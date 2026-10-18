"""FastAPI backend for the two-arm bottle opener.

Serves the single-page frontend (frontend/index.html) and exposes REST endpoints
for robot control. The frontend talks to this over HTTP; all Viam calls live here.

Run with:
    uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import json
import os
import time
from functools import wraps
from pathlib import Path

import numpy as np
from typing import Any

import importlib.util
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from robot import (
    connect, disconnect, joints, end_position, move_joint_positions,
    move_position, gripper_action, camera_frame, list_arm_names,
    list_all_resource_names, arm_stop, gripper_stop, transform_pose,
    world_point, capture_frame, yolo_detect, run as viam_run,
    motion_session, cancel_motion, check_motion,
)
_ROBOT = None
def _get_robot():
    global _ROBOT
    if _ROBOT is None:
        from robot import _robot as r
        _ROBOT = r._sync_robot()
    return _ROBOT
from yolo_grab import (DEFAULTS as YOLO_DEFAULTS, pick_and_place as yg_pick_and_place, reset_arm as yg_reset_arm, measure as yg_measure, plan as yg_plan, level_target as yg_level_target, yaw_target as yg_yaw_target, _unit, _straight)
from cap_unscrew import UnscrewParams, Unscrewer, describe as unscrew_describe
from bottle_task import BottleTask, go_to_keyframe as bt_go_to_keyframe, apply_gripper as bt_apply_gripper
from cap_line_search import (line_search_cap, detect_s_template, size_to_distance,
                             _render_letter_template, _HAS_TESSERACT)

# Load poses_io from the repo root without adding it to sys.path (avoids shadowing backend modules).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_poses_spec = importlib.util.spec_from_file_location("poses_io", _REPO_ROOT / "poses_io.py")
_poses_io = importlib.util.module_from_spec(_poses_spec)
_poses_spec.loader.exec_module(_poses_io)
sequence = _poses_io.sequence
load_poses = _poses_io.load
save_poses = _poses_io.save

_cap_spec = importlib.util.spec_from_file_location("shared_cap_vision", _REPO_ROOT / "cap_vision.py")
_cap_vision = importlib.util.module_from_spec(_cap_spec)
_cap_spec.loader.exec_module(_cap_vision)

app = FastAPI(title="Viam Bottle Opener Backend")

# Resolve paths relative to this file so cwd doesn't matter.
BACKEND_DIR = Path(__file__).resolve().parent
POSES_FILE = Path(os.environ.get("POSES_FILE", str(_REPO_ROOT / "poses.json")))
if not POSES_FILE.is_absolute():
    POSES_FILE = BACKEND_DIR / POSES_FILE

# Singletons (created at startup after robot connects)
_UNSCREWER: Unscrewer | None = None
_BOTTLE_TASK: BottleTask | None = None

# YOLO step-by-step state (persists between HTTP requests, like Streamlit session state)
YOLO_STATE = {"meas": None, "plan": None, "cfg": None}


def motion_endpoint(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            with motion_session():
                return fn(*args, **kwargs)
        except (RuntimeError, InterruptedError) as exc:
            raise HTTPException(status_code=409, detail=str(exc))
    return wrapped


@app.post("/api/stop_all")
def api_stop_all():
    cancel_motion()
    # Signal the sequence threads before stopping hardware, so they cannot
    # advance while the stop requests are being processed.
    for task in (_BOTTLE_TASK, _UNSCREWER):
        if task is not None:
            task._stop.set()
    errors = []
    for name in ("arm", "arm2"):
        for stop in (arm_stop, gripper_stop):
            try:
                stop(name)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
    if errors:
        raise HTTPException(status_code=500, detail="; ".join(errors))
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class JointMove(BaseModel):
    values: list[float]


class PoseMove(BaseModel):
    x: float
    y: float
    z: float
    o_x: float
    o_y: float
    o_z: float
    theta: float


class GripperAction(BaseModel):
    action: str  # "grab" or "open"


class SavePose(BaseModel):
    arm: str
    name: str
    kind: str  # "joints" or "pose"
    values: list[float] | None = None
    pose: dict[str, float] | None = None
    gripper: str | None = None
    delay: float | None = None


class ReorderPoses(BaseModel):
    poses: list[dict]


class YoloLocate(BaseModel):
    conf: float | None = None
    frames: int | None = None
    radius: float | None = None
    grasp_frac: float | None = None
    pad_mm: float | None = None
    pre_mm: float | None = None
    stroke_mm: float | None = None
    lat_trim_mm: float | None = None
    depth_trim_mm: float | None = None
    level: bool | None = None


class YoloMove(BaseModel):
    kind: str  # "pre", "grab", "back"


class Nudge(BaseModel):
    direction: str  # left, right, up, down, in, out
    mm: float = 5.0


class LevelGripper(BaseModel):
    pass


class Yaw(BaseModel):
    deg: float = 1.0


class CapDetect(BaseModel):
    min_r: int = 40
    max_r: int = 250
    sensitivity: int = 50
    min_silver: float = 0.5
    min_rim: float = 0.55
    cap_d: float = 30.0


class AlignPlan(BaseModel):
    circle_x: float
    circle_y: float
    circle_r: float
    circle_z: float
    hover: float = 80.0
    keep_height: bool = True


class LineSearchRequest(BaseModel):
    cap_d_mm: float = 34.0
    search_range_px: int = 200
    step_px: int = 10
    use_ocr: bool = False


class Keyframe(BaseModel):
    arm: str
    name: str
    values: list[float] | None = None
    pose: dict[str, float] | None = None
    gripper: str | None = None
    delay: float | None = None


class RunSequence(BaseModel):
    keyframes: list[Keyframe]


class UnscrewRequest(BaseModel):
    twists: int = 4
    twist_deg: float = 90.0
    loosen_sign: int = -1
    release_pos: int = 500
    final_lift_mm: float = 30.0
    speed_deg_s: float = 20.0
    normal_speed_deg_s: float = 60.0
    grip_force_pct: int = 50
    normal_grip_force_pct: int = 50
    max_level_tilt_deg: float = 15.0
    start_pose: str = ""
    start_joints: list[float] | None = None
    start_target: list[float] | None = None
    dry_run: bool = False


class PickPlaceRequest(BaseModel):
    # Uses defaults from yolo_grab.DEFAULTS; only override what's needed
    pass


class FullRunRequest(BaseModel):
    # Runs all keyframes (arm then arm2), then unscrew, then retreat/drop
    unscrew: UnscrewRequest | None = None


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------
@app.on_event("startup")
def _startup() -> None:
    connect()
    global _UNSCREWER, _BOTTLE_TASK
    _UNSCREWER = Unscrewer(arm_name="arm2", gripper_name="gripper2")
    _BOTTLE_TASK = BottleTask(
        arms={"arm": "arm", "arm2": "arm2"},
        grippers={"arm": "gripper", "arm2": "gripper2"},
        unscrewer=_UNSCREWER,
    )


@app.on_event("shutdown")
def _shutdown() -> None:
    disconnect()


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
@app.get("/")
def index() -> FileResponse:
    return FileResponse(BACKEND_DIR.parent / "frontend" / "index.html")


# ---------------------------------------------------------------------------
# Robot resources
# ---------------------------------------------------------------------------
@app.get("/api/arms")
def api_arms() -> dict[str, Any]:
    try:
        names = list_arm_names()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"arms": names}


@app.get("/api/resources")
def api_resources() -> dict[str, Any]:
    try:
        return list_all_resource_names()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/arm/{name}/joints")
def api_joints(name: str) -> dict[str, Any]:
    try:
        return {"arm": name, "values": joints(name)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/arm/{name}/position")
def api_position(name: str) -> dict[str, Any]:
    try:
        return {"arm": name, **end_position(name)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/arm/{name}/move/joints")
@motion_endpoint
def api_move_joints(name: str, move: JointMove) -> dict[str, Any]:
    if len(move.values) != 6:
        raise HTTPException(status_code=400, detail="Expected 6 joint values")
    try:
        move_joint_positions(name, move.values)
        return {"status": "ok", "arm": name}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/arm/{name}/move/position")
@motion_endpoint
def api_move_position(name: str, move: PoseMove) -> dict[str, Any]:
    try:
        move_position(name, move.x, move.y, move.z, move.o_x, move.o_y, move.o_z, move.theta)
        return {"status": "ok", "arm": name}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/arm/{name}/gripper/{action}")
@motion_endpoint
def api_gripper(name: str, action: str) -> dict[str, Any]:
    if action not in ("grab", "open"):
        raise HTTPException(status_code=400, detail="action must be 'grab' or 'open'")
    try:
        gripper_action(name, action)
        return {"status": "ok", "arm": name, "action": action}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/arm/{name}/stop")
def api_arm_stop(name: str) -> dict[str, Any]:
    cancel_motion()
    try:
        arm_stop(name)
        gripper_stop(name)
        return {"status": "ok", "arm": name}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Keyframes / poses
# ---------------------------------------------------------------------------
@app.get("/api/poses")
def api_poses() -> dict[str, Any]:
    if not POSES_FILE.exists():
        return {"poses": {}}
    data = json.loads(POSES_FILE.read_text())
    return {"poses": data}


@app.post("/api/poses/run")
@motion_endpoint
def api_run_poses(seq: RunSequence) -> dict[str, Any]:
    """Play a list of keyframes in order, one arm at a time (blocking)."""
    try:
        for kf in seq.keyframes:
            check_motion()
            arm_name = kf.arm
            d = kf.model_dump()
            if d.get("pose") is not None:
                p = d["pose"]
                move_position(arm_name,
                              float(p["x"]), float(p["y"]), float(p["z"]),
                              float(p["o_x"]), float(p["o_y"]), float(p["o_z"]), float(p["theta"]))
            else:
                move_joint_positions(arm_name, kf.values or [])
            if kf.gripper:
                gripper_action(arm_name, kf.gripper)
            if kf.delay and kf.delay > 0:
                time.sleep(float(kf.delay))
        return {"status": "ok", "played": len(seq.keyframes)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# YOLO pick & place (arm 1)
# ---------------------------------------------------------------------------
@app.post("/api/pick_and_place")
@motion_endpoint
def api_pick_and_place() -> dict[str, Any]:
    """Run the full YOLO pick & place on arm 1 (blocking)."""
    try:
        cfg = dict(YOLO_DEFAULTS)
        keyframes = load_poses(POSES_FILE)
        targets = {}
        for name in ("NewMid", "parking"):
            frame = next((p for p in keyframes
                          if p.get("arm") == "arm" and p.get("name") == name), None)
            values = frame.get("values") if frame else None
            if (not isinstance(values, list) or len(values) != 6
                    or not all(isinstance(v, (int, float)) and np.isfinite(v) for v in values)):
                raise ValueError(f"Arm 1 needs a {name} keyframe with six finite joint angles")
            targets[name] = values
        result = yg_pick_and_place(
            arm_name="arm", gripper_name="gripper",
            cam_name="cam", cam_frame="cam", grip_frame="gripper",
            newmid=targets["NewMid"],
            parking=targets["parking"],
            cfg=cfg, check=check_motion,
        )
        return {"status": "ok", "frames": result[0]["frames"], "width_mm": result[0]["width_mm"]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/reset_arm1")
@motion_endpoint
def api_reset_arm1() -> dict[str, Any]:
    """Reset arm 1: open gripper, move to parking keyframe."""
    try:
        parking = next((p for p in load_poses(POSES_FILE)
                        if p.get("arm") == "arm" and p.get("name") == "parking"), None)
        if parking is None or len(parking.get("values", [])) != 6:
            raise ValueError("Arm 1 needs a parking keyframe with six joint angles")
        yg_reset_arm(
            arm_name="arm", gripper_name="gripper",
            parking=parking["values"], check=check_motion,
        )
        return {"status": "ok"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Unscrew (arm 2)
# ---------------------------------------------------------------------------
@app.post("/api/unscrew")
@motion_endpoint
def api_unscrew(req: UnscrewRequest | None = None) -> dict[str, Any]:
    """Run the cap unscrew sequence on arm 2 (blocking)."""
    if _UNSCREWER is None:
        raise HTTPException(status_code=500, detail="Unscrewer not initialized")
    if _UNSCREWER.running:
        raise HTTPException(status_code=409, detail="Unscrew already running")
    try:
        p = UnscrewParams(**(req.model_dump() if req else {}))
        if p.start_pose:
            frame = next((k for k in load_poses(POSES_FILE)
                          if k["arm"] == "arm2" and k["name"] == p.start_pose), None)
            if frame is None:
                raise ValueError("Unknown arm2 start pose: " + p.start_pose)
            p.start_joints = tuple(frame.get("values") or ())
            p.start_target = tuple(frame["pose"][k] for k in ("x", "y", "z", "o_x", "o_y", "o_z", "theta")) if frame.get("pose") else ()
        _UNSCREWER.start(p)
        # Wait for completion (blocking endpoint)
        while _UNSCREWER.running:
            time.sleep(0.1)
        u = _UNSCREWER.status()
        if u["state"] != "done":
            raise HTTPException(status_code=500, detail=f"unscrew {u['state']} at '{u['step']}': {u.get('error')}")
        return {"status": "ok", "step": u["step"]}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/unscrew/describe")
def api_unscrew_describe() -> dict[str, Any]:
    """Human-readable step list for the current unscrew params."""
    p = UnscrewParams()
    return {"steps": unscrew_describe(p)}


@app.post("/api/unscrew/stop")
def api_unscrew_stop() -> dict[str, Any]:
    if _UNSCREWER:
        _UNSCREWER.stop()
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Full run (arm's keyframes -> arm2 keyframes -> unscrew -> retreat -> drop)
# ---------------------------------------------------------------------------
@app.post("/api/full_run")
@motion_endpoint
def api_full_run(req: FullRunRequest | None = None) -> dict[str, Any]:
    """Run the complete bottle-opening sequence (blocking)."""
    if _BOTTLE_TASK is None:
        raise HTTPException(status_code=500, detail="BottleTask not initialized")
    if _BOTTLE_TASK.running:
        raise HTTPException(status_code=409, detail="Full run already running")
    try:
        all_kf = load_poses(POSES_FILE)
        keyframes = sequence(all_kf, "arm_then_arm2")
        required = {"safe_bottle_start_pose", "open_bottle_cap_pose", "open_bottle_cap_up_pose", "trash_drop_pose"}
        missing = required - {k["name"] for k in keyframes if k["arm"] == "arm2"}
        if missing:
            raise ValueError("Missing arm2 keyframes: " + ", ".join(sorted(missing)))
        # Remove trash_drop from keyframe list (it's handled separately as drop)
        drop_kf = next((k for k in keyframes if k["arm"] == "arm2" and k["name"] == "trash_drop_pose"), None)
        pre_kf = [k for k in keyframes if k != drop_kf]
        # alongside: arm2 starts safe_bottle_start_pose together with arm's first keyframe
        alongside = next((k for k in pre_kf if k["arm"] == "arm2" and k["name"] == "safe_bottle_start_pose"), None)
        if alongside is not None and pre_kf and pre_kf[0]["arm"] != "arm2":
            pre_kf.remove(alongside)
        # via: safe_bottle_start_pose (move only, after unscrew)
        via = [k for k in keyframes if k["arm"] == "arm2" and k["name"] == "safe_bottle_start_pose"]
        # retreat: open_bottle_cap_up_pose (rise up holding cap)
        retreat = next((k for k in pre_kf if k["arm"] == "arm2" and k["name"] == "open_bottle_cap_up_pose"), None)
        p = UnscrewParams(**(req.unscrew.model_dump() if req and req.unscrew else {}))
        p.start_joints = ()
        p.start_target = ()
        p.start_pose = ""
        _BOTTLE_TASK.start(pre_kf, p, drop_kf, retreat, via, alongside)
        while _BOTTLE_TASK.running:
            time.sleep(0.1)
        s = _BOTTLE_TASK.status()
        if s["state"] != "done":
            raise HTTPException(status_code=500, detail=f"full run {s['state']} at '{s['step']}': {s.get('error')}")
        return {"status": "ok", "step": s["step"]}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/full_run/stop")
def api_full_run_stop() -> dict[str, Any]:
    if _BOTTLE_TASK:
        _BOTTLE_TASK.stop()
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
@app.get("/api/status")
def api_status() -> dict[str, Any]:
    result = {"arms": {}}
    for name in ("arm", "arm2"):
        try:
            result["arms"][name] = {
                "joints": joints(name),
                "position": end_position(name),
            }
        except Exception:
            result["arms"][name] = {"error": "unavailable"}
    if _UNSCREWER:
        result["unscrew"] = _UNSCREWER.status()
    if _BOTTLE_TASK:
        result["full_run"] = _BOTTLE_TASK.status()
    return result


# ---------------------------------------------------------------------------
# Keyframes: save, reorder, delete
# ---------------------------------------------------------------------------
@app.post("/api/poses/save")
def api_save_pose(sp: SavePose) -> dict[str, Any]:
    try:
        poses = load_poses(POSES_FILE)
        poses = [p for p in poses if (p.get("arm"), p.get("name")) != (sp.arm, sp.name)]
        entry = {"arm": sp.arm, "name": sp.name}
        if sp.kind == "pose" and sp.pose is not None:
            entry["pose"] = sp.pose
        elif sp.values is not None:
            entry["values"] = list(sp.values)
        if sp.gripper:
            entry["gripper"] = sp.gripper
        if sp.delay and sp.delay > 0:
            entry["delay"] = float(sp.delay)
        poses.append(entry)
        save_poses(POSES_FILE, poses)
        return {"status": "ok", "saved": sp.name}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/poses/reorder")
def api_reorder_poses(rp: ReorderPoses) -> dict[str, Any]:
    try:
        poses = rp.poses
        for p in poses:
            if "arm" not in p:
                p["arm"] = "arm"
        save_poses(POSES_FILE, poses)
        return {"status": "ok", "count": len(poses)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# YOLO step-by-step (arm 1)
# ---------------------------------------------------------------------------
@app.post("/api/yolo/locate")
def api_yolo_locate(req: YoloLocate | None = None) -> dict[str, Any]:
    try:
        cfg = dict(YOLO_DEFAULTS)
        if req:
            for k, v in req.model_dump().items():
                if v is not None:
                    cfg[k] = v
        m = yg_measure("arm", "cam", "cam", "gripper", cfg)
        plan = yg_plan(m, cfg)
        YOLO_STATE["meas"] = m
        YOLO_STATE["plan"] = plan
        YOLO_STATE["cfg"] = cfg
        return {
            "status": "ok",
            "frames": m["frames"],
            "of": m["of"],
            "bottle_x": round(m["axis"][0], 1),
            "bottle_y": round(m["axis"][1], 1),
            "grasp_z": round(plan["zg"], 1),
            "width_mm": round(m["width_mm"], 1),
            "height_mm": round(plan["height_mm"], 1),
            "camera_to_bottle_mm": round(m["z"], 1),
            "checks": plan["checks"],
        }
    except Exception as exc:
        YOLO_STATE["meas"] = None
        YOLO_STATE["plan"] = None
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/yolo/move")
@motion_endpoint
def api_yolo_move(req: YoloMove) -> dict[str, Any]:
    try:
        plan = YOLO_STATE["plan"]
        meas = YOLO_STATE["meas"]
        cfg = YOLO_STATE["cfg"]
        if plan is None or meas is None:
            raise HTTPException(status_code=400, detail="Locate the bottle first.")
        kind = req.kind
        errors = [text for level, text in plan["checks"] if level == "error"]
        if kind in ("pre", "grab") and errors:
            raise HTTPException(status_code=400, detail="; ".join(errors))
        if kind in ("pre", "grab"):
            now = end_position("arm")
            expected = meas["end"] if kind == "pre" else plan["pre"]
            distance = np.linalg.norm([now[k] - expected[k] for k in ("x", "y", "z")])
            same_orientation = (np.allclose([now[k] for k in ("o_x", "o_y", "o_z")],
                                           [expected[k] for k in ("o_x", "o_y", "o_z")], atol=0.01)
                                and abs((now["theta"] - expected["theta"] + 180) % 360 - 180) < 1.0)
            if distance > (2 if kind == "pre" else 5) or not same_orientation:
                detail = "Arm moved since Locate; locate again" if kind == "pre" else "Go to the pre-grasp first"
                raise HTTPException(status_code=400, detail=detail)
        if kind == "pre":
            gripper_action("arm", "open")
            _straight("arm", plan["pre"], cfg["pre_step_mm"])
        elif kind == "grab":
            gripper_action("arm", "open")
            _straight("arm", plan["grab"], cfg["in_step_mm"])
            held = gripper_action("arm", "grab")
            if held is False:
                gripper_action("arm", "open")
                _straight("arm", plan["pre"], cfg["in_step_mm"])
                raise HTTPException(status_code=500, detail="Gripper closed on nothing; backed out to pre-grasp.")
        elif kind == "back":
            now = end_position("arm")
            a = plan["a"]
            k = sum((now[q] - plan["pre"][q]) * a[idx] for idx, q in enumerate(("x", "y", "z")))
            if k < 2:
                return {"status": "ok", "note": "Already at or behind the pre-grasp."}
            goal = dict(now, x=now["x"] - k * a[0], y=now["y"] - k * a[1], z=now["z"] - k * a[2])
            _straight("arm", goal, cfg["in_step_mm"])
        else:
            raise HTTPException(status_code=400, detail="kind must be 'pre', 'grab', or 'back'")
        return {"status": "ok", "kind": kind}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/yolo/nudge")
@motion_endpoint
def api_yolo_nudge(req: Nudge) -> dict[str, Any]:
    try:
        e = end_position("arm")
        a = _unit([e["o_x"], e["o_y"], e["o_z"]])
        up = np.array([0.0, 0.0, 1.0])
        left = np.cross(up, _unit(a - a.dot(up) * up))
        dirs = {"left": left, "right": -left, "up": up, "down": -up, "in": a, "out": -a}
        if req.direction not in dirs:
            raise HTTPException(status_code=400, detail=f"direction must be one of {list(dirs)}")
        d = dirs[req.direction] * min(float(req.mm), 20.0)
        goal = dict(e, x=e["x"] + d[0], y=e["y"] + d[1], z=e["z"] + d[2])
        move_position("arm", goal["x"], goal["y"], goal["z"],
                      goal["o_x"], goal["o_y"], goal["o_z"], goal["theta"])
        return {"status": "ok", "direction": req.direction, "mm": float(req.mm)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/yolo/level")
@motion_endpoint
def api_yolo_level(_: LevelGripper | None = None) -> dict[str, Any]:
    try:
        e = end_position("arm")
        PIVOT_MM = 105.0 + YOLO_DEFAULTS["pad_mm"]
        target, turn = yg_level_target(e, PIVOT_MM)
        move_position("arm", target["x"], target["y"], target["z"],
                      target["o_x"], target["o_y"], target["o_z"], target["theta"])
        return {"status": "ok", "turn_deg": round(turn, 1)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/yolo/yaw")
@motion_endpoint
def api_yolo_yaw(req: Yaw) -> dict[str, Any]:
    try:
        e = end_position("arm")
        PIVOT_MM = 105.0 + YOLO_DEFAULTS["pad_mm"]
        target, turn = yg_yaw_target(e, req.deg, PIVOT_MM)
        move_position("arm", target["x"], target["y"], target["z"],
                      target["o_x"], target["o_y"], target["o_z"], target["theta"])
        return {"status": "ok", "deg": req.deg, "turn_deg": round(turn, 1)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/yolo/state")
def api_yolo_state() -> dict[str, Any]:
    m = YOLO_STATE["meas"]
    p = YOLO_STATE["plan"]
    if m is None or p is None:
        return {"available": False}
    return {
        "available": True,
        "frames": m["frames"],
        "of": m["of"],
        "bottle_x": round(m["axis"][0], 1),
        "bottle_y": round(m["axis"][1], 1),
        "grasp_z": round(p["zg"], 1),
        "width_mm": round(m["width_mm"], 1),
        "height_mm": round(p["height_mm"], 1),
        "camera_to_bottle_mm": round(m["z"], 1),
        "checks": p["checks"],
    }


# ---------------------------------------------------------------------------
# Cap detection + Align (arm 2)
# ---------------------------------------------------------------------------
@app.post("/api/cap/detect")
def api_cap_detect(req: CapDetect | None = None) -> dict[str, Any]:
    try:
        from types import SimpleNamespace
        cv_detect_circles = _cap_vision.detect_circles
        import cv2
        from robot import capture_frame
        cap = capture_frame("cam2", "cam2", "arm2")
        bgr = cap["bgr"]
        depth = cap["depth"]
        vision = _cap_vision.CapVision(None, None, None, None, SimpleNamespace(
            focal_x_px=cap["fx"], focal_y_px=cap["fy"], center_x_px=cap["cx"], center_y_px=cap["cy"]))
        detect = (req.min_r, req.max_r, req.sensitivity, req.min_silver, req.min_rim) if req else (40, 250, 50, 0.5, 0.55)
        circles, rejected = cv_detect_circles(bgr, *detect)
        result = {"circles": [], "rejected": []}
        for u, v, r, _ in circles:
            p = vision.measure((u, v, r, 0), depth, req.cap_d if req else 30.0)
            result["circles"].append({
                "u": round(u, 1), "v": round(v, 1), "r": round(r, 1),
                "x": round(p["x"], 1), "y": round(p["y"], 1), "z": round(p["z"], 1),
                "method": p["method"], "valid": round(p["valid"], 2),
            })
        for u, v, r, _ in rejected:
            result["rejected"].append({"u": round(u, 1), "v": round(v, 1), "r": round(r, 1)})
        img = cv2.cvtColor(vision.annotate(bgr, circles, rejected), cv2.COLOR_RGB2BGR)
        ok, enc = cv2.imencode(".jpg", img)
        result["jpeg_base64"] = base64.b64encode(enc.tobytes()).decode("ascii") if ok else None
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Cap detection: line-search (alternative to Hough circles)
# ---------------------------------------------------------------------------
@app.post("/api/cap/line-search")
def api_cap_line_search(req: LineSearchRequest | None = None) -> dict[str, Any]:
    """Line-search cap detection: walk down the bottle axis looking for the cap.

    Uses known bottle geometry to collapse the 2D circle search into a 1D line
    search. Optionally template-matches / OCRs for the 'S' on the cap.
    """
    try:
        import cv2
        from robot import capture_frame
        cap = capture_frame("cam2", "cam2", "arm2")
        bgr = cap["bgr"]
        depth = cap["depth"]
        i = cap["intrinsics"] if "intrinsics" in cap else None

        # Build a minimal intrinsics object from camera capture
        class _Intr:
            def __init__(self, cap_dict):
                self.focal_x_px = float(cap_dict.get("fx", 600.0))
                self.focal_y_px = float(cap_dict.get("fy", 600.0))
                self.center_x_px = float(cap_dict.get("cx", cap["bgr"].shape[1] / 2))
                self.center_y_px = float(cap_dict.get("cy", cap["bgr"].shape[0] / 2))

        intrinsics = _Intr(cap)

        # Use image center as approx bottle center, axis pointing down in image
        h, w = bgr.shape[:2]
        bottle_center_px = (w / 2, h / 2)
        axis_dir_px = (0.0, 1.0)

        template = _render_letter_template('S')
        candidates = line_search_cap(
            bgr, depth, intrinsics, bottle_center_px, axis_dir_px,
            cap_d_mm=req.cap_d_mm if req else 34.0,
            search_range_px=req.search_range_px if req else 200,
            step_px=req.step_px if req else 10,
            use_ocr=req.use_ocr if req else False,
            template=template,
        )

        result = {
            "candidates": [],
            "has_ocr": _HAS_TESSERACT,
        }
        for c in candidates:
            result["candidates"].append({
                "u": round(c.u, 1), "v": round(c.v, 1), "r": round(c.r, 1),
                "x_mm": round(c.x_mm, 1), "y_mm": round(c.y_mm, 1), "z_mm": round(c.z_mm, 1),
                "score": round(c.score, 3),
                "has_s": c.has_s,
                "s_confidence": round(c.s_confidence, 3),
            })

        # Annotate image with detections
        for c in candidates[:5]:
            color = (0, 255, 0) if c.has_s else (0, 200, 255)
            cv2.circle(bgr, (int(c.u), int(c.v)), int(c.r), color, 3)
            cv2.circle(bgr, (int(c.u), int(c.v)), 4, color, -1)
            label = f"S:{c.s_confidence:.2f}" if c.has_s else f"{c.score:.2f}"
            cv2.putText(bgr, label, (int(c.u) + 5, int(c.v) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        ok, enc = cv2.imencode(".jpg", bgr)
        result["jpeg_base64"] = base64.b64encode(enc.tobytes()).decode("ascii") if ok else None
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/cap/line-search/template")
def api_cap_line_search_template() -> dict[str, Any]:
    """Return the 'S' template image used for template matching."""
    try:
        import cv2
        import base64
        template = _render_letter_template('S')
        ok, enc = cv2.imencode(".jpg", template)
        return {"jpeg_base64": base64.b64encode(enc.tobytes()).decode("ascii") if ok else None}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/align/plan")
def api_align_plan(req: AlignPlan) -> dict[str, Any]:
    try:
        import numpy as np
        from robot import transform_pose
        cap_world = transform_pose("cam2", {"x": req.circle_x, "y": req.circle_y, "z": req.circle_z,
                                              "o_x": 0, "o_y": 0, "o_z": 1, "theta": 0}, "arm2_origin")
        grip_world = transform_pose("gripper2", {"x": 0, "y": 0, "z": 0, "o_x": 0, "o_y": 0, "o_z": 1, "theta": 0}, "arm2_origin")
        end = end_position("arm2")
        keep_height = req.keep_height
        dz = 0.0 if keep_height else cap_world["z"] + req.hover - grip_world["z"]
        delta = np.array([cap_world["x"] - grip_world["x"],
                          cap_world["y"] - grip_world["y"],
                          dz])
        norm = float(np.linalg.norm(delta))
        clamped = norm > 100.0
        if clamped:
            delta *= 100.0 / norm
        target = {
            "x": end["x"] + delta[0], "y": end["y"] + delta[1], "z": end["z"] + delta[2],
            "o_x": end["o_x"], "o_y": end["o_y"], "o_z": end["o_z"], "theta": end["theta"],
        }
        return {
            "cap_world": {k: round(v, 1) for k, v in cap_world.items()},
            "grip_world": {k: round(v, 1) for k, v in grip_world.items()},
            "delta": {k: round(v, 1) for k, v in zip(("x", "y", "z"), delta)},
            "norm": round(norm, 1),
            "clamped": clamped,
            "target": {k: round(v, 3) for k, v in target.items()},
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/align/move")
@motion_endpoint
def api_align_move(req: AlignPlan) -> dict[str, Any]:
    try:
        plan = api_align_plan(req)
        move_position("arm2",
                      plan["target"]["x"], plan["target"]["y"], plan["target"]["z"],
                      plan["target"]["o_x"], plan["target"]["o_y"], plan["target"]["o_z"], plan["target"]["theta"])
        return {"status": "ok", "norm": plan["norm"], "clamped": plan["clamped"]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Camera: annotated preview images
# ---------------------------------------------------------------------------
@app.get("/api/camera/{name}/frame")
def api_depth_frame(name: str) -> dict[str, Any]:
    import cv2
    cap = capture_frame(name, name, "arm2" if name == "cam2" else "arm")
    if cap["depth"] is None:
        raise HTTPException(status_code=400, detail="Camera did not return depth")
    rgb = _cap_vision.colorize_depth(cap["depth"])
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return {"jpeg_base64": base64.b64encode(enc.tobytes()).decode("ascii") if ok else None}


@app.get("/api/camera/{name}/preview")
def api_camera_preview(name: str) -> dict[str, Any]:
    import base64
    try:
        jpg = camera_frame(name)
        return {"jpeg_base64": base64.b64encode(jpg).decode("ascii")}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# YOLO annotated preview (arm 1 camera)
# ---------------------------------------------------------------------------
@app.get("/api/yolo/preview")
def api_yolo_preview(confidence: float = 0.65, show_all: bool = False) -> dict[str, Any]:
    """Return an annotated JPEG frame from cam with YOLO bottle boxes drawn."""
    import base64
    import cv2
    try:
        jpg = camera_frame("cam")
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        detections = [d for d in yolo_detect(YOLO_DEFAULTS["service"], jpg) if d["confidence"] >= confidence]
        bottles = sorted((d for d in detections if d["class_name"].lower() == "bottle"), key=lambda d: d["x_min"] + d["x_max"])
        count = len(bottles)
        for d in (detections if show_all else bottles):
            color = (0, 255, 0) if bottles and d is bottles[0] else (0, 200, 255)
            cv2.rectangle(img, (int(d["x_min"]), int(d["y_min"])), (int(d["x_max"]), int(d["y_max"])), color, 2)
            cv2.putText(img, f"{d['class_name']}: {d['confidence']:.0%}", (int(d["x_min"]), max(20, int(d["y_min"]))), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2)
        ok, enc = cv2.imencode(".jpg", img)
        return {"jpeg_base64": base64.b64encode(enc.tobytes()).decode("ascii") if ok else None, "count": count}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health")
def api_health() -> dict[str, Any]:
    return {"status": "ok"}
