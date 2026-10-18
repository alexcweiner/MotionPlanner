"""Viam robot client wrapper for the bottle-opener backend.

Holds a single RobotClient connection (cached) and exposes small async helpers
used by the FastAPI routes. All Viam calls go through `run()` so the async SDK
can be called from FastAPI's sync route functions via asyncio.
"""

from __future__ import annotations

import asyncio
import os
import threading
from contextlib import contextmanager
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv
from viam.components.arm import Arm, JointPositions
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.media.video import CameraMimeType, ViamImage
from viam.proto.common import Pose, PoseInFrame
from viam.robot.client import RobotClient
from viam.services.vision import VisionClient

# Load .env from the backend directory (and fall back to repo root)
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)

ADDRESS = os.environ.get("VIAM_MACHINE_ADDRESS", "armfarm16-main.310sld03v2.viam.cloud")
API_KEY_ID = os.environ.get("VIAM_API_KEY_ID", "")
API_KEY = os.environ.get("VIAM_API_KEY", "")

RIGS = {"arm": ("cam", "gripper"), "arm2": ("cam2", "gripper2")}


_motion_lock = threading.Lock()
_motion_context = threading.local()
_stop_generation = 0


def cancel_motion():
    """Invalidate in-flight HTTP motion routines before sending hardware stops."""
    global _stop_generation
    _stop_generation += 1


def check_motion():
    generation = getattr(_motion_context, "generation", _stop_generation)
    if generation != _stop_generation:
        raise InterruptedError("Motion stopped; no further steps will run")


@contextmanager
def motion_session():
    if not _motion_lock.acquire(blocking=False):
        raise RuntimeError("Another motion operation is running")
    _motion_context.generation = _stop_generation
    try:
        yield
        check_motion()
    finally:
        del _motion_context.generation
        _motion_lock.release()


class Robot:
    """Cached robot connection + convenience accessors."""

    def __init__(self) -> None:
        self._robot: RobotClient | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> RobotClient:
        if self._robot is None:
            async with self._lock:
                if self._robot is None:
                    if not API_KEY_ID.strip() or not API_KEY.strip():
                        raise RuntimeError(
                            "Set VIAM_API_KEY_ID and VIAM_API_KEY in .env or "
                            "backend/.env (or env.sh when using ./run.sh), then restart the backend."
                        )
                    opts = RobotClient.Options.with_api_key(
                        api_key=API_KEY,
                        api_key_id=API_KEY_ID,
                        refresh_interval=0,
                    )
                    self._robot = await RobotClient.at_address(ADDRESS, opts)
        return self._robot

    async def close(self) -> None:
        if self._robot is not None:
            await self._robot.close()
            self._robot = None

    def arm(self, name: str) -> Arm:
        return Arm.from_robot(self._sync_robot(), name)

    def gripper(self, name: str) -> Gripper:
        return Gripper.from_robot(self._sync_robot(), name)

    def camera(self, name: str) -> Camera:
        return Camera.from_robot(self._sync_robot(), name)

    def _sync_robot(self) -> RobotClient:
        """Blocking accessor: use inside run() only."""
        if self._robot is None:
            raise RuntimeError("Robot not connected; call connect() first")
        return self._robot


# Module-level singleton used by route handlers.
_robot = Robot()
_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()
_loop_thread: threading.Thread | None = None


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()

            def serve() -> None:
                asyncio.set_event_loop(_loop)
                _loop.run_forever()

            _loop_thread = threading.Thread(target=serve, name="viam-sdk", daemon=True)
            _loop_thread.start()
    return _loop


def run(coro) -> Any:
    """Submit every SDK call to one persistent loop, including connect and close.

    FastAPI worker threads can submit concurrently. Keeping the loop alive also
    preserves the client's background tasks between requests.
    """
    loop = _get_loop()
    if threading.current_thread() is _loop_thread:
        coro.close()
        raise RuntimeError("Cannot call blocking run() from the Viam event loop")
    return asyncio.run_coroutine_threadsafe(coro, loop).result()


def connect() -> None:
    """Connect the robot synchronously. Does not raise if the machine is unreachable."""
    try:
        run(_robot.connect())
    except Exception:
        # Robot unavailable: endpoints will return errors until it connects.
        pass


def disconnect() -> None:
    run(_robot.close())


def joints(name: str) -> list[float]:
    """Read current joint positions (values list) for an arm."""
    return list(run(_robot.arm(name).get_joint_positions()).values)


def end_position(name: str) -> dict[str, float]:
    """Read the current end position pose for an arm."""
    e = run(_robot.arm(name).get_end_position())
    return {
        "x": float(e.x), "y": float(e.y), "z": float(e.z),
        "o_x": float(e.o_x), "o_y": float(e.o_y), "o_z": float(e.o_z), "theta": float(e.theta),
    }


def move_joint_positions(name: str, values: list[float]) -> None:
    """Move an arm to joint positions (joint-space move)."""
    check_motion()
    run(_robot.arm(name).move_to_joint_positions(JointPositions(values=list(values))))


def move_position(name: str, x: float, y: float, z: float,
                  o_x: float, o_y: float, o_z: float, theta: float) -> None:
    """Move an arm to a Cartesian pose via the motion planner."""
    check_motion()
    run(_robot.arm(name).move_to_position(
        Pose(x=x, y=y, z=z, o_x=o_x, o_y=o_y, o_z=o_z, theta=theta)
    ))


def gripper_action(arm_name: str, action: str) -> bool | None:
    """Grab or open the gripper on an arm. Returns True/False for grab, None for open."""
    check_motion()
    grip_name = RIGS[arm_name][1]
    if action == "grab":
        return run(_robot.gripper(grip_name).grab())
    elif action == "open":
        run(_robot.gripper(grip_name).open())
    else:
        raise ValueError(f"Unknown gripper action: {action}")


def arm_do_command(arm_name: str, command: dict) -> None:
    """Send a DoCommand to an arm (e.g. {"set_speed": 20.0})."""
    run(_robot.arm(arm_name).do_command(command))


def gripper_do_command(arm_name: str, command: dict) -> None:
    """Send a DoCommand to an arm's gripper (e.g. {"set_gripper_torque": 60})."""
    grip_name = RIGS[arm_name][1]
    run(_robot.gripper(grip_name).do_command(command))


def transform_pose(src_frame: str, pose_dict: dict, dst_frame: str, extra=None) -> dict:
    """Transform a pose from src_frame to dst_frame using the machine's frame system."""
    q = PoseInFrame(reference_frame=src_frame, pose=Pose(**pose_dict))
    r = run(_robot._sync_robot().transform_pose(q, dst_frame, extra)).pose
    return {"x": float(r.x), "y": float(r.y), "z": float(r.z),
            "o_x": float(r.o_x), "o_y": float(r.o_y), "o_z": float(r.o_z), "theta": float(r.theta)}


def arm_stop(arm_name: str) -> None:
    run(_robot.arm(arm_name).stop())


def gripper_stop(arm_name: str) -> None:
    grip_name = RIGS[arm_name][1]
    run(_robot.gripper(grip_name).stop())


def camera_frame(cam_name: str) -> bytes:
    """Capture a JPEG color frame from a camera. Returns raw JPEG bytes."""
    images, _ = run(_robot.camera(cam_name).get_images())
    by_name = {i.name: i for i in images}
    image = next((im for im in images if "color" in (im.name or "").lower()
                  and ("jpeg" in (im.mime_type or "") or "png" in (im.mime_type or ""))), None)
    if image is None:
        raise RuntimeError(f"No color image from camera {cam_name}")
    return image.data


def list_arm_names() -> list[str]:
    """Return the arm resource names configured on the machine."""
    robot = run(_robot.connect())
    return [r.name for r in robot.resource_names if r.type == "component" and r.subtype == "arm"]


def list_all_resource_names() -> dict[str, list[str]]:
    robot = run(_robot.connect())
    return {
        subtype: [r.name for r in robot.resource_names if r.type == "component" and r.subtype == subtype]
        for subtype in ("arm", "camera", "gripper")
    }


# ---------------------------------------------------------------------------
# Vision: color + depth capture, frame transforms, YOLO detections
# ---------------------------------------------------------------------------
def world_point(frame_name: str, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    """A point given in `frame_name` -> world coordinates (mm)."""
    q = PoseInFrame(reference_frame=frame_name,
                    pose=Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=1, theta=0))
    r = run(_robot._sync_robot().transform_pose(q, "world")).pose
    return np.array([r.x, r.y, r.z])


def capture_frame(cam_name: str, cam_frame: str, arm_name: str) -> dict:
    """One color frame (+ depth) plus camera pose and intrinsics. Read-only."""

    async def grab():
        cam = _robot.camera(cam_name)
        images, _ = await cam.get_images()
        props = await cam.get_properties()
        return images, props

    images, props = run(grab())
    pick = next((im for im in images if "color" in (getattr(im, "name", "") or "").lower()), None)
    if pick is None:
        pick = next(im for im in images if "jpeg" in im.mime_type or "png" in im.mime_type)
    bgr = cv2.imdecode(np.frombuffer(pick.data, np.uint8), cv2.IMREAD_COLOR)

    dep = next((im for im in images if "dep" in im.mime_type), None)
    depth = None
    depth_matches_color = False
    if dep is not None:
        depth = np.asarray(dep.bytes_to_depth_array(), dtype=np.uint16)
        depth_matches_color = depth.shape == bgr.shape[:2]
        if depth.shape != bgr.shape[:2]:
            depth = cv2.resize(depth, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    k = props.intrinsic_parameters
    origin = world_point(cam_frame)
    axes = np.stack([(world_point(cam_frame, *a) - origin) / 1000.0
                     for a in ((1000, 0, 0), (0, 1000, 0), (0, 0, 1000))], axis=1)
    return {"bgr": bgr, "depth": depth, "depth_matches_color": depth_matches_color, "arm": arm_name,
            "fx": float(k.focal_x_px), "fy": float(k.focal_y_px),
            "cx": float(k.center_x_px), "cy": float(k.center_y_px),
            "origin": origin, "R": axes,
            "time": time.strftime("%H:%M:%S")}


def _strip_depth(depth, box, lo, hi, rows=(0.45, 0.85), half=0.15):
    x0, y0, x1, y1 = box
    u, w, h = (x0 + x1) / 2, x1 - x0, y1 - y0
    c0, c1 = int(round(u - half * w)), int(round(u + half * w)) + 1
    r0, r1 = int(round(y0 + rows[0] * h)), int(round(y0 + rows[1] * h)) + 1
    roi = depth[max(r0, 0):r1, max(c0, 0):c1].astype(float)
    vals = roi[(roi >= lo) & (roi <= hi)]
    frac = vals.size / max(roi.size, 1)
    if vals.size < 30:
        return None, frac, None, None
    near = np.percentile(vals, 20)
    front = vals[vals <= near + 40]
    return float(np.median(front)), frac, float(np.percentile(front, 75) - np.percentile(front, 25)), (r0 + r1) / 2


def _row_width_px(cap, box, v, z, radius):
    x0, y0, x1, y1 = box
    r = int(round(v))
    if not 0 <= r < cap["bgr"].shape[0]:
        return None
    hsv = cv2.cvtColor(cap["bgr"][r:r + 1], cv2.COLOR_BGR2HSV)[0]
    dep = cap["depth"][r].astype(float)
    blue = (hsv[:, 0] >= 95) & (hsv[:, 0] <= 135) & (hsv[:, 1] >= 70) & (hsv[:, 2] >= 25)
    near = (dep >= z - 25) & (dep <= z + radius + 25)
    ok = blue | near
    u = int(round((x0 + x1) / 2))
    lo, hi = max(int(x0) - 15, 0), min(int(x1) + 15, len(ok) - 1)

    def grow(step):
        i, miss, last = u, 0, u
        while lo <= i + step <= hi and miss < 4:
            i += step
            if ok[i]:
                last, miss = i, 0
            else:
                miss += 1
        return last
    return grow(-1), grow(1)


def _surface_to_world(cap, det, z, radius):
    u, v = det["px"]
    surface = np.array([(u - cap["cx"]) / cap["fx"] * z, (v - cap["cy"]) / cap["fy"] * z, z])
    axis = surface + radius * surface / np.linalg.norm(surface)
    return cap["origin"] + cap["R"] @ axis



def yolo_detect(service_name: str, jpeg_bytes: bytes) -> list[dict]:
    """Run YOLO on a JPEG image; return list of {class_name, confidence, x_min, y_min, x_max, y_max}."""
    detector = VisionClient.from_robot(_robot._sync_robot(), service_name)
    dets = run(detector.get_detections(ViamImage(jpeg_bytes, CameraMimeType.JPEG)))
    out = []
    for d in dets:
        out.append({
            "class_name": d.class_name,
            "confidence": float(d.confidence),
            "x_min": float(d.x_min), "y_min": float(d.y_min),
            "x_max": float(d.x_max), "y_max": float(d.y_max),
        })
    return out
