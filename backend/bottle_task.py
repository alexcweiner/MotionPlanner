"""Full bottle-opening run: keyframe sequence -> unscrew -> retreat up -> via poses -> drop the cap at the trash pose.

Backend port of bottle_task.py. Runs in a background thread so the FastAPI server stays responsive.
Stop / STOP ALL can interrupt it.
"""
from __future__ import annotations

import threading
from viam.proto.common import Pose

from robot import (
    arm_stop, end_position, gripper_action, move_joint_positions, move_position,
)

POSE_KEYS = ("x", "y", "z", "o_x", "o_y", "o_z", "theta")


def go_to_keyframe(arm_name: str, kf: dict):
    """Joint keyframes ("values") move in joint space. XYZ keyframes ("pose") go through the motion planner."""
    if kf.get("pose") is not None:
        p = kf["pose"]
        move_position(arm_name, float(p["x"]), float(p["y"]), float(p["z"]),
                      float(p["o_x"]), float(p["o_y"]), float(p["o_z"]), float(p["theta"]))
    else:
        move_joint_positions(arm_name, list(kf["values"]))


def apply_gripper(arm_name: str, action: str):
    if action == "grab":
        gripper_action(arm_name, "grab")
    elif action == "open":
        gripper_action(arm_name, "open")


class BottleTask:
    def __init__(self, arms, grippers, unscrewer):
        self.arms, self.grippers, self.unscrewer = arms, grippers, unscrewer
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._status = {"state": "idle", "step": "", "error": None}

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def status(self):
        with self._lock:
            return dict(self._status)

    def start(self, keyframes, unscrew_params, drop, retreat=None, via=(), alongside=None):
        if self.running:
            return
        self._stop.clear()
        self._set(state="running", step="starting", error=None)
        self._thread = threading.Thread(target=self._sequence,
                                        args=(list(keyframes), unscrew_params, drop, retreat, list(via), alongside),
                                        name="bottle-task", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self.unscrewer.stop()
        self._halt()
        if self._thread is not None:
            self._thread.join(timeout=10)

    # ---------- internals ----------
    def _set(self, **kw):
        with self._lock:
            self._status.update(kw)

    def _halt(self):
        for arm_name in self.arms:
            try:
                arm_stop(arm_name)
            except Exception:
                pass

    def _check(self):
        if self._stop.is_set():
            raise InterruptedError("stopped")

    def _keyframe(self, label, kf, move_only=False):
        self._check()
        self._set(step=f"{label}: moving {kf['arm']} to {kf['name']}")
        go_to_keyframe(kf["arm"], kf)
        if move_only:
            return
        if kf.get("gripper"):
            self._check()
            self._set(step=f"{label}: gripper {kf['gripper']} at {kf['name']}")
            apply_gripper(kf["arm"], kf["gripper"])
        if kf.get("delay", 0) > 0 and self._stop.wait(kf["delay"]):
            raise InterruptedError("stopped")

    def _rise_to(self, kf):
        """Straight up to kf's height, same x, y and orientation as now."""
        self._check()
        arm_name = kf["arm"]
        e = end_position(arm_name)
        z = float(kf["pose"]["z"]) if "pose" in kf else None
        if z is None or z <= e["z"]:
            return
        self._set(step=f"retreat: {arm_name} up {z - e['z']:.0f} mm to {kf['name']} height, holding the cap")
        target = Pose(x=e["x"], y=e["y"], z=z, o_x=e["o_x"], o_y=e["o_y"], o_z=e["o_z"], theta=e["theta"])
        move_position(arm_name, target.x, target.y, target.z, target.o_x, target.o_y, target.o_z, target.theta)

    def _move_and_grip(self, kf, errors):
        try:
            go_to_keyframe(kf["arm"], kf)
            if kf.get("gripper") and not self._stop.is_set():
                apply_gripper(kf["arm"], kf["gripper"])
        except Exception as e:
            errors.append(e)

    def _sequence(self, keyframes, unscrew_params, drop, retreat, via, alongside):
        side, errors = None, []

        def wait_for_side(reason):
            nonlocal side
            if side is not None:
                if side.is_alive():
                    self._set(step=f"waiting for {alongside['arm']} to reach {alongside['name']} ({reason})")
                side.join()
                side = None
            if errors:
                raise RuntimeError(f"{alongside['arm']} to {alongside['name']}: {errors[0]}")

        try:
            for n, kf in enumerate(keyframes, 1):
                label = f"1/3 keyframes {n}/{len(keyframes)}"
                if n == 1 and alongside is not None:
                    self._check()
                    side = threading.Thread(target=self._move_and_grip, args=(alongside, errors), daemon=True)
                    side.start()
                    label += f" (and {alongside['arm']} → {alongside['name']})"
                elif alongside is not None and kf["arm"] == alongside["arm"]:
                    wait_for_side(f"before its {kf['name']}")
                self._keyframe(label, kf)
                if n == 1:
                    wait_for_side(f"{kf['arm']} waits at {kf['name']}")
            if alongside is not None:
                wait_for_side("before the unscrew")

            self._check()
            self.unscrewer.start(unscrew_params)
            while self.unscrewer.running:
                self._set(step=f"2/3 unscrew: {self.unscrewer.status()['step']}")
                if self._stop.wait(0.2):
                    raise InterruptedError("stopped")
            u = self.unscrewer.status()
            if u["state"] != "done":
                raise RuntimeError(f"unscrew {u['state']} at '{u['step']}': {u['error']}")

            if retreat is not None:
                self._rise_to(retreat)
            for kf in via:
                self._keyframe("3/3 via (holding cap)", kf, move_only=True)
            self._keyframe("3/3 drop", drop)
            self._set(state="done", step="complete")
        except Exception as e:
            self._halt()
            if side is not None:
                side.join(timeout=10)
            self._set(state="stopped" if isinstance(e, InterruptedError) else "error", error=str(e))
