"""Regrasp unscrew: twist the cap off with the wrist joint — backend port.

Pure robot logic (no HTTP, no Streamlit). Uses backend/robot.py for all Viam calls.
Unscrewer.start(params) / stop() / status() mirror the original class.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np
from viam.proto.common import Pose, PoseInFrame, Transform

from robot import (
    arm_do_command, arm_stop, gripper_stop, end_position, gripper_action, gripper_do_command,
    joints, move_joint_positions, move_position, transform_pose,
)

WRIST = 5
_LEVEL_FRAME = "_level_target"


@dataclass
class UnscrewParams:
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
    start_joints: tuple = ()
    start_target: tuple = ()
    dry_run: bool = False


def describe(p: UnscrewParams):
    turn = "counter-clockwise" if p.loosen_sign < 0 else "clockwise"
    steps = [f"Slow arm to {p.speed_deg_s:g}°/s, set grip force to {p.grip_force_pct}%",
             f"Open gripper to {p.release_pos}"]
    if p.start_joints:
        steps.append(f"Move to **{p.start_pose}** ({', '.join(f'{v:.1f}' for v in p.start_joints)}°)")
    elif p.start_target:
        x, y, z, ox, oy, oz, th = p.start_target
        steps.append(f"Move to **{p.start_pose}**: x {x:.1f}, y {y:.1f}, z {z:.1f} mm, "
                     f"orientation ({ox:.3f}, {oy:.3f}, {oz:.3f}) θ {th:.1f}° (motion planner)")
    steps.append(f"Level the tool to point straight down (pivot at the gripper, roll kept; abort if more than "
                 f"{p.max_level_tilt_deg:g}° off)")
    for i in range(1, p.twists):
        steps.append(f"{i}. Grab → unscrew {p.twist_deg:g}° {turn} (seen from above) → open to {p.release_pos} "
                     "→ twist back (same height)")
    steps.append(f"{p.twists}. Grab → unscrew {p.twist_deg:g}° {turn} → keep holding and lift {p.final_lift_mm:g} mm; "
                 "end there")
    if p.dry_run:
        steps.append("Dry run: empty grabs are allowed")
    steps.append(f"Restore speed to {p.normal_speed_deg_s:g}°/s and grip force to {p.normal_grip_force_pct}%")
    return steps


class Unscrewer:
    def __init__(self, arm_name="arm2", gripper_name="gripper2"):
        self.arm_name = arm_name
        self.gripper_name = gripper_name
        self.base = f"{arm_name}_origin"
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

    def check(self, max_tilt_deg=15.0):
        tilt = self._tilt_deg(end_position(self.arm_name))
        if tilt > max_tilt_deg:
            return (f"Tool is {tilt:.1f}° from pointing straight down (limit {max_tilt_deg:g}°); "
                    "re-teach the start pose closer to vertical.")
        return None

    # ---------- leveling geometry (frame system; nothing here moves) ----------
    def _tf(self, src, pose_dict, dst, extra=None):
        return transform_pose(src, pose_dict, dst, extra)

    def _point(self, src, x=0.0, y=0.0, z=0.0, dst=None, extra=None):
        p = self._tf(src, {"x": x, "y": y, "z": z, "o_z": 1.0}, dst or self.base, extra)
        return np.array([p["x"], p["y"], p["z"]])

    @staticmethod
    def _tilt_deg(end):
        return math.degrees(math.acos(max(-1.0, min(1.0, -end["o_z"]))))

    def tool_length(self):
        return float(np.linalg.norm(self._point(self.gripper_name, dst=self.arm_name)))

    def level_plan(self, max_tilt_deg, tool=None, extra=None):
        tool = tool or self.arm_name
        L = self.tool_length()
        flange = self._point(tool, extra=extra)
        below = self._point(self.base, *(flange - [0, 0, 100]), dst=tool, extra=extra)
        d = below / np.linalg.norm(below)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, d[2]))))
        if tilt > max_tilt_deg:
            raise RuntimeError(f"Tool is {tilt:.1f}° from pointing straight down (limit {max_tilt_deg:g}°); "
                               "re-teach the start pose closer to vertical.")
        if tilt < 1.0:
            return {"tilt": tilt, "target": None}
        lon = math.degrees(math.atan2(d[1], d[0]))
        rel = Pose(x=-L * d[0], y=-L * d[1], z=L * (1 - d[2]), o_x=d[0], o_y=d[1], o_z=d[2], theta=-lon)
        target = self._tf(tool, {"x": rel.x, "y": rel.y, "z": rel.z,
                                 "o_x": rel.o_x, "o_y": rel.o_y, "o_z": rel.o_z, "theta": rel.theta}, self.base, extra)
        vt = [Transform(reference_frame=_LEVEL_FRAME,
                        pose_in_observer_frame=PoseInFrame(reference_frame=tool, pose=rel))] + list(extra or [])
        origin_new = self._point(_LEVEL_FRAME, extra=vt)
        axis_new = self._point(_LEVEL_FRAME, z=100, extra=vt) - origin_new
        pivot_shift = np.linalg.norm(self._point(_LEVEL_FRAME, z=L, extra=vt) - self._point(tool, z=L, extra=extra))
        x_old = self._point(tool, x=100, extra=extra) - flange
        x_new = self._point(_LEVEL_FRAME, x=100, extra=vt) - origin_new
        roll = abs((math.degrees(math.atan2(x_new[1], x_new[0]) - math.atan2(x_old[1], x_old[0])) + 180) % 360 - 180)
        down_err = math.degrees(math.acos(max(-1.0, min(1.0, -axis_new[2] / 100))))
        if down_err > 0.5 or pivot_shift > 1.0 or roll > tilt + 1.0:
            raise RuntimeError(f"leveling check failed (axis {down_err:.2f}° off, pivot moved {pivot_shift:.2f} mm, "
                               f"roll changed {roll:.1f}°); not moving")
        return {"tilt": tilt, "target": target, "roll_change": roll}

    # ---------- control ----------
    def start(self, params: UnscrewParams):
        if self.running:
            return
        self._stop.clear()
        self._set(state="running", step="starting", error=None)
        self._thread = threading.Thread(target=self._sequence, args=(params,), name="cap-unscrew", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        for part_name, part_fn in ((self.arm_name, arm_stop), (self.arm_name, gripper_stop)):
            try:
                part_fn(part_name)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=10)

    # ---------- internals ----------
    def _set(self, **kw):
        with self._lock:
            self._status.update(kw)

    def _do(self, label, coro_fn):
        if self._stop.is_set():
            raise InterruptedError("stopped")
        self._set(step=label)
        return coro_fn()

    def _joints(self):
        return list(joints(self.arm_name))

    def _rotate_wrist(self, label, delta_deg):
        j = self._joints()
        j[WRIST] += delta_deg
        self._do(label, lambda: move_joint_positions(self.arm_name, j))

    def _lift(self, label, mm):
        if mm == 0:
            return
        e = end_position(self.arm_name)
        target = Pose(x=e["x"], y=e["y"], z=e["z"] + mm,
                      o_x=e["o_x"], o_y=e["o_y"], o_z=e["o_z"], theta=e["theta"])
        self._do(label, lambda: move_position(self.arm_name, target.x, target.y, target.z,
                                              target.o_x, target.o_y, target.o_z, target.theta))

    def _open(self, label, pos):
        if pos >= 850:
            self._do(label, lambda: gripper_action(self.arm_name, "open"))
        else:
            self._do(label, lambda: gripper_do_command(self.arm_name, {"set": float(pos)}))

    def _grab(self, label, require=True):
        holding = self._do(label, lambda: gripper_action(self.arm_name, "grab"))
        if require and holding is False:
            raise RuntimeError(f"{label}: gripper closed on nothing; is it around the cap?")

    def _level(self, p):
        problem = self.check(p.max_level_tilt_deg)
        if problem:
            raise RuntimeError(problem)
        if self._tilt_deg(end_position(self.arm_name)) < 1.0:
            return
        plan = self.level_plan(p.max_level_tilt_deg)
        if plan["target"] is not None:
            t = plan["target"]
            self._do(f"leveling tool ({plan['tilt']:.1f}° off vertical)",
                     lambda: move_position(self.arm_name, t["x"], t["y"], t["z"],
                                           t["o_x"], t["o_y"], t["o_z"], t["theta"]))

    def _sequence(self, p: UnscrewParams):
        twist = p.loosen_sign * p.twist_deg
        try:
            self._do("slowing arm", lambda: arm_do_command(self.arm_name, {"set_speed": p.speed_deg_s}))
            self._do("setting grip force", lambda: gripper_do_command(self.arm_name, {"set_gripper_torque": p.grip_force_pct}))
            self._open("opening gripper", p.release_pos)
            if p.start_joints:
                self._do(f"moving to {p.start_pose}",
                         lambda: move_joint_positions(self.arm_name, list(p.start_joints)))
            elif p.start_target:
                x, y, z, ox, oy, oz, th = p.start_target
                target = Pose(x=x, y=y, z=z, o_x=ox, o_y=oy, o_z=oz, theta=th)
                self._do(f"moving to {p.start_pose}",
                         lambda: move_position(self.arm_name, target.x, target.y, target.z,
                                               target.o_x, target.o_y, target.o_z, target.theta))
            self._level(p)
            for i in range(1, p.twists + 1):
                tag = f"twist {i}/{p.twists}"
                self._grab(f"{tag}: grab", require=not p.dry_run)
                self._rotate_wrist(f"{tag}: unscrewing", twist)
                if i < p.twists:
                    self._open(f"{tag}: open", p.release_pos)
                    self._rotate_wrist(f"{tag}: twisting back", -twist)
            self._lift("lifting cap", p.final_lift_mm)
            self._set(state="done", step="complete")
        except Exception as e:
            for fn in (arm_stop, gripper_stop):
                try:
                    fn(self.arm_name)
                except Exception:
                    pass
            self._set(state="stopped" if isinstance(e, InterruptedError) else "error", error=str(e))
        finally:
            for what, fn, cmd in (("speed", arm_do_command, {"set_speed": p.normal_speed_deg_s}),
                                  ("grip force", gripper_do_command, {"set_gripper_torque": p.normal_grip_force_pct})):
                try:
                    fn(self.arm_name, cmd)
                except Exception as ex:
                    self._set(error=f"could not restore {what}: {ex}")
