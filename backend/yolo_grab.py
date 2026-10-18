"""Head-on YOLO grab of the leftmost bottle with arm 1 — backend port.

Pure robot logic (no HTTP, no Streamlit). Uses backend/robot.py for all Viam calls.
pick_and_place() / reset_arm() are the routines behind the frontend's Simple tab.
"""
from __future__ import annotations

import math
from typing import Callable

import cv2
import numpy as np

from robot import (
    capture_frame, yolo_detect, world_point, _surface_to_world, _strip_depth,
    joints, end_position, move_joint_positions, move_position, gripper_action,
    arm_do_command, gripper_do_command, RIGS,
)

POSE_KEYS = ("x", "y", "z", "o_x", "o_y", "o_z", "theta")
DEFAULTS = {
    "service": "bottle-yolo",
    "conf": 0.5,
    "frames": 3,
    "radius": 40.0,
    "grasp_frac": 0.35,
    "pad_mm": 60.0,
    "tip_mm": 67.0,
    "pre_mm": 100.0,
    "stroke_mm": 84.0,
    "bottle_diam_mm": 80.0,
    "bottle_h_mm": 304.8,   # Saratoga 28 fl oz, bottom to cap top (same as cap_line_search.BOTTLE_HEIGHT_MM)
    "size_tol": 0.25,       # drop a frame whose depth makes the bottle more than this fraction off that height
    "max_width_mm": 115.0,
    "finger_mm": 12.0,
    "grip_half_h_mm": 40.0,
    "max_spread_mm": 10.0,
    "depth_lo": 250, "depth_hi": 1200,
    "pre_step_mm": 40.0,
    "in_step_mm": 20.0,
    "lat_trim_mm": 18.0,
    "depth_trim_mm": 20.0,
    "level": True,
    "backup_mm": 100.0,
    "max_backups": 2,
}


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _cam_point(cap, u, v, Z):
    p = np.array([(u - cap["cx"]) / cap["fx"] * Z, (v - cap["cy"]) / cap["fy"] * Z, Z])
    return cap["origin"] + cap["R"] @ p


def project(cap, world):
    p = cap["R"].T @ (np.asarray(world, float) - cap["origin"])
    return cap["fx"] * p[0] / p[2] + cap["cx"], cap["fy"] * p[1] / p[2] + cap["cy"]


PROFILE = (0.25, 0.35, 0.45, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85)


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


def _profile(cap, box, z, radius):
    x0, y0, x1, y1 = box
    out = {}
    for f in PROFILE:
        v = y1 - f * (y1 - y0)
        lr = _row_width_px(cap, box, v, z, radius)
        if lr:
            a, b = _cam_point(cap, lr[0] - .5, v, z + radius), _cam_point(cap, lr[1] + .5, v, z + radius)
            out[f] = float(np.linalg.norm(b - a))
    return out


def _bottle(cap, d, cfg, radius, W, H):
    box = (d["x_min"], d["y_min"], d["x_max"], d["y_max"])
    z, frac, spread, v_mid = _strip_depth(cap["depth"], box, cfg["depth_lo"], cfg["depth_hi"])
    if z is None:
        return None
    u, Zc = (d["x_min"] + d["x_max"]) / 2, z + radius
    left, right = _cam_point(cap, d["x_min"], v_mid, Zc), _cam_point(cap, d["x_max"], v_mid, Zc)
    return {"box": [int(b) for b in box], "conf": float(d["confidence"]), "z": z, "valid": frac, "spread": spread,
            "axis": _surface_to_world(cap, {"px": (u, v_mid)}, z, radius),
            "surface": _cam_point(cap, u, v_mid, z),
            "top": _cam_point(cap, u, d["y_min"], Zc), "bottom": _cam_point(cap, u, d["y_max"], Zc),
            "left": left, "right": right,
            "width_mm": float(np.linalg.norm(right - left)),
            "profile": _profile(cap, box, z, radius) if radius > 0 else {},
            "edges": [name for name, cut in (("left", d["x_min"] <= 2), ("top", d["y_min"] <= 2),
                                             ("right", d["x_max"] >= W - 3), ("bottom", d["y_max"] >= H - 3)) if cut]}


def _size_mismatch(b, cfg):
    """Why this frame's depth can't be right, or None. The bottle's height in mm follows from its box height and
    the depth used, so depth read through the glass (the background behind it) makes it far too tall. Skipped when
    the box is cut off at the top or bottom; locate() backs up for that."""
    if {"top", "bottom"} & set(b["edges"]):
        return None
    h, H = float(np.linalg.norm(np.asarray(b["top"]) - np.asarray(b["bottom"]))), cfg["bottle_h_mm"]
    if abs(h / H - 1) > cfg["size_tol"]:
        return f"depth {b['z']:.0f} mm makes the bottle {h:.0f} mm tall (it is {H:.0f})"
    return None


def frame_axes(arm_name, cam_frame, grip_frame):
    o = world_point(grip_frame)
    ax = [(world_point(grip_frame, *a) - o) / 100.0 for a in ((100, 0, 0), (0, 100, 0), (0, 0, 100))]
    return {"origin": o, "x": ax[0], "y": ax[1], "z": ax[2]}


def measure(arm_name, cam_name, cam_frame, grip_frame, cfg, check=None):
    """Read-only. Raises ValueError with a plain reason when the bottle can't be measured reliably."""
    import base64
    import io
    joints_before = list(joints(arm_name))
    per, last, dropped = [], None, []
    for _ in range(int(cfg["frames"])):
        if check:
            check()
        cap = capture_frame(cam_name, cam_frame, arm_name)
        if cap["depth"] is None or not cap["depth_matches_color"]:
            raise ValueError("Need aligned, equal-resolution color and depth from the camera.")
        ok, enc = cv2.imencode(".jpg", cap["bgr"])
        jpeg = enc.tobytes()
        dets = yolo_detect(cfg["service"], jpeg)
        H, W = cap["bgr"].shape[:2]
        bottles = sorted((d for d in dets if d["class_name"].lower() == "bottle" and d["confidence"] >= cfg["conf"]),
                         key=lambda d: d["x_min"] + d["x_max"])
        last = (cap, bottles)
        t = _bottle(cap, bottles[0], cfg, cfg["radius"], W, H) if bottles else None
        n = _bottle(cap, bottles[1], cfg, 0.0, W, H) if len(bottles) > 1 else None
        if t is not None and (why := _size_mismatch(t, cfg)):
            dropped.append(why)
            t = None
        if n is not None and _size_mismatch(n, cfg):
            n = None
        per.append({"target": t, "neighbor": n, "count": len(bottles)} if t else None)
    if np.max(np.abs(np.array(joints(arm_name)) - joints_before)) > 0.2:
        raise ValueError("The arm moved while measuring; measure again with the arm still.")
    good = [p for p in per if p]
    need = max(2, int(cfg["frames"]) // 2 + 1)
    if len(good) < need:
        raise ValueError(f"The leftmost bottle had usable depth in only {len(good)} of {int(cfg['frames'])} frames "
                         f"(need {need}). Check the bottle is in view and 25-120 cm from the camera."
                         + (f" Dropped: {'; '.join(dropped)}." if dropped else ""))

    def med(key, who="target"):
        vals = [p[who][key] for p in good if p[who] is not None]
        return np.median(np.array(vals, float), axis=0) if vals else None

    axes = np.array([p["target"]["axis"] for p in good])
    axis = np.median(axes, axis=0)
    out = {"frames": len(good), "of": int(cfg["frames"]),
           "spread_mm": float(np.max(np.linalg.norm(axes[:, :2] - axis[:2], axis=1))),
           "axis": axis, "surface": med("surface"), "top": med("top"), "bottom": med("bottom"),
           "width_mm": float(med("width_mm")), "z": float(med("z")), "valid": float(med("valid")),
           "depth_spread": float(med("spread")), "conf": float(med("conf")),
           "clipped_edges": sorted({edge for p in good for edge in p["target"]["edges"]}),
           "counts": [p["count"] for p in good],
           "z_frames": [float(p["target"]["z"]) for p in good], "dropped": dropped,
           "neighbor": None, "joints": joints_before}
    out["profile"] = {f: float(np.median(v)) for f in PROFILE
                      if len(v := [p["target"]["profile"][f] for p in good if f in p["target"]["profile"]]) >= need}
    out["clipped"] = bool(out["clipped_edges"])
    nb = [p["neighbor"] for p in good if p["neighbor"] is not None]
    if nb:
        out["neighbor"] = {k: np.median(np.array([n[k] for n in nb], float), axis=0)
                           for k in ("left", "right", "surface", "z", "width_mm", "top")}
        out["neighbor"]["frames"] = len(nb)
    end = end_position(arm_name)
    out["end"] = {k: float(end[k]) for k in POSE_KEYS}
    out["gripper"] = frame_axes(arm_name, cam_frame, grip_frame)
    out["cap"], out["boxes"] = last[0], [(d["x_min"], d["y_min"], d["x_max"], d["y_max"], d["confidence"]) for d in last[1]]
    return out


def plan(m, cfg):
    """Pre-grasp and grasp poses for the arm's end (flange), plus safety checks."""
    g, up, e = m["gripper"], np.array([0.0, 0.0, 1.0]), m["end"]
    R = ov_to_R(e)
    if cfg.get("level", False):
        z = _unit(R[:, 2] - R[:, 2].dot(up) * up)
        x = up if R[:, 0].dot(up) >= 0 else -up
        R_d = np.column_stack([x, np.cross(z, x), z])
    else:
        R_d = R
    a = R_d[:, 2]
    a_h = _unit(a - a.dot(up) * up)
    lat = np.cross(up, a_h)
    e_pos = np.array([e["x"], e["y"], e["z"]])
    reach = float((np.asarray(g["origin"]) - e_pos).dot(R[:, 2])) + cfg["pad_mm"]
    top, bottom = m["top"][2], m["bottom"][2]
    zg = bottom + cfg["grasp_frac"] * (top - bottom)
    A = (np.array([m["axis"][0], m["axis"][1], zg]) - cfg.get("lat_trim_mm", 0.0) * lat
         + cfg.get("depth_trim_mm", 0.0) * a_h)
    T0 = e_pos + reach * R[:, 2]
    ori = R_to_ov(R_d)
    at = lambda q: dict(ori, x=float(q[0]), y=float(q[1]), z=float(q[2]))
    grab_pos, pre_pos = A - reach * a, A - (reach + cfg["pre_mm"]) * a
    d_grab, d_pre = grab_pos - e_pos, pre_pos - e_pos
    p = {"A": A, "zg": zg, "a": a, "lat": lat, "T0": T0, "pre": at(pre_pos), "grab": at(grab_pos), "reach": reach,
         "d_pre": d_pre, "d_grab": d_grab, "height_mm": float(top - bottom), "checks": [],
         "turn_deg": math.degrees(math.acos(float(np.clip((np.trace(R.T @ R_d) - 1) / 2, -1, 1))))}
    c = p["checks"]
    lateral_now = float((A - T0).dot(lat))
    vertical_now = float(A[2] - T0[2])
    p["offset_now"] = (lateral_now, vertical_now, float((A - T0).dot(a)))
    if cfg.get("lat_trim_mm") or cfg.get("depth_trim_mm"):
        c.append(("ok", f"Trims applied: {cfg.get('lat_trim_mm', 0):+.0f} mm right, "
                        f"{cfg.get('depth_trim_mm', 0):+.0f} mm deeper than the bottle axis."))
    if p["turn_deg"] > 10:
        c.append(("error", f"Levelling would turn the gripper {p['turn_deg']:.0f}°; over the 10° limit."))
    if m["frames"] < m["of"]:
        c.append(("warn", f"Bottle measured in {m['frames']}/{m['of']} frames."
                          + (f" Dropped: {'; '.join(m['dropped'])}." if m.get("dropped") else "")))
    if m["spread_mm"] > cfg["max_spread_mm"]:
        zs = ", ".join(f"{v:.0f}" for v in m.get("z_frames", []))
        c.append(("error", f"Frames disagree by {m['spread_mm']:.0f} mm (limit {cfg['max_spread_mm']:.0f}); "
                           f"detection is unstable (depth per frame: {zs} mm). Nothing will move."))
    if m["clipped"]:
        c.append(("error", f"The leftmost bottle touches the {', '.join(m['clipped_edges'])} edge of the image, "
                           "so its box may be cut off."))
    if not 150 <= p["height_mm"] <= 450:
        c.append(("error", f"Bottle height reads {p['height_mm']:.0f} mm, which is implausible. Nothing will move."))
    prof = m.get("profile", {})
    near_f = [f for f in sorted(prof) if abs(f - cfg["grasp_frac"]) <= 0.051]
    seen = max(prof[f] for f in near_f) if near_f else m["width_mm"]
    p["grip_width"] = seen
    diam = cfg["bottle_diam_mm"]
    side = (cfg["stroke_mm"] - diam) / 2
    if seen > cfg["max_width_mm"]:
        c.append(("error", f"The target reads {seen:.0f} mm wide, far over the known {diam:.0f} mm bottle; it may be "
                           "two bottles in one detection. Nothing will move."))
    else:
        c.append(("ok", f"Known {diam:.0f} mm bottle in a {cfg['stroke_mm']:.0f} mm opening ({side:.0f} mm per side); "
                        f"it reads {seen:.0f} mm in the image."))
    n = m.get("neighbor")
    if n is not None:
        edges = [n["left"], n["right"]]
        s_t = [((E - A).dot(lat), (E - A).dot(a)) for E in edges]
        s, t = min(s_t, key=lambda st: abs(st[0]))
        outer = cfg["stroke_mm"] / 2 + cfg["finger_mm"]
        tip = cfg["tip_mm"] - cfg["pad_mm"]
        p["neighbor_gap"] = (float(abs(s) - outer), float(t - tip))
        below = zg - cfg["grip_half_h_mm"] - n["top"][2]
        p["neighbor_below"] = float(below)
        if below >= 15:
            c.append(("ok", f"Neighbour is {below:.0f} mm below the lowest part of the gripper at this grasp height "
                            f"(its top at z {n['top'][2]:.0f} mm)."))
        elif abs(s) < outer + 5 and t < tip + 10:
            c.append(("error", f"The neighbour's edge is {abs(s):.0f} mm to the side of the bottle axis and "
                               f"{t:+.0f} mm along the approach; an open finger reaches {outer:.0f} mm to the side "
                               f"and {tip:+.0f} mm along. The finger would hit it."))
        else:
            c.append(("ok", f"Neighbour clear: edge {abs(s):.0f} mm to the side (finger reaches {outer:.0f}), "
                            f"{t:+.0f} mm along the approach (fingertips reach {tip:+.0f})."))
    if np.linalg.norm(d_pre) > 700:
        c.append(("error", f"Pre-grasp is {np.linalg.norm(d_pre):.0f} mm away; over the 700 mm limit."))
    if (A - T0).dot(a) < cfg["pre_mm"] - 1:
        c.append(("error", "The bottle is already closer than the pre-grasp distance; back off first."))
    return p


def steps(start, goal, step_mm):
    s = np.array([start[k] for k in ("x", "y", "z")])
    g = np.array([goal[k] for k in ("x", "y", "z")])
    n = max(1, math.ceil(np.linalg.norm(g - s) / step_mm))
    return [dict(goal, x=float(q[0]), y=float(q[1]), z=float(q[2]))
            for q in (s + (g - s) * i / n for i in range(1, n + 1))]


def annotate(m, p):
    cap = m["cap"]
    img = cap["bgr"].copy()
    for i, (x0, y0, x1, y1, conf) in enumerate(m["boxes"]):
        color = (0, 255, 0) if i == 0 else (0, 200, 255)
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
        cv2.putText(img, f"{'TARGET' if i == 0 else f'bottle {i + 1}'} {conf:.0%}", (int(x0), max(20, int(y0) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2)
    if p is not None:
        for pt, col in ((p["A"], (0, 0, 255)), (p["A"] - p["lat"] * 42, (255, 0, 0)),
                        (p["A"] + p["lat"] * 42, (255, 0, 0))):
            u, v = project(cap, pt)
            cv2.drawMarker(img, (int(u), int(v)), col, cv2.MARKER_CROSS, 28, 3)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------- gripper attitude ----------
def _Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def ov_to_R(e):
    o = _unit([e["o_x"], e["o_y"], e["o_z"]])
    lat = math.acos(float(np.clip(o[2], -1, 1)))
    lon = math.atan2(o[1], o[0]) if 1 - abs(o[2]) > 1e-9 else 0.0
    return _Rz(lon) @ _Ry(lat) @ _Rz(math.radians(e["theta"]))


def R_to_ov(R):
    o = R[:, 2]
    lat = math.acos(float(np.clip(o[2], -1, 1)))
    lon = math.atan2(o[1], o[0]) if 1 - abs(o[2]) > 1e-9 else 0.0
    M = (_Rz(lon) @ _Ry(lat)).T @ R
    return {"o_x": float(o[0]), "o_y": float(o[1]), "o_z": float(o[2]),
            "theta": math.degrees(math.atan2(M[1, 0], M[0, 0]))}


def attitude(e):
    R = ov_to_R(e)
    z, y = R[:, 2], R[:, 1]
    return (math.degrees(math.asin(float(np.clip(z[2], -1, 1)))), math.degrees(math.asin(float(np.clip(y[2], -1, 1)))),
            math.degrees(math.atan2(z[1], z[0])))


def _pivot_target(e, R_d, reach_mm, max_deg):
    R = ov_to_R(e)
    turn = math.degrees(math.acos(float(np.clip((np.trace(R.T @ R_d) - 1) / 2, -1, 1))))
    if turn > max_deg:
        raise ValueError(f"That is a {turn:.1f}° turn; refusing anything over {max_deg:.0f}° in one move.")
    P = np.array([e["x"], e["y"], e["z"]]) + reach_mm * R[:, 2]
    pos = P - reach_mm * R_d[:, 2]
    return dict(R_to_ov(R_d), x=float(pos[0]), y=float(pos[1]), z=float(pos[2])), turn


def level_target(e, reach_mm, max_deg=10.0):
    R = ov_to_R(e)
    up = np.array([0.0, 0.0, 1.0])
    z = _unit(R[:, 2] - R[:, 2].dot(up) * up)
    x = up if R[:, 0].dot(up) >= 0 else -up
    if abs(R[:, 0].dot(up)) < 0.9:
        raise ValueError("The gripper is rolled too far for an automatic level.")
    return _pivot_target(e, np.column_stack([x, np.cross(z, x), z]), reach_mm, max_deg)


def yaw_target(e, deg, reach_mm, max_deg=10.0):
    return _pivot_target(e, _Rz(math.radians(deg)) @ ov_to_R(e), reach_mm, max_deg)


# ---------- taught constants ----------
PLACE_HEADING_DEG = -31.9
PLACE_XY = (-91.1, -379.5)
PLACE_ABOVE_JOINTS = [-117.43, 11.48, -39.05, 93.96, 87.93, 207.5]
TABLE_Z = -8.7
LIFT_MM = 100.0
SETDOWN_SLOW_MM = 20.0
RESET_RETREAT_MM = 100.0
RESET_CLEAR_Z = 380.0
AT_KEYFRAME_DEG = 1.0


def _pose(e):
    return {k: float(getattr(e, k)) for k in POSE_KEYS} if not isinstance(e, dict) else dict(e)


def _straight(arm_name, goal, step_mm, check=None):
    e = _pose(end_position(arm_name))
    for w in steps(e, goal, step_mm):
        if check:
            check()
        move_position(arm_name, w["x"], w["y"], w["z"], w["o_x"], w["o_y"], w["o_z"], w["theta"])


def _near(q, ref, tol=AT_KEYFRAME_DEG):
    return ref is not None and max(abs(a - b) for a, b in zip(q, ref)) <= tol


def reset_arm(arm_name, gripper_name, parking, check=None, log=print):
    """Open the gripper; unless already at parking: back straight out, rise to RESET_CLEAR_Z, then move to parking."""
    log("Opening the gripper")
    gripper_action(arm_name, "open")
    if _near(list(joints(arm_name)), parking):
        log("Already at parking")
        return
    e = end_position(arm_name)
    a = ov_to_R(e)[:, 2]
    a_h = np.array([a[0], a[1], 0.0])
    if np.linalg.norm(a_h) > 0.5:
        d = -RESET_RETREAT_MM * _unit(a_h)
        log(f"Backing straight out {RESET_RETREAT_MM:.0f} mm")
        _straight(arm_name, dict(e, x=e["x"] + d[0], y=e["y"] + d[1]), 20.0, check)
    e = end_position(arm_name)
    if e["z"] < RESET_CLEAR_Z:
        log(f"Rising to z {RESET_CLEAR_Z:.0f} mm")
        _straight(arm_name, dict(e, z=RESET_CLEAR_Z), 40.0, check)
    if check:
        check()
    log("Moving to parking")
    move_joint_positions(arm_name, list(parking))


def locate(arm_name, cam_name, cam_frame, grip_frame, cfg, check=None, log=print):
    """measure() from where the arm is; back up in -x if the bottle is cut off, at most max_backups times."""
    backups = int(cfg["max_backups"])
    for attempt in range(backups + 1):
        if check:
            check()
        log(f"Locating the leftmost bottle (YOLO + depth, {int(cfg['frames'])} frames)")
        try:
            m = measure(arm_name, cam_name, cam_frame, grip_frame, cfg, check)
            if not m["clipped"]:
                return m
            why = f"the bottle is cut off at the {', '.join(m['clipped_edges'])} edge of the image"
        except ValueError as ex:
            why = str(ex)
        if attempt == backups:
            raise ValueError(f"Could not see the whole bottle after backing up {backups * cfg['backup_mm']:.0f} mm: "
                             f"{why} (Reset arm 1 parks it)")
        log(f"Not fully in view ({why}); backing up {cfg['backup_mm']:.0f} mm in x to look again")
        e = end_position(arm_name)
        _straight(arm_name, dict(e, x=e["x"] - cfg["backup_mm"]), 25.0, check)


def pick_and_place(arm_name, gripper_name, cam_name, cam_frame, grip_frame, newmid, parking, cfg,
                   check=None, log=print, on_grab=None):
    """From parking or NewMid: locate -> pre-grasp -> straight in -> grab -> lift -> place. Ends holding the bottle.

    on_grab: optional no-arg callable fired right after a successful grab (e.g. to start a TTS cue).
    """
    q = list(joints(arm_name))
    if not (_near(q, newmid) or _near(q, parking)):
        raise ValueError("Arm 1 must start at parking or NewMid. Press Reset arm 1 first.")
    log("Opening the gripper")
    gripper_action(arm_name, "open")
    if not _near(q, newmid):
        if check:
            check()
        log("Moving to NewMid")
        move_joint_positions(arm_name, list(newmid))
    m = locate(arm_name, cam_name, cam_frame, grip_frame, cfg, check, log)
    p = plan(m, cfg)
    for level, text in p["checks"]:
        if level == "error":
            raise ValueError(text + " (arm 1 stays where it looked from; Reset arm 1 parks it)")
        if level == "warn":
            log("Note: " + text)
    log(f"Bottle at x {p['A'][0]:.0f}, y {p['A'][1]:.0f} (grasp z {p['zg']:.0f}); moving to the pre-grasp")
    _straight(arm_name, p["pre"], cfg["pre_step_mm"], check)
    log(f"Going straight in {cfg['pre_mm']:.0f} mm")
    _straight(arm_name, p["grab"], cfg["in_step_mm"], check)
    if check:
        check()
    log("Grabbing")
    held = gripper_action(arm_name, "grab")
    if held is False:
        gripper_action(arm_name, "open")
        _straight(arm_name, p["pre"], cfg["in_step_mm"], check)
        raise ValueError("The gripper closed on nothing; opened and backed out to the pre-grasp.")
    if on_grab is not None:
        on_grab()
    e = end_position(arm_name)
    log(f"Lifting {LIFT_MM:.0f} mm")
    _straight(arm_name, dict(e, z=e["z"] + LIFT_MM), 20.0, check)
    if check:
        check()
    log("Carrying to above the place spot")
    move_joint_positions(arm_name, PLACE_ABOVE_JOINTS)
    e = end_position(arm_name)
    above_bottom = float(p["zg"] - m["bottom"][2])
    place_z = min(max(TABLE_Z + above_bottom, TABLE_Z + 40.0), e["z"])
    log(f"Setting down: pads {above_bottom:.0f} mm above the bottle's bottom, so arm end to z {place_z:.0f}")
    _straight(arm_name, dict(e, z=min(place_z + SETDOWN_SLOW_MM, e["z"])), 20.0, check)
    _straight(arm_name, dict(e, z=place_z), 5.0, check)
    log("Placed. Arm 1 is holding the bottle at the place spot.")
    return m, p
