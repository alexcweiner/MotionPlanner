"""Exercise the backend port with synthetic camera data; no hardware calls."""
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
from backend import robot

spec = importlib.util.spec_from_file_location('backend_yolo_under_test', Path(__file__).resolve().parents[1] / 'backend/yolo_grab.py')
yg = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'robot': robot}):
    spec.loader.exec_module(yg)


def test_bottle_depth_and_profile():
    cap = {'depth': np.full((100, 100), 500, dtype=np.uint16),
           'bgr': np.zeros((100, 100, 3), dtype=np.uint8),
           'fx': 500, 'fy': 500, 'cx': 50, 'cy': 50,
           'origin': np.zeros(3), 'R': np.eye(3)}
    detection = dict(x_min=30, x_max=70, y_min=10, y_max=90, confidence=.9)
    result = yg._bottle(cap, detection, yg.DEFAULTS, 40, 100, 100)
    assert result['z'] == 500
    assert result['valid'] == 1
    assert result['profile']
    # Radius is measured along the pixel ray, not along the optical z axis.
    np.testing.assert_allclose(np.linalg.norm(result['axis'] - result['surface']), 40)


def test_gripper_geometry_uses_gripper_frame(monkeypatch):
    frames = []
    def point(frame, x=0, y=0, z=0):
        frames.append(frame)
        return np.array([10+x, 20+y, 30+z], dtype=float)
    monkeypatch.setattr(yg, 'world_point', point)
    axes = yg.frame_axes('arm', 'cam', 'gripper')
    assert frames == ['gripper'] * 4
    np.testing.assert_array_equal(axes['origin'], [10, 20, 30])
    np.testing.assert_array_equal(axes['x'], [1, 0, 0])


# ---------- known bottle size vs depth read through the glass ----------
FX = 640.0
# Camera looking level along world +x (camera x -> world -y, camera y -> world -z), as at NewMid.
LEVEL_R = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _scene(depths, surface_mm=450.0):
    """measure() inputs for one upright 304.8 x 80 mm bottle whose front is surface_mm from the camera.
    Frame i's depth image reads depths[i] everywhere (450 = the glass; ~1150 = the wall seen through it)."""
    axis = surface_mm + 40.0
    half_h, half_w = FX * 304.8 / axis / 2, FX * 80.0 / axis / 2
    det = dict(class_name='bottle', confidence=.9, x_min=640 - half_w, x_max=640 + half_w,
               y_min=360 - half_h, y_max=360 + half_h)
    frames = iter([{'bgr': np.zeros((720, 1280, 3), np.uint8), 'depth': np.full((720, 1280), d, np.uint16),
                    'depth_matches_color': True, 'fx': FX, 'fy': FX, 'cx': 640.0, 'cy': 360.0,
                    'origin': np.zeros(3), 'R': LEVEL_R} for d in depths])
    return {'capture_frame': lambda *a: next(frames), 'yolo_detect': lambda *a: [dict(det)],
            'joints': lambda *a: [0.0] * 6,
            'end_position': lambda *a: {k: 0.0 for k in yg.POSE_KEYS} | {'o_x': 1.0},
            'world_point': lambda frame, x=0, y=0, z=0: np.array([x, y, z], float)}


def _measure(monkeypatch, depths):
    for name, fn in _scene(depths).items():
        monkeypatch.setattr(yg, name, fn)
    return yg.measure('arm', 'cam', 'cam', 'gripper', dict(yg.DEFAULTS))


def test_a_frame_whose_depth_reads_the_background_is_dropped(monkeypatch):
    m = _measure(monkeypatch, [450, 450, 1150])
    assert (m['frames'], m['of']) == (2, 3)
    assert m['spread_mm'] < 1                      # was ~700 mm with the background frame in the median
    assert m['z_frames'] == [450.0, 450.0]
    assert len(m['dropped']) == 1 and 'depth 1150 mm' in m['dropped'][0]
    np.testing.assert_allclose(m['axis'][0], 490.0, atol=1)


def test_good_frames_are_all_kept(monkeypatch):
    m = _measure(monkeypatch, [450, 452, 448])
    assert (m['frames'], m['dropped']) == (3, [])


def test_when_most_frames_read_the_background_nothing_is_measured(monkeypatch):
    import pytest
    with pytest.raises(ValueError, match='Dropped: depth 1150 mm'):
        _measure(monkeypatch, [450, 1150, 1150])


def test_spread_error_lists_each_frames_depth():
    m = {'spread_mm': 25.0, 'z_frames': [450.0, 470.0, 452.0]}
    text = ' '.join(t for level, t in _plan_checks(m) if level == 'error')
    assert 'depth per frame: 450, 470, 452 mm' in text


def _plan_checks(overrides):
    up = {k: 0.0 for k in yg.POSE_KEYS} | {'x': 3.3, 'y': -359.1, 'z': 235.1, 'o_x': 0.9985, 'o_y': -0.0461,
                                            'o_z': -0.028, 'theta': -5.2}
    m = {'frames': 3, 'of': 3, 'spread_mm': 0.0, 'axis': np.array([560., -292., 120.]),
         'top': np.array([560., -292., 280.]), 'bottom': np.array([560., -292., -20.]), 'width_mm': 90.0,
         'clipped': False, 'clipped_edges': [], 'neighbor': None, 'profile': {}, 'end': up,
         'gripper': {'origin': np.array([170., -359., 235.])}} | overrides
    return yg.plan(m, yg.DEFAULTS)['checks']


def test_orientation_matrix_is_a_rotation_at_any_theta():
    for theta in (-90.0, -5.2, 40.0, 180.0):
        for o in ((0.857, -0.515, 0.0), (0.0, 0.6, 0.8), (0.9985, -0.0461, -0.028)):
            R = yg.ov_to_R({'o_x': o[0], 'o_y': o[1], 'o_z': o[2], 'theta': theta})
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)
            np.testing.assert_allclose(R[:, 2], np.array(o) / np.linalg.norm(o), atol=1e-9)
