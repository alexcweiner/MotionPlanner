"""Offline regression tests for the FastAPI port. No startup or real hardware."""
import importlib
import sys
import threading
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'backend'))
    with patch.dict(sys.modules):
        for name in ('main', 'robot', 'yolo_grab', 'cap_unscrew', 'bottle_task', 'cap_line_search'):
            sys.modules.pop(name, None)
        m = importlib.import_module('main')
        monkeypatch.setattr(m, 'POSES_FILE', tmp_path / 'poses.json')
        yield m


def test_save_preserves_other_arm_and_roundtrips(api):
    for arm in ('arm', 'arm2'):
        api.api_save_pose(api.SavePose(arm=arm, name='home', kind='joints', values=[1]*6))
    assert {(p['arm'],p['name']) for p in api.load_poses(api.POSES_FILE)} == {('arm','home'),('arm2','home')}


def test_full_run_preserves_return_waypoint(api, monkeypatch):
    names = ['safe_bottle_start_pose','open_bottle_cap_up_pose','open_bottle_cap_pose','trash_drop_pose']
    frames = [{'arm':'arm','name':'parking','values':[0]*6}] + [dict(arm='arm2',name=n,values=[0]*6) for n in names]
    monkeypatch.setattr(api,'load_poses',lambda path: frames)
    task = N(running=False, start=Mock(), status=lambda: {'state':'done','step':'complete'})
    monkeypatch.setattr(api,'_BOTTLE_TASK',task)
    api.api_full_run()
    pre, params, drop, retreat, via, alongside = task.start.call_args.args
    assert alongside['name'] == names[0]
    assert alongside not in pre
    assert via == [alongside]
    assert drop['name'] == names[-1]


def test_unscrew_resolves_selected_pose(api, monkeypatch):
    monkeypatch.setattr(api,'load_poses',lambda path: [dict(arm='arm2',name='cap',values=[1]*6)])
    task=N(running=False,start=Mock(),status=lambda:{'state':'done','step':'complete'})
    monkeypatch.setattr(api,'_UNSCREWER',task)
    api.api_unscrew(api.UnscrewRequest(start_pose='cap'))
    assert task.start.call_args.args[0].start_joints == (1,)*6


def test_bad_yolo_plan_blocks_motion(api, monkeypatch):
    api.YOLO_STATE.update(plan={'checks':[('error','unstable')]},meas={},cfg={})
    move=Mock(); grip=Mock()
    monkeypatch.setattr(api,'_straight',move); monkeypatch.setattr(api,'gripper_action',grip)
    for kind in ('pre','grab'):
        with pytest.raises(api.HTTPException, match='unstable'):
            api.api_yolo_move(api.YoloMove(kind=kind))
    move.assert_not_called(); grip.assert_not_called()


def test_stop_invalidates_remaining_commands(api, monkeypatch):
    r=sys.modules['robot']
    with r.motion_session():
        r.cancel_motion()
        with pytest.raises(InterruptedError):
            r.move_joint_positions('arm',[0]*6)
        # Restore only this test session so its exit check can finish.
        r._motion_context.generation = r._stop_generation


def test_stop_all_attempts_every_resource_on_failure(api, monkeypatch):
    arm=Mock(side_effect=RuntimeError('offline')); grip=Mock()
    monkeypatch.setattr(api,'arm_stop',arm); monkeypatch.setattr(api,'gripper_stop',grip)
    with pytest.raises(api.HTTPException): api.api_stop_all()
    assert arm.call_count == grip.call_count == 2


def test_concurrent_motion_rejected(api, monkeypatch):
    entered=threading.Event(); release=threading.Event()
    def hold(*args): entered.set(); release.wait(2)
    monkeypatch.setattr(api,'move_joint_positions',hold)
    t=threading.Thread(target=lambda:api.api_move_joints('arm',api.JointMove(values=[0]*6)))
    t.start(); assert entered.wait(2)
    try:
        with pytest.raises(api.HTTPException) as error:
            api.api_move_joints('arm2',api.JointMove(values=[0]*6))
        assert error.value.status_code == 409
    finally: release.set(); t.join()


def test_level_endpoint_accepts_empty_body(api, monkeypatch):
    pose=dict(x=0,y=0,z=0,o_x=0,o_y=0,o_z=1,theta=0)
    monkeypatch.setattr(api,'end_position',lambda name:pose)
    monkeypatch.setattr(api,'yg_level_target',lambda *args:(pose,0))
    monkeypatch.setattr(api,'move_position',Mock())
    assert TestClient(api.app).post('/api/yolo/level').status_code == 200


def test_cap_detection_and_alignment_use_measured_depth(api, monkeypatch):
    cap=dict(bgr=np.zeros((100,100,3),np.uint8),depth=np.full((100,100),400,np.uint16),fx=200,fy=200,cx=50,cy=50)
    monkeypatch.setattr(sys.modules['robot'],'capture_frame',lambda *args:cap)
    monkeypatch.setattr(api._cap_vision,'detect_circles',lambda *args:([(60,50,10,1)],[]))
    detected=api.api_cap_detect()['circles'][0]
    assert (detected['x'],detected['y'],detected['z']) == (20,0,400)
    tf=Mock(side_effect=lambda src,p,dst:dict(p,theta=0,o_x=0,o_y=0))
    monkeypatch.setattr(sys.modules['robot'],'transform_pose',tf)
    monkeypatch.setattr(api,'end_position',lambda name:dict(x=0,y=0,z=500,o_x=0,o_y=0,o_z=-1,theta=0))
    api.api_align_plan(api.AlignPlan(circle_x=20,circle_y=0,circle_z=400,circle_r=10))
    assert tf.call_args_list[0].args[1]['z'] == 400


def test_unscrew_stop_and_extra_frames(api, monkeypatch):
    u=sys.modules['cap_unscrew']; stopped=[]
    monkeypatch.setattr(u,'arm_stop',lambda name:stopped.append(name))
    monkeypatch.setattr(u,'gripper_stop',lambda name:stopped.append(name))
    worker=u.Unscrewer(); worker.stop()
    assert stopped == ['arm2','arm2']
    transform=Mock(return_value=dict(x=1,y=2,z=3))
    monkeypatch.setattr(u,'transform_pose',transform)
    extra=[object()]; worker._point('_virtual',extra=extra)
    assert transform.call_args.args[3] is extra


def test_pick_place_http_response_and_saved_targets(api, monkeypatch):
    frames=[dict(arm='arm',name=name,values=[n]*6) for n,name in enumerate(('NewMid','parking'),1)]
    monkeypatch.setattr(api,'load_poses',lambda path:frames)
    routine=Mock(return_value=({'frames':3,'width_mm':80.0},{}))
    monkeypatch.setattr(api,'yg_pick_and_place',routine)
    response=TestClient(api.app).post('/api/pick_and_place')
    assert response.status_code == 200
    assert response.json()['frames'] == 3
    assert routine.call_args.kwargs['newmid'] == [1]*6
    assert routine.call_args.kwargs['parking'] == [2]*6


def test_sequence_stops_before_next_keyframe(api, monkeypatch):
    moves=[]
    def move(name, values):
        moves.append(name)
        api.cancel_motion()
    monkeypatch.setattr(api,'move_joint_positions',move)
    seq=api.RunSequence(keyframes=[api.Keyframe(arm='arm',name=str(i),values=[0]*6) for i in range(2)])
    with pytest.raises(api.HTTPException): api.api_run_poses(seq)
    assert moves == ['arm']


def test_yolo_preview_has_working_adapter(api, monkeypatch):
    import cv2
    _,png=cv2.imencode('.png',np.zeros((50,50,3),np.uint8))
    monkeypatch.setattr(api,'camera_frame',lambda name:png.tobytes())
    monkeypatch.setattr(api,'yolo_detect',lambda *args:[dict(class_name='bottle',confidence=.9,x_min=5,y_min=5,x_max=30,y_max=40)])
    response=TestClient(api.app).get('/api/yolo/preview')
    assert response.status_code == 200
    assert response.json()['count'] == 1


def test_depth_preview_endpoint(api, monkeypatch):
    monkeypatch.setattr(api,'capture_frame',lambda *args:{'depth':np.full((4,4),400,np.uint16)})
    response=TestClient(api.app).get('/api/camera/cam/frame')
    assert response.status_code == 200
    assert response.json()['jpeg_base64']


@pytest.mark.parametrize('kind', ['pre','grab'])
def test_yolo_rejects_moved_arm_or_skipped_pregrasp(api,monkeypatch,kind):
    expected=dict(x=0,y=0,z=100,o_x=1,o_y=0,o_z=0,theta=0)
    api.YOLO_STATE.update(plan={'checks':[], 'pre':expected},meas={'end':expected},cfg={})
    monkeypatch.setattr(api,'end_position',lambda name:dict(expected,x=20))
    move=Mock(); grip=Mock()
    monkeypatch.setattr(api,'_straight',move);monkeypatch.setattr(api,'gripper_action',grip)
    with pytest.raises(api.HTTPException): api.api_yolo_move(api.YoloMove(kind=kind))
    move.assert_not_called();grip.assert_not_called()
