"""Connection lifecycle regressions; these tests never contact hardware."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock

from backend import robot


def test_shared_loop_survives_async_startup_and_concurrent_requests():
    async def identify():
        await asyncio.sleep(0.005)
        return asyncio.get_running_loop()

    async def startup():
        return robot.run(identify())

    loop = asyncio.run(startup())
    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(lambda _: robot.run(identify()), range(32)))
    assert all(result is loop for result in results)
    assert loop.is_running()


def test_runtime_error_is_not_retried():
    calls = []

    async def fail():
        calls.append(1)
        raise RuntimeError('SDK failure')

    async def request():
        try:
            robot.run(fail())
        except RuntimeError as exc:
            assert str(exc) == 'SDK failure'
        else:
            raise AssertionError('Expected SDK failure')

    asyncio.run(request())
    assert calls == [1]


def test_authenticated_connection_and_resource_listing(monkeypatch):
    key_id = '00000000-0000-4000-8000-000000000001'
    monkeypatch.setattr(robot, 'API_KEY', 'test-key')
    monkeypatch.setattr(robot, 'API_KEY_ID', key_id)
    fake = SimpleNamespace(resource_names=[
        SimpleNamespace(type='component', subtype=subtype, name=name)
        for subtype, name in [('arm', 'arm'), ('arm', 'arm2'), ('camera', 'cam'), ('gripper', 'gripper')]
    ], close=AsyncMock())
    dial = AsyncMock(return_value=fake)
    monkeypatch.setattr(robot.RobotClient, 'at_address', dial)
    monkeypatch.setattr(robot, '_robot', robot.Robot())
    assert robot.list_arm_names() == ['arm', 'arm2']
    assert robot.list_all_resource_names() == {
        'arm': ['arm', 'arm2'], 'camera': ['cam'], 'gripper': ['gripper']}
    dial.assert_awaited_once()
    opts = dial.call_args.args[1]
    assert opts.dial_options.auth_entity == key_id
    assert opts.dial_options.credentials.payload == 'test-key'
    robot.disconnect()
    fake.close.assert_awaited_once()


def test_capture_unpacks_images_and_uses_client_transforms(monkeypatch):
    import cv2
    import numpy as np
    from viam.media.video import NamedImage, CameraMimeType

    _, encoded = cv2.imencode('.png', np.zeros((2, 3, 3), dtype=np.uint8))
    color = NamedImage('color', encoded.tobytes(), CameraMimeType.PNG)
    depth = SimpleNamespace(mime_type='image/vnd.viam.dep', name='depth',
                            bytes_to_depth_array=lambda: [[100, 200, 300], [400, 500, 600]])
    camera = SimpleNamespace(
        get_images=AsyncMock(return_value=([color, depth], object())),
        get_properties=AsyncMock(return_value=SimpleNamespace(intrinsic_parameters=SimpleNamespace(
            focal_x_px=100, focal_y_px=100, center_x_px=1, center_y_px=1))))

    async def transform(q, destination):
        assert destination == 'world'
        return q

    client = SimpleNamespace(transform_pose=AsyncMock(side_effect=transform))
    wrapper = robot.Robot()
    wrapper._robot = client
    monkeypatch.setattr(wrapper, 'camera', lambda name: camera)
    monkeypatch.setattr(robot, '_robot', wrapper)
    frame = robot.capture_frame('cam', 'cam', 'arm')
    assert frame['bgr'].shape == (2, 3, 3)
    assert frame['depth_matches_color']
    assert frame['depth'][1, 2] == 600
    np.testing.assert_array_equal(frame['R'], np.eye(3))
    assert client.transform_pose.await_count == 4
