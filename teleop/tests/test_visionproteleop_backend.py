"""Deterministic no-hardware tests for the native Vision Pro backend."""

from __future__ import annotations

import copy
import math
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


TELEOP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TELEOP_ROOT))
sys.path.insert(0, str(TELEOP_ROOT / "televuer" / "src"))

from televuer import tv_wrapper as vuer_conventions  # noqa: E402
from utils import visionproteleop_backend as backend_module  # noqa: E402
from utils.visionproteleop_backend import (  # noqa: E402
    HOLD,
    REANCHOR_REQUIRED,
    TRACKING_OK,
    TRACKING_STALE,
    SyntheticVisionProStreamer,
    VisionProTeleopBackend,
)


def rotation_x(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    result = np.eye(4)
    result[:3, :3] = np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]
    )
    return result


def rotation_z(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    result = np.eye(4)
    result[:3, :3] = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    return result


def pose(translation, rotation: np.ndarray | None = None) -> np.ndarray:
    result = np.eye(4) if rotation is None else rotation.copy()
    result[:3, 3] = np.asarray(translation, dtype=np.float64)
    return result


def local_joints(count: int = 25) -> np.ndarray:
    joints = np.repeat(np.eye(4)[None, :, :], count, axis=0)
    for index in range(count):
        joints[index, :3, 3] = np.array(
            [0.004 * index, -0.002 * index, 0.001 * index]
        )
    return joints


def make_packet(
    *,
    joint_count: int = 25,
    use_arm_keys: bool = False,
    head_openxr: np.ndarray | None = None,
    left_wrist_openxr: np.ndarray | None = None,
    right_wrist_openxr: np.ndarray | None = None,
) -> tuple[dict, dict]:
    head_openxr = (
        pose([0.1, 1.6, -0.2], rotation_z(0.2))
        if head_openxr is None
        else head_openxr
    )
    left_wrist_openxr = (
        pose([-0.25, 1.25, -0.45], rotation_z(-0.15))
        if left_wrist_openxr is None
        else left_wrist_openxr
    )
    right_wrist_openxr = (
        pose([0.28, 1.22, -0.42], rotation_z(0.12))
        if right_wrist_openxr is None
        else right_wrist_openxr
    )
    left_joints = local_joints(joint_count)
    right_joints = local_joints(joint_count)
    right_joints[:, 1, 3] *= -1.0

    avp_basis = backend_module.T_AVP_Y_UP_TO_Z_UP
    packet = {
        "head": (avp_basis @ head_openxr @ rotation_x(-math.pi / 2.0))[None],
        "left_wrist": (avp_basis @ left_wrist_openxr)[None],
        "right_wrist": (avp_basis @ right_wrist_openxr)[None],
        "left_pinch_distance": 0.012,
        "right_pinch_distance": 0.031,
    }
    key_suffix = "arm" if use_arm_keys else "fingers"
    packet[f"left_{key_suffix}"] = left_joints
    packet[f"right_{key_suffix}"] = right_joints
    raw = {
        "head": head_openxr,
        "left_wrist": left_wrist_openxr,
        "right_wrist": right_wrist_openxr,
        "left_joints": left_joints,
        "right_joints": right_joints,
    }
    return packet, raw


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class TrackingWrapper:
    def __init__(self, raw) -> None:
        self._raw = raw

    @property
    def raw(self):
        return self._raw


class FakeStreamer:
    def __init__(self) -> None:
        self.latest = None
        self.record = False
        self.video_config = None
        self.video_callback = None
        self.webrtc_args = None
        self.updated_frames = []
        self.cleanup_calls = 0
        self.cleanup_error = None
        self.update_error = None

    def get_latest(self):
        if self.latest is None:
            return None
        # Stock avp_stream creates a new TrackingData wrapper on every poll, but
        # keeps the same .raw dictionary until a network update arrives.
        return TrackingWrapper(self.latest)

    def configure_video(self, **kwargs):
        self.video_config = kwargs

    def register_frame_callback(self, callback):
        self.video_callback = callback

    def start_webrtc(self, **kwargs):
        self.webrtc_args = kwargs

    def update_frame(self, frame):
        if self.update_error is not None:
            raise self.update_error
        self.updated_frames.append(frame)

    def cleanup(self):
        self.cleanup_calls += 1
        if self.cleanup_error is not None:
            raise self.cleanup_error


class VisionProTeleopBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.streamer = FakeStreamer()
        self.logs = []

    def make_backend(self, **kwargs) -> VisionProTeleopBackend:
        settings = {
            "ip": "192.0.2.10",
            "streamer": self.streamer,
            "clock": self.clock,
            "status_callback": self.logs.append,
            "start_video": False,
            "settling_time_s": 0.0,
        }
        settings.update(kwargs)
        return VisionProTeleopBackend(**settings)

    def prime_and_enable(self, backend, packet=None):
        if packet is None:
            packet, _ = make_packet()
        self.streamer.latest = packet
        waiting = backend.get_tele_data()
        self.assertEqual(waiting.tracking_state, REANCHOR_REQUIRED)
        self.assertFalse(waiting.tracking_active)
        self.assertTrue(backend.request_enable())
        active = backend.get_tele_data()
        self.assertEqual(active.tracking_state, TRACKING_OK)
        self.assertTrue(active.tracking_active)
        return active

    def test_streamer_is_lazy_optional_dependency(self):
        with mock.patch.object(
            backend_module.importlib,
            "import_module",
            side_effect=ModuleNotFoundError("avp_stream"),
        ):
            with self.assertRaisesRegex(RuntimeError, "optional 'avp_stream'"):
                VisionProTeleopBackend(
                    ip="192.0.2.10",
                    clock=self.clock,
                    status_callback=None,
                    start_video=False,
                )

    def test_factory_disables_recording_and_configures_mono_video(self):
        captured = {}

        def factory(**kwargs):
            captured.update(kwargs)
            return self.streamer

        backend = VisionProTeleopBackend(
            ip="192.0.2.10",
            img_shape=(360, 640),
            display_fps=29.7,
            streamer_factory=factory,
            clock=self.clock,
            status_callback=None,
        )
        self.assertFalse(captured["record"])
        self.assertEqual(captured["origin"], "avp")
        self.assertEqual(captured["ht_backend"], "grpc")
        self.assertEqual(
            self.streamer.video_config,
            {"device": None, "size": "640x360", "fps": 30, "stereo": False},
        )
        self.assertIsNotNone(self.streamer.video_callback)
        self.assertEqual(self.streamer.webrtc_args, {"port": 9999, "blocking": False})
        backend.close()

    def test_non_finite_safety_limits_are_rejected(self):
        cases = (
            {"display_fps": np.nan},
            {"tracking_timeout_s": np.nan},
            {"settling_time_s": np.inf},
            {"rigid_atol": np.nan},
            {"max_wrist_translation_jump_m": np.nan},
            {"max_wrist_rotation_jump_rad": np.inf},
            {"max_wrist_velocity_m_s": np.nan},
            {"max_wrist_angular_velocity_rad_s": np.inf},
            {"pinch_threshold_m": np.nan},
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    self.make_backend(**kwargs)

    def test_late_streamer_after_close_is_cleaned_without_starting_video(self):
        backend = self.make_backend()
        backend.close()
        late_streamer = FakeStreamer()
        backend._finish_streamer_startup(
            late_streamer,
            True,
            {"device": None, "size": "640x480", "fps": 30, "stereo": False},
            9999,
        )
        self.assertEqual(late_streamer.cleanup_calls, 1)
        self.assertIsNone(late_streamer.video_config)
        self.assertIsNone(late_streamer.video_callback)
        self.assertIsNone(late_streamer.webrtc_args)
        self.assertIsNone(backend.streamer)

    def test_converts_exactly_like_existing_vuer_wrapper(self):
        backend = self.make_backend(return_hand_rot_data=True)
        packet, raw = make_packet()
        data = self.prime_and_enable(backend, packet)

        transform = vuer_conventions.T_ROBOT_OPENXR
        inverse_transform = vuer_conventions.T_OPENXR_ROBOT
        expected_head = transform @ raw["head"] @ inverse_transform
        expected_left_world = transform @ raw["left_wrist"] @ inverse_transform
        expected_right_world = transform @ raw["right_wrist"] @ inverse_transform
        expected_left = (
            expected_left_world @ vuer_conventions.T_TO_UNITREE_HUMANOID_LEFT_ARM
        )
        expected_right = (
            expected_right_world @ vuer_conventions.T_TO_UNITREE_HUMANOID_RIGHT_ARM
        )
        expected_left[:3, 3] -= expected_head[:3, 3]
        expected_right[:3, 3] -= expected_head[:3, 3]
        expected_left[:3, 3] += np.array([0.15, 0.0, 0.45])
        expected_right[:3, 3] += np.array([0.15, 0.0, 0.45])

        np.testing.assert_allclose(data.head_pose, expected_head, atol=1e-12)
        np.testing.assert_allclose(data.left_wrist_pose, expected_left, atol=1e-12)
        np.testing.assert_allclose(data.right_wrist_pose, expected_right, atol=1e-12)

        for side in ("left", "right"):
            raw_wrist = raw[f"{side}_wrist"]
            joints = raw[f"{side}_joints"][:25]
            local_points = joints[:, :, 3].T
            world_points = raw_wrist @ local_points
            robot_world_points = transform @ world_points
            robot_world_wrist = transform @ raw_wrist @ inverse_transform
            robot_arm_points = np.linalg.inv(robot_world_wrist) @ robot_world_points
            expected_hand = (
                vuer_conventions.T_TO_UNITREE_HAND @ robot_arm_points
            )[:3].T
            np.testing.assert_allclose(
                getattr(data, f"{side}_hand_pos"), expected_hand, atol=1e-12
            )

        self.assertEqual(data.left_hand_pos.shape, (25, 3))
        self.assertEqual(data.right_hand_pos.shape, (25, 3))
        self.assertEqual(data.left_hand_rot.shape, (25, 3, 3))
        self.assertEqual(data.right_hand_rot.shape, (25, 3, 3))
        self.assertTrue(data.left_hand_pinch)
        self.assertAlmostEqual(data.left_hand_pinchValue, 1.2)
        self.assertFalse(data.right_hand_pinch)
        self.assertAlmostEqual(data.right_hand_pinchValue, 3.1)
        self.assertFalse(data.left_hand_squeeze)
        self.assertFalse(data.right_hand_squeeze)
        self.assertTrue(data.head_pose_is_valid)
        self.assertTrue(data.left_arm_is_valid)
        self.assertTrue(data.right_arm_is_valid)
        self.assertFalse(data.native_tracking_status_available)

    def test_head_identity_forward_and_xyz_axes_have_robot_signs(self):
        backend = self.make_backend()
        identity = np.eye(4)
        axis_cases = (
            ("right", np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])),
            ("up", np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            ("back", np.array([0.0, 0.0, 1.0]), np.array([-1.0, 0.0, 0.0])),
            ("forward", np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0])),
        )
        packet, _ = make_packet(
            head_openxr=identity,
            left_wrist_openxr=identity,
            right_wrist_openxr=identity,
        )
        neutral = backend._convert_packet(packet)
        np.testing.assert_allclose(neutral.head_pose, np.eye(4), atol=1e-12)

        for name, openxr_axis, robot_axis in axis_cases:
            with self.subTest(name=name):
                moved_head = pose(openxr_axis)
                moved_packet, _ = make_packet(
                    head_openxr=moved_head,
                    left_wrist_openxr=identity,
                    right_wrist_openxr=identity,
                )
                moved = backend._convert_packet(moved_packet)
                np.testing.assert_allclose(
                    moved.head_pose[:3, 3], robot_axis, atol=1e-12
                )

    def test_both_wrists_map_left_right_forward_back_up_down(self):
        backend = self.make_backend()
        identity = np.eye(4)
        baseline_packet, _ = make_packet(
            head_openxr=identity,
            left_wrist_openxr=identity,
            right_wrist_openxr=identity,
        )
        baseline = backend._convert_packet(baseline_packet)
        direction_cases = (
            ("right", [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
            ("left", [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ("forward", [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]),
            ("back", [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]),
            ("up", [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
            ("down", [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]),
        )
        for name, openxr_delta, expected_delta in direction_cases:
            with self.subTest(name=name):
                wrist = pose(openxr_delta)
                packet, _ = make_packet(
                    head_openxr=identity,
                    left_wrist_openxr=wrist,
                    right_wrist_openxr=wrist,
                )
                moved = backend._convert_packet(packet)
                for side in ("left", "right"):
                    actual_delta = (
                        getattr(moved, f"{side}_wrist_pose")[:3, 3]
                        - getattr(baseline, f"{side}_wrist_pose")[:3, 3]
                    )
                    np.testing.assert_allclose(
                        actual_delta, np.asarray(expected_delta), atol=1e-12
                    )

    def test_head_roll_pitch_yaw_rotation_directions(self):
        backend = self.make_backend()
        identity = np.eye(4)
        angle = 0.3
        # OpenXR +Y yaw -> robot +Z yaw; OpenXR +X -> robot -Y;
        # OpenXR +Z -> robot -X after the existing basis conversion.
        rotation_cases = (
            ("yaw", rotation_x(0.0) @ backend_module._rotation_y(angle), rotation_z(angle)),
            ("pitch", rotation_x(angle), backend_module._rotation_y(-angle)),
            ("roll", rotation_z(angle), rotation_x(-angle)),
        )
        for name, openxr_rotation, expected_robot_rotation in rotation_cases:
            with self.subTest(name=name):
                packet, _ = make_packet(
                    head_openxr=openxr_rotation,
                    left_wrist_openxr=identity,
                    right_wrist_openxr=identity,
                )
                data = backend._convert_packet(packet)
                np.testing.assert_allclose(
                    data.head_pose[:3, :3],
                    expected_robot_rotation[:3, :3],
                    atol=1e-12,
                )

    def test_finger_order_is_preserved_and_local_curl_axis_is_mapped(self):
        backend = self.make_backend()
        identity = np.eye(4)
        packet, _ = make_packet(
            head_openxr=identity,
            left_wrist_openxr=identity,
            right_wrist_openxr=identity,
        )
        ordered = np.repeat(np.eye(4)[None, :, :], 25, axis=0)
        for index in range(25):
            ordered[index, :3, 3] = np.array(
                [0.001 * (index + 1), 0.002 * (index + 1), 0.003 * (index + 1)]
            )
        packet["left_fingers"] = ordered
        packet["right_fingers"] = ordered.copy()
        baseline = backend._convert_packet(packet)
        expected = (
            backend_module.T_TO_UNITREE_HAND
            @ backend_module.T_ROBOT_OPENXR
            @ ordered[:, :, 3].T
        )[:3].T
        np.testing.assert_allclose(baseline.left_hand_pos, expected, atol=1e-12)
        np.testing.assert_allclose(baseline.right_hand_pos, expected, atol=1e-12)

        curled_packet = copy.deepcopy(packet)
        curled_packet["left_fingers"][9, 2, 3] += 0.02
        curled = backend._convert_packet(curled_packet)
        delta = curled.left_hand_pos[9] - baseline.left_hand_pos[9]
        # Local OpenXR +Z curl displacement becomes Unitree-hand +Y.
        np.testing.assert_allclose(delta, np.array([0.0, 0.02, 0.0]), atol=1e-12)

    def test_accepts_25_and_27_joint_inputs_but_always_returns_25(self):
        outputs = []
        for count, arm_keys in ((25, False), (27, False), (27, True)):
            with self.subTest(count=count, arm_keys=arm_keys):
                self.clock = FakeClock()
                self.streamer = FakeStreamer()
                backend = self.make_backend()
                packet, _ = make_packet(joint_count=count, use_arm_keys=arm_keys)
                data = self.prime_and_enable(backend, packet)
                self.assertEqual(data.left_hand_pos.shape, (25, 3))
                self.assertEqual(data.right_hand_pos.shape, (25, 3))
                outputs.append(data.left_hand_pos)
        np.testing.assert_allclose(outputs[0], outputs[1])
        np.testing.assert_allclose(outputs[0], outputs[2])

    def test_same_raw_object_does_not_refresh_monotonic_arrival_time(self):
        backend = self.make_backend(tracking_timeout_s=0.2)
        packet, _ = make_packet()
        self.prime_and_enable(backend, packet)
        first_arrival = backend.last_packet_monotonic

        self.clock.advance(0.1)
        still_active = backend.get_tele_data()
        self.assertEqual(backend.last_packet_monotonic, first_arrival)
        self.assertTrue(still_active.tracking_active)

        self.clock.advance(0.11)
        stale = backend.get_tele_data()
        self.assertEqual(stale.tracking_state, TRACKING_STALE)
        self.assertFalse(stale.tracking_active)
        self.assertFalse(stale.session_alive)

    def test_enable_latches_until_distinct_valid_packets_finish_settling(self):
        backend = self.make_backend(settling_time_s=0.2, tracking_timeout_s=0.25)
        packet, _ = make_packet()
        self.streamer.latest = packet
        self.assertEqual(backend.get_tele_data().tracking_state, REANCHOR_REQUIRED)
        self.assertFalse(backend.request_enable())
        self.assertTrue(backend.get_status()["enable_requested"])

        # Re-polling the same raw object does not count toward settling.
        self.clock.advance(0.1)
        self.assertEqual(backend.get_tele_data().tracking_state, REANCHOR_REQUIRED)
        self.assertFalse(backend.get_status()["settling_complete"])

        self.streamer.latest = copy.deepcopy(packet)
        backend.get_tele_data()
        self.clock.advance(0.11)
        self.streamer.latest = copy.deepcopy(packet)
        settled = backend.get_tele_data()
        self.assertEqual(settled.tracking_state, TRACKING_OK)
        self.assertTrue(settled.tracking_active)

    def test_reconnect_requires_a_new_explicit_enable(self):
        backend = self.make_backend(tracking_timeout_s=0.2)
        packet, _ = make_packet()
        active = self.prime_and_enable(backend, packet)
        held_left = active.left_wrist_pose.copy()

        self.clock.advance(0.21)
        self.assertEqual(backend.get_tele_data().tracking_state, TRACKING_STALE)
        self.assertFalse(backend.request_enable())

        self.clock.advance(0.01)
        self.streamer.latest = copy.deepcopy(packet)
        reconnected = backend.get_tele_data()
        self.assertEqual(reconnected.tracking_state, REANCHOR_REQUIRED)
        self.assertFalse(reconnected.tracking_active)
        np.testing.assert_allclose(reconnected.left_wrist_pose, held_left)
        self.assertTrue(backend.request_enable())
        self.assertTrue(backend.get_tele_data().tracking_active)

    def test_nan_or_non_rigid_transform_fails_closed_and_holds_last_pose(self):
        backend = self.make_backend()
        packet, _ = make_packet()
        active = self.prime_and_enable(backend, packet)
        held_head = active.head_pose.copy()

        invalid = copy.deepcopy(packet)
        invalid["left_fingers"][7, 0, 3] = np.nan
        self.clock.advance(0.02)
        self.streamer.latest = invalid
        failed = backend.get_tele_data()
        self.assertEqual(failed.tracking_state, HOLD)
        self.assertFalse(failed.tracking_active)
        self.assertTrue(failed.session_alive)
        np.testing.assert_allclose(failed.head_pose, held_head)
        self.assertFalse(backend.request_enable())

        non_rigid = copy.deepcopy(packet)
        non_rigid["head"][0, 0, 0] = 2.0
        self.clock.advance(0.02)
        self.streamer.latest = non_rigid
        failed = backend.get_tele_data()
        self.assertEqual(failed.tracking_state, HOLD)
        self.assertIn("not orthonormal", failed.tracking_reason)

    def test_translation_jump_and_rotation_jump_require_reanchor(self):
        backend = self.make_backend(
            max_wrist_translation_jump_m=0.05,
            max_wrist_rotation_jump_rad=0.5,
            max_wrist_velocity_m_s=100.0,
            max_wrist_angular_velocity_rad_s=100.0,
        )
        packet, raw = make_packet()
        self.prime_and_enable(backend, packet)

        moved_wrist = raw["left_wrist"].copy()
        moved_wrist[0, 3] += 0.2
        moved, _ = make_packet(left_wrist_openxr=moved_wrist)
        self.clock.advance(0.1)
        self.streamer.latest = moved
        self.assertEqual(backend.get_tele_data().tracking_state, HOLD)
        self.assertIn("translation jump", backend.state_reason)
        self.assertFalse(backend.request_enable())
        self.clock.advance(0.05)
        self.streamer.latest = copy.deepcopy(moved)
        self.assertEqual(backend.get_tele_data().tracking_state, REANCHOR_REQUIRED)
        self.assertTrue(backend.request_enable())

        rotated_wrist = moved_wrist @ rotation_z(0.8)
        rotated, _ = make_packet(left_wrist_openxr=rotated_wrist)
        self.clock.advance(0.1)
        self.streamer.latest = rotated
        self.assertEqual(backend.get_tele_data().tracking_state, HOLD)
        self.assertIn("rotation jump", backend.state_reason)

    def test_wrist_velocity_is_rejected_even_below_jump_threshold(self):
        backend = self.make_backend(
            max_wrist_translation_jump_m=1.0,
            max_wrist_velocity_m_s=1.0,
            max_wrist_rotation_jump_rad=math.pi,
            max_wrist_angular_velocity_rad_s=100.0,
        )
        packet, raw = make_packet()
        self.prime_and_enable(backend, packet)
        moved_wrist = raw["left_wrist"].copy()
        moved_wrist[0, 3] += 0.02
        moved, _ = make_packet(left_wrist_openxr=moved_wrist)
        self.clock.advance(0.01)
        self.streamer.latest = moved
        self.assertEqual(backend.get_tele_data().tracking_state, HOLD)
        self.assertIn("wrist velocity", backend.state_reason)

    def test_operator_hold_never_auto_resumes(self):
        backend = self.make_backend()
        packet, _ = make_packet()
        self.prime_and_enable(backend, packet)
        backend.set_hold(True, "U key hold")
        self.assertEqual(backend.get_tele_data().tracking_state, HOLD)
        self.assertFalse(backend.get_tele_data().tracking_active)
        # Fresh packets cannot move an asserted external hold to REANCHOR.
        self.clock.advance(0.01)
        self.streamer.latest = copy.deepcopy(packet)
        self.assertEqual(backend.get_tele_data().tracking_state, HOLD)
        self.assertFalse(backend.request_enable())
        backend.set_hold(False)
        self.assertFalse(backend.get_tele_data().tracking_active)
        self.assertEqual(backend.get_tele_data().tracking_state, REANCHOR_REQUIRED)
        self.assertTrue(backend.request_enable())
        self.assertTrue(backend.get_tele_data().tracking_active)

    def test_video_callback_and_direct_update_copy_mono_bgr_frame(self):
        backend = VisionProTeleopBackend(
            ip="192.0.2.10",
            img_shape=(4, 6),
            streamer=self.streamer,
            clock=self.clock,
            status_callback=None,
            start_video=True,
        )
        blank = self.streamer.video_callback(np.ones((4, 6, 3), dtype=np.uint8))
        self.assertEqual(blank.shape, (4, 6, 3))
        self.assertFalse(np.any(blank))

        source = np.zeros((2, 2, 3), dtype=np.uint8)
        source[0, 0] = np.array([3, 5, 7])
        backend.render_to_xr(source)
        source[:] = 255
        callback_frame = self.streamer.video_callback(np.zeros((4, 6, 3), dtype=np.uint8))
        self.assertEqual(callback_frame.shape, (4, 6, 3))
        self.assertEqual(callback_frame.dtype, np.uint8)
        np.testing.assert_array_equal(callback_frame[0, 0], np.array([3, 5, 7]))
        np.testing.assert_array_equal(
            self.streamer.updated_frames[-1], callback_frame
        )
        with self.assertRaises(ValueError):
            callback_frame[:] = 99
        self.assertEqual(self.streamer.video_callback(None)[0, 0, 0], 3)

        self.streamer.update_error = RuntimeError("video down")
        backend.render_to_xr(np.zeros((4, 6, 3), dtype=np.uint8))
        self.assertIsInstance(backend.last_video_error, RuntimeError)

    def test_status_reports_tracking_video_and_rejection_metrics(self):
        backend = self.make_backend(img_shape=(4, 6))
        packet, raw = make_packet()
        self.prime_and_enable(backend, packet)
        self.clock.advance(0.05)
        self.streamer.latest = copy.deepcopy(packet)
        backend.get_tele_data()
        backend.render_to_xr(np.zeros((4, 6, 3), dtype=np.uint8))
        self.clock.advance(0.05)
        backend.render_to_xr(np.zeros((4, 6, 3), dtype=np.uint8))
        backend._video_frame_callback(None)
        self.clock.advance(0.05)
        backend._video_frame_callback(None)

        jumped_wrist = raw["left_wrist"].copy()
        jumped_wrist[0, 3] += 1.0
        jumped, _ = make_packet(left_wrist_openxr=jumped_wrist)
        self.streamer.latest = jumped
        backend.get_tele_data()
        status = backend.get_status()
        self.assertEqual(status["packet_count"], 3)
        self.assertGreater(status["tracking_rate_hz"], 0.0)
        self.assertEqual(status["video_input_count"], 2)
        self.assertGreater(status["video_input_rate_hz"], 0.0)
        self.assertEqual(status["video_callback_count"], 2)
        self.assertGreater(status["video_callback_rate_hz"], 0.0)
        self.assertEqual(status["rejected_jump_count"], 1)
        self.assertIsNone(status["transport_latency_ms"])
        self.assertEqual(status["head_position_m"].shape, (3,))
        self.assertTrue(math.isfinite(status["head_yaw_rad"]))
        self.assertEqual(status["left_wrist_position_m"].shape, (3,))
        self.assertEqual(status["right_wrist_position_m"].shape, (3,))
        self.assertEqual(status["left_hand_joint_count"], 25)
        self.assertEqual(status["right_hand_joint_count"], 25)
        self.assertGreater(status["left_hand_nonzero_points"], 0)
        self.assertGreater(status["right_hand_nonzero_points"], 0)
        self.assertGreater(status["left_pinch_distance_cm"], 0.0)
        self.assertGreater(status["right_pinch_distance_cm"], 0.0)

    def test_stock_streamer_construction_runs_in_daemon_worker(self):
        started = threading.Event()
        release = threading.Event()
        created = []

        def blocking_factory(**_kwargs):
            started.set()
            release.wait(timeout=2.0)
            streamer = FakeStreamer()
            created.append(streamer)
            return streamer

        with mock.patch.object(
            backend_module, "_load_streamer_class", return_value=blocking_factory
        ):
            backend = VisionProTeleopBackend(
                ip="192.0.2.10",
                start_video=False,
                clock=self.clock,
                status_callback=None,
            )
        self.assertTrue(started.wait(timeout=0.5))
        self.assertIsNone(backend.streamer)
        self.assertTrue(backend._startup_thread.daemon)
        backend.close()
        release.set()
        backend._startup_thread.join(timeout=1.0)
        self.assertFalse(backend._startup_thread.is_alive())
        self.assertEqual(created[0].cleanup_calls, 1)

    def test_synthetic_streamer_matches_stock_shapes_and_changes_every_poll(self):
        synthetic = SyntheticVisionProStreamer(fps=60.0, joint_count=27)
        first = synthetic.get_latest()
        second = synthetic.get_latest()
        self.assertIsNot(first, second)
        self.assertEqual(first["head"].shape, (1, 4, 4))
        self.assertEqual(first["left_wrist"].shape, (1, 4, 4))
        self.assertEqual(first["right_wrist"].shape, (1, 4, 4))
        self.assertEqual(first["left_fingers"].shape, (25, 4, 4))
        self.assertEqual(first["right_fingers"].shape, (25, 4, 4))
        self.assertEqual(first["left_arm"].shape, (27, 4, 4))
        self.assertEqual(first["right_arm"].shape, (27, 4, 4))
        self.assertFalse(np.array_equal(first["head"], second["head"]))

        backend = VisionProTeleopBackend(
            ip="synthetic",
            streamer=synthetic,
            settling_time_s=0.0,
            clock=self.clock,
            status_callback=None,
            img_shape=(4, 6),
        )
        self.assertEqual(backend.get_tele_data().tracking_state, REANCHOR_REQUIRED)
        self.assertTrue(backend.request_enable())
        self.clock.advance(1.0 / 60.0)
        data = backend.get_tele_data()
        self.assertTrue(data.tracking_active)
        self.assertEqual(data.left_hand_pos.shape, (25, 3))
        backend.render_to_xr(np.zeros((4, 6, 3), dtype=np.uint8))
        self.assertEqual(synthetic.pull_video_frame().shape, (4, 6, 3))
        self.assertEqual(synthetic.updated_frame_count, 1)
        self.assertEqual(synthetic.video_callback_count, 1)

    def test_tactile_overlay_compatibility_hook_accepts_positional_data(self):
        backend = self.make_backend()
        self.assertIsNone(backend.set_tactile_overlay({"left": object()}))

    def test_cleanup_is_idempotent_and_best_effort(self):
        backend = self.make_backend()
        self.streamer.cleanup_error = RuntimeError("cleanup failed")
        backend.close()
        backend.close()
        self.assertEqual(self.streamer.cleanup_calls, 1)
        self.assertEqual(len(backend.last_cleanup_errors), 1)
        closed = backend.get_tele_data()
        self.assertEqual(closed.tracking_state, HOLD)
        self.assertFalse(closed.session_alive)
        self.assertFalse(closed.tracking_active)

    def test_console_state_names_are_explicit(self):
        backend = self.make_backend(tracking_timeout_s=0.1)
        packet, _ = make_packet()
        self.streamer.latest = packet
        backend.get_tele_data()
        backend.request_enable()
        backend.request_hold("test hold")
        backend.request_enable()
        self.clock.advance(0.11)
        backend.get_tele_data()
        output = "\n".join(self.logs)
        for state in (TRACKING_STALE, REANCHOR_REQUIRED, TRACKING_OK, HOLD):
            self.assertIn(state, output)


if __name__ == "__main__":
    unittest.main()
