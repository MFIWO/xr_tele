import threading
import time
import unittest
from types import SimpleNamespace
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from teleop.robot_control.robotis_image_client import (
    AI_WORKER_CAMERA_TOPICS,
    RobotisROSImageClient,
)
from teleop.robot_control.robotis_leader_observer import (
    AI_WORKER_HEAD_JOINTS,
    AI_WORKER_LEADER_TOPICS,
    AI_WORKER_LEFT_SG2_JOINTS,
    AI_WORKER_LIFT_JOINTS,
    AI_WORKER_RIGHT_SG2_JOINTS,
    AIWorkerLeaderObserver,
    AIWorkerROSObserver,
    _initialize_observer_state,
)
from teleop.robot_control.robotis_webxr_bridge_client import (
    AIWorkerWebXRBridgeClient,
)
from teleop.utils.loop_streamer import AIWorkerLeaderLoopStreamer


def _trajectory(names, positions):
    return SimpleNamespace(
        joint_names=list(names),
        points=[SimpleNamespace(positions=list(positions))],
    )


def _empty_observer(stale_after=0.5):
    observer = AIWorkerLeaderObserver.__new__(AIWorkerLeaderObserver)
    observer.stale_after = stale_after
    observer._lock = threading.Lock()
    observer._stop = threading.Event()
    observer._core_ready = threading.Event()
    observer._last_error = None
    observer._updated_at = {
        "leader_left": None,
        "leader_right": None,
        "leader_head": None,
        "leader_lift": None,
        "leader_mobile": None,
        "follower": None,
        "follower_mobile": None,
    }
    observer._values = AIWorkerLeaderLoopStreamer.empty_snapshot()
    observer._values.pop("valid")
    observer._values = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in observer._values.items()
    }
    observer._readers = {}
    observer._threads = []
    return observer


class _FakeROSNode:
    def __init__(self):
        self.subscriptions = []
        self.destroyed = []

    def create_subscription(self, message_type, topic, callback, qos_profile):
        subscription = SimpleNamespace(
            message_type=message_type,
            topic=topic,
            callback=callback,
            qos_profile=qos_profile,
        )
        self.subscriptions.append(subscription)
        return subscription

    def destroy_subscription(self, subscription):
        self.destroyed.append(subscription)


class _FakeExecutor:
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def remove_node(self, node):
        self.nodes.remove(node)

    def spin_once(self, timeout_sec):
        time.sleep(min(float(timeout_sec), 0.001))

    def wake(self):
        pass


class AIWorkerLeaderLoopTest(unittest.TestCase):
    def test_observer_extracts_complete_eight_joint_actions_in_config_order(self):
        observer = _empty_observer()
        left_values = np.arange(8, dtype=np.float64) + 10.0
        right_values = np.arange(8, dtype=np.float64) + 20.0

        observer._process(
            "leader_left",
            _trajectory(reversed(AI_WORKER_LEFT_SG2_JOINTS), reversed(left_values)),
        )
        observer._process(
            "leader_right",
            _trajectory(reversed(AI_WORKER_RIGHT_SG2_JOINTS), reversed(right_values)),
        )

        snapshot = observer.snapshot()
        np.testing.assert_allclose(snapshot["left_action_position"], left_values)
        np.testing.assert_allclose(snapshot["right_action_position"], right_values)
        self.assertTrue(snapshot["valid"]["leader_left"])
        self.assertTrue(snapshot["valid"]["leader_right"])

    def test_follower_state_supplies_arm_gripper_head_and_lift_observations(self):
        observer = _empty_observer()
        names = list(
            AI_WORKER_LEFT_SG2_JOINTS
            + AI_WORKER_RIGHT_SG2_JOINTS
            + AI_WORKER_HEAD_JOINTS
            + AI_WORKER_LIFT_JOINTS
        )
        positions = np.arange(len(names), dtype=np.float64)
        velocities = positions + 100.0
        observer._process(
            "follower",
            SimpleNamespace(
                name=names,
                position=positions.tolist(),
                velocity=velocities.tolist(),
            ),
        )

        snapshot = observer.snapshot()
        np.testing.assert_allclose(snapshot["left_state_position"], positions[:8])
        np.testing.assert_allclose(snapshot["right_state_position"], positions[8:16])
        np.testing.assert_allclose(snapshot["head_state_position"], positions[16:18])
        np.testing.assert_allclose(snapshot["lift_state_position"], positions[18:19])
        np.testing.assert_allclose(snapshot["left_state_velocity"], velocities[:8])
        self.assertTrue(snapshot["valid"]["follower"])

    def test_core_ready_requires_follower_and_both_leader_arms(self):
        observer = _empty_observer()
        observer._mark_updated("follower", {"left_state_position": np.zeros(8)})
        observer._mark_updated("leader_left", {"left_action_position": np.zeros(8)})
        self.assertFalse(observer.wait_until_ready(0.0))

        observer._mark_updated("leader_right", {"right_action_position": np.zeros(8)})
        self.assertTrue(observer.wait_until_ready(0.0))

    def test_snapshot_marks_old_sources_stale(self):
        observer = _empty_observer(stale_after=0.1)
        observer._updated_at["leader_left"] = time.monotonic() - 1.0
        self.assertFalse(observer.snapshot()["valid"]["leader_left"])

    def test_loop_schema_separates_follower_observation_and_leader_action(self):
        streamer = AIWorkerLeaderLoopStreamer("localhost:50051", 30)
        snapshot = streamer.empty_snapshot()
        snapshot["left_state_position"] = list(range(8))
        snapshot["right_state_position"] = list(range(8, 16))
        snapshot["left_action_position"] = list(range(20, 28))
        snapshot["right_action_position"] = list(range(30, 38))
        snapshot["valid"] = {"follower": True, "leader_left": True}

        robot = streamer._build_snapshot_step(snapshot)["ai_worker"]
        self.assertEqual(
            robot["observation"]["left_arm"]["joint_position"], list(range(7))
        )
        self.assertEqual(robot["observation"]["left_ee"]["joint_position"], [7.0])
        self.assertEqual(
            robot["action"]["right_arm"]["joint_position"],
            [float(value) for value in range(30, 37)],
        )
        self.assertEqual(robot["action"]["right_ee"]["joint_position"], [37.0])
        self.assertTrue(robot["observation"]["source_valid"]["follower"])
        self.assertFalse(robot["observation"]["source_valid"]["leader_right"])

        layout = streamer._channel_layout()
        self.assertIn("ai_worker.action.left_ee.joint_position", layout)
        self.assertIn("ai_worker.observation.head.joint_position", layout)
        self.assertIn("ai_worker.action.mobile.velocity", layout)

    def test_ros_observer_subscribes_to_all_sources_without_publishers(self):
        node = _FakeROSNode()
        executor = _FakeExecutor()
        message_types = {
            "trajectory": "JointTrajectory",
            "twist": "Twist",
            "joint_state": "JointState",
            "odometry": "Odometry",
        }
        observer = AIWorkerROSObserver(
            node=node,
            executor=executor,
            message_types=message_types,
            qos_profile="sensor-data-qos",
        )
        try:
            self.assertEqual(
                {subscription.topic for subscription in node.subscriptions},
                set(AI_WORKER_LEADER_TOPICS.values()),
            )
            left = next(
                subscription
                for subscription in node.subscriptions
                if subscription.topic == AI_WORKER_LEADER_TOPICS["leader_left"]
            )
            values = np.arange(8, dtype=np.float64)
            left.callback(_trajectory(AI_WORKER_LEFT_SG2_JOINTS, values))
            np.testing.assert_allclose(
                observer.snapshot()["left_action_position"],
                values,
            )
        finally:
            observer.close()

        self.assertEqual(len(node.destroyed), len(AI_WORKER_LEADER_TOPICS))
        self.assertEqual(executor.nodes, [])

    def test_ros_image_client_decodes_compressed_camera_frames(self):
        node = _FakeROSNode()
        client = RobotisROSImageClient(
            node,
            compressed_image_type="CompressedImage",
            qos_profile="sensor-data-qos",
        )
        try:
            self.assertEqual(
                {subscription.topic for subscription in node.subscriptions},
                set(AI_WORKER_CAMERA_TOPICS.values()),
            )
            image = np.full((8, 12, 3), 77, dtype=np.uint8)
            encoded_ok, encoded = cv2.imencode(".jpg", image)
            self.assertTrue(encoded_ok)
            head = next(
                subscription
                for subscription in node.subscriptions
                if subscription.topic == AI_WORKER_CAMERA_TOPICS["head"]
            )
            head.callback(
                SimpleNamespace(
                    data=encoded.tobytes(),
                    header=SimpleNamespace(
                        stamp=SimpleNamespace(sec=12, nanosec=34)
                    ),
                )
            )
            frame = client.get_head_frame()
            self.assertEqual(frame.bgr.shape, image.shape)
            self.assertEqual(frame.jpg, encoded.tobytes())
            self.assertEqual(
                client.get_frame_timestamps()["head"]["source_time_ns"],
                12_000_000_034,
            )
            self.assertEqual(
                client.get_cam_config()["head_camera"]["source"],
                "robotis_ros2",
            )
            self.assertIsNone(client.last_error)
        finally:
            client.close()

        self.assertEqual(len(node.destroyed), len(AI_WORKER_CAMERA_TOPICS))

    def test_webxr_bridge_events_use_the_existing_loop_snapshot_schema(self):
        client = AIWorkerWebXRBridgeClient.__new__(AIWorkerWebXRBridgeClient)
        _initialize_observer_state(client, 0.5)
        client._bridge_ready = threading.Event()
        client._bridge_info = None
        client._camera_error = None

        client._dispatch_event(
            {
                "kind": "bridge_ready",
                "rmw": "rmw_zenoh_cpp",
                "domain_id": 30,
            }
        )
        values = np.arange(8, dtype=np.float64)
        client._dispatch_event(
            {
                "kind": "trajectory",
                "source": "leader_right",
                "joint_names": list(AI_WORKER_RIGHT_SG2_JOINTS),
                "positions": values.tolist(),
            }
        )

        self.assertTrue(client._bridge_ready.is_set())
        self.assertEqual(client.bridge_info["rmw"], "rmw_zenoh_cpp")
        np.testing.assert_allclose(
            client.snapshot()["right_action_position"],
            values,
        )

    def test_help_exposes_meta_quest_loop_without_importing_rclpy(self):
        repo_root = Path(__file__).resolve().parents[2]
        entrypoint = repo_root / "teleop" / "teleop_hand_and_arm.py"
        result = subprocess.run(
            [sys.executable, str(entrypoint), "--help"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--meta-quest-loop", result.stdout)
        self.assertIn("--robotis-webxr-loop", result.stdout)
        self.assertIn("--meta-quest-ros-python", result.stdout)

    def test_meta_quest_and_physical_leader_modes_are_mutually_exclusive(self):
        repo_root = Path(__file__).resolve().parents[2]
        entrypoint = repo_root / "teleop" / "teleop_hand_and_arm.py"
        result = subprocess.run(
            [
                sys.executable,
                str(entrypoint),
                "--meta-quest-loop",
                "--leader-master-loop",
            ],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutually exclusive", result.stderr)


if __name__ == "__main__":
    unittest.main()
