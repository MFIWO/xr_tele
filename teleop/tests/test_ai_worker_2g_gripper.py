import threading
import time
import unittest

import numpy as np

from teleop.robot_control.robotis_ai_worker import (
    AI_WORKER_2G_GRIPPER_JOINTS,
    AI_WORKER_ARM_JOINTS,
    AIWorkerArmController,
    ai_worker_2g_gripper_position,
)
from teleop.utils.ai_worker_command_sink import AIWorkerArmCommandSink


class _CaptureTransport:
    def __init__(self):
        self.live_calls = []
        self.trajectory_calls = []

    def publish(self, key, joint_names, positions, duration):
        self.live_calls.append(
            (key, tuple(joint_names), np.asarray(positions, dtype=np.float64), duration)
        )

    def publish_trajectory(
        self,
        key,
        joint_names,
        positions,
        times_from_start,
        velocities=None,
        accelerations=None,
    ):
        self.trajectory_calls.append(
            {
                "key": key,
                "joint_names": tuple(joint_names),
                "positions": np.asarray(positions, dtype=np.float64),
                "velocities": np.asarray(velocities, dtype=np.float64),
                "accelerations": np.asarray(accelerations, dtype=np.float64),
                "times": np.asarray(times_from_start, dtype=np.float64),
            }
        )


def _controller():
    controller = AIWorkerArmController.__new__(AIWorkerArmController)
    controller.transport = _CaptureTransport()
    controller.command_duration = 0.08
    controller.include_2g_gripper = True
    controller.arm_velocity_limit = 1000.0
    controller.home_q = np.zeros(14, dtype=np.float64)
    controller.ready_q = np.zeros(14, dtype=np.float64)
    controller._q = np.zeros(14, dtype=np.float64)
    controller._dq = np.zeros(14, dtype=np.float64)
    controller._last_command = np.zeros(14, dtype=np.float64)
    controller._gripper_q = np.zeros(2, dtype=np.float64)
    controller._last_gripper_command = np.zeros(2, dtype=np.float64)
    controller._last_command_time = time.monotonic() - 1.0
    controller._last_write_ok = False
    controller._joint_state_received = threading.Event()
    return controller


class AIWorker2GGripperTest(unittest.TestCase):
    def test_pinch_mapping_matches_sg2_closed_positive_convention(self):
        self.assertAlmostEqual(
            ai_worker_2g_gripper_position(
                5.0, closed_input=5.0, open_input=7.0, max_position=1.1
            ),
            1.1,
        )
        self.assertAlmostEqual(
            ai_worker_2g_gripper_position(
                6.0, closed_input=5.0, open_input=7.0, max_position=1.1
            ),
            0.55,
        )
        self.assertAlmostEqual(
            ai_worker_2g_gripper_position(
                7.0, closed_input=5.0, open_input=7.0, max_position=1.1
            ),
            0.0,
        )

    def test_live_messages_match_physical_ai_tools_eight_joint_schema(self):
        controller = _controller()
        controller.set_dual_gripper_command([0.25, 0.75])

        controller.ctrl_dual_arm(np.zeros(14, dtype=np.float64))

        self.assertEqual(len(controller.transport.live_calls), 2)
        for side_index, (key, names, positions, duration) in enumerate(
            controller.transport.live_calls
        ):
            self.assertEqual(key, ("left", "right")[side_index])
            self.assertEqual(
                names[:7],
                AI_WORKER_ARM_JOINTS[side_index * 7:(side_index + 1) * 7],
            )
            self.assertEqual(names[7], AI_WORKER_2G_GRIPPER_JOINTS[side_index])
            self.assertEqual(positions.shape, (8,))
            self.assertAlmostEqual(positions[7], (0.25, 0.75)[side_index])
            self.assertAlmostEqual(duration, 0.08)

    def test_smooth_arm_trajectory_holds_gripper_in_eighth_column(self):
        controller = _controller()
        controller.set_dual_gripper_command([0.2, 0.8])

        controller.ctrl_dual_arm_smooth_to(
            np.zeros(14, dtype=np.float64), duration=1.0, num_points=4
        )

        self.assertEqual(len(controller.transport.trajectory_calls), 2)
        for side_index, call in enumerate(controller.transport.trajectory_calls):
            self.assertEqual(call["positions"].shape, (4, 8))
            np.testing.assert_allclose(
                call["positions"][:, 7], (0.2, 0.8)[side_index]
            )
            np.testing.assert_allclose(call["velocities"][:, 7], 0.0)
            np.testing.assert_allclose(call["accelerations"][:, 7], 0.0)

    def test_startup_wait_requires_both_gripper_states_when_enabled(self):
        controller = _controller()

        class Message:
            pass

        arms_only = Message()
        arms_only.name = list(AI_WORKER_ARM_JOINTS)
        arms_only.position = [0.0] * 14
        arms_only.velocity = [0.0] * 14
        controller._joint_state_cb(arms_only)
        self.assertFalse(controller._joint_state_received.is_set())

        complete = Message()
        complete.name = list(AI_WORKER_ARM_JOINTS + AI_WORKER_2G_GRIPPER_JOINTS)
        complete.position = [0.0] * 14 + [0.3, 0.7]
        complete.velocity = [0.0] * 16
        controller._joint_state_cb(complete)
        self.assertTrue(controller._joint_state_received.is_set())
        np.testing.assert_allclose(controller.get_current_dual_gripper_q(), [0.3, 0.7])
        np.testing.assert_allclose(
            controller.get_last_commanded_dual_gripper_q(), [0.3, 0.7]
        )

    def test_command_sink_exposes_same_gripper_surface_without_dds(self):
        sink = AIWorkerArmCommandSink(include_2g_gripper=True, report_interval=0.0)
        sink.set_dual_gripper_command([0.4, 2.0])

        np.testing.assert_allclose(sink.get_current_dual_gripper_q(), [0.4, 1.1])
        np.testing.assert_allclose(
            sink.get_last_commanded_dual_gripper_q(), [0.4, 1.1]
        )
        sink.close()


if __name__ == "__main__":
    unittest.main()
