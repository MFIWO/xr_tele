import time
import unittest

import numpy as np

from teleop.robot_control.robotis_ai_worker import (
    AI_WORKER_ARM_JOINTS,
    AI_WORKER_SG2_READY_Q,
    AIWorkerArmController,
)


class _CaptureTransport:
    def __init__(self):
        self.calls = []
        self.live_calls = []

    def publish(self, key, joint_names, positions, duration):
        self.live_calls.append(
            {
                "key": key,
                "joint_names": tuple(joint_names),
                "positions": np.asarray(positions, dtype=np.float64),
                "duration": float(duration),
            }
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
        self.calls.append(
            {
                "key": key,
                "joint_names": tuple(joint_names),
                "positions": np.asarray(positions, dtype=np.float64),
                "times": np.asarray(times_from_start, dtype=np.float64),
                "velocities": np.asarray(velocities, dtype=np.float64),
                "accelerations": np.asarray(accelerations, dtype=np.float64),
            }
        )


def _controller(start_q=None, velocity_limit=3.0):
    controller = AIWorkerArmController.__new__(AIWorkerArmController)
    controller.transport = _CaptureTransport()
    controller.command_duration = 0.08
    controller.arm_velocity_limit = float(velocity_limit)
    controller._q = (
        np.zeros(14, dtype=np.float64)
        if start_q is None
        else np.asarray(start_q, dtype=np.float64).reshape(14).copy()
    )
    controller._last_command = controller._q.copy()
    controller._last_command_time = time.monotonic()
    controller._last_write_ok = False
    return controller


class AIWorkerSmoothTrajectoryTest(unittest.TestCase):
    def test_quintic_trajectory_has_exact_stationary_endpoints(self):
        controller = _controller()

        duration = controller.ctrl_dual_arm_smooth_to(
            AI_WORKER_SG2_READY_Q,
            duration=7.0,
            num_points=100,
        )

        self.assertAlmostEqual(duration, 7.0)
        self.assertEqual([call["key"] for call in controller.transport.calls], ["left", "right"])
        for side, call in enumerate(controller.transport.calls):
            expected_names = AI_WORKER_ARM_JOINTS[side * 7:(side + 1) * 7]
            expected_target = AI_WORKER_SG2_READY_Q[side * 7:(side + 1) * 7]
            self.assertEqual(call["joint_names"], expected_names)
            self.assertEqual(call["positions"].shape, (100, 7))
            self.assertTrue(np.all(np.diff(call["times"]) > 0.0))
            self.assertAlmostEqual(call["times"][0], 0.0)
            self.assertAlmostEqual(call["times"][-1], 7.0)
            np.testing.assert_allclose(call["positions"][0], 0.0, atol=1e-12)
            np.testing.assert_allclose(call["positions"][-1], expected_target, atol=1e-12)
            np.testing.assert_allclose(call["velocities"][[0, -1]], 0.0, atol=1e-12)
            np.testing.assert_allclose(call["accelerations"][[0, -1]], 0.0, atol=1e-12)

        np.testing.assert_allclose(controller._last_command, AI_WORKER_SG2_READY_Q)
        self.assertTrue(controller._last_write_ok)

    def test_duration_expands_to_respect_velocity_limit(self):
        controller = _controller(velocity_limit=0.2)

        duration = controller.ctrl_dual_arm_smooth_to(
            AI_WORKER_SG2_READY_Q,
            duration=1.0,
            num_points=100,
        )

        self.assertGreater(duration, 1.0)
        peak_speed = max(
            float(np.max(np.abs(call["velocities"])))
            for call in controller.transport.calls
        )
        self.assertLessEqual(peak_speed, 0.2 + 1e-9)

    def test_command_limiter_can_reanchor_to_measured_endpoint(self):
        controller = _controller()
        controller._last_command = AI_WORKER_SG2_READY_Q.copy()
        measured = AI_WORKER_SG2_READY_Q * 0.5
        controller._q = measured.copy()
        previous_time = controller._last_command_time

        anchored = controller.sync_arm_command_to_measured()

        np.testing.assert_allclose(anchored, measured)
        np.testing.assert_allclose(controller._last_command, measured)
        self.assertGreaterEqual(controller._last_command_time, previous_time)

    def test_last_command_getter_returns_a_copy(self):
        controller = _controller()
        expected = np.linspace(-0.2, 0.2, 14)
        controller._last_command = expected.copy()

        returned = controller.get_last_commanded_dual_arm_q()
        returned[0] = 999.0

        np.testing.assert_allclose(controller.get_last_commanded_dual_arm_q(), expected)

    def test_live_control_stays_on_the_original_single_point_path(self):
        controller = _controller()
        controller._last_command_time -= 1.0
        target = AI_WORKER_SG2_READY_Q * 0.25

        controller.ctrl_dual_arm(target)

        self.assertEqual(controller.transport.calls, [])
        self.assertEqual(
            [call["key"] for call in controller.transport.live_calls],
            ["left", "right"],
        )
        for call in controller.transport.live_calls:
            self.assertAlmostEqual(call["duration"], controller.command_duration)
        np.testing.assert_allclose(
            controller.get_last_commanded_dual_arm_q(),
            np.concatenate(
                [call["positions"] for call in controller.transport.live_calls]
            ),
        )


if __name__ == "__main__":
    unittest.main()
