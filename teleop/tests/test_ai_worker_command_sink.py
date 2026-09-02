import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from teleop.robot_control.robotis_ai_worker import AIWorkerArmIK
from teleop.utils.ai_worker_command_sink import (
    AIWorkerArmCommandSink,
    HX5D20CommandSink,
    preview_teleop_step,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SH5_URDF = (
    REPO_ROOT.parent
    / "ai_worker"
    / "ffw_description"
    / "urdf"
    / "ffw_sh5_rev1_follower"
    / "ffw_sh5_follower.urdf"
)


def _open_hand_landmarks():
    points = np.zeros((25, 3), dtype=np.float64)
    points[1:5] = (
        (0.020, 0.010, 0.0),
        (0.035, 0.025, 0.0),
        (0.050, 0.040, 0.0),
        (0.065, 0.055, 0.0),
    )
    chains = (
        (5, 6, 7, 8, 9),
        (10, 11, 12, 13, 14),
        (15, 16, 17, 18, 19),
        (20, 21, 22, 23, 24),
    )
    for x, chain in zip((0.040, 0.015, -0.015, -0.040), chains):
        for step, index in enumerate(chain, start=1):
            points[index] = (x, 0.020 * step, 0.0)
    return points


def _bent_hand_landmarks():
    points = _open_hand_landmarks()
    points[7] = (0.040, 0.055, 0.020)
    points[8] = (0.040, 0.055, 0.040)
    points[9] = (0.040, 0.040, 0.055)
    return points


def _shared_hand_buffers(points):
    left_input = mp.Array("d", points.reshape(-1))
    right_input = mp.Array("d", points.reshape(-1))
    state = mp.Array("d", 40)
    action = mp.Array("d", 40)
    return left_input, right_input, mp.Lock(), state, action


class AIWorkerCommandSinkTest(unittest.TestCase):
    def test_arm_sink_limits_and_captures_target_without_transport(self):
        reports = []
        sink = AIWorkerArmCommandSink(
            arm_velocity_limit=1000.0,
            report_interval=0.0,
            reporter=reports.append,
        )
        initial = sink.get_current_dual_arm_q()
        target = initial.copy()
        target[[0, 7]] += 0.2

        sink.ctrl_dual_arm(target)

        np.testing.assert_allclose(sink.get_last_commanded_dual_arm_q(), target)
        np.testing.assert_allclose(sink.get_current_dual_arm_q(), target)
        self.assertEqual(sink.command_count, 1)
        self.assertTrue(sink.get_last_write_ok())
        report = sink.print_report()
        self.assertIn("DDS", sink.command_topic_description)
        self.assertIn("count=1", report)
        self.assertEqual(reports[-1], report)
        sink.stop()

    def test_geometric_hx5_sink_updates_shared_state_and_action(self):
        open_points = _open_hand_landmarks()
        buffers = _shared_hand_buffers(open_points)
        sink = HX5D20CommandSink(
            *buffers,
            retarget_mode="geometric",
            smoothing_alpha=1.0,
            report_interval=0.0,
            start_thread=False,
        )
        first_left, first_right = sink.update_target()
        second_left, second_right = sink.update_target(
            _bent_hand_landmarks(),
            _bent_hand_landmarks(),
        )

        self.assertEqual(first_left.shape, (20,))
        self.assertEqual(first_right.shape, (20,))
        self.assertGreater(np.max(np.abs(second_left - first_left)), 0.1)
        self.assertGreater(np.max(np.abs(second_right - first_right)), 0.1)
        expected = np.concatenate((second_left, second_right))
        np.testing.assert_allclose(np.asarray(buffers[3][:]), expected)
        np.testing.assert_allclose(np.asarray(buffers[4][:]), expected)
        self.assertEqual(sink.command_count, 2)
        sink.close()

    @unittest.skipUnless(SH5_URDF.is_file(), "AI Worker SH5 URDF is not installed")
    def test_preview_step_routes_production_ik_and_hx5_outputs_to_memory(self):
        try:
            arm_ik = AIWorkerArmIK(urdf_path=SH5_URDF, max_iterations=8)
        except RuntimeError as exc:
            self.skipTest(str(exc))
        points = _open_hand_landmarks()
        hand_sink = HX5D20CommandSink(
            *_shared_hand_buffers(points),
            retarget_mode="geometric",
            smoothing_alpha=1.0,
            report_interval=0.0,
            start_thread=False,
        )
        arm_sink = AIWorkerArmCommandSink(
            home_q=arm_ik.ready_q,
            ready_q=arm_ik.ready_q,
            arm_velocity_limit=1000.0,
            report_interval=0.0,
        )
        left_wrist = np.eye(4, dtype=np.float64)
        right_wrist = np.eye(4, dtype=np.float64)
        left_wrist[:3, 3] = (0.2, 0.2, 0.5)
        right_wrist[:3, 3] = (0.2, -0.2, 0.5)
        first = preview_teleop_step(
            arm_ik,
            arm_sink,
            hand_sink,
            left_wrist,
            right_wrist,
            points,
            points,
        )
        shifted_left = left_wrist.copy()
        shifted_right = right_wrist.copy()
        shifted_left[0, 3] += 0.04
        shifted_right[0, 3] += 0.04
        second = preview_teleop_step(
            arm_ik,
            arm_sink,
            hand_sink,
            shifted_left,
            shifted_right,
            points,
            points,
        )

        self.assertEqual(second.arm.shape, (14,))
        self.assertEqual(second.left_hand.shape, (20,))
        self.assertEqual(second.right_hand.shape, (20,))
        self.assertGreater(np.max(np.abs(second.arm - first.arm)), 1e-3)
        self.assertEqual(arm_sink.command_count, 2)
        self.assertEqual(hand_sink.command_count, 2)
        hand_sink.close()
        arm_sink.close()

    def test_fresh_process_import_and_construction_never_import_robotis_dds(self):
        script = """
import builtins
import multiprocessing as mp
import numpy as np

real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'teleop.robot_control.robotis_dds':
        raise AssertionError('command sink attempted to import motor DDS')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

from teleop.utils.ai_worker_command_sink import AIWorkerArmCommandSink, HX5D20CommandSink

points = np.zeros((25, 3), dtype=np.float64)
points[1:, 0] = np.linspace(0.01, 0.24, 24)
points[1:, 1] = np.linspace(0.02, 0.48, 24)
arm = AIWorkerArmCommandSink(report_interval=0.0)
hand = HX5D20CommandSink(
    mp.Array('d', points.reshape(-1)),
    mp.Array('d', points.reshape(-1)),
    mp.Lock(),
    mp.Array('d', 40),
    mp.Array('d', 40),
    retarget_mode='geometric',
    report_interval=0.0,
    start_thread=False,
)
arm.close()
hand.close()
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
