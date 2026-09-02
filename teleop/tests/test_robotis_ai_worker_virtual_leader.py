import unittest
from pathlib import Path

import numpy as np

from teleop.robot_control.robotis_ai_worker import AIWorkerArmIK, AIWorkerVirtualLeaderIK


REPO_ROOT = Path(__file__).resolve().parents[2]
_SG2_RELATIVE = (
    Path("ffw_description")
    / "urdf"
    / "ffw_sg2_rev1_follower"
    / "ffw_sg2_follower.urdf"
)
_SG2_CANDIDATES = (
    REPO_ROOT.parent / "ai_worker" / _SG2_RELATIVE,
    REPO_ROOT.parent / "external_repos" / "ai_worker" / _SG2_RELATIVE,
)
SG2_URDF = next((path for path in _SG2_CANDIDATES if path.is_file()), _SG2_CANDIDATES[0])


@unittest.skipUnless(SG2_URDF.is_file(), f"Official SG2 URDF not found: {SG2_URDF}")
class AIWorkerVirtualLeaderIKTest(unittest.TestCase):
    def _run_reachable_path(self, target_q, frames=90):
        ik = AIWorkerVirtualLeaderIK(urdf_path=SG2_URDF)
        start_q = ik.ready_q.copy()
        base_frames = ik._forward(start_q)
        identity = np.eye(4, dtype=np.float64)
        q, _ = ik.solve_ik(identity, identity, start_q, np.zeros(14))
        max_joint_step = 0.0

        for index in range(1, frames + 1):
            phase = index / frames
            phase = phase * phase * (3.0 - 2.0 * phase)
            oracle_q = (1.0 - phase) * start_q + phase * target_q
            desired_frames = ik._forward(oracle_q)
            inputs = []
            for desired, base in zip(desired_frames, base_frames):
                pose = np.eye(4, dtype=np.float64)
                pose[:3, :3] = desired.rotation @ base.rotation.T
                pose[:3, 3] = desired.translation - base.translation
                inputs.append(pose)
            previous_q = q.copy()
            q, _ = ik.solve_ik(inputs[0], inputs[1], q, np.zeros(14))
            max_joint_step = max(max_joint_step, float(np.max(np.abs(q - previous_q))))

        actual_frames = ik._forward(q)
        target_frames = ik._forward(target_q)
        position_error = max(
            np.linalg.norm(actual.translation - target.translation)
            for actual, target in zip(actual_frames, target_frames)
        )
        rotation_error = max(
            np.linalg.norm(ik.pin.log3(actual.rotation.T @ target.rotation))
            for actual, target in zip(actual_frames, target_frames)
        )
        return q, position_error, rotation_error, max_joint_step

    def test_legacy_solver_regression_is_unchanged(self):
        ik = AIWorkerArmIK(urdf_path=SG2_URDF)
        identity = np.eye(4, dtype=np.float64)
        ik.solve_ik(identity, identity, ik.ready_q, np.zeros(14))
        left = identity.copy()
        right = identity.copy()
        left[:3, 3] = [0.04, 0.02, 0.03]
        right[:3, 3] = [0.04, -0.02, 0.03]
        q, _ = ik.solve_ik(left, right, ik.ready_q, np.zeros(14))
        expected = np.array(
            [
                -0.1262719564, 0.0268393266, 0.0273603344, -1.5302894237,
                0.0232221581, 0.0872600865, -0.0285272755,
                -0.1262719564, -0.0268393266, -0.0273603344, -1.5302894237,
                -0.0232221581, 0.0872600865, 0.0285272755,
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(q, expected, atol=1e-8, rtol=0.0)

    def test_virtual_leader_reaches_straight_attention_pose(self):
        target_q = np.zeros(14, dtype=np.float64)
        q, position_error, rotation_error, max_joint_step = self._run_reachable_path(target_q)
        self.assertLess(abs(q[3]), 0.03)
        self.assertLess(abs(q[10]), 0.03)
        self.assertLess(position_error, 1e-3)
        self.assertLess(rotation_error, 1e-3)
        self.assertLessEqual(max_joint_step, 0.120001)

    def test_virtual_leader_reaches_straight_overhead_pose(self):
        target_q = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] * 2)
        q, position_error, rotation_error, max_joint_step = self._run_reachable_path(target_q)
        self.assertLess(abs(q[3]), 0.03)
        self.assertLess(abs(q[10]), 0.03)
        self.assertLess(position_error, 1e-3)
        self.assertLess(rotation_error, 1e-3)
        self.assertLessEqual(max_joint_step, 0.120001)


if __name__ == "__main__":
    unittest.main()
