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
    def _run_reachable_path(
        self,
        target_q,
        frames=90,
        ik_class=AIWorkerVirtualLeaderIK,
    ):
        ik = ik_class(urdf_path=SG2_URDF)
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

    def test_relative_local_maps_operator_local_axes_to_robot_local_axes(self):
        ik = AIWorkerArmIK(
            urdf_path=SG2_URDF,
            wrist_orientation_mode="relative-local",
        )
        angle = 0.30

        def rotation_x(value):
            cosine, sine = np.cos(value), np.sin(value)
            return np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, cosine, -sine],
                    [0.0, sine, cosine],
                ],
                dtype=np.float64,
            )

        def rotation_y(value):
            cosine, sine = np.cos(value), np.sin(value)
            return np.array(
                [
                    [cosine, 0.0, sine],
                    [0.0, 1.0, 0.0],
                    [-sine, 0.0, cosine],
                ],
                dtype=np.float64,
            )

        def rotation_z(value):
            cosine, sine = np.cos(value), np.sin(value)
            return np.array(
                [
                    [cosine, -sine, 0.0],
                    [sine, cosine, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

        robot_rotation = rotation_z(-0.4) @ rotation_y(0.2)
        robot_anchor = ik.pin.SE3(robot_rotation, np.array([0.4, 0.2, 1.0]))
        for side, operator_anchor in (
            ("left", rotation_x(np.pi / 2.0)),
            ("right", rotation_x(-np.pi / 2.0)),
        ):
            for motion, local_delta in (
                ("pitch", rotation_y(angle)),
                ("terminal", rotation_z(angle)),
            ):
                with self.subTest(side=side, motion=motion):
                    input_anchor = np.eye(4, dtype=np.float64)
                    input_anchor[:3, :3] = operator_anchor
                    input_pose = input_anchor.copy()
                    input_pose[:3, :3] = operator_anchor @ local_delta
                    target = ik._target(
                        input_pose,
                        input_anchor,
                        robot_anchor,
                        robot_rotation,
                        np.eye(3),
                        np.eye(3),
                    )
                    np.testing.assert_allclose(
                        target.rotation,
                        robot_rotation @ local_delta,
                        atol=1e-12,
                    )
                    np.testing.assert_allclose(
                        target.translation,
                        robot_anchor.translation,
                        atol=1e-12,
                    )

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

    def test_sg2_ik_keeps_wrist_pitch_and_terminal_axis_distinct(self):
        ready_q = np.array(
            [0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0] * 2,
            dtype=np.float64,
        )
        angle = 0.30
        for ik_class in (AIWorkerArmIK, AIWorkerVirtualLeaderIK):
            for side, offset in (("left", 0), ("right", 7)):
                other_arm = slice(7, 14) if offset == 0 else slice(0, 7)
                pitch_index = offset + 5
                terminal_index = offset + 6
                for sign in (-1.0, 1.0):
                    with self.subTest(
                        ik=ik_class.__name__, side=side, sign=sign, motion="pitch"
                    ):
                        pitch_target = ready_q.copy()
                        pitch_target[pitch_index] = sign * angle
                        q, position_error, rotation_error, max_joint_step = (
                            self._run_reachable_path(
                                pitch_target,
                                ik_class=ik_class,
                            )
                        )
                        pitch_delta = q[pitch_index] - ready_q[pitch_index]
                        terminal_delta = q[terminal_index] - ready_q[terminal_index]
                        self.assertGreater(sign * pitch_delta, 0.29)
                        self.assertLess(abs(terminal_delta), 0.005)
                        self.assertGreater(
                            abs(pitch_delta),
                            10.0 * max(abs(terminal_delta), 1e-9),
                        )
                        self.assertLess(
                            np.max(np.abs(q[other_arm] - ready_q[other_arm])),
                            0.002,
                        )
                        self.assertLess(position_error, 5e-4)
                        self.assertLess(rotation_error, 2e-3)
                        self.assertLessEqual(max_joint_step, 0.120001)

                    with self.subTest(
                        ik=ik_class.__name__,
                        side=side,
                        sign=sign,
                        motion="terminal_axis",
                    ):
                        terminal_target = ready_q.copy()
                        terminal_target[terminal_index] = sign * angle
                        q, position_error, rotation_error, max_joint_step = (
                            self._run_reachable_path(
                                terminal_target,
                                ik_class=ik_class,
                            )
                        )
                        pitch_delta = q[pitch_index] - ready_q[pitch_index]
                        terminal_delta = q[terminal_index] - ready_q[terminal_index]
                        # The seven-axis arm can redistribute terminal rotation
                        # through upstream joints near asymmetric joint limits,
                        # so verify the terminal joint's sign/dominance rather
                        # than freezing one non-unique full joint solution.
                        self.assertGreater(sign * terminal_delta, 0.22)
                        self.assertLess(abs(pitch_delta), 0.01)
                        self.assertGreater(
                            abs(terminal_delta),
                            10.0 * max(abs(pitch_delta), 1e-9),
                        )
                        self.assertLess(
                            np.max(np.abs(q[other_arm] - ready_q[other_arm])),
                            0.002,
                        )
                        self.assertLess(position_error, 5e-4)
                        self.assertLess(rotation_error, 2e-3)
                        self.assertLessEqual(max_joint_step, 0.120001)


if __name__ == "__main__":
    unittest.main()
