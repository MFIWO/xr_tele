import unittest
from pathlib import Path

import numpy as np

from teleop.robot_control.robotis_ai_worker import AIWorkerVirtualLeaderIK


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
class AIWorkerVirtualLeaderStabilityTest(unittest.TestCase):
    SIM_RESET_Q = np.array(
        [0.75, 0.0, 0.0, -2.30, 0.0, 0.0, 0.0] * 2,
        dtype=np.float64,
    )

    @staticmethod
    def _translated_wrist(x):
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = x
        return pose

    @staticmethod
    def _rotation_x(angle):
        cosine, sine = np.cos(angle), np.sin(angle)
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cosine, -sine],
                [0.0, sine, cosine],
            ],
            dtype=np.float64,
        )

    def _anchored_solver(self):
        ik = AIWorkerVirtualLeaderIK(
            urdf_path=SG2_URDF,
            wrist_orientation_mode="relative-local",
        )
        identity = np.eye(4, dtype=np.float64)
        q, _ = ik.solve_ik(identity, identity, ik.ready_q, np.zeros(14))
        return ik, q

    def test_extension_round_trip_never_crosses_the_straight_elbow_branch(self):
        ik, q = self._anchored_solver()
        largest_elbow = np.full(2, -np.inf, dtype=np.float64)

        path = np.concatenate(
            (
                np.linspace(0.0, 0.20, 41)[1:],
                np.linspace(0.20, 0.0, 41)[1:],
            )
        )
        for x in path:
            wrist = self._translated_wrist(x)
            q, _ = ik.solve_ik(wrist, wrist, q, np.zeros(14))
            largest_elbow = np.maximum(largest_elbow, q[[3, 10]])

        # q4 == 0 is the straight-arm configuration. Positive q4 is the
        # backwards-elbow branch that made the simulated arms wind around.
        self.assertLessEqual(float(np.max(largest_elbow)), 1e-9)

    def test_out_and_back_returns_to_the_anchor_joint_branch(self):
        ik, q = self._anchored_solver()
        anchor_q = q.copy()
        path = np.concatenate(
            (
                np.linspace(0.0, 0.20, 41)[1:],
                np.linspace(0.20, 0.0, 41)[1:],
                np.zeros(20, dtype=np.float64),
            )
        )

        for x in path:
            wrist = self._translated_wrist(x)
            q, _ = ik.solve_ik(wrist, wrist, q, np.zeros(14))

        self.assertLess(float(np.max(np.abs(q - anchor_q))), 0.15)
        self.assertLessEqual(float(np.max(q[[3, 10]])), 1e-9)

        actual_frames = ik._forward(q)
        anchor_frames = ik._forward(anchor_q)
        for actual, anchor in zip(actual_frames, anchor_frames):
            self.assertLess(
                float(np.linalg.norm(actual.translation - anchor.translation)),
                5e-3,
            )
            self.assertLess(
                float(np.linalg.norm(ik.pin.log3(actual.rotation.T @ anchor.rotation))),
                5e-3,
            )

    def test_measured_joint_state_reseeds_each_solve(self):
        ik, command_q = self._anchored_solver()
        anchor_q = command_q.copy()

        for x in np.linspace(0.0, 0.20, 41)[1:]:
            wrist = self._translated_wrist(x)
            command_q, _ = ik.solve_ik(
                wrist,
                wrist,
                command_q,
                np.zeros(14),
            )

        self.assertGreater(float(np.max(np.abs(command_q - anchor_q))), 0.20)

        # Emulate a controller that stayed at the measured anchor while the
        # previous virtual command ran ahead. With a measured-state seed, the
        # exact anchor EE target must recover in one solve instead of continuing
        # from the stale virtual branch.
        identity = np.eye(4, dtype=np.float64)
        recovered_q, _ = ik.solve_ik(
            identity,
            identity,
            anchor_q,
            np.zeros(14),
        )
        self.assertLess(float(np.max(np.abs(recovered_q - anchor_q))), 0.03)

    def test_sim_reset_pose_does_not_move_without_an_ee_delta(self):
        ik = AIWorkerVirtualLeaderIK(
            urdf_path=SG2_URDF,
            wrist_orientation_mode="relative-local",
        )
        identity = np.eye(4, dtype=np.float64)
        q = self.SIM_RESET_Q.copy()
        for _ in range(30):
            q, _ = ik.solve_ik(identity, identity, q, np.zeros(14))

        np.testing.assert_allclose(q, self.SIM_RESET_Q, atol=1e-8, rtol=0.0)

    def test_held_wrist_roll_converges_without_whole_arm_windup(self):
        ik = AIWorkerVirtualLeaderIK(
            urdf_path=SG2_URDF,
            wrist_orientation_mode="relative-local",
        )
        identity = np.eye(4, dtype=np.float64)
        q, _ = ik.solve_ik(
            identity,
            identity,
            self.SIM_RESET_Q,
            np.zeros(14),
        )
        anchor_frames = ik._forward(q)
        right = identity.copy()
        right[:3, :3] = self._rotation_x(0.25)

        for _ in range(40):
            q, _ = ik.solve_ik(identity, right, q, np.zeros(14))
        settled_q = q.copy()
        for _ in range(80):
            q, _ = ik.solve_ik(identity, right, q, np.zeros(14))

        self.assertLess(float(np.max(np.abs(q - settled_q))), 0.03)
        self.assertLessEqual(float(np.max(q[[3, 10]])), 1e-9)
        actual_right = ik._forward(q)[1]
        target_rotation = anchor_frames[1].rotation @ self._rotation_x(0.25)
        self.assertLess(
            float(np.linalg.norm(actual_right.translation - anchor_frames[1].translation)),
            1e-3,
        )
        self.assertLess(
            float(np.linalg.norm(ik.pin.log3(actual_right.rotation.T @ target_rotation))),
            1e-2,
        )

    def test_sim_reset_path_reaches_straight_elbows_without_branch_flip(self):
        ik = AIWorkerVirtualLeaderIK(
            urdf_path=SG2_URDF,
            wrist_orientation_mode="relative-local",
        )
        identity = np.eye(4, dtype=np.float64)
        q, _ = ik.solve_ik(
            identity,
            identity,
            self.SIM_RESET_Q,
            np.zeros(14),
        )
        anchor_frames = ik._forward(self.SIM_RESET_Q)
        straight_q = self.SIM_RESET_Q.copy()
        straight_q[[3, 10]] = 0.0
        largest_elbow = np.full(2, -np.inf, dtype=np.float64)

        for index in range(1, 91):
            phase = index / 90.0
            phase = phase * phase * (3.0 - 2.0 * phase)
            oracle_q = (1.0 - phase) * self.SIM_RESET_Q + phase * straight_q
            desired_frames = ik._forward(oracle_q)
            inputs = []
            for desired, anchor in zip(desired_frames, anchor_frames):
                pose = np.eye(4, dtype=np.float64)
                pose[:3, :3] = anchor.rotation.T @ desired.rotation
                pose[:3, 3] = desired.translation - anchor.translation
                inputs.append(pose)
            q, _ = ik.solve_ik(inputs[0], inputs[1], q, np.zeros(14))
            largest_elbow = np.maximum(largest_elbow, q[[3, 10]])

        self.assertLess(float(np.max(np.abs(q[[3, 10]]))), 0.04)
        self.assertLessEqual(float(np.max(largest_elbow)), 1e-9)
        actual_frames = ik._forward(q)
        target_frames = ik._forward(straight_q)
        for actual, target in zip(actual_frames, target_frames):
            self.assertLess(
                float(np.linalg.norm(actual.translation - target.translation)),
                2e-3,
            )
            self.assertLess(
                float(np.linalg.norm(ik.pin.log3(actual.rotation.T @ target.rotation))),
                1e-2,
            )

    def test_lateral_extension_prefers_straight_elbows_over_axial_corkscrew(self):
        ik, q = self._anchored_solver()
        anchor_frames = ik._forward(q)
        left = np.eye(4, dtype=np.float64)
        right = np.eye(4, dtype=np.float64)
        left[1, 3] = 0.30
        right[1, 3] = -0.30

        for _ in range(120):
            q, _ = ik.solve_ik(left, right, q, np.zeros(14))

        self.assertLess(float(np.max(np.abs(q[[3, 10]]))), 0.04)
        self.assertLessEqual(float(np.max(q[[3, 10]])), 1e-9)
        self.assertLess(float(np.max(np.abs(q[[2, 4, 9, 11]]))), 0.50)
        actual_frames = ik._forward(q)
        for side, (actual, anchor) in enumerate(zip(actual_frames, anchor_frames)):
            expected_translation = anchor.translation.copy()
            expected_translation[1] += 0.30 if side == 0 else -0.30
            self.assertLess(
                float(np.linalg.norm(actual.translation - expected_translation)),
                0.035,
            )
            self.assertLess(
                float(np.linalg.norm(ik.pin.log3(actual.rotation.T @ anchor.rotation))),
                0.20,
            )

    def test_lateral_extension_ramp_does_not_hold_then_jump(self):
        ik, q = self._anchored_solver()
        consecutive_stationary = 0
        maximum_stationary_run = 0
        maximum_elbow_step = 0.0

        for lateral in np.linspace(0.0, 0.30, 121)[1:]:
            left = np.eye(4, dtype=np.float64)
            right = np.eye(4, dtype=np.float64)
            left[1, 3] = lateral
            right[1, 3] = -lateral
            previous_q = q.copy()
            q, _ = ik.solve_ik(left, right, q, np.zeros(14))
            joint_step = float(np.max(np.abs(q - previous_q)))
            elbow_step = float(
                np.max(np.abs(q[[3, 10]] - previous_q[[3, 10]]))
            )
            maximum_elbow_step = max(maximum_elbow_step, elbow_step)
            if joint_step < 1e-6:
                consecutive_stationary += 1
                maximum_stationary_run = max(
                    maximum_stationary_run,
                    consecutive_stationary,
                )
            else:
                consecutive_stationary = 0

        self.assertLessEqual(maximum_stationary_run, 2)
        self.assertLess(maximum_elbow_step, 0.10)
        self.assertLess(float(np.max(np.abs(q[[3, 10]]))), 0.04)

    def test_large_lateral_step_makes_progress_instead_of_latching_hold(self):
        ik = AIWorkerVirtualLeaderIK(
            urdf_path=SG2_URDF,
            wrist_orientation_mode="relative-local",
        )
        identity = np.eye(4, dtype=np.float64)
        q, _ = ik.solve_ik(
            identity,
            identity,
            self.SIM_RESET_Q,
            np.zeros(14),
        )
        left = np.eye(4, dtype=np.float64)
        right = np.eye(4, dtype=np.float64)
        left[1, 3] = 0.30
        right[1, 3] = -0.30

        first_q, _ = ik.solve_ik(left, right, q, np.zeros(14))
        self.assertGreater(float(np.max(np.abs(first_q - q))), 1e-4)
        for _ in range(60):
            first_q, _ = ik.solve_ik(
                left,
                right,
                first_q,
                np.zeros(14),
            )
        self.assertGreater(
            float(np.max(np.abs(first_q - self.SIM_RESET_Q))),
            0.20,
        )

    def test_constant_reach_closed_loops_do_not_accumulate_nullspace_windup(self):
        ik, q = self._anchored_solver()
        anchor_q = q.copy()
        tangents = []
        for side in range(2):
            radial = (
                ik._robot_anchor[side].translation - ik._shoulder_roots[side]
            )
            radius = np.linalg.norm(radial)
            normal = radial / radius
            tangent_1 = np.cross(normal, np.array([0.0, 0.0, 1.0]))
            tangent_1 /= np.linalg.norm(tangent_1)
            tangent_2 = np.cross(normal, tangent_1)
            tangents.append((radius, normal, tangent_1, tangent_2))

        for _ in range(10):
            for phase in np.linspace(0.0, 2.0 * np.pi, 121)[1:]:
                inputs = []
                for side, (radius, normal, tangent_1, tangent_2) in enumerate(
                    tangents
                ):
                    direction = normal + 0.20 * (
                        (np.cos(phase) - 1.0) * tangent_1
                        + np.sin(phase) * tangent_2
                    )
                    direction /= np.linalg.norm(direction)
                    desired = ik._shoulder_roots[side] + radius * direction
                    pose = np.eye(4, dtype=np.float64)
                    pose[:3, 3] = desired - ik._robot_anchor[side].translation
                    inputs.append(pose)
                q, _ = ik.solve_ik(inputs[0], inputs[1], q, np.zeros(14))

        self.assertLess(float(np.max(np.abs(q - anchor_q))), 0.02)
        self.assertLessEqual(float(np.max(q[[3, 10]])), 1e-9)

    def test_lateral_closed_path_returns_without_joint_windup(self):
        ik, q = self._anchored_solver()
        anchor_q = q.copy()

        path = []
        for value in np.linspace(0.0, 0.12, 25)[1:]:
            path.append((value, 0.0))
        for value in np.linspace(0.0, 0.12, 25)[1:]:
            path.append((0.12, value))
        for value in np.linspace(0.12, 0.0, 25)[1:]:
            path.append((value, 0.12))
        for value in np.linspace(0.12, 0.0, 25)[1:]:
            path.append((0.0, value))
        path.extend([(0.0, 0.0)] * 30)

        for forward, lateral in path:
            left = np.eye(4, dtype=np.float64)
            right = np.eye(4, dtype=np.float64)
            left[:3, 3] = [forward, lateral, 0.0]
            right[:3, 3] = [forward, -lateral, 0.0]
            q, _ = ik.solve_ik(left, right, q, np.zeros(14))

        self.assertLess(float(np.max(np.abs(q - anchor_q))), 0.08)
        self.assertLessEqual(float(np.max(q[[3, 10]])), 1e-9)

    def test_unreachable_translation_is_projected_and_stays_finite(self):
        ik, q = self._anchored_solver()
        target = ik.pin.SE3(np.eye(3), ik._shoulder_roots[0] + np.array([2.0, 0.0, 0.0]))
        projected = ik._project_target_to_workspace(target, 0)
        projected_reach = np.linalg.norm(
            projected.translation - ik._shoulder_roots[0]
        )
        self.assertAlmostEqual(
            float(projected_reach),
            float(ik._straight_reach[0]),
            places=10,
        )

        far = self._translated_wrist(0.80)
        for _ in range(100):
            q, _ = ik.solve_ik(far, far, q, np.zeros(14))
        self.assertTrue(np.all(np.isfinite(q)))
        self.assertTrue(np.all(q >= ik.lower - 1e-12))
        self.assertTrue(np.all(q <= ik.upper + 1e-12))
        self.assertLessEqual(float(np.max(q[[3, 10]])), 1e-9)
        raw_targets = (
            ik._target(
                far,
                ik._input_anchor[0],
                ik._robot_anchor[0],
                ik._robot_orientation_anchor[0],
                ik._wrist_mount_rotations[0],
                ik._wrist_roll_corrections[0],
            ),
            ik._target(
                far,
                ik._input_anchor[1],
                ik._robot_anchor[1],
                ik._robot_orientation_anchor[1],
                ik._wrist_mount_rotations[1],
                ik._wrist_roll_corrections[1],
            ),
        )
        targets = tuple(
            ik._project_target_to_workspace(target, side)
            for side, target in enumerate(raw_targets)
        )
        actual_frames = ik._forward(q)
        for actual, target in zip(actual_frames, targets):
            self.assertLess(
                float(np.linalg.norm(actual.translation - target.translation)),
                ik._virtual_max_position_residual,
            )

    def test_backwards_elbow_startup_is_rejected_instead_of_moving_silently(self):
        ik = AIWorkerVirtualLeaderIK(
            urdf_path=SG2_URDF,
            wrist_orientation_mode="relative-local",
        )
        backwards_q = ik.ready_q.copy()
        backwards_q[[3, 10]] = 0.50
        identity = np.eye(4, dtype=np.float64)
        with self.assertRaisesRegex(RuntimeError, "backwards-elbow"):
            ik.solve_ik(identity, identity, backwards_q, np.zeros(14))


if __name__ == "__main__":
    unittest.main()
