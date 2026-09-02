"""ROBOTIS AI Worker dual-arm IK and DDS trajectory controller.

The kinematic model is loaded from the official ``ai_worker`` checkout instead
of duplicating its URDF in xr_tele.  Vision Pro and AI Worker wrist frames are
aligned on the first valid IK frame.  Wrist rotation can then follow either a
spatial delta in the shared robot/world basis or a body-local delta expressed
in the aligned wrist frames.
"""

from pathlib import Path
import os
import threading
import time

import numpy as np


AI_WORKER_ARM_JOINTS = tuple(
    [f"arm_l_joint{i}" for i in range(1, 8)]
    + [f"arm_r_joint{i}" for i in range(1, 8)]
)
AI_WORKER_LEFT_ARM_JOINTS = AI_WORKER_ARM_JOINTS[:7]
AI_WORKER_RIGHT_ARM_JOINTS = AI_WORKER_ARM_JOINTS[7:]
AI_WORKER_ARM_LOWER = np.array(
    [-3.14, 0.0, -3.14, -2.9361, -3.14, -1.57, -1.8201,
     -3.14, -3.14, -3.14, -2.9361, -3.14, -1.57, -1.5804]
)
AI_WORKER_ARM_UPPER = np.array(
    [3.14, 3.14, 3.14, 1.0786, 3.14, 1.57, 1.5804,
     3.14, 0.0, 3.14, 1.0786, 3.14, 1.57, 1.8201]
)
AI_WORKER_SH5_HOME_Q = np.array(
    [0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0] * 2,
    dtype=np.float64,
)
AI_WORKER_SG2_HOME_Q = np.zeros(14, dtype=np.float64)
AI_WORKER_SH5_READY_Q = AI_WORKER_SH5_HOME_Q.copy()
AI_WORKER_SG2_READY_Q = np.array(
    [0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0] * 2,
    dtype=np.float64,
)


def ai_worker_home_q_for_urdf(urdf_path):
    """Return the official arm initial pose for the selected AI Worker model."""
    normalized_path = str(Path(urdf_path).expanduser()).lower()
    if "ffw_sg2" in normalized_path:
        return AI_WORKER_SG2_HOME_Q.copy()
    return AI_WORKER_SH5_HOME_Q.copy()


def ai_worker_ready_q_for_urdf(urdf_path):
    """Return the bent-elbow pose used while waiting to start teleoperation."""
    normalized_path = str(Path(urdf_path).expanduser()).lower()
    if "ffw_sg2" in normalized_path:
        return AI_WORKER_SG2_READY_Q.copy()
    return AI_WORKER_SH5_READY_Q.copy()


def default_ai_worker_urdf() -> Path:
    env_path = os.environ.get("AI_WORKER_URDF")
    if env_path:
        return Path(env_path).expanduser().resolve()
    external_repos = Path(__file__).resolve().parents[3]
    return external_repos / "ai_worker/ffw_description/urdf/ffw_sh5_rev1_follower/ffw_sh5_follower.urdf"


class AIWorkerArmIK:
    """Damped least-squares IK for the two seven-axis AI Worker arms."""

    def __init__(
        self,
        urdf_path=None,
        translation_scale=1.0,
        max_iterations=24,
        wrist_orientation_mode="relative",
        left_wrist_roll_offset_deg=0.0,
        right_wrist_roll_offset_deg=0.0,
    ):
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise RuntimeError("AI Worker IK requires Pinocchio (python package 'pin').") from exc

        self.pin = pin
        self.urdf_path = Path(urdf_path).expanduser().resolve() if urdf_path else default_ai_worker_urdf()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(
                f"AI Worker URDF not found: {self.urdf_path}. "
                "Clone robotis-ai/ai_worker beside xr_tele or pass --ai-worker-urdf."
            )

        full_model = pin.buildModelFromUrdf(str(self.urdf_path))
        keep = set(AI_WORKER_ARM_JOINTS)
        locked = [jid for jid, name in enumerate(full_model.names) if jid > 0 and name not in keep]
        self.model = pin.buildReducedModel(full_model, locked, pin.neutral(full_model))
        reduced_names = tuple(self.model.names[1:])
        if reduced_names != AI_WORKER_ARM_JOINTS or self.model.nq != 14:
            raise RuntimeError(
                "Unexpected AI Worker reduced joint order: "
                f"{reduced_names}; expected {AI_WORKER_ARM_JOINTS}."
            )

        # TeleVuer uses H1_2's semantic wrist convention: local Y is wrist
        # pitch and local Z is terminal wrist yaw.  AI Worker's link7 frame is
        # physically equivalent, but its local axes are rotated: link7 local X
        # is terminal yaw, local Y is pitch, and local Z points opposite the
        # semantic X axis.  Add zero-offset operational frames that express the
        # H1_2 convention without changing the URDF or the HX5 hand model.
        link7_from_semantic = np.array(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        self.left_frame_id = self.model.addFrame(
            pin.Frame(
                "ai_worker_left_wrist_ee",
                self.model.getJointId("arm_l_joint7"),
                pin.SE3(link7_from_semantic, np.zeros(3)),
                pin.FrameType.OP_FRAME,
            )
        )
        self.right_frame_id = self.model.addFrame(
            pin.Frame(
                "ai_worker_right_wrist_ee",
                self.model.getJointId("arm_r_joint7"),
                pin.SE3(link7_from_semantic, np.zeros(3)),
                pin.FrameType.OP_FRAME,
            )
        )
        self.data = self.model.createData()

        if wrist_orientation_mode not in ("absolute", "relative", "relative-local"):
            raise ValueError(
                "wrist_orientation_mode must be 'absolute', 'relative', "
                "or 'relative-local'."
            )
        self.wrist_orientation_mode = wrist_orientation_mode

        # The virtual target frames have the same semantic orientation as
        # H1_2's wrist-yaw EE, so no HX5 mount rotation belongs in the target.
        self._wrist_mount_rotations = (np.eye(3), np.eye(3))
        self._wrist_roll_corrections = (
            self._rotation_z(np.deg2rad(left_wrist_roll_offset_deg)),
            self._rotation_z(np.deg2rad(right_wrist_roll_offset_deg)),
        )
        self._has_wrist_roll_correction = (
            not np.isclose(left_wrist_roll_offset_deg, 0.0),
            not np.isclose(right_wrist_roll_offset_deg, 0.0),
        )

        self.lower = np.asarray(self.model.lowerPositionLimit, dtype=np.float64)
        self.upper = np.asarray(self.model.upperPositionLimit, dtype=np.float64)
        self.home_q = ai_worker_home_q_for_urdf(self.urdf_path)
        self.ready_q = ai_worker_ready_q_for_urdf(self.urdf_path)
        ready_frames = self._forward(self.ready_q)
        self._corrected_home_rotations = (
            ready_frames[0].rotation @ self._wrist_roll_corrections[0],
            ready_frames[1].rotation @ self._wrist_roll_corrections[1],
        )
        self.translation_scale = float(translation_scale)
        self.max_iterations = int(max_iterations)
        self._input_anchor = None
        self._robot_anchor = None
        self._robot_orientation_anchor = None
        self._last_q = self.ready_q.copy()

    @staticmethod
    def _pose(matrix):
        pose = np.asarray(matrix, dtype=np.float64)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError("Wrist target must be a finite 4x4 transform.")
        return pose

    @staticmethod
    def _rotation_z(angle):
        cosine = np.cos(angle)
        sine = np.sin(angle)
        return np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def reset_anchor(self):
        self._input_anchor = None
        self._robot_anchor = None
        self._robot_orientation_anchor = None

    def _forward(self, q):
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        return (
            self.data.oMf[self.left_frame_id].copy(),
            self.data.oMf[self.right_frame_id].copy(),
        )

    def _target(
        self,
        input_pose,
        input_anchor,
        robot_anchor,
        robot_orientation_anchor,
        wrist_mount_rotation,
        wrist_roll_correction,
    ):
        if self.wrist_orientation_mode == "absolute":
            # The virtual wrist EE uses the same absolute orientation convention
            # as H1_2; the HX5 mount is intentionally outside the arm IK target.
            rotation = input_pose[:3, :3] @ wrist_mount_rotation
        elif self.wrist_orientation_mode == "relative":
            # Preserve the robot's initial wrist orientation while applying the
            # VR rotation in the shared/world basis.  Multiplication in the old
            # order applied a human-local axis directly in the robot-local basis,
            # which made the two wrists rotate about different apparent axes.
            world_delta = input_pose[:3, :3] @ input_anchor[:3, :3].T
            rotation = world_delta @ robot_orientation_anchor
        else:
            # Preserve the robot's initial wrist orientation while mapping a
            # rotation about an operator-wrist local axis to the same local axis
            # of the aligned robot wrist.  This keeps pitch and terminal wrist
            # rotation distinct even when the two anchor frames differ by the
            # left/right hand-convention correction.
            local_delta = input_anchor[:3, :3].T @ input_pose[:3, :3]
            rotation = robot_orientation_anchor @ local_delta
        if self.wrist_orientation_mode == "absolute":
            # Absolute targets have no startup anchor, so apply their static
            # hand-axis correction directly.
            rotation = rotation @ wrist_roll_correction
        translation = robot_anchor.translation + self.translation_scale * (
            input_pose[:3, 3] - input_anchor[:3, 3]
        )
        return self.pin.SE3(rotation, translation)

    def solve_ik(self, left_wrist, right_wrist, current_lr_arm_q=None, current_lr_arm_dq=None):
        del current_lr_arm_dq
        left_input = self._pose(left_wrist)
        right_input = self._pose(right_wrist)
        q = np.asarray(current_lr_arm_q, dtype=np.float64).reshape(-1) if current_lr_arm_q is not None else self._last_q.copy()
        if q.size != 14 or not np.all(np.isfinite(q)):
            q = self._last_q.copy()
        q = np.clip(q, self.lower, self.upper)

        if self._input_anchor is None:
            self._input_anchor = (left_input.copy(), right_input.copy())
            self._robot_anchor = self._forward(q)
            # A non-zero offset is an absolute calibration from SH5 home, not
            # an increment to apply on every teleop restart.  A zero-offset side
            # preserves its measured startup orientation exactly.
            self._robot_orientation_anchor = tuple(
                self._corrected_home_rotations[index]
                if self._has_wrist_roll_correction[index]
                else self._robot_anchor[index].rotation.copy()
                for index in range(2)
            )

        targets = (
            self._target(
                left_input,
                self._input_anchor[0],
                self._robot_anchor[0],
                self._robot_orientation_anchor[0],
                self._wrist_mount_rotations[0],
                self._wrist_roll_corrections[0],
            ),
            self._target(
                right_input,
                self._input_anchor[1],
                self._robot_anchor[1],
                self._robot_orientation_anchor[1],
                self._wrist_mount_rotations[1],
                self._wrist_roll_corrections[1],
            ),
        )
        frame_ids = (self.left_frame_id, self.right_frame_id)

        for _ in range(self.max_iterations):
            self.pin.forwardKinematics(self.model, self.data, q)
            self.pin.updateFramePlacements(self.model, self.data)
            errors = []
            jacobians = []
            for frame_id, target in zip(frame_ids, targets):
                current = self.data.oMf[frame_id]
                errors.append(self.pin.log6(current.inverse() * target).vector)
                jacobians.append(
                    self.pin.computeFrameJacobian(
                        self.model, self.data, q, frame_id, self.pin.ReferenceFrame.LOCAL
                    )
                )
            error = np.concatenate(errors)
            if np.linalg.norm(error) < 1e-4:
                break
            jacobian = np.vstack(jacobians)
            damping = 2e-3
            dq = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(12), error
            )
            dq += 0.003 * (self.ready_q - q)
            max_step = 0.12
            step_norm = np.max(np.abs(dq))
            if step_norm > max_step:
                dq *= max_step / step_norm
            q = np.clip(self.pin.integrate(self.model, q, dq), self.lower, self.upper)

        self._last_q = q.copy()
        return q, np.zeros(14, dtype=np.float64)


class AIWorkerVirtualLeaderIK(AIWorkerArmIK):
    """Reach-aware SG2 IK that behaves like a continuous virtual 7-axis leader.

    The legacy solver intentionally remains unchanged.  This opt-in solver uses
    the measured joint state as its primary seed, keeps the elbows on the
    anatomical branch, and selects elbow bend from hand reach.  Whole-arm
    continuity comes from measured-state reseeding plus a weak shoulder
    null-space anchor.  During extension, elbow and axial-arm posture enter the
    weighted least-squares objective so a small wrist-pose residual is preferred
    over a corkscrewed or backwards-elbow solution.
    """

    _ELBOW_INDICES = (3, 10)
    _AXIAL_POSTURE_INDICES = ((2, 4), (9, 11))
    _WRIST_JOINT_INDICES = (4, 5, 6, 11, 12, 13)
    _SHOULDER_JOINT_NAMES = ("arm_l_joint1", "arm_r_joint1")

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("max_iterations", 32)
        super().__init__(*args, **kwargs)
        if "ffw_sg2" not in str(self.urdf_path).lower():
            raise ValueError(
                "AI Worker virtual-leader IK currently supports the SG2 follower URDF only."
            )

        self._virtual_damping = 2e-3
        self._virtual_max_step = 0.12
        self._virtual_translation_weight = np.sqrt(50.0)
        self._virtual_rotation_weight = 1.0
        self._virtual_anchor_nullspace_gain = 0.05
        self._virtual_elbow_regularization = 1.0
        self._virtual_axial_regularization = 0.10
        self._virtual_position_tolerance = 1e-4
        self._virtual_rotation_tolerance = 1e-3
        self._virtual_branch_retry_threshold = 0.10
        self._virtual_max_position_residual = 0.08
        self._virtual_startup_elbow_tolerance = 0.05
        self._posture_anchor_q = None
        self._anchor_reach = None
        self._shoulder_joint_ids = tuple(
            self.model.getJointId(name) for name in self._SHOULDER_JOINT_NAMES
        )

        self._forward(self.ready_q)
        self._shoulder_roots = tuple(
            self.data.oMi[joint_id].translation.copy()
            for joint_id in self._shoulder_joint_ids
        )
        ready_frames = self._forward(self.ready_q)
        straight_frames = self._forward(self.home_q)
        self._ready_reach = np.array(
            [
                np.linalg.norm(frame.translation - root)
                for frame, root in zip(ready_frames, self._shoulder_roots)
            ],
            dtype=np.float64,
        )
        self._straight_reach = np.array(
            [
                np.linalg.norm(frame.translation - root)
                for frame, root in zip(straight_frames, self._shoulder_roots)
            ],
            dtype=np.float64,
        )
        if np.any(self._straight_reach <= self._ready_reach + 1e-6):
            raise RuntimeError(
                "SG2 virtual-leader IK requires the home pose to reach farther than the ready pose."
            )

    def reset_anchor(self):
        super().reset_anchor()
        self._posture_anchor_q = None
        self._anchor_reach = None

    def _project_target_to_workspace(self, target, side):
        """Project targets onto the straight, anatomical-elbow reach sphere."""
        root = self._shoulder_roots[side]
        offset = target.translation - root
        reach = float(np.linalg.norm(offset))
        maximum_reach = self._straight_reach[side]
        if reach <= maximum_reach or reach <= 1e-9:
            return target
        translation = root + offset * (maximum_reach / reach)
        return self.pin.SE3(target.rotation.copy(), translation)

    def _elbow_posture_target(self, targets):
        posture = np.zeros(14, dtype=np.float64)
        anchor_q = self._posture_anchor_q
        anchor_reach = self._anchor_reach
        if anchor_q is None or anchor_reach is None:
            anchor_q = self.ready_q
            anchor_reach = self._ready_reach
        for side, (target, root, elbow_index) in enumerate(
            zip(targets, self._shoulder_roots, self._ELBOW_INDICES)
        ):
            reach = np.linalg.norm(target.translation - root)
            start_reach = anchor_reach[side]
            end_reach = max(self._straight_reach[side], start_reach + 1e-3)
            reach_span = end_reach - start_reach
            extension = np.clip(
                (reach - start_reach) / max(reach_span, 1e-6),
                0.0,
                1.0,
            )
            extension = extension * extension * (3.0 - 2.0 * extension)
            anchor_elbow = min(anchor_q[elbow_index], 0.0)
            elbow_target = (1.0 - extension) * anchor_elbow
            posture[elbow_index] = elbow_target
        return posture

    def _extension_from_elbow_target(self, elbow_target):
        extension = np.zeros(2, dtype=np.float64)
        for side, elbow_index in enumerate(self._ELBOW_INDICES):
            anchor_elbow = min(self._posture_anchor_q[elbow_index], 0.0)
            if anchor_elbow < -1e-6:
                extension[side] = np.clip(
                    1.0 - elbow_target[elbow_index] / anchor_elbow,
                    0.0,
                    1.0,
                )
            else:
                extension[side] = 1.0
        return extension

    def _task_terms(self, q, targets):
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        errors = []
        jacobians = []
        max_position_error = 0.0
        max_rotation_error = 0.0
        for frame_id, target in zip((self.left_frame_id, self.right_frame_id), targets):
            current = self.data.oMf[frame_id]
            motion_error = self.pin.log6(current.inverse() * target)
            errors.append(motion_error.vector)
            jacobians.append(
                self.pin.computeFrameJacobian(
                    self.model,
                    self.data,
                    q,
                    frame_id,
                    self.pin.ReferenceFrame.LOCAL,
                )
            )
            max_position_error = max(
                max_position_error,
                float(np.linalg.norm(motion_error.linear)),
            )
            max_rotation_error = max(
                max_rotation_error,
                float(np.linalg.norm(motion_error.angular)),
            )

        error = np.concatenate(errors)
        jacobian = np.vstack(jacobians)
        weights = []
        for _ in targets:
            weights.extend(
                [self._virtual_translation_weight] * 3
                + [self._virtual_rotation_weight] * 3
            )
        weights = np.asarray(weights, dtype=np.float64)
        return (
            weights * error,
            weights[:, None] * jacobian,
            max_position_error,
            max_rotation_error,
        )

    def _task_metrics(self, q, targets):
        frames = self._forward(q)
        metrics = []
        for current, target in zip(frames, targets):
            motion_error = self.pin.log6(current.inverse() * target)
            position_error = float(np.linalg.norm(motion_error.linear))
            rotation_error = float(np.linalg.norm(motion_error.angular))
            rotation_cost = self._virtual_rotation_weight**2
            cost = self._virtual_translation_weight**2 * position_error**2 + (
                rotation_cost * rotation_error**2
            )
            metrics.append((position_error, rotation_error, cost))
        return metrics

    def _objective_costs(self, q, targets, elbow_target, extension):
        metrics = self._task_metrics(q, targets)
        costs = np.array([metric[2] for metric in metrics], dtype=np.float64)
        for side, elbow_index in enumerate(self._ELBOW_INDICES):
            side_extension = extension[side]
            costs[side] += (
                self._virtual_elbow_regularization
                * side_extension
                * (q[elbow_index] - elbow_target[elbow_index]) ** 2
            )
            for axial_index in self._AXIAL_POSTURE_INDICES[side]:
                costs[side] += (
                    self._virtual_axial_regularization
                    * side_extension
                    * (q[axial_index] - self._posture_anchor_q[axial_index]) ** 2
                )
        return costs

    def _solve_candidate(self, seed_q, targets, elbow_target, extension):
        q = np.asarray(seed_q, dtype=np.float64).reshape(14).copy()
        q = np.clip(q, self.lower, self.upper)
        first_step_q = q.copy()
        safe_upper = self.upper.copy()
        safe_upper[list(self._ELBOW_INDICES)] = 0.0
        regularization = np.zeros(14, dtype=np.float64)
        posture_reference = self._posture_anchor_q.copy()
        for side, elbow_index in enumerate(self._ELBOW_INDICES):
            side_extension = extension[side]
            regularization[elbow_index] += (
                self._virtual_elbow_regularization * side_extension
            )
            posture_reference[elbow_index] = elbow_target[elbow_index]
            for axial_index in self._AXIAL_POSTURE_INDICES[side]:
                regularization[axial_index] += (
                    self._virtual_axial_regularization * side_extension
                )

        for iteration in range(self.max_iterations):
            (
                weighted_error,
                weighted_jacobian,
                position_error,
                rotation_error,
            ) = self._task_terms(q, targets)
            elbow_error = np.max(
                np.abs(
                    q[list(self._ELBOW_INDICES)]
                    - elbow_target[list(self._ELBOW_INDICES)]
                )
            )
            system = (
                weighted_jacobian.T @ weighted_jacobian
                + np.diag(regularization)
                + self._virtual_damping * np.eye(14)
            )
            rhs = (
                weighted_jacobian.T @ weighted_error
                + regularization * (posture_reference - q)
            )
            dq = np.linalg.solve(system, rhs)
            task_projector = (
                np.eye(14)
                - np.linalg.pinv(weighted_jacobian, rcond=1e-5)
                @ weighted_jacobian
            )
            anchor_error = posture_reference - q
            anchor_error[list(self._WRIST_JOINT_INDICES)] = 0.0
            dq += self._virtual_anchor_nullspace_gain * (
                task_projector @ anchor_error
            )
            if not np.all(np.isfinite(dq)):
                return None
            max_step = np.max(np.abs(dq))
            if (
                max_step <= 1e-6
                or (
                    position_error <= self._virtual_position_tolerance
                    and rotation_error <= self._virtual_rotation_tolerance
                    and elbow_error <= 0.01
                    and np.max(
                        np.abs(task_projector @ anchor_error)
                    )
                    <= 1e-5
                )
            ):
                break
            if max_step > self._virtual_max_step:
                dq *= self._virtual_max_step / max_step
            q = self.pin.integrate(self.model, q, dq)
            q = np.clip(q, self.lower, safe_upper)
            if iteration == 0:
                first_step_q = q.copy()
        return q, first_step_q

    def solve_ik(self, left_wrist, right_wrist, current_lr_arm_q=None, current_lr_arm_dq=None):
        del current_lr_arm_dq
        left_input = self._pose(left_wrist)
        right_input = self._pose(right_wrist)
        measured_q = (
            np.asarray(current_lr_arm_q, dtype=np.float64).reshape(-1)
            if current_lr_arm_q is not None
            else self._last_q.copy()
        )
        if measured_q.size != 14 or not np.all(np.isfinite(measured_q)):
            measured_q = self._last_q.copy()
        measured_q = np.clip(measured_q, self.lower, self.upper)

        if self._input_anchor is None:
            startup_elbows = measured_q[list(self._ELBOW_INDICES)]
            if np.any(startup_elbows > self._virtual_startup_elbow_tolerance):
                raise RuntimeError(
                    "AI Worker virtual-leader cannot anchor from the backwards-elbow "
                    f"branch (q4={np.round(startup_elbows, 4).tolist()}). "
                    "Reset the simulator or move the arms to a supervised safe pose first."
                )
            self._input_anchor = (left_input.copy(), right_input.copy())
            self._robot_anchor = self._forward(measured_q)
            self._robot_orientation_anchor = tuple(
                self._corrected_home_rotations[index]
                if self._has_wrist_roll_correction[index]
                else self._robot_anchor[index].rotation.copy()
                for index in range(2)
            )
            self._posture_anchor_q = measured_q.copy()
            self._anchor_reach = np.array(
                [
                    np.linalg.norm(frame.translation - root)
                    for frame, root in zip(self._robot_anchor, self._shoulder_roots)
                ],
                dtype=np.float64,
            )
            self._last_q = measured_q.copy()

        raw_targets = (
            self._target(
                left_input,
                self._input_anchor[0],
                self._robot_anchor[0],
                self._robot_orientation_anchor[0],
                self._wrist_mount_rotations[0],
                self._wrist_roll_corrections[0],
            ),
            self._target(
                right_input,
                self._input_anchor[1],
                self._robot_anchor[1],
                self._robot_orientation_anchor[1],
                self._wrist_mount_rotations[1],
                self._wrist_roll_corrections[1],
            ),
        )
        targets = tuple(
            self._project_target_to_workspace(target, side)
            for side, target in enumerate(raw_targets)
        )
        previous_q = measured_q.copy()
        elbow_target = self._elbow_posture_target(targets)
        extension = self._extension_from_elbow_target(elbow_target)

        try:
            candidate = self._solve_candidate(
                measured_q,
                targets,
                elbow_target,
                extension,
            )
            if candidate is None:
                return previous_q, np.zeros(14, dtype=np.float64)
            q, measured_first_step_q = candidate

            elbow_mismatch = np.abs(
                q[list(self._ELBOW_INDICES)]
                - elbow_target[list(self._ELBOW_INDICES)]
            )
            retry_sides = elbow_mismatch > self._virtual_branch_retry_threshold
            if np.any(retry_sides):
                anchor_result = self._solve_candidate(
                    self._posture_anchor_q,
                    targets,
                    elbow_target,
                    extension,
                )
                if anchor_result is not None:
                    anchor_candidate, _ = anchor_result
                    measured_metrics = self._task_metrics(q, targets)
                    anchor_metrics = self._task_metrics(
                        anchor_candidate,
                        targets,
                    )
                    for side, retry_side in enumerate(retry_sides):
                        if not retry_side:
                            continue
                        arm_slice = slice(side * 7, side * 7 + 7)
                        elbow_index = self._ELBOW_INDICES[side]
                        measured_position, measured_rotation, measured_cost = (
                            measured_metrics[side]
                        )
                        anchor_position, anchor_rotation, anchor_cost = (
                            anchor_metrics[side]
                        )
                        task_is_equivalent = (
                            anchor_position <= max(1e-3, measured_position + 1e-3)
                            and anchor_rotation <= max(1e-2, measured_rotation + 1e-2)
                        )
                        anchor_elbow_error = abs(
                            anchor_candidate[elbow_index] - elbow_target[elbow_index]
                        )
                        measured_elbow_error = abs(
                            q[elbow_index] - elbow_target[elbow_index]
                        )
                        posture_is_better = (
                            anchor_elbow_error + 1e-4 < measured_elbow_error
                        )
                        if (
                            task_is_equivalent and posture_is_better
                        ) or anchor_cost + 1e-8 < measured_cost:
                            q[arm_slice] = anchor_candidate[arm_slice]
        except np.linalg.LinAlgError:
            return previous_q, np.zeros(14, dtype=np.float64)

        solved_metrics = self._task_metrics(q, targets)
        previous_costs = self._objective_costs(
            previous_q,
            targets,
            elbow_target,
            extension,
        )
        solved_q = q.copy()
        q = previous_q.copy()
        for side, previous_cost in enumerate(previous_costs):
            arm_slice = slice(side * 7, side * 7 + 7)
            elbow_index = self._ELBOW_INDICES[side]
            position_error = solved_metrics[side][0]
            if (
                not np.isfinite(position_error)
                or position_error > self._virtual_max_position_residual
            ):
                continue

            best_cost = previous_cost
            best_arm_q = previous_q[arm_slice].copy()
            for step_target in (solved_q, measured_first_step_q):
                command_delta = (
                    step_target[arm_slice] - previous_q[arm_slice]
                )
                command_step = float(np.max(np.abs(command_delta)))
                if command_step > self._virtual_max_step:
                    command_delta *= self._virtual_max_step / command_step
                for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
                    trial_q = q.copy()
                    trial_q[arm_slice] = (
                        previous_q[arm_slice] + alpha * command_delta
                    )
                    trial_q = np.clip(trial_q, self.lower, self.upper)
                    if previous_q[elbow_index] <= 0.0:
                        trial_q[elbow_index] = min(trial_q[elbow_index], 0.0)
                    else:
                        trial_q[elbow_index] = min(
                            trial_q[elbow_index],
                            previous_q[elbow_index],
                        )
                    trial_cost = self._objective_costs(
                        trial_q,
                        targets,
                        elbow_target,
                        extension,
                    )[side]
                    if np.isfinite(trial_cost) and trial_cost < best_cost - 1e-12:
                        best_cost = trial_cost
                        best_arm_q = trial_q[arm_slice].copy()
            q[arm_slice] = best_arm_q
        if not np.all(np.isfinite(q)):
            return previous_q, np.zeros(14, dtype=np.float64)
        self._last_q = q.copy()
        return q, np.zeros(14, dtype=np.float64)


class AIWorkerArmController:
    """Publish AI Worker arm targets using ROBOTIS CycloneDDS messages."""

    command_topic_description = "ROBOTIS DDS JointTrajectory (AI Worker left/right arms)"

    def __init__(
        self,
        command_duration=0.08,
        node_name="xr_tele_ai_worker_arms",
        home_q=None,
        ready_q=None,
    ):
        del node_name
        from teleop.robot_control.robotis_dds import RobotisJointTrajectoryTransport

        self.command_duration = max(0.02, float(command_duration))
        self.home_q = (
            AI_WORKER_SH5_HOME_Q.copy()
            if home_q is None
            else np.asarray(home_q, dtype=np.float64).reshape(14).copy()
        )
        self.ready_q = (
            self.home_q.copy()
            if ready_q is None
            else np.asarray(ready_q, dtype=np.float64).reshape(14).copy()
        )
        self.arm_velocity_limit = 3.0
        self._q = self.home_q.copy()
        self._dq = np.zeros(14)
        self._last_command = self.home_q.copy()
        self._last_command_time = time.monotonic()
        self._last_write_ok = False
        self._joint_state_received = threading.Event()
        self.transport = RobotisJointTrajectoryTransport(
            {
                "left": "/leader/joint_trajectory_command_broadcaster_left/joint_trajectory",
                "right": "/leader/joint_trajectory_command_broadcaster_right/joint_trajectory",
            },
            joint_state_callback=self._joint_state_cb,
        )

    def _joint_state_cb(self, msg):
        positions = dict(zip(msg.name, msg.position))
        velocities = dict(zip(msg.name, msg.velocity)) if len(msg.velocity) == len(msg.name) else {}
        for i, name in enumerate(AI_WORKER_ARM_JOINTS):
            if name in positions:
                self._q[i] = positions[name]
            if name in velocities:
                self._dq[i] = velocities[name]
        if (
            not self._joint_state_received.is_set()
            and all(name in positions for name in AI_WORKER_ARM_JOINTS)
        ):
            self._last_command = self._q.copy()
            self._last_command_time = time.monotonic()
            self._joint_state_received.set()

    def wait_for_joint_state(self, timeout=5.0):
        """Wait until all 14 measured arm joints have initialized the controller."""
        return self._joint_state_received.wait(timeout=max(0.0, float(timeout)))

    def sync_arm_command_to_measured(self):
        """Re-anchor the velocity limiter to the latest measured arm pose."""
        measured = np.asarray(self.get_current_dual_arm_q(), dtype=np.float64).reshape(14)
        if not np.all(np.isfinite(measured)):
            raise RuntimeError("Cannot anchor arm commands to invalid joint state.")
        self._last_command = np.clip(
            measured,
            AI_WORKER_ARM_LOWER,
            AI_WORKER_ARM_UPPER,
        )
        self._last_command_time = time.monotonic()
        return self._last_command.copy()

    def _publish_side(self, key, names, positions):
        self.transport.publish(key, names, positions, self.command_duration)

    def get_current_dual_arm_q(self):
        return self._q.copy()

    def get_current_dual_arm_dq(self):
        return self._dq.copy()

    def get_last_commanded_dual_arm_q(self):
        """Return the most recent post-limit arm position command."""
        return self._last_command.copy()

    def get_current_motor_q(self):
        return self.get_current_dual_arm_q()

    def ctrl_dual_arm(self, q, tau=None):
        del tau
        target = np.asarray(q, dtype=np.float64).reshape(14)
        target = np.clip(target, AI_WORKER_ARM_LOWER, AI_WORKER_ARM_UPPER)
        now = time.monotonic()
        dt = max(now - self._last_command_time, 1.0 / 100.0)
        max_delta = max(0.01, float(self.arm_velocity_limit)) * dt
        target = self._last_command + np.clip(target - self._last_command, -max_delta, max_delta)
        self._publish_side("left", AI_WORKER_LEFT_ARM_JOINTS, target[:7])
        self._publish_side("right", AI_WORKER_RIGHT_ARM_JOINTS, target[7:])
        self._last_command = target
        self._last_command_time = now
        self._last_write_ok = True

    def ctrl_dual_arm_smooth_to(self, q, duration, num_points=100):
        """Send one zero-velocity quintic trajectory from measured q to target q."""
        target = np.asarray(q, dtype=np.float64).reshape(14)
        target = np.clip(target, AI_WORKER_ARM_LOWER, AI_WORKER_ARM_UPPER)
        start = np.asarray(self.get_current_dual_arm_q(), dtype=np.float64).reshape(14)
        if not np.all(np.isfinite(start)):
            raise RuntimeError("Cannot start a smooth arm trajectory from invalid joint state.")

        delta = target - start
        requested_duration = max(0.1, float(duration))
        velocity_limit = max(0.01, float(self.arm_velocity_limit))
        # The peak derivative of 10u^3 - 15u^4 + 6u^5 is 1.875.
        minimum_duration = 1.875 * float(np.max(np.abs(delta))) / velocity_limit
        trajectory_duration = max(requested_duration, minimum_duration)
        num_points = max(2, int(num_points))

        times = np.linspace(0.0, trajectory_duration, num_points, dtype=np.float64)
        u = times / trajectory_duration
        position_coeff = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        velocity_coeff = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / trajectory_duration
        acceleration_coeff = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / (
            trajectory_duration * trajectory_duration
        )
        positions = start[None, :] + position_coeff[:, None] * delta[None, :]
        velocities = velocity_coeff[:, None] * delta[None, :]
        accelerations = acceleration_coeff[:, None] * delta[None, :]

        self.transport.publish_trajectory(
            "left",
            AI_WORKER_LEFT_ARM_JOINTS,
            positions[:, :7],
            times,
            velocities[:, :7],
            accelerations[:, :7],
        )
        self.transport.publish_trajectory(
            "right",
            AI_WORKER_RIGHT_ARM_JOINTS,
            positions[:, 7:],
            times,
            velocities[:, 7:],
            accelerations[:, 7:],
        )
        self._last_command = target.copy()
        self._last_command_time = time.monotonic()
        self._last_write_ok = True
        return trajectory_duration

    def speed_gradual_max(self):
        # Unitree controllers ramp an internal DDS gain here. ROS trajectory
        # controllers already apply their configured limits.
        return None

    def ctrl_dual_arm_go_home(self):
        self._publish_side("left", AI_WORKER_LEFT_ARM_JOINTS, self.home_q[:7])
        self._publish_side("right", AI_WORKER_RIGHT_ARM_JOINTS, self.home_q[7:])
        self._last_command = self.home_q.copy()
        self._last_write_ok = True

    def ctrl_dual_arm_go_ready(self):
        self._publish_side("left", AI_WORKER_LEFT_ARM_JOINTS, self.ready_q[:7])
        self._publish_side("right", AI_WORKER_RIGHT_ARM_JOINTS, self.ready_q[7:])
        self._last_command = self.ready_q.copy()
        self._last_command_time = time.monotonic()
        self._last_write_ok = True

    def get_last_write_ok(self):
        return self._last_write_ok

    def close(self):
        self.transport.close()
