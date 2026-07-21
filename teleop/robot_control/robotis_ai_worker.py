"""ROBOTIS AI Worker dual-arm IK and DDS trajectory controller.

The kinematic model is loaded from the official ``ai_worker`` checkout instead
of duplicating its URDF in xr_tele.  Vision Pro and AI Worker wrist frames are
aligned on the first valid IK frame, then translation and spatial rotation
deltas are applied in the shared robot/world basis.
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

        if wrist_orientation_mode not in ("absolute", "relative"):
            raise ValueError("wrist_orientation_mode must be 'absolute' or 'relative'.")
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
        self.home_q = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0] * 2)
        home_frames = self._forward(self.home_q)
        self._corrected_home_rotations = (
            home_frames[0].rotation @ self._wrist_roll_corrections[0],
            home_frames[1].rotation @ self._wrist_roll_corrections[1],
        )
        self.translation_scale = float(translation_scale)
        self.max_iterations = int(max_iterations)
        self._input_anchor = None
        self._robot_anchor = None
        self._robot_orientation_anchor = None
        self._last_q = self.home_q.copy()

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
        else:
            # Preserve the robot's initial wrist orientation while applying the
            # VR rotation in the shared/world basis.  Multiplication in the old
            # order applied a human-local axis directly in the robot-local basis,
            # which made the two wrists rotate about different apparent axes.
            world_delta = input_pose[:3, :3] @ input_anchor[:3, :3].T
            rotation = world_delta @ robot_orientation_anchor
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
            dq += 0.003 * (self.home_q - q)
            max_step = 0.12
            step_norm = np.max(np.abs(dq))
            if step_norm > max_step:
                dq *= max_step / step_norm
            q = np.clip(self.pin.integrate(self.model, q, dq), self.lower, self.upper)

        self._last_q = q.copy()
        return q, np.zeros(14, dtype=np.float64)


class AIWorkerArmController:
    """Publish AI Worker arm targets using ROBOTIS CycloneDDS messages."""

    command_topic_description = "ROBOTIS DDS JointTrajectory (AI Worker left/right arms)"

    def __init__(self, command_duration=0.08, node_name="xr_tele_ai_worker_arms"):
        del node_name
        from teleop.robot_control.robotis_dds import RobotisJointTrajectoryTransport

        self.command_duration = max(0.02, float(command_duration))
        self.home_q = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0] * 2)
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

    def _publish_side(self, key, names, positions):
        self.transport.publish(key, names, positions, self.command_duration)

    def get_current_dual_arm_q(self):
        return self._q.copy()

    def get_current_dual_arm_dq(self):
        return self._dq.copy()

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

    def speed_gradual_max(self):
        # Unitree controllers ramp an internal DDS gain here. ROS trajectory
        # controllers already apply their configured limits.
        return None

    def ctrl_dual_arm_go_home(self):
        self._publish_side("left", AI_WORKER_LEFT_ARM_JOINTS, self.home_q[:7])
        self._publish_side("right", AI_WORKER_RIGHT_ARM_JOINTS, self.home_q[7:])
        self._last_command = self.home_q.copy()
        self._last_write_ok = True

    def get_last_write_ok(self):
        return self._last_write_ok

    def close(self):
        self.transport.close()
