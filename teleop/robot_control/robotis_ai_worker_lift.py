"""Bounded position control for the AI Worker lift joint."""

import threading

import numpy as np


class AIWorkerLiftController:
    """Map relative XR head height to a bounded AI Worker lift position."""

    LOWER = -0.5
    UPPER = 0.0
    TOPIC = "/leader/joystick_controller_right/joint_trajectory"
    JOINT = "lift_joint"

    def __init__(
        self,
        gain=1.0,
        deadband=0.015,
        smoothing_alpha=0.2,
        max_step=0.01,
        command_duration=0.08,
    ):
        from teleop.robot_control.robotis_dds import RobotisJointTrajectoryTransport

        self.gain = float(gain)
        self.deadband = max(0.0, float(deadband))
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.max_step = max(0.0, float(max_step))
        self.command_duration = max(0.02, float(command_duration))
        self._joint_state_received = threading.Event()
        self._measured_position = None
        self._neutral_head_z = None
        self._neutral_lift_position = None
        self._command = None
        self.transport = RobotisJointTrajectoryTransport(
            {"lift": self.TOPIC}, joint_state_callback=self._joint_state_cb
        )

    def _joint_state_cb(self, msg):
        positions = dict(zip(msg.name, msg.position))
        if self.JOINT in positions:
            self._measured_position = float(positions[self.JOINT])
            self._joint_state_received.set()

    def wait_for_joint_state(self, timeout=5.0):
        return self._joint_state_received.wait(timeout=max(0.0, float(timeout)))

    @staticmethod
    def _extract_head_z(head_pose):
        pose = np.asarray(head_pose, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("head_pose must be a finite 4x4 matrix")
        return float(pose[2, 3])

    def reset_neutral(self):
        """Re-anchor on the next valid XR frame without moving the lift."""
        self._neutral_head_z = None
        self._neutral_lift_position = None
        self._command = None

    def hold(self):
        """Hold the last commanded (or currently measured) lift position."""
        if self._measured_position is None:
            raise RuntimeError("lift_joint state has not been received")
        if self._command is None:
            self._command = float(np.clip(self._measured_position, self.LOWER, self.UPPER))
        self.transport.publish(
            "lift", (self.JOINT,), (self._command,), self.command_duration
        )
        return self._command

    def nudge(self, direction, distance):
        """Move the lift by a signed bounded increment and publish the new target."""
        if direction not in (-1, 0, 1):
            raise ValueError("direction must be -1, 0, or 1")
        distance = float(distance)
        if not np.isfinite(distance) or distance < 0.0:
            raise ValueError("distance must be finite and zero or greater")
        if self._measured_position is None:
            raise RuntimeError("lift_joint state has not been received")
        if self._command is None:
            self._command = float(np.clip(self._measured_position, self.LOWER, self.UPPER))
        self._command = float(
            np.clip(self._command + int(direction) * distance, self.LOWER, self.UPPER)
        )
        self.transport.publish(
            "lift", (self.JOINT,), (self._command,), self.command_duration
        )
        return self._command

    def update(self, head_pose):
        if self._measured_position is None:
            raise RuntimeError("lift_joint state has not been received")

        head_z = self._extract_head_z(head_pose)
        if self._neutral_head_z is None:
            self._neutral_head_z = head_z
            self._neutral_lift_position = float(
                np.clip(self._measured_position, self.LOWER, self.UPPER)
            )
            self._command = self._neutral_lift_position

        delta_z = head_z - self._neutral_head_z
        if abs(delta_z) < self.deadband:
            delta_z = 0.0
        target = float(
            np.clip(
                self._neutral_lift_position + self.gain * delta_z,
                self.LOWER,
                self.UPPER,
            )
        )
        filtered = self._command + self.smoothing_alpha * (target - self._command)
        if self.max_step > 0.0:
            filtered = self._command + float(
                np.clip(filtered - self._command, -self.max_step, self.max_step)
            )
        self._command = float(np.clip(filtered, self.LOWER, self.UPPER))
        self.transport.publish(
            "lift", (self.JOINT,), (self._command,), self.command_duration
        )
        return self._command, target, head_z, delta_z

    def close(self):
        self.transport.close()
