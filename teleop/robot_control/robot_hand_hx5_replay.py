"""Direct, validated HX5-D20 joint replay for the ROBOTIS AI Worker.

This publisher deliberately does not instantiate the live hand retargeter.  It
only accepts already-retargeted joint positions in the existing HX5 hardware
order: 20 left-hand joints followed by 20 right-hand joints.
"""

import threading

import numpy as np

from teleop.robot_control.robot_hand_hx5_d20 import (
    HX5_D20_NUM_JOINTS,
    LEFT_JOINT_NAMES,
    LEFT_LOWER,
    LEFT_UPPER,
    RIGHT_JOINT_NAMES,
    RIGHT_LOWER,
    RIGHT_UPPER,
)


_HX5_REPLAY_TOPICS = {
    "left": "/leader/joint_trajectory_command_broadcaster_left_hand/joint_trajectory",
    "right": "/leader/joint_trajectory_command_broadcaster_right_hand/joint_trajectory",
}
_ALL_JOINT_NAMES = LEFT_JOINT_NAMES + RIGHT_JOINT_NAMES


class AIWorkerHX5ReplayPublisher:
    """Publish recorded HX5 qpos without running landmark retargeting."""

    def __init__(self, command_duration=0.08, transport_factory=None):
        command_duration = float(command_duration)
        if not np.isfinite(command_duration) or command_duration <= 0.0:
            raise ValueError("command_duration must be a finite positive number.")

        if transport_factory is None:
            from teleop.robot_control.robotis_dds import RobotisJointTrajectoryTransport

            transport_factory = RobotisJointTrajectoryTransport

        self.command_duration = max(0.02, command_duration)
        self._state = np.zeros(HX5_D20_NUM_JOINTS * 2, dtype=np.float64)
        self._last_command = np.zeros(HX5_D20_NUM_JOINTS * 2, dtype=np.float64)
        self._seen_joint_names = set()
        self._joint_state_received = threading.Event()
        self._state_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._closed = False
        self.transport = transport_factory(
            dict(_HX5_REPLAY_TOPICS),
            joint_state_callback=self._joint_state_cb,
        )

    def _joint_state_cb(self, msg):
        positions = dict(zip(msg.name, msg.position))
        with self._state_lock:
            for index, name in enumerate(_ALL_JOINT_NAMES):
                if name not in positions:
                    continue
                value = float(positions[name])
                if not np.isfinite(value):
                    continue
                self._state[index] = value
                self._seen_joint_names.add(name)
            if len(self._seen_joint_names) == len(_ALL_JOINT_NAMES):
                self._joint_state_received.set()

    @staticmethod
    def _validate_target(side, values, lower, upper):
        target = np.asarray(values, dtype=np.float64).reshape(-1)
        if target.size != HX5_D20_NUM_JOINTS:
            raise ValueError(
                f"{side} HX5 target expected {HX5_D20_NUM_JOINTS} values, "
                f"got {target.size}."
            )
        if not np.isfinite(target).all():
            raise ValueError(f"{side} HX5 target must contain only finite values.")
        below = np.flatnonzero(target < lower)
        above = np.flatnonzero(target > upper)
        if below.size or above.size:
            invalid = np.concatenate((below, above))
            invalid = np.unique(invalid).tolist()
            raise ValueError(
                f"{side} HX5 target violates joint limits at indices {invalid}."
            )
        return target.copy()

    def wait_for_joint_state(self, timeout=5.0):
        """Wait until finite positions have been observed for all 40 joints."""
        return self._joint_state_received.wait(timeout=max(0.0, float(timeout)))

    def write(self, left, right):
        """Validate and publish one left/right frame, returning the sent 40-vector."""
        # Validate both sides before the first DDS write so malformed input can
        # never intentionally produce a one-sided command.
        left_target = self._validate_target("left", left, LEFT_LOWER, LEFT_UPPER)
        right_target = self._validate_target("right", right, RIGHT_LOWER, RIGHT_UPPER)
        command = np.concatenate((left_target, right_target))

        with self._write_lock:
            if self._closed:
                raise RuntimeError("HX5 replay publisher is closed.")
            self.transport.publish(
                "left", LEFT_JOINT_NAMES, left_target, self.command_duration
            )
            self.transport.publish(
                "right", RIGHT_JOINT_NAMES, right_target, self.command_duration
            )
            with self._state_lock:
                self._last_command = command.copy()
        return command.copy()

    def get_current_dual_hand_q(self):
        with self._state_lock:
            return self._state.copy()

    def get_last_commanded_dual_hand_q(self):
        with self._state_lock:
            return self._last_command.copy()

    def hold_current(self):
        """Command measured q when initialized, otherwise repeat the last command."""
        with self._state_lock:
            target = (
                self._state.copy()
                if self._joint_state_received.is_set()
                else self._last_command.copy()
            )
        return self.write(target[:HX5_D20_NUM_JOINTS], target[HX5_D20_NUM_JOINTS:])

    def close(self):
        with self._write_lock:
            if self._closed:
                return
            self._closed = True
            self.transport.close()
