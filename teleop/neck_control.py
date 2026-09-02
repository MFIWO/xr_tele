import math
import socket

import numpy as np


def _wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class VisionProNeckController:
    """Convert Vision Pro head orientation into bounded UDP pan/tilt commands."""

    def __init__(
        self,
        host,
        port=9091,
        yaw_limit=math.pi / 2.0,
        pitch_limit=math.pi / 2.0,
        smoothing_alpha=0.25,
        max_step=0.08,
        command_deadband=0.0,
    ):
        self.address = (host, int(port))
        self.yaw_limit = abs(float(yaw_limit))
        self.pitch_limit = abs(float(pitch_limit))
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.max_step = abs(float(max_step))
        self.command_deadband = abs(float(command_deadband))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.neutral = None
        self.command = np.zeros(2, dtype=np.float64)
        self.last_sent_command = None

    @staticmethod
    def _extract_yaw_pitch(head_pose):
        pose = np.asarray(head_pose, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("head_pose must be a finite 4x4 matrix")

        forward = pose[:3, 0]
        horizontal = math.hypot(float(forward[0]), float(forward[1]))
        if horizontal < 1e-6 and abs(float(forward[2])) < 1e-6:
            raise ValueError("head_pose has an invalid forward axis")

        yaw = math.atan2(float(forward[1]), float(forward[0]))
        pitch = math.atan2(float(forward[2]), horizontal)
        return np.array([yaw, pitch], dtype=np.float64)

    def reset_neutral(self):
        self.neutral = None
        self.command[:] = 0.0
        self.last_sent_command = None

    def update(self, head_pose):
        measured = self._extract_yaw_pitch(head_pose)
        if self.neutral is None:
            self.neutral = measured
            target = np.zeros(2, dtype=np.float64)
        else:
            target = measured - self.neutral
            target[0] = _wrap_angle(float(target[0]))
        return self._send_target(target)

    def command_absolute(self, yaw_pitch):
        """Send an already-neutral-relative yaw/pitch target (policy inference path).

        A trained policy predicts the same quantity this class sends over UDP, so the
        neutral offset must not be applied again — only the clamp and rate limit.
        """
        target = np.asarray(yaw_pitch, dtype=np.float64).reshape(-1)
        if target.size != 2 or not np.isfinite(target).all():
            raise ValueError("yaw_pitch must be two finite values")
        return self._send_target(target.copy())

    def _send_target(self, target):
        target[0] = np.clip(target[0], -self.yaw_limit, self.yaw_limit)
        target[1] = np.clip(target[1], -self.pitch_limit, self.pitch_limit)

        filtered = self.command + self.smoothing_alpha * (target - self.command)
        if self.max_step > 0.0:
            filtered = self.command + np.clip(
                filtered - self.command,
                -self.max_step,
                self.max_step,
            )
        self.command = filtered

        should_send = (
            self.last_sent_command is None
            or self.command_deadband <= 0.0
            or np.max(np.abs(self.command - self.last_sent_command)) >= self.command_deadband
        )
        if should_send:
            payload = f"{self.command[0]:.5f},{self.command[1]:.5f}".encode("ascii")
            self.socket.sendto(payload, self.address)
            self.last_sent_command = self.command.copy()

        effective_command = self.command if self.last_sent_command is None else self.last_sent_command
        return effective_command.copy(), target.copy()

    def close(self):
        self.socket.close()
