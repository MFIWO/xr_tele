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
    ):
        self.address = (host, int(port))
        self.yaw_limit = abs(float(yaw_limit))
        self.pitch_limit = abs(float(pitch_limit))
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.max_step = abs(float(max_step))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.neutral = None
        self.command = np.zeros(2, dtype=np.float64)

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

    def update(self, head_pose):
        measured = self._extract_yaw_pitch(head_pose)
        if self.neutral is None:
            self.neutral = measured
            target = np.zeros(2, dtype=np.float64)
        else:
            target = measured - self.neutral
            target[0] = _wrap_angle(float(target[0]))

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

        payload = f"{self.command[0]:.5f},{self.command[1]:.5f}".encode("ascii")
        self.socket.sendto(payload, self.address)
        return self.command.copy(), target.copy()

    def close(self):
        self.socket.close()
