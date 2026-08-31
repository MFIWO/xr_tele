import math
import socket
import time

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


class AIWorkerNeckController:
    """Drive the AI Worker pan/tilt head from Vision Pro orientation over DDS."""

    # Limits from ffw_sg2_follower.urdf. head_joint1 rotates about Y (pitch),
    # and head_joint2 rotates about Z (yaw).
    PITCH_LOWER = -0.2317
    PITCH_UPPER = 0.6951
    YAW_LOWER = -0.35
    YAW_UPPER = 0.35

    def __init__(
        self,
        yaw_limit=0.35,
        pitch_limit=0.6951,
        smoothing_alpha=0.25,
        max_step=0.08,
        command_duration=0.08,
        pitch_gain=1.0,
        pitch_invert=True,
    ):
        from teleop.robot_control.robotis_dds import RobotisJointTrajectoryTransport

        self.yaw_limit = min(abs(float(yaw_limit)), self.YAW_UPPER)
        self.pitch_limit = min(abs(float(pitch_limit)), self.PITCH_UPPER)
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.max_step = abs(float(max_step))
        self.command_duration = max(0.02, float(command_duration))
        self.pitch_gain = max(0.0, float(pitch_gain))
        self.pitch_sign = -1.0 if pitch_invert else 1.0
        self.neutral = None
        self.command = np.zeros(2, dtype=np.float64)  # yaw, pitch
        self.transport = RobotisJointTrajectoryTransport(
            {"head": "/leader/joystick_controller_left/joint_trajectory"}
        )

    _extract_yaw_pitch = staticmethod(VisionProNeckController._extract_yaw_pitch)

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
        target[1] *= self.pitch_gain * self.pitch_sign

        target[0] = np.clip(target[0], -self.yaw_limit, self.yaw_limit)
        target[1] = np.clip(target[1], -self.pitch_limit, self.pitch_limit)
        filtered = self.command + self.smoothing_alpha * (target - self.command)
        if self.max_step > 0.0:
            filtered = self.command + np.clip(
                filtered - self.command, -self.max_step, self.max_step
            )
        filtered[0] = np.clip(filtered[0], self.YAW_LOWER, self.YAW_UPPER)
        filtered[1] = np.clip(filtered[1], self.PITCH_LOWER, self.PITCH_UPPER)
        self.command = filtered

        self.transport.publish(
            "head",
            ("head_joint1", "head_joint2"),
            (self.command[1], self.command[0]),  # pitch, yaw
            self.command_duration,
        )
        return self.command.copy(), target.copy()

    def close(self):
        try:
            self.transport.publish(
                "head",
                ("head_joint1", "head_joint2"),
                (0.0, 0.0),
                max(0.5, self.command_duration),
            )
            time.sleep(0.05)
        except Exception:
            pass
        self.transport.close()
