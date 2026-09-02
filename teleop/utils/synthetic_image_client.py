"""Deterministic camera source for DDS-free teleoperation diagnostics."""

from dataclasses import dataclass, field
import time

import numpy as np


@dataclass(frozen=True)
class SyntheticImage:
    """Small subset of ``teleimager.TeleImage`` consumed by teleoperation."""

    fps: float
    jpg: bytes | None
    bgr: np.ndarray | None = field(repr=False)


class SyntheticImageClient:
    """Generate three colored BGR streams without opening network or DDS resources."""

    def __init__(self, fps=30.0):
        self.fps = max(1.0, float(fps))
        self._started = time.monotonic()
        self._closed = False
        self._config = {
            "head_camera": self._camera_config((240, 424)),
            "left_wrist_camera": self._camera_config((180, 240)),
            "right_wrist_camera": self._camera_config((180, 240)),
        }

    def _camera_config(self, shape):
        return {
            "enable_zmq": True,
            "enable_webrtc": False,
            "zmq_port": None,
            "webrtc_port": None,
            "image_shape": list(shape),
            "binocular": False,
            "fps": self.fps,
            "source": "synthetic_no_hardware",
        }

    def get_cam_config(self):
        return self._config

    def _frame(self, name, shape, color):
        if self._closed:
            return SyntheticImage(fps=0.0, jpg=None, bgr=None)
        height, width = shape
        frame = np.empty((height, width, 3), dtype=np.uint8)
        frame[:] = color

        # Moving bars make a frozen or swapped stream obvious without requiring
        # fonts or OpenCV in the deterministic test environment.
        phase = int((time.monotonic() - self._started) * 80.0) % width
        frame[:, max(0, phase - 3) : min(width, phase + 4)] = (255, 255, 255)
        side_phase = int((time.monotonic() - self._started) * 50.0) % height
        frame[max(0, side_phase - 2) : min(height, side_phase + 3), :] //= 2
        frame[0:8, 0:8] = {
            "head": (0, 255, 255),
            "left_wrist": (255, 255, 0),
            "right_wrist": (255, 0, 255),
        }[name]
        return SyntheticImage(fps=self.fps, jpg=None, bgr=frame)

    def get_head_frame(self):
        return self._frame("head", (240, 424), (38, 70, 120))

    def get_left_wrist_frame(self):
        return self._frame("left_wrist", (180, 240), (90, 55, 25))

    def get_right_wrist_frame(self):
        return self._frame("right_wrist", (180, 240), (25, 55, 90))

    def get_frame_timestamps(self):
        now = time.time_ns()
        return {
            name: {"source_time_ns": now, "receive_time_ns": now}
            for name in ("head", "left_wrist", "right_wrist")
        }

    def close(self):
        self._closed = True
