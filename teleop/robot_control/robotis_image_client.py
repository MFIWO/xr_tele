"""AI Worker camera client backed by ROS 2 CompressedImage over CycloneDDS."""

import threading
import time
from collections import deque

import cv2
import numpy as np

from teleimager.image_client import TeleImage


AI_WORKER_CAMERA_TOPICS = {
    "head": "/zed/zed_node/left/image_rect_color/compressed",
    "left_wrist": "/camera_left/camera_left/color/image_rect_raw/compressed",
    "right_wrist": "/camera_right/camera_right/color/image_rect_raw/compressed",
}


class RobotisDDSImageClient:
    """Expose AI Worker ROS camera topics through the TeleImager client API."""

    def __init__(self, domain_id=None):
        from cyclonedds.core import Qos, Policy
        from robotis_dds_python.idl.sensor_msgs.msg import CompressedImage_
        from robotis_dds_python.tools.topic_manager import TopicManager

        self._manager = TopicManager(domain_id=domain_id)
        # image_transport's compressed publishers normally use sensor-data QoS
        # (best effort). A reliable reader does not match a best-effort writer.
        qos = Qos(
            Policy.Reliability.BestEffort,
            Policy.Durability.Volatile,
            Policy.History.KeepLast(1),
        )
        self._readers = {
            name: self._manager.topic_reader(topic_name=topic, topic_type=CompressedImage_, qos=qos)
            for name, topic in AI_WORKER_CAMERA_TOPICS.items()
        }
        self._frames = {name: TeleImage(fps=0.0, jpg=None, bgr=None) for name in self._readers}
        self._times = {name: deque(maxlen=20) for name in self._readers}
        self._lock = threading.Lock()
        self._running = True
        self._threads = [
            threading.Thread(target=self._reader_loop, args=(name, reader), daemon=True)
            for name, reader in self._readers.items()
        ]
        for thread in self._threads:
            thread.start()

        # Shapes are only initial buffer hints. teleop_hand_and_arm replaces
        # them with the first decoded frame's actual dimensions.
        self._config = {
            "head_camera": self._camera_config((720, 1280)),
            "left_wrist_camera": self._camera_config((480, 640)),
            "right_wrist_camera": self._camera_config((480, 640)),
        }

    @staticmethod
    def _camera_config(shape):
        return {
            "enable_zmq": True,
            "enable_webrtc": False,
            "zmq_port": None,
            "webrtc_port": None,
            "image_shape": list(shape),
            "binocular": False,
            "fps": 30,
            "source": "robotis_dds",
        }

    def _reader_loop(self, name, reader):
        while self._running:
            try:
                samples = reader.take(N=1)
                if not samples:
                    time.sleep(0.002)
                    continue
                message = samples[-1]
                jpg = bytes(message.data)
                bgr = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                now = time.monotonic()
                stamps = self._times[name]
                stamps.append(now)
                fps = 0.0
                if len(stamps) > 1 and stamps[-1] > stamps[0]:
                    fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
                with self._lock:
                    self._frames[name] = TeleImage(fps=fps, jpg=jpg, bgr=bgr)
            except Exception:
                if self._running:
                    time.sleep(0.02)

    def get_cam_config(self):
        return self._config

    def _get(self, name):
        with self._lock:
            return self._frames[name]

    def get_head_frame(self):
        return self._get("head")

    def get_left_wrist_frame(self):
        return self._get("left_wrist")

    def get_right_wrist_frame(self):
        return self._get("right_wrist")

    def close(self):
        self._running = False
        for thread in self._threads:
            thread.join(timeout=0.5)
        for reader in self._readers.values():
            try:
                reader.close()
            except Exception:
                pass

