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
        self._frame_timestamps = {
            name: {"source_time_ns": None, "receive_time_ns": None}
            for name in self._readers
        }
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

    @staticmethod
    def _source_time_ns(message):
        stamp = getattr(getattr(message, "header", None), "stamp", None)
        sec = getattr(stamp, "sec", None)
        nanosec = getattr(stamp, "nanosec", None)
        if sec is None or nanosec is None:
            return None
        return int(sec) * 1_000_000_000 + int(nanosec)

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
                receive_time_ns = time.time_ns()
                source_time_ns = self._source_time_ns(message)
                stamps = self._times[name]
                stamps.append(now)
                fps = 0.0
                if len(stamps) > 1 and stamps[-1] > stamps[0]:
                    fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
                with self._lock:
                    self._frames[name] = TeleImage(fps=fps, jpg=jpg, bgr=bgr)
                    self._frame_timestamps[name] = {
                        "source_time_ns": source_time_ns,
                        "receive_time_ns": receive_time_ns,
                    }
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

    def get_frame_timestamps(self):
        """Return source and host-receive times for each latest DDS camera frame."""
        with self._lock:
            return {name: dict(values) for name, values in self._frame_timestamps.items()}

    def close(self):
        self._running = False
        for thread in self._threads:
            thread.join(timeout=0.5)
        for reader in self._readers.values():
            try:
                reader.close()
            except Exception:
                pass


class RobotisROSImageClient:
    """Expose AI Worker camera topics through an existing rclpy node.

    This variant follows the node's selected RMW, including ROBOTIS' default
    ``rmw_zenoh_cpp`` Meta Quest deployment.  The owner of ``node`` is
    responsible for spinning it and shutting it down.
    """

    def __init__(
        self,
        node,
        *,
        compressed_image_type=None,
        qos_profile=None,
    ):
        if compressed_image_type is None:
            from sensor_msgs.msg import CompressedImage

            compressed_image_type = CompressedImage
        if qos_profile is None:
            from rclpy.qos import (
                DurabilityPolicy,
                HistoryPolicy,
                QoSProfile,
                ReliabilityPolicy,
            )

            qos_profile = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )

        self._node = node
        self._lock = threading.Lock()
        self._frames = {
            name: TeleImage(fps=0.0, jpg=None, bgr=None)
            for name in AI_WORKER_CAMERA_TOPICS
        }
        self._times = {name: deque(maxlen=20) for name in AI_WORKER_CAMERA_TOPICS}
        self._frame_timestamps = {
            name: {"source_time_ns": None, "receive_time_ns": None}
            for name in AI_WORKER_CAMERA_TOPICS
        }
        self._last_error = None
        self._subscriptions = []
        for name, topic in AI_WORKER_CAMERA_TOPICS.items():
            callback = lambda message, name=name: self._on_image(name, message)
            self._subscriptions.append(
                self._node.create_subscription(
                    compressed_image_type,
                    topic,
                    callback,
                    qos_profile,
                )
            )

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
            "source": "robotis_ros2",
        }

    def _on_image(self, name, message):
        try:
            jpg = bytes(message.data)
            bgr = cv2.imdecode(
                np.frombuffer(jpg, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if bgr is None:
                raise ValueError("compressed image could not be decoded")
            now = time.monotonic()
            receive_time_ns = time.time_ns()
            source_time_ns = RobotisDDSImageClient._source_time_ns(message)
            stamps = self._times[name]
            stamps.append(now)
            fps = 0.0
            if len(stamps) > 1 and stamps[-1] > stamps[0]:
                fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
            with self._lock:
                self._frames[name] = TeleImage(fps=fps, jpg=jpg, bgr=bgr)
                self._frame_timestamps[name] = {
                    "source_time_ns": source_time_ns,
                    "receive_time_ns": receive_time_ns,
                }
        except Exception as exc:
            self._last_error = f"{name}: {exc}"

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

    def get_frame_timestamps(self):
        with self._lock:
            return {name: dict(values) for name, values in self._frame_timestamps.items()}

    @property
    def last_error(self):
        return self._last_error

    def close(self):
        for subscription in self._subscriptions:
            try:
                self._node.destroy_subscription(subscription)
            except Exception:
                pass
        self._subscriptions.clear()
