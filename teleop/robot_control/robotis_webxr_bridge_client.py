"""Receive ROBOTIS WebXR ROS 2 samples from the Jazzy helper process."""

from __future__ import annotations

from multiprocessing.connection import Client
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np

from teleimager.image_client import TeleImage
from teleop.robot_control.robotis_image_client import AI_WORKER_CAMERA_TOPICS
from teleop.robot_control.robotis_leader_observer import (
    AIWorkerLeaderObserver,
    _initialize_observer_state,
)


class AIWorkerWebXRBridgeClient(AIWorkerLeaderObserver):
    """Expose Zenoh-backed ROS state and cameras through the Loop client APIs."""

    def __init__(
        self,
        domain_id=30,
        stale_after=0.5,
        *,
        ros_python="/usr/bin/python3",
        ros_setup="/opt/ros/jazzy/setup.bash",
        rmw_implementation="rmw_zenoh_cpp",
        startup_timeout=15.0,
    ):
        _initialize_observer_state(self, stale_after)
        self._manager = None
        self._readers = {}
        self._threads = []
        self._frames = {
            name: TeleImage(fps=0.0, jpg=None, bgr=None)
            for name in AI_WORKER_CAMERA_TOPICS
        }
        self._times = {name: [] for name in AI_WORKER_CAMERA_TOPICS}
        self._frame_timestamps = {
            name: {"source_time_ns": None, "receive_time_ns": None}
            for name in AI_WORKER_CAMERA_TOPICS
        }
        self._bridge_ready = threading.Event()
        self._bridge_info = None
        self._camera_error = None
        self._closed = False

        self._config = {
            "head_camera": self._camera_config((720, 1280)),
            "left_wrist_camera": self._camera_config((480, 640)),
            "right_wrist_camera": self._camera_config((480, 640)),
        }

        self._temp_dir = tempfile.mkdtemp(prefix="xr-tele-webxr-")
        self._socket_path = os.path.join(self._temp_dir, "ros.sock")
        authkey = secrets.token_bytes(32)
        helper = Path(__file__).resolve().parents[1] / "robotis_webxr_ros_bridge.py"
        environment = os.environ.copy()
        environment["XR_TELE_ROS_BRIDGE_AUTHKEY"] = authkey.hex()
        environment["ROS_DOMAIN_ID"] = str(int(domain_id))
        environment["RMW_IMPLEMENTATION"] = str(rmw_implementation)

        ros_python = str(Path(ros_python).expanduser())
        ros_setup = str(Path(ros_setup).expanduser()) if ros_setup else ""
        if ros_setup:
            command = [
                "/bin/bash",
                "-c",
                'source "$1" && exec "$2" "$3" --socket "$4" --domain-id "$5"',
                "xr-tele-webxr-bridge",
                ros_setup,
                ros_python,
                str(helper),
                self._socket_path,
                str(int(domain_id)),
            ]
        else:
            command = [
                ros_python,
                str(helper),
                "--socket",
                self._socket_path,
                "--domain-id",
                str(int(domain_id)),
            ]
        self._bridge_process = subprocess.Popen(
            command,
            env=environment,
            stdin=subprocess.DEVNULL,
        )

        self._connection = None
        deadline = time.monotonic() + max(0.1, float(startup_timeout))
        last_connect_error = None
        while time.monotonic() < deadline:
            if self._bridge_process.poll() is not None:
                self.close()
                raise RuntimeError(
                    "ROBOTIS WebXR ROS bridge exited before accepting a connection "
                    f"(status {self._bridge_process.returncode})"
                )
            try:
                self._connection = Client(
                    self._socket_path,
                    family="AF_UNIX",
                    authkey=authkey,
                )
                break
            except (FileNotFoundError, ConnectionRefusedError, OSError) as exc:
                last_connect_error = exc
                time.sleep(0.05)
        if self._connection is None:
            self.close()
            raise RuntimeError(
                "timed out connecting to ROBOTIS WebXR ROS bridge: "
                f"{last_connect_error}"
            )

        self._receive_thread = threading.Thread(
            target=self._receive_loop,
            name="robotis-webxr-ros-bridge-client",
            daemon=True,
        )
        self._receive_thread.start()
        if not self._bridge_ready.wait(timeout=max(0.1, float(startup_timeout))):
            error = self._last_error or "bridge did not report ready"
            self.close()
            raise RuntimeError(f"ROBOTIS WebXR ROS bridge startup failed: {error}")
        if self._bridge_info is None:
            error = self._last_error or "bridge exited before reporting its RMW"
            self.close()
            raise RuntimeError(f"ROBOTIS WebXR ROS bridge startup failed: {error}")

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
            "source": "robotis_ros2_bridge",
        }

    @property
    def bridge_info(self):
        return None if self._bridge_info is None else dict(self._bridge_info)

    @property
    def last_error(self):
        return self._camera_error

    def _receive_loop(self):
        while not self._stop.is_set() and self._connection is not None:
            try:
                event = self._connection.recv()
            except (EOFError, OSError) as exc:
                if not self._stop.is_set():
                    self._last_error = f"ROS bridge connection: {exc}"
                    self._bridge_ready.set()
                break
            self._dispatch_event(event)

    def _dispatch_event(self, event):
        kind = event.get("kind")
        if kind == "bridge_ready":
            self._bridge_info = dict(event)
            self._bridge_ready.set()
            return
        if kind == "bridge_error":
            self._last_error = f"ROS bridge: {event.get('message', 'unknown error')}"
            self._bridge_ready.set()
            return
        if kind == "trajectory":
            self._process(
                event["source"],
                SimpleNamespace(
                    joint_names=event.get("joint_names", ()),
                    points=[SimpleNamespace(positions=event.get("positions", ()))],
                ),
            )
            return
        if kind == "joint_state":
            self._process(
                "follower",
                SimpleNamespace(
                    name=event.get("name", ()),
                    position=event.get("position", ()),
                    velocity=event.get("velocity", ()),
                ),
            )
            return
        if kind == "twist":
            self._process(
                "leader_mobile",
                self._twist_message(event),
            )
            return
        if kind == "odometry":
            self._process(
                "follower_mobile",
                SimpleNamespace(twist=SimpleNamespace(twist=self._twist_message(event))),
            )
            return
        if kind == "image":
            self._process_image(event)

    @staticmethod
    def _twist_message(event):
        linear = list(event.get("linear", (0.0, 0.0)))
        while len(linear) < 2:
            linear.append(0.0)
        return SimpleNamespace(
            linear=SimpleNamespace(x=linear[0], y=linear[1]),
            angular=SimpleNamespace(z=event.get("angular_z", 0.0)),
        )

    def _process_image(self, event):
        name = event.get("name")
        if name not in self._frames:
            return
        try:
            jpg = bytes(event.get("data", b""))
            bgr = cv2.imdecode(
                np.frombuffer(jpg, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if bgr is None:
                raise ValueError("compressed image could not be decoded")
            now = time.monotonic()
            stamps = self._times[name]
            stamps.append(now)
            if len(stamps) > 20:
                del stamps[:-20]
            fps = 0.0
            if len(stamps) > 1 and stamps[-1] > stamps[0]:
                fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
            with self._lock:
                self._frames[name] = TeleImage(fps=fps, jpg=jpg, bgr=bgr)
                self._frame_timestamps[name] = {
                    "source_time_ns": event.get("source_time_ns"),
                    "receive_time_ns": time.time_ns(),
                }
        except Exception as exc:
            self._camera_error = f"{name}: {exc}"

    def get_cam_config(self):
        return self._config

    def _get_frame(self, name):
        with self._lock:
            return self._frames[name]

    def get_head_frame(self):
        return self._get_frame("head")

    def get_left_wrist_frame(self):
        return self._get_frame("left_wrist")

    def get_right_wrist_frame(self):
        return self._get_frame("right_wrist")

    def get_frame_timestamps(self):
        with self._lock:
            return {name: dict(value) for name, value in self._frame_timestamps.items()}

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        connection = getattr(self, "_connection", None)
        if connection is not None:
            try:
                connection.send("close")
            except Exception:
                pass
            try:
                connection.close()
            except Exception:
                pass
        receive_thread = getattr(self, "_receive_thread", None)
        if receive_thread is not None:
            receive_thread.join(timeout=1.0)

        process = getattr(self, "_bridge_process", None)
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
        temp_dir = getattr(self, "_temp_dir", None)
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
