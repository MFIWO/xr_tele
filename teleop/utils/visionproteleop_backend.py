"""Native VisionProTeleop input/output adapter for the existing teleop ABI.

The stock ``avp_stream`` package reports head and wrist poses after applying its
Y-up-to-Z-up basis transform.  Finger matrices are wrist-local and are not
changed by that basis transform.  This adapter deliberately reproduces the
coordinate conventions used by ``televuer.tv_wrapper.TeleVuerWrapper``:

* head and wrists use the robot basis (x forward, y left, z up),
* wrist poses use the existing Unitree left/right wrist frame convention,
* hand positions contain the first 25 joints in the existing Unitree hand
  convention, whether VisionProTeleop supplies a 25- or 27-joint skeleton.

Freshness is measured locally with ``time.monotonic()``.  ``avp_stream`` does
not expose an ARKit ``isTracked`` flag or a tracking sequence/timestamp, so a
new network sample is detected by the identity of ``TrackingData.raw``.  That
matches the stock streamer, which replaces its raw dictionary for each hand
update.  Numeric/rigid checks and packet freshness are safety signals; they are
not a claim that native ARKit tracking status is observable.
"""

from __future__ import annotations

import copy
import importlib
import math
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

try:
    import cv2
except ModuleNotFoundError:  # Keep command-sink/unit tests dependency-light.
    cv2 = None


TRACKING_OK = "TRACKING_OK"
TRACKING_STALE = "TRACKING_STALE"
REANCHOR_REQUIRED = "REANCHOR_REQUIRED"
HOLD = "HOLD"


@dataclass
class TeleData:
    """Dependency-free attribute ABI shared with televuer's TeleData."""

    head_pose: np.ndarray
    left_wrist_pose: np.ndarray
    right_wrist_pose: np.ndarray
    left_hand_pos: Optional[np.ndarray] = None
    right_hand_pos: Optional[np.ndarray] = None
    left_hand_rot: Optional[np.ndarray] = None
    right_hand_rot: Optional[np.ndarray] = None
    left_hand_pinch: bool = False
    left_hand_pinchValue: float = 10.0
    left_hand_squeeze: bool = False
    left_hand_squeezeValue: float = 0.0
    right_hand_pinch: bool = False
    right_hand_pinchValue: float = 10.0
    right_hand_squeeze: bool = False
    right_hand_squeezeValue: float = 0.0
    left_ctrl_trigger: bool = False
    left_ctrl_triggerValue: float = 10.0
    left_ctrl_squeeze: bool = False
    left_ctrl_squeezeValue: float = 0.0
    left_ctrl_aButton: bool = False
    left_ctrl_bButton: bool = False
    left_ctrl_thumbstick: bool = False
    left_ctrl_thumbstickValue: np.ndarray = field(
        default_factory=lambda: np.zeros(2)
    )
    right_ctrl_trigger: bool = False
    right_ctrl_triggerValue: float = 10.0
    right_ctrl_squeeze: bool = False
    right_ctrl_squeezeValue: float = 0.0
    right_ctrl_aButton: bool = False
    right_ctrl_bButton: bool = False
    right_ctrl_thumbstick: bool = False
    right_ctrl_thumbstickValue: np.ndarray = field(
        default_factory=lambda: np.zeros(2)
    )
    tracking_active: bool = False
    session_alive: bool = False
    head_pose_is_valid: bool = False
    left_arm_is_valid: bool = False
    right_arm_is_valid: bool = False
    tracking_state: str = TRACKING_STALE
    tracking_reason: str = "waiting for first tracking packet"
    tracking_sample_age_s: float = math.inf
    native_tracking_status_available: bool = False


# Existing Vuer/OpenXR convention: x right, y up, z back -> robot convention:
# x forward, y left, z up.
T_ROBOT_OPENXR = np.array(
    [
        [0.0, 0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
T_OPENXR_ROBOT = np.linalg.inv(T_ROBOT_OPENXR)

# Applied internally by stock avp_stream to head/wrist world poses.
T_AVP_Y_UP_TO_Z_UP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

T_TO_UNITREE_HUMANOID_LEFT_ARM = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
T_TO_UNITREE_HUMANOID_RIGHT_ARM = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
T_TO_UNITREE_HAND = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _rotation_x(angle_radians: float) -> np.ndarray:
    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cosine, -sine, 0.0],
            [0.0, sine, cosine, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _rotation_y(angle_radians: float) -> np.ndarray:
    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)
    return np.array(
        [
            [cosine, 0.0, sine, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sine, 0.0, cosine, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _rotation_z(angle_radians: float) -> np.ndarray:
    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)
    return np.array(
        [
            [cosine, -sine, 0.0, 0.0],
            [sine, cosine, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


# avp_stream post-multiplies head pose by Rx(-90 degrees).
T_AVP_HEAD_CORRECTION_INV = _rotation_x(math.pi / 2.0)
T_AVP_TO_ROBOT_WORLD = T_ROBOT_OPENXR @ np.linalg.inv(T_AVP_Y_UP_TO_Z_UP)


def _load_streamer_class():
    """Load the optional native backend only when it is selected."""
    try:
        module = importlib.import_module("avp_stream")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "VisionProTeleop backend requires the optional 'avp_stream' package. "
            "Install the repository's pinned VisionProTeleop requirement before "
            "using --xr-backend visionproteleop."
        ) from exc

    try:
        return module.VisionProStreamer
    except AttributeError as exc:
        raise RuntimeError(
            "Installed avp_stream does not export VisionProStreamer; use the "
            "version pinned by this repository."
        ) from exc


def _squeeze_pose(value: Any, name: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape == (1, 4, 4):
        pose = pose[0]
    if pose.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4) or (1, 4, 4), got {pose.shape}")
    return pose


def _validate_rigid_transform(transform: np.ndarray, name: str, atol: float) -> None:
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(transform[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=atol):
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=atol):
        raise ValueError(f"{name} rotation determinant is not +1")


def _extract_joints(packet: Mapping[str, Any], side: str) -> np.ndarray:
    joints = packet.get(f"{side}_fingers")
    if joints is None:
        joints = packet.get(f"{side}_arm")
    if joints is None:
        raise ValueError(f"missing {side} hand joints")

    joints = np.asarray(joints, dtype=np.float64)
    if joints.shape not in ((25, 4, 4), (27, 4, 4)):
        raise ValueError(
            f"{side} hand joints must have shape (25, 4, 4) or (27, 4, 4), "
            f"got {joints.shape}"
        )
    return joints[:25]


def _rotation_distance(left: np.ndarray, right: np.ndarray) -> float:
    relative = left[:3, :3].T @ right[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.acos(cosine)


class SyntheticVisionProStreamer:
    """Deterministic, dependency-free source with the stock streamer surface.

    Every :meth:`get_latest` call creates a distinct raw dictionary and advances
    one fixed-rate sample.  The motion exercises head yaw, bilateral wrist XYZ,
    wrist orientation, finger curl, and a periodic thumb/index pinch.  It never
    imports ``avp_stream`` and performs no network, DDS, or motor I/O.
    """

    def __init__(
        self,
        ip: str = "synthetic",
        record: bool = False,
        ht_backend: str = "grpc",
        origin: str = "avp",
        verbose: bool = False,
        *,
        fps: float = 60.0,
        joint_count: int = 27,
        **_unused: Any,
    ) -> None:
        if not math.isfinite(float(fps)) or fps <= 0.0:
            raise ValueError("synthetic fps must be finite and positive")
        if joint_count not in (25, 27):
            raise ValueError("synthetic joint_count must be 25 or 27")
        self.ip = ip
        self.record = False
        self.ht_backend = ht_backend
        self.origin = origin
        self.verbose = verbose
        self.fps = float(fps)
        self.joint_count = int(joint_count)
        self.sample_index = 0
        self.closed = False
        self.video_config: Optional[dict[str, Any]] = None
        self.frame_callback: Optional[Callable[[np.ndarray], np.ndarray]] = None
        self.user_frame: Optional[np.ndarray] = None
        self.webrtc_started = False
        self.updated_frame_count = 0
        self.video_callback_count = 0

    def get_latest(self) -> Optional[dict[str, Any]]:
        if self.closed:
            return None
        sample_time = self.sample_index / self.fps
        self.sample_index += 1

        head_openxr = _rotation_y(0.08 * math.sin(0.55 * sample_time))
        head_openxr[:3, 3] = np.array(
            [
                0.015 * math.sin(0.31 * sample_time),
                1.60 + 0.01 * math.sin(0.47 * sample_time),
                -0.20 + 0.012 * math.cos(0.37 * sample_time),
            ]
        )
        left_wrist_openxr = (
            _rotation_z(-0.12 + 0.08 * math.sin(0.61 * sample_time))
            @ _rotation_x(0.06 * math.cos(0.43 * sample_time))
        )
        right_wrist_openxr = (
            _rotation_z(0.12 - 0.08 * math.sin(0.59 * sample_time))
            @ _rotation_x(-0.06 * math.cos(0.41 * sample_time))
        )
        left_wrist_openxr[:3, 3] = np.array(
            [
                -0.27 + 0.035 * math.sin(0.73 * sample_time),
                1.25 + 0.025 * math.sin(0.89 * sample_time),
                -0.44 + 0.030 * math.cos(0.67 * sample_time),
            ]
        )
        right_wrist_openxr[:3, 3] = np.array(
            [
                0.27 - 0.035 * math.sin(0.71 * sample_time),
                1.25 + 0.025 * math.cos(0.83 * sample_time),
                -0.44 + 0.030 * math.sin(0.69 * sample_time),
            ]
        )

        curl = 0.5 + 0.5 * math.sin(0.8 * sample_time)
        pinch = 0.5 + 0.5 * math.sin(0.93 * sample_time + 0.4)
        left_arm = self._hand_skeleton(-1.0, curl, pinch)
        right_arm = self._hand_skeleton(1.0, 1.0 - curl, pinch)

        packet = {
            "head": (
                T_AVP_Y_UP_TO_Z_UP
                @ head_openxr
                @ _rotation_x(-math.pi / 2.0)
            )[None],
            "left_wrist": (T_AVP_Y_UP_TO_Z_UP @ left_wrist_openxr)[None],
            "right_wrist": (T_AVP_Y_UP_TO_Z_UP @ right_wrist_openxr)[None],
            "left_fingers": left_arm[:25].copy(),
            "right_fingers": right_arm[:25].copy(),
            "left_arm": left_arm,
            "right_arm": right_arm,
            "left_pinch_distance": float(
                np.linalg.norm(left_arm[4, :3, 3] - left_arm[9, :3, 3])
            ),
            "right_pinch_distance": float(
                np.linalg.norm(right_arm[4, :3, 3] - right_arm[9, :3, 3])
            ),
        }
        return packet

    def configure_video(self, **kwargs: Any) -> None:
        self.video_config = dict(kwargs)

    def register_frame_callback(
        self, callback: Callable[[np.ndarray], np.ndarray]
    ) -> None:
        self.frame_callback = callback

    def start_webrtc(self, **_kwargs: Any) -> None:
        self.webrtc_started = True

    def update_frame(self, frame: np.ndarray) -> None:
        self.user_frame = frame
        self.updated_frame_count += 1

    def pull_video_frame(self) -> np.ndarray:
        """Invoke the registered callback once for an offline diagnostic."""
        if self.video_config is None:
            width, height = 640, 480
        else:
            width, height = (
                int(value) for value in self.video_config["size"].split("x", 1)
            )
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        self.video_callback_count += 1
        if self.frame_callback is None:
            return blank
        return self.frame_callback(blank)

    def cleanup(self) -> None:
        self.closed = True

    def is_connected(self) -> bool:
        return self.webrtc_started and not self.closed

    def _hand_skeleton(
        self, mirror: float, curl: float, pinch: float
    ) -> np.ndarray:
        joints = np.repeat(np.eye(4)[None, :, :], self.joint_count, axis=0)
        # wrist(0), thumb(1-4), index(5-9), middle(10-14), ring(15-19),
        # little(20-24), then optional forearmWrist/forearmArm(25-26).
        thumb_positions = (
            (0.020, -0.026, 0.000),
            (0.038, -0.039, 0.004),
            (0.057, -0.046, 0.008),
            (0.074, -0.050, 0.012),
        )
        for index, values in enumerate(thumb_positions, start=1):
            joints[index, :3, 3] = np.array(
                [values[0], mirror * values[1], values[2]]
            )

        finger_specs = (
            (5, -0.032, 0.105),
            (10, -0.010, 0.112),
            (15, 0.014, 0.103),
            (20, 0.034, 0.090),
        )
        progress = np.linspace(0.15, 1.0, 5)
        for start, lateral, length in finger_specs:
            for offset, fraction in enumerate(progress):
                bend = curl * fraction
                joints[start + offset, :3, 3] = np.array(
                    [
                        length * fraction * (1.0 - 0.38 * bend),
                        mirror * lateral,
                        0.055 * bend * fraction,
                    ]
                )

        # Blend thumb tip toward index tip so retargeting sees a real periodic
        # pinch rather than a separate synthetic boolean.
        open_thumb_tip = joints[4, :3, 3].copy()
        pinch_target = joints[9, :3, 3] + np.array([0.0, mirror * 0.004, 0.0])
        joints[4, :3, 3] = (1.0 - pinch) * open_thumb_tip + pinch * pinch_target
        if self.joint_count == 27:
            joints[25, :3, 3] = np.array([-0.035, 0.0, 0.0])
            joints[26, :3, 3] = np.array([-0.180, 0.0, 0.0])
        return joints


class VisionProTeleopBackend:
    """Adapt stock VisionProTeleop tracking/video to the current TeleData ABI.

    A fresh, valid sample starts in :data:`REANCHOR_REQUIRED`; callers must call
    :meth:`request_enable` explicitly.  A timeout, reconnect, invalid transform,
    or rejected wrist motion disables every validity flag.  A reconnect or hold
    never resumes merely because packets start arriving again.

    Parameters named ``streamer``, ``streamer_factory``, and ``clock`` exist for
    deterministic no-hardware tests.  Production code normally supplies only
    ``ip`` and video/safety settings.
    """

    def __init__(
        self,
        ip: str,
        *,
        use_hand_tracking: bool = True,
        binocular: bool = False,
        img_shape: tuple[int, int] = (480, 1280),
        display_fps: float = 30.0,
        return_hand_rot_data: bool = False,
        tracking_timeout_s: float = 0.25,
        settling_time_s: float = 0.25,
        rigid_atol: float = 2e-2,
        max_wrist_translation_jump_m: float = 0.35,
        max_wrist_rotation_jump_rad: float = math.radians(100.0),
        max_wrist_velocity_m_s: float = 6.0,
        max_wrist_angular_velocity_rad_s: float = 12.0,
        motion_rejection_enabled: bool = True,
        pinch_threshold_m: float = 0.025,
        webrtc_port: int = 9999,
        start_video: bool = True,
        tracking_transport: str = "grpc",
        verbose: bool = False,
        streamer: Any = None,
        streamer_factory: Optional[Callable[..., Any]] = None,
        clock: Callable[[], float] = time.monotonic,
        status_callback: Optional[Callable[[str], None]] = print,
    ) -> None:
        if not ip:
            raise ValueError("Vision Pro IP or room code is required")
        if not use_hand_tracking:
            raise ValueError("VisionProTeleop backend supports hand tracking only")
        if binocular:
            raise ValueError("VisionProTeleop backend currently supports mono video only")
        if len(img_shape) != 2 or min(int(img_shape[0]), int(img_shape[1])) <= 0:
            raise ValueError("img_shape must be a positive (height, width) tuple")
        if not math.isfinite(float(display_fps)) or display_fps <= 0.0:
            raise ValueError("display_fps must be finite and positive")
        if not math.isfinite(float(tracking_timeout_s)) or tracking_timeout_s <= 0.0:
            raise ValueError("tracking_timeout_s must be positive")
        if not math.isfinite(float(settling_time_s)) or settling_time_s < 0.0:
            raise ValueError("settling_time_s must be non-negative")
        if not math.isfinite(float(rigid_atol)) or rigid_atol <= 0.0:
            raise ValueError("rigid_atol must be finite and positive")
        if not math.isfinite(float(pinch_threshold_m)) or pinch_threshold_m < 0.0:
            raise ValueError("pinch_threshold_m must be finite and non-negative")
        if not 1 <= int(webrtc_port) <= 65535:
            raise ValueError("webrtc_port must be between 1 and 65535")
        for name, value in (
            ("max_wrist_translation_jump_m", max_wrist_translation_jump_m),
            ("max_wrist_rotation_jump_rad", max_wrist_rotation_jump_rad),
            ("max_wrist_velocity_m_s", max_wrist_velocity_m_s),
            ("max_wrist_angular_velocity_rad_s", max_wrist_angular_velocity_rad_s),
        ):
            if not math.isfinite(float(value)) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

        self.ip = ip
        self.img_shape = (int(img_shape[0]), int(img_shape[1]))
        self.display_fps = float(display_fps)
        self.return_hand_rot_data = bool(return_hand_rot_data)
        self.tracking_timeout_s = float(tracking_timeout_s)
        self.settling_time_s = float(settling_time_s)
        self.rigid_atol = float(rigid_atol)
        self.max_wrist_translation_jump_m = float(max_wrist_translation_jump_m)
        self.max_wrist_rotation_jump_rad = float(max_wrist_rotation_jump_rad)
        self.max_wrist_velocity_m_s = float(max_wrist_velocity_m_s)
        self.max_wrist_angular_velocity_rad_s = float(max_wrist_angular_velocity_rad_s)
        self.motion_rejection_enabled = bool(motion_rejection_enabled)
        self.pinch_threshold_m = float(pinch_threshold_m)
        self._clock = clock
        self._status_callback = status_callback

        self.packet_count = 0
        self.invalid_packet_count = 0
        self.rejected_jump_count = 0
        self.stale_transition_count = 0
        self.reconnect_count = 0
        self.hold_count = 0
        self.video_input_count = 0
        self.video_callback_count = 0
        self._packet_times: deque[float] = deque(maxlen=512)
        self._video_input_times: deque[float] = deque(maxlen=512)
        self._video_callback_times: deque[float] = deque(maxlen=512)

        self.state = TRACKING_STALE
        self.state_reason = "waiting for first tracking packet"
        if self._status_callback is not None:
            self._status_callback(
                f"[VisionProTeleop] {self.state}: {self.state_reason}"
            )
        self._closed = False
        self._last_raw_object: Any = None
        self._last_packet_monotonic: Optional[float] = None
        self._last_observed_wrist_poses: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._last_observed_monotonic: Optional[float] = None
        self._pending_data: Optional[TeleData] = None
        self._last_good_data = self._make_neutral_data()
        self._needs_reanchor = True
        self._enable_requested = False
        self._continuous_valid_since: Optional[float] = None
        self._continuous_valid_packet_count = 0
        self._settling_complete = False
        self._external_hold = False

        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self.last_video_error: Optional[Exception] = None
        self.last_cleanup_errors: list[Exception] = []
        self.startup_error: Optional[Exception] = None
        self._startup_thread: Optional[threading.Thread] = None
        self._streamer_lock = threading.Lock()
        self.streamer = None

        streamer_kwargs = {
            "ip": ip,
            "record": False,
            "ht_backend": tracking_transport,
            "origin": "avp",
            "verbose": verbose,
        }
        video_kwargs = {
            "device": None,
            "size": f"{self.img_shape[1]}x{self.img_shape[0]}",
            "fps": max(1, int(round(self.display_fps))),
            "stereo": False,
        }

        if streamer is not None:
            if getattr(streamer, "record", False):
                # Injected streamers are primarily for tests, but keep recording
                # off if a caller injects an already-created stock streamer.
                streamer.record = False
            self._finish_streamer_startup(
                streamer, start_video, video_kwargs, int(webrtc_port)
            )
        elif streamer_factory is not None:
            # An explicit factory is an injection seam for deterministic tests.
            # The stock class path below is always asynchronous because its
            # constructor waits indefinitely for the first tracking packet.
            created = streamer_factory(**streamer_kwargs)
            self._finish_streamer_startup(
                created, start_video, video_kwargs, int(webrtc_port)
            )
        else:
            factory = _load_streamer_class()
            self._startup_thread = threading.Thread(
                target=self._streamer_startup_worker,
                args=(factory, streamer_kwargs, start_video, video_kwargs, int(webrtc_port)),
                name="visionproteleop-startup",
                daemon=True,
            )
            self._startup_thread.start()

    @property
    def last_packet_monotonic(self) -> Optional[float]:
        return self._last_packet_monotonic

    @property
    def tracking_active(self) -> bool:
        return (
            self.state == TRACKING_OK
            and not self._closed
            and self._is_fresh(self._clock())
        )

    def _streamer_startup_worker(
        self,
        factory: Callable[..., Any],
        streamer_kwargs: dict[str, Any],
        start_video: bool,
        video_kwargs: dict[str, Any],
        webrtc_port: int,
    ) -> None:
        created = None
        try:
            created = factory(**streamer_kwargs)
            self._finish_streamer_startup(
                created, start_video, video_kwargs, webrtc_port
            )
        except Exception as exc:
            self.startup_error = exc
            self._needs_reanchor = True
            self._transition(HOLD, f"streamer startup failed: {exc}")

    def _finish_streamer_startup(
        self,
        streamer: Any,
        start_video: bool,
        video_kwargs: dict[str, Any],
        webrtc_port: int,
    ) -> None:
        # The stock constructor may unblock only after the outer teleop process
        # has already begun shutdown.  Check before configuring WebRTC so a
        # late constructor cannot resurrect video/network work after close().
        with self._streamer_lock:
            already_closed = self._closed
        if already_closed:
            try:
                streamer.cleanup()
            except Exception as cleanup_exc:
                self.last_cleanup_errors.append(cleanup_exc)
            return

        try:
            if start_video:
                streamer.configure_video(**video_kwargs)
                streamer.register_frame_callback(self._video_frame_callback)
                streamer.start_webrtc(port=webrtc_port, blocking=False)
        except Exception:
            try:
                streamer.cleanup()
            except Exception as cleanup_exc:
                self.last_cleanup_errors.append(cleanup_exc)
            raise

        with self._streamer_lock:
            if self._closed:
                try:
                    streamer.cleanup()
                except Exception as cleanup_exc:
                    self.last_cleanup_errors.append(cleanup_exc)
                return
            self.streamer = streamer

    def request_enable(self) -> bool:
        """Explicitly accept the latest fresh pose as the new tracking anchor."""
        if self._closed or self._external_hold or self._pending_data is None:
            return False
        now = self._clock()
        if not self._is_fresh(now):
            return False
        self._enable_requested = True
        if not self._settling_complete:
            self._enable_requested = False
            elapsed = self._settling_elapsed(now)
            self._transition(
                REANCHOR_REQUIRED,
                "operator enable rejected; valid tracking settling "
                f"{elapsed:.3f}/{self.settling_time_s:.3f}s",
            )
            return False
        self._activate_pending()
        return True

    def request_hold(self, reason: str = "operator hold") -> None:
        """Fail closed until a subsequent explicit :meth:`request_enable`."""
        self._reset_reanchor(reset_motion=True)
        self._transition(HOLD, reason)

    def set_hold(self, enabled: bool, reason: str = "operator hold") -> None:
        """Convenience hook for an external U-key/e-stop state.

        Clearing ``enabled`` does not resume tracking.  The caller must still
        invoke :meth:`request_enable` so a released hold cannot auto-resume.
        """
        enabled = bool(enabled)
        if enabled and not self._external_hold:
            self._external_hold = True
            self.request_hold(reason)
        elif enabled:
            self._transition(HOLD, reason)
        elif self._external_hold:
            self._external_hold = False
            if self._is_fresh(self._clock()) and self._pending_data is not None:
                self._transition(
                    REANCHOR_REQUIRED,
                    "external hold released; explicit enable required",
                )
            else:
                self._transition(TRACKING_STALE, "external hold released without fresh pose")

    def get_status(self) -> dict[str, Any]:
        now = self._clock()
        if (
            not self._closed
            and self.startup_error is None
            and self._last_packet_monotonic is not None
            and not self._is_fresh(now)
        ):
            self._mark_stale("tracking packet timeout")
        with self._frame_lock:
            video_input_times = tuple(self._video_input_times)
            video_callback_times = tuple(self._video_callback_times)
        diagnostic_data = (
            self._pending_data
            if self._pending_data is not None
            else self._last_good_data
        )
        left_hand = np.asarray(diagnostic_data.left_hand_pos, dtype=np.float64)
        right_hand = np.asarray(diagnostic_data.right_hand_pos, dtype=np.float64)
        return {
            "state": self.state,
            "reason": self.state_reason,
            "tracking_active": self.tracking_active,
            "session_alive": self._is_fresh(now) and not self._closed,
            "sample_age_s": self._sample_age(now),
            "transport_latency_ms": None,
            "packet_count": self.packet_count,
            "tracking_rate_hz": self._recent_rate(self._packet_times, now),
            "invalid_packet_count": self.invalid_packet_count,
            "rejected_jump_count": self.rejected_jump_count,
            "reconnect_count": self.reconnect_count,
            "stale_transition_count": self.stale_transition_count,
            "hold_count": self.hold_count,
            "video_input_count": self.video_input_count,
            "video_input_rate_hz": self._recent_rate(video_input_times, now),
            "video_callback_count": self.video_callback_count,
            "video_callback_rate_hz": self._recent_rate(video_callback_times, now),
            "settling_elapsed_s": self._settling_elapsed(now),
            "settling_complete": self._settling_complete,
            "enable_requested": self._enable_requested,
            "external_hold": self._external_hold,
            "startup_pending": (
                self.streamer is None
                and self.startup_error is None
                and not self._closed
            ),
            "startup_error": None if self.startup_error is None else str(self.startup_error),
            "native_tracking_status_available": False,
            "head_position_m": diagnostic_data.head_pose[:3, 3].copy(),
            "head_yaw_rad": float(
                math.atan2(
                    diagnostic_data.head_pose[1, 0],
                    diagnostic_data.head_pose[0, 0],
                )
            ),
            "left_wrist_position_m": diagnostic_data.left_wrist_pose[:3, 3].copy(),
            "right_wrist_position_m": diagnostic_data.right_wrist_pose[:3, 3].copy(),
            "left_hand_joint_count": int(left_hand.shape[0]),
            "right_hand_joint_count": int(right_hand.shape[0]),
            "left_hand_nonzero_points": int(
                np.count_nonzero(np.linalg.norm(left_hand, axis=1) > 1e-6)
            ),
            "right_hand_nonzero_points": int(
                np.count_nonzero(np.linalg.norm(right_hand, axis=1) > 1e-6)
            ),
            "left_pinch_distance_cm": float(diagnostic_data.left_hand_pinchValue),
            "right_pinch_distance_cm": float(diagnostic_data.right_hand_pinchValue),
        }

    def get_tele_data(self) -> TeleData:
        """Poll once and return a fail-closed TeleData-compatible snapshot."""
        now = self._clock()
        if self._closed:
            return self._annotate_output(now)

        with self._streamer_lock:
            streamer = self.streamer
        if streamer is None:
            return self._annotate_output(now)

        try:
            latest = streamer.get_latest()
        except Exception as exc:
            self._mark_stale(f"tracking receive failed: {exc}")
            return self._annotate_output(now)

        raw = self._raw_packet(latest)
        if raw is not None and raw is not self._last_raw_object:
            # Retain the object itself, not only id(raw), so Python cannot reuse
            # an id and accidentally make an old sample look fresh.
            previous_arrival = self._last_packet_monotonic
            reconnect = (
                previous_arrival is not None
                and (
                    self.state == TRACKING_STALE
                    or now - previous_arrival > self.tracking_timeout_s
                )
            )
            if reconnect:
                self.reconnect_count += 1
                self._reset_reanchor(reset_motion=True)
                if self._external_hold:
                    self._transition(HOLD, "external hold active during reconnect")
                else:
                    self._transition(
                        REANCHOR_REQUIRED,
                        "tracking stream reconnected; explicit enable required",
                    )
            self._last_raw_object = raw
            self._last_packet_monotonic = now
            self.packet_count += 1
            self._packet_times.append(now)
            self._process_new_packet(raw, now)

        if not self._is_fresh(now):
            self._mark_stale("tracking packet timeout")
        return self._annotate_output(now)

    def render_to_xr(self, image: np.ndarray) -> None:
        """Publish one mono BGR frame to the native Vision Pro video plane."""
        frame = np.asarray(image)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Vision Pro frame must be HxWx3 BGR, got {frame.shape}")
        if not np.issubdtype(frame.dtype, np.number):
            raise ValueError("Vision Pro frame must use a numeric dtype")
        if not np.all(np.isfinite(frame)):
            raise ValueError("Vision Pro frame contains non-finite values")

        frame = np.clip(frame, 0, 255).astype(np.uint8, copy=False)
        frame = self._resize_bgr(frame, self.img_shape)
        frame = np.ascontiguousarray(frame).copy()
        frame.setflags(write=False)
        now = self._clock()
        with self._frame_lock:
            self._latest_frame = frame
            self.video_input_count += 1
            self._video_input_times.append(now)

        # The registered callback is authoritative.  update_frame also makes a
        # new frame immediately visible to stock avp_stream's synthetic track.
        try:
            with self._streamer_lock:
                streamer = self.streamer
            update_frame = None if streamer is None else getattr(streamer, "update_frame", None)
            if update_frame is not None:
                update_frame(frame)
            self.last_video_error = None
        except Exception as exc:
            # A video-plane failure must not mutate tracking safety state or
            # crash a control loop.  The callback can still serve the frame.
            self.last_video_error = exc

    def set_tactile_overlay(self, *_args: Any, **_overlay: Any) -> None:
        """Compatibility no-op; AI Worker/HX5 has no RH5 tactile overlay."""

    def close(self) -> None:
        """Best-effort, idempotent streamer cleanup."""
        if self._closed:
            return
        self._closed = True
        self._reset_reanchor(reset_motion=True)
        self._transition(HOLD, "backend closed")
        with self._streamer_lock:
            streamer = self.streamer
            self.streamer = None
        if streamer is not None:
            try:
                streamer.cleanup()
            except Exception as exc:
                self.last_cleanup_errors.append(exc)

    def _process_new_packet(self, packet: Any, now: float) -> None:
        if not isinstance(packet, Mapping):
            self.invalid_packet_count += 1
            self._reset_reanchor(reset_motion=True)
            self._transition(HOLD, "tracking payload is not a mapping")
            return

        try:
            candidate = self._convert_packet(packet)
        except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
            self.invalid_packet_count += 1
            self._reset_reanchor(reset_motion=True)
            self._transition(HOLD, f"invalid tracking packet: {exc}")
            return

        motion_error = self._wrist_motion_error(candidate, now)
        if motion_error is not None:
            self.rejected_jump_count += 1
            self._reset_reanchor(reset_motion=True)
            self._transition(HOLD, motion_error)
            return

        self._pending_data = candidate
        self._record_continuous_valid(now)
        if self._external_hold:
            self._transition(HOLD, "external hold active")
            return
        if self.state == TRACKING_OK and not self._needs_reanchor:
            self._last_good_data = copy.deepcopy(candidate)
        else:
            self._needs_reanchor = True
            if self._enable_requested and self._settling_complete:
                self._activate_pending()
            else:
                elapsed = self._settling_elapsed(now)
                self._transition(
                    REANCHOR_REQUIRED,
                    "fresh tracking requires explicit enable; settling "
                    f"{elapsed:.3f}/{self.settling_time_s:.3f}s",
                )

    def _convert_packet(self, packet: Mapping[str, Any]) -> TeleData:
        head_avp = _squeeze_pose(packet.get("head"), "head")
        left_wrist_avp = _squeeze_pose(packet.get("left_wrist"), "left_wrist")
        right_wrist_avp = _squeeze_pose(packet.get("right_wrist"), "right_wrist")
        left_joints = _extract_joints(packet, "left")
        right_joints = _extract_joints(packet, "right")

        for name, transform in (
            ("head", head_avp),
            ("left_wrist", left_wrist_avp),
            ("right_wrist", right_wrist_avp),
        ):
            _validate_rigid_transform(transform, name, self.rigid_atol)
        for side, joints in (("left", left_joints), ("right", right_joints)):
            for index, transform in enumerate(joints):
                _validate_rigid_transform(
                    transform, f"{side}_hand[{index}]", self.rigid_atol
                )

        # avp_stream head = A @ raw_openxr_head @ Rx(-90deg).
        head_robot = (
            T_AVP_TO_ROBOT_WORLD
            @ head_avp
            @ T_AVP_HEAD_CORRECTION_INV
            @ T_OPENXR_ROBOT
        )
        left_wrist_robot = (
            T_AVP_TO_ROBOT_WORLD
            @ left_wrist_avp
            @ T_OPENXR_ROBOT
            @ T_TO_UNITREE_HUMANOID_LEFT_ARM
        )
        right_wrist_robot = (
            T_AVP_TO_ROBOT_WORLD
            @ right_wrist_avp
            @ T_OPENXR_ROBOT
            @ T_TO_UNITREE_HUMANOID_RIGHT_ARM
        )

        # Match Vuer: WORLD -> HEAD is translation-only, then use the fixed
        # head-to-waist IK origin offset.
        left_wrist_robot = left_wrist_robot.copy()
        right_wrist_robot = right_wrist_robot.copy()
        left_wrist_robot[:3, 3] -= head_robot[:3, 3]
        right_wrist_robot[:3, 3] -= head_robot[:3, 3]
        left_wrist_robot[0, 3] += 0.15
        right_wrist_robot[0, 3] += 0.15
        left_wrist_robot[2, 3] += 0.45
        right_wrist_robot[2, 3] += 0.45

        left_hand_pos = self._convert_hand_positions(left_joints)
        right_hand_pos = self._convert_hand_positions(right_joints)

        if self.return_hand_rot_data:
            left_hand_rot = self._convert_hand_rotations(left_joints)
            right_hand_rot = self._convert_hand_rotations(right_joints)
        else:
            left_hand_rot = None
            right_hand_rot = None

        left_pinch_distance = self._pinch_distance(packet, left_joints, "left")
        right_pinch_distance = self._pinch_distance(packet, right_joints, "right")

        return TeleData(
            head_pose=head_robot,
            left_wrist_pose=left_wrist_robot,
            right_wrist_pose=right_wrist_robot,
            left_hand_pos=left_hand_pos,
            right_hand_pos=right_hand_pos,
            left_hand_rot=left_hand_rot,
            right_hand_rot=right_hand_rot,
            left_hand_pinch=left_pinch_distance <= self.pinch_threshold_m,
            left_hand_pinchValue=left_pinch_distance * 100.0,
            left_hand_squeeze=False,
            left_hand_squeezeValue=0.0,
            right_hand_pinch=right_pinch_distance <= self.pinch_threshold_m,
            right_hand_pinchValue=right_pinch_distance * 100.0,
            right_hand_squeeze=False,
            right_hand_squeezeValue=0.0,
        )

    @staticmethod
    def _convert_hand_positions(joints: np.ndarray) -> np.ndarray:
        local_points = joints[:, :, 3].T
        converted = T_TO_UNITREE_HAND @ T_ROBOT_OPENXR @ local_points
        return converted[:3].T.copy()

    @staticmethod
    def _convert_hand_rotations(joints: np.ndarray) -> np.ndarray:
        rotations = joints[:, :3, :3]
        basis = T_ROBOT_OPENXR[:3, :3]
        inverse_basis = T_OPENXR_ROBOT[:3, :3]
        return np.einsum("ij,njk,kl->nil", basis, rotations, inverse_basis)

    @staticmethod
    def _pinch_distance(
        packet: Mapping[str, Any], joints: np.ndarray, side: str
    ) -> float:
        value = packet.get(f"{side}_pinch_distance")
        if value is None:
            value = np.linalg.norm(joints[4, :3, 3] - joints[9, :3, 3])
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{side}_pinch_distance is invalid")
        return value

    def _wrist_motion_error(self, candidate: TeleData, now: float) -> Optional[str]:
        current = (candidate.left_wrist_pose, candidate.right_wrist_pose)
        previous = self._last_observed_wrist_poses
        previous_time = self._last_observed_monotonic
        self._last_observed_wrist_poses = tuple(pose.copy() for pose in current)
        self._last_observed_monotonic = now

        if not self.motion_rejection_enabled:
            return None
        if previous is None or previous_time is None:
            return None
        delta_time = now - previous_time
        for side, old_pose, new_pose in zip(("left", "right"), previous, current):
            translation = float(np.linalg.norm(new_pose[:3, 3] - old_pose[:3, 3]))
            rotation = _rotation_distance(old_pose, new_pose)
            if translation > self.max_wrist_translation_jump_m:
                return (
                    f"{side} wrist translation jump {translation:.3f}m exceeds "
                    f"{self.max_wrist_translation_jump_m:.3f}m"
                )
            if rotation > self.max_wrist_rotation_jump_rad:
                return (
                    f"{side} wrist rotation jump {rotation:.3f}rad exceeds "
                    f"{self.max_wrist_rotation_jump_rad:.3f}rad"
                )
            if delta_time > 1e-6:
                velocity = translation / delta_time
                angular_velocity = rotation / delta_time
                if velocity > self.max_wrist_velocity_m_s:
                    return (
                        f"{side} wrist velocity {velocity:.3f}m/s exceeds "
                        f"{self.max_wrist_velocity_m_s:.3f}m/s"
                    )
                if angular_velocity > self.max_wrist_angular_velocity_rad_s:
                    return (
                        f"{side} wrist angular velocity {angular_velocity:.3f}rad/s "
                        f"exceeds {self.max_wrist_angular_velocity_rad_s:.3f}rad/s"
                    )
        return None

    @staticmethod
    def _raw_packet(latest: Any) -> Any:
        if latest is None:
            return None
        if isinstance(latest, Mapping):
            return latest
        raw = getattr(latest, "raw", None)
        if raw is None:
            raw = getattr(latest, "_raw", None)
        return raw

    def _record_continuous_valid(self, now: float) -> None:
        was_settled = self._settling_complete
        if self._continuous_valid_since is None:
            self._continuous_valid_since = now
            self._continuous_valid_packet_count = 1
        else:
            self._continuous_valid_packet_count += 1
        elapsed = self._settling_elapsed(now)
        self._settling_complete = (
            elapsed >= self.settling_time_s
            and (
                self.settling_time_s == 0.0
                or self._continuous_valid_packet_count >= 2
            )
        )
        if (
            self._settling_complete
            and not was_settled
            and self._status_callback is not None
        ):
            self._status_callback(
                "[VisionProTeleop] REANCHOR_REQUIRED: tracking settled; press R once to enable."
            )

    def _settling_elapsed(self, now: float) -> float:
        if self._continuous_valid_since is None:
            return 0.0
        return max(0.0, now - self._continuous_valid_since)

    def _activate_pending(self) -> None:
        if self._pending_data is None:
            return
        self._last_good_data = copy.deepcopy(self._pending_data)
        self._needs_reanchor = False
        self._enable_requested = False
        self._transition(TRACKING_OK, "operator enabled settled tracking anchor")

    def _reset_reanchor(self, *, reset_motion: bool) -> None:
        self._needs_reanchor = True
        self._enable_requested = False
        self._pending_data = None
        self._continuous_valid_since = None
        self._continuous_valid_packet_count = 0
        self._settling_complete = False
        if reset_motion:
            self._last_observed_wrist_poses = None
            self._last_observed_monotonic = None

    def _mark_stale(self, reason: str) -> None:
        if self._external_hold:
            if "tracking packet timeout" not in self.state_reason:
                self.stale_transition_count += 1
            self._reset_reanchor(reset_motion=True)
            self._transition(HOLD, f"external hold active; {reason}")
            return
        if self.state != TRACKING_STALE:
            self.stale_transition_count += 1
            self._reset_reanchor(reset_motion=True)
            self._transition(TRACKING_STALE, reason)
        elif self.state_reason != reason:
            self.state_reason = reason

    def _transition(self, state: str, reason: str) -> None:
        state_changed = state != self.state
        if state == HOLD and self.state != HOLD:
            self.hold_count += 1
        self.state = state
        self.state_reason = reason
        # Settling reasons contain a continuously changing elapsed time.  Emit
        # only state changes here so a 60 Hz tracking stream cannot flood the
        # control-loop console; get_status() exposes the current detailed reason.
        if state_changed and self._status_callback is not None:
            self._status_callback(f"[VisionProTeleop] {state}: {reason}")

    def _is_fresh(self, now: float) -> bool:
        return (
            self._last_packet_monotonic is not None
            and now - self._last_packet_monotonic <= self.tracking_timeout_s
        )

    def _sample_age(self, now: float) -> float:
        if self._last_packet_monotonic is None:
            return math.inf
        return max(0.0, now - self._last_packet_monotonic)

    @staticmethod
    def _recent_rate(timestamps: Any, now: float, window_s: float = 2.0) -> float:
        recent = [stamp for stamp in timestamps if now - stamp <= window_s]
        if len(recent) < 2:
            return 0.0
        duration = recent[-1] - recent[0]
        if duration <= 0.0:
            return 0.0
        return (len(recent) - 1) / duration

    def _annotate_output(self, now: float) -> TeleData:
        output = copy.deepcopy(self._last_good_data)
        is_ok = self.state == TRACKING_OK and self._is_fresh(now) and not self._closed
        output.tracking_active = is_ok
        output.session_alive = self._is_fresh(now) and not self._closed
        output.head_pose_is_valid = is_ok
        output.left_arm_is_valid = is_ok
        output.right_arm_is_valid = is_ok
        output.tracking_state = self.state
        output.tracking_reason = self.state_reason
        output.tracking_sample_age_s = self._sample_age(now)
        output.native_tracking_status_available = False
        return output

    def _make_neutral_data(self) -> TeleData:
        left_wrist = np.eye(4)
        right_wrist = np.eye(4)
        left_wrist[:3, 3] = np.array([0.15, 0.0, 0.45])
        right_wrist[:3, 3] = np.array([0.15, 0.0, 0.45])
        rotations = (
            np.repeat(np.eye(3)[None, :, :], 25, axis=0)
            if self.return_hand_rot_data
            else None
        )
        return TeleData(
            head_pose=np.eye(4),
            left_wrist_pose=left_wrist,
            right_wrist_pose=right_wrist,
            left_hand_pos=np.zeros((25, 3)),
            right_hand_pos=np.zeros((25, 3)),
            left_hand_rot=None if rotations is None else rotations.copy(),
            right_hand_rot=None if rotations is None else rotations.copy(),
        )

    def _video_frame_callback(self, _source_frame: np.ndarray) -> np.ndarray:
        now = self._clock()
        with self._frame_lock:
            self.video_callback_count += 1
            self._video_callback_times.append(now)
            if self._latest_frame is None:
                frame = np.zeros((*self.img_shape, 3), dtype=np.uint8)
                frame.setflags(write=False)
                return frame
            return self._latest_frame

    @staticmethod
    def _resize_bgr(frame: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        target_height, target_width = target_shape
        if frame.shape[:2] == target_shape:
            return frame
        source_height, source_width = frame.shape[:2]
        if cv2 is not None:
            interpolation = (
                cv2.INTER_AREA
                if target_height < source_height or target_width < source_width
                else cv2.INTER_LINEAR
            )
            return cv2.resize(
                frame,
                (target_width, target_height),
                interpolation=interpolation,
            )
        y_indices = np.linspace(0, source_height - 1, target_height).astype(np.intp)
        x_indices = np.linspace(0, source_width - 1, target_width).astype(np.intp)
        return frame[y_indices][:, x_indices]


__all__ = [
    "HOLD",
    "REANCHOR_REQUIRED",
    "SyntheticVisionProStreamer",
    "TeleData",
    "TRACKING_OK",
    "TRACKING_STALE",
    "VisionProTeleopBackend",
]
