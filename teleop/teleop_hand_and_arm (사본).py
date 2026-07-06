import time
import argparse
import copy
import json
from multiprocessing import Value, Array, Lock
import threading
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)
try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np

import os 
import sys
import socket
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds 
from televuer import TeleVuerWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController, H2_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK, H2_ArmIK
from teleimager.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.audio_recorder import BackgroundAudioRecorder, AudioRecorderError
from teleop.utils.ipc import IPC_Server
from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
from teleop.utils.rh5dg2_tactile import RH5DG2TactileHeatMapper
from teleop.neck_control import VisionProNeckController
from sshkeyboard import listen_keyboard, stop_listening

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
def publish_reset_category(category: int, publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
TACTILE_VR_OVERLAY_VISIBLE = True  # Toggle RH5DG2 tactile overlay visibility in the XR viewer
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_TOGGLE   False       True          False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: when READY is True, set RECORD_TOGGLE=True to transition.
#  --> auto  : Auto-transition after saving data.

def on_press(key):
    global STOP, START, READY, RECORD_RUNNING, RECORD_TOGGLE, TACTILE_VR_OVERLAY_VISIBLE
    if key == 'r':
        START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's':
        if START == True and (READY or RECORD_RUNNING):
            RECORD_TOGGLE = True
        elif START == True:
            logger_mp.warning("[teleop record] ignored [s] because the previous episode is still saving. Please wait until READY.")
        else:
            logger_mp.warning("[teleop record] ignored [s] because teleop has not started. Press [r] first, then [s] to record.")
    elif key == 't':
        TACTILE_VR_OVERLAY_VISIBLE = not TACTILE_VR_OVERLAY_VISIBLE
        logger_mp.info(
            "[RH5DG2 tactile VR overlay] keyboard toggle visible=%s",
            TACTILE_VR_OVERLAY_VISIBLE,
        )
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
    }

def _fmt_hand_debug(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return "len=0"
    flat = arr.reshape(-1)
    point0 = arr.reshape(25, 3)[0].tolist() if arr.size == 75 else flat[:3].tolist()
    return (
        f"shape={arr.shape} min={flat.min():.4f} max={flat.max():.4f} "
        f"allzero={np.allclose(flat, 0.0, atol=1e-5)} p0={np.round(point0, 4).tolist()}"
    )

def _hand_tracking_status(left_values, right_values):
    left = np.asarray(left_values, dtype=np.float64).reshape(-1, 3)
    right = np.asarray(right_values, dtype=np.float64).reshape(-1, 3)
    left_norm = np.linalg.norm(left, axis=1) if left.size else np.array([])
    right_norm = np.linalg.norm(right, axis=1) if right.size else np.array([])
    left_allzero = bool(left_norm.size == 0 or np.allclose(left, 0.0, atol=1e-5))
    right_allzero = bool(right_norm.size == 0 or np.allclose(right, 0.0, atol=1e-5))
    left_valid_points = int(np.sum(left_norm > 1e-5))
    right_valid_points = int(np.sum(right_norm > 1e-5))
    return {
        "hand_tracking_ready": bool(not left_allzero and not right_allzero),
        "left_allzero": left_allzero,
        "right_allzero": right_allzero,
        "left_valid_points": left_valid_points,
        "right_valid_points": right_valid_points,
    }

def _fmt_pose_debug(values):
    arr = np.asarray(values, dtype=np.float64)
    flat = arr.reshape(-1)
    return (
        f"shape={arr.shape} finite={np.isfinite(flat).all()} "
        f"first={np.round(flat[: min(7, flat.size)], 4).tolist()}"
    )

def _fmt_vec_debug(values):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return "len=0"
    return (
        f"len={arr.size} min={arr.min():.4f} max={arr.max():.4f} "
        f"first7={np.round(arr[:7], 4).tolist()} last7={np.round(arr[-7:], 4).tolist()}"
    )


def _read_body_qpos(arm_ctrl, enabled=True):
    if not enabled or arm_ctrl is None or not hasattr(arm_ctrl, "get_current_motor_q"):
        return []
    try:
        qpos = np.asarray(arm_ctrl.get_current_motor_q(), dtype=np.float64).reshape(-1)
    except Exception as exc:
        logger_mp.debug("[teleop record body] failed to read body qpos: %s", exc)
        return []
    if qpos.size == 0 or not np.isfinite(qpos).all():
        return []
    return qpos.tolist()

def _safe_render_to_xr(tv_wrapper, image, log_prefix):
    try:
        tv_wrapper.render_to_xr(image)
        return True
    except Exception as exc:
        logger_mp.warning(f"{log_prefix} render_to_xr failed: {exc}")
        return False

def _tcp_check(host, port, timeout=0.35):
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True, "ok"
    except Exception as exc:
        return False, f"{exc.__class__.__name__}: {exc}"


class NeckFeedbackReceiver:
    def __init__(self, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("0.0.0.0", int(port)))
        self.socket.setblocking(False)
        self.latest = None

    def read_latest(self):
        data = None
        try:
            while True:
                data, _ = self.socket.recvfrom(256)
        except BlockingIOError:
            pass
        if data is not None:
            try:
                parts = data.decode("ascii").strip().split(",")
                if len(parts) == 2:
                    yaw, pitch = float(parts[0]), float(parts[1])
                    if np.isfinite([yaw, pitch]).all():
                        self.latest = {
                            "timestamp": time.time(),
                            "yaw_pitch": [yaw, pitch],
                        }
            except Exception:
                logger_mp.debug("[teleop neck feedback] malformed packet=%r", data)
        return None if self.latest is None else dict(self.latest)

    def close(self):
        self.socket.close()


class RH5DG2TactileUDPReceiver:
    def __init__(self, port, host="0.0.0.0", timeout=1.0, debug_rate=0.0, recv_size=8192):
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self.debug_rate = max(float(debug_rate), 0.0)
        self.recv_size = int(recv_size)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.settimeout(0.1)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest = {}
        self.last_rx_time = 0.0
        self.packets = 0
        self.errors = 0
        self.last_debug = 0.0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                raw, addr = self.socket.recvfrom(self.recv_size)
            except socket.timeout:
                self._debug()
                continue
            except OSError:
                if not self.stop_event.is_set():
                    with self.lock:
                        self.errors += 1
                break
            try:
                packet = json.loads(raw.decode("utf-8"))
                if not isinstance(packet, dict):
                    raise ValueError("packet is not a JSON object")
                with self.lock:
                    self.latest = packet
                    self.last_rx_time = time.time()
                    self.packets += 1
            except Exception as exc:
                with self.lock:
                    self.errors += 1
                logger_mp.debug("[RH5DG2 tactile UDP] malformed packet from %s: %s", addr, exc)
            self._debug()

    def _debug(self):
        if self.debug_rate <= 0.0:
            return
        now = time.time()
        if now - self.last_debug < 1.0 / self.debug_rate:
            return
        self.last_debug = now
        with self.lock:
            age = None if self.last_rx_time <= 0.0 else now - self.last_rx_time
            keys = list(self.latest.keys()) if isinstance(self.latest, dict) else []
            logger_mp.info(
                "[RH5DG2 tactile UDP] packets=%s errors=%s age=%s keys=%s",
                self.packets,
                self.errors,
                None if age is None else round(age, 3),
                keys,
            )

    def read_latest(self):
        now = time.time()
        with self.lock:
            if not self.latest:
                return {}
            latest = copy.deepcopy(self.latest)
            age = now - self.last_rx_time
        if age > self.timeout:
            latest["_stale"] = True
            latest["_age_sec"] = age
        return latest

    def close(self):
        self.stop_event.set()
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass
        self.socket.close()


class AudioUDPReceiver:
    def __init__(self, port, host="0.0.0.0", timeout=1.0, recv_size=65535):
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self.recv_size = int(recv_size)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.settimeout(0.1)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest = {}
        self.last_rx_time = 0.0
        self.packets = 0
        self.errors = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _coerce_packet(self, raw):
        try:
            packet = json.loads(raw.decode("utf-8"))
            if isinstance(packet, dict):
                mic = str(packet.get("mic", "mic_0"))
                samples = packet.get("samples", packet.get("pcm", packet.get("data")))
                if samples is None:
                    raise ValueError("audio JSON packet is missing samples/pcm/data")
                return {mic: np.asarray(samples, dtype=np.int16)}
        except UnicodeDecodeError:
            pass
        except json.JSONDecodeError:
            pass

        audio = np.frombuffer(raw, dtype=np.int16).copy()
        if audio.size == 0:
            raise ValueError("empty audio packet")
        return {"mic_0": audio}

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                raw, _ = self.socket.recvfrom(self.recv_size)
            except socket.timeout:
                continue
            except OSError:
                if not self.stop_event.is_set():
                    with self.lock:
                        self.errors += 1
                break
            try:
                packet = self._coerce_packet(raw)
                with self.lock:
                    self.latest = packet
                    self.last_rx_time = time.time()
                    self.packets += 1
            except Exception as exc:
                with self.lock:
                    self.errors += 1
                logger_mp.debug("[record audio UDP] malformed packet: %s", exc)

    def read_latest(self):
        now = time.time()
        with self.lock:
            if not self.latest or self.last_rx_time <= 0.0:
                return None
            if now - self.last_rx_time > self.timeout:
                return None
            return {mic: np.asarray(audio, dtype=np.int16).copy() for mic, audio in self.latest.items()}

    def close(self):
        self.stop_event.set()
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass
        self.socket.close()


def _local_ip_for_remote(remote_host):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.2)
        sock.connect((remote_host, 1))
        local_ip = sock.getsockname()[0]
        sock.close()
        return local_ip
    except Exception:
        return "127.0.0.1"

def _camera_config_key(camera_name):
    if camera_name == "left_wrist":
        return "left_wrist_camera"
    if camera_name == "right_wrist":
        return "right_wrist_camera"
    return "head_camera"

def _get_camera_frame(img_client, camera_name):
    if camera_name == "left_wrist":
        return img_client.get_left_wrist_frame()
    if camera_name == "right_wrist":
        return img_client.get_right_wrist_frame()
    return img_client.get_head_frame()

def _is_both_wrist_camera(camera_name):
    return camera_name in ("both", "both_wrist")

def _is_composite_wrist_camera(camera_name):
    return camera_name in ("both", "both_wrist", "head_and_wrist", "head_wrist")

def _apply_camera_orientation(image, camera_name, args):
    if image is None:
        return None
    oriented = image
    if camera_name == "left_wrist" and args.left_wrist_camera_vflip:
        oriented = np.flipud(oriented).copy()
    if camera_name == "right_wrist" and args.right_wrist_camera_vflip:
        oriented = np.flipud(oriented).copy()
    return oriented

def _compose_both_wrist_bgr(left_bgr, right_bgr):
    if left_bgr is None:
        return None if right_bgr is None else np.ascontiguousarray(right_bgr)
    if right_bgr is None:
        return np.ascontiguousarray(left_bgr)
    left = np.asarray(left_bgr)
    right = np.asarray(right_bgr)
    if left.ndim != 3 or right.ndim != 3:
        return None
    if left.shape[0] != right.shape[0]:
        target_h = min(left.shape[0], right.shape[0])
        if cv2 is not None:
            left_w = max(1, int(round(left.shape[1] * target_h / left.shape[0])))
            right_w = max(1, int(round(right.shape[1] * target_h / right.shape[0])))
            left = cv2.resize(left, (left_w, target_h), interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, (right_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            left = left[:target_h]
            right = right[:target_h]
    return np.ascontiguousarray(np.concatenate((left, right), axis=1))

def _get_both_wrist_bgr(img_client, args):
    left_frame = img_client.get_left_wrist_frame()
    right_frame = img_client.get_right_wrist_frame()
    left_bgr = None if left_frame is None else getattr(left_frame, "bgr", None)
    right_bgr = None if right_frame is None else getattr(right_frame, "bgr", None)
    left_bgr = _apply_camera_orientation(left_bgr, "left_wrist", args)
    right_bgr = _apply_camera_orientation(right_bgr, "right_wrist", args)
    return _compose_both_wrist_bgr(left_bgr, right_bgr), left_frame, right_frame

def _compose_head_and_wrist_bgr(head_bgr, wrist_bgr):
    if head_bgr is None:
        return None if wrist_bgr is None else np.ascontiguousarray(wrist_bgr)
    if wrist_bgr is None:
        return np.ascontiguousarray(head_bgr)
    head = np.asarray(head_bgr)
    wrist = np.asarray(wrist_bgr)
    if head.ndim != 3 or wrist.ndim != 3:
        return None
    target_w = max(head.shape[1], wrist.shape[1])
    if cv2 is not None:
        if head.shape[1] != target_w:
            head_h = max(1, int(round(head.shape[0] * target_w / head.shape[1])))
            head = cv2.resize(head, (target_w, head_h), interpolation=cv2.INTER_AREA)
        if wrist.shape[1] != target_w:
            wrist_h = max(1, int(round(wrist.shape[0] * target_w / wrist.shape[1])))
            wrist = cv2.resize(wrist, (target_w, wrist_h), interpolation=cv2.INTER_AREA)
    else:
        target_w = min(head.shape[1], wrist.shape[1])
        head = head[:, :target_w]
        wrist = wrist[:, :target_w]
    return np.ascontiguousarray(np.concatenate((head, wrist), axis=0))

def _fit_bgr_to_shape(image, target_shape):
    if image is None:
        return None
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    if target_h <= 0 or target_w <= 0:
        return image
    src = np.asarray(image)
    if src.ndim != 3:
        return None
    if src.shape[:2] == (target_h, target_w):
        return np.ascontiguousarray(src)
    if cv2 is None:
        out = np.zeros((target_h, target_w, src.shape[2]), dtype=src.dtype)
        crop_h = min(target_h, src.shape[0])
        crop_w = min(target_w, src.shape[1])
        out[:crop_h, :crop_w] = src[:crop_h, :crop_w]
        return out
    scale = min(target_w / src.shape[1], target_h / src.shape[0])
    resized_w = max(1, int(round(src.shape[1] * scale)))
    resized_h = max(1, int(round(src.shape[0] * scale)))
    resized = cv2.resize(src, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    out = np.zeros((target_h, target_w, src.shape[2]), dtype=src.dtype)
    y0 = (target_h - resized_h) // 2
    x0 = (target_w - resized_w) // 2
    out[y0:y0 + resized_h, x0:x0 + resized_w] = resized
    return out

def _get_head_and_wrist_bgr(img_client, args):
    head_frame = img_client.get_head_frame()
    head_bgr = None if head_frame is None else getattr(head_frame, "bgr", None)
    wrist_bgr, left_frame, right_frame = _get_both_wrist_bgr(img_client, args)
    return _compose_head_and_wrist_bgr(head_bgr, wrist_bgr), head_frame, left_frame, right_frame

def _log_camera_reachability(host, camera_config):
    checks = [("config", 60000)]
    for name in ("head_camera", "left_wrist_camera", "right_wrist_camera"):
        camera = camera_config.get(name, {})
        if camera.get("enable_webrtc"):
            checks.append((f"{name}.webrtc", camera.get("webrtc_port")))
        if camera.get("enable_zmq"):
            checks.append((f"{name}.zmq", camera.get("zmq_port")))

    parts = []
    for label, port in checks:
        if port is None:
            parts.append(f"{label}=missing_port")
            continue
        ok, detail = _tcp_check(host, port)
        parts.append(f"{label}={host}:{port} reachable={ok} detail={detail}")
    logger_mp.info(f"[teleop camera server check] {'; '.join(parts)}")

def _select_viewer_camera_route(display_mode, viewer_camera_mode, camera):
    if display_mode == "pass-through" or viewer_camera_mode == "none":
        return False, False, "none"

    enable_webrtc = bool(camera.get("enable_webrtc"))
    enable_zmq = bool(camera.get("enable_zmq"))
    if viewer_camera_mode == "auto":
        if enable_webrtc:
            return True, False, "webrtc"
        if enable_zmq:
            return False, True, "zmq"
        return False, False, "none"
    if viewer_camera_mode == "webrtc":
        return enable_webrtc, False, "webrtc" if enable_webrtc else "none"
    if viewer_camera_mode == "zmq":
        return False, enable_zmq, "zmq" if enable_zmq else "none"
    return False, False, "none"

def _rate_hz(count, start_time):
    elapsed = max(time.time() - start_time, 1e-6)
    return count / elapsed

def _frame_debug(frame):
    if frame is None:
        return "frame=None"
    bgr = getattr(frame, "bgr", None)
    jpg = getattr(frame, "jpg", None)
    return (
        f"frame={frame!r} fps={getattr(frame, 'fps', None)} "
        f"jpg_bytes={len(jpg) if jpg else 0} "
        f"bgr_shape={None if bgr is None else bgr.shape} "
        f"bgr_dtype={None if bgr is None else bgr.dtype}"
    )

def _sample_camera_frame(img_client, camera_name, attempts=30, sleep_s=0.05):
    for _ in range(attempts):
        frame = _get_camera_frame(img_client, camera_name)
        if frame is not None and getattr(frame, "bgr", None) is not None:
            return frame
        time.sleep(sleep_s)
    return frame if "frame" in locals() else None

def _parse_int_list(raw):
    if raw is None or str(raw).strip() == "":
        return None
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]

def _audio_info_disabled(args, error=None):
    info = {
        "enabled": False,
        "device": args.audio_device,
        "sample_rate": args.audio_sample_rate,
        "channels": args.audio_channels,
        "dtype": args.audio_dtype,
        "format": "wav",
        "path": "audios/audio.wav",
        "start_timestamp": None,
        "end_timestamp": None,
        "chunk_size": args.audio_chunk_size,
    }
    if error is not None:
        info["error"] = str(error)
    return info

def _start_episode_audio(args, recorder):
    if not args.enable_audio:
        return None

    audio_path = os.path.join(recorder.audio_dir, "audio.wav")
    audio_recorder = BackgroundAudioRecorder(
        audio_path,
        device=args.audio_device,
        sample_rate=args.audio_sample_rate,
        channels=args.audio_channels,
        dtype=args.audio_dtype,
        chunk_size=args.audio_chunk_size,
        rel_path="audios/audio.wav",
    )
    try:
        audio_recorder.start()
    except Exception as exc:
        recorder.update_info({"audio": _audio_info_disabled(args, error=exc)})
        raise AudioRecorderError(str(exc)) from exc

    recorder.update_info(
        {
            "audio": audio_recorder.metadata(),
            "audio_chunks": [],
        }
    )
    logger_mp.info(
        "[teleop audio] START path=%s device=%s sample_rate=%s channels=%s dtype=%s chunk_size=%s backend=%s",
        audio_recorder.rel_path,
        args.audio_device,
        args.audio_sample_rate,
        args.audio_channels,
        args.audio_dtype,
        args.audio_chunk_size,
        audio_recorder.active_backend,
    )
    return audio_recorder

def _stop_episode_audio(audio_recorder, recorder):
    if audio_recorder is None:
        return None
    metadata = audio_recorder.stop()
    updates = {"audio": metadata}
    if audio_recorder.chunk_timestamps:
        updates["audio_chunks"] = audio_recorder.chunk_timestamps
    recorder.update_info(updates)
    duration = None
    if metadata.get("start_timestamp") is not None and metadata.get("end_timestamp") is not None:
        duration = metadata["end_timestamp"] - metadata["start_timestamp"]
    logger_mp.info(
        "[teleop audio] STOP path=%s duration=%s chunks=%s samples=%s dropped_chunks=%s backend=%s",
        metadata.get("path"),
        None if duration is None else round(duration, 3),
        metadata.get("total_chunks"),
        metadata.get("total_samples"),
        metadata.get("dropped_chunks"),
        metadata.get("backend"),
    )
    return metadata


RH5DG2_VENDOR_DEMO_OPEN_BASELINE_LEFT = [
    1800, 1800, 1800,
    0,
    1650,
    0,
    1900, 1900, 1900,
    1870,
    1750, 1600, 1930,
]
RH5DG2_VENDOR_DEMO_OPEN_BASELINE_RIGHT = [
    1800, 1800, 1800,
    0,
    1650,
    0,
    1900, 1900, 1900,
    1870,
    1750, 1600, 1930,
]

def _parse_float_list(raw):
    if raw is None or str(raw).strip() == "":
        return None
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]

def _resolve_rh5dg2_safe_baseline(raw):
    if raw is None:
        return None
    value = str(raw).strip()
    if value == "" or value.lower() in ("none", "current", "startup"):
        return None
    if value.lower() in ("demo", "demo_open", "vendor_demo", "vendor_demo_open"):
        return {
            "left": list(RH5DG2_VENDOR_DEMO_OPEN_BASELINE_LEFT),
            "right": list(RH5DG2_VENDOR_DEMO_OPEN_BASELINE_RIGHT),
        }
    parsed = _parse_float_list(value)
    if parsed is not None and len(parsed) != 13:
        raise ValueError("--rh5dg2-safe-baseline must be demo_open/current or exactly 13 comma-separated values.")
    return parsed

def _resolve_rh5dg2_tactile_vr_sides(raw):
    value = str(raw or "right_ee").strip().lower()
    if value in ("both", "all", "dual", "left_right", "right_left"):
        return ["left_ee", "right_ee"]

    sides = []
    for item in value.replace("+", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if item in ("left", "left_ee", "left_hand"):
            side = "left_ee"
        elif item in ("right", "right_ee", "right_hand"):
            side = "right_ee"
        else:
            raise ValueError(
                "--rh5dg2-tactile-vr-side must be left_ee, right_ee, both, or a comma-separated pair."
            )
        if side not in sides:
            sides.append(side)
    return sides or ["right_ee"]

def _axis_angle_from_matrix(rot):
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    cos_angle = np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    if angle < 1e-7:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64), 0.0
    axis = np.array(
        [
            rot[2, 1] - rot[1, 2],
            rot[0, 2] - rot[2, 0],
            rot[1, 0] - rot[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * np.sin(angle))
    return axis / max(np.linalg.norm(axis), 1e-9), angle

def _matrix_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )

def _resolve_arm_sensitivity_args(args):
    explicit = any(
        value is not None
        for value in (
            args.arm_pos_gain_x,
            args.arm_pos_gain_y,
            args.arm_pos_gain_z,
            args.arm_rot_gain,
            args.arm_max_delta,
            args.arm_smoothing_alpha,
        )
    )
    if not explicit:
        return {
            "enabled": False,
            "pos_gain": np.ones(3, dtype=np.float64),
            "rot_gain": 1.0,
            "max_delta": float("inf"),
            "smoothing_alpha": 1.0,
        }
    sim_gain = 1.6 if args.sim else 1.0
    sim_z_gain = 1.9 if args.sim else 1.0
    return {
        "enabled": True,
        "pos_gain": np.array(
            [
                args.arm_pos_gain_x if args.arm_pos_gain_x is not None else sim_gain,
                args.arm_pos_gain_y if args.arm_pos_gain_y is not None else sim_gain,
                args.arm_pos_gain_z if args.arm_pos_gain_z is not None else sim_z_gain,
            ],
            dtype=np.float64,
        ),
        "rot_gain": float(args.arm_rot_gain if args.arm_rot_gain is not None else (1.15 if args.sim else 1.0)),
        "max_delta": float(args.arm_max_delta if args.arm_max_delta is not None else (0.85 if args.sim else 0.45)),
        "smoothing_alpha": float(
            args.arm_smoothing_alpha if args.arm_smoothing_alpha is not None else (0.85 if args.sim else 0.65)
        ),
    }

def _apply_arm_sensitivity(left_pose, right_pose, state, config, enabled=True):
    def convert(side, pose):
        arr = np.asarray(pose, dtype=np.float64)
        if not enabled or not config.get("enabled", False) or arr.shape != (4, 4) or not np.isfinite(arr).all():
            return pose, None

        base_key = f"{side}_base"
        smooth_key = f"{side}_smooth"
        if base_key not in state:
            state[base_key] = arr.copy()
            state[smooth_key] = arr.copy()

        base = state[base_key]
        human_delta = arr[:3, 3] - base[:3, 3]
        scaled_delta = human_delta * config["pos_gain"]
        clamped_delta = np.clip(scaled_delta, -config["max_delta"], config["max_delta"])

        rel_rot = base[:3, :3].T @ arr[:3, :3]
        axis, angle = _axis_angle_from_matrix(rel_rot)
        gained_rot = base[:3, :3] @ _matrix_from_axis_angle(axis, angle * config["rot_gain"])

        target = arr.copy()
        target[:3, 3] = base[:3, 3] + clamped_delta
        target[:3, :3] = gained_rot

        alpha = float(np.clip(config["smoothing_alpha"], 0.0, 1.0))
        prev = state[smooth_key]
        final = target.copy()
        final[:3, 3] = prev[:3, 3] + alpha * (target[:3, 3] - prev[:3, 3])
        state[smooth_key] = final.copy()
        debug = {
            "human_delta": human_delta,
            "scaled_delta": scaled_delta,
            "clamped_delta": clamped_delta,
            "final_pos": final[:3, 3].copy(),
        }
        return final, debug

    left_adjusted, left_debug = convert("left", left_pose)
    right_adjusted, right_debug = convert("right", right_pose)
    return left_adjusted, right_adjusted, left_debug, right_debug

def _fmt_arm_sensitivity_debug(side, debug):
    if debug is None:
        return f"{side}=unavailable"
    return (
        f"{side}:human_delta={np.round(debug['human_delta'], 4).tolist()},"
        f"scaled_delta={np.round(debug['scaled_delta'], 4).tolist()},"
        f"clamped_delta={np.round(debug['clamped_delta'], 4).tolist()},"
        f"final_pos={np.round(debug['final_pos'], 4).tolist()}"
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1', 'H2'], default='H1_2', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'inspire_dg2', 'rh5dg2_ftp', 'rh5dg2_dfx', 'rh56f1', 'brainco'], default='rh5dg2_dfx', help='Select end effector controller')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--viewer-host-ip', type=str, default=None, help='Host IP advertised to the XR browser for the HTTPS/WSS viewer. If omitted, infer it from the route to --img-server-ip.')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    parser.add_argument('--camera', '--viewer-camera', dest='camera', type=str, choices=['head', 'left_wrist', 'right_wrist', 'both', 'both_wrist', 'head_and_wrist', 'head_wrist', 'all'], default='head', help='Camera stream shown in the 8012 XR viewer. Use head_and_wrist to show the head view with both wrist cameras below it.')
    parser.add_argument('--viewer-camera-mode', type=str, choices=['auto', 'webrtc', 'zmq', 'none'], default='zmq', help='Select how the 8012 XR viewer receives the selected camera.')
    parser.add_argument('--viewer-display-fps', type=float, default=15.0, help='XR JPEG push rate for ZMQ camera mode. Lower this on congested Wi-Fi.')
    parser.add_argument('--viewer-jpeg-quality', type=int, default=60, help='XR JPEG quality for ZMQ camera mode, from 1 to 100.')
    parser.add_argument('--no-left-wrist-camera-vflip', dest='left_wrist_camera_vflip', action='store_false', help='Disable vertical flip correction for the left wrist camera.')
    parser.add_argument('--right-wrist-camera-vflip', action='store_true', help='Enable vertical flip correction for the right wrist camera.')
    parser.add_argument('--hand-control-hz', type=float, default=50.0, help='RH5DG2 hand retarget/publish loop frequency.')
    parser.add_argument('--hand-debug-rate', type=float, default=0.0, help='Teleop hand input debug log rate in Hz. Set 0 to disable periodic hand input logs.')
    parser.add_argument('--inspire-dg2-port', type=str, default='/dev/ttyUSB0', help='Shortcut serial port for both Inspire RH5DG2 hands when side-specific ports are omitted.')
    parser.add_argument('--inspire-dg2-left-port', type=str, default=None, help='Inspire RH5DG2 left hand RS485 serial port.')
    parser.add_argument('--inspire-dg2-right-port', type=str, default=None, help='Inspire RH5DG2 right hand RS485 serial port.')
    parser.add_argument('--inspire-dg2-baudrate', type=int, default=115200, help='Inspire RH5DG2 RS485 serial baudrate.')
    parser.add_argument('--inspire-dg2-left-id', type=int, default=2, help='Inspire RH5DG2 left hand ID on the RS485 bus.')
    parser.add_argument('--inspire-dg2-right-id', type=int, default=1, help='Inspire RH5DG2 right hand ID on the RS485 bus.')
    parser.add_argument('--inspire-dg2-state-hz', type=float, default=20.0, help='Inspire RH5DG2 angleAct polling frequency.')
    parser.add_argument('--inspire-dg2-tactile-hz', type=float, default=30.0, help='Inspire RH5DG2 tactile polling frequency.')
    parser.add_argument('--inspire-dg2-transport', type=str, choices=['dds', 'serial'], default='dds', help='Inspire RH5DG2 transport. Use dds with rh5dg2_serial_dds_bridge.py on the robot PC.')
    parser.add_argument('--inspire-dg2-bridge-host', type=str, default=None, help='Reserved for legacy UDP bridge mode.')
    parser.add_argument('--inspire-dg2-bridge-port', type=int, default=9720, help='Reserved for legacy UDP bridge mode.')
    parser.add_argument('--rh5dg2-log-throttle', type=float, default=1.0, help='RH5DG2 controller debug log rate in Hz.')
    parser.add_argument('--rh5dg2-hand-swap', action='store_true', help='Enable RH5DG2-only left/right hand input swap for devices that report swapped hand labels.')
    parser.add_argument('--rh5dg2-fast-mode', action=argparse.BooleanOptionalAction, default=True, help='Enable lower-latency RH5DG2 retarget settings.')
    parser.add_argument('--rh5dg2-retarget-mode', type=str, choices=['config', 'vector', 'dexpilot'], default='config', help='RH5DG2 retargeting mode. config uses assets/RH5DG2/RH5DG2.yml; dexpilot enables DexPilot without editing the YAML type.')
    parser.add_argument('--rh56f1-retarget-mode', type=str, choices=['config', 'vector', 'dexpilot'], default='config', help='RH56F1 retargeting mode. config uses assets/RH56F1/RH56F1.yml; dexpilot enables DexPilot without editing the YAML type.')
    parser.add_argument('--disable-hand-smoothing', action='store_true', help='Reserved flag for RH5DG2 hand path; current RH5DG2 path has no smoothing enabled.')
    parser.add_argument('--arm-pos-gain-x', type=float, default=1, help='Arm wrist translation gain on X; sim default is higher than real.')
    parser.add_argument('--arm-pos-gain-y', type=float, default=1, help='Arm wrist translation gain on Y; sim default is higher than real.')
    parser.add_argument('--arm-pos-gain-z', type=float, default=1, help='Arm wrist translation gain on Z; sim default is higher than real.')
    parser.add_argument('--arm-rot-gain', type=float, default=1, help='Arm wrist rotation gain.')
    parser.add_argument('--arm-max-delta', type=float, default=1, help='Per-axis clamp for gain-scaled wrist translation delta.')
    parser.add_argument('--arm-smoothing-alpha', type=float, default=1.0, help='Arm target smoothing alpha, where 1.0 means no translation lag.')
    parser.add_argument('--arm-startup-duration', type=float, default=2.0, help='Seconds to apply extra-small arm q target clamp after teleop starts.')
    parser.add_argument('--arm-startup-max-step', type=float, default=0.08, help='Max per-joint arm q target delta during startup clamp, in radians.')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--headless', action=argparse.BooleanOptionalAction, default=True, help='Enable headless mode and disable Rerun recording visualization by default. Use --no-headless to enable Rerun.')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--disable-arm', action='store_true', help='Disable arm IK/control while keeping XR and hand paths alive.')
    parser.add_argument('--disable-body', action=argparse.BooleanOptionalAction, default=True, help='Disable high-level body/loco command publishing.')
    parser.add_argument('--hand-only', action='store_true', help='Run XR input and end-effector hand control only; arm and body control stay off.')
    parser.add_argument('--enable-neck', action=argparse.BooleanOptionalAction, default=True, help='Send Vision Pro head yaw/pitch to the external UDP neck controller.')
    parser.add_argument('--neck-host', type=str, default=None, help='External neck controller host. Defaults to --img-server-ip.')
    parser.add_argument('--neck-port', type=int, default=9091, help='External neck controller UDP command port.')
    parser.add_argument('--neck-yaw-limit', type=float, default=1.2, help='Absolute relative neck yaw command limit in radians.')
    parser.add_argument('--neck-pitch-limit', type=float, default=0.8, help='Absolute relative neck pitch command limit in radians.')
    parser.add_argument('--neck-smoothing-alpha', type=float, default=0.25, help='Neck command low-pass alpha from 0 to 1.')
    parser.add_argument('--neck-max-step', type=float, default=0.08, help='Maximum neck command change per control frame in radians.')
    parser.add_argument('--neck-feedback-port', type=int, default=9093, help='UDP port that receives actual neck yaw,pitch feedback from the pan/tilt process.')
    parser.add_argument('--neck-log-rate', type=float, default=0.0, help='Neck debug log rate in Hz. Set 0 to disable periodic neck logs.')
    parser.add_argument('--enable-waist-follow-neck', action='store_true', help='Make the H1_2 waist yaw slowly follow the neck yaw command, including in --motion mode.')
    parser.add_argument('--waist-yaw-gain', type=float, default=0.5, help='H1_2 waist-yaw gain applied to the neck yaw command.')
    parser.add_argument('--waist-yaw-limit', type=float, default=0.1745, help='H1_2 relative waist-yaw limit in radians; default is about 10 degrees.')
    parser.add_argument('--waist-yaw-velocity', type=float, default=0.25, help='H1_2 waist-yaw velocity limit in radians per second.')
    parser.add_argument('--skip-arm-go-home-on-exit', action='store_true', help='Do not command arm zero/home pose during shutdown.')
    parser.add_argument('--rh5dg2-safe-mode', action=argparse.BooleanOptionalAction, default=True, help='Publish restricted RH5DG2 raw hand commands for real-hardware bringup.')
    parser.add_argument('--rh5dg2-active-hand', type=str, choices=['right', 'left', 'both'], default='both', help='RH5DG2 safe mode active DDS command hand.')
    parser.add_argument('--rh5dg2-enabled-indices', type=str, default='0,1,2,3,4,5,6,7,8,9,10,11,12', help='Comma-separated RH5DG2 raw actuator indices allowed in safe mode.')
    parser.add_argument('--rh5dg2-pitch-only', action='store_true', help='RH5DG2 safe preset: enable only pitch actuators 0,1,2,4,6,7,8,9.')
    parser.add_argument('--rh5dg2-gain', type=float, default=1.0, help='RH5DG2 safe raw command gain from baseline toward retarget target.')
    parser.add_argument('--rh5dg2-raw-close-direction', type=float, default=-1.0, help='RH5DG2 safe raw close direction; use -1 if raw pitch closes in the opposite direction.')
    parser.add_argument('--rh5dg2-safe-baseline', type=str, default='demo_open', help='RH5DG2 safe raw open baseline: demo_open, current, or 13 comma-separated angleSet values.')
    parser.add_argument('--rh5dg2-restore-repeat', type=int, default=80, help='RH5DG2 init-pose restore publish count on exit.')
    parser.add_argument('--rh5dg2-restore-interval', type=float, default=0.1, help='RH5DG2 init-pose restore publish interval in seconds.')
    parser.add_argument('--rh5dg2-restore-settle', type=float, default=0.75, help='Extra wait after RH5DG2 init-pose restore publishes.')
    parser.add_argument('--rh5dg2-curl-scale', type=float, default=1.2, help='Global RH5DG2 landmark curl multiplier before clipping.')
    parser.add_argument('--rh5dg2-index-curl-scale', type=float, default=1.8, help='Additional RH5DG2 index-finger curl multiplier before clipping.')
    parser.add_argument('--rh5dg2-enable-thumb', action=argparse.BooleanOptionalAction, default=True, help='Enable RH5DG2 safe thumb actuators 10,11,12.')
    parser.add_argument('--rh5dg2-thumb-source', type=str, choices=['curl', 'raw'], default='raw', help='RH5DG2 safe thumb close-ratio source.')
    parser.add_argument('--rh5dg2-thumb10-scale', type=float, default=1.5, help='RH5DG2 thumb actuator 10 curl scale.')
    parser.add_argument('--rh5dg2-thumb11-scale', type=float, default=1.0, help='RH5DG2 thumb actuator 11 curl scale.')
    parser.add_argument('--rh5dg2-thumb12-scale', type=float, default=1.5, help='RH5DG2 thumb actuator 12 curl scale.')
    parser.add_argument('--rh56f1-tactile-port', type=str, default=None, help='Shortcut for a single RH56F1 tactile serial port; records as right_ee.')
    parser.add_argument('--rh56f1-tactile-left-port', type=str, default=None, help='RH56F1 left_ee tactile serial port.')
    parser.add_argument('--rh56f1-tactile-right-port', type=str, default=None, help='RH56F1 right_ee tactile serial port.')
    parser.add_argument('--rh56f1-tactile-baudrate', type=int, default=115200, help='RH56F1 tactile serial baudrate.')
    parser.add_argument('--rh56f1-tactile-id', type=int, default=1, help='RH56F1 tactile serial hand ID.')
    parser.add_argument('--rh56f1-tactile-hz', type=float, default=30.0, help='RH56F1 tactile polling frequency.')
    parser.add_argument('--rh56f1-tactile-debug-rate', type=float, default=0.0, help='RH56F1 tactile debug log rate in Hz. Set 0 to disable.')
    parser.add_argument('--rh5dg2-tactile-udp-port', type=int, default=9105, help='UDP port for robot-side RH5DG2 tactile JSON packets. Set 0 to disable.')
    parser.add_argument('--rh5dg2-tactile-udp-timeout', type=float, default=1.0, help='Seconds before the latest RH5DG2 UDP tactile packet is marked stale.')
    parser.add_argument('--rh5dg2-tactile-debug-rate', type=float, default=0.0, help='RH5DG2 tactile UDP debug log rate in Hz. Set 0 to disable.')
    parser.add_argument('--enable-rh5dg2-tactile-vr-overlay', action=argparse.BooleanOptionalAction, default=True, help='Show an RH5DG2 tactile heat HUD over the Vuer camera image.')
    parser.add_argument('--rh5dg2-tactile-vr-side', type=str, default='both', help='RH5DG2 tactile side to visualize in Vuer: right_ee, left_ee, both, or left_ee,right_ee.')
    parser.add_argument('--rh5dg2-tactile-vr-baseline-seconds', type=float, default=0.0, help='Quiet baseline duration for Vuer RH5DG2 tactile heat.')
    parser.add_argument('--rh5dg2-tactile-vr-ema-alpha', type=float, default=0.25, help='EMA alpha for Vuer RH5DG2 tactile heat.')
    parser.add_argument('--rh5dg2-tactile-vr-deadband', type=float, default=1.0, help='Baseline-corrected deadband for Vuer RH5DG2 tactile heat.')
    parser.add_argument('--rh5dg2-tactile-vr-normal-max', type=float, default=2500.0, help='Raw normal force value mapped to full Vuer heat.')
    parser.add_argument('--rh5dg2-tactile-vr-tangent-max', type=float, default=2500.0, help='Raw tangential force value mapped to full Vuer heat.')
    parser.add_argument('--rh5dg2-tactile-vr-proximity-max', type=float, default=65535.0, help='Raw finger proximity value mapped to full Vuer heat.')
    parser.add_argument('--rh5dg2-tactile-vr-proximity-weight', type=float, default=0.65, help='Finger proximity contribution weight for Vuer RH5DG2 tactile heat.')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--affinity', action = 'store_true', help = 'Enable high priority and set CPU affinity mode')
    # record mode and task info
    parser.add_argument('--record', action=argparse.BooleanOptionalAction, default=True, help='Enable data recording mode')
    parser.add_argument('--screen-record', action='store_true', help='Record only the head camera view to MP4; toggle with s.')
    parser.add_argument('--screen-record-dir', type=str, default='./screen_records', help='Directory for head camera MP4 recordings.')
    parser.add_argument('--record-body-state', action=argparse.BooleanOptionalAction, default=True, help='Record full robot/body qpos from arm controller lowstate when available.')
    parser.add_argument('--enable-audio', action='store_true', help='Record host-side microphone audio continuously into episode_xxxx/audios/audio.wav.')
    parser.add_argument('--audio-device', type=str, default='plughw:2,0', help='ALSA/sounddevice input device for host-side continuous audio.')
    parser.add_argument('--audio-sample-rate', type=int, default=48000, help='Host-side continuous audio sample rate in Hz.')
    parser.add_argument('--audio-channels', type=int, default=1, help='Host-side continuous audio channel count.')
    parser.add_argument('--audio-dtype', type=str, choices=['int16'], default='int16', help='Host-side continuous audio sample dtype.')
    parser.add_argument('--audio-chunk-size', type=int, default=1024, help='Host-side continuous audio callback/read chunk size in samples.')
    parser.add_argument('--audio-required', action='store_true', help='Abort starting an episode if host-side continuous audio cannot start.')
    parser.add_argument('--record-audio-udp-port', type=int, default=0, help='UDP port for recording int16 PCM audio packets. Set 0 to disable.')
    parser.add_argument('--record-audio-timeout', type=float, default=1.0, help='Seconds before the latest UDP audio packet is considered unavailable.')
    parser.add_argument(
        '--task-dir',
        type=str,
        default=os.path.join(current_dir, 'utils', 'data'),
        help='path to save data (default: teleop/utils/data, independent of current working directory)',
    )
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')

    args = parser.parse_args()
    if args.viewer_display_fps <= 0:
        raise ValueError("--viewer-display-fps must be greater than zero.")
    if not 1 <= args.viewer_jpeg_quality <= 100:
        raise ValueError("--viewer-jpeg-quality must be between 1 and 100.")
    if not 0.0 <= args.neck_smoothing_alpha <= 1.0:
        raise ValueError("--neck-smoothing-alpha must be between 0 and 1.")
    if args.neck_yaw_limit <= 0.0 or args.neck_pitch_limit <= 0.0:
        raise ValueError("--neck-yaw-limit and --neck-pitch-limit must be greater than zero.")
    if args.neck_max_step < 0.0:
        raise ValueError("--neck-max-step must be zero or greater.")
    if args.neck_log_rate < 0.0:
        raise ValueError("--neck-log-rate must be zero or greater.")
    if args.record_audio_udp_port < 0:
        raise ValueError("--record-audio-udp-port must be zero or greater.")
    if args.record_audio_timeout <= 0.0:
        raise ValueError("--record-audio-timeout must be greater than zero.")
    if args.audio_required and not args.enable_audio:
        raise ValueError("--audio-required requires --enable-audio.")
    if args.audio_sample_rate <= 0:
        raise ValueError("--audio-sample-rate must be greater than zero.")
    if args.audio_channels <= 0:
        raise ValueError("--audio-channels must be greater than zero.")
    if args.audio_chunk_size <= 0:
        raise ValueError("--audio-chunk-size must be greater than zero.")
    if args.rh56f1_tactile_port and not args.rh56f1_tactile_right_port:
        args.rh56f1_tactile_right_port = args.rh56f1_tactile_port
    if args.ee == "inspire_dg2":
        if args.inspire_dg2_port:
            if not args.inspire_dg2_left_port:
                args.inspire_dg2_left_port = args.inspire_dg2_port
            if not args.inspire_dg2_right_port:
                args.inspire_dg2_right_port = args.inspire_dg2_port
        if args.inspire_dg2_baudrate <= 0:
            raise ValueError("--inspire-dg2-baudrate must be greater than zero.")
        if args.inspire_dg2_transport != "dds" and args.inspire_dg2_bridge_port <= 0:
            raise ValueError("--inspire-dg2-bridge-port must be greater than zero.")
        if args.inspire_dg2_state_hz <= 0.0 or args.inspire_dg2_tactile_hz <= 0.0:
            raise ValueError("--inspire-dg2-state-hz and --inspire-dg2-tactile-hz must be greater than zero.")
    if args.enable_waist_follow_neck:
        if not args.enable_neck:
            raise ValueError("--enable-waist-follow-neck requires --enable-neck.")
        if args.arm != "H1_2" or args.disable_arm:
            raise ValueError("--enable-waist-follow-neck requires --arm H1_2 with arm control enabled.")
        if args.motion:
            logger_mp.warning(
                "[teleop waist] --motion enabled: publishing H1_2 waist yaw through rt/arm_sdk. "
                "Some motion-service firmware versions may ignore non-arm joint targets."
            )
    if args.waist_yaw_limit <= 0.0 or args.waist_yaw_velocity <= 0.0:
        raise ValueError("--waist-yaw-limit and --waist-yaw-velocity must be greater than zero.")
    if args.record and args.screen_record:
        raise ValueError("--record and --screen-record cannot be enabled together.")
    if args.ee == "rh56f1" and args.arm != "H1_2":
        raise ValueError("--ee rh56f1 currently supports the H1_2 arm path only; use --arm H1_2.")
    rh5dg2_safe_baseline = _resolve_rh5dg2_safe_baseline(args.rh5dg2_safe_baseline)
    if args.hand_only:
        args.disable_arm = True
        args.disable_body = True
    preserve_zero_ready_mode = args.disable_arm
    rh5dg2_enabled_indices = _parse_int_list(args.rh5dg2_enabled_indices)
    if args.rh5dg2_pitch_only:
        rh5dg2_enabled_indices = [0, 1, 2, 4, 6, 7, 8, 9]
    if args.rh5dg2_enable_thumb:
        if rh5dg2_enabled_indices is None:
            rh5dg2_enabled_indices = [0, 1, 2, 4, 6, 7, 8, 9]
        rh5dg2_enabled_indices = sorted(set(rh5dg2_enabled_indices) | {10, 11, 12})
    if (
        args.ee in ("rh5dg2_dfx", "rh5dg2_ftp")
        and not args.sim
        and (args.hand_only or args.rh5dg2_pitch_only or rh5dg2_enabled_indices is not None)
    ):
        args.rh5dg2_safe_mode = True
    if args.rh5dg2_fast_mode:
        if args.rh5dg2_log_throttle == 1.0:
            args.rh5dg2_log_throttle = 2.0
        if args.hand_debug_rate == 1.0:
            args.hand_debug_rate = 0.5
    logger_mp.debug(f"args: {args}")
    if args.hand_only or args.disable_arm or args.disable_body:
        logger_mp.warning(
            "[teleop safety mode] hand_only=%s disable_arm=%s disable_body=%s motion_requested=%s",
            args.hand_only,
            args.disable_arm,
            args.disable_body,
            args.motion,
        )
    logger_mp.info(
        "[teleop zero-ready policy] preserve_zero_ready_mode=%s reason=%s",
        preserve_zero_ready_mode,
        "arm disabled" if preserve_zero_ready_mode else "arm enabled; allow debug-mode handoff for arm teleop init",
    )
    if args.ee in ("rh5dg2_dfx", "rh5dg2_ftp") and args.rh5dg2_safe_mode:
        logger_mp.warning("RH5DG2 SAFE MODE")
        logger_mp.warning("enabled actuators: %s", " ".join(str(i) for i in (rh5dg2_enabled_indices or [0, 1, 2, 4, 6, 7, 8, 9])))
        logger_mp.warning("disabled actuators: %s", " ".join(str(i) for i in sorted(set(range(13)) - set(rh5dg2_enabled_indices or [0, 1, 2, 4, 6, 7, 8, 9]))))
        logger_mp.warning("arm control: %s", "OFF" if args.disable_arm else "ON")
        logger_mp.warning("body control: %s", "OFF" if args.disable_body else "ON")
        logger_mp.warning("gain: %.3f", args.rh5dg2_gain)
        logger_mp.warning("active hand: %s", args.rh5dg2_active_hand)
        logger_mp.warning(
            "curl scale: global=%.3f index=%.3f",
            args.rh5dg2_curl_scale,
            args.rh5dg2_index_curl_scale,
        )
        logger_mp.warning(
            "thumb: enabled=%s source=%s scales={10: %.3f, 11: %.3f, 12: %.3f}",
            args.rh5dg2_enable_thumb,
            args.rh5dg2_thumb_source,
            args.rh5dg2_thumb10_scale,
            args.rh5dg2_thumb11_scale,
            args.rh5dg2_thumb12_scale,
        )
        logger_mp.warning(
            "restore: repeat=%s interval=%.3f settle=%.3f",
            args.rh5dg2_restore_repeat,
            args.rh5dg2_restore_interval,
            args.rh5dg2_restore_settle,
        )
        logger_mp.warning(
            "safe baseline: %s",
            "startup current_state" if rh5dg2_safe_baseline is None else rh5dg2_safe_baseline,
        )

    arm_ctrl = None
    arm_ik = None
    hand_ctrl = None
    img_client = None
    tv_wrapper = None
    ipc_server = None
    listen_keyboard_thread = None
    screen_record_writer = None
    screen_record_path = None
    recorder = None
    sim_state_subscriber = None
    neck_ctrl = None
    neck_feedback = None
    rh56f1_tactile_reader = None
    rh5dg2_tactile_udp = None
    rh5dg2_tactile_heat_mappers = {}
    audio_udp_receiver = None
    episode_audio_recorder = None

    try:
        # setup dds communication domains id
        if args.sim:
            ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        else:
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)
        if args.enable_neck:
            neck_ctrl = VisionProNeckController(
                host=args.neck_host or args.img_server_ip,
                port=args.neck_port,
                yaw_limit=args.neck_yaw_limit,
                pitch_limit=args.neck_pitch_limit,
                smoothing_alpha=args.neck_smoothing_alpha,
                max_step=args.neck_max_step,
            )
            logger_mp.info(
                f"[teleop neck] enabled target={args.neck_host or args.img_server_ip}:{args.neck_port} "
                f"yaw_limit={args.neck_yaw_limit:.3f} pitch_limit={args.neck_pitch_limit:.3f} "
                f"alpha={args.neck_smoothing_alpha:.3f} max_step={args.neck_max_step:.3f}"
            )
            try:
                neck_feedback = NeckFeedbackReceiver(args.neck_feedback_port)
                logger_mp.info(
                    f"[teleop neck feedback] listening udp=0.0.0.0:{args.neck_feedback_port}"
                )
            except OSError as e:
                neck_feedback = None
                logger_mp.warning(f"[teleop neck feedback] disabled: {e}")

        if args.rh5dg2_tactile_udp_port > 0:
            try:
                rh5dg2_tactile_udp = RH5DG2TactileUDPReceiver(
                    port=args.rh5dg2_tactile_udp_port,
                    timeout=args.rh5dg2_tactile_udp_timeout,
                    debug_rate=args.rh5dg2_tactile_debug_rate,
                )
                logger_mp.info(
                    f"[RH5DG2 tactile UDP] listening udp=0.0.0.0:{args.rh5dg2_tactile_udp_port} "
                    f"timeout={args.rh5dg2_tactile_udp_timeout:.3f}s"
                )
            except OSError as e:
                rh5dg2_tactile_udp = None
                logger_mp.warning(f"[RH5DG2 tactile UDP] disabled: {e}")

        if args.enable_rh5dg2_tactile_vr_overlay:
            if args.input_mode != "hand":
                logger_mp.warning("[RH5DG2 tactile VR overlay] disabled: hand tracking input is required.")
            elif rh5dg2_tactile_udp is None:
                logger_mp.warning("[RH5DG2 tactile VR overlay] disabled: --rh5dg2-tactile-udp-port is not active.")
            else:
                rh5dg2_tactile_vr_sides = _resolve_rh5dg2_tactile_vr_sides(args.rh5dg2_tactile_vr_side)
                rh5dg2_tactile_heat_mappers = {
                    side: RH5DG2TactileHeatMapper(
                        side=side,
                        baseline_seconds=args.rh5dg2_tactile_vr_baseline_seconds,
                        ema_alpha=args.rh5dg2_tactile_vr_ema_alpha,
                        deadband=args.rh5dg2_tactile_vr_deadband,
                        normal_max=args.rh5dg2_tactile_vr_normal_max,
                        tangent_max=args.rh5dg2_tactile_vr_tangent_max,
                        proximity_max=args.rh5dg2_tactile_vr_proximity_max,
                        proximity_weight=args.rh5dg2_tactile_vr_proximity_weight,
                    )
                    for side in rh5dg2_tactile_vr_sides
                }
                logger_mp.info(
                    "[RH5DG2 tactile VR overlay] enabled sides=%s baseline=%.3fs ema_alpha=%.3f",
                    ",".join(rh5dg2_tactile_vr_sides),
                    args.rh5dg2_tactile_vr_baseline_seconds,
                    args.rh5dg2_tactile_vr_ema_alpha,
                )

        if args.record and args.record_audio_udp_port > 0:
            try:
                audio_udp_receiver = AudioUDPReceiver(
                    port=args.record_audio_udp_port,
                    timeout=args.record_audio_timeout,
                )
                logger_mp.info(
                    f"[record audio UDP] listening udp=0.0.0.0:{args.record_audio_udp_port} "
                    f"timeout={args.record_audio_timeout:.3f}s format=int16 PCM"
                )
            except OSError as e:
                audio_udp_receiver = None
                logger_mp.warning(f"[record audio UDP] disabled: {e}")

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press,get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication mode
        else:
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                                      kwargs={"on_press": on_press, "until": None, "sequential": False,}, 
                                                      daemon=True)
            listen_keyboard_thread.start()

        # image client
        img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
        camera_config = img_client.get_cam_config()
        selected_camera_name = "head" if args.camera == "all" else args.camera
        if selected_camera_name == "both":
            selected_camera_name = "both_wrist"
        if selected_camera_name == "head_wrist":
            selected_camera_name = "head_and_wrist"
        selected_both_wrist = _is_both_wrist_camera(selected_camera_name)
        selected_composite_wrist = _is_composite_wrist_camera(selected_camera_name)
        if selected_composite_wrist:
            head_config = camera_config["head_camera"]
            left_wrist_config = camera_config["left_wrist_camera"]
            right_wrist_config = camera_config["right_wrist_camera"]
            both_wrist_zmq = bool(left_wrist_config.get("enable_zmq")) and bool(right_wrist_config.get("enable_zmq"))
            composite_zmq = both_wrist_zmq if selected_both_wrist else bool(head_config.get("enable_zmq"))
            if selected_both_wrist:
                configured_shape = tuple(left_wrist_config["image_shape"][:1]) + (
                    left_wrist_config["image_shape"][1] + right_wrist_config["image_shape"][1],
                )
            else:
                wrist_width = left_wrist_config["image_shape"][1] + right_wrist_config["image_shape"][1]
                configured_shape = (
                    head_config["image_shape"][0] + left_wrist_config["image_shape"][0],
                    max(head_config["image_shape"][1], wrist_width),
                )
            selected_camera_key = f"{selected_camera_name}_camera"
            selected_camera_config = {
                "binocular": False,
                "enable_zmq": composite_zmq,
                "enable_webrtc": False,
                "fps": min(
                    head_config.get("fps", 0) if not selected_both_wrist else left_wrist_config.get("fps", 0),
                    left_wrist_config.get("fps", 0),
                    right_wrist_config.get("fps", 0),
                ),
                "image_shape": configured_shape,
                "zmq_port": None,
                "webrtc_port": None,
            }
            if args.viewer_camera_mode == "webrtc":
                logger_mp.warning(
                    f"[teleop camera selected] {selected_camera_name} viewer requires ZMQ because the frame is composed locally; forcing --viewer-camera-mode=zmq."
                )
                args.viewer_camera_mode = "zmq"
        else:
            selected_camera_key = _camera_config_key(selected_camera_name)
            selected_camera_config = camera_config[selected_camera_key]
        logger_mp.info(
            f"[teleop image client] created host={args.img_server_ip} "
            f"request_bgr=True selected_camera={selected_camera_name}"
        )
        logger_mp.info(
            f"[teleop camera config] img_server_ip={args.img_server_ip} "
            f"display_mode={args.display_mode} head={camera_config['head_camera']} "
            f"left_wrist={camera_config['left_wrist_camera']} "
            f"right_wrist={camera_config['right_wrist_camera']}"
        )
        _log_camera_reachability(args.img_server_ip, camera_config)
        viewer_webrtc, viewer_zmq, viewer_route = _select_viewer_camera_route(
            args.display_mode,
            args.viewer_camera_mode,
            selected_camera_config,
        )
        if args.display_mode == "pass-through":
            logger_mp.warning(
                "[teleop camera route] display_mode=pass-through does not render robot camera frames in Vuer. "
                "Use --display-mode=immersive for full ZED view or --display-mode=ego for pass-through plus a camera window."
            )
        if (
            selected_camera_name == "left_wrist"
            and args.left_wrist_camera_vflip
            and viewer_webrtc
        ):
            if selected_camera_config.get("enable_zmq"):
                viewer_webrtc = False
                viewer_zmq = True
                viewer_route = "zmq"
                logger_mp.info(
                    "[teleop camera orientation] selected_camera=left_wrist "
                    "vertical_flip=True; forcing viewer route to ZMQ because WebRTC planes cannot be pixel-flipped in Python."
                )
            else:
                logger_mp.warning(
                    "[teleop camera orientation] selected_camera=left_wrist vertical_flip=True "
                    "but ZMQ is disabled; WebRTC viewer may remain vertically inverted."
                )
        xr_need_local_img = args.display_mode != 'pass-through' and viewer_zmq
        viewer_host_ip = args.viewer_host_ip or _local_ip_for_remote(args.img_server_ip)
        viewer_host_source = "explicit" if args.viewer_host_ip else "route_to_img_server"
        viewer_url = f"https://{viewer_host_ip}:8012/?ws=wss://{viewer_host_ip}:8012"
        selected_webrtc_port = selected_camera_config.get("webrtc_port")
        selected_zmq_port = selected_camera_config.get("zmq_port")
        webrtc_offer_url = f"https://{args.img_server_ip}:{selected_webrtc_port}/offer" if selected_webrtc_port else None
        selected_img_shape = tuple(selected_camera_config['image_shape'])
        if xr_need_local_img:
            if selected_camera_name == "head_and_wrist":
                sample_bgr, head_sample, left_sample, right_sample = _get_head_and_wrist_bgr(img_client, args)
                sample_debug = f"head={_frame_debug(head_sample)} left={_frame_debug(left_sample)} right={_frame_debug(right_sample)}"
            elif selected_both_wrist:
                sample_bgr, left_sample, right_sample = _get_both_wrist_bgr(img_client, args)
                sample_debug = f"left={_frame_debug(left_sample)} right={_frame_debug(right_sample)}"
            else:
                sample_frame = _sample_camera_frame(img_client, selected_camera_name)
                sample_bgr = getattr(sample_frame, "bgr", None)
                sample_debug = _frame_debug(sample_frame)
            logger_mp.info(
                f"[teleop camera sample] camera_name={selected_camera_name} "
                f"configured_shape={selected_img_shape} binocular={selected_camera_config.get('binocular')} "
                f"{sample_debug}"
            )
            if sample_bgr is not None:
                actual_img_shape = tuple(sample_bgr.shape[:2])
                if actual_img_shape != selected_img_shape and not selected_composite_wrist:
                    logger_mp.warning(
                        f"[teleop camera shape] camera_name={selected_camera_name} "
                        f"configured_shape={selected_img_shape} actual_shape={actual_img_shape}; "
                        "using actual_shape for Vuer ZMQ shared image buffer."
                    )
                    selected_img_shape = actual_img_shape
                elif actual_img_shape != selected_img_shape:
                    logger_mp.warning(
                        f"[teleop camera shape] camera_name={selected_camera_name} "
                        f"configured_shape={selected_img_shape} sample_shape={actual_img_shape}; "
                        "keeping configured_shape and resizing composed frames for Vuer."
                    )
        logger_mp.info(
            f"[teleop camera selected] requested_camera={args.camera} "
            f"selected_camera={selected_camera_name} selected_key={selected_camera_key} "
            f"all_mode_displays=head_only={args.camera == 'all'} "
            f"left_wrist_vflip={args.left_wrist_camera_vflip} right_wrist_vflip={args.right_wrist_camera_vflip}"
        )
        for camera_name in ("head", "left_wrist", "right_wrist"):
            camera = camera_config[_camera_config_key(camera_name)]
            webrtc_port = camera.get("webrtc_port")
            url = f"https://{args.img_server_ip}:{webrtc_port}/offer" if webrtc_port else None
            reachable = False
            detail = "missing_port"
            if webrtc_port:
                reachable, detail = _tcp_check(args.img_server_ip, webrtc_port)
            logger_mp.info(
                f"[teleop camera stream] camera_name={camera_name} "
                f"webrtc_url={url} reachable={reachable} detail={detail} "
                f"fps={camera.get('fps')} zmq={camera.get('enable_zmq')} "
                f"webrtc={camera.get('enable_webrtc')}"
            )
        if args.camera == "all":
            logger_mp.warning("[teleop camera selected] --camera=all logs all streams but displays head camera in the current 8012 viewer. Use --viewer-camera head_and_wrist to display head and wrist camera views together.")
        logger_mp.info(
            f"[teleop viewer 8012] url={viewer_url} bind=0.0.0.0:8012 "
            f"host_source={viewer_host_source} "
            f"display_mode={args.display_mode} requested_camera_mode={args.viewer_camera_mode} "
            f"selected_camera_mode={viewer_route} selected_camera={selected_camera_name} "
            f"display_fps={args.viewer_display_fps:.1f} jpeg_quality={args.viewer_jpeg_quality}"
        )
        if args.display_mode in ("immersive", "ego"):
            if viewer_webrtc:
                logger_mp.info(
                    f"[teleop camera route] mode=webrtc url={webrtc_offer_url} "
                    f"viewer_url={viewer_url} selected_camera={selected_camera_name}"
                )
            elif viewer_zmq:
                logger_mp.info(
                    f"[teleop camera route] mode=zmq host={args.img_server_ip} "
                    f"port={selected_zmq_port} viewer_url={viewer_url} selected_camera={selected_camera_name}"
                )
            else:
                logger_mp.warning(f"[teleop camera route] immersive/ego requested but selected camera {selected_camera_name} has no ZMQ/WebRTC enabled.")
            logger_mp.info(
                f"[teleop viewer stream bind] selected_camera={selected_camera_name} mode={viewer_route} "
                f"webrtc_url={webrtc_offer_url if viewer_webrtc else None} "
                f"zmq={args.img_server_ip}:{selected_zmq_port if viewer_zmq else None}"
            )

        # televuer_wrapper: obtain hand pose data from the XR device and transmit the selected camera image to the XR device.
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand", 
                                     binocular=selected_camera_config['binocular'],
                                     img_shape=selected_img_shape,
                                     display_fps=args.viewer_display_fps,
                                     jpeg_quality=args.viewer_jpeg_quality,
                                     display_mode=args.display_mode,
                                     zmq=viewer_zmq,
                                     webrtc=viewer_webrtc,
                                     webrtc_url=webrtc_offer_url,
                                     arm_reference_mode="head_yaw",
                                     )
        
        # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        if preserve_zero_ready_mode:
            motion_switcher = None
            logger_mp.warning(
                "[teleop body control] OFF: skip MotionSwitcher.Enter_Debug_Mode because arm control is disabled."
            )
        elif args.motion:
            if args.input_mode == "controller":
                loco_wrapper = LocoClientWrapper()
        else:
            motion_switcher = MotionSwitcher()
            status, result = motion_switcher.Enter_Debug_Mode()
            logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

        # arm
        if args.disable_arm:
            logger_mp.warning("[teleop arm control] OFF: arm controller and IK are not initialized.")
        elif args.arm == "G1_29":
            arm_ik = G1_29_ArmIK()
            arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "G1_23":
            arm_ik = G1_23_ArmIK()
            arm_ctrl = G1_23_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1_2":
            arm_ik = H1_2_ArmIK()
            arm_ctrl = H1_2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1":
            arm_ik = H1_ArmIK()
            arm_ctrl = H1_ArmController(simulation_mode=args.sim)
        elif args.arm == "H2":
            arm_ik = H2_ArmIK()
            arm_ctrl = H2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)

        # end-effector
        if args.ee == "dex3":
            from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
            hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                          dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "dex1":
            from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, 
                                                     dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_dfx":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX, Inspire_Num_Motors
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', Inspire_Num_Motors * 2, lock = False)   # [output] current left, right hand state data.
            dual_hand_action_array = Array('d', Inspire_Num_Motors * 2, lock = False)  # [output] current left, right hand action data.
            hand_ctrl = Inspire_Controller_DFX(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_ftp":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_FTP, Inspire_Num_Motors
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', Inspire_Num_Motors * 2, lock = False)   # [output] current left, right hand state data.
            dual_hand_action_array = Array('d', Inspire_Num_Motors * 2, lock = False)  # [output] current left, right hand action data.
            hand_ctrl = Inspire_Controller_FTP(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "brainco":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller, brainco_Num_Motors
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', brainco_Num_Motors * 2, lock = False)   # [output] current left, right hand state data.
            dual_hand_action_array = Array('d', brainco_Num_Motors * 2, lock = False)  # [output] current left, right hand action data.
            hand_ctrl = Brainco_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                           dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "rh5dg2_dfx":
            from teleop.robot_control.robot_hand_RH5DG2 import RH5DG2_Controller_DFX, RH5DG2_Num_Motors
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            hand_input_timestamp = Value('d', 0.0, lock=True)
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', RH5DG2_Num_Motors * 2, lock = False)
            dual_hand_action_array = Array('d', RH5DG2_Num_Motors * 2, lock = False)
            hand_ctrl = RH5DG2_Controller_DFX(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock,
                                              dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim,
                                              network_interface=args.network_interface, fps=args.hand_control_hz,
                                              input_timestamp_value=hand_input_timestamp,
                                              log_throttle_s=args.rh5dg2_log_throttle,
                                              fast_mode=args.rh5dg2_fast_mode,
                                              safe_mode=args.rh5dg2_safe_mode,
                                              enabled_indices=rh5dg2_enabled_indices,
                                              pitch_only=args.rh5dg2_pitch_only,
                                              safe_gain=args.rh5dg2_gain,
                                              raw_close_direction=args.rh5dg2_raw_close_direction,
                                              safe_active_hand=args.rh5dg2_active_hand,
                                              safe_baseline=rh5dg2_safe_baseline,
                                              restore_repeat=args.rh5dg2_restore_repeat,
                                              restore_interval_s=args.rh5dg2_restore_interval,
                                              restore_settle_s=args.rh5dg2_restore_settle,
                                              curl_scale=args.rh5dg2_curl_scale,
                                              index_curl_scale=args.rh5dg2_index_curl_scale,
                                              enable_thumb=args.rh5dg2_enable_thumb,
                                              thumb_source=args.rh5dg2_thumb_source,
                                              thumb10_scale=args.rh5dg2_thumb10_scale,
                                              thumb11_scale=args.rh5dg2_thumb11_scale,
                                              thumb12_scale=args.rh5dg2_thumb12_scale,
                                              retarget_mode=args.rh5dg2_retarget_mode)
        elif args.ee == "rh5dg2_ftp":
            from teleop.robot_control.robot_hand_RH5DG2 import RH5DG2_Controller_FTP, RH5DG2_Num_Motors
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            hand_input_timestamp = Value('d', 0.0, lock=True)
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', RH5DG2_Num_Motors * 2, lock = False)
            dual_hand_action_array = Array('d', RH5DG2_Num_Motors * 2, lock = False)
            hand_ctrl = RH5DG2_Controller_FTP(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock,
                                               dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim,
                                               network_interface=args.network_interface, fps=args.hand_control_hz,
                                               input_timestamp_value=hand_input_timestamp,
                                               log_throttle_s=args.rh5dg2_log_throttle,
                                               fast_mode=args.rh5dg2_fast_mode,
                                               retarget_mode=args.rh5dg2_retarget_mode)
        elif args.ee == "inspire_dg2":
            from teleop.robot_control.robot_hand_inspire_dg2 import InspireDG2_Controller, InspireDG2_Num_Motors

            left_hand_pos_array = Array('d', 75, lock=True)
            right_hand_pos_array = Array('d', 75, lock=True)
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', InspireDG2_Num_Motors * 2, lock=False)
            dual_hand_action_array = Array('d', InspireDG2_Num_Motors * 2, lock=False)
            hand_ctrl = InspireDG2_Controller(
                left_hand_pos_array,
                right_hand_pos_array,
                dual_hand_data_lock,
                dual_hand_state_array,
                dual_hand_action_array,
                fps=args.hand_control_hz,
                simulation_mode=args.sim,
                left_port=args.inspire_dg2_left_port,
                right_port=args.inspire_dg2_right_port,
                baudrate=args.inspire_dg2_baudrate,
                left_hand_id=args.inspire_dg2_left_id,
                right_hand_id=args.inspire_dg2_right_id,
                state_hz=args.inspire_dg2_state_hz,
                tactile_hz=args.inspire_dg2_tactile_hz,
                transport=args.inspire_dg2_transport,
                bridge_host=args.inspire_dg2_bridge_host,
                bridge_port=args.inspire_dg2_bridge_port,
                dds_domain_id=1 if args.sim else 0,
                network_interface=args.network_interface,
                fast_mode=args.rh5dg2_fast_mode,
                retarget_mode=args.rh5dg2_retarget_mode,
            )
        elif args.ee == "rh56f1":
            from teleop.robot_control.robot_hand_RH56F1 import (
                RH56F1_Controller,
                RH56F1_Num_Retarget_Joints,
                RH56F1TactileReader,
            )
            rh56f1_record_motors = RH56F1_Num_Retarget_Joints
            left_hand_pos_array = Array('d', 75, lock=True)
            right_hand_pos_array = Array('d', 75, lock=True)
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', rh56f1_record_motors * 2, lock=False)
            dual_hand_action_array = Array('d', rh56f1_record_motors * 2, lock=False)
            hand_ctrl = RH56F1_Controller(
                left_hand_pos_array,
                right_hand_pos_array,
                dual_hand_data_lock,
                dual_hand_state_array,
                dual_hand_action_array,
                fps=args.hand_control_hz,
                simulation_mode=args.sim,
                network_interface=args.network_interface,
                retarget_mode=args.rh56f1_retarget_mode,
            )
            if args.rh56f1_tactile_left_port or args.rh56f1_tactile_right_port:
                rh56f1_tactile_reader = RH56F1TactileReader(
                    left_port=args.rh56f1_tactile_left_port,
                    right_port=args.rh56f1_tactile_right_port,
                    baudrate=args.rh56f1_tactile_baudrate,
                    hand_id=args.rh56f1_tactile_id,
                    fps=args.rh56f1_tactile_hz,
                    debug_rate=args.rh56f1_tactile_debug_rate,
                )
                logger_mp.info(
                    "[teleop RH56F1 tactile] enabled left_port=%s right_port=%s hz=%.2f",
                    args.rh56f1_tactile_left_port,
                    args.rh56f1_tactile_right_port,
                    args.rh56f1_tactile_hz,
                )
        else:
            pass

        if (
            (args.rh56f1_tactile_left_port or args.rh56f1_tactile_right_port)
            and rh56f1_tactile_reader is None
        ):
            from teleop.robot_control.robot_hand_RH56F1 import RH56F1TactileReader

            rh56f1_tactile_reader = RH56F1TactileReader(
                left_port=args.rh56f1_tactile_left_port,
                right_port=args.rh56f1_tactile_right_port,
                baudrate=args.rh56f1_tactile_baudrate,
                hand_id=args.rh56f1_tactile_id,
                fps=args.rh56f1_tactile_hz,
                debug_rate=args.rh56f1_tactile_debug_rate,
            )
            logger_mp.info(
                "[teleop serial tactile] enabled ee=%s left_port=%s right_port=%s hz=%.2f",
                args.ee,
                args.rh56f1_tactile_left_port,
                args.rh56f1_tactile_right_port,
                args.rh56f1_tactile_hz,
            )

        if args.ee in ["dex3", "inspire_dfx", "inspire_ftp", "inspire_dg2", "rh5dg2_dfx", "rh5dg2_ftp", "rh56f1", "brainco"]:
            logger_mp.info(f"[teleop ee] ee={args.ee} hand_controller={hand_ctrl.__class__.__name__}")
        if args.ee in ("rh5dg2_dfx", "rh5dg2_ftp"):
            logger_mp.info(
                f"[teleop hand side mapping] ee={args.ee} "
                f"rh5dg2_hand_swap={args.rh5dg2_hand_swap} "
                f"rh5dg2_fast_mode={args.rh5dg2_fast_mode} "
                f"rh5dg2_retarget_mode={args.rh5dg2_retarget_mode} "
                "scope=hand_landmarks_only arm_wrist_pose_uses_arm_sensitivity=True"
            )
        if args.ee == "inspire_dg2":
            logger_mp.info(
                f"[teleop hand side mapping] ee={args.ee} "
                f"transport={args.inspire_dg2_transport} "
                f"left_port={args.inspire_dg2_left_port} left_id={args.inspire_dg2_left_id} "
                f"right_port={args.inspire_dg2_right_port} right_id={args.inspire_dg2_right_id} "
                "scope=hand_landmarks_only arm_wrist_pose_uses_arm_sensitivity=True"
            )
        if args.ee == "rh56f1":
            logger_mp.info(
                f"[teleop hand side mapping] ee={args.ee} "
                f"rh56f1_retarget_mode={args.rh56f1_retarget_mode} "
                "scope=hand_landmarks_only arm_wrist_pose_uses_arm_sensitivity=True"
            )
        
        # affinity mode (if you dont know what it is, then you probably don't need it)
        if args.affinity:
            import psutil
            p = psutil.Process(os.getpid())
            p.cpu_affinity([0,1,2,3]) # Set CPU affinity to cores 0-3
            try:
                p.nice(-20)           # Set highest priority
                logger_mp.info("Set high priority successfully.")
            except psutil.AccessDenied:
                logger_mp.warning("Failed to set high priority. Please run as root.")
                
            for child in p.children(recursive=True):
                try:
                    logger_mp.info(f"Child process {child.pid} name: {child.name()}")
                    child.cpu_affinity([5,6])
                    child.nice(-20)
                except psutil.AccessDenied:
                    pass

        # simulation mode
        if args.sim:
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        # record + headless / non-headless mode
        if args.record:
            recording_metadata = {
                "robot": {
                    "arm": args.arm,
                    "end_effector": args.ee,
                    "input_mode": args.input_mode,
                    "simulation": args.sim,
                    "disable_arm": args.disable_arm,
                    "disable_body": args.disable_body,
                    "hand_only": args.hand_only,
                    "rh56f1_retarget_mode": args.rh56f1_retarget_mode,
                },
                "camera": {
                    "selected": args.camera,
                    "display_mode": args.display_mode,
                    "viewer_camera_mode": args.viewer_camera_mode,
                    "server_ip": args.img_server_ip,
                },
                "control": {
                    "frequency": args.frequency,
                    "network_interface": args.network_interface,
                    "motion": args.motion,
                },
                "body_state": {
                    "enabled": args.record_body_state,
                    "source": "arm_ctrl.get_current_motor_q",
                },
                "audio": {
                    "enabled": args.record_audio_udp_port > 0,
                    "udp_port": args.record_audio_udp_port,
                    "timeout": args.record_audio_timeout,
                    "format": "int16_pcm",
                },
                "rh56f1_tactile": {
                    "left_port": args.rh56f1_tactile_left_port,
                    "right_port": args.rh56f1_tactile_right_port,
                    "baudrate": args.rh56f1_tactile_baudrate,
                    "hand_id": args.rh56f1_tactile_id,
                    "hz": args.rh56f1_tactile_hz,
                },
                "inspire_dg2": {
                    "left_port": args.inspire_dg2_left_port,
                    "right_port": args.inspire_dg2_right_port,
                    "baudrate": args.inspire_dg2_baudrate,
                    "left_id": args.inspire_dg2_left_id,
                    "right_id": args.inspire_dg2_right_id,
                    "state_hz": args.inspire_dg2_state_hz,
                    "tactile_hz": args.inspire_dg2_tactile_hz,
                    "transport": args.inspire_dg2_transport,
                    "bridge_host": args.inspire_dg2_bridge_host,
                    "bridge_port": args.inspire_dg2_bridge_port,
                },
                "rh5dg2_tactile_udp": {
                    "enabled": args.rh5dg2_tactile_udp_port > 0,
                    "port": args.rh5dg2_tactile_udp_port,
                    "timeout": args.rh5dg2_tactile_udp_timeout,
                    "vr_overlay": {
                        "enabled": bool(rh5dg2_tactile_heat_mappers),
                        "side": args.rh5dg2_tactile_vr_side,
                        "resolved_sides": list(rh5dg2_tactile_heat_mappers.keys()),
                        "baseline_seconds": args.rh5dg2_tactile_vr_baseline_seconds,
                        "ema_alpha": args.rh5dg2_tactile_vr_ema_alpha,
                    },
                },
                "neck": {
                    "enabled": args.enable_neck,
                    "command_host": args.neck_host or args.img_server_ip,
                    "command_port": args.neck_port,
                    "feedback_port": args.neck_feedback_port,
                },
            }
            if args.enable_audio:
                recording_metadata["continuous_audio"] = {
                    "enabled": True,
                    "device": args.audio_device,
                    "sample_rate": args.audio_sample_rate,
                    "channels": args.audio_channels,
                    "dtype": args.audio_dtype,
                    "format": "wav",
                    "path": "audios/audio.wav",
                    "chunk_size": args.audio_chunk_size,
                    "required": args.audio_required,
                }
            recorder = EpisodeWriter(
                task_dir=os.path.abspath(os.path.join(args.task_dir, args.task_name)),
                task_goal=args.task_goal,
                task_desc=args.task_desc,
                task_steps=args.task_steps,
                frequency=args.frequency,
                rerun_log=not args.headless,
                metadata=recording_metadata,
            )
            logger_mp.info(
                f"[teleop record] task_dir={recorder.task_dir} "
                "Press [s] once to create an episode and start writing states; "
                "press [s] again to finalize data.json."
            )

        logger_mp.info("----------------------------------------------------------------")
        logger_mp.info("🟢  Press [r] to start syncing the robot with your movements.")
        if args.record:
            logger_mp.info("🟡  Press [s] to START or SAVE recording (toggle cycle).")
        elif args.screen_record:
            logger_mp.info("🟡  Press [s] to START or STOP head camera screen recording.")
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        if rh5dg2_tactile_heat_mappers:
            logger_mp.info("🟣  Press [t] to SHOW or HIDE the RH5DG2 tactile VR overlay.")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        READY = True                  # now ready to (1) enter START state
        logger_mp.info(f"[teleop ready state] READY={READY} START={START} STOP={STOP} disable_arm={args.disable_arm}")
        while not START and not STOP: # wait for start or stop signal.
            time.sleep(0.033)
            if selected_camera_config.get('enable_zmq') and xr_need_local_img:
                if selected_camera_name == "head_and_wrist":
                    prestart_bgr, head_prestart_img, left_prestart_img, right_prestart_img = _get_head_and_wrist_bgr(img_client, args)
                    prestart_debug = f"head={_frame_debug(head_prestart_img)} left={_frame_debug(left_prestart_img)} right={_frame_debug(right_prestart_img)}"
                elif selected_both_wrist:
                    prestart_bgr, left_prestart_img, right_prestart_img = _get_both_wrist_bgr(img_client, args)
                    prestart_debug = f"left={_frame_debug(left_prestart_img)} right={_frame_debug(right_prestart_img)}"
                else:
                    prestart_img = _get_camera_frame(img_client, selected_camera_name)
                    prestart_bgr = None if prestart_img is None else getattr(prestart_img, "bgr", None)
                    prestart_bgr = _apply_camera_orientation(prestart_bgr, selected_camera_name, args)
                    prestart_debug = _frame_debug(prestart_img)
                if prestart_bgr is not None:
                    prestart_bgr = _fit_bgr_to_shape(prestart_bgr, selected_img_shape)
                    _safe_render_to_xr(tv_wrapper, prestart_bgr, "[teleop camera prestart]")
                else:
                    logger_mp.warning(f"[teleop camera prestart] no {selected_camera_name} frame received for XR display. {prestart_debug}")

        logger_mp.info("---------------------🚀start Tracking🚀-------------------------")
        logger_mp.info(f"[teleop ready state] READY={READY} START={START} STOP={STOP} disable_arm={args.disable_arm}")
        if args.disable_arm:
            logger_mp.warning("[teleop arm disabled reason] --disable-arm set; IK/control publish will be skipped.")
        else:
            arm_ctrl.speed_gradual_max()
            logger_mp.info(
                f"[teleop arm startup safety] reset wrist base at START; "
                f"startup_duration={args.arm_startup_duration:.3f}s "
                f"startup_max_step={args.arm_startup_max_step:.4f}rad"
            )

        head_img = None
        left_wrist_img = None
        right_wrist_img = None
        left_wrist_bgr = None
        right_wrist_bgr = None
        record_save_pending = False
        viewer_frame_count = 0
        hand_input_count = 0
        hand_input_rate_start = time.time()
        hand_debug_last_ts = 0.0
        hand_debug_interval = 1.0 / args.hand_debug_rate if args.hand_debug_rate > 0 else None
        arm_sensitivity_config = _resolve_arm_sensitivity_args(args)
        arm_sensitivity_state = {}
        tracking_start_time = time.time()
        logger_mp.info(
            f"[teleop arm sensitivity config] pos_gain={np.round(arm_sensitivity_config['pos_gain'], 4).tolist()} "
            f"rot_gain={arm_sensitivity_config['rot_gain']:.3f} "
            f"max_delta={arm_sensitivity_config['max_delta']:.3f} "
            f"smoothing_alpha={arm_sensitivity_config['smoothing_alpha']:.3f} "
            f"enabled={arm_sensitivity_config.get('enabled', False)} sim={args.sim}"
        )

        # main loop. robot start to follow VR user's motion
        loop_count = 0
        neck_log_last_ts = 0.0
        neck_log_interval = 1.0 / args.neck_log_rate if args.neck_log_rate > 0 else None
        while not STOP:
            loop_count += 1
            start_time = time.time()
            neck_record = None
            left_wrist_bgr = None
            right_wrist_bgr = None
            # get image
            if camera_config['head_camera']['enable_zmq']:
                if args.record or args.screen_record or (xr_need_local_img and selected_camera_name == "head"):
                    head_img = img_client.get_head_frame()
            if xr_need_local_img:
                if selected_camera_name == "head_and_wrist":
                    viewer_bgr, head_viewer_frame, left_viewer_frame, right_viewer_frame = _get_head_and_wrist_bgr(img_client, args)
                    viewer_frame = head_viewer_frame
                    viewer_frame_debug = f"head={_frame_debug(head_viewer_frame)} left={_frame_debug(left_viewer_frame)} right={_frame_debug(right_viewer_frame)}"
                elif selected_both_wrist:
                    viewer_bgr, left_viewer_frame, right_viewer_frame = _get_both_wrist_bgr(img_client, args)
                    viewer_frame = None
                    viewer_frame_debug = f"left={_frame_debug(left_viewer_frame)} right={_frame_debug(right_viewer_frame)}"
                else:
                    viewer_frame = head_img if selected_camera_name == "head" else _get_camera_frame(img_client, selected_camera_name)
                    viewer_bgr = None if viewer_frame is None else getattr(viewer_frame, "bgr", None)
                    viewer_bgr = _apply_camera_orientation(viewer_bgr, selected_camera_name, args)
                    viewer_frame_debug = _frame_debug(viewer_frame)
                viewer_frame_count += 1
                frame_timestamp = time.time()
                if viewer_bgr is not None:
                    viewer_bgr = _fit_bgr_to_shape(viewer_bgr, selected_img_shape)
                    _safe_render_to_xr(tv_wrapper, viewer_bgr, "[teleop camera loop]")
                    if loop_count % 50 == 0:
                        logger_mp.debug(
                            f"[teleop viewer frame] camera_name={selected_camera_name} "
                            f"received_frame_count={viewer_frame_count} frame_timestamp={frame_timestamp:.6f} "
                            f"fps={getattr(viewer_frame, 'fps', None)} "
                            f"image_shape={viewer_bgr.shape} configured_shape={selected_img_shape} "
                            f"binocular={selected_camera_config.get('binocular')} "
                            f"route={viewer_route} display_mode={args.display_mode}"
                        )
                        logger_mp.debug(
                            f"[teleop viewer latency] camera_name={selected_camera_name} "
                            f"latency_ms={(time.time() - frame_timestamp) * 1000.0:.2f} "
                            "source=local_receive_timestamp"
                        )
                elif loop_count % 50 == 0:
                    logger_mp.warning(
                        f"[teleop camera loop] no frame received for XR display "
                        f"camera_name={selected_camera_name} {viewer_frame_debug}"
                    )
            if camera_config['left_wrist_camera']['enable_zmq']:
                if args.record:
                    left_wrist_img = img_client.get_left_wrist_frame()
                    if left_wrist_img is not None and left_wrist_img.bgr is not None and cv2 is not None:
                        # Store the camera frame exactly as received from the image server.
                        left_wrist_bgr = left_wrist_img.bgr
            
            if camera_config['right_wrist_camera']['enable_zmq']:
                if args.record:
                    right_wrist_img = img_client.get_right_wrist_frame()
                    if right_wrist_img is not None and right_wrist_img.bgr is not None and cv2 is not None:
                        # Store the camera frame exactly as received from the image server.
                        right_wrist_bgr = right_wrist_img.bgr

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if recorder.create_episode():
                        try:
                            episode_audio_recorder = _start_episode_audio(args, recorder)
                            RECORD_RUNNING = True
                            logger_mp.info(
                                f"[teleop record] START data_json={recorder.json_path} "
                                f"robot={args.arm}/{args.ee}"
                            )
                        except AudioRecorderError as exc:
                            episode_audio_recorder = None
                            if args.audio_required:
                                RECORD_RUNNING = False
                                recorder.save_episode()
                                logger_mp.error(
                                    "[teleop audio] required audio could not start; episode start aborted: %s",
                                    exc,
                                )
                            else:
                                RECORD_RUNNING = True
                                logger_mp.warning(
                                    "[teleop audio] disabled for this episode; continuing without audio: %s",
                                    exc,
                                )
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    try:
                        if episode_audio_recorder is not None:
                            _stop_episode_audio(episode_audio_recorder, recorder)
                            episode_audio_recorder = None
                    except Exception as exc:
                        logger_mp.exception("[teleop audio] failed to stop cleanly: %s", exc)
                    recorder.save_episode()
                    record_save_pending = True
                    logger_mp.info(
                        f"[teleop record] STOP requested data_json={recorder.json_path} "
                        f"queued_frames={recorder.item_id + 1}"
                    )
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)
            elif args.screen_record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    frame = None if head_img is None else head_img.bgr
                    if frame is None:
                        logger_mp.error("[teleop screen record] Cannot start: no decoded head camera frame.")
                    elif cv2 is None:
                        logger_mp.error("[teleop screen record] Cannot start: OpenCV is unavailable.")
                    else:
                        os.makedirs(args.screen_record_dir, exist_ok=True)
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        screen_record_path = os.path.join(
                            args.screen_record_dir,
                            f"head_{timestamp}.mp4",
                        )
                        height, width = frame.shape[:2]
                        screen_record_writer = cv2.VideoWriter(
                            screen_record_path,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            float(args.frequency),
                            (width, height),
                        )
                        if screen_record_writer.isOpened():
                            RECORD_RUNNING = True
                            logger_mp.info(
                                f"[teleop screen record] START path={screen_record_path} "
                                f"size={width}x{height} fps={args.frequency:.2f}"
                            )
                        else:
                            logger_mp.error(
                                f"[teleop screen record] Failed to open video writer: {screen_record_path}"
                            )
                            screen_record_writer.release()
                            screen_record_writer = None
                            screen_record_path = None
                else:
                    RECORD_RUNNING = False
                    if screen_record_writer is not None:
                        screen_record_writer.release()
                        screen_record_writer = None
                    logger_mp.info(f"[teleop screen record] SAVED path={screen_record_path}")

            if args.screen_record and RECORD_RUNNING and screen_record_writer is not None:
                frame = None if head_img is None else head_img.bgr
                if frame is not None:
                    screen_record_writer.write(frame)

            # get xr's tele data
            tele_data = tv_wrapper.get_tele_data()
            arm_tracking_ready = bool(
                getattr(tele_data, "head_pose_is_valid", True)
                and getattr(tele_data, "left_arm_is_valid", True)
                and getattr(tele_data, "right_arm_is_valid", True)
            )
            latest_tactiles = None
            if rh5dg2_tactile_udp is not None:
                latest_tactiles = rh5dg2_tactile_udp.read_latest()
            if not latest_tactiles and hand_ctrl is not None and hasattr(hand_ctrl, "read_latest_tactile"):
                latest_tactiles = hand_ctrl.read_latest_tactile()
            if rh5dg2_tactile_heat_mappers:
                try:
                    if TACTILE_VR_OVERLAY_VISIBLE and latest_tactiles and not latest_tactiles.get("_stale", False):
                        tactile_views = [
                            mapper.update(latest_tactiles)
                            for mapper in rh5dg2_tactile_heat_mappers.values()
                        ]
                        tv_wrapper.set_tactile_overlay(tactile_views)
                    else:
                        tv_wrapper.set_tactile_overlay(None)
                except Exception as exc:
                    if loop_count % 30 == 0:
                        logger_mp.warning("[RH5DG2 tactile VR overlay] update skipped: %s", exc)
            if neck_ctrl is not None and getattr(tele_data, "head_pose_is_valid", True):
                try:
                    neck_measured = neck_ctrl._extract_yaw_pitch(tele_data.head_pose)
                    neck_command, neck_target = neck_ctrl.update(tele_data.head_pose)
                    neck_actual = neck_feedback.read_latest() if neck_feedback is not None else None
                    neck_record = {
                        "raw_head_yaw_pitch": neck_measured.tolist(),
                        "target_yaw_pitch": neck_target.tolist(),
                        "command_yaw_pitch": neck_command.tolist(),
                        "actual_yaw_pitch": None if neck_actual is None else neck_actual.get("yaw_pitch"),
                        "actual_timestamp": None if neck_actual is None else neck_actual.get("timestamp"),
                    }
                    waist_command = None
                    if args.enable_waist_follow_neck:
                        waist_command = arm_ctrl.ctrl_waist_yaw(
                            neck_command[0] * args.waist_yaw_gain,
                            limit=args.waist_yaw_limit,
                            velocity_limit=args.waist_yaw_velocity,
                        )
                    now = time.time()
                    should_log_neck = neck_log_interval is not None and now - neck_log_last_ts >= neck_log_interval
                    if should_log_neck:
                        neck_log_last_ts = now
                        waist_actual = None
                        waist_error = None
                        if waist_command is not None:
                            waist_actual = arm_ctrl.get_waist_yaw_relative_position()
                            waist_error = waist_command - waist_actual
                        logger_mp.debug(
                            f"[teleop neck] raw_head={np.round(neck_measured, 4).tolist()} "
                            f"target={np.round(neck_target, 4).tolist()} "
                            f"command={np.round(neck_command, 4).tolist()} "
                            f"actual={None if neck_actual is None else np.round(neck_actual.get('yaw_pitch'), 4).tolist()} "
                            f"waist_yaw={None if waist_command is None else round(waist_command, 4)} "
                            f"waist_actual={None if waist_actual is None else round(waist_actual, 4)} "
                            f"waist_error={None if waist_error is None else round(waist_error, 4)}"
                        )
                except (ValueError, OSError) as e:
                    if loop_count % 30 == 0:
                        logger_mp.warning(f"[teleop neck] command skipped: {e}")

            # [수정 부분: 강제 Swap 로직 제거하고 있는 그대로(Left->Left, Right->Right) 할당]
            left_hand_pos = tele_data.left_hand_pos
            right_hand_pos = tele_data.right_hand_pos
            if args.ee in ("rh5dg2_dfx", "rh5dg2_ftp") and args.rh5dg2_hand_swap:
                left_hand_pos, right_hand_pos = right_hand_pos, left_hand_pos

            left_wrist_pose = tele_data.left_wrist_pose
            right_wrist_pose = tele_data.right_wrist_pose
            left_wrist_pose, right_wrist_pose, left_arm_sens_debug, right_arm_sens_debug = _apply_arm_sensitivity(
                left_wrist_pose,
                right_wrist_pose,
                arm_sensitivity_state,
                arm_sensitivity_config,
                enabled=not args.disable_arm and arm_tracking_ready,
            )
            if loop_count % 50 == 0:
                logger_mp.debug(
                    f"[teleop arm input] ready={READY} start={START} "
                    f"tracking_ready={arm_tracking_ready} "
                    f"head_valid={getattr(tele_data, 'head_pose_is_valid', None)} "
                    f"left_arm_valid={getattr(tele_data, 'left_arm_is_valid', None)} "
                    f"right_arm_valid={getattr(tele_data, 'right_arm_is_valid', None)} "
                    f"left_wrist={_fmt_pose_debug(left_wrist_pose)} "
                    f"right_wrist={_fmt_pose_debug(right_wrist_pose)} "
                    f"head={_fmt_pose_debug(getattr(tele_data, 'head_pose', []))}"
                )
                logger_mp.debug(
                    f"[teleop arm sensitivity] "
                    f"{_fmt_arm_sensitivity_debug('left', left_arm_sens_debug)} "
                    f"{_fmt_arm_sensitivity_debug('right', right_arm_sens_debug)} "
                    f"pos_gain={np.round(arm_sensitivity_config['pos_gain'], 4).tolist()} "
                    f"rot_gain={arm_sensitivity_config['rot_gain']:.3f} "
                    f"max_delta={arm_sensitivity_config['max_delta']:.3f} "
                    f"smoothing_alpha={arm_sensitivity_config['smoothing_alpha']:.3f} "
                    f"enabled={arm_sensitivity_config.get('enabled', False)}"
                )

            left_hand_pinchValue = tele_data.left_hand_pinchValue
            right_hand_pinchValue = tele_data.right_hand_pinchValue

            left_hand_pinch = tele_data.left_hand_pinch
            right_hand_pinch = tele_data.right_hand_pinch

            left_hand_squeeze = tele_data.left_hand_squeeze
            right_hand_squeeze = tele_data.right_hand_squeeze

            left_hand_squeezeValue = tele_data.left_hand_squeezeValue
            right_hand_squeezeValue = tele_data.right_hand_squeezeValue

            if args.ee in ("dex3", "inspire_dfx", "inspire_ftp", "inspire_dg2", "rh5dg2_dfx", "rh5dg2_ftp", "rh56f1", "brainco") and args.input_mode == "hand":
                hand_input_count += 1
                now = time.time()
                should_hand_debug = hand_debug_interval is not None and now - hand_debug_last_ts >= hand_debug_interval
                if should_hand_debug:
                    hand_debug_last_ts = now
                hand_status = _hand_tracking_status(left_hand_pos, right_hand_pos)
                if should_hand_debug:
                    logger_mp.info(
                        f"[teleop hand input before write] ee={args.ee} input={args.input_mode} "
                        f"left={_fmt_hand_debug(left_hand_pos)} right={_fmt_hand_debug(right_hand_pos)} "
                        f"timestamp={now:.6f}"
                    )
                    logger_mp.info(
                        "[teleop hand tracking status] "
                        f"hand_tracking_ready={hand_status['hand_tracking_ready']} "
                        f"left_allzero={hand_status['left_allzero']} "
                        f"right_allzero={hand_status['right_allzero']} "
                        f"left_valid_points={hand_status['left_valid_points']} "
                        f"right_valid_points={hand_status['right_valid_points']}"
                    )
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = left_hand_pos.flatten()
                    left_shared_debug = np.array(left_hand_pos_array[:]).reshape(25, 3).copy() if should_hand_debug else None
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = right_hand_pos.flatten()
                    right_shared_debug = np.array(right_hand_pos_array[:]).reshape(25, 3).copy() if should_hand_debug else None
                if args.ee in ("rh5dg2_dfx", "rh5dg2_ftp"):
                    with hand_input_timestamp.get_lock():
                        hand_input_timestamp.value = now
                if should_hand_debug:
                    logger_mp.info(
                        f"[teleop hand input after write] ee={args.ee} "
                        f"left_shared={_fmt_hand_debug(left_shared_debug)} right_shared={_fmt_hand_debug(right_shared_debug)} "
                        f"write_latency_ms={(time.time() - now) * 1000.0:.2f}"
                    )
                    logger_mp.info(
                        f"[teleop hand input hz] hz={_rate_hz(hand_input_count, hand_input_rate_start):.2f} "
                        f"count={hand_input_count}"
                    )
            elif args.ee == "dex1" and args.input_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_ctrl_triggerValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_ctrl_triggerValue
            elif args.ee == "dex1" and args.input_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = right_hand_pinchValue
            else:
                pass
            
            # high level control
            if args.input_mode == "controller" and args.motion and not args.disable_body:
                # quit teleoperate
                if tele_data.right_ctrl_aButton:
                    START = False
                    STOP = True
                # command robot to enter damping mode. soft emergency stop function
                if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
                    loco_wrapper.Damp()
                # https://github.com/unitreerobotics/xr_teleoperate/issues/135, control, limit velocity to within 0.3
                loco_wrapper.Move(-tele_data.left_ctrl_thumbstickValue[1] * 0.3,
                                  -tele_data.left_ctrl_thumbstickValue[0] * 0.3,
                                  -tele_data.right_ctrl_thumbstickValue[0]* 0.3)

            # get current robot state data.
            if arm_ctrl is None:
                current_lr_arm_q = np.array([], dtype=np.float64)
                current_lr_arm_dq = np.array([], dtype=np.float64)
            else:
                current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
                current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

            # solve ik using motor data and wrist pose, then use ik results to control arms.
            if args.disable_arm or arm_ctrl is None:
                sol_q = np.asarray(current_lr_arm_q, dtype=np.float64).copy()
                sol_tauff = np.zeros_like(sol_q)
                if loop_count % 50 == 0:
                    logger_mp.warning(
                        f"[teleop arm disabled reason] loop={loop_count} reason=--disable-arm "
                        f"current_q={_fmt_vec_debug(current_lr_arm_q)}"
                    )
            elif not arm_tracking_ready:
                sol_q = np.asarray(current_lr_arm_q, dtype=np.float64).copy()
                sol_tauff = np.zeros_like(sol_q)
                if loop_count % 10 == 0:
                    logger_mp.warning(
                        f"[teleop arm tracking hold] loop={loop_count} "
                        f"head_valid={getattr(tele_data, 'head_pose_is_valid', None)} "
                        f"left_arm_valid={getattr(tele_data, 'left_arm_is_valid', None)} "
                        f"right_arm_valid={getattr(tele_data, 'right_arm_is_valid', None)} "
                        f"current_q={_fmt_vec_debug(current_lr_arm_q)}"
                    )
            else:
                time_ik_start = time.time()
                if loop_count % 50 == 0:
                    logger_mp.debug(
                        f"[teleop arm ik enter] current_q={_fmt_vec_debug(current_lr_arm_q)} "
                        f"current_dq={_fmt_vec_debug(current_lr_arm_dq)} "
                        f"left_wrist={_fmt_pose_debug(left_wrist_pose)} "
                        f"right_wrist={_fmt_pose_debug(right_wrist_pose)}"
                    )
                sol_q, sol_tauff  = arm_ik.solve_ik(left_wrist_pose, right_wrist_pose, current_lr_arm_q, current_lr_arm_dq)
                time_ik_end = time.time()
                logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
                if loop_count % 50 == 0:
                    logger_mp.debug(
                        f"[teleop arm ik result] dt={time_ik_end - time_ik_start:.6f} "
                        f"current_q={_fmt_vec_debug(current_lr_arm_q)} sol_q={_fmt_vec_debug(sol_q)} "
                        f"tauff={_fmt_vec_debug(sol_tauff)}"
                    )
                startup_elapsed = time.time() - tracking_start_time
                if args.arm_startup_duration > 0.0 and startup_elapsed < args.arm_startup_duration:
                    max_step = max(0.0, args.arm_startup_max_step)
                    sol_q = current_lr_arm_q + np.clip(sol_q - current_lr_arm_q, -max_step, max_step)
                    sol_tauff = np.zeros_like(sol_q)
                    if loop_count % 10 == 0:
                        logger_mp.info(
                            f"[teleop arm startup clamp] elapsed={startup_elapsed:.3f}s "
                            f"max_step={max_step:.4f} target={_fmt_vec_debug(sol_q)}"
                        )
            # ---- [추가 부분: 안전 장치 (Safety Mechanism)] ----
            # 로봇 스펙에 맞게 최대/최소 라디안 및 급격한 움직임 허용치 설정
            #MAX_RAD = 2.5    # 예: 절대적인 최대 관절 각도
            #MIN_RAD = -2.5   # 예: 절대적인 최소 관절 각도
            #MAX_DELTA = 0.5  # 예: 한 프레임(약 0.03초) 내 허용되는 최대 각도 변화량

            #q_delta_abs = np.abs(sol_q - current_lr_arm_q)
            
            #if np.any(sol_q > MAX_RAD) or np.any(sol_q < MIN_RAD) or np.any(q_delta_abs > MAX_DELTA):
            #    logger_mp.error("🚨 Safety Triggered: Abnormal joint movement detected!")
                
                # High-level controller가 있을 경우 Damping 모드 실행
            #    if args.motion and args.input_mode == "controller":
            #        try:
            #            loco_wrapper.Damp()
            #            logger_mp.info("Entered Damping Mode successfully.")
            #        except NameError:
            #            pass
                
                # 텔레오퍼레이션 즉시 정지 상태로 전환 (로봇에 비정상 sol_q 전달 방지)
            #    START = False
            #    STOP = True
            #    continue 
            # ---------------------------------------------------
            if not args.disable_arm and arm_ctrl is not None:
                if loop_count % 50 == 0:
                    logger_mp.debug(
                        f"[teleop arm publish enter] controller={arm_ctrl.__class__.__name__} "
                        f"sim={args.sim} target={_fmt_vec_debug(sol_q)}"
                    )
                arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
                if loop_count % 50 == 0:
                    arm_write_ok = arm_ctrl.get_last_write_ok() if hasattr(arm_ctrl, "get_last_write_ok") else None
                    logger_mp.debug(
                        f"[teleop arm publish] controller={arm_ctrl.__class__.__name__} "
                        f"sim={args.sim} write_ok={arm_write_ok} topic=rt/lowcmd domain={1 if args.sim else 0} "
                        f"target={_fmt_vec_debug(sol_q)} tauff={_fmt_vec_debug(sol_tauff)}"
                    )

            # record data
            if args.record:
                READY = recorder.is_ready() # now ready to (2) enter RECORD_RUNNING state
                if record_save_pending and READY and not RECORD_RUNNING:
                    record_save_pending = False
                    logger_mp.info(
                        f"[teleop record] SAVE COMPLETE data_json={recorder.json_path} "
                        f"frames_written={getattr(recorder, 'items_written', None)} "
                        f"frames_failed={getattr(recorder, 'items_failed', None)} "
                        "ready_for_next_episode=True Press [s] to start next recording."
                    )
                # dex hand or gripper
                if args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist() if arm_ctrl is not None else []
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = []
                        current_body_action = []

                elif (args.ee == "rh5dg2_dfx" or args.ee == "rh5dg2_ftp" or args.ee == "inspire_dg2") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:13]
                        right_ee_state = dual_hand_state_array[-13:]
                        left_hand_action = dual_hand_action_array[:13]
                        right_hand_action = dual_hand_action_array[-13:]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "rh56f1" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        rh56f1_count = len(dual_hand_state_array) // 2
                        left_ee_state = dual_hand_state_array[:rh56f1_count]
                        right_ee_state = dual_hand_state_array[-rh56f1_count:]
                        left_hand_action = dual_hand_action_array[:rh56f1_count]
                        right_hand_action = dual_hand_action_array[-rh56f1_count:]
                        current_body_state = []
                        current_body_action = []
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []

                recorded_body_state = _read_body_qpos(arm_ctrl, enabled=args.record_body_state)
                if recorded_body_state:
                    current_body_state = recorded_body_state

                # arm state and action
                left_arm_state  = current_lr_arm_q[:7]
                right_arm_state = current_lr_arm_q[-7:]
                left_arm_action = sol_q[:7]
                right_arm_action = sol_q[-7:]
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    if camera_config['head_camera']['binocular']:
                        if head_img is not None:
                            head_split_x = head_img.bgr.shape[1] // 2
                            colors[f"color_{0}"] = head_img.bgr[:, :head_split_x]
                            colors[f"color_{1}"] = head_img.bgr[:, head_split_x:]
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_bgr is not None:
                                colors[f"color_{2}"] = left_wrist_bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_bgr is not None:
                                colors[f"color_{3}"] = right_wrist_bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    else:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_bgr is not None:
                                colors[f"color_{1}"] = left_wrist_bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_bgr is not None:
                                colors[f"color_{2}"] = right_wrist_bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   [],                          
                            "torque": [],                        
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   [],                          
                            "torque": [],                         
                        },                        
                        "left_ee": {                                                                    
                            "qpos":   left_ee_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_ee": {                                                                    
                            "qpos":   right_ee_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        }, 
                        "body": {
                            "qpos": current_body_state,
                        }, 
                    }
                    actions = {
                        "left_arm": {                                   
                            "qpos":   left_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],      
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],       
                        },                         
                        "left_ee": {                                   
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_ee": {                                   
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        }, 
                        "body": {
                            "qpos": current_body_action,
                        }, 
                    }
                    if neck_record is not None:
                        states["neck"] = {
                            "raw_head_yaw_pitch": neck_record["raw_head_yaw_pitch"],
                            "actual_yaw_pitch": neck_record["actual_yaw_pitch"],
                            "actual_timestamp": neck_record["actual_timestamp"],
                        }
                        actions["neck"] = {
                            "target_yaw_pitch": neck_record["target_yaw_pitch"],
                            "command_yaw_pitch": neck_record["command_yaw_pitch"],
                        }
                    tactiles = None
                    if latest_tactiles:
                        tactiles = latest_tactiles
                    if not tactiles and rh56f1_tactile_reader is not None:
                        tactiles = rh56f1_tactile_reader.read_latest()
                    audios = None
                    if audio_udp_receiver is not None:
                        audios = audio_udp_receiver.read_latest()
                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, tactiles=tactiles, audios=audios, sim_state=sim_state)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, tactiles=tactiles, audios=audios)

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception:
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        try:
            if episode_audio_recorder is not None and recorder is not None:
                _stop_episode_audio(episode_audio_recorder, recorder)
                episode_audio_recorder = None
        except Exception as e:
            logger_mp.error(f"Failed to stop episode audio recorder: {e}")

        try:
            if recorder is not None:
                logger_mp.info("[teleop record] finalizing recorder before device shutdown...")
                recorder.close()
                recorder = None
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")

        try:
            if neck_ctrl is not None:
                neck_ctrl.close()
        except Exception as e:
            logger_mp.error(f"Failed to close neck controller: {e}")

        try:
            if neck_feedback is not None:
                neck_feedback.close()
        except Exception as e:
            logger_mp.error(f"Failed to close neck feedback receiver: {e}")

        try:
            if rh56f1_tactile_reader is not None:
                rh56f1_tactile_reader.stop()
        except Exception as e:
            logger_mp.error(f"Failed to stop RH56F1 tactile reader: {e}")

        try:
            if rh5dg2_tactile_udp is not None:
                rh5dg2_tactile_udp.close()
        except Exception as e:
            logger_mp.error(f"Failed to stop RH5DG2 tactile UDP receiver: {e}")

        try:
            if audio_udp_receiver is not None:
                audio_udp_receiver.close()
        except Exception as e:
            logger_mp.error(f"Failed to stop audio UDP receiver: {e}")

        try:
            if hand_ctrl is not None and hasattr(hand_ctrl, "stop"):
                hand_ctrl.stop()
        except Exception as e:
            logger_mp.error(f"Failed to stop hand controller: {e}")

        try:
            if hand_ctrl is not None and hasattr(hand_ctrl, "restore_initial_pose"):
                hand_ctrl.restore_initial_pose()
        except Exception as e:
            logger_mp.error(f"Failed to restore RH5DG2 initial hand pose: {e}")

        try:
            if args.skip_arm_go_home_on_exit:
                logger_mp.warning("[teleop arm shutdown] skip ctrl_dual_arm_go_home because --skip-arm-go-home-on-exit is set.")
            elif arm_ctrl is not None:
                arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")
        
        try:
            if args.ipc:
                ipc_server.stop()
            else:
                stop_listening()
                listen_keyboard_thread.join(timeout=1.0)
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")
        
        try:
            if img_client is not None:
                img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            if tv_wrapper is not None:
                tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        try:
            if not args.motion:
                pass
                # status, result = motion_switcher.Exit_Debug_Mode()
                # logger_mp.info(f"Exit debug mode: {'Success' if status == 3104 else 'Failed'}")
        except Exception as e:
            logger_mp.error(f"Failed to exit debug mode: {e}")

        try:
            if args.sim and sim_state_subscriber is not None:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")

        try:
            if screen_record_writer is not None:
                screen_record_writer.release()
                logger_mp.info(f"[teleop screen record] SAVED on exit path={screen_record_path}")
        except Exception as e:
            logger_mp.error(f"Failed to close screen recorder: {e}")
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)
