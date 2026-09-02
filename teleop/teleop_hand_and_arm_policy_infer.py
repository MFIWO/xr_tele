"""Closed-loop policy rollout on the H1-2, driven by a remote LeRobot policy server.

Copied from `teleop_hand_and_arm_manus_and_vive.py` — the robot bring-up, cameras,
XR viewer, keyboard FSM, recording and shutdown paths are unchanged. What differs is
where the commands come from: instead of Vive trackers + Manus gloves + IK, every
joint target is an action predicted by the ACT / GR00T checkpoint trained on the data
that same teleop script recorded.

    this script (robot PC, no GPU)  --TCP:5601-->  lerobot policy server (GPU PC)
      observation: state(42) + 4 cameras          examples/h1_2_deploy/policy_bridge_server.py
      action:      state-shaped 42-dim chunk

Vector layout, identical to the training dataset built by
`lerobot/examples/port_datasets/port_h1_2_loop_kit.py`::

    [0:7]   left arm joint positions (rad)
    [7:20]  left hand motor positions (Inspire DG2 raw units, 0..2040)
    [20:27] right arm joint positions (rad)
    [27:40] right hand motor positions
    [40:42] head yaw, pitch (rad) — the command the pan/tilt neck receives

Checklist before running:
    - Same scene/lighting/robot start pose as data collection ([h] drives the robot to
      that home pose; [r] then hands control to the policy).
    - Cameras mounted identically; wrist vflip matches how the episodes were recorded
      (see --policy-wrist-vflip).
    - E-stop within reach. Start with --dry-run, then --policy-arm-max-step small.

Usage:
    python teleop/teleop_hand_and_arm_policy_infer.py \
        --ee inspire_dg2 --policy-server 192.168.123.50:5601 \
        --policy-task "Turn on the turn signal ..." --dry-run
"""

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

import glob
import os
import sys
import socket
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds 
from televuer import TeleVuerWrapper
from teleop.robot_control.robot_arm import (
    G1_29_ArmController,
    G1_29_JointArmIndex,
    G1_23_ArmController,
    G1_23_JointArmIndex,
    H1_2_ArmController,
    H1_2_JointArmIndex,
    H1_ArmController,
    H1_JointArmIndex,
    H2_ArmController,
    H2_JointArmIndex,
)
from teleimager.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.audio_recorder import BackgroundAudioRecorder, AudioRecorderError
from teleop.utils.ipc import IPC_Server
from teleop.utils.motion_switcher import MotionSwitcher
from teleop.utils.rh5dg2_tactile import RH5DG2TactileHeatMapper
from teleop.utils.policy_bridge import PolicyBridgeClient
from teleop.neck_control import VisionProNeckController
from sshkeyboard import listen_keyboard, stop_listening

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
def publish_reset_category(category: int, publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")


START_SYNC_DELAY = 1.0
START_SYNC_AT = None

# Guards the [r]/[p] state transitions shared between the sshkeyboard listener
# thread (on_press) and the main control loop (delayed start).
STATE_LOCK = threading.Lock()


def _maybe_start_delayed_run_locked():
    """Fire the queued [r] once its delay elapsed. Returns True when START flips on."""
    global START, START_SYNC_AT, PAUSE_TO_READY
    if START_SYNC_AT is None:
        return False
    if STOP:
        START_SYNC_AT = None
        return False
    if time.monotonic() < START_SYNC_AT:
        return False
    if START:
        # A stale [r] raced with an already-running rollout; do not re-fire mid-run.
        START_SYNC_AT = None
        return False
    START_SYNC_AT = None
    START = True
    PAUSE_TO_READY = False
    if STEP_MODE_ENABLED:
        logger_mp.info("[policy run] START (step mode): press [n] to run one inference per step. [p] pauses, [q] quits.")
    else:
        logger_mp.info("[policy run] START: the policy now drives arms/hands/neck. Press [p] to pause, [q] to quit.")
    return True


def maybe_start_delayed_run():
    with STATE_LOCK:
        return _maybe_start_delayed_run_locked()


def _update_neck_from_action(
    args,
    neck_ctrl,
    neck_feedback,
    head_action,
    arm_ctrl,
    loop_count,
    neck_log_last_ts,
    neck_log_interval,
    allow_waist=True,
):
    """Drive the pan/tilt neck from the policy's 2-dim head action.

    The recorded `head` action channel is the yaw/pitch *command* the teleop run sent
    over UDP, so the policy output goes straight back into the same command path — no
    neutral offset, only the controller's clamp + rate limit.
    """
    if neck_ctrl is None or head_action is None:
        return None, neck_log_last_ts
    try:
        neck_command, neck_target = neck_ctrl.command_absolute(head_action)
        neck_actual = neck_feedback.read_latest() if neck_feedback is not None else None
        neck_record = {
            "raw_head_yaw_pitch": [float(head_action[0]), float(head_action[1])],
            "target_yaw_pitch": neck_target.tolist(),
            "command_yaw_pitch": neck_command.tolist(),
            "actual_yaw_pitch": None if neck_actual is None else neck_actual.get("yaw_pitch"),
            "actual_timestamp": None if neck_actual is None else neck_actual.get("timestamp"),
        }
        waist_command = None
        if allow_waist and args.enable_waist_follow_neck and arm_ctrl is not None:
            waist_direction = -1.0 if args.waist_follow_neck_invert else 1.0
            waist_command = arm_ctrl.ctrl_waist_yaw(
                neck_command[0] * args.waist_yaw_gain * waist_direction,
                limit=args.waist_yaw_limit,
                velocity_limit=args.waist_yaw_velocity,
            )
        now = time.time()
        if neck_log_interval is not None and now - neck_log_last_ts >= neck_log_interval:
            neck_log_last_ts = now
            logger_mp.info(
                f"[policy neck] action={np.round(head_action, 4).tolist()} "
                f"target={np.round(neck_target, 4).tolist()} "
                f"command={np.round(neck_command, 4).tolist()} "
                f"actual={None if neck_actual is None else np.round(neck_actual.get('yaw_pitch'), 4).tolist()} "
                f"waist_yaw={None if waist_command is None else round(waist_command, 4)}"
            )
        return neck_record, neck_log_last_ts
    except (ValueError, OSError) as e:
        if loop_count % 30 == 0:
            logger_mp.warning(f"[policy neck] command skipped: {e}")
        return None, neck_log_last_ts

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
PAUSE_TO_READY = False  # True while teleop is paused by [p]; arms are driven back to the ready pose
GO_HOME        = False  # True while [h] eases the arms/hands/neck to the home pose before a rollout
TACTILE_VR_OVERLAY_VISIBLE = True  # Toggle RH5DG2 tactile overlay visibility in the XR viewer
# waist keyboard control (H1_2 only): [j]/[k] nudge the waist yaw left/right during teleop
WAIST_YAW_REL     = 0.0     # current relative waist-yaw target (rad, relative to startup home)
WAIST_KEY_STEP    = 0.05    # rad added/removed per [j]/[k] press; overwritten from args in main
WAIST_KEY_LIMIT   = 0.1745  # +/- clamp for the accumulator (rad); overwritten from args in main
WAIST_KEY_ENABLED = False   # set True when --enable-waist-keyboard is passed
WAIST_KEY_INVERT  = False   # swap [j]/[k] direction; overwritten from args in main
# synchronous single-step rollout: [n] sends exactly one observation, the returned
# action becomes the held target, and nothing else moves until the next [n]
STEP_MODE_ENABLED = False   # set True when --policy-step-mode is passed
STEP_REQUEST      = False   # [n] pressed; cleared once the observation is actually sent
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
    with STATE_LOCK:
        _on_press_locked(key)


def _on_press_locked(key):
    global STOP, START, READY, RECORD_RUNNING, RECORD_TOGGLE, TACTILE_VR_OVERLAY_VISIBLE, WAIST_YAW_REL
    global START_SYNC_AT, PAUSE_TO_READY, GO_HOME, STEP_REQUEST
    if key == 'r':
        if START:
            logger_mp.warning("[policy run] ignored [r]: the rollout is already running. Press [p] to pause, or [q] to quit.")
            return
        START = False
        PAUSE_TO_READY = False
        GO_HOME = False
        START_SYNC_AT = time.monotonic() + START_SYNC_DELAY
        logger_mp.info(
            "[policy run] [r] pressed. The policy takes over in %.1fs — stand clear and keep the E-stop in reach.",
            START_SYNC_DELAY,
        )
    elif key == 'p':
        if START:
            # Pause the rollout: drive the arms back to the ready pose and wait for [r].
            START = False
            START_SYNC_AT = None
            PAUSE_TO_READY = True
            logger_mp.info(
                "[policy pause] [p] pressed. Arms return to the ready pose and the policy is disengaged; "
                "press [r] to resume."
            )
        else:
            logger_mp.warning("[policy pause] ignored [p] because the rollout is not running. Press [r] to start.")
    elif key == 'h':
        # Send the robot to the home pose the episodes started from, before handing over
        # to the policy: the first predicted chunk assumes that starting configuration.
        if START:
            logger_mp.warning("[policy home] ignored [h]: the rollout is running. Press [p] to pause first.")
        else:
            GO_HOME = True
            PAUSE_TO_READY = False
            START_SYNC_AT = None  # a queued [r] must not fire mid-homing
            WAIST_YAW_REL = 0.0
            logger_mp.info("[policy home] [h] pressed. Easing arms/hands/neck to the home pose; press [r] when it settles.")
    elif key == 'q':
        START = False
        START_SYNC_AT = None
        GO_HOME = False
        STOP = True
    elif key == 's':
        if START == True and (READY or RECORD_RUNNING):
            RECORD_TOGGLE = True
        elif START == True:
            logger_mp.warning("[policy record] ignored [s] because the previous episode is still saving. Please wait until READY.")
        else:
            logger_mp.warning("[policy record] ignored [s] because the rollout has not started. Press [r] first, then [s] to record.")
    elif key == 't':
        TACTILE_VR_OVERLAY_VISIBLE = not TACTILE_VR_OVERLAY_VISIBLE
        logger_mp.info(
            "[RH5DG2 tactile VR overlay] keyboard toggle visible=%s",
            TACTILE_VR_OVERLAY_VISIBLE,
        )
    elif key == 'j' or key == 'k':
        if not WAIST_KEY_ENABLED:
            logger_mp.warning("[teleop waist keyboard] ignored [%s]: pass --enable-waist-keyboard to use it.", key)
        else:
            direction = 1.0 if key == 'j' else -1.0
            if WAIST_KEY_INVERT:
                direction = -direction
            WAIST_YAW_REL = float(np.clip(WAIST_YAW_REL + direction * WAIST_KEY_STEP,
                                          -WAIST_KEY_LIMIT, WAIST_KEY_LIMIT))
            logger_mp.info("[teleop waist keyboard] [%s] waist_yaw_rel=%.4f rad (%.1f deg)",
                           key, WAIST_YAW_REL, np.degrees(WAIST_YAW_REL))
    elif key == 'i':
        if not WAIST_KEY_ENABLED:
            logger_mp.warning("[teleop waist keyboard] ignored [i]: pass --enable-waist-keyboard to use it.")
        else:
            WAIST_YAW_REL = 0.0
            logger_mp.info("[teleop waist keyboard] [i] waist reset to home (0.0 deg)")
    elif key == 'n':
        if not STEP_MODE_ENABLED:
            logger_mp.warning("[policy step] ignored [n]: pass --policy-step-mode to use single-step inference.")
        elif not START:
            logger_mp.warning("[policy step] ignored [n]: press [r] first to engage the rollout.")
        elif STEP_REQUEST:
            logger_mp.warning("[policy step] ignored [n]: the previous step is still in flight.")
        else:
            STEP_REQUEST = True
            logger_mp.info("[policy step] [n] pressed: sending one observation to the policy server.")
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
    if values is None:
        return "none"
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
    if left_values is None or right_values is None:
        return {
            "hand_tracking_ready": False,
            "left_allzero": True,
            "right_allzero": True,
            "left_valid_points": 0,
            "right_valid_points": 0,
        }
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
    if tv_wrapper is None:
        return False
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
        self.side_last_rx_time = {"left_ee": 0.0, "right_ee": 0.0}
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
                    now = time.time()
                    side_updates = {}
                    for side in ("left_ee", "right_ee"):
                        if isinstance(packet.get(side), dict):
                            side_updates[side] = packet[side]
                    tactiles = packet.get("tactiles")
                    if isinstance(tactiles, dict):
                        for side in ("left_ee", "right_ee"):
                            if isinstance(tactiles.get(side), dict):
                                side_updates[side] = tactiles[side]
                    if side_updates:
                        merged = {
                            key: value
                            for key, value in self.latest.items()
                            if key in ("left_ee", "right_ee")
                        }
                        merged.update(copy.deepcopy(side_updates))
                        merged["timestamp"] = packet.get("timestamp", now)
                        merged["source"] = packet.get("source", "rh5dg2_tactile_udp")
                        self.latest = merged
                        for side in side_updates:
                            self.side_last_rx_time[side] = now
                    else:
                        self.latest = packet
                    self.last_rx_time = now
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
            stale_sides = [
                side
                for side, stamp in self.side_last_rx_time.items()
                if stamp > 0.0 and now - stamp > self.timeout
            ]
        if age > self.timeout:
            latest["_stale"] = True
            latest["_age_sec"] = age
        if stale_sides:
            latest["_stale_sides"] = stale_sides
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

def _draw_record_indicator(bgr, recording, binocular=False):
    """Draw a small record-state badge in the top-left of the XR image.

    Returns a COPY so the overlay never lands in the recorded frames (viewer_bgr
    may alias head_img.bgr). Red filled dot + 'REC' while recording, dim hollow
    dot + 'STBY' otherwise. For binocular images the badge is drawn on both eyes.
    """
    if bgr is None or cv2 is None:
        return bgr
    src = np.asarray(bgr)
    if src.ndim != 3:
        return bgr
    out = src.copy()
    h, w = out.shape[:2]
    eye_w = w // 2 if binocular else w
    n_eyes = 2 if binocular else 1
    r = max(10, int(round(h * 0.032)))               # 2x size
    margin_y = max(10, int(round(h * 0.03)))
    color = (0, 0, 255) if recording else (170, 170, 170)  # BGR: red / gray
    label = "REC" if recording else "STBY"
    thick = max(2, int(round(h * 0.004)))            # 2x size
    fscale = max(0.7, h * 0.0022)                    # 2x size
    font = cv2.FONT_HERSHEY_SIMPLEX
    gap = max(6, r // 2)
    (tw, th), _ = cv2.getTextSize(label, font, fscale, thick)
    badge_w = 2 * r + gap + tw
    cy = margin_y + r
    for e in range(n_eyes):
        eye_center = e * eye_w + eye_w // 2
        cx = eye_center - badge_w // 2 + r           # dot center; badge centered on eye
        if recording:
            cv2.circle(out, (cx, cy), r, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(out, (cx, cy), r, color, thick, cv2.LINE_AA)
        cv2.putText(out, label, (cx + r + gap, cy + th // 2), font,
                    fscale, color, thick, cv2.LINE_AA)
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

# Home/ready arm target for [h]/[p]. None -> built-in fallback (zeros + elbows at -0.3);
# set from --policy-home-q so [h] can go to the pose the episodes were actually recorded from.
POLICY_HOME_ARM_Q = None

def _make_arm_ready_q(current_q):
    q_now = np.asarray(current_q, dtype=np.float64).reshape(-1)
    if POLICY_HOME_ARM_Q is not None and POLICY_HOME_ARM_Q.size == q_now.size:
        return POLICY_HOME_ARM_Q.copy()
    q = np.zeros_like(q_now)
    if q.size == 0:
        return q
    half = q.size // 2
    if q.size >= 8 and half + 3 < q.size:
        q[3] = -0.3
        q[half + 3] = -0.3
    return q


def _resolve_policy_home_q(raw):
    """Resolve --policy-home-q into a 14-dim arm target (left 7 + right 7, radians).

    Accepts: 14 comma-separated floats; a recorded episode `data.json` (or its episode
    directory) — takes the first frame's left_arm/right_arm qpos; or a LeRobot dataset
    root — takes the arm joints of an `observation.state` row, understanding the
    known layouts: 42-dim robot order ([larm7, lhand13, rarm7, rhand13, head2]), the
    63-dim deploy-bundle order (head2 + legs12 + waist1 + larm7 + rarm7 + pad8 + hands),
    or a plain 14-dim vector. A dataset root may carry a `#ep=N` and/or `#frame=M`
    suffix (e.g. `episodes/val#ep=0#frame=60`) to pick the episode and row; default is
    episode 0, frame 0. Returns None when raw is empty (keep the fallback pose)."""
    if raw is None or str(raw).strip() == "":
        return None
    text = str(raw).strip()

    try:
        parsed = _parse_float_list(text)
    except ValueError:
        parsed = None  # not a number list -> treat as a path below
    if parsed is not None:
        if len(parsed) != 14:
            raise ValueError(f"--policy-home-q expects 14 comma-separated values, got {len(parsed)}.")
        return np.asarray(parsed, dtype=np.float64)

    # Optional dataset selectors: <root>#ep=N#frame=M (default episode 0, frame 0).
    ep_index, frame_index = 0, 0
    base, _, _ = text.partition("#")
    for part in text.split("#")[1:]:
        key, _, value = part.partition("=")
        if key == "ep":
            ep_index = int(value)
        elif key == "frame":
            frame_index = int(value)
        else:
            raise ValueError(f"--policy-home-q: unknown selector {part!r}; use #ep=N and/or #frame=M.")

    path = os.path.expanduser(base)
    if os.path.isdir(path):
        episode_json = os.path.join(path, "data.json")
        if os.path.isfile(episode_json):
            path = episode_json
        else:
            # LeRobot dataset root: read one row of one data parquet.
            parquets = sorted(glob.glob(os.path.join(path, "data", "**", "*.parquet"), recursive=True))
            if not parquets:
                raise ValueError(f"--policy-home-q: no data.json or data/**/*.parquet under {path}")
            wanted = [p for p in parquets if p.endswith(f"episode_{ep_index:06d}.parquet")]
            if not wanted:
                raise ValueError(f"--policy-home-q: episode {ep_index} not found under {path}/data/")
            import pyarrow.parquet as pq  # optional dep; only needed for the dataset-root form
            table = pq.read_table(wanted[0], columns=["observation.state"])
            if not 0 <= frame_index < table.num_rows:
                raise ValueError(
                    f"--policy-home-q: frame {frame_index} out of range (episode has {table.num_rows} rows)."
                )
            state = np.asarray(table["observation.state"][frame_index].as_py(), dtype=np.float64).reshape(-1)
            if state.size == 42:    # robot order: [larm7, lhand13, rarm7, rhand13, head2]
                return np.concatenate((state[0:7], state[20:27]))
            if state.size == 63:    # deploy bundle: head2 + legs12 + waist1 + larm7 + rarm7 + ...
                return np.concatenate((state[15:22], state[22:29]))
            if state.size == 14:
                return state
            raise ValueError(
                f"--policy-home-q: observation.state has {state.size} dims; known layouts are 14, 42, 63."
            )
    if not os.path.isfile(path):
        raise ValueError(f"--policy-home-q: {text} is neither 14 floats nor an existing path.")
    with open(path) as f:
        episode = json.load(f)
    frames = episode["data"] if isinstance(episode, dict) else episode
    first = frames[0]["states"]
    home = np.asarray(
        list(first["left_arm"]["qpos"]) + list(first["right_arm"]["qpos"]), dtype=np.float64
    )
    if home.size != 14:
        raise ValueError(f"--policy-home-q: first frame arm qpos has {home.size} dims, expected 14.")
    return home

def _step_arms_toward_ready(arm_ctrl, current_q, max_step, tolerance=0.05, cmd_q=None, meas_margin=0.15,
                            gravity_comp=None):
    """Advance the *commanded* arm target one control tick toward the ready pose, clamped
    to `max_step` per joint, and return (reached, cmd_q). Pass the returned cmd_q back in
    on the next tick: integrating the command (like _smooth_arm_go_home) instead of
    stepping from the measured q avoids the stall where gravity droop keeps the measured
    position a few hundredths of a radian behind every command, so "measured + max_step"
    never actually rises above the pose the arm is already holding. `meas_margin` bounds
    the command to the measured pose (like the rollout's _policy_arm_target) so a stuck
    or lagging arm can never wind up a large PD error; the margin is well above gravity
    droop, so it does not reintroduce the stall. `reached` is judged against the measured
    q so the caller only reports home once the robot got there."""
    q_meas = np.asarray(current_q, dtype=np.float64).reshape(-1)
    if q_meas.size == 0 or not np.isfinite(q_meas).all():
        return False, cmd_q
    if cmd_q is None or np.asarray(cmd_q).size != q_meas.size:
        cmd_q = q_meas.copy()
    target = _make_arm_ready_q(q_meas)
    cmd_q = cmd_q + np.clip(target - cmd_q, -max_step, max_step)
    if meas_margin > 0.0:
        cmd_q = np.clip(cmd_q, q_meas - meas_margin, q_meas + meas_margin)
    tauff = gravity_comp.tau(cmd_q) if gravity_comp is not None else np.zeros_like(cmd_q)
    arm_ctrl.ctrl_dual_arm(cmd_q, tauff)
    reached = bool(np.max(np.abs(target - cmd_q)) <= 1e-9 and np.max(np.abs(target - q_meas)) <= tolerance)
    return reached, cmd_q


def _go_home_tick(args, arm_ctrl, hand_ctrl, neck_ctrl, neck_feedback,
                  loop_count, neck_log_last_ts, neck_log_interval, current_arm_q=None,
                  cmd_q=None, gravity_comp=None):
    """One control tick of the [h] home move: ease the arms toward the ready pose, drop the
    hand override back to the controller's open baseline, centre the neck and unwind the
    waist. Shared by the pre-[r] wait loop and the disengaged branch of the control loop so
    [h] behaves the same before the first rollout and after a [p] pause. Pass the returned
    cmd_q back in each tick (reset it to None when the move starts). Returns
    (home_reached, neck_log_last_ts, cmd_q)."""
    if args.dry_run:
        logger_mp.info("[policy home] --dry-run: nothing published; treat the robot as already home.")
        return True, neck_log_last_ts, cmd_q

    home_reached = True
    if not args.disable_arm and arm_ctrl is not None:
        try:
            if current_arm_q is None:
                current_arm_q = arm_ctrl.get_current_dual_arm_q()
            home_reached, cmd_q = _step_arms_toward_ready(
                arm_ctrl, current_arm_q, max(0.0, args.policy_arm_startup_max_step), cmd_q=cmd_q,
                meas_margin=getattr(args, "policy_arm_meas_margin", 0.15),
                gravity_comp=gravity_comp,
            )
        except Exception as exc:
            home_reached = False
            if loop_count % 30 == 0:
                logger_mp.warning("[policy home] arm command skipped: %s", exc)
    if not args.disable_hand and hand_ctrl is not None:
        hand_ctrl.clear_raw_command()
    if args.enable_neck and neck_ctrl is not None:
        _, neck_log_last_ts = _update_neck_from_action(
            args,
            neck_ctrl,
            neck_feedback,
            np.zeros(2, dtype=np.float64),
            arm_ctrl,
            loop_count,
            neck_log_last_ts,
            neck_log_interval,
            allow_waist=False,
        )
    # The [j]/[k] waist block only runs while engaged, so unwind the waist here instead.
    if arm_ctrl is not None and hasattr(arm_ctrl, "ctrl_waist_yaw"):
        try:
            arm_ctrl.ctrl_waist_yaw(
                0.0,
                limit=args.waist_yaw_limit,
                velocity_limit=args.waist_yaw_velocity,
            )
        except Exception as exc:
            if loop_count % 30 == 0:
                logger_mp.warning("[policy home] waist command skipped: %s", exc)
    return home_reached, neck_log_last_ts, cmd_q


def _smooth_arm_go_home(arm_ctrl, duration=3.0, frequency=100.0, velocity_cap=3.0):
    """Gradually lower both arms (and the H1_2 waist) from the current pose to the
    zero/home pose using a cosine ease, so they descend smoothly on exit instead of
    dropping. Returns True when the smooth descent ran."""
    if arm_ctrl is None or not hasattr(arm_ctrl, "get_current_dual_arm_q") or not hasattr(arm_ctrl, "ctrl_dual_arm"):
        return False
    try:
        start_q = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=np.float64).reshape(-1)
    except Exception as exc:
        logger_mp.warning("[teleop arm shutdown] smooth go-home skipped: cannot read arm q (%s)", exc)
        return False
    if start_q.size == 0 or not np.isfinite(start_q).all():
        logger_mp.warning("[teleop arm shutdown] smooth go-home skipped: invalid arm q.")
        return False

    target_q = np.zeros_like(start_q)
    tauff = np.zeros_like(start_q)

    # Cap the controller's own velocity clip so it cannot add a jump on top of the interpolation.
    prev_velocity_limit = getattr(arm_ctrl, "arm_velocity_limit", None)
    if prev_velocity_limit is not None and velocity_cap > 0.0:
        arm_ctrl.arm_velocity_limit = min(prev_velocity_limit, float(velocity_cap))

    # H1_2 waist: ramp back to home over the same window if the arm is already turned.
    has_waist = hasattr(arm_ctrl, "ctrl_waist_yaw") and hasattr(arm_ctrl, "get_waist_yaw_relative_position")
    start_waist = 0.0
    if has_waist:
        try:
            start_waist = float(arm_ctrl.get_waist_yaw_relative_position())
        except Exception:
            has_waist = False
    waist_velocity = max(abs(start_waist) * 2.0, 1.0)

    duration = max(float(duration), 0.1)
    frequency = max(float(frequency), 1.0)
    n_steps = max(int(duration * frequency), 1)
    dt = duration / n_steps
    logger_mp.info(
        "[teleop arm shutdown] smooth go-home: lowering arms to home over %.2fs at %.0f Hz "
        "(waist_start=%.3f rad).",
        duration, frequency, start_waist,
    )
    for i in range(1, n_steps + 1):
        # cosine ease-in-out from 0 -> 1
        s = 0.5 - 0.5 * np.cos(np.pi * i / n_steps)
        q = start_q + (target_q - start_q) * s
        arm_ctrl.ctrl_dual_arm(q, tauff)
        if has_waist and abs(start_waist) > 1e-4:
            waist_rel = start_waist * (1.0 - s)
            try:
                arm_ctrl.ctrl_waist_yaw(waist_rel, limit=abs(start_waist) + 1e-3, velocity_limit=waist_velocity)
            except Exception as exc:
                logger_mp.debug("[teleop arm shutdown] waist ramp step failed: %s", exc)
                has_waist = False
        time.sleep(dt)
    return True

_ARM_JOINT_INDEX_BY_MODEL = {
    "G1_29": G1_29_JointArmIndex,
    "G1_23": G1_23_JointArmIndex,
    "H1_2": H1_2_JointArmIndex,
    "H1": H1_JointArmIndex,
    "H2": H2_JointArmIndex,
}


def _format_motor_temperature(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return "[" + ",".join(str(int(item)) for item in value) + "]"
    return str(int(value))


def _maybe_log_arm_joint_temperatures(args, arm_ctrl, last_log_time):
    """Log the latest DDS arm-joint temperatures without another DDS reader."""
    interval = max(0.0, float(args.joint_temperature_interval))
    now = time.monotonic()
    if interval <= 0.0 or arm_ctrl is None or now - last_log_time < interval:
        return last_log_time

    try:
        temperatures = arm_ctrl.get_current_motor_temperatures()
        joint_indices = _ARM_JOINT_INDEX_BY_MODEL[args.arm]
        left = []
        right = []
        for joint in joint_indices:
            value = temperatures[joint.value]
            if value is None:
                continue
            item = f"{joint.name}={_format_motor_temperature(value)}"
            (left if joint.name.startswith("kLeft") else right).append(item)
        logger_mp.info(
            "[arm joint temperature °C] robot=%s left={%s} right={%s}",
            args.arm,
            " ".join(left),
            " ".join(right),
        )
    except Exception as exc:
        logger_mp.warning("[arm joint temperature] read skipped: %s", exc)
    return now

# ---------------------------------------------------------------------------
# Policy inference: observation assembly and action execution
# ---------------------------------------------------------------------------
# The layout below is the contract with the training dataset. It is produced by
# lerobot/examples/port_datasets/port_h1_2_loop_kit.py from the loop_porting_kit
# recordings, so changing anything here silently feeds the policy a permuted vector.
POLICY_ARM_DIM = 7
POLICY_EE_DIM = 13                 # Inspire DG2 motor channels, raw units
POLICY_HEAD_DIM = 2
POLICY_VECTOR_DIM = 2 * POLICY_ARM_DIM + 2 * POLICY_EE_DIM + POLICY_HEAD_DIM  # 42
POLICY_SLICES = {
    "left_arm":   slice(0, 7),
    "left_hand":  slice(7, 20),
    "right_arm":  slice(20, 27),
    "right_hand": slice(27, 40),
    "head":       slice(40, 42),
}
POLICY_CAMERAS = ["head_left", "head_right", "wrist_left", "wrist_right"]
# --ee values whose action channel is 13 raw DG2 motor targets per hand.
POLICY_SUPPORTED_EE = ("inspire_dg2", "rh5dg2_dfx")
DG2_RAW_MIN = 0.0
DG2_RAW_MAX = 2040.0


def _policy_state_vector(arm_q, left_ee_state, right_ee_state, head_yaw_pitch):
    """Assemble the 42-dim observation.state, or None when a source is unusable.

    Every element must be an *observed* value (not the last command), matching how
    `h1_2.observation.*` was recorded.
    """
    arm = np.asarray(arm_q, dtype=np.float64).reshape(-1)
    left_ee = np.asarray(left_ee_state, dtype=np.float64).reshape(-1)
    right_ee = np.asarray(right_ee_state, dtype=np.float64).reshape(-1)
    head = np.asarray(head_yaw_pitch, dtype=np.float64).reshape(-1)
    if arm.size != 2 * POLICY_ARM_DIM:
        return None, f"arm state has {arm.size} dims, expected {2 * POLICY_ARM_DIM}"
    if left_ee.size != POLICY_EE_DIM or right_ee.size != POLICY_EE_DIM:
        return None, f"hand state has {left_ee.size}/{right_ee.size} dims, expected {POLICY_EE_DIM}"
    if head.size != POLICY_HEAD_DIM:
        return None, f"head state has {head.size} dims, expected {POLICY_HEAD_DIM}"
    state = np.concatenate((arm[:POLICY_ARM_DIM], left_ee, arm[-POLICY_ARM_DIM:], right_ee, head))
    if not np.isfinite(state).all():
        return None, "state contains non-finite values"
    return state.astype(np.float32), None


def _policy_camera_frames(head_bgr, left_wrist_bgr, right_wrist_bgr, wrist_vflip=False):
    """Map the teleop camera streams onto the dataset's camera keys.

    The head stream is the binocular side-by-side ZED frame; the sidecar that wrote
    camera-head_left.mp4 / camera-head_right.mp4 split it the same way.
    """
    frames = {}
    if head_bgr is not None and head_bgr.ndim == 3 and head_bgr.shape[1] >= 2:
        split = head_bgr.shape[1] // 2
        frames["head_left"] = head_bgr[:, :split]
        frames["head_right"] = head_bgr[:, split:]
    if left_wrist_bgr is not None:
        frames["wrist_left"] = cv2.flip(left_wrist_bgr, 0) if wrist_vflip else left_wrist_bgr
    if right_wrist_bgr is not None:
        frames["wrist_right"] = right_wrist_bgr
    return frames


def _split_policy_action(action):
    """42-dim action -> {left_arm, left_hand, right_arm, right_hand, head}."""
    vec = np.asarray(action, dtype=np.float64).reshape(-1)
    return {name: vec[sl].copy() for name, sl in POLICY_SLICES.items()}


class _ArmGravityComp:
    """Gravity-compensation feedforward for the policy arm targets.

    The recording-time teleop sends tau_ff from the IK's reduced pinocchio model
    (pin.rnea) alongside every position target; sending the rollout q with tau=0
    leaves the PD loop fighting gravity alone, so the arms ride visibly lower
    than the recordings the policy was trained on. This rebuilds the same reduced
    model (cached pickle when available) and computes the pure gravity term
    rnea(q, 0, 0) per tick."""

    _IK_BY_ARM = {
        "G1_29": "G1_29_ArmIK",
        "G1_23": "G1_23_ArmIK",
        "H1_2": "H1_2_ArmIK",
        "H1": "H1_ArmIK",
        "H2": "H2_ArmIK",
    }

    def __init__(self, arm_type, expected_dim):
        import pinocchio as pin
        from teleop.robot_control import robot_arm_ik
        self._pin = pin
        ik = getattr(robot_arm_ik, self._IK_BY_ARM[arm_type])(Unit_Test=False, Visualization=False)
        self._model = ik.reduced_robot.model
        self._data = ik.reduced_robot.data
        if self._model.nq != expected_dim:
            raise ValueError(
                f"reduced model nq={self._model.nq} does not match the "
                f"{expected_dim}-dim dual-arm command."
            )
        self._zeros = np.zeros(self._model.nv)

    def tau(self, q):
        return np.asarray(
            self._pin.rnea(self._model, self._data, np.asarray(q, dtype=np.float64),
                           self._zeros, self._zeros)
        ).copy()


def _policy_arm_target(last_cmd, current_q, action_parts, max_step, meas_margin=0.15):
    """Rate-limited 14-dim arm target, integrated from the last *commanded* pose (like
    _policy_hand_target). Basing each step on the measured pose instead lets gravity
    droop drag the whole rollout a tracking-error below the recorded trajectory — the
    arms dip at [r] and hover low. The measured-pose clamp keeps the command within
    `meas_margin` rad of the real arms so the PD loop can never wind up a large error."""
    current = np.asarray(current_q, dtype=np.float64).reshape(-1)
    target = np.concatenate((action_parts["left_arm"], action_parts["right_arm"]))
    if current.size != target.size or not np.isfinite(current).all():
        return None
    base = np.asarray(last_cmd, dtype=np.float64).reshape(-1) if last_cmd is not None else current
    if base.size != target.size or not np.isfinite(base).all():
        base = current
    cmd = target if max_step <= 0.0 else base + np.clip(target - base, -max_step, max_step)
    if meas_margin > 0.0:
        cmd = np.clip(cmd, current - meas_margin, current + meas_margin)
    return cmd


def _policy_hand_target(last_cmd, action_slice, max_step):
    """Rate-limited raw DG2 target. Based on the last command, not the measured
    angle: the fingers lag the command by a lot and would stall the ramp."""
    target = np.clip(np.asarray(action_slice, dtype=np.float64), DG2_RAW_MIN, DG2_RAW_MAX)
    if last_cmd is None or max_step <= 0.0:
        return target
    return np.clip(last_cmd + np.clip(target - last_cmd, -max_step, max_step), DG2_RAW_MIN, DG2_RAW_MAX)


def _read_neck_yaw_pitch(neck_feedback, latched):
    """Actual pan/tilt angles for observation.state[40:42], latched across gaps."""
    if neck_feedback is None:
        return latched
    sample = neck_feedback.read_latest()
    if sample is None or sample.get("yaw_pitch") is None:
        return latched
    return [float(v) for v in sample["yaw_pitch"][:2]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 20.0, help = 'control and record \'s frequency')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1', 'H2'], default='H1_2', help='Select arm controller')
    parser.add_argument('--joint-temperature-interval', type=float, default=0.0, help='Seconds between arm-joint temperature logs from rt/lowstate. Set a positive value to enable (default: disabled).')
    parser.add_argument('--ee', type=str, choices=['inspire_dg2', 'rh5dg2_dfx'], default='inspire_dg2', help='End effector controller. Only the 13-channel Inspire DG2 hands match the trained action layout.')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--viewer-host-ip', type=str, default=None, help='Host IP advertised to the XR browser for the HTTPS/WSS viewer. If omitted, infer it from the route to --img-server-ip.')
    parser.add_argument('--network-interface', type=str, default='enp44s0', help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    # policy inference (see teleop/utils/policy_bridge.py and lerobot examples/h1_2_deploy)
    parser.add_argument('--policy-server', type=str, default='127.0.0.1:5601', help='host:port of the LeRobot policy bridge server running next to the checkpoint.')
    parser.add_argument('--policy-task', type=str, default='', help='Task instruction sent with every observation. Must match the string the dataset was ported with.')
    parser.add_argument('--policy-cameras', type=str, nargs='+', choices=POLICY_CAMERAS, default=POLICY_CAMERAS, help='Dataset camera keys to send. Must match the cameras the checkpoint was trained on.')
    parser.add_argument('--policy-image-encoding', type=str, choices=['jpg', 'raw'], default='jpg', help="Camera wire encoding. 'raw' avoids JPEG loss but needs ~10x the bandwidth.")
    parser.add_argument('--policy-jpeg-quality', type=int, default=95, help='JPEG quality for --policy-image-encoding jpg.')
    parser.add_argument('--policy-chunk-threshold', type=float, default=0.5, help='Send a new observation once the queued action chunk drops to this fraction.')
    parser.add_argument('--policy-aggregate', type=str, default='weighted_average', choices=['weighted_average', 'latest_only', 'average', 'conservative'], help='How an incoming chunk blends with actions already queued for the same step.')
    parser.add_argument('--policy-action-timeout', type=float, default=0.5, help='Seconds without a usable action before the robot holds its current pose.')
    parser.add_argument('--policy-step-mode', action='store_true', help='Synchronous single-step rollout: after [r], nothing is sent to the policy server until [n] is pressed. Each [n] ships exactly one observation, the returned action chunk is executed one action per control tick, and the robot holds the last target until the next [n].')
    parser.add_argument('--policy-step-actions', type=int, default=0, help='How many actions of the returned chunk each [n] executes (one per control tick) before holding. Default 0: the whole chunk.')
    parser.add_argument('--policy-arm-max-step', type=float, default=0.05, help='Max per-joint arm target change per control frame in radians. At 20Hz, 0.05 caps the arms at 1 rad/s.')
    parser.add_argument('--no-arm-gravity-comp', action='store_true', help='Send zero torque feedforward with the arm targets instead of the rnea gravity term. The recording teleop always sends gravity compensation, so disabling this makes the rollout arms sag below the trained trajectories; only useful for A/B testing.')
    parser.add_argument('--policy-arm-meas-margin', type=float, default=0.15, help='Max radians the arm command may lead the measured pose, for the rollout and the [h] home ramp. Keeps a stuck arm from winding up PD error; the recorded teleop lead peaks at ~0.13 rad, so raise this if the replayed arms ride visibly lower than the recording.')
    parser.add_argument('--policy-arm-startup-max-step', type=float, default=0.02, help='Tighter arm step limit during --policy-arm-startup-duration after [r].')
    parser.add_argument('--policy-home-q', type=str, default=None, help='Arm home pose for [h]/[p]: 14 comma-separated radians (left 7 + right 7), a recorded episode data.json (or its episode directory), or a LeRobot dataset root — the first frame arm state becomes the home target. Default: zeros with elbows at -0.3 rad.')
    parser.add_argument('--policy-arm-startup-duration', type=float, default=2.0, help='Seconds of extra-small arm steps right after the policy engages.')
    parser.add_argument('--policy-hand-max-step', type=float, default=250.0, help='Max per-actuator hand target change per control frame in raw DG2 units.')
    parser.add_argument('--policy-wrist-vflip', action='store_true', help='Vertically flip the left wrist frame before sending it. Only set this if the recorded camera-wrist_left.mp4 is flipped relative to the live stream.')
    parser.add_argument('--policy-log-rate', type=float, default=1.0, help='Policy action/queue debug log rate in Hz. Set 0 to disable.')
    parser.add_argument('--dry-run', action='store_true', help='Run the full observe/predict loop but publish nothing to the arms, hands or neck.')
    parser.add_argument('--start-sync-delay', type=float, default=START_SYNC_DELAY, help='Seconds to wait after pressing [r] before the policy takes over.')
    parser.add_argument('--camera', '--viewer-camera', dest='camera', type=str, choices=['head', 'left_wrist', 'right_wrist', 'both', 'both_wrist', 'head_and_wrist', 'head_wrist', 'all'], default='head', help='Camera stream shown in the 8012 XR viewer. Use head_and_wrist to show the head view with both wrist cameras below it.')
    parser.add_argument('--viewer-camera-mode', type=str, choices=['auto', 'webrtc', 'zmq', 'none'], default='zmq', help='Select how the 8012 XR viewer receives the selected camera.')
    parser.add_argument('--viewer-display-fps', type=float, default=15.0, help='XR JPEG push rate for ZMQ camera mode. Lower this on congested Wi-Fi.')
    parser.add_argument('--viewer-jpeg-quality', type=int, default=60, help='XR JPEG quality for ZMQ camera mode, from 1 to 100.')
    parser.add_argument('--no-left-wrist-camera-vflip', dest='left_wrist_camera_vflip', action='store_false', help='Disable vertical flip correction for the left wrist camera.')
    parser.add_argument('--right-wrist-camera-vflip', action='store_true', help='Enable vertical flip correction for the right wrist camera.')
    parser.add_argument('--hand-control-hz', type=float, default=20.0, help='RH5DG2 hand retarget/publish loop frequency.')
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
    parser.add_argument('--inspire-dg2-thumb-curl-gain', type=float, default=1.0, help='Inspire DG2 thumb landmark curl gain before thresholding.')
    parser.add_argument('--inspire-dg2-right-thumb-curl-gain', type=float, default=1.0, help='Extra Inspire DG2 right-thumb multiplier after --inspire-dg2-thumb-curl-gain.')
    parser.add_argument('--inspire-dg2-thumb-curl-threshold', type=float, default=0.12, help='Inspire DG2 thumb curl deadzone before the thumb boost activates.')
    parser.add_argument('--inspire-dg2-thumb-curl-strength', type=float, default=0.0, help='Inspire DG2 thumb boost strength toward the closed raw target.')
    parser.add_argument('--inspire-dg2-thumb-curl-log-rate', type=float, default=0.0, help='Inspire DG2 thumb curl boost debug log rate in Hz. Set 0 to disable.')
    parser.add_argument('--rh5dg2-log-throttle', type=float, default=0.0, help='RH5DG2 controller debug log rate in Hz. Set 0 to disable debug prints.')
    parser.add_argument('--rh5dg2-fast-mode', action=argparse.BooleanOptionalAction, default=True, help='Enable lower-latency RH5DG2 retarget settings.')
    parser.add_argument('--rh5dg2-retarget-mode', type=str, choices=['config', 'vector', 'dexpilot'], default='config', help='RH5DG2 retargeting mode. config uses assets/RH5DG2/RH5DG2.yml; dexpilot enables DexPilot without editing the YAML type.')
    parser.add_argument('--arm-lost-timeout', type=float, default=0.5, help='Seconds of XR session loss before the viewer reports tracking gone. Display only; the policy path does not depend on XR tracking.')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--headless', action=argparse.BooleanOptionalAction, default=True, help='Enable headless mode and disable Rerun recording visualization by default. Use --no-headless to enable Rerun.')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--viewer', action=argparse.BooleanOptionalAction, default=True, help='Start the XR viewer so an operator can watch the rollout. --no-viewer runs fully headless; camera frames still go to the policy.')
    parser.add_argument('--disable-arm', action='store_true', help='Do not publish the arm part of the action; hands and neck still follow the policy.')
    parser.add_argument('--disable-hand', action='store_true', help='Do not publish the hand part of the action; arms and neck still follow the policy.')
    parser.add_argument('--disable-body', action=argparse.BooleanOptionalAction, default=True, help='Disable high-level body/loco command publishing.')
    parser.add_argument('--enable-neck', action=argparse.BooleanOptionalAction, default=True, help='Send the policy head action to the external UDP neck controller.')
    parser.add_argument('--neck-host', type=str, default=None, help='External neck controller host. Defaults to --img-server-ip.')
    parser.add_argument('--neck-port', type=int, default=9091, help='External neck controller UDP command port.')
    parser.add_argument('--neck-yaw-limit', type=float, default=1.2, help='Absolute relative neck yaw command limit in radians.')
    parser.add_argument('--neck-pitch-limit', type=float, default=0.8, help='Absolute relative neck pitch command limit in radians.')
    parser.add_argument('--neck-smoothing-alpha', type=float, default=0.25, help='Neck command low-pass alpha from 0 to 1.')
    parser.add_argument('--neck-max-step', type=float, default=0.08, help='Maximum neck command change per control frame in radians.')
    parser.add_argument('--neck-command-deadband', type=float, default=0.04, help='Minimum yaw/pitch command change in radians before sending a new neck UDP command. Set 0 to send every frame.')
    parser.add_argument('--neck-feedback-port', type=int, default=9093, help='UDP port that receives actual neck yaw,pitch feedback from the pan/tilt process.')
    parser.add_argument('--neck-log-rate', type=float, default=0.0, help='Neck debug log rate in Hz. Set 0 to disable periodic neck logs.')
    parser.add_argument('--enable-waist-follow-neck', action='store_true', help='Make the H1_2 waist yaw slowly follow the neck yaw command, including in --motion mode.')
    parser.add_argument('--waist-follow-neck-invert', action=argparse.BooleanOptionalAction, default=True, help='Rotate the H1_2 waist OPPOSITE to the head/neck yaw direction. Use --no-waist-follow-neck-invert to make the waist turn the same way as the head.')
    parser.add_argument('--waist-yaw-gain', type=float, default=0.5, help='H1_2 waist-yaw gain applied to the neck yaw command.')
    parser.add_argument('--waist-yaw-limit', type=float, default=0.2618, help='H1_2 relative waist-yaw limit in radians; default is about 15 degrees.')
    parser.add_argument('--waist-yaw-velocity', type=float, default=0.25, help='H1_2 waist-yaw velocity limit in radians per second.')
    parser.add_argument('--enable-waist-keyboard', action='store_true', help='Rotate the H1_2 waist yaw with the [j]/[k] keys during teleop (sshkeyboard mode). Reuses --waist-yaw-limit and --waist-yaw-velocity; mutually exclusive with --enable-waist-follow-neck.')
    parser.add_argument('--log-waist-angle', action='store_true', help='Periodically log the current H1_2 waist angle (default: disabled).')
    parser.add_argument('--waist-keyboard-step', type=float, default=0.05, help='Radians the H1_2 waist yaw moves per [j]/[k] key press (default ~2.9 deg).')
    parser.add_argument('--waist-keyboard-invert', action='store_true', help='Swap the [j]/[k] waist rotation direction.')
    parser.add_argument('--skip-arm-go-home-on-exit', action='store_true', help='Do not command arm zero/home pose during shutdown.')
    parser.add_argument('--arm-shutdown-duration', type=float, default=3.0, help='Seconds to smoothly lower both arms (and H1_2 waist) from the current pose back to home on exit. 0 uses the legacy fast go-home.')
    parser.add_argument('--arm-shutdown-velocity', type=float, default=3.0, help='Arm velocity limit in rad/s applied as a safety cap during the smooth shutdown descent.')
    parser.add_argument('--rh5dg2-safe-mode', action=argparse.BooleanOptionalAction, default=True, help='Publish restricted RH5DG2 raw hand commands for real-hardware bringup.')
    parser.add_argument('--rh5dg2-active-hand', type=str, choices=['right', 'left', 'both'], default='both', help='RH5DG2 safe mode active DDS command hand.')
    parser.add_argument('--rh5dg2-enabled-indices', type=str, default='0,1,2,3,4,5,6,7,8,9,10,11,12', help='Comma-separated RH5DG2 raw actuator indices allowed in safe mode.')
    parser.add_argument('--rh5dg2-pitch-only', action='store_true', help='RH5DG2 safe preset: enable only pitch actuators 0,1,2,4,6,7,8,9.')
    parser.add_argument('--rh5dg2-gain', type=float, default=1.0, help='RH5DG2 safe raw command gain from baseline toward retarget target.')
    parser.add_argument('--rh5dg2-raw-close-direction', type=float, default=-1.0, help='RH5DG2 safe raw close direction; use -1 if raw pitch closes in the opposite direction.')
    parser.add_argument('--rh5dg2-safe-baseline', type=str, default='demo_open', help='RH5DG2 safe raw open baseline: demo_open, current, or 13 comma-separated angleSet values.')
    parser.add_argument('--rh5dg2-lock-spread-joints', action=argparse.BooleanOptionalAction, default=True, help='Hold RH5DG2 spread raw actuators 3 and 5 at the safe baseline value.')
    parser.add_argument('--rh5dg2-restore-repeat', type=int, default=80, help='RH5DG2 init-pose restore publish count on exit.')
    parser.add_argument('--rh5dg2-restore-interval', type=float, default=0.1, help='RH5DG2 init-pose restore publish interval in seconds.')
    parser.add_argument('--rh5dg2-restore-settle', type=float, default=0.75, help='Extra wait after RH5DG2 init-pose restore publishes.')
    parser.add_argument('--rh5dg2-curl-scale', type=float, default=1.2, help='Global RH5DG2 landmark curl multiplier before clipping.')
    parser.add_argument('--rh5dg2-index-curl-scale', type=float, default=1.2, help='Additional RH5DG2 index-finger curl multiplier before clipping.')
    parser.add_argument('--rh5dg2-middle-curl-scale', type=float, default=0.85, help='Additional RH5DG2 middle-finger curl multiplier before clipping.')
    parser.add_argument('--rh5dg2-ring-curl-scale', type=float, default=0.85, help='Additional RH5DG2 ring-finger curl multiplier before clipping.')
    parser.add_argument('--rh5dg2-little-curl-scale', type=float, default=0.85, help='Additional RH5DG2 little-finger curl multiplier before clipping.')
    parser.add_argument('--rh5dg2-thumb-curl-scale', type=float, default=3.0, help='Additional RH5DG2 thumb landmark curl multiplier before clipping.')
    parser.add_argument('--rh5dg2-thumb-curl-threshold', type=float, default=0.15, help='RH5DG2 thumb curl deadzone for curl-source thumb control; higher values make the thumb open more easily.')
    parser.add_argument('--rh5dg2-enable-thumb', action=argparse.BooleanOptionalAction, default=True, help='Enable RH5DG2 safe thumb actuators 10,11,12.')
    parser.add_argument('--rh5dg2-thumb-source', type=str, choices=['curl', 'raw'], default='curl', help='RH5DG2 safe thumb close-ratio source.')
    parser.add_argument('--rh5dg2-thumb10-scale', type=float, default=1.8, help='RH5DG2 thumb actuator 10 curl scale.')
    parser.add_argument('--rh5dg2-thumb11-scale', type=float, default=1.0, help='RH5DG2 thumb actuator 11 curl scale.')
    parser.add_argument('--rh5dg2-thumb12-scale', type=float, default=1.8, help='RH5DG2 thumb actuator 12 curl scale.')
    parser.add_argument('--rh5dg2-right-thumb-close-gain', type=float, default=2.8, help='Extra close-ratio gain for the RH5DG2 right thumb after thumb source selection.')
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
    parser.add_argument('--enable-rh5dg2-tactile-vr-overlay', action=argparse.BooleanOptionalAction, default=False, help='Show an RH5DG2 tactile heat HUD over the Vuer camera image.')
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
    # Config Loop streaming (on by default; see teleop/utils/loop_streamer.py). The thin
    # client only ships frames to the loop_porting_kit sidecar over TCP and quietly drops
    # them when the sidecar is absent, so keeping it on costs nothing and needs no flag.
    parser.add_argument('--loop', action='store_true', default=False, help='Stream robot state + cameras to the Config Loop sidecar. Off by default here: a rollout is not training data.')
    parser.add_argument('--no-loop', dest='loop', action='store_false', help='Disable Config Loop streaming')
    parser.add_argument('--loop-addr', type=str, default='127.0.0.1:5590', help='Config Loop sidecar TCP host:port')
    parser.add_argument('--loop-hand-name', type=str, default='rh5dg2', help='Config Loop source name for the separate RH5DG2 tactile stream.')
    # record mode and task info
    parser.add_argument('--record', action=argparse.BooleanOptionalAction, default=False, help='Record the rollout as an episode (same writer the teleop script uses). Off by default.')
    parser.add_argument('--screen-record', action='store_true', help='Record only the head camera view to MP4; toggle with s.')
    parser.add_argument('--screen-record-dir', type=str, default='./screen_records', help='Directory for head camera MP4 recordings.')
    parser.add_argument('--record-body-state', action=argparse.BooleanOptionalAction, default=True, help='Record full robot/body qpos from arm controller lowstate when available.')
    parser.add_argument('--record-depth', action=argparse.BooleanOptionalAction, default=True, help='Compute and save head stereo depth while recording (ZED factory calibration + SGBM on this PC; robot side unchanged).')
    parser.add_argument('--record-depth-scale', type=float, default=0.5, help='Depth computation scale relative to one eye image (0.5 -> 640x360 from 1280x720).')
    parser.add_argument('--zed-calib', type=str, default='', help='Path enable-neckto the ZED factory calibration .conf. Default: assets/zed_calib/SN19294463.conf in this repo.')
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
    parser.add_argument('--task-desc', type = str, default = 'distance:o', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')

    args = parser.parse_args()
    if args.viewer_display_fps <= 0:
        raise ValueError("--viewer-display-fps must be greater than zero.")
    if not 1 <= args.viewer_jpeg_quality <= 100:
        raise ValueError("--viewer-jpeg-quality must be between 1 and 100.")
    if not args.policy_server:
        raise ValueError("--policy-server must not be empty.")
    if not args.policy_cameras:
        raise ValueError("--policy-cameras must list at least one camera.")
    if not 0.0 <= args.policy_chunk_threshold <= 1.0:
        raise ValueError("--policy-chunk-threshold must be between 0 and 1.")
    if args.policy_action_timeout <= 0.0:
        raise ValueError("--policy-action-timeout must be greater than zero.")
    if args.policy_arm_max_step < 0.0 or args.policy_arm_startup_max_step < 0.0:
        raise ValueError("--policy-arm-max-step and --policy-arm-startup-max-step must be zero or greater.")
    if args.policy_arm_meas_margin < 0.0:
        raise ValueError("--policy-arm-meas-margin must be zero or greater (0 disables the clamp).")
    if args.policy_step_actions < 0:
        raise ValueError("--policy-step-actions must be zero (whole chunk) or greater.")
    if args.policy_step_mode:
        STEP_MODE_ENABLED = True
        logger_mp.info(
            "[policy step] single-step mode: after [r], press [n] to run one inference; the robot "
            "executes %s of the returned chunk, then holds until the next [n].",
            "all actions" if args.policy_step_actions == 0 else f"the first {args.policy_step_actions} action(s)",
        )

    POLICY_HOME_ARM_Q = _resolve_policy_home_q(args.policy_home_q)
    if POLICY_HOME_ARM_Q is not None:
        logger_mp.info(
            "[policy home] home pose from --policy-home-q (%s):\n  left  %s\n  right %s",
            args.policy_home_q,
            np.round(POLICY_HOME_ARM_Q[:7], 4).tolist(),
            np.round(POLICY_HOME_ARM_Q[7:], 4).tolist(),
        )
    else:
        logger_mp.info("[policy home] no --policy-home-q given; [h] uses the built-in fallback pose (zeros, elbows -0.3 rad).")
    if args.policy_arm_startup_duration < 0.0:
        raise ValueError("--policy-arm-startup-duration must be zero or greater.")
    if args.policy_hand_max_step < 0.0:
        raise ValueError("--policy-hand-max-step must be zero or greater.")
    if not 1 <= args.policy_jpeg_quality <= 100:
        raise ValueError("--policy-jpeg-quality must be between 1 and 100.")
    if args.policy_log_rate < 0.0:
        raise ValueError("--policy-log-rate must be zero or greater.")
    if args.ee not in POLICY_SUPPORTED_EE:
        raise ValueError(
            f"--ee {args.ee} has no 13-channel raw command path; the trained action layout "
            f"needs one of {POLICY_SUPPORTED_EE}."
        )
    if args.start_sync_delay < 0.0:
        raise ValueError("--start-sync-delay must be zero or greater.")
    START_SYNC_DELAY = float(args.start_sync_delay)
    if not 0.0 <= args.neck_smoothing_alpha <= 1.0:
        raise ValueError("--neck-smoothing-alpha must be between 0 and 1.")
    if args.neck_yaw_limit <= 0.0 or args.neck_pitch_limit <= 0.0:
        raise ValueError("--neck-yaw-limit and --neck-pitch-limit must be greater than zero.")
    if args.neck_max_step < 0.0:
        raise ValueError("--neck-max-step must be zero or greater.")
    if args.neck_command_deadband < 0.0:
        raise ValueError("--neck-command-deadband must be zero or greater.")
    if args.neck_log_rate < 0.0:
        raise ValueError("--neck-log-rate must be zero or greater.")
    if "head_left" not in args.policy_cameras and "head_right" not in args.policy_cameras:
        logger_mp.warning("[policy obs] no head camera selected; the policy sees wrist views only.")
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
        if args.inspire_dg2_thumb_curl_gain < 0.0 or args.inspire_dg2_right_thumb_curl_gain < 0.0:
            raise ValueError("--inspire-dg2-thumb-curl-gain and --inspire-dg2-right-thumb-curl-gain must be non-negative.")
        if not 0.0 <= args.inspire_dg2_thumb_curl_strength <= 1.0:
            raise ValueError("--inspire-dg2-thumb-curl-strength must be between 0 and 1.")
        if not 0.0 <= args.inspire_dg2_thumb_curl_threshold <= 0.95:
            raise ValueError("--inspire-dg2-thumb-curl-threshold must be between 0 and 0.95.")
        if args.inspire_dg2_thumb_curl_log_rate < 0.0:
            raise ValueError("--inspire-dg2-thumb-curl-log-rate must be non-negative.")
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
    if args.enable_waist_keyboard:
        if args.enable_waist_follow_neck:
            raise ValueError("--enable-waist-keyboard and --enable-waist-follow-neck cannot be used together.")
        if args.arm != "H1_2" or args.disable_arm:
            raise ValueError("--enable-waist-keyboard requires --arm H1_2 with arm control enabled.")
        if args.waist_keyboard_step <= 0.0:
            raise ValueError("--waist-keyboard-step must be greater than zero.")
        if args.ipc:
            raise ValueError("--enable-waist-keyboard needs sshkeyboard input; do not combine it with --ipc.")
        WAIST_KEY_ENABLED = True
        WAIST_KEY_STEP = float(args.waist_keyboard_step)
        WAIST_KEY_LIMIT = float(abs(args.waist_yaw_limit))
        WAIST_KEY_INVERT = bool(args.waist_keyboard_invert)
        logger_mp.info(
            "[teleop waist keyboard] enabled: [j]/[k] step=%.4f rad (%.1f deg), limit=+/-%.4f rad (%.1f deg), velocity=%.3f rad/s, invert=%s",
            WAIST_KEY_STEP, np.degrees(WAIST_KEY_STEP),
            WAIST_KEY_LIMIT, np.degrees(WAIST_KEY_LIMIT),
            args.waist_yaw_velocity, WAIST_KEY_INVERT,
        )
    if args.record and args.screen_record:
        raise ValueError("--record and --screen-record cannot be enabled together.")
    rh5dg2_safe_baseline = _resolve_rh5dg2_safe_baseline(args.rh5dg2_safe_baseline)
    # The arm controller is always needed: observation.state[0:7]/[20:27] are its measured
    # joint angles, so --disable-arm only suppresses the command publish.
    preserve_zero_ready_mode = False
    rh5dg2_enabled_indices = _parse_int_list(args.rh5dg2_enabled_indices)
    if args.rh5dg2_pitch_only:
        rh5dg2_enabled_indices = [0, 1, 2, 4, 6, 7, 8, 9]
    if args.rh5dg2_enable_thumb:
        if rh5dg2_enabled_indices is None:
            rh5dg2_enabled_indices = [0, 1, 2, 4, 6, 7, 8, 9]
        rh5dg2_enabled_indices = sorted(set(rh5dg2_enabled_indices) | {10, 11, 12})
    if args.rh5dg2_fast_mode:
        if args.rh5dg2_log_throttle == 1.0:
            args.rh5dg2_log_throttle = 2.0
        if args.hand_debug_rate == 1.0:
            args.hand_debug_rate = 0.5
    logger_mp.debug(f"args: {args}")
    logger_mp.warning(
        "[policy safety mode] dry_run=%s disable_arm=%s disable_hand=%s enable_neck=%s disable_body=%s motion_requested=%s",
        args.dry_run,
        args.disable_arm,
        args.disable_hand,
        args.enable_neck,
        args.disable_body,
        args.motion,
    )
    logger_mp.info(
        "[policy action limits] arm_max_step=%.4f rad/frame (%.2f rad/s at %.1fHz) "
        "startup=%.4f for %.1fs hand_max_step=%.1f raw/frame action_timeout=%.2fs",
        args.policy_arm_max_step,
        args.policy_arm_max_step * args.frequency,
        args.frequency,
        args.policy_arm_startup_max_step,
        args.policy_arm_startup_duration,
        args.policy_hand_max_step,
        args.policy_action_timeout,
    )
    if args.ee in ("rh5dg2_dfx", "rh5dg2_ftp") and args.rh5dg2_safe_mode:
        logger_mp.warning("RH5DG2 SAFE MODE")
        logger_mp.warning("enabled actuators: %s", " ".join(str(i) for i in (rh5dg2_enabled_indices or [0, 1, 2, 4, 6, 7, 8, 9])))
        logger_mp.warning("disabled actuators: %s", " ".join(str(i) for i in sorted(set(range(13)) - set(rh5dg2_enabled_indices or [0, 1, 2, 4, 6, 7, 8, 9]))))
        logger_mp.warning("arm control: %s", "OFF" if args.disable_arm else "ON")
        logger_mp.warning("body control: %s", "OFF" if args.disable_body else "ON")
        logger_mp.warning("gain: %.3f", args.rh5dg2_gain)
        logger_mp.warning("active hand: %s", args.rh5dg2_active_hand)
        logger_mp.warning("lock spread joints: %s", args.rh5dg2_lock_spread_joints)
        logger_mp.warning(
            "curl scale: global=%.3f index=%.3f middle=%.3f ring=%.3f little=%.3f thumb=%.3f thumb_threshold=%.3f",
            args.rh5dg2_curl_scale,
            args.rh5dg2_index_curl_scale,
            args.rh5dg2_middle_curl_scale,
            args.rh5dg2_ring_curl_scale,
            args.rh5dg2_little_curl_scale,
            args.rh5dg2_thumb_curl_scale,
            args.rh5dg2_thumb_curl_threshold,
        )
        logger_mp.warning(
            "thumb: enabled=%s source=%s scales={10: %.3f, 11: %.3f, 12: %.3f} right_close_gain=%.3f",
            args.rh5dg2_enable_thumb,
            args.rh5dg2_thumb_source,
            args.rh5dg2_thumb10_scale,
            args.rh5dg2_thumb11_scale,
            args.rh5dg2_thumb12_scale,
            args.rh5dg2_right_thumb_close_gain,
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
    arm_gravity_comp = None
    hand_ctrl = None
    img_client = None
    tv_wrapper = None
    ipc_server = None
    listen_keyboard_thread = None
    screen_record_writer = None
    screen_record_path = None
    recorder = None
    depth_worker = None
    sim_state_subscriber = None
    neck_ctrl = None
    neck_feedback = None
    rh56f1_tactile_reader = None
    rh5dg2_tactile_udp = None
    rh5dg2_tactile_heat_mappers = {}
    policy_client = None
    joint_temperature_last_log = 0.0
    audio_udp_receiver = None
    episode_audio_recorder = None
    loop_robot = None
    loop_hand = None
    loop_camera = None

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
                command_deadband=args.neck_command_deadband,
            )
            logger_mp.info(
                f"[policy neck] enabled target={args.neck_host or args.img_server_ip}:{args.neck_port} "
                f"yaw_limit={args.neck_yaw_limit:.3f} pitch_limit={args.neck_pitch_limit:.3f} "
                f"alpha={args.neck_smoothing_alpha:.3f} max_step={args.neck_max_step:.3f} "
                f"deadband={args.neck_command_deadband:.3f} input_source=policy_action[40:42]"
            )
        # actual neck yaw,pitch feedback (UDP from zed_pantilt). Independent of
        # --enable-neck: the pan/tilt process may run standalone on the robot, and
        # the Config Loop head channels only need this passive listener.
        if args.neck_feedback_port > 0:
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

        policy_client = PolicyBridgeClient(
            addr=args.policy_server,
            task=args.policy_task,
            cameras=args.policy_cameras,
            state_dim=POLICY_VECTOR_DIM,
            action_dim=POLICY_VECTOR_DIM,
            fps=args.frequency,
            chunk_size_threshold=args.policy_chunk_threshold,
            aggregate_fn_name=args.policy_aggregate,
            image_encoding=args.policy_image_encoding,
            jpeg_quality=args.policy_jpeg_quality,
        )
        policy_client.start()
        logger_mp.info(
            "[policy bridge] server=%s task=%r cameras=%s state_dim=%s enc=%s aggregate=%s",
            args.policy_server, args.policy_task, args.policy_cameras,
            POLICY_VECTOR_DIM, args.policy_image_encoding, args.policy_aggregate,
        )

        if args.enable_rh5dg2_tactile_vr_overlay:
            if rh5dg2_tactile_udp is None:
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

        # televuer_wrapper: purely a monitor here — the policy never reads XR input, but an
        # operator watching the rollout through the headset is worth the extra process.
        if args.viewer:
            tv_wrapper = TeleVuerWrapper(use_hand_tracking=False,
                                         binocular=selected_camera_config['binocular'],
                                         img_shape=selected_img_shape,
                                         display_fps=args.viewer_display_fps,
                                         jpeg_quality=args.viewer_jpeg_quality,
                                         display_mode=args.display_mode,
                                         zmq=viewer_zmq,
                                         webrtc=viewer_webrtc,
                                         webrtc_url=webrtc_offer_url,
                                         arm_reference_mode="head_yaw",
                                         tracking_timeout=args.arm_lost_timeout,
                                         session_timeout=max(2.0, args.arm_lost_timeout * 4.0),
                                         )
        else:
            xr_need_local_img = False
            logger_mp.info("[policy viewer] --no-viewer: XR display is off; camera frames still feed the policy.")

        # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        if args.motion:
            pass
        else:
            motion_switcher = MotionSwitcher()
            status, result = motion_switcher.Enter_Debug_Mode()
            logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

        # arm — always constructed: observation.state needs the measured joint angles even
        # when --disable-arm suppresses the command publish.
        if args.arm == "G1_29":
            arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "G1_23":
            arm_ctrl = G1_23_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1_2":
            arm_ctrl = H1_2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1":
            arm_ctrl = H1_ArmController(simulation_mode=args.sim)
        elif args.arm == "H2":
            arm_ctrl = H2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)

        # Gravity feedforward for the policy targets, matching what the
        # recording teleop sent with every ctrl_dual_arm() call.
        arm_gravity_comp = None
        if arm_ctrl is not None and not args.disable_arm and not args.no_arm_gravity_comp:
            try:
                arm_dim = len(arm_ctrl.get_current_dual_arm_q())
                arm_gravity_comp = _ArmGravityComp(args.arm, arm_dim)
                logger_mp.info("[policy gravity] rnea gravity feedforward enabled (%s, %s joints).",
                               args.arm, arm_dim)
            except Exception as exc:
                logger_mp.warning(
                    "[policy gravity] could not build the gravity model (%s); arm targets will be "
                    "sent with zero tau and may sag below the recorded pose.", exc,
                )

        # end-effector
        if args.ee == "rh5dg2_dfx":
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
                                              middle_curl_scale=args.rh5dg2_middle_curl_scale,
                                              ring_curl_scale=args.rh5dg2_ring_curl_scale,
                                              little_curl_scale=args.rh5dg2_little_curl_scale,
                                              thumb_curl_scale=args.rh5dg2_thumb_curl_scale,
                                              thumb_curl_threshold=args.rh5dg2_thumb_curl_threshold,
                                              enable_thumb=args.rh5dg2_enable_thumb,
                                              thumb_source=args.rh5dg2_thumb_source,
                                              thumb10_scale=args.rh5dg2_thumb10_scale,
                                              thumb11_scale=args.rh5dg2_thumb11_scale,
                                              thumb12_scale=args.rh5dg2_thumb12_scale,
                                              right_thumb_close_gain=args.rh5dg2_right_thumb_close_gain,
                                              lock_spread_joints=args.rh5dg2_lock_spread_joints,
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
                thumb_curl_gain=args.inspire_dg2_thumb_curl_gain,
                right_thumb_curl_gain=args.inspire_dg2_right_thumb_curl_gain,
                thumb_curl_threshold=args.inspire_dg2_thumb_curl_threshold,
                thumb_curl_strength=args.inspire_dg2_thumb_curl_strength,
                thumb_curl_log_rate=args.inspire_dg2_thumb_curl_log_rate,
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

        logger_mp.info(f"[policy ee] ee={args.ee} hand_controller={hand_ctrl.__class__.__name__}")
        if not hasattr(hand_ctrl, "set_raw_command"):
            raise RuntimeError(
                f"{hand_ctrl.__class__.__name__} has no set_raw_command(); the policy hand action "
                "cannot bypass the landmark retargeting path."
            )
        # Until the first action arrives the landmark arrays stay all-zero, so the hand
        # controller reports "input not ready" and holds its open baseline. The raw
        # override takes over from the first set_raw_command() call onwards.

        # Unified EE handles for Config Loop streaming.
        loop_ee_state_array = dual_hand_state_array
        loop_ee_action_array = dual_hand_action_array
        loop_ee_lock = dual_hand_data_lock

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
                    "input_mode": "policy",
                    "simulation": args.sim,
                    "disable_arm": args.disable_arm,
                    "disable_hand": args.disable_hand,
                    "disable_body": args.disable_body,
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
                    "start_sync_delay": args.start_sync_delay,
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
                "policy": {
                    "server": args.policy_server,
                    "task": args.policy_task,
                    "cameras": list(args.policy_cameras),
                    "state_dim": POLICY_VECTOR_DIM,
                    "action_dim": POLICY_VECTOR_DIM,
                    "image_encoding": args.policy_image_encoding,
                    "aggregate": args.policy_aggregate,
                    "chunk_threshold": args.policy_chunk_threshold,
                    "arm_max_step": args.policy_arm_max_step,
                    "hand_max_step": args.policy_hand_max_step,
                    "action_timeout": args.policy_action_timeout,
                    "step_mode": args.policy_step_mode,
                    "step_actions": args.policy_step_actions,
                    "wrist_vflip": args.policy_wrist_vflip,
                    "gravity_comp": arm_gravity_comp is not None,
                    "dry_run": args.dry_run,
                },
                "neck": {
                    "enabled": args.enable_neck,
                    "input_source": "policy_action",
                    "command_host": args.neck_host or args.img_server_ip,
                    "command_port": args.neck_port,
                    "feedback_port": args.neck_feedback_port,
                    "command_deadband": args.neck_command_deadband,
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
            if args.record_depth:
                try:
                    if cv2 is None:
                        raise RuntimeError("cv2 unavailable")
                    head_cfg = camera_config['head_camera']
                    if not head_cfg.get('binocular'):
                        raise RuntimeError("head camera is not binocular; no stereo pair for depth")
                    from teleop.utils.stereo_depth import StereoDepthEstimator, AsyncStereoDepthWorker
                    head_shape = head_cfg['image_shape']  # [H, W*2] side-by-side
                    depth_eye_size = (int(head_shape[1]) // 2, int(head_shape[0]))
                    depth_calib_path = args.zed_calib or os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        '..', 'assets', 'zed_calib', 'SN19294463.conf')
                    depth_worker = AsyncStereoDepthWorker(
                        StereoDepthEstimator(depth_calib_path, depth_eye_size,
                                             scale=args.record_depth_scale))
                    logger_mp.info(
                        "[teleop record depth] enabled eye=%s scale=%.2f calib=%s",
                        depth_eye_size, args.record_depth_scale, depth_calib_path)
                except Exception as exc:
                    depth_worker = None
                    logger_mp.warning("[teleop record depth] disabled: %s", exc)

        # Config Loop streaming (optional sink). Robot-state gRPC is declared
        # here; sends continue non-blocking inside the control loop.
        if args.loop:
            from teleop.utils.loop_streamer import LoopRobotStreamer, LoopHandStreamer, LoopCameraStreamer
            # probe full-body qpos once to size the fixed-width body channel
            # (same source --record-body-state uses; empty probe -> no channel)
            loop_body_probe = _read_body_qpos(arm_ctrl, enabled=args.record_body_state)
            if args.record_body_state and not loop_body_probe:
                logger_mp.warning("[loop] body qpos unavailable at connect; body channel disabled for this run.")
            loop_robot = LoopRobotStreamer(
                args.loop_addr,
                args.ee,
                args.frequency,
                arm=args.arm,
                head_dim=2 if (neck_ctrl is not None or neck_feedback is not None) else 0,
                body_dim=len(loop_body_probe),
                raw_head_dim=2 if neck_ctrl is not None else 0,
            )
            loop_robot.connect()
            # session snapshot -> sidecar writes teleop_session.json next to each
            # Loop-saved episode (full args dump covers everything data.json's
            # info.recording block derives from; the structured block rides along
            # when --record built it).
            loop_robot.set_session_metadata({
                "script": os.path.basename(__file__),
                "argv": sys.argv[1:],
                "args": dict(sorted(vars(args).items())),
                "recording": recording_metadata if args.record else None,
            })
            if args.ee in ("inspire_dg2", "rh5dg2_dfx", "rh5dg2_ftp"):
                loop_hand = LoopHandStreamer(
                    args.loop_addr,
                    args.frequency,
                    hand_key=args.loop_hand_name,
                    source_key=args.loop_hand_name,
                )
                loop_hand.connect()
            loop_camera = LoopCameraStreamer(args.loop_addr, camera_config)
            loop_camera.connect()
            logger_mp.info(f"[loop] streaming robot state + tactile + cameras to Config Loop at {args.loop_addr}")

        logger_mp.info("----------------------------------------------------------------")
        logger_mp.info(f"🟢  Press [r] to hand control to the policy after {START_SYNC_DELAY:.1f}s.")
        logger_mp.info("🟠  Press [h] BEFORE [r] to send the arms/hands/neck to the home pose the episodes started from.")
        logger_mp.info("🟣  Press [p] while running to PAUSE the rollout and return the arms to the ready pose; press [r] to resume.")
        if args.dry_run:
            logger_mp.warning("🧪  --dry-run: actions are logged only. Nothing is published to arms, hands or neck.")
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
        logger_mp.info(f"[policy ready state] READY={READY} START={START} STOP={STOP} disable_arm={args.disable_arm}")
        prestart_loop_count = 0
        prestart_bridge_log = 0.0
        home_cmd_q = None       # integrated arm command for the [h]/[p] home ramp
        policy_arm_cmd_q = None  # integrated arm command for the policy rollout
        neck_log_last_ts = 0.0
        neck_log_interval = 1.0 / args.neck_log_rate if args.neck_log_rate > 0 else None
        while not START and not STOP: # wait for start or stop signal.
            prestart_loop_count += 1
            time.sleep(0.033)
            joint_temperature_last_log = _maybe_log_arm_joint_temperatures(
                args, arm_ctrl, joint_temperature_last_log
            )
            if GO_HOME:
                # [h] pressed before the first [r]: the control loop below has not started
                # yet, so run the home move from here — otherwise [h] would only work after
                # a [p] pause and the pre-rollout homing would silently do nothing.
                home_reached, neck_log_last_ts, home_cmd_q = _go_home_tick(
                    args, arm_ctrl, hand_ctrl, neck_ctrl, neck_feedback,
                    prestart_loop_count, neck_log_last_ts, neck_log_interval,
                    cmd_q=home_cmd_q, gravity_comp=arm_gravity_comp,
                )
                if home_reached:
                    with STATE_LOCK:
                        GO_HOME = False
                    logger_mp.info("[policy home] home pose reached. Press [r] to start the policy.")
                elif prestart_loop_count % 40 == 0:
                    logger_mp.info("[policy home] easing to the home pose...")
            else:
                home_cmd_q = None  # restart the command ramp from the live pose on the next [h]
            if maybe_start_delayed_run() and not policy_client.ready:
                # Starting without a policy server would leave the arms holding forever;
                # say so once instead of silently idling.
                logger_mp.warning(
                    "[policy run] started, but the policy server at %s is not connected yet. "
                    "The robot holds its pose until the first action chunk arrives.",
                    args.policy_server,
                )
            now = time.monotonic()
            if now - prestart_bridge_log >= 2.0:
                prestart_bridge_log = now
                if not policy_client.ready:
                    logger_mp.warning(
                        "[policy bridge] waiting for %s (connected=%s). Start policy_bridge_server.py on the GPU host.",
                        args.policy_server, policy_client.connected,
                    )
            # feed Config Loop cameras during the pre-start wait too, so the Loop
            # previews go READY before [r] is pressed (camera frames only; the
            # robot-step source stays silent until tracking actually starts).
            if loop_camera is not None:
                if camera_config['head_camera']['enable_zmq']:
                    prestart_loop_img = img_client.get_head_frame()
                    if prestart_loop_img is not None:
                        loop_camera.set_head(prestart_loop_img.bgr, prestart_loop_img.jpg)
                if camera_config['left_wrist_camera']['enable_zmq']:
                    prestart_loop_img = img_client.get_left_wrist_frame()
                    if prestart_loop_img is not None:
                        loop_camera.set_left_wrist(prestart_loop_img.bgr, prestart_loop_img.jpg)
                if camera_config['right_wrist_camera']['enable_zmq']:
                    prestart_loop_img = img_client.get_right_wrist_frame()
                    if prestart_loop_img is not None:
                        loop_camera.set_right_wrist(prestart_loop_img.bgr, prestart_loop_img.jpg)
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

        logger_mp.info("---------------------🚀 policy rollout 🚀-------------------------")
        logger_mp.info(
            f"[policy ready state] READY={READY} START={START} STOP={STOP} "
            f"disable_arm={args.disable_arm} disable_hand={args.disable_hand} dry_run={args.dry_run}"
        )
        if args.disable_arm:
            logger_mp.warning("[policy arm] OFF: --disable-arm set; the arm part of the action is not published.")
        else:
            arm_ctrl.speed_gradual_max()
            # Seed the arm target so the first action eases up from where the arms already
            # are instead of snapping toward the zeros/home target. Prefer the held [h]
            # home command over the measured pose: re-commanding the measured (gravity-
            # drooped) q would visibly drop the arms the moment [r] is pressed.
            try:
                _seed_q = home_cmd_q
                if _seed_q is None:
                    _seed_q = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=np.float64).reshape(-1)
                if _seed_q.size and np.isfinite(_seed_q).all() and not args.dry_run:
                    _seed_tau = arm_gravity_comp.tau(_seed_q) if arm_gravity_comp is not None \
                        else np.zeros_like(_seed_q)
                    arm_ctrl.ctrl_dual_arm(_seed_q, _seed_tau)
                    policy_arm_cmd_q = _seed_q.copy()
            except Exception as _seed_exc:
                logger_mp.debug("[policy arm startup] q_target seed skipped: %s", _seed_exc)
            logger_mp.info(
                f"[policy arm startup safety] startup_duration={args.policy_arm_startup_duration:.3f}s "
                f"startup_max_step={args.policy_arm_startup_max_step:.4f}rad "
                f"steady_max_step={args.policy_arm_max_step:.4f}rad"
            )

        head_img = None
        left_wrist_img = None
        right_wrist_img = None
        left_wrist_bgr = None
        right_wrist_bgr = None
        record_save_pending = False
        viewer_frame_count = 0
        # policy rollout state
        policy_started_at = time.time()
        policy_head_latched = [0.0, 0.0]
        policy_last_left_hand_cmd = None
        policy_last_right_hand_cmd = None
        policy_last_action = None
        policy_last_action_time = 0.0
        policy_obs_skipped = 0
        policy_hold_logged = False
        policy_step_actions_left = 0  # step mode: actions of the pending chunk still to execute
        policy_step_got_action = False  # step mode: at least one action of this step's chunk arrived
        policy_step_executed = 0        # step mode: actions executed for the current [n]
        policy_log_interval = 1.0 / args.policy_log_rate if args.policy_log_rate > 0 else None
        policy_log_last_ts = 0.0

        # main loop. the robot now follows the policy's action chunks
        loop_count = 0
        neck_log_last_ts = 0.0
        neck_log_interval = 1.0 / args.neck_log_rate if args.neck_log_rate > 0 else None
        # latest neck yaw,pitch for the Config Loop robot step (head channels are
        # fixed-width, so latch the last known values across invalid-pose ticks)
        loop_head_state = [0.0, 0.0]
        loop_head_action = [0.0, 0.0]
        loop_raw_head = [0.0, 0.0]
        loop_body_q = []
        while not STOP:
            loop_count += 1
            start_time = time.time()
            joint_temperature_last_log = _maybe_log_arm_joint_temperatures(
                args, arm_ctrl, joint_temperature_last_log
            )
            neck_record = None
            left_wrist_bgr = None
            right_wrist_bgr = None
            # get image (the policy needs every camera every tick, independent of --record)
            if camera_config['head_camera']['enable_zmq']:
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
                    if args.record or args.screen_record:
                        viewer_bgr = _draw_record_indicator(
                            viewer_bgr, RECORD_RUNNING,
                            binocular=bool(selected_camera_config.get('binocular')),
                        )
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
                left_wrist_img = img_client.get_left_wrist_frame()
                if left_wrist_img is not None and left_wrist_img.bgr is not None:
                    # Store the camera frame exactly as received from the image server —
                    # the recordings the policy trained on were not reoriented either.
                    left_wrist_bgr = left_wrist_img.bgr

            if camera_config['right_wrist_camera']['enable_zmq']:
                right_wrist_img = img_client.get_right_wrist_frame()
                if right_wrist_img is not None and right_wrist_img.bgr is not None:
                    right_wrist_bgr = right_wrist_img.bgr

            # hand the freshest frames to Config Loop (control thread only swaps
            # references; JPEG handling + RTSP happen off-thread in loop_camera).
            # Pass teleimager's original .jpg too: monocular cams skip a
            # decode->re-encode round trip and packetize those bytes directly.
            if loop_camera is not None:
                if head_img is not None:
                    loop_camera.set_head(head_img.bgr, head_img.jpg)
                if left_wrist_img is not None:
                    loop_camera.set_left_wrist(left_wrist_img.bgr, left_wrist_img.jpg)
                if right_wrist_img is not None:
                    loop_camera.set_right_wrist(right_wrist_img.bgr, right_wrist_img.jpg)

            # feed the depth worker while recording (non-blocking; compute runs off-thread)
            if RECORD_RUNNING and depth_worker is not None and head_img is not None:
                try:
                    if head_img.bgr is not None:
                        depth_worker.submit(head_img.bgr)
                except Exception as exc:
                    if loop_count % 50 == 0:
                        logger_mp.warning("[teleop record depth] submit skipped: %s", exc)

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if recorder.create_episode():
                        if depth_worker is not None:
                            recorder.save_depth_info(depth_worker.estimator.intrinsics())
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
            maybe_start_delayed_run()

            # Robot observation. Read it every tick regardless of START so the pause path
            # and the recorder see the same numbers the policy would.
            if arm_ctrl is None:
                current_lr_arm_q = np.array([], dtype=np.float64)
                current_lr_arm_dq = np.array([], dtype=np.float64)
            else:
                current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
                current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()
            with dual_hand_data_lock:
                policy_left_ee_state = list(dual_hand_state_array[:POLICY_EE_DIM])
                policy_right_ee_state = list(dual_hand_state_array[-POLICY_EE_DIM:])
            policy_head_latched = _read_neck_yaw_pitch(neck_feedback, policy_head_latched)

            if not START:
                if PAUSE_TO_READY and not args.disable_arm and arm_ctrl is not None and not args.dry_run:
                    # [p] pause: ease the arms back to the ready pose while disengaged.
                    try:
                        _, home_cmd_q = _step_arms_toward_ready(
                            arm_ctrl, current_lr_arm_q, max(0.0, args.policy_arm_startup_max_step),
                            cmd_q=home_cmd_q, gravity_comp=arm_gravity_comp,
                        )
                    except Exception as exc:
                        if loop_count % 30 == 0:
                            logger_mp.warning("[policy pause] ready pose command skipped: %s", exc)
                    # Drop the raw override so the hands fall back to the controller's open
                    # baseline instead of freezing on the last action they were given.
                    hand_ctrl.clear_raw_command()
                    if loop_count % 100 == 0:
                        logger_mp.info("[policy pause] paused at ready pose. Press [r] to resume.")

                if GO_HOME:
                    # [h] home: same ready pose as the pause path, plus an open-hand
                    # baseline and a centred neck, so the rollout starts from the pose
                    # the episodes were recorded from.
                    home_reached, neck_log_last_ts, home_cmd_q = _go_home_tick(
                        args, arm_ctrl, hand_ctrl, neck_ctrl, neck_feedback,
                        loop_count, neck_log_last_ts, neck_log_interval,
                        current_arm_q=current_lr_arm_q,
                        cmd_q=home_cmd_q, gravity_comp=arm_gravity_comp,
                    )
                    if home_reached:
                        with STATE_LOCK:
                            GO_HOME = False
                        logger_mp.info("[policy home] home pose reached. Press [r] to start the policy.")
                    elif loop_count % 40 == 0:
                        logger_mp.info("[policy home] easing to the home pose...")
                if not GO_HOME and not PAUSE_TO_READY:
                    home_cmd_q = None  # restart the command ramp from the live pose next time
                # Drop queued actions so resuming never replays a chunk predicted from a
                # pre-pause observation.
                policy_client.reset()
                policy_last_left_hand_cmd = None
                policy_last_right_hand_cmd = None
                policy_arm_cmd_q = None
                if args.policy_step_mode:
                    # Step mode holds the last action forever, so a pause must forget it:
                    # resuming waits for a fresh [n] instead of ramping back to a stale target.
                    policy_last_action = None
                    policy_last_action_time = 0.0
                    policy_step_actions_left = 0
                    policy_step_got_action = False
                    policy_step_executed = 0
                    STEP_REQUEST = False
                policy_started_at = time.time()
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / args.frequency) - time_elapsed)
                time.sleep(sleep_time)
                continue

            # Engaged: seed the policy arm command from the held home/pause command (if
            # any) so the rollout starts from the pose [h] drove to instead of dipping
            # to the gravity-drooped measured pose; later [p]/[h] ramps restart live.
            if policy_arm_cmd_q is None and home_cmd_q is not None:
                policy_arm_cmd_q = home_cmd_q.copy()
            home_cmd_q = None

            # Stage the observation for the bridge (non-blocking; the sender thread
            # decides when the action queue is low enough to actually ship it).
            policy_state, policy_state_error = _policy_state_vector(
                current_lr_arm_q, policy_left_ee_state, policy_right_ee_state, policy_head_latched
            )
            policy_frames = _policy_camera_frames(
                None if head_img is None else head_img.bgr,
                left_wrist_bgr,
                right_wrist_bgr,
                wrist_vflip=args.policy_wrist_vflip,
            )
            missing_cams = [c for c in args.policy_cameras if c not in policy_frames]
            if policy_state is None or missing_cams:
                policy_obs_skipped += 1
                if loop_count % 20 == 0:
                    logger_mp.warning(
                        "[policy obs] skipped: state_error=%s missing_cameras=%s (total skipped=%s)",
                        policy_state_error, missing_cams, policy_obs_skipped,
                    )
            elif not args.policy_step_mode:
                policy_client.submit(policy_state, policy_frames)
            elif STEP_REQUEST:
                # [n]: ship exactly this observation. The queue is empty in step mode, so
                # the sender is always hungry and ships it immediately.
                policy_client.reset()
                policy_client.submit(policy_state, policy_frames)
                # 0 = the whole chunk: consume until the queue drains.
                policy_step_actions_left = args.policy_step_actions or (1 << 30)
                policy_step_got_action = False
                policy_step_executed = 0
                with STATE_LOCK:
                    STEP_REQUEST = False
                logger_mp.info("[policy step] observation sent; waiting for the action chunk...")

            # Pop the action for this tick. An empty queue means the server is late or
            # gone: hold the current pose rather than replaying a stale target.
            policy_action = None
            if not args.policy_step_mode:
                policy_action = policy_client.pop_action()
            elif policy_step_actions_left > 0:
                policy_action = policy_client.pop_action()
                if policy_action is not None:
                    if not policy_step_got_action:
                        logger_mp.info("[policy step] chunk arrived; executing one action per tick...")
                    policy_step_got_action = True
                    policy_step_executed += 1
                    policy_step_actions_left -= 1
                elif policy_step_got_action:
                    # The queue drained: the whole chunk has been executed.
                    policy_step_actions_left = 0
                if policy_step_actions_left == 0:
                    # Drop any unexecuted remainder and hold the last target until [n].
                    policy_client.reset()
                    policy_step_got_action = False
                    held = policy_action if policy_action is not None else policy_last_action
                    logger_mp.info(
                        "[policy step] %s action(s) executed and held. left_arm=%s right_arm=%s head=%s "
                        "— press [n] for the next step.",
                        policy_step_executed,
                        np.round(held[POLICY_SLICES['left_arm']], 3).tolist(),
                        np.round(held[POLICY_SLICES['right_arm']], 3).tolist(),
                        np.round(held[POLICY_SLICES['head']], 3).tolist(),
                    )
            now = time.time()
            if policy_action is not None:
                policy_last_action = policy_action
                policy_last_action_time = now
                policy_hold_logged = False
            policy_action_age = now - policy_last_action_time if policy_last_action_time else float("inf")
            # Step mode ignores the freshness timeout on purpose: the whole point is to
            # hold the last returned target while the operator inspects it.
            policy_engaged = policy_last_action is not None and (
                args.policy_step_mode or policy_action_age <= args.policy_action_timeout)
            if not policy_engaged and not policy_hold_logged:
                policy_hold_logged = True
                if args.policy_step_mode:
                    logger_mp.info("[policy step] engaged and holding the current pose. Press [n] to run one inference.")
                else:
                    logger_mp.warning(
                        "[policy action] no fresh action for %.2fs (timeout %.2fs); holding pose. "
                        "bridge=%s",
                        policy_action_age, args.policy_action_timeout, policy_client.stats(),
                    )
            action_parts = _split_policy_action(policy_last_action) if policy_engaged else None

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
                        if tv_wrapper is not None:
                            tv_wrapper.set_tactile_overlay(tactile_views)
                    elif tv_wrapper is not None:
                        tv_wrapper.set_tactile_overlay(None)
                except Exception as exc:
                    if loop_count % 30 == 0:
                        logger_mp.warning("[RH5DG2 tactile VR overlay] update skipped: %s", exc)

            # ---- neck: policy head action -> pan/tilt UDP command
            if action_parts is not None and not args.dry_run:
                neck_record, neck_log_last_ts = _update_neck_from_action(
                    args,
                    neck_ctrl,
                    neck_feedback,
                    action_parts["head"],
                    arm_ctrl,
                    loop_count,
                    neck_log_last_ts,
                    neck_log_interval,
                    allow_waist=True,
                )

            # keyboard-driven H1_2 waist yaw ([j]/[k]); independent of the policy
            if WAIST_KEY_ENABLED and arm_ctrl is not None and hasattr(arm_ctrl, "ctrl_waist_yaw") and not args.dry_run:
                try:
                    arm_ctrl.ctrl_waist_yaw(
                        WAIST_YAW_REL,
                        limit=args.waist_yaw_limit,
                        velocity_limit=args.waist_yaw_velocity,
                    )
                except Exception as exc:
                    if loop_count % 30 == 0:
                        logger_mp.warning("[teleop waist keyboard] command skipped: %s", exc)

            # periodic current-waist-angle log
            if args.log_waist_angle and loop_count % 50 == 0 and arm_ctrl is not None \
               and hasattr(arm_ctrl, "get_waist_yaw_relative_position") \
               and (WAIST_KEY_ENABLED or args.enable_waist_follow_neck):
                try:
                    waist_deg = float(np.degrees(arm_ctrl.get_waist_yaw_relative_position()))
                    logger_mp.info("현재 허리 각도 : %.1f 도", waist_deg)
                except Exception:
                    pass

            # ---- arms: policy joint targets, rate limited against the measured pose
            sol_q = np.asarray(current_lr_arm_q, dtype=np.float64).copy()
            sol_tauff = np.zeros_like(sol_q)
            arm_step_limit = args.policy_arm_max_step
            if args.policy_arm_startup_duration > 0.0 and \
               time.time() - policy_started_at < args.policy_arm_startup_duration:
                arm_step_limit = min(arm_step_limit, args.policy_arm_startup_max_step)
            if action_parts is not None and arm_ctrl is not None:
                arm_target = _policy_arm_target(
                    policy_arm_cmd_q, current_lr_arm_q, action_parts, arm_step_limit,
                    meas_margin=args.policy_arm_meas_margin,
                )
                if arm_target is None:
                    if loop_count % 20 == 0:
                        logger_mp.warning(
                            "[policy arm] target rejected: measured q=%s", _fmt_vec_debug(current_lr_arm_q)
                        )
                else:
                    sol_q = arm_target
                    policy_arm_cmd_q = arm_target
            elif args.policy_step_mode and policy_arm_cmd_q is not None:
                # Step mode waits between [n] presses with no action: keep commanding the
                # held target instead of the measured pose, otherwise gravity droop would
                # ratchet the arms down a tracking-error per tick while waiting.
                sol_q = np.asarray(policy_arm_cmd_q, dtype=np.float64).copy()
            if not args.disable_arm and arm_ctrl is not None and not args.dry_run:
                if arm_gravity_comp is not None:
                    sol_tauff = arm_gravity_comp.tau(sol_q)
                arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
                if loop_count % 50 == 0:
                    arm_write_ok = arm_ctrl.get_last_write_ok() if hasattr(arm_ctrl, "get_last_write_ok") else None
                    logger_mp.debug(
                        f"[policy arm publish] controller={arm_ctrl.__class__.__name__} "
                        f"sim={args.sim} write_ok={arm_write_ok} step_limit={arm_step_limit:.4f} "
                        f"target={_fmt_vec_debug(sol_q)}"
                    )

            # ---- hands: policy raw DG2 targets, bypassing the landmark retargeting path
            if action_parts is not None:
                policy_last_left_hand_cmd = _policy_hand_target(
                    policy_last_left_hand_cmd, action_parts["left_hand"], args.policy_hand_max_step
                )
                policy_last_right_hand_cmd = _policy_hand_target(
                    policy_last_right_hand_cmd, action_parts["right_hand"], args.policy_hand_max_step
                )
                if not args.disable_hand and not args.dry_run:
                    try:
                        hand_ctrl.set_raw_command(policy_last_left_hand_cmd, policy_last_right_hand_cmd)
                    except Exception as exc:
                        if loop_count % 30 == 0:
                            logger_mp.warning("[policy hand] raw command skipped: %s", exc)

            if policy_log_interval is not None and time.time() - policy_log_last_ts >= policy_log_interval:
                policy_log_last_ts = time.time()
                bridge = policy_client.stats()
                logger_mp.info(
                    "[policy step] engaged=%s queued=%s/%s rtt=%.0fms infer=%.0fms sent=%s recv=%s "
                    "dropped=%s obs_skipped=%s",
                    policy_engaged, bridge["queued"], bridge["chunk_size"], bridge["rtt_ms"],
                    bridge["infer_ms"], bridge["sent"], bridge["received"], bridge["dropped"],
                    policy_obs_skipped,
                )
                if action_parts is not None:
                    logger_mp.info(
                        "[policy action] l_arm=%s r_arm=%s l_hand=%s r_hand=%s head=%s",
                        np.round(action_parts["left_arm"], 3).tolist(),
                        np.round(action_parts["right_arm"], 3).tolist(),
                        np.round(action_parts["left_hand"], 0).tolist(),
                        np.round(action_parts["right_hand"], 0).tolist(),
                        np.round(action_parts["head"], 3).tolist(),
                    )

            # stream robot state to Config Loop (non-blocking; never breaks teleop)
            if loop_robot is not None:
                if loop_ee_lock is not None:
                    with loop_ee_lock:
                        loop_ee_state = list(loop_ee_state_array)
                        loop_ee_action = list(loop_ee_action_array)
                else:
                    loop_ee_state = []
                    loop_ee_action = []
                # hand-only / no-arm runs have empty arm arrays; send zeros so the
                # channel layout declared at connect() stays 7-dim per arm.
                if len(current_lr_arm_q) == 14:
                    loop_arm_q, loop_arm_dq, loop_arm_action = current_lr_arm_q, current_lr_arm_dq, sol_q
                else:
                    loop_arm_q = loop_arm_dq = loop_arm_action = np.zeros(14)
                # neck yaw,pitch -> head channels: observation = actual angles from
                # zed_pantilt (UDP feedback, latched inside the receiver); action =
                # the commanded target when teleop drives the neck (--enable-neck),
                # otherwise mirrors the observation (neck holding its position).
                if neck_feedback is not None:
                    _neck_fb = neck_feedback.read_latest()
                    if _neck_fb is not None and _neck_fb.get("yaw_pitch") is not None:
                        loop_head_state = [float(v) for v in _neck_fb["yaw_pitch"][:2]]
                if neck_record is not None and neck_record.get("command_yaw_pitch") is not None:
                    loop_head_action = [float(v) for v in neck_record["command_yaw_pitch"][:2]]
                else:
                    loop_head_action = list(loop_head_state)
                # operator XR head angles + full-body motor q (latched on failure)
                if neck_record is not None and neck_record.get("raw_head_yaw_pitch") is not None:
                    loop_raw_head = [float(v) for v in neck_record["raw_head_yaw_pitch"][:2]]
                _body_q = _read_body_qpos(arm_ctrl, enabled=args.record_body_state)
                if _body_q:
                    loop_body_q = _body_q
                loop_robot.send(time.time_ns() // 1000, loop_arm_q, loop_arm_dq,
                                loop_arm_action, loop_ee_state, loop_ee_action,
                                loop_head_state, loop_head_action,
                                loop_body_q, loop_raw_head)
                if loop_hand is not None:
                    loop_hand.send(time.time_ns() // 1000, loop_ee_state, loop_ee_action, latest_tactiles)

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
                with dual_hand_data_lock:
                    left_ee_state = dual_hand_state_array[:POLICY_EE_DIM]
                    right_ee_state = dual_hand_state_array[-POLICY_EE_DIM:]
                    left_hand_action = dual_hand_action_array[:POLICY_EE_DIM]
                    right_hand_action = dual_hand_action_array[-POLICY_EE_DIM:]
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
                    if depth_worker is not None:
                        latest_depth = depth_worker.get_latest()
                        if latest_depth is not None:
                            depths["depth_0"] = latest_depth[0]  # uint16 mm, left-eye aligned
                        elif loop_count % 50 == 0:
                            logger_mp.warning("[teleop record depth] no depth computed yet; frame saved without depth.")
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
                    if WAIST_KEY_ENABLED and arm_ctrl is not None and hasattr(arm_ctrl, "get_waist_yaw_relative_position"):
                        try:
                            waist_actual_rel = float(arm_ctrl.get_waist_yaw_relative_position())
                        except Exception:
                            waist_actual_rel = None
                        states["waist"] = {"yaw_relative": waist_actual_rel}
                        actions["waist"] = {"yaw_relative_target": WAIST_YAW_REL}
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
            if depth_worker is not None:
                stats = depth_worker.stats()
                logger_mp.info(
                    "[teleop record depth] stopping worker (frames=%s avg=%.1fms)",
                    stats["count"], stats["avg_ms"])
                depth_worker.stop()
                depth_worker = None
        except Exception as e:
            logger_mp.error(f"Failed to stop depth worker: {e}")

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
            if policy_client is not None:
                logger_mp.info("[policy bridge] final stats: %s", policy_client.stats())
                policy_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close policy bridge client: {e}")

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
                if args.arm_shutdown_duration > 0.0:
                    _smooth_arm_go_home(
                        arm_ctrl,
                        duration=args.arm_shutdown_duration,
                        velocity_cap=args.arm_shutdown_velocity,
                    )
                # Final settle: confirm home and, in motion mode, ramp down the arm_sdk weight.
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
            if loop_robot is not None:
                loop_robot.close()
        except Exception as e:
            logger_mp.error(f"Failed to close Loop robot streamer: {e}")

        try:
            if loop_hand is not None:
                loop_hand.close()
        except Exception as e:
            logger_mp.error(f"Failed to close Loop hand streamer: {e}")

        try:
            if loop_camera is not None:
                loop_camera.close()
        except Exception as e:
            logger_mp.error(f"Failed to close Loop camera streamer: {e}")

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
