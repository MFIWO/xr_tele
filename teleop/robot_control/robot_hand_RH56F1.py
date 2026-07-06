from __future__ import annotations

from multiprocessing import Array
import os
from pathlib import Path
import sys
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
import yaml

_LOCAL_DEX_RETARGETING_SRC = Path(__file__).resolve().parent / "dex-retargeting" / "src"
if _LOCAL_DEX_RETARGETING_SRC.exists() and str(_LOCAL_DEX_RETARGETING_SRC) not in sys.path:
    sys.path.insert(0, str(_LOCAL_DEX_RETARGETING_SRC))

from dex_retargeting import RetargetingConfig
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

import logging_mp

logger_mp = logging_mp.getLogger(__name__)

RH56F1_Num_Motors = 6
RH56F1_Num_Hand_Joints = RH56F1_Num_Motors * 2
RH56F1_Num_Retarget_Joints = 12
RH56F1_CMD_TOPIC = "rt/rh56f1/cmd"
RH56F1_STATE_TOPIC = "rt/rh56f1/state"
RH56F1_JOINT_SUFFIXES = [
    "thumb_1_joint",
    "thumb_2_joint",
    "thumb_3_joint",
    "thumb_4_joint",
    "index_1_joint",
    "index_2_joint",
    "middle_1_joint",
    "middle_2_joint",
    "ring_1_joint",
    "ring_2_joint",
    "little_1_joint",
    "little_2_joint",
]
RH56F1_THUMB_3_RATIO = 1.2953
RH56F1_THUMB_4_RATIO = 0.8962
RH56F1_ACTUATOR_NAMES = [
    "little",
    "ring",
    "middle",
    "index",
    "thumb_flexion",
    "thumb_abduction",
]
RH56F1_COMMAND_NAMES = [suffix.removesuffix("_joint") for suffix in RH56F1_JOINT_SUFFIXES]
RH56F1_FINGER_LANDMARKS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8, 9),
    "middle": (10, 11, 12, 13, 14),
    "ring": (15, 16, 17, 18, 19),
    "little": (20, 21, 22, 23, 24),
}
RH56F1_COMMAND_FINGER_INDICES = {
    "thumb": (0, 1, 2, 3),
    "index": (4, 5),
    "middle": (6, 7),
    "ring": (8, 9),
    "little": (10, 11),
}
RH56F1_FOLDED_ABS = np.array(
    [896.0, 896.0, 896.0, 896.0, 1120.0, 1802.0],
    dtype=np.float64,
)
RH56F1_OPEN_ABS = np.array(
    [1748.0, 1748.0, 1748.0, 1748.0, 1350.0, 600.0],
    dtype=np.float64,
)
RH56F1_SAFE_ABS_MIN = np.minimum(RH56F1_FOLDED_ABS, RH56F1_OPEN_ABS)
RH56F1_SAFE_ABS_MAX = np.maximum(RH56F1_FOLDED_ABS, RH56F1_OPEN_ABS)
RH56F1_DEPLOY_COMMAND_MODE = os.environ.get("RH56F1_DEPLOY_COMMAND_MODE", "normalized").strip().lower()
_RH56F1_RETARGET_MODES = ("vector", "dexpilot")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ASSETS_ROOT = _REPO_ROOT / "assets"
_CONFIG_PATH = _ASSETS_ROOT / "RH56F1" / "RH56F1.yml"
_HAND_FRAME_AXES = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def _normalize_rh56f1_retarget_mode(mode):
    raw = os.getenv("RH56F1_RETARGET_MODE") if mode is None else mode
    if raw is None:
        return None
    normalized = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "": None,
        "config": None,
        "default": None,
        "yaml": None,
        "none": None,
        "dex": "dexpilot",
        "dex_pilot": "dexpilot",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized is None:
        return None
    if normalized not in _RH56F1_RETARGET_MODES:
        raise ValueError(
            f"RH56F1 retarget mode must be one of config, {', '.join(_RH56F1_RETARGET_MODES)}; got {raw!r}"
        )
    return normalized


def _normalize(values, limits):
    values = np.asarray(values, dtype=np.float64)
    limits = np.asarray(limits, dtype=np.float64)
    spans = limits[:, 1] - limits[:, 0]
    return np.clip((values - limits[:, 0]) / np.maximum(spans, 1e-8), 0.0, 1.0)


def _retarget_to_actuator_flex(values, limits):
    flex = _normalize(values, limits)
    actuator_flex = np.array(
        [
            flex[10],
            flex[8],
            flex[6],
            flex[4],
            np.mean(flex[1:4]),
            flex[0],
        ],
        dtype=np.float64,
    )
    return np.clip(actuator_flex, 0.0, 1.0)


def _normalized_to_actuator_flex(values):
    flex = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    if flex.size == RH56F1_Num_Motors:
        return flex
    if flex.size != RH56F1_Num_Retarget_Joints:
        raise ValueError(f"RH56F1 expected 6 or 12 values, got {flex.size}")
    return np.array(
        [
            flex[10],
            flex[8],
            flex[6],
            flex[4],
            np.mean(flex[1:4]),
            flex[0],
        ],
        dtype=np.float64,
    )


def _actuator_flex_to_abs(values):
    flex = _normalized_to_actuator_flex(values)
    open_amount = 1.0 - flex
    open_amount = np.clip(open_amount, 0.0, 1.0)
    abs_cmd = RH56F1_FOLDED_ABS + open_amount * (RH56F1_OPEN_ABS - RH56F1_FOLDED_ABS)
    return np.clip(abs_cmd, RH56F1_SAFE_ABS_MIN, RH56F1_SAFE_ABS_MAX)


def _apply_thumb_coupling(values, limits):
    """Restore the RH56F1 thumb linkage after independent SIM retargeting."""
    values = np.asarray(values, dtype=np.float64).copy()
    limits = np.asarray(limits, dtype=np.float64)

    # URDF: thumb_3 mimics thumb_2, then thumb_4 mimics thumb_3.
    values[2] = values[1] * RH56F1_THUMB_3_RATIO
    values[3] = values[2] * RH56F1_THUMB_4_RATIO
    values[2:4] = np.clip(values[2:4], limits[2:4, 0], limits[2:4, 1])
    return values


def _retarget_to_normalized(values, limits):
    return _normalize(_apply_thumb_coupling(values, limits), limits)


def _prepare_vector_reference(hand_data, indices):
    data = np.asarray(hand_data, dtype=np.float64).reshape(25, 3)
    indices = np.asarray(indices, dtype=np.int64)
    vectors = data[indices[1]] - data[indices[0]]
    return vectors @ _HAND_FRAME_AXES.T


def _hand_tracking_ready(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return bool(
        np.isfinite(left).all()
        and np.isfinite(right).all()
        and np.count_nonzero(np.linalg.norm(left, axis=1) > 1e-5) >= 8
        and np.count_nonzero(np.linalg.norm(right, axis=1) > 1e-5) >= 8
    )


def _env_flag(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_csv_set(name, default):
    raw = os.getenv(name)
    if raw is None:
        return set(default)
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def _finger_extension_scores(hand_data):
    data = np.asarray(hand_data, dtype=np.float64).reshape(25, 3)
    scores = {}
    for finger, landmarks in RH56F1_FINGER_LANDMARKS.items():
        points = data[list(landmarks)]
        segments = np.diff(points, axis=0)
        path_length = float(np.sum(np.linalg.norm(segments, axis=1)))
        chord_length = float(np.linalg.norm(points[-1] - points[0]))
        straightness = chord_length / max(path_length, 1e-6)

        base_to_tip = float(np.linalg.norm(points[-1] - points[0]))
        wrist_to_tip = float(np.linalg.norm(points[-1] - data[0]))
        reach = min(base_to_tip, wrist_to_tip)
        # Straight fingers usually sit above ~0.94; curled/contact-stuck
        # fingers drop quickly. Reach prevents tiny noisy straight segments
        # from being treated as an intentional open command.
        straight_score = float(np.clip((straightness - 0.82) / 0.15, 0.0, 1.0))
        reach_score = float(np.clip((reach - 0.045) / 0.055, 0.0, 1.0))
        scores[finger] = float(np.clip(straight_score * reach_score, 0.0, 1.0))
    return scores


def _apply_open_recovery(target, hand_data, side):
    target = np.clip(np.asarray(target, dtype=np.float64), 0.0, 1.0).copy()
    if not _env_flag("RH56F1_ENABLE_OPEN_RECOVERY", True):
        return target, {}

    enabled_hands = _env_csv_set("RH56F1_OPEN_RECOVERY_HANDS", ("right",))
    if side.lower() not in enabled_hands and "both" not in enabled_hands:
        return target, {}

    enabled_fingers = _env_csv_set(
        "RH56F1_OPEN_RECOVERY_FINGERS",
        ("index", "middle", "little"),
    )
    threshold = float(np.clip(float(os.getenv("RH56F1_OPEN_RECOVERY_THRESHOLD", "0.62")), 0.0, 0.98))
    strength = float(np.clip(float(os.getenv("RH56F1_OPEN_RECOVERY_STRENGTH", "1.0")), 0.0, 1.0))
    open_value = float(np.clip(float(os.getenv("RH56F1_OPEN_RECOVERY_VALUE", "0.0")), 0.0, 1.0))
    scores = _finger_extension_scores(hand_data)
    before = target.copy()

    activations = {}
    for finger in enabled_fingers:
        if finger not in RH56F1_COMMAND_FINGER_INDICES:
            continue
        score = scores.get(finger, 0.0)
        activation = float(np.clip((score - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0))
        activation *= strength
        activations[finger] = activation
        if activation <= 0.0:
            continue
        for idx in RH56F1_COMMAND_FINGER_INDICES[finger]:
            target[idx] = min(target[idx], target[idx] * (1.0 - activation) + open_value * activation)

    if not any(value > 0.0 for value in activations.values()):
        return target, {}

    return target, {
        "side": side,
        "scores": scores,
        "activations": activations,
        "delta": target - before,
        "threshold": threshold,
        "strength": strength,
        "open_value": open_value,
    }


def _fmt_named(names, values):
    return [
        f"{name}={float(value):.4f}"
        for name, value in zip(names, np.asarray(values, dtype=np.float64))
    ]


def _fmt_motor_fields(cmds, start=0, count=6):
    fields = []
    for idx in range(start, min(start + count, len(cmds))):
        cmd = cmds[idx]
        try:
            reserve = list(getattr(cmd, "reserve", []))[:3]
        except TypeError:
            reserve = getattr(cmd, "reserve", None)
        fields.append(
            {
                "idx": idx,
                "q": round(float(getattr(cmd, "q", 0.0)), 4),
                "mode": int(getattr(cmd, "mode", 0)),
                "reserve": reserve,
                "kp": round(float(getattr(cmd, "kp", 0.0)), 4),
                "kd": round(float(getattr(cmd, "kd", 0.0)), 4),
            }
        )
    return fields


def _rh56f1_touch_command(hand_id):
    data = [0xEB, 0x90, int(hand_id) & 0xFF, 0x04, 0x11, 0xB8, 0x0B, 0x44]
    data.append(sum(data[2:]) & 0xFF)
    return bytes(data)


def _parse_rh56f1_touch_response(recv):
    if not recv:
        return None
    start_idx = recv.find(b"\xB8\x0B")
    if start_idx < 0:
        return None
    data_start = start_idx + 2
    if data_start + 68 > len(recv):
        return None

    fingers = ["little", "ring", "middle", "index", "thumb"]
    finger_results = {}
    for idx, finger in enumerate(fingers):
        base_idx = data_start + idx * 10
        data = recv[base_idx : base_idx + 10]
        finger_results[finger] = [
            data[0] | (data[1] << 8),
            data[2] | (data[3] << 8),
            data[4] | (data[5] << 8),
            data[6] | (data[7] << 8) | (data[8] << 16),
        ]

    palm_start = data_start + len(fingers) * 10
    palm = []
    for idx in range(9):
        base_idx = palm_start + idx * 2
        palm.append(recv[base_idx] | (recv[base_idx + 1] << 8))

    return {
        "fingers": finger_results,
        "palm": palm,
    }


class RH56F1TactileReader:
    """Background serial reader for RH56F1 tactile frames."""

    def __init__(
        self,
        left_port=None,
        right_port=None,
        baudrate=115200,
        hand_id=1,
        fps=30.0,
        debug_rate=0.0,
    ):
        self.left_port = left_port
        self.right_port = right_port
        self.baudrate = int(baudrate)
        self.hand_id = int(hand_id)
        self.fps = max(float(fps), 1.0)
        self.debug_rate = max(float(debug_rate), 0.0)
        self.running = True
        self.lock = threading.Lock()
        self.latest = {}
        self.errors = {}
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _open_serials(self):
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("pyserial is required for RH56F1 tactile reading") from exc

        serials = {}
        for side, port in (("left_ee", self.left_port), ("right_ee", self.right_port)):
            if not port:
                continue
            ser = serial.Serial(port, self.baudrate, timeout=0.03)
            ser.reset_input_buffer()
            serials[side] = ser
            logger_mp.info(
                "[RH56F1 tactile] opened side=%s port=%s baudrate=%s hand_id=%s",
                side,
                port,
                self.baudrate,
                self.hand_id,
            )
        return serials

    def _read_one(self, ser):
        ser.write(_rh56f1_touch_command(self.hand_id))
        time.sleep(0.025)
        return _parse_rh56f1_touch_response(ser.read_all())

    def _loop(self):
        period = 1.0 / self.fps
        last_debug = 0.0
        try:
            serials = self._open_serials()
        except Exception as exc:
            logger_mp.exception("[RH56F1 tactile] failed to initialize: %s", exc)
            with self.lock:
                self.errors["init"] = repr(exc)
            return

        try:
            while self.running:
                start = time.perf_counter()
                for side, ser in serials.items():
                    try:
                        data = self._read_one(ser)
                        if data is not None:
                            with self.lock:
                                self.latest[side] = {
                                    "timestamp": time.time(),
                                    **data,
                                }
                        else:
                            with self.lock:
                                self.errors[side] = "empty_or_malformed_frame"
                    except Exception as exc:
                        with self.lock:
                            self.errors[side] = repr(exc)
                        if self.debug_rate > 0.0 and time.time() - last_debug >= 1.0 / self.debug_rate:
                            logger_mp.warning("[RH56F1 tactile] read failed side=%s error=%s", side, exc)
                if self.debug_rate > 0.0 and time.time() - last_debug >= 1.0 / self.debug_rate:
                    last_debug = time.time()
                    with self.lock:
                        snapshot = dict(self.latest)
                    logger_mp.info("[RH56F1 tactile] latest=%s", snapshot)
                time.sleep(max(0.0, period - (time.perf_counter() - start)))
        finally:
            for ser in serials.values():
                try:
                    ser.close()
                except Exception:
                    pass

    def read_latest(self):
        with self.lock:
            return {
                side: {
                    "timestamp": data.get("timestamp"),
                    "fingers": {name: list(values) for name, values in data.get("fingers", {}).items()},
                    "palm": list(data.get("palm", [])),
                }
                for side, data in self.latest.items()
            }

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


def _set_cmd_active(cmd, active=True):
    cmd.mode = 0b0001 if active else 0
    reserve_value = [1, 0, 0] if active else [0, 0, 0]
    try:
        cmd.reserve = reserve_value
    except (AttributeError, TypeError):
        try:
            cmd.reserve = tuple(reserve_value)
        except (AttributeError, TypeError):
            try:
                cmd.reserve[0] = reserve_value[0]
                cmd.reserve[1] = reserve_value[1]
                cmd.reserve[2] = reserve_value[2]
            except (AttributeError, TypeError, IndexError):
                pass


def _debug_test_pose_target(num_motors):
    pose = os.environ.get("RH56F1_TEST_POSE", "").strip().lower()
    if not pose:
        return None, "retarget"
    if pose in ("open", "zero", "0"):
        return np.zeros(num_motors, dtype=np.float64), f"test_pose:{pose}"
    if pose in ("half", "mid", "0.5"):
        return np.full(num_motors, 0.5, dtype=np.float64), f"test_pose:{pose}"
    if pose in ("close", "closed", "one", "1"):
        return np.ones(num_motors, dtype=np.float64), f"test_pose:{pose}"
    logger_mp.warning(
        "[RH56F1 debug] unsupported RH56F1_TEST_POSE=%r; use open, half, or close",
        pose,
    )
    return None, f"test_pose_invalid:{pose}"


class RH56F1Retargeting:
    def __init__(self, retarget_mode=None):
        RetargetingConfig.set_default_urdf_dir(str(_ASSETS_ROOT))
        self.retarget_mode = _normalize_rh56f1_retarget_mode(retarget_mode)
        cfg = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
        if self.retarget_mode is not None:
            for side in ("left", "right"):
                cfg[side]["type"] = self.retarget_mode

        self.left_retargeting = RetargetingConfig.from_dict(cfg["left"]).build()
        self.right_retargeting = RetargetingConfig.from_dict(cfg["right"]).build()
        # The retargeting config optimizes the 12 URDF joints directly, then
        # the deploy command path collapses them into the 6 RH56F1 actuators.
        self.left_joint_names = [f"left_{suffix}" for suffix in RH56F1_JOINT_SUFFIXES]
        self.right_joint_names = [f"right_{suffix}" for suffix in RH56F1_JOINT_SUFFIXES]
        self.left_indices = self.left_retargeting.optimizer.target_link_human_indices
        self.right_indices = self.right_retargeting.optimizer.target_link_human_indices
        self.left_retargeting_type = self.left_retargeting.optimizer.retargeting_type.lower()
        self.right_retargeting_type = self.right_retargeting.optimizer.retargeting_type.lower()
        self.left_to_command = [
            self.left_retargeting.joint_names.index(name) for name in self.left_joint_names
        ]
        self.right_to_command = [
            self.right_retargeting.joint_names.index(name) for name in self.right_joint_names
        ]
        self.left_joint_limits = self._joint_limits(cfg["left"]["urdf_path"], self.left_joint_names)
        self.right_joint_limits = self._joint_limits(cfg["right"]["urdf_path"], self.right_joint_names)
        logger_mp.info(
            "[RH56F1 retargeting] left=%s right=%s override=%s left_indices_shape=%s right_indices_shape=%s command_joints=%s",
            self.left_retargeting_type,
            self.right_retargeting_type,
            self.retarget_mode or "config",
            np.asarray(self.left_indices).shape,
            np.asarray(self.right_indices).shape,
            len(self.left_joint_names),
        )

    @staticmethod
    def _joint_limits(urdf_path, joint_names):
        root = ET.parse(_ASSETS_ROOT / urdf_path).getroot()
        limits = {}
        for joint in root.findall("joint"):
            limit = joint.find("limit")
            if limit is not None:
                limits[joint.get("name")] = (
                    float(limit.get("lower", "0")),
                    float(limit.get("upper", "0")),
                )
        missing = [name for name in joint_names if name not in limits]
        if missing:
            raise ValueError(f"RH56F1 joint limits missing: {missing}")
        return [limits[name] for name in joint_names]

    def _retarget_raw(self, left_hand_data, right_hand_data):
        left_ref = _prepare_vector_reference(left_hand_data, self.left_indices)
        right_ref = _prepare_vector_reference(right_hand_data, self.right_indices)
        left_q = self.left_retargeting.retarget(left_ref)[self.left_to_command]
        right_q = self.right_retargeting.retarget(right_ref)[self.right_to_command]
        return left_q, right_q

    def retarget_sim(self, left_hand_data, right_hand_data):
        left_q, right_q = self._retarget_raw(left_hand_data, right_hand_data)
        return (
            _retarget_to_normalized(left_q, self.left_joint_limits),
            _retarget_to_normalized(right_q, self.right_joint_limits),
        )

    def retarget_abs(self, left_hand_data, right_hand_data):
        left_q, right_q = self._retarget_raw(left_hand_data, right_hand_data)
        left_q = _apply_thumb_coupling(left_q, self.left_joint_limits)
        right_q = _apply_thumb_coupling(right_q, self.right_joint_limits)
        return (
            _actuator_flex_to_abs(_retarget_to_actuator_flex(left_q, self.left_joint_limits)),
            _actuator_flex_to_abs(_retarget_to_actuator_flex(right_q, self.right_joint_limits)),
        )

    def retarget_deploy(self, left_hand_data, right_hand_data):
        return self.retarget_sim(left_hand_data, right_hand_data)

    def retarget(self, left_hand_data, right_hand_data):
        return self.retarget_abs(left_hand_data, right_hand_data)


class RH56F1_Controller:
    """RH56F1 controller using right-then-left actuator commands."""

    def __init__(
        self,
        left_hand_array,
        right_hand_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
        fps=50.0,
        simulation_mode=False,
        network_interface=None,
        retarget_mode=None,
    ):
        self.simulation_mode = bool(simulation_mode)
        self.dds_domain_id = 1 if self.simulation_mode else 0
        self.network_interface = network_interface
        self.fps = max(float(fps), 1.0)
        self.running = True
        self.command_motors = RH56F1_Num_Retarget_Joints
        self.command_mode = "sim_normalized" if self.simulation_mode else RH56F1_DEPLOY_COMMAND_MODE
        self.hand_retargeting = RH56F1Retargeting(retarget_mode=retarget_mode)
        self.retarget_mode = self.hand_retargeting.retarget_mode
        self.left_state = Array("d", self.command_motors, lock=True)
        self.right_state = Array("d", self.command_motors, lock=True)

        ChannelFactoryInitialize(self.dds_domain_id, networkInterface=self.network_interface)
        self.publisher = ChannelPublisher(RH56F1_CMD_TOPIC, MotorCmds_)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber(RH56F1_STATE_TOPIC, MotorStates_)
        self.subscriber.Init()

        self.state_thread = threading.Thread(target=self._state_loop, daemon=True)
        self.state_thread.start()
        self.control_thread = threading.Thread(
            target=self._control_loop,
            args=(
                left_hand_array,
                right_hand_array,
                dual_hand_data_lock,
                dual_hand_state_array,
                dual_hand_action_array,
            ),
            daemon=True,
        )
        self.control_thread.start()
        logger_mp.info(
            "[RH56F1 DDS] initialized mode=%s domain=%s network_interface=%s cmd_topic=%s state_topic=%s joints_per_hand=%s retarget=%s/%s override=%s",
            "sim" if self.simulation_mode else "deploy",
            self.dds_domain_id,
            self.network_interface,
            RH56F1_CMD_TOPIC,
            RH56F1_STATE_TOPIC,
            self.command_motors,
            self.hand_retargeting.left_retargeting_type,
            self.hand_retargeting.right_retargeting_type,
            self.retarget_mode or "config",
        )

    def _state_loop(self):
        while self.running:
            msg = self.subscriber.Read()
            states = getattr(msg, "states", []) if msg is not None else []
            num_joints = self.command_motors * 2
            if len(states) >= num_joints:
                with self.right_state.get_lock():
                    self.right_state[:] = [states[i].q for i in range(self.command_motors)]
                with self.left_state.get_lock():
                    self.left_state[:] = [
                        states[self.command_motors + i].q for i in range(self.command_motors)
                    ]
            time.sleep(0.002)

    def _command_message(self, right_target, left_target):
        right_wire = self._wire_command(right_target)
        left_wire = self._wire_command(left_target)
        msg = MotorCmds_()
        msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(self.command_motors * 2)]
        for idx, value in enumerate(np.concatenate((right_wire, left_wire))):
            _set_cmd_active(msg.cmds[idx], True)
            msg.cmds[idx].q = float(value)
            msg.cmds[idx].dq = 0.0
            msg.cmds[idx].tau = 0.0
            msg.cmds[idx].kp = 1.0
            msg.cmds[idx].kd = 0.05
        return msg

    def _write_command(self, right_target, left_target):
        msg = self._command_message(right_target, left_target)
        return self.publisher.Write(msg), msg

    def _wire_command(self, target):
        target = np.asarray(target, dtype=np.float64)
        if self.simulation_mode or self.command_mode == "normalized":
            return target
        logger_mp.warning(
            "[RH56F1 command] unknown mode=%r; falling back to normalized q",
            self.command_mode,
        )
        return target

    def _control_loop(
        self,
        left_hand_array,
        right_hand_array,
        dual_hand_data_lock,
        dual_hand_state_array,
        dual_hand_action_array,
    ):
        if self.simulation_mode:
            left_target = np.zeros(self.command_motors, dtype=np.float64)
            right_target = np.zeros(self.command_motors, dtype=np.float64)
        else:
            left_target = np.zeros(self.command_motors, dtype=np.float64)
            right_target = np.zeros(self.command_motors, dtype=np.float64)
        period = 1.0 / self.fps
        loop_count = 0

        while self.running:
            try:
                start = time.perf_counter()
                loop_count += 1
                command_source = "hold"
                left_recovery_debug = {}
                right_recovery_debug = {}
                with left_hand_array.get_lock():
                    left_hand = np.asarray(left_hand_array[:], dtype=np.float64).reshape(25, 3)
                with right_hand_array.get_lock():
                    right_hand = np.asarray(right_hand_array[:], dtype=np.float64).reshape(25, 3)

                test_target, test_source = _debug_test_pose_target(self.command_motors)
                if test_target is not None and not self.simulation_mode:
                    left_target = test_target.copy()
                    right_target = test_target.copy()
                    command_source = test_source
                elif _hand_tracking_ready(left_hand, right_hand):
                    if self.simulation_mode:
                        left_target, right_target = self.hand_retargeting.retarget_sim(left_hand, right_hand)
                        command_source = "retarget_sim"
                    else:
                        left_target, right_target = self.hand_retargeting.retarget_deploy(left_hand, right_hand)
                        command_source = "retarget_deploy"
                    left_target, left_recovery_debug = _apply_open_recovery(left_target, left_hand, "left")
                    right_target, right_recovery_debug = _apply_open_recovery(right_target, right_hand, "right")
                    if left_recovery_debug or right_recovery_debug:
                        command_source += "+open_recovery"
                else:
                    command_source = test_source if test_source != "retarget" else "hold_no_hand_tracking"

                state = np.concatenate((np.asarray(self.left_state[:]), np.asarray(self.right_state[:])))
                action = np.concatenate((left_target, right_target))
                if dual_hand_state_array is not None and dual_hand_action_array is not None:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state
                        dual_hand_action_array[:] = action

                write_ok, cmd_msg = self._write_command(right_target, left_target)
                if loop_count % max(1, int(self.fps)) == 0:
                    left_wire = self._wire_command(left_target)
                    right_wire = self._wire_command(right_target)
                    logger_mp.debug(
                        "[RH56F1 publish] mode=%s source=%s write_ok=%s joints_per_hand=%s "
                        "topic=%s domain=%s "
                        "left_min=%.4f left_max=%.4f right_min=%.4f right_max=%.4f "
                        "left_cmd=%s right_cmd=%s left_cmd_abs=%s right_cmd_abs=%s "
                        "left_wire=%s right_wire=%s "
                        "open_recovery_left=%s open_recovery_right=%s "
                        "right_fields=%s left_fields=%s "
                        "left_state=%s right_state=%s",
                        self.command_mode,
                        command_source,
                        write_ok,
                        self.command_motors,
                        RH56F1_CMD_TOPIC,
                        self.dds_domain_id,
                        float(np.min(left_target)),
                        float(np.max(left_target)),
                        float(np.min(right_target)),
                        float(np.max(right_target)),
                        _fmt_named(RH56F1_COMMAND_NAMES, left_target)
                        if not self.simulation_mode
                        else np.round(left_target, 4).tolist(),
                        _fmt_named(RH56F1_COMMAND_NAMES, right_target)
                        if not self.simulation_mode
                        else np.round(right_target, 4).tolist(),
                        _fmt_named(RH56F1_ACTUATOR_NAMES, _actuator_flex_to_abs(left_target))
                        if not self.simulation_mode
                        else "n/a",
                        _fmt_named(RH56F1_ACTUATOR_NAMES, _actuator_flex_to_abs(right_target))
                        if not self.simulation_mode
                        else "n/a",
                        _fmt_named(RH56F1_COMMAND_NAMES, left_wire)
                        if not self.simulation_mode
                        else np.round(left_wire, 4).tolist(),
                        _fmt_named(RH56F1_COMMAND_NAMES, right_wire)
                        if not self.simulation_mode
                        else np.round(right_wire, 4).tolist(),
                        left_recovery_debug if "left_recovery_debug" in locals() else {},
                        right_recovery_debug if "right_recovery_debug" in locals() else {},
                        _fmt_motor_fields(cmd_msg.cmds, 0, self.command_motors),
                        _fmt_motor_fields(cmd_msg.cmds, self.command_motors, self.command_motors),
                        _fmt_named(RH56F1_COMMAND_NAMES, state[: self.command_motors])
                        if not self.simulation_mode
                        else np.round(state[: self.command_motors], 4).tolist(),
                        _fmt_named(RH56F1_COMMAND_NAMES, state[-self.command_motors :])
                        if not self.simulation_mode
                        else np.round(state[-self.command_motors :], 4).tolist(),
                    )
                time.sleep(max(0.0, period - (time.perf_counter() - start)))
            except Exception:
                logger_mp.exception("[RH56F1 control] control loop crashed")
                self.running = False

    def stop(self):
        self.running = False


__all__ = [
    "RH56F1_ACTUATOR_NAMES",
    "RH56F1_CMD_TOPIC",
    "RH56F1_FOLDED_ABS",
    "RH56F1_OPEN_ABS",
    "RH56F1_SAFE_ABS_MAX",
    "RH56F1_SAFE_ABS_MIN",
    "RH56F1_STATE_TOPIC",
    "RH56F1_Controller",
    "RH56F1_Num_Motors",
    "RH56F1_Num_Retarget_Joints",
    "RH56F1TactileReader",
    "RH56F1Retargeting",
]
