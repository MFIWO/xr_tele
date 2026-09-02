import os
import json
import numpy as np
import socket
import sys
import threading
import time
import xml.etree.ElementTree as ET
from multiprocessing import Process, Array
from pathlib import Path

import yaml

_LOCAL_DEX_RETARGETING_SRC = Path(__file__).resolve().parent / "dex-retargeting" / "src"
if _LOCAL_DEX_RETARGETING_SRC.exists() and str(_LOCAL_DEX_RETARGETING_SRC) not in sys.path:
    sys.path.insert(0, str(_LOCAL_DEX_RETARGETING_SRC))

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
from dex_retargeting import RetargetingConfig
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

import logging_mp

logger_mp = logging_mp.getLogger(__name__)


InspireDG2_Num_Motors = 13
InspireDG2_Tactile_Num_Values = 29
kTopicInspireDG2Command = "rt/rh5dg2/cmd"
kTopicInspireDG2State = "rt/rh5dg2/state"

_XR_TELEOPERATE_ROOT = Path(__file__).resolve().parents[2]
_ASSETS_ROOT = _XR_TELEOPERATE_ROOT / "assets"
_DG2_ASSET_DIR = _ASSETS_ROOT / "RH5DG2"
_DG2_CONFIG_PATH = _DG2_ASSET_DIR / "RH5DG2.yml"
_DG2_URDF_CACHE_DIR = Path("/tmp") / "xr_teleoperate_rh5dg2_urdf"

_DG2_POSITION_AXES = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)

_REG = {
    "ID": 1000,
    "clearErr": 1003,
    "forceClb": 1007,
    "angleSet": 1080,
    "forceSet": 1094,
    "speedSet": 1108,
    "angleAct": 1136,
    "forceAct": 1150,
    "errCode": 1178,
    "statusCode": 1192,
    "temp": 1206,
    "mode": 1220,
}

_WRITE_13_BYTES = InspireDG2_Num_Motors * 2
_READ_13_BYTES = 28
_TOUCH_ADDRESS = 3000
_TOUCH_BYTES = 68

_DG2_ANGLE_OPEN = np.array(
    [1650, 1650, 1800, 150, 1650, 150, 1900, 1900, 1900, 1900, 1750, 1600, 2040],
    dtype=np.float64,
)
_DG2_ANGLE_CLOSED = np.array(
    [900, 900, 900, -150, 900, -150, 1050, 1050, 1050, 1050, 1140, 1220, 1500],
    dtype=np.float64,
)
_DG2_ANGLE_MIN = np.array(
    [900, 900, 900, -150, 900, -150, 1050, 1050, 1050, 1050, 1140, 1220, 1500],
    dtype=np.float64,
)
_DG2_ANGLE_MAX = np.array(
    [1650, 1650, 1800, 150, 1650, 150, 1900, 1900, 1900, 1900, 1750, 1600, 2040],
    dtype=np.float64,
)
_DG2_SAFE_COMMAND_MIN = np.array(
    [1030, 1030, 1030, -150, 1030, -150, 1150, 1150, 1150, 1150, 1140, 1220, 1520],
    dtype=np.float64,
)
_DG2_SAFE_COMMAND_MAX = np.array(
    [1650, 1650, 1800, 150, 1650, 150, 1900, 1900, 1900, 1870, 1950, 1800, 2040],
    dtype=np.float64,
)
_DG2_SPREAD_INDICES = (3, 5)
_DG2_SIDE_SWING_RAW_INDICES = (3, 5)
_DG2_SPREAD_NEUTRAL = {
    "left": np.array([0.0, 0.0], dtype=np.float64),
    "right": np.array([0.0, 0.0], dtype=np.float64),
}
_DG2_RAW_FROM_RETARGET = np.array(
    [
        11,  # little root <- pinky_mcp
        9,   # ring root
        7,   # middle root
        6,   # middle side swing
        4,   # index root
        3,   # index side swing
        12,  # little middle <- pinky_pip
        10,  # ring middle
        8,   # middle middle
        5,   # index middle
        0,   # thumb side swing
        1,   # thumb first bend
        2,   # thumb second bend
    ],
    dtype=np.int64,
)
_DG2_PINCH_RAW_INDICES = (4, 9, 10, 11, 12)
_DG2_THUMB_BEND_RAW_INDICES = (11, 12)
_DG2_PINCH_CLOSE_TARGET = np.array(
    [1030, 1030, 1030, -150, 900, -150, 1150, 1150, 1150, 1050, 1140, 1220, 1520],
    dtype=np.float64,
)
_DG2_MIDDLE_RAW_INDICES = (2, 8)
_DG2_MIDDLE_CLOSE_TARGET = np.array(
    [1030, 1030, 1030, -150, 1030, -150, 1150, 1150, 1150, 1150, 1140, 1220, 1520],
    dtype=np.float64,
)
_DG2_FINGER_LANDMARKS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8, 9),
    "middle": (10, 11, 12, 13, 14),
    "ring": (15, 16, 17, 18, 19),
    "little": (20, 21, 22, 23, 24),
}


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name, default):
    value = os.getenv(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        logger_mp.warning("Ignoring invalid float env %s=%r; using %.4f", name, value, float(default))
        return float(default)


def _checksum(data):
    return sum(data[2:]) & 0xFF


def _build_read_frame(hand_id, address, num_bytes):
    frame = [
        0xEB,
        0x90,
        int(hand_id) & 0xFF,
        0x04,
        0x11,
        int(address) & 0xFF,
        (int(address) >> 8) & 0xFF,
        int(num_bytes) & 0xFF,
    ]
    frame.append(_checksum(frame))
    return bytes(frame)


def _build_write_frame(hand_id, address, payload):
    frame = [
        0xEB,
        0x90,
        int(hand_id) & 0xFF,
        len(payload) + 3,
        0x12,
        int(address) & 0xFF,
        (int(address) >> 8) & 0xFF,
    ]
    frame.extend(int(v) & 0xFF for v in payload)
    frame.append(_checksum(frame))
    return bytes(frame)


def _pack_int16_values(values):
    payload = []
    for value in values:
        raw = int(round(float(value))) & 0xFFFF
        payload.append(raw & 0xFF)
        payload.append((raw >> 8) & 0xFF)
    return payload


def _int16_from_le(lo, hi):
    value = (int(lo) & 0xFF) | ((int(hi) & 0xFF) << 8)
    if value >= 0x8000:
        value -= 0x10000
    return value


def _find_response_payload(recv, hand_id, address, command, num_bytes):
    if not recv:
        return None
    address_l = int(address) & 0xFF
    address_h = (int(address) >> 8) & 0xFF
    expected_min = 7 + int(num_bytes)
    for idx in range(max(0, len(recv) - expected_min + 1)):
        if (
            recv[idx] == 0x90
            and recv[idx + 1] == 0xEB
            and recv[idx + 2] == (int(hand_id) & 0xFF)
            and recv[idx + 4] == command
            and recv[idx + 5] == address_l
            and recv[idx + 6] == address_h
        ):
            data_start = idx + 7
            data_end = data_start + int(num_bytes)
            if data_end <= len(recv):
                return bytes(recv[data_start:data_end])
    return None


def _read_register(ser, hand_id, address, num_bytes, timeout_s=0.06):
    if hasattr(ser, "reset_input_buffer"):
        ser.reset_input_buffer()
    ser.write(_build_read_frame(hand_id, address, num_bytes))
    deadline = time.time() + timeout_s
    recv = bytearray()
    while time.time() < deadline:
        chunk = ser.read_all()
        if chunk:
            recv.extend(chunk)
            payload = _find_response_payload(recv, hand_id, address, 0x11, num_bytes)
            if payload is not None:
                return payload
        time.sleep(0.002)
    chunk = ser.read_all()
    if chunk:
        recv.extend(chunk)
    return _find_response_payload(recv, hand_id, address, 0x11, num_bytes)


def _write_register(ser, hand_id, address, values):
    ser.write(_build_write_frame(hand_id, address, _pack_int16_values(values)))


def _read_13_int16(ser, hand_id, register_name):
    payload = _read_register(ser, hand_id, _REG[register_name], _READ_13_BYTES)
    if payload is None or len(payload) < _WRITE_13_BYTES:
        return None
    return [_int16_from_le(payload[i], payload[i + 1]) for i in range(0, _WRITE_13_BYTES, 2)]


def _parse_touch_payload(payload):
    if payload is None or len(payload) < _TOUCH_BYTES:
        return None

    fingers = {}
    for idx, finger in enumerate(("little", "ring", "middle", "index", "thumb")):
        base = idx * 10
        proximity = payload[base + 6] | (payload[base + 7] << 8) | (payload[base + 8] << 16)
        fingers[finger] = [
            payload[base] | (payload[base + 1] << 8),
            payload[base + 2] | (payload[base + 3] << 8),
            payload[base + 4] | (payload[base + 5] << 8),
            proximity,
        ]

    palm_start = 5 * 10
    palm = []
    for idx in range(9):
        base = palm_start + idx * 2
        palm.append(payload[base] | (payload[base + 1] << 8))

    return {"fingers": fingers, "palm": palm}


def _read_touch_data(ser, hand_id):
    payload = _read_register(ser, hand_id, _TOUCH_ADDRESS, _TOUCH_BYTES, timeout_s=0.08)
    return _parse_touch_payload(payload)


def _normalize_retarget_mode(mode):
    if mode is None:
        return None
    mode = str(mode).strip().lower()
    if mode in ("", "config", "default"):
        return None
    if mode == "vector":
        return "Vector"
    if mode == "dexpilot":
        return "DexPilot"
    raise ValueError(f"Unsupported RH5DG2 retarget mode: {mode}")


def _clip_to_joint_limits(values, joint_limits):
    values = np.asarray(values, dtype=np.float64)
    clipped = np.empty(len(values), dtype=np.float64)
    for idx, (value, (lower, upper)) in enumerate(zip(values, joint_limits)):
        clipped[idx] = np.clip(value, lower, upper)
    return clipped


def _normalize_to_open_ratio(values, joint_limits):
    values = np.asarray(values, dtype=np.float64)
    normalized = np.empty(len(values), dtype=np.float64)
    for idx, (value, (lower, upper)) in enumerate(zip(values, joint_limits)):
        if np.isclose(upper, lower):
            normalized[idx] = 0.5
        else:
            normalized[idx] = np.clip((upper - value) / (upper - lower), 0.0, 1.0)
    return normalized


def _open_joint_value(joint_name, joint_limit):
    lower, upper = joint_limit
    if "_yaw_joint" in joint_name and ("index_" in joint_name or "middle_" in joint_name):
        return float(np.clip(0.0, lower, upper))
    return float(lower)


def _prepare_vector_reference(hand_data, indices):
    data = np.asarray(hand_data, dtype=np.float64).reshape(25, 3)
    idx = np.asarray(indices, dtype=np.int64)
    source = data[idx[0, :]]
    target = data[idx[1, :]]
    return (target - source) @ _DG2_POSITION_AXES.T


def _prepare_position_reference(hand_data, indices):
    data = np.asarray(hand_data, dtype=np.float64).reshape(25, 3)
    idx = np.asarray(indices, dtype=np.int64)
    wrist = data[0].copy()
    selected = data[idx]
    return (selected - wrist) @ _DG2_POSITION_AXES.T


def _valid_hand_landmarks(hand_data):
    data = np.asarray(hand_data, dtype=np.float64).reshape(25, 3)
    if not np.isfinite(data).all():
        return False
    if np.allclose(data, 0.0, atol=1e-5):
        return False
    if np.allclose(data[4], np.array([-1.13, 0.3, 0.15]), atol=1e-3):
        return False
    return True


def _pinch_activation(hand_data):
    if not _valid_hand_landmarks(hand_data):
        return 0.0, float("inf")
    data = np.asarray(hand_data, dtype=np.float64).reshape(25, 3)
    distance = float(np.linalg.norm(data[4] - data[9]))
    close_dist = _env_float("INSPIRE_DG2_PINCH_CLOSE_DIST", 0.028)
    open_dist = _env_float("INSPIRE_DG2_PINCH_OPEN_DIST", 0.075)
    if open_dist <= close_dist:
        open_dist = close_dist + 1e-3
    activation = np.clip((open_dist - distance) / (open_dist - close_dist), 0.0, 1.0)
    return float(activation), distance


def _finger_curl_score(hand_data, finger):
    if not _valid_hand_landmarks(hand_data):
        return 0.0
    data = np.asarray(hand_data, dtype=np.float64).reshape(25, 3)
    points = data[list(_DG2_FINGER_LANDMARKS[finger])]
    segments = np.diff(points, axis=0)
    path_length = float(np.sum(np.linalg.norm(segments, axis=1)))
    chord_length = float(np.linalg.norm(points[-1] - points[0]))
    straightness = chord_length / max(path_length, 1e-6)
    return float(1.0 - np.clip((straightness - 0.72) / 0.25, 0.0, 1.0))


class _InspireDG2Retargeting:
    def __init__(self, fast_mode=True, retarget_mode=None):
        RetargetingConfig.set_default_urdf_dir(str(_ASSETS_ROOT))
        self.fast_mode = bool(fast_mode)
        self.retarget_mode = _normalize_retarget_mode(retarget_mode)
        cfg = self._load_config()
        if self.retarget_mode is not None:
            for side in ("left", "right"):
                cfg[side]["type"] = self.retarget_mode
        if self.fast_mode:
            self._apply_fast_config(cfg)

        self.left_retargeting = RetargetingConfig.from_dict(cfg["left"]).build()
        self.right_retargeting = RetargetingConfig.from_dict(cfg["right"]).build()
        if self.fast_mode:
            self._configure_fast_optimizer(self.left_retargeting)
            self._configure_fast_optimizer(self.right_retargeting)

        self.left_joint_names = list(cfg["left"]["target_joint_names"])
        self.right_joint_names = list(cfg["right"]["target_joint_names"])
        self.left_indices = self.left_retargeting.optimizer.target_link_human_indices
        self.right_indices = self.right_retargeting.optimizer.target_link_human_indices
        self.left_retargeting_to_hardware = [
            self.left_retargeting.joint_names.index(name) for name in self.left_joint_names
        ]
        self.right_retargeting_to_hardware = [
            self.right_retargeting.joint_names.index(name) for name in self.right_joint_names
        ]
        self.left_joint_limits = self._load_joint_limits(Path(cfg["left"]["urdf_path"]), self.left_joint_names)
        self.right_joint_limits = self._load_joint_limits(Path(cfg["right"]["urdf_path"]), self.right_joint_names)
        self._set_open_initial_qpos(self.left_retargeting, self.left_joint_names, self.left_joint_limits)
        self._set_open_initial_qpos(self.right_retargeting, self.right_joint_names, self.right_joint_limits)
        logger_mp.info(
            "[InspireDG2 retargeting] mode_left=%s mode_right=%s override=%s fast_mode=%s",
            self.left_retargeting.optimizer.retargeting_type,
            self.right_retargeting.optimizer.retargeting_type,
            self.retarget_mode or "config",
            self.fast_mode,
        )

    def _load_config(self):
        with _DG2_CONFIG_PATH.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        fixed_cfg = {}
        for side in ("left", "right"):
            side_cfg = dict(cfg[side])
            side_cfg["urdf_path"] = str(self._resolve_urdf_path(side_cfg["urdf_path"]))
            fixed_cfg[side] = side_cfg
        return fixed_cfg

    def _resolve_urdf_path(self, urdf_path_str):
        urdf_path = Path(urdf_path_str)
        source_path = _ASSETS_ROOT / urdf_path
        if not source_path.exists():
            fallback_path = urdf_path.with_name(urdf_path.name.replace("RH5DG2_R", "RH56DG2_R"))
            source_path = _ASSETS_ROOT / fallback_path
            if not source_path.exists():
                raise FileNotFoundError(f"RH5DG2 URDF not found: {urdf_path}")
            urdf_path = fallback_path

        urdf_text = source_path.read_text(encoding="utf-8")
        mesh_uri = (_DG2_ASSET_DIR / "meshes").resolve().as_uri() + "/"
        rewritten_text = urdf_text
        for package_prefix in (
            "package://RH5DG2_R/meshes/",
            "package://RH5DG2_L/meshes/",
            "package://RH5DG2/meshes/",
        ):
            rewritten_text = rewritten_text.replace(package_prefix, mesh_uri)
        if rewritten_text == urdf_text:
            return urdf_path

        _DG2_URDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        rewritten_path = _DG2_URDF_CACHE_DIR / urdf_path.name
        rewritten_path.write_text(rewritten_text, encoding="utf-8")
        return rewritten_path

    def _load_joint_limits(self, urdf_path, joint_names):
        xml_path = Path(urdf_path)
        if not xml_path.is_absolute():
            xml_path = _ASSETS_ROOT / xml_path
        xml_root = ET.parse(xml_path).getroot()
        limits = {}
        for joint in xml_root.findall("joint"):
            joint_name = joint.get("name")
            limit = joint.find("limit")
            if joint_name is None or limit is None:
                continue
            limits[joint_name] = (float(limit.get("lower", "0.0")), float(limit.get("upper", "0.0")))
        missing = [name for name in joint_names if name not in limits]
        if missing:
            raise ValueError(f"Missing RH5DG2 joint limits for: {missing}")
        return [limits[name] for name in joint_names]

    def _apply_fast_config(self, cfg):
        normal_delta = float(os.getenv("INSPIRE_DG2_FAST_NORMAL_DELTA", "0.0015"))
        low_pass_alpha = float(os.getenv("INSPIRE_DG2_FAST_LOW_PASS_ALPHA", "1.0"))
        for side in ("left", "right"):
            cfg[side]["normal_delta"] = normal_delta
            cfg[side]["low_pass_alpha"] = low_pass_alpha

    def _configure_fast_optimizer(self, retargeting):
        opt = retargeting.optimizer.opt
        maxeval = int(os.getenv("INSPIRE_DG2_FAST_MAXEVAL", "14"))
        ftol_abs = float(os.getenv("INSPIRE_DG2_FAST_FTOL_ABS", "1e-4"))
        xtol_abs = float(os.getenv("INSPIRE_DG2_FAST_XTOL_ABS", "1e-4"))
        maxtime = float(os.getenv("INSPIRE_DG2_FAST_MAXTIME", "0.006"))
        if maxeval > 0 and hasattr(opt, "set_maxeval"):
            opt.set_maxeval(maxeval)
        if ftol_abs > 0.0 and hasattr(opt, "set_ftol_abs"):
            opt.set_ftol_abs(ftol_abs)
        if xtol_abs > 0.0 and hasattr(opt, "set_xtol_abs"):
            opt.set_xtol_abs(xtol_abs)
        if maxtime > 0.0 and hasattr(opt, "set_maxtime"):
            opt.set_maxtime(maxtime)
        if retargeting.filter is not None and float(os.getenv("INSPIRE_DG2_FAST_LOW_PASS_ALPHA", "1.0")) >= 1.0:
            retargeting.filter = None

    def _set_open_initial_qpos(self, retargeting, joint_names, joint_limits):
        robot_qpos = np.zeros(retargeting.optimizer.robot.dof, dtype=np.float32)
        name_to_open = {
            name: _open_joint_value(name, limit)
            for name, limit in zip(joint_names, joint_limits)
        }
        for idx, joint_name in zip(retargeting.optimizer.idx_pin2target, retargeting.optimizer.target_joint_names):
            robot_qpos[idx] = name_to_open.get(joint_name, 0.0)
        retargeting.set_qpos(robot_qpos)
        if retargeting.filter is not None:
            retargeting.filter.reset()


def _dg2_raw_command_from_retarget(q_target, joint_limits, safe_limits=True):
    q = _clip_to_joint_limits(q_target, joint_limits)
    normalized = _normalize_to_open_ratio(q, joint_limits)
    cmd = _DG2_ANGLE_CLOSED.copy()
    for raw_idx, retarget_idx in enumerate(_DG2_RAW_FROM_RETARGET):
        if raw_idx in _DG2_SIDE_SWING_RAW_INDICES:
            cmd[raw_idx] = np.degrees(q[retarget_idx]) * 10.0
        else:
            cmd[raw_idx] = _DG2_ANGLE_CLOSED[raw_idx] + normalized[retarget_idx] * (
                _DG2_ANGLE_OPEN[raw_idx] - _DG2_ANGLE_CLOSED[raw_idx]
            )
    cmd = np.clip(cmd, _DG2_ANGLE_MIN, _DG2_ANGLE_MAX)
    if safe_limits:
        cmd = np.clip(cmd, _DG2_SAFE_COMMAND_MIN, _DG2_SAFE_COMMAND_MAX)
    if _env_flag("INSPIRE_DG2_LOCK_V_JOINTS", True):
        cmd[list(_DG2_SPREAD_INDICES)] = _DG2_SPREAD_NEUTRAL["right"]
    return np.rint(cmd).astype(np.int16)


def _lock_dg2_spread_joints(cmd, side="right"):
    locked = np.asarray(cmd, dtype=np.float64).copy()
    spread_neutral = _DG2_SPREAD_NEUTRAL.get(str(side).lower(), _DG2_SPREAD_NEUTRAL["right"])
    locked[list(_DG2_SPREAD_INDICES)] = spread_neutral
    return np.rint(locked).astype(np.int16)


def _normalize_inspire_sixdof_targets(left_q_target, right_q_target):
    def normalize(val, min_val, max_val):
        return np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)

    for idx in range(6):
        if idx <= 3:
            left_q_target[idx] = normalize(left_q_target[idx], 0.0, 1.7)
            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.7)
        elif idx == 4:
            left_q_target[idx] = normalize(left_q_target[idx], 0.0, 0.5)
            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 0.5)
        elif idx == 5:
            left_q_target[idx] = normalize(left_q_target[idx], -0.1, 1.3)
            right_q_target[idx] = normalize(right_q_target[idx], -0.1, 1.3)


def _dg2_command_from_inspire_sixdof(q_target, side="right", safe_limits=True):
    q = np.clip(np.asarray(q_target, dtype=np.float64), 0.0, 1.0)
    cmd = _DG2_ANGLE_OPEN.copy()

    # Inspire order: pinky, ring, middle, index, thumb bend, thumb rotation.
    mapping = {
        0: (0, 6),
        1: (1, 7),
        2: (2, 8),
        3: (4, 9),
        4: (11, 12),
        5: (10,),
    }
    for src_idx, dg2_indices in mapping.items():
        for dst_idx in dg2_indices:
            cmd[dst_idx] = _DG2_ANGLE_CLOSED[dst_idx] + q[src_idx] * (
                _DG2_ANGLE_OPEN[dst_idx] - _DG2_ANGLE_CLOSED[dst_idx]
            )

    spread_neutral = _DG2_SPREAD_NEUTRAL.get(str(side).lower(), _DG2_SPREAD_NEUTRAL["right"])
    cmd[list(_DG2_SPREAD_INDICES)] = spread_neutral
    cmd = np.clip(cmd, _DG2_ANGLE_MIN, _DG2_ANGLE_MAX)
    if safe_limits:
        cmd = np.clip(cmd, _DG2_SAFE_COMMAND_MIN, _DG2_SAFE_COMMAND_MAX)
    cmd[list(_DG2_SPREAD_INDICES)] = spread_neutral
    return np.rint(cmd).astype(np.int16)


def _apply_pinch_boost(cmd, hand_data):
    if not _env_flag("INSPIRE_DG2_ENABLE_PINCH_BOOST", True):
        return cmd
    boosted = np.asarray(cmd, dtype=np.float64).copy()
    activation, _distance = _pinch_activation(hand_data)
    if activation <= 0.0:
        return np.rint(boosted).astype(np.int16)

    strength = np.clip(_env_float("INSPIRE_DG2_PINCH_BOOST_STRENGTH", 0.35), 0.0, 1.0)
    index_scale = np.clip(_env_float("INSPIRE_DG2_PINCH_INDEX_SCALE", 0.60), 0.0, 2.0)
    thumb_scale = np.clip(_env_float("INSPIRE_DG2_PINCH_THUMB_SCALE", 0.25), 0.0, 2.0)
    for raw_idx in _DG2_PINCH_RAW_INDICES:
        scale = index_scale if raw_idx in (4, 9) else thumb_scale
        gain = np.clip(activation * strength * scale, 0.0, 1.0)
        target = _DG2_PINCH_CLOSE_TARGET[raw_idx]
        boosted[raw_idx] = boosted[raw_idx] + gain * (target - boosted[raw_idx])

    boosted = np.clip(boosted, _DG2_SAFE_COMMAND_MIN, _DG2_SAFE_COMMAND_MAX)
    return np.rint(boosted).astype(np.int16)


def _apply_thumb_curl_boost(
    cmd,
    hand_data,
    side="right",
    curl_gain=1.0,
    threshold=0.12,
    strength=0.0,
    first_scale=1.0,
    second_scale=1.0,
    safe_limits=True,
):
    boosted = np.asarray(cmd, dtype=np.float64).copy()
    raw_curl = _finger_curl_score(hand_data, "thumb")
    curl_gain = float(np.clip(curl_gain, 0.0, 10.0))
    threshold = float(np.clip(threshold, 0.0, 0.95))
    strength = float(np.clip(strength, 0.0, 1.0))
    scaled_curl = float(np.clip(raw_curl * curl_gain, 0.0, 1.0))
    activation = float(np.clip((scaled_curl - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0))
    before = boosted[list(_DG2_THUMB_BEND_RAW_INDICES)].copy()

    if activation > 0.0 and strength > 0.0:
        for raw_idx, scale in zip(_DG2_THUMB_BEND_RAW_INDICES, (first_scale, second_scale)):
            gain = float(np.clip(activation * strength * float(scale), 0.0, 1.0))
            target = float(_DG2_ANGLE_CLOSED[raw_idx])
            boosted[raw_idx] = boosted[raw_idx] + gain * (target - boosted[raw_idx])

    boosted = np.clip(boosted, _DG2_ANGLE_MIN, _DG2_ANGLE_MAX)
    if safe_limits:
        boosted = np.clip(boosted, _DG2_SAFE_COMMAND_MIN, _DG2_SAFE_COMMAND_MAX)
    after = boosted[list(_DG2_THUMB_BEND_RAW_INDICES)].copy()
    return np.rint(boosted).astype(np.int16), {
        "side": str(side),
        "raw_curl": float(raw_curl),
        "scaled_curl": scaled_curl,
        "curl_gain": curl_gain,
        "threshold": threshold,
        "activation": activation,
        "strength": strength,
        "first_scale": float(first_scale),
        "second_scale": float(second_scale),
        "before": np.round(before, 4).tolist(),
        "after": np.round(after, 4).tolist(),
        "delta": np.round(after - before, 4).tolist(),
    }


def _apply_middle_curl_boost(cmd, hand_data):
    if not _env_flag("INSPIRE_DG2_ENABLE_MIDDLE_CURL_BOOST", False):
        return cmd
    boosted = np.asarray(cmd, dtype=np.float64).copy()
    curl = _finger_curl_score(hand_data, "middle")
    threshold = np.clip(_env_float("INSPIRE_DG2_MIDDLE_CURL_THRESHOLD", 0.18), 0.0, 0.95)
    activation = np.clip((curl - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    if activation <= 0.0:
        return np.rint(boosted).astype(np.int16)

    strength = np.clip(_env_float("INSPIRE_DG2_MIDDLE_CURL_STRENGTH", 0.65), 0.0, 1.0)
    mcp_scale = np.clip(_env_float("INSPIRE_DG2_MIDDLE_MCP_SCALE", 1.0), 0.0, 2.0)
    pip_scale = np.clip(_env_float("INSPIRE_DG2_MIDDLE_PIP_SCALE", 0.35), 0.0, 2.0)
    for raw_idx in _DG2_MIDDLE_RAW_INDICES:
        scale = mcp_scale if raw_idx == 2 else pip_scale
        gain = np.clip(activation * strength * scale, 0.0, 1.0)
        target = _DG2_MIDDLE_CLOSE_TARGET[raw_idx]
        boosted[raw_idx] = boosted[raw_idx] + gain * (target - boosted[raw_idx])

    boosted = np.clip(boosted, _DG2_SAFE_COMMAND_MIN, _DG2_SAFE_COMMAND_MAX)
    return np.rint(boosted).astype(np.int16)


def _apply_middle_open_recovery(cmd, hand_data, side="right"):
    if not _env_flag("INSPIRE_DG2_ENABLE_MIDDLE_OPEN_RECOVERY", True):
        return cmd
    enabled_sides = {
        item.strip().lower()
        for item in os.getenv("INSPIRE_DG2_MIDDLE_OPEN_RECOVERY_SIDES", "right").split(",")
        if item.strip()
    }
    side_name = str(side).lower()
    if side_name not in enabled_sides and "both" not in enabled_sides:
        return cmd

    curl = _finger_curl_score(hand_data, "middle")
    threshold = np.clip(_env_float("INSPIRE_DG2_MIDDLE_OPEN_RECOVERY_THRESHOLD", 0.14), 0.01, 0.95)
    activation = np.clip((threshold - curl) / threshold, 0.0, 1.0)
    if activation <= 0.0:
        return cmd

    boosted = np.asarray(cmd, dtype=np.float64).copy()
    mcp_strength = np.clip(_env_float("INSPIRE_DG2_MIDDLE_OPEN_MCP_STRENGTH", 0.65), 0.0, 1.0)
    pip_strength = np.clip(_env_float("INSPIRE_DG2_MIDDLE_OPEN_PIP_STRENGTH", 0.0), 0.0, 1.0)
    boosted[2] = boosted[2] + activation * mcp_strength * (_DG2_ANGLE_OPEN[2] - boosted[2])
    if pip_strength > 0.0:
        boosted[8] = boosted[8] + activation * pip_strength * (_DG2_ANGLE_OPEN[8] - boosted[8])
    boosted = np.clip(boosted, _DG2_SAFE_COMMAND_MIN, _DG2_SAFE_COMMAND_MAX)
    return np.rint(boosted).astype(np.int16)


def _apply_middle_open_ratio_recovery(cmd, q_target, side="right"):
    if not _env_flag("INSPIRE_DG2_ENABLE_MIDDLE_OPEN_RATIO_RECOVERY", True):
        return cmd
    enabled_sides = {
        item.strip().lower()
        for item in os.getenv("INSPIRE_DG2_MIDDLE_OPEN_RATIO_RECOVERY_SIDES", "right").split(",")
        if item.strip()
    }
    side_name = str(side).lower()
    if side_name not in enabled_sides and "both" not in enabled_sides:
        return cmd

    q = np.asarray(q_target, dtype=np.float64)
    if q.shape[0] <= 2:
        return cmd
    open_ratio = float(np.clip(q[2], 0.0, 1.0))
    threshold = np.clip(_env_float("INSPIRE_DG2_MIDDLE_OPEN_RATIO_THRESHOLD", 0.72), 0.0, 0.99)
    if open_ratio <= threshold:
        return cmd

    activation = (open_ratio - threshold) / max(1.0 - threshold, 1e-6)
    open_floor_start = _env_float("INSPIRE_DG2_MIDDLE_OPEN_RATIO_FLOOR_START", 1680.0)
    open_floor = open_floor_start + activation * (_DG2_ANGLE_OPEN[2] - open_floor_start)
    boosted = np.asarray(cmd, dtype=np.float64).copy()
    boosted[2] = max(boosted[2], open_floor)
    boosted = np.clip(boosted, _DG2_SAFE_COMMAND_MIN, _DG2_SAFE_COMMAND_MAX)
    return np.rint(boosted).astype(np.int16)


class InspireDG2_Controller:
    def __init__(
        self,
        left_hand_array,
        right_hand_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
        fps=50.0,
        Unit_Test=False,
        simulation_mode=False,
        left_port="/dev/ttyUSB0",
        right_port="/dev/ttyUSB0",
        baudrate=115200,
        left_hand_id=2,
        right_hand_id=1,
        state_hz=20.0,
        tactile_hz=30.0,
        transport="dds",
        bridge_host="192.168.123.164",
        bridge_port=9720,
        dds_domain_id=0,
        network_interface=None,
        cmd_topic=kTopicInspireDG2Command,
        state_topic=kTopicInspireDG2State,
        fast_mode=True,
        retarget_mode="config",
        safe_command_limits=True,
        use_inspire6dof=True,
        thumb_curl_gain=1.0,
        right_thumb_curl_gain=1.0,
        thumb_curl_threshold=0.12,
        thumb_curl_strength=0.0,
        thumb_curl_first_scale=1.0,
        thumb_curl_second_scale=1.0,
        thumb_curl_log_rate=0.0,
    ):
        logger_mp.info("Initialize InspireDG2_Controller...")
        self.fps = float(fps)
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode
        self.transport = str(transport).lower()
        self.left_port = left_port
        self.right_port = right_port
        self.baudrate = int(baudrate)
        self.left_hand_id = int(left_hand_id)
        self.right_hand_id = int(right_hand_id)
        self.state_hz = float(state_hz)
        self.tactile_hz = float(tactile_hz)
        self.bridge_host = bridge_host
        self.bridge_port = int(bridge_port)
        self.dds_domain_id = int(dds_domain_id)
        self.network_interface = network_interface
        self.cmd_topic = cmd_topic
        self.state_topic = state_topic
        self.fast_mode = bool(fast_mode)
        self.retarget_mode = retarget_mode
        self.safe_command_limits = bool(safe_command_limits)
        self.use_inspire6dof = _env_flag("INSPIRE_DG2_USE_6DOF", use_inspire6dof)
        self.thumb_curl_gain = float(np.clip(thumb_curl_gain, 0.0, 10.0))
        self.right_thumb_curl_gain = float(np.clip(right_thumb_curl_gain, 0.0, 10.0))
        self.thumb_curl_threshold = float(np.clip(thumb_curl_threshold, 0.0, 0.95))
        self.thumb_curl_strength = float(np.clip(thumb_curl_strength, 0.0, 1.0))
        self.thumb_curl_first_scale = float(np.clip(thumb_curl_first_scale, 0.0, 3.0))
        self.thumb_curl_second_scale = float(np.clip(thumb_curl_second_scale, 0.0, 3.0))
        self.thumb_curl_log_rate = max(float(thumb_curl_log_rate), 0.0)
        self._last_thumb_curl_log_ts = {"left": 0.0, "right": 0.0}

        logger_mp.info(
            "[InspireDG2 thumb curl boost] gain=%.3f right_gain=%.3f threshold=%.3f strength=%.3f first_scale=%.3f second_scale=%.3f log_rate=%.3f",
            self.thumb_curl_gain,
            self.right_thumb_curl_gain,
            self.thumb_curl_threshold,
            self.thumb_curl_strength,
            self.thumb_curl_first_scale,
            self.thumb_curl_second_scale,
            self.thumb_curl_log_rate,
        )

        if self.use_inspire6dof:
            if not self.Unit_Test:
                self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND)
            else:
                self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND_Unit_Test)
            logger_mp.info("[InspireDG2] using Inspire 6DOF retargeting compatibility mode; DG2 V joints are locked neutral.")
        else:
            self.hand_retargeting = _InspireDG2Retargeting(
                fast_mode=self.fast_mode,
                retarget_mode=self.retarget_mode,
            )

        # Raw command override, used by the policy-inference path: [enabled, left*N, right*N].
        # A trained policy predicts the same raw angleSet values this loop publishes, so it
        # must bypass landmark retargeting and every curl/pinch heuristic applied after it.
        self.raw_command_array = Array("d", 1 + 2 * InspireDG2_Num_Motors, lock=True)

        self.left_hand_state_array = Array("d", InspireDG2_Num_Motors, lock=True)
        self.right_hand_state_array = Array("d", InspireDG2_Num_Motors, lock=True)
        self.left_tactile_array = Array("d", InspireDG2_Tactile_Num_Values, lock=True)
        self.right_tactile_array = Array("d", InspireDG2_Tactile_Num_Values, lock=True)
        self.tactile_timestamp_array = Array("d", 2, lock=True)

        control_target = self.control_process
        control_args = (
            left_hand_array,
            right_hand_array,
            self.left_hand_state_array,
            self.right_hand_state_array,
            self.left_tactile_array,
            self.right_tactile_array,
            self.tactile_timestamp_array,
            dual_hand_data_lock,
            dual_hand_state_array,
            dual_hand_action_array,
        )
        if self.transport == "dds":
            hand_control_worker = threading.Thread(target=control_target, args=control_args, daemon=True)
        else:
            hand_control_worker = Process(target=control_target, args=control_args)
            hand_control_worker.daemon = True
        hand_control_worker.start()
        self.hand_control_process = hand_control_worker

        logger_mp.info("Initialize InspireDG2_Controller OK!")

    def set_raw_command(self, left_cmd, right_cmd):
        """Publish these raw angleSet values instead of the retargeted command.

        Both arrays are InspireDG2_Num_Motors long, in raw motor units. Stays in effect
        until clear_raw_command(); the control loop keeps its own rate and DDS/serial
        transport, so this only replaces where the numbers come from.
        """
        left = np.asarray(left_cmd, dtype=np.float64).reshape(-1)
        right = np.asarray(right_cmd, dtype=np.float64).reshape(-1)
        if left.size != InspireDG2_Num_Motors or right.size != InspireDG2_Num_Motors:
            raise ValueError(
                f"raw command needs {InspireDG2_Num_Motors} values per hand, "
                f"got {left.size}/{right.size}"
            )
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("raw command contains non-finite values")
        with self.raw_command_array.get_lock():
            self.raw_command_array[0] = 1.0
            self.raw_command_array[1:1 + InspireDG2_Num_Motors] = left
            self.raw_command_array[1 + InspireDG2_Num_Motors:] = right

    def clear_raw_command(self):
        """Hand control back to the landmark retargeting path."""
        with self.raw_command_array.get_lock():
            self.raw_command_array[0] = 0.0

    def _read_raw_command(self):
        with self.raw_command_array.get_lock():
            if self.raw_command_array[0] < 0.5:
                return None, None
            left = np.array(self.raw_command_array[1:1 + InspireDG2_Num_Motors], dtype=np.float64)
            right = np.array(self.raw_command_array[1 + InspireDG2_Num_Motors:], dtype=np.float64)
        return left, right

    def _open_udp_socket(self):
        if self.simulation_mode:
            return None
        if not self.bridge_host:
            raise RuntimeError("Inspire RH5DG2 UDP bridge host is not configured.")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)
        logger_mp.info("[InspireDG2] using UDP bridge host=%s port=%s", self.bridge_host, self.bridge_port)
        return sock

    def _send_udp_command(self, sock, left_cmd, right_cmd, seq):
        if sock is None:
            return
        left_cmd = _lock_dg2_spread_joints(left_cmd, side="left")
        right_cmd = _lock_dg2_spread_joints(right_cmd, side="right")
        packet = {
            "type": "command",
            "seq": int(seq),
            "timestamp": time.time(),
            "left": {
                "id": self.left_hand_id,
                "angle_set": [int(v) for v in left_cmd],
            },
            "right": {
                "id": self.right_hand_id,
                "angle_set": [int(v) for v in right_cmd],
            },
        }
        data = json.dumps(packet, separators=(",", ":")).encode("utf-8")
        sock.sendto(data, (self.bridge_host, self.bridge_port))

    def _drain_udp_state(self, sock, left_hand_state_array, right_hand_state_array,
                         left_tactile_array, right_tactile_array, tactile_timestamp_array):
        if sock is None:
            return
        while True:
            try:
                raw, _addr = sock.recvfrom(65535)
            except BlockingIOError:
                return
            except OSError as exc:
                logger_mp.warning("[InspireDG2] UDP state receive failed: %s", exc)
                return
            try:
                packet = json.loads(raw.decode("utf-8"))
            except Exception as exc:
                logger_mp.warning("[InspireDG2] UDP state packet parse failed: %s", exc)
                continue
            self._apply_udp_state_packet(
                packet,
                left_hand_state_array,
                right_hand_state_array,
                left_tactile_array,
                right_tactile_array,
                tactile_timestamp_array,
            )

    def _apply_udp_state_packet(self, packet, left_hand_state_array, right_hand_state_array,
                                left_tactile_array, right_tactile_array, tactile_timestamp_array):
        for side_name, state_array, tactile_array, tactile_idx in (
            ("left_ee", left_hand_state_array, left_tactile_array, 0),
            ("right_ee", right_hand_state_array, right_tactile_array, 1),
        ):
            data = packet.get(side_name)
            if not isinstance(data, dict):
                data = packet.get(side_name.replace("_ee", ""))
            if not isinstance(data, dict):
                continue
            angle_act = data.get("angle_act")
            if isinstance(angle_act, (list, tuple)) and len(angle_act) >= InspireDG2_Num_Motors:
                self._write_state(state_array, angle_act[:InspireDG2_Num_Motors])
            tactile = data.get("tactile")
            if isinstance(tactile, dict):
                self._write_tactile(tactile_array, tactile_timestamp_array, tactile_idx, tactile)

    def _open_dds(self):
        ChannelFactoryInitialize(self.dds_domain_id, networkInterface=self.network_interface)
        publisher = ChannelPublisher(self.cmd_topic, MotorCmds_)
        publisher.Init()
        subscriber = ChannelSubscriber(self.state_topic, MotorStates_)
        subscriber.Init()
        logger_mp.info(
            "[InspireDG2] using DDS bridge cmd_topic=%s state_topic=%s domain=%s network_interface=%s",
            self.cmd_topic,
            self.state_topic,
            self.dds_domain_id,
            self.network_interface,
        )
        return publisher, subscriber

    @staticmethod
    def _set_dds_cmd_active(cmd, active=True):
        cmd.mode = 0b0001 if active else 0
        reserve = [1, 0, 0] if active else [0, 0, 0]
        try:
            cmd.reserve = reserve
        except Exception:
            try:
                cmd.reserve[0] = reserve[0]
            except Exception:
                pass

    def _ensure_dds_cmd_msg(self):
        if hasattr(self, "hand_msg"):
            return
        self.hand_msg = MotorCmds_()
        self.hand_msg.cmds = [
            unitree_go_msg_dds__MotorCmd_()
            for _ in range(InspireDG2_Num_Motors * 2)
        ]

    def _publish_dds_command(self, publisher, left_cmd, right_cmd):
        if publisher is None:
            return
        self._ensure_dds_cmd_msg()
        right_cmd = np.asarray(right_cmd, dtype=np.float64).reshape(InspireDG2_Num_Motors)
        left_cmd = np.asarray(left_cmd, dtype=np.float64).reshape(InspireDG2_Num_Motors)
        right_cmd = _lock_dg2_spread_joints(right_cmd, side="right")
        left_cmd = _lock_dg2_spread_joints(left_cmd, side="left")
        for idx in range(InspireDG2_Num_Motors):
            cmd = self.hand_msg.cmds[idx]
            cmd.q = float(right_cmd[idx])
            cmd.dq = 0.0
            cmd.tau = 0.0
            cmd.kp = 1.0
            cmd.kd = 0.05
            self._set_dds_cmd_active(cmd, True)
        for idx in range(InspireDG2_Num_Motors):
            cmd = self.hand_msg.cmds[InspireDG2_Num_Motors + idx]
            cmd.q = float(left_cmd[idx])
            cmd.dq = 0.0
            cmd.tau = 0.0
            cmd.kp = 1.0
            cmd.kd = 0.05
            self._set_dds_cmd_active(cmd, True)
        publisher.Write(self.hand_msg)

    def _read_dds_state(self, subscriber, left_hand_state_array, right_hand_state_array):
        if subscriber is None:
            return
        msg = subscriber.Read()
        if msg is None or not hasattr(msg, "states"):
            return
        states = list(msg.states)
        if len(states) < InspireDG2_Num_Motors * 2:
            return
        right_state = [float(states[idx].q) for idx in range(InspireDG2_Num_Motors)]
        left_state = [
            float(states[InspireDG2_Num_Motors + idx].q)
            for idx in range(InspireDG2_Num_Motors)
        ]
        self._write_state(left_hand_state_array, left_state)
        self._write_state(right_hand_state_array, right_state)

    def _start_dds_state_thread(self, subscriber, left_hand_state_array, right_hand_state_array):
        stop_event = threading.Event()

        def state_loop():
            while not stop_event.is_set() and getattr(self, "running", False):
                try:
                    self._read_dds_state(subscriber, left_hand_state_array, right_hand_state_array)
                except Exception as exc:
                    logger_mp.debug("[InspireDG2] DDS state read failed: %s", exc)
                time.sleep(0.002)

        thread = threading.Thread(target=state_loop, daemon=True)
        thread.start()
        return stop_event, thread

    def _open_serials(self):
        if self.simulation_mode:
            return {}
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is required for Inspire RH5DG2 serial control. "
                "Install it in the G1 robot computer/container that runs teleop: "
                "python3 -m pip install pyserial"
            ) from exc

        serials = {}
        for port in {self.left_port, self.right_port}:
            if not port:
                continue
            ser = serial.Serial(port, self.baudrate, timeout=0.01)
            if hasattr(ser, "reset_input_buffer"):
                ser.reset_input_buffer()
            serials[port] = ser
            logger_mp.info("[InspireDG2] opened serial port=%s baudrate=%s", port, self.baudrate)
        return serials

    def _side_configs(self):
        return {
            "left": {"port": self.left_port, "hand_id": self.left_hand_id},
            "right": {"port": self.right_port, "hand_id": self.right_hand_id},
        }

    def _maybe_log_thumb_curl_debug(self, debug):
        if self.thumb_curl_log_rate <= 0.0 or not debug:
            return
        side = debug.get("side", "unknown")
        now = time.time()
        interval = 1.0 / max(self.thumb_curl_log_rate, 1e-6)
        last = self._last_thumb_curl_log_ts.get(side, 0.0)
        if now - last < interval:
            return
        self._last_thumb_curl_log_ts[side] = now
        logger_mp.info(
            "[InspireDG2 thumb curl] side=%s raw=%.4f scaled=%.4f gain=%.3f threshold=%.3f activation=%.4f strength=%.3f before=%s after=%s delta=%s",
            side,
            debug["raw_curl"],
            debug["scaled_curl"],
            debug["curl_gain"],
            debug["threshold"],
            debug["activation"],
            debug["strength"],
            debug["before"],
            debug["after"],
            debug["delta"],
        )

    def _retarget(self, left_hand_data, right_hand_data, left_q_target, right_q_target):
        if np.allclose(right_hand_data, 0.0, atol=1e-5) or np.allclose(left_hand_data, 0.0, atol=1e-5):
            return left_q_target, right_q_target
        if np.allclose(left_hand_data[4], np.array([-1.13, 0.3, 0.15]), atol=1e-3):
            return left_q_target, right_q_target

        if self.use_inspire6dof:
            ref_left_value = (
                left_hand_data[self.hand_retargeting.left_indices[1, :]]
                - left_hand_data[self.hand_retargeting.left_indices[0, :]]
            )
            ref_right_value = (
                right_hand_data[self.hand_retargeting.right_indices[1, :]]
                - right_hand_data[self.hand_retargeting.right_indices[0, :]]
            )
            left_q_target = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[
                self.hand_retargeting.left_dex_retargeting_to_hardware
            ]
            right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[
                self.hand_retargeting.right_dex_retargeting_to_hardware
            ]
            _normalize_inspire_sixdof_targets(left_q_target, right_q_target)
            return left_q_target, right_q_target

        left_idx = np.asarray(self.hand_retargeting.left_indices)
        right_idx = np.asarray(self.hand_retargeting.right_indices)
        if left_idx.ndim == 2:
            ref_left_value = _prepare_vector_reference(left_hand_data, left_idx)
            ref_right_value = _prepare_vector_reference(right_hand_data, right_idx)
        else:
            ref_left_value = _prepare_position_reference(left_hand_data, left_idx)
            ref_right_value = _prepare_position_reference(right_hand_data, right_idx)

        left_q_target = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[
            self.hand_retargeting.left_retargeting_to_hardware
        ]
        right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[
            self.hand_retargeting.right_retargeting_to_hardware
        ]
        return (
            _clip_to_joint_limits(left_q_target, self.hand_retargeting.left_joint_limits),
            _clip_to_joint_limits(right_q_target, self.hand_retargeting.right_joint_limits),
        )

    def _write_state(self, target_array, values):
        with target_array.get_lock():
            target_array[:] = [float(v) for v in values]

    def _write_tactile(self, target_array, timestamp_array, side_idx, data):
        if data is None:
            return
        flat = []
        for finger in ("little", "ring", "middle", "index", "thumb"):
            flat.extend(data["fingers"].get(finger, [0, 0, 65535, 0]))
        flat.extend(data.get("palm", [0] * 9))
        with target_array.get_lock():
            target_array[:] = [float(v) for v in flat[:InspireDG2_Tactile_Num_Values]]
        with timestamp_array.get_lock():
            timestamp_array[side_idx] = time.time()

    def _read_side_state(self, serials, side, cfg, fallback_cmd):
        port = cfg["port"]
        if self.simulation_mode:
            return fallback_cmd
        if not port or port not in serials:
            return None
        return _read_13_int16(serials[port], cfg["hand_id"], "angleAct")

    def _write_side_command(self, serials, cfg, command):
        port = cfg["port"]
        if self.simulation_mode or not port or port not in serials:
            return
        command = _lock_dg2_spread_joints(command)
        _write_register(serials[port], cfg["hand_id"], _REG["angleSet"], command)

    def _read_side_tactile(self, serials, cfg):
        port = cfg["port"]
        if self.simulation_mode or not port or port not in serials:
            return None
        return _read_touch_data(serials[port], cfg["hand_id"])

    def control_process(
        self,
        left_hand_array,
        right_hand_array,
        left_hand_state_array,
        right_hand_state_array,
        left_tactile_array,
        right_tactile_array,
        tactile_timestamp_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
    ):
        logger_mp.info("[InspireDG2] Control process started.")
        self.running = True
        if self.use_inspire6dof:
            left_q_target = np.full(6, 1.0, dtype=np.float64)
            right_q_target = np.full(6, 1.0, dtype=np.float64)
            left_cmd = _dg2_command_from_inspire_sixdof(
                left_q_target,
                side="left",
                safe_limits=self.safe_command_limits,
            )
            right_cmd = _dg2_command_from_inspire_sixdof(
                right_q_target,
                side="right",
                safe_limits=self.safe_command_limits,
            )
        else:
            left_q_target = np.array(
                [
                    _open_joint_value(name, limit)
                    for name, limit in zip(
                        self.hand_retargeting.left_joint_names,
                        self.hand_retargeting.left_joint_limits,
                    )
                ],
                dtype=np.float64,
            )
            right_q_target = np.array(
                [
                    _open_joint_value(name, limit)
                    for name, limit in zip(
                        self.hand_retargeting.right_joint_names,
                        self.hand_retargeting.right_joint_limits,
                    )
                ],
                dtype=np.float64,
            )
            left_cmd = _dg2_raw_command_from_retarget(
                left_q_target,
                self.hand_retargeting.left_joint_limits,
                safe_limits=self.safe_command_limits,
            )
            right_cmd = _dg2_raw_command_from_retarget(
                right_q_target,
                self.hand_retargeting.right_joint_limits,
                safe_limits=self.safe_command_limits,
            )
        side_configs = self._side_configs()
        udp_sock = None
        dds_publisher = None
        dds_subscriber = None
        dds_state_stop = None
        dds_state_thread = None
        seq = 0

        if self.transport == "dds":
            try:
                dds_publisher, dds_subscriber = self._open_dds()
                dds_state_stop, dds_state_thread = self._start_dds_state_thread(
                    dds_subscriber,
                    left_hand_state_array,
                    right_hand_state_array,
                )
            except Exception as exc:
                logger_mp.exception("[InspireDG2] Failed to open DDS bridge client: %s", exc)
            serials = {}
        elif self.transport == "udp":
            try:
                udp_sock = self._open_udp_socket()
            except Exception as exc:
                logger_mp.exception("[InspireDG2] Failed to open UDP bridge client: %s", exc)
            serials = {}
        else:
            try:
                serials = self._open_serials()
            except Exception as exc:
                logger_mp.exception("[InspireDG2] Failed to open serial port: %s", exc)
                serials = {}

        if self.transport == "serial" and not self.simulation_mode and not serials:
            logger_mp.warning("[InspireDG2] No serial ports opened; command loop will run without hardware writes.")
        if self.transport == "dds" and not self.simulation_mode and dds_publisher is None:
            logger_mp.warning("[InspireDG2] DDS bridge unavailable; command loop will run without hardware writes.")
        if self.transport == "udp" and not self.simulation_mode and udp_sock is None:
            logger_mp.warning("[InspireDG2] UDP bridge unavailable; command loop will run without hardware writes.")

        period = 1.0 / max(self.fps, 1.0)
        state_period = 1.0 / max(self.state_hz, 1.0)
        tactile_period = 1.0 / max(self.tactile_hz, 1.0)
        next_state_time = 0.0
        next_tactile_time = 0.0

        try:
            while self.running:
                start_time = time.time()
                with left_hand_array.get_lock():
                    left_hand_data = np.array(left_hand_array[:]).reshape(25, 3).copy()
                with right_hand_array.get_lock():
                    right_hand_data = np.array(right_hand_array[:]).reshape(25, 3).copy()

                left_q_target, right_q_target = self._retarget(
                    left_hand_data,
                    right_hand_data,
                    left_q_target,
                    right_q_target,
                )
                if self.use_inspire6dof:
                    left_cmd = _dg2_command_from_inspire_sixdof(
                        left_q_target,
                        side="left",
                        safe_limits=self.safe_command_limits,
                    )
                    right_cmd = _dg2_command_from_inspire_sixdof(
                        right_q_target,
                        side="right",
                        safe_limits=self.safe_command_limits,
                    )
                    left_cmd = _apply_pinch_boost(left_cmd, left_hand_data)
                    right_cmd = _apply_pinch_boost(right_cmd, right_hand_data)
                    left_cmd, left_thumb_debug = _apply_thumb_curl_boost(
                        left_cmd,
                        left_hand_data,
                        side="left",
                        curl_gain=self.thumb_curl_gain,
                        threshold=self.thumb_curl_threshold,
                        strength=self.thumb_curl_strength,
                        first_scale=self.thumb_curl_first_scale,
                        second_scale=self.thumb_curl_second_scale,
                        safe_limits=self.safe_command_limits,
                    )
                    right_cmd, right_thumb_debug = _apply_thumb_curl_boost(
                        right_cmd,
                        right_hand_data,
                        side="right",
                        curl_gain=self.thumb_curl_gain * self.right_thumb_curl_gain,
                        threshold=self.thumb_curl_threshold,
                        strength=self.thumb_curl_strength,
                        first_scale=self.thumb_curl_first_scale,
                        second_scale=self.thumb_curl_second_scale,
                        safe_limits=self.safe_command_limits,
                    )
                    self._maybe_log_thumb_curl_debug(left_thumb_debug)
                    self._maybe_log_thumb_curl_debug(right_thumb_debug)
                    left_cmd = _apply_middle_open_recovery(left_cmd, left_hand_data, side="left")
                    right_cmd = _apply_middle_open_recovery(right_cmd, right_hand_data, side="right")
                    left_cmd = _apply_middle_open_ratio_recovery(left_cmd, left_q_target, side="left")
                    right_cmd = _apply_middle_open_ratio_recovery(right_cmd, right_q_target, side="right")
                    left_cmd = _lock_dg2_spread_joints(left_cmd, side="left")
                    right_cmd = _lock_dg2_spread_joints(right_cmd, side="right")
                else:
                    left_cmd = _dg2_raw_command_from_retarget(
                        left_q_target,
                        self.hand_retargeting.left_joint_limits,
                        safe_limits=self.safe_command_limits,
                    )
                    left_cmd = _apply_middle_curl_boost(left_cmd, left_hand_data)
                    left_cmd = _apply_pinch_boost(left_cmd, left_hand_data)
                    right_cmd = _dg2_raw_command_from_retarget(
                        right_q_target,
                        self.hand_retargeting.right_joint_limits,
                        safe_limits=self.safe_command_limits,
                    )
                    right_cmd = _apply_middle_curl_boost(right_cmd, right_hand_data)
                    right_cmd = _apply_pinch_boost(right_cmd, right_hand_data)
                    left_cmd, left_thumb_debug = _apply_thumb_curl_boost(
                        left_cmd,
                        left_hand_data,
                        side="left",
                        curl_gain=self.thumb_curl_gain,
                        threshold=self.thumb_curl_threshold,
                        strength=self.thumb_curl_strength,
                        first_scale=self.thumb_curl_first_scale,
                        second_scale=self.thumb_curl_second_scale,
                        safe_limits=self.safe_command_limits,
                    )
                    right_cmd, right_thumb_debug = _apply_thumb_curl_boost(
                        right_cmd,
                        right_hand_data,
                        side="right",
                        curl_gain=self.thumb_curl_gain * self.right_thumb_curl_gain,
                        threshold=self.thumb_curl_threshold,
                        strength=self.thumb_curl_strength,
                        first_scale=self.thumb_curl_first_scale,
                        second_scale=self.thumb_curl_second_scale,
                        safe_limits=self.safe_command_limits,
                    )
                    self._maybe_log_thumb_curl_debug(left_thumb_debug)
                    self._maybe_log_thumb_curl_debug(right_thumb_debug)
                    left_cmd = _apply_middle_open_recovery(left_cmd, left_hand_data, side="left")
                    right_cmd = _apply_middle_open_recovery(right_cmd, right_hand_data, side="right")

                left_cmd = _lock_dg2_spread_joints(left_cmd, side="left")
                right_cmd = _lock_dg2_spread_joints(right_cmd, side="right")

                raw_left, raw_right = self._read_raw_command()
                if raw_left is not None:
                    # Policy inference: the action already is a raw angleSet, so it replaces
                    # the retargeted command wholesale (spread lock included).
                    left_cmd, right_cmd = raw_left, raw_right

                now = time.time()
                if self.transport == "dds":
                    self._publish_dds_command(dds_publisher, left_cmd, right_cmd)
                elif self.transport == "udp":
                    seq += 1
                    self._send_udp_command(udp_sock, left_cmd, right_cmd, seq)
                    self._drain_udp_state(
                        udp_sock,
                        left_hand_state_array,
                        right_hand_state_array,
                        left_tactile_array,
                        right_tactile_array,
                        tactile_timestamp_array,
                    )
                elif now >= next_state_time:
                    left_state = self._read_side_state(serials, "left", side_configs["left"], left_cmd)
                    right_state = self._read_side_state(serials, "right", side_configs["right"], right_cmd)
                    if left_state is not None:
                        self._write_state(left_hand_state_array, left_state)
                    if right_state is not None:
                        self._write_state(right_hand_state_array, right_state)
                    next_state_time = now + state_period

                if self.transport == "serial" and now >= next_tactile_time:
                    left_touch = self._read_side_tactile(serials, side_configs["left"])
                    right_touch = self._read_side_tactile(serials, side_configs["right"])
                    self._write_tactile(left_tactile_array, tactile_timestamp_array, 0, left_touch)
                    self._write_tactile(right_tactile_array, tactile_timestamp_array, 1, right_touch)
                    next_tactile_time = now + tactile_period

                if self.transport == "serial":
                    self._write_side_command(serials, side_configs["left"], left_cmd)
                    self._write_side_command(serials, side_configs["right"], right_cmd)

                state_data = np.concatenate(
                    (np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:]))
                )
                action_data = np.concatenate((left_cmd.astype(np.float64), right_cmd.astype(np.float64)))
                if dual_hand_state_array is not None and dual_hand_action_array is not None:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                time.sleep(max(0.0, period - (time.time() - start_time)))
        finally:
            if dds_state_stop is not None:
                dds_state_stop.set()
            if dds_state_thread is not None:
                try:
                    dds_state_thread.join(timeout=0.2)
                except Exception:
                    pass
            for ser in serials.values():
                try:
                    ser.close()
                except Exception:
                    pass
            if udp_sock is not None:
                try:
                    udp_sock.close()
                except Exception:
                    pass
            logger_mp.info("InspireDG2_Controller has been closed.")

    def read_latest_tactile(self):
        result = {}
        for side, side_idx, tactile_array in (
            ("left_ee", 0, self.left_tactile_array),
            ("right_ee", 1, self.right_tactile_array),
        ):
            timestamp = self.tactile_timestamp_array[side_idx]
            if timestamp <= 0.0:
                continue
            with tactile_array.get_lock():
                values = list(tactile_array[:])
            fingers = {}
            for idx, finger in enumerate(("little", "ring", "middle", "index", "thumb")):
                base = idx * 4
                fingers[finger] = values[base : base + 4]
            result[side] = {
                "timestamp": timestamp,
                "fingers": fingers,
                "palm": values[20:29],
            }
        return result


Inspire_Controller_DG2 = InspireDG2_Controller
Inspire_Num_Motors = InspireDG2_Num_Motors
Inspire_Num_Motors_DG2 = InspireDG2_Num_Motors

__all__ = [
    "InspireDG2_Controller",
    "InspireDG2_Num_Motors",
    "InspireDG2_Tactile_Num_Values",
    "Inspire_Controller_DG2",
    "Inspire_Num_Motors",
    "Inspire_Num_Motors_DG2",
]
