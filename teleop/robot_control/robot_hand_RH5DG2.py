from enum import IntEnum
from multiprocessing import Array, Process
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
from unitree_sdk2py.core.channel import ChannelFactory, ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

import logging_mp

_logger_factory = getattr(logging_mp, "getLogger", None) or getattr(logging_mp, "get_logger")
logger_mp = _logger_factory(__name__)


RH5DG2_Num_Motors = 13
RH5DG2_GRASP_SHARPNESS = 1.35
RH5DG2_TELEOP_CLOSE_GAIN = 1.0
RH5DG2_RAW_PITCH_INDICES = (0, 1, 2, 4, 6, 7, 8, 9)
RH5DG2_RAW_SPREAD_INDICES = (3, 5)
RH5DG2_RAW_THUMB_INDICES = (10, 11, 12)
RH5DG2_SAFE_DELTA_LIMIT = {
    0: 600.0,
    1: 600.0,
    2: 600.0,
    3: 200.0,
    4: 600.0,
    5: 200.0,
    6: 750.0,
    7: 750.0,
    8: 750.0,
    9: 720.0,
    10: 600.0,
    11: 150.0,
    12: 450.0,
}
RH5DG2_SAFE_ABS_MAX = np.array(
    [
        1650, 1650, 1650, 200,
        1650, 0,
        1900, 1900, 1900, 1870,
        1950, 1800, 2040,
    ],
    dtype=np.float64,
)
RH5DG2_SAFE_ABS_MIN = np.array(
    [
        1030, 1030, 1030, 0,
        1030, -200,
        1150, 1150, 1150, 1150,
        940, 1250, 1550,
    ],
    dtype=np.float64,
)
RH5DG2_VENDOR_DEMO_OPEN_BASELINE_LEFT = np.array(
    [
        1800, 1800, 1800,
        0,
        1650,
        0,
        1900, 1900, 1900,
        1870,
        1750, 1600, 1930,
    ],
    dtype=np.float64,
)
RH5DG2_VENDOR_DEMO_OPEN_BASELINE_RIGHT = np.array(
    [
        1800, 1800, 1800,
        0,
        1650,
        0,
        1900, 1900, 1900,
        1870,
        1750, 1600, 1930,
    ],
    dtype=np.float64,
)
RH5DG2_SAFE_RAW_FROM_NORMALIZED = {
    # raw angleSet index -> semantic normalized retarget index
    4: 4,   # index root pitch
    9: 5,   # index middle pitch
    2: 7,   # middle root pitch
    8: 8,   # middle middle pitch
    1: 9,   # ring root pitch
    7: 10,  # ring middle pitch
    0: 11,  # little root pitch
    6: 12,  # little middle pitch
}
RH5DG2_SAFE_THUMB_OPEN = {
    10: 1750.0,
    11: 1600.0,
    12: 1930.0,
}
RH5DG2_SAFE_THUMB_CLOSE = {
    10: 1140.0,
    11: 1320.0,
    12: 1880.0,
}
RH5DG2_SAFE_SPREAD_DIRECTION = {
    3: 1.0,
    5: -1.0,
}
RH5DG2_SAFE_ACTIVE_HAND_DEFAULT = "right"
RH5DG2_FINGER_OPEN_CALIBRATION = {
    "thumb": {"indices": (1, 2), "tip": 4, "close": 0.055, "open": 0.105, "gain": 1.2, "bias": 0.0},
    "index": {"indices": (4, 5), "tip": 9, "close": 0.075, "open": 0.140, "gain": 1.8, "bias": 0.0},
    "middle": {"indices": (7, 8), "tip": 14, "close": 0.085, "open": 0.155, "gain": 1.3, "bias": 0.0},
    "ring": {"indices": (9, 10), "tip": 19, "close": 0.075, "open": 0.135, "gain": 2.0, "bias": 0.0},
    "little": {"indices": (11, 12), "tip": 24, "close": 0.065, "open": 0.115, "gain": 2.0, "bias": 0.0},
}
RH5DG2_SAFE_CLOSE_FLOOR = np.array(
    [
        0.45,
        0.35,
        0.45,
        0.50,
        0.12,
        0.12,
        0.50,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
    ],
    dtype=np.float64,
)
_RH5DG2_POSITION_AXES = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
_RH5DG2_PALM_INDICES = np.array([0, 5, 10, 15, 20], dtype=np.int64)
_RH5DG2_FINGER_LANDMARKS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8, 9),
    "middle": (10, 11, 12, 13, 14),
    "ring": (15, 16, 17, 18, 19),
    "little": (20, 21, 22, 23, 24),
}
_RH5DG2_FINGER_FLEX_JOINTS = {
    "thumb": (1, 2),
    "index": (4, 5),
    "middle": (7, 8),
    "ring": (9, 10),
    "little": (11, 12),
}
_RH5DG2_FINGER_DEBUG_NAMES = {
    "thumb": "thumb",
    "index": "index",
    "middle": "middle",
    "ring": "ring",
    "little": "pinky",
}
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ASSETS_ROOT = _REPO_ROOT / "assets"
_RH5DG2_ASSET_DIR = _ASSETS_ROOT / "RH5DG2"
_RH5DG2_CONFIG_PATH = _RH5DG2_ASSET_DIR / "RH5DG2.yml"
_RH5DG2_URDF_CACHE_DIR = Path("/tmp/opencode/rh5dg2_urdf")
_RH5DG2_RETARGET_MODES = ("vector", "dexpilot")

kTopicRH5DG2DFXCommand = "rt/rh5dg2/cmd"
kTopicRH5DG2DFXState = "rt/rh5dg2/state"
kTopicRH5DG2FTPLeftCommand = "rt/rh5dg2_hand/ctrl/l"
kTopicRH5DG2FTPRightCommand = "rt/rh5dg2_hand/ctrl/r"
kTopicRH5DG2FTPLeftState = "rt/rh5dg2_hand/state/l"
kTopicRH5DG2FTPRightState = "rt/rh5dg2_hand/state/r"

# Old Inspire-style names kept so legacy imports resolve to the RH5DG2 topics
# within this compatibility module.
kTopicInspireDFXCommand = kTopicRH5DG2DFXCommand
kTopicInspireDFXState = kTopicRH5DG2DFXState
kTopicInspireFTPLeftCommand = kTopicRH5DG2FTPLeftCommand
kTopicInspireFTPRightCommand = kTopicRH5DG2FTPRightCommand
kTopicInspireFTPLeftState = kTopicRH5DG2FTPLeftState
kTopicInspireFTPRightState = kTopicRH5DG2FTPRightState


def _env_flag(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _normalize_rh5dg2_retarget_mode(mode):
    raw = os.getenv("RH5DG2_RETARGET_MODE") if mode is None else mode
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
    if normalized not in _RH5DG2_RETARGET_MODES:
        raise ValueError(
            f"RH5DG2 retarget mode must be one of config, {', '.join(_RH5DG2_RETARGET_MODES)}; got {raw!r}"
        )
    return normalized


def _fmt_debug(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return "len=0"
    return (
        f"len={arr.size} min={arr.min():.4f} max={arr.max():.4f} "
        f"first5={np.round(arr[:5], 4).tolist()} last5={np.round(arr[-5:], 4).tolist()}"
    )


def _fmt_motor_fields(cmds, start=0, count=5):
    rows = []
    for idx in range(start, min(start + count, len(cmds))):
        motor = cmds[idx]
        reserve = getattr(motor, "reserve", None)
        rows.append(
            f"i={idx} q={getattr(motor, 'q', None)} dq={getattr(motor, 'dq', None)} "
            f"tau={getattr(motor, 'tau', None)} kp={getattr(motor, 'kp', None)} "
            f"kd={getattr(motor, 'kd', None)} mode={getattr(motor, 'mode', None)} "
            f"reserve={list(reserve) if reserve is not None else None}"
        )
    return " | ".join(rows)


def _fmt_motor_fields_for_indices(cmds, indices, offset=0):
    rows = []
    for local_idx in indices:
        idx = offset + int(local_idx)
        if idx < 0 or idx >= len(cmds):
            continue
        motor = cmds[idx]
        reserve = getattr(motor, "reserve", None)
        rows.append(
            f"i={idx} q={getattr(motor, 'q', None)} "
            f"mode={getattr(motor, 'mode', None)} "
            f"reserve={list(reserve) if reserve is not None else None}"
        )
    return " | ".join(rows) if rows else "none"


def _fmt_index_list(values):
    return [int(idx) for idx in sorted(values)]


def _fmt_hand_input(hand_data, indices):
    data = np.asarray(hand_data, dtype=np.float64)
    idx = np.asarray(indices)
    summary = (
        f"shape={data.shape} min={data.min():.4f} max={data.max():.4f} "
        f"p0={np.round(data[0], 4).tolist()}"
    )
    if idx.ndim == 2:
        start_idx = idx[0, :]
        end_idx = idx[1, :]
        start_points = data[start_idx]
        end_points = data[end_idx]
        selected = end_points - start_points
        return (
            f"{summary} idx_start={start_idx.tolist()} idx_end={end_idx.tolist()} "
            f"start_first3={np.round(start_points[:3], 4).tolist()} "
            f"end_first3={np.round(end_points[:3], 4).tolist()} "
            f"ref_first3={np.round(selected[:3], 4).tolist()}"
        )
    selected = data[idx]
    return (
        f"{summary} idx={idx.tolist()} "
        f"selected={np.round(selected, 4).tolist()}"
    )


def _fmt_matrix(values):
    arr = np.asarray(values, dtype=np.float64)
    return np.round(arr, 4).tolist()


def _fmt_named_values(names, values):
    return [
        f"{name}={float(value):.4f}"
        for name, value in zip(names, np.asarray(values, dtype=np.float64))
    ]


def _finger_command_values(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "index": np.round(arr[list(_RH5DG2_FINGER_FLEX_JOINTS["index"])], 4).tolist(),
        "middle": np.round(arr[list(_RH5DG2_FINGER_FLEX_JOINTS["middle"])], 4).tolist(),
        "ring": np.round(arr[list(_RH5DG2_FINGER_FLEX_JOINTS["ring"])], 4).tolist(),
        "pinky": np.round(arr[list(_RH5DG2_FINGER_FLEX_JOINTS["little"])], 4).tolist(),
    }


def _fmt_named_limits(names, joint_limits):
    rows = []
    for name, (lower, upper) in zip(names, joint_limits):
        rows.append(f"{name}=[{lower:.4f},{upper:.4f}] range={upper - lower:.4f}")
    return rows


def _fmt_named_mapping(names, joint_limits):
    rows = []
    for name, (lower, upper) in zip(names, joint_limits):
        rows.append(
            f"{name}: norm1_open_rad={lower:.4f},norm0_close_rad={upper:.4f},"
            f"lower={lower:.4f},upper={upper:.4f}"
        )
    return rows


def _denormalize_from_unit_interval(values, joint_limits):
    normalized = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    out = np.empty(len(normalized), dtype=np.float64)
    for idx, (value, (lower, upper)) in enumerate(zip(normalized, joint_limits)):
        out[idx] = upper - value * (upper - lower)
    return out


def _saturation_summary(raw_values, normalized_values, joint_limits):
    raw = np.asarray(raw_values, dtype=np.float64)
    norm = np.asarray(normalized_values, dtype=np.float64)
    limits = np.asarray(joint_limits, dtype=np.float64)
    lower = limits[:, 0]
    upper = limits[:, 1]
    return {
        "norm0": int(np.sum(norm <= 1e-4)),
        "norm1": int(np.sum(norm >= 1.0 - 1e-4)),
        "raw_near_lower": int(np.sum(raw <= lower + 1e-3)),
        "raw_near_upper": int(np.sum(raw >= upper - 1e-3)),
    }


def _reset_channel_factory_after_fork():
    """Force the Unitree DDS singleton to create a fresh participant in child processes."""
    try:
        ChannelFactory._ChannelFactory__initialized = False
        ChannelFactory._ChannelFactory__domain = None
        ChannelFactory._ChannelFactory__participant = None
        ChannelFactory._ChannelFactory__qos = None
    except Exception as exc:
        logger_mp.warning("[RH5DG2 DDS] Failed to reset ChannelFactory after fork: %s", exc)


def _publisher_debug_status(publisher):
    try:
        channel = getattr(publisher, "_ChannelPublisher__channel", None)
        writer_holder = getattr(channel, "_Channel__writer", None)
        matched = getattr(writer_holder, "_Channel__Writer__publication_matched_count", None)
        writer = getattr(writer_holder, "_Channel__Writer__writer", None)
        return (
            f"publisher_id={id(publisher)} channel_id={id(channel) if channel is not None else None} "
            f"writer_holder_id={id(writer_holder) if writer_holder is not None else None} "
            f"data_writer_id={id(writer) if writer is not None else None} matched={matched}"
        )
    except Exception as exc:
        return f"publisher_status_error={exc}"


def _rate_hz(count, start_time):
    elapsed = max(time.time() - start_time, 1e-6)
    return count / elapsed


class RH5DG2_Right_Hand_JointIndex(IntEnum):
    kRightHandThumbYaw = 0
    kRightHandThumbMcp = 1
    kRightHandThumbDip = 2
    kRightHandIndexYaw = 3
    kRightHandIndexMcp = 4
    kRightHandIndexPip = 5
    kRightHandMiddleYaw = 6
    kRightHandMiddleMcp = 7
    kRightHandMiddlePip = 8
    kRightHandRingMcp = 9
    kRightHandRingPip = 10
    kRightHandPinkyMcp = 11
    kRightHandPinkyPip = 12


class RH5DG2_Left_Hand_JointIndex(IntEnum):
    kLeftHandThumbYaw = 13
    kLeftHandThumbMcp = 14
    kLeftHandThumbDip = 15
    kLeftHandIndexYaw = 16
    kLeftHandIndexMcp = 17
    kLeftHandIndexPip = 18
    kLeftHandMiddleYaw = 19
    kLeftHandMiddleMcp = 20
    kLeftHandMiddlePip = 21
    kLeftHandRingMcp = 22
    kLeftHandRingPip = 23
    kLeftHandPinkyMcp = 24
    kLeftHandPinkyPip = 25


# Old names kept so existing imports do not break.
Inspire_Num_Motors = RH5DG2_Num_Motors
Inspire_Right_Hand_JointIndex = RH5DG2_Right_Hand_JointIndex
Inspire_Left_Hand_JointIndex = RH5DG2_Left_Hand_JointIndex


def _is_hand_tracking_ready(left_hand_data, right_hand_data):
    return _hand_landmark_status(left_hand_data, right_hand_data)["hand_tracking_ready"]


def _is_legacy_sim_hand_tracking_ready(left_hand_data, right_hand_data):
    return (
        not np.allclose(right_hand_data, 0.0, atol=1e-5)
        and not np.allclose(left_hand_data[4], np.array([-1.13, 0.3, 0.15]), atol=1e-3)
    )


def _hand_landmark_status(left_hand_data, right_hand_data):
    left = np.asarray(left_hand_data, dtype=np.float64).reshape(-1, 3)
    right = np.asarray(right_hand_data, dtype=np.float64).reshape(-1, 3)
    left_finite = bool(np.isfinite(left).all())
    right_finite = bool(np.isfinite(right).all())
    left_norm = np.linalg.norm(left, axis=1) if left.size else np.array([])
    right_norm = np.linalg.norm(right, axis=1) if right.size else np.array([])
    left_allzero = bool(left_norm.size == 0 or np.allclose(left, 0.0, atol=1e-5))
    right_allzero = bool(right_norm.size == 0 or np.allclose(right, 0.0, atol=1e-5))
    left_sentinel = bool(np.allclose(left[4], np.array([-1.13, 0.3, 0.15]), atol=1e-3)) if left.shape[0] > 4 else True
    left_valid_points = int(np.sum(left_norm > 1e-5))
    right_valid_points = int(np.sum(right_norm > 1e-5))
    min_valid_points = int(os.getenv("RH5DG2_MIN_VALID_HAND_POINTS", "8"))
    return {
        "hand_tracking_ready": bool(
            left_finite
            and right_finite
            and not left_allzero
            and not right_allzero
            and left_valid_points >= min_valid_points
            and right_valid_points >= min_valid_points
        ),
        "left_finite": left_finite,
        "right_finite": right_finite,
        "left_allzero": left_allzero,
        "right_allzero": right_allzero,
        "left_valid_points": left_valid_points,
        "right_valid_points": right_valid_points,
        "min_valid_points": min_valid_points,
        "left_sentinel": left_sentinel,
    }


def _normalize_to_unit_interval(values, joint_limits, return_unclamped=False):
    normalized = np.empty(len(values), dtype=np.float64)
    unclamped = np.empty(len(values), dtype=np.float64)
    for idx, (value, (lower, upper)) in enumerate(zip(values, joint_limits)):
        if np.isclose(upper, lower):
            unclamped[idx] = 0.5
            normalized[idx] = 0.5
            continue

        # 수정됨: (value - lower) 대신 다시 (upper - value)를 사용하여 역방향으로 매핑
        unclamped[idx] = (upper - value) / (upper - lower)
        normalized[idx] = np.clip(unclamped[idx], 0.0, 1.0)

    if return_unclamped:
        return normalized, unclamped
    return normalized


def _clip_to_joint_limits(values, joint_limits):
    clipped = np.empty(len(values), dtype=np.float64)
    for idx, (value, (lower, upper)) in enumerate(zip(values, joint_limits)):
        clipped[idx] = np.clip(value, lower, upper)
    return clipped


def _coerce_rh5dg2_safe_baseline_values(values, label="baseline"):
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size != RH5DG2_Num_Motors:
        logger_mp.warning(
            "[RH5DG2 safe baseline] ignoring invalid %s length=%s expected=%s values=%s",
            label,
            arr.size,
            RH5DG2_Num_Motors,
            arr.tolist(),
        )
        return None
    if not np.isfinite(arr).all():
        logger_mp.warning("[RH5DG2 safe baseline] ignoring non-finite %s=%s", label, arr.tolist())
        return None
    return arr.copy()


def _coerce_rh5dg2_safe_baseline(values):
    if values is None:
        return None, None
    if isinstance(values, dict):
        left = _coerce_rh5dg2_safe_baseline_values(values.get("left"), "left baseline")
        right = _coerce_rh5dg2_safe_baseline_values(values.get("right"), "right baseline")
        return left, right
    common = _coerce_rh5dg2_safe_baseline_values(values, "common baseline")
    if common is None:
        return None, None
    return common.copy(), common.copy()


def _apply_safe_close_floor(values):
    normalized = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    if not _env_flag("RH5DG2_ENABLE_TELEOP_SAFE_CLOSE"):
        return normalized.copy()
    floor = RH5DG2_SAFE_CLOSE_FLOOR
    raw = os.getenv("RH5DG2_SAFE_CLOSE_FLOOR")
    if raw:
        parsed = np.array([float(item.strip()) for item in raw.split(",") if item.strip()], dtype=np.float64)
        if parsed.size == RH5DG2_Num_Motors:
            floor = np.clip(parsed, 0.0, 1.0)
        else:
            logger_mp.warning(
                "Ignoring RH5DG2_SAFE_CLOSE_FLOOR with %s values; expected %s",
                parsed.size,
                RH5DG2_Num_Motors,
            )
    return np.maximum(normalized, floor)


def _apply_teleop_close_gain(values):
    normalized = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    gain = RH5DG2_TELEOP_CLOSE_GAIN
    raw = os.getenv("RH5DG2_TELEOP_CLOSE_GAIN")
    if raw:
        try:
            gain = float(raw)
        except ValueError:
            logger_mp.warning("Ignoring invalid RH5DG2_TELEOP_CLOSE_GAIN=%r", raw)
            gain = RH5DG2_TELEOP_CLOSE_GAIN
    gain = max(0.0, gain)

    closure = 1.0 - normalized
    return np.clip(1.0 - closure * gain, 0.0, 1.0)


def _finger_open_scores(hand_data):
    data = np.asarray(hand_data, dtype=np.float64).reshape(25, 3)
    wrist = data[0]
    scores = {}
    for finger, cfg in RH5DG2_FINGER_OPEN_CALIBRATION.items():
        tip_dist = float(np.linalg.norm(data[cfg["tip"]] - wrist))
        span = max(cfg["open"] - cfg["close"], 1e-6)
        raw_score = np.clip((tip_dist - cfg["close"]) / span, 0.0, 1.0)
        gain = cfg["gain"]
        bias = cfg["bias"]
        raw_gain = os.getenv(f"RH5DG2_{finger.upper()}_OPEN_GAIN")
        raw_bias = os.getenv(f"RH5DG2_{finger.upper()}_OPEN_BIAS")
        if raw_gain is not None:
            try:
                gain = float(raw_gain)
            except ValueError:
                logger_mp.warning("Ignoring invalid RH5DG2_%s_OPEN_GAIN=%r", finger.upper(), raw_gain)
        if raw_bias is not None:
            try:
                bias = float(raw_bias)
            except ValueError:
                logger_mp.warning("Ignoring invalid RH5DG2_%s_OPEN_BIAS=%r", finger.upper(), raw_bias)
        scores[finger] = {
            "distance": tip_dist,
            "raw": raw_score,
            "calibrated": np.clip(raw_score * gain + bias, 0.0, 1.0),
        }
    return scores


def _apply_finger_open_calibration(values, hand_data):
    calibrated = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0).copy()
    scores = _finger_open_scores(hand_data)
    if not _env_flag("RH5DG2_ENABLE_OPEN_CALIBRATION"):
        return calibrated, scores
    for finger, cfg in RH5DG2_FINGER_OPEN_CALIBRATION.items():
        open_value = scores[finger]["calibrated"]
        for idx in cfg["indices"]:
            calibrated[idx] = max(calibrated[idx], open_value)
    return calibrated, scores


def _human_finger_shape_scores(hand_data):
    data = np.asarray(hand_data, dtype=np.float64).reshape(25, 3)
    open_scores = _finger_open_scores(data)
    scores = {}
    for finger, landmarks in _RH5DG2_FINGER_LANDMARKS.items():
        points = data[list(landmarks)]
        segments = np.diff(points, axis=0)
        path_length = float(np.sum(np.linalg.norm(segments, axis=1)))
        chord_length = float(np.linalg.norm(points[-1] - points[0]))
        straightness = chord_length / max(path_length, 1e-6)
        straight_score = float(np.clip((straightness - 0.82) / 0.15, 0.0, 1.0))
        open_score = float(open_scores[finger]["raw"])
        extension_score = float(np.clip(straight_score * open_score, 0.0, 1.0))
        scores[finger] = {
            "straightness": straightness,
            "straight_score": straight_score,
            "open_score": open_score,
            "extension_score": extension_score,
            "path_length": path_length,
            "chord_length": chord_length,
        }
    return scores


def _human_finger_curl_scores(hand_data):
    data = np.asarray(hand_data, dtype=np.float64).reshape(25, 3)
    scores = {}
    for finger, landmarks in _RH5DG2_FINGER_LANDMARKS.items():
        points = data[list(landmarks)]
        segments = np.diff(points, axis=0)
        path_length = float(np.sum(np.linalg.norm(segments, axis=1)))
        chord_length = float(np.linalg.norm(points[-1] - points[0]))
        straightness = chord_length / max(path_length, 1e-6)

        # 0.97+ is visually straight/open, 0.72 or below is a clear curl.
        curl_from_shape = 1.0 - np.clip((straightness - 0.72) / 0.25, 0.0, 1.0)
        curl_from_tip = 1.0 - _finger_open_scores(data)[finger]["raw"]
        curl_score = float(np.clip(max(curl_from_shape, curl_from_tip), 0.0, 1.0))
        scores[finger] = {
            "curl": curl_score,
            "curl_shape": float(curl_from_shape),
            "curl_tip": float(curl_from_tip),
            "straightness": straightness,
            "path_length": path_length,
            "chord_length": chord_length,
        }
    return scores


def _apply_human_curl_command(values, hand_data, curl_scale=1.0, finger_curl_scales=None):
    normalized = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0).copy()
    scores = _human_finger_curl_scores(hand_data)
    strength = float(os.getenv("RH5DG2_CURL_COMMAND_STRENGTH", "1.0"))
    threshold = float(os.getenv("RH5DG2_CURL_COMMAND_THRESHOLD", "0.10"))
    decouple_enabled = _env_flag("RH5DG2_DECOUPLE_FINGERS", True)
    decouple_strength = float(os.getenv("RH5DG2_DECOUPLE_STRENGTH", "0.75"))
    decouple_threshold = float(os.getenv("RH5DG2_DECOUPLE_THRESHOLD", "0.12"))
    curl_scale = float(np.clip(curl_scale, 0.0, 10.0))
    finger_curl_scales = finger_curl_scales or {}
    strength = float(np.clip(strength, 0.0, 1.0))
    threshold = float(np.clip(threshold, 0.0, 0.95))
    decouple_strength = float(np.clip(decouple_strength, 0.0, 1.0))
    decouple_threshold = float(np.clip(decouple_threshold, 0.0, 1.0))

    before = normalized.copy()
    curl_targets = np.ones_like(normalized)
    for finger, joint_indices in _RH5DG2_FINGER_FLEX_JOINTS.items():
        raw_curl = float(scores[finger]["curl"])
        finger_scale = float(np.clip(finger_curl_scales.get(finger, 1.0), 0.0, 10.0))
        curl = float(np.clip(raw_curl * curl_scale * finger_scale, 0.0, 1.0))
        scores[finger]["raw_curl"] = raw_curl
        scores[finger]["curl"] = curl
        scores[finger]["curl_scale"] = curl_scale
        scores[finger]["finger_scale"] = finger_scale
        command_curl = np.clip((curl - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0) * strength
        # Normalized convention: 1.0=open, 0.0=closed.
        target_open_value = float(np.clip(1.0 - command_curl, 0.0, 1.0))
        for idx in joint_indices:
            curl_targets[idx] = min(curl_targets[idx], target_open_value)
            normalized[idx] = min(normalized[idx], target_open_value)

    before_decouple = normalized.copy()
    if decouple_enabled and decouple_strength > 0.0:
        for primary, neighbor in (("index", "middle"), ("middle", "index"), ("ring", "little"), ("little", "ring")):
            if scores[primary]["curl"] - scores[neighbor]["curl"] < decouple_threshold:
                continue
            for idx in _RH5DG2_FINGER_FLEX_JOINTS[neighbor]:
                desired_open = curl_targets[idx]
                normalized[idx] = max(normalized[idx], normalized[idx] * (1.0 - decouple_strength) + desired_open * decouple_strength)

    return normalized, {
        "scores": scores,
        "targets": curl_targets,
        "delta": normalized - before,
        "decouple_delta": normalized - before_decouple,
        "decouple_enabled": decouple_enabled,
        "decouple_strength": decouple_strength,
        "decouple_threshold": decouple_threshold,
        "curl_scale": curl_scale,
        "finger_curl_scales": dict(finger_curl_scales),
        "strength": strength,
        "threshold": threshold,
    }


def _fmt_shape_scores(scores):
    return [
        (
            f"{finger}:straight={data['straightness']:.4f},"
            f"straight_score={data['straight_score']:.4f},"
            f"open_score={data['open_score']:.4f},"
            f"extension={data['extension_score']:.4f}"
        )
        for finger, data in scores.items()
    ]


def _fmt_curl_scores(scores):
    return [
        (
            f"{_RH5DG2_FINGER_DEBUG_NAMES.get(finger, finger)}:"
            f"curl={data['curl']:.4f},raw={data.get('raw_curl', data['curl']):.4f},"
            f"scale={data.get('curl_scale', 1.0):.2f}x{data.get('finger_scale', 1.0):.2f},"
            f"shape={data['curl_shape']:.4f},"
            f"tip={data['curl_tip']:.4f},straight={data['straightness']:.4f}"
        )
        for finger, data in scores.items()
    ]


def _finger_command_delta(debug):
    delta = np.asarray(debug["curl_debug"]["delta"], dtype=np.float64)
    result = {}
    for finger in ("index", "middle", "ring", "little"):
        name = _RH5DG2_FINGER_DEBUG_NAMES.get(finger, finger)
        indices = _RH5DG2_FINGER_FLEX_JOINTS[finger]
        result[f"{name}_delta"] = np.round(delta[list(indices)], 4).tolist()
    return result


def _v_pose_spread_score(hand_data):
    data = np.asarray(hand_data, dtype=np.float64).reshape(25, 3)
    distance = float(np.linalg.norm(data[9] - data[14]))
    score = float(np.clip((distance - 0.025) / 0.055, 0.0, 1.0))
    return {"distance": distance, "score": score}


def _finger_landmark_debug(hand_data, finger):
    data = np.asarray(hand_data, dtype=np.float64).reshape(25, 3)
    landmarks = _RH5DG2_FINGER_LANDMARKS[finger]
    points = data[list(landmarks)]
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    return {
        "indices": list(landmarks),
        "points": np.round(points, 4).tolist(),
        "segments": np.round(segments, 4).tolist(),
        "lengths": np.round(lengths, 4).tolist(),
    }


def _apply_open_hand_prior(values, hand_data):
    normalized = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0).copy()
    scores = _human_finger_shape_scores(hand_data)
    strength = float(os.getenv("RH5DG2_OPEN_PRIOR_STRENGTH", "1.0"))
    threshold = float(os.getenv("RH5DG2_OPEN_PRIOR_THRESHOLD", "0.55"))
    strength = np.clip(strength, 0.0, 1.0)
    threshold = np.clip(threshold, 0.0, 1.0)

    before = normalized.copy()
    prior_targets = np.zeros_like(normalized)
    for finger, joint_indices in _RH5DG2_FINGER_FLEX_JOINTS.items():
        extension = scores[finger]["extension_score"]
        prior = np.clip((extension - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
        prior *= strength
        for idx in joint_indices:
            prior_targets[idx] = max(prior_targets[idx], prior)
            normalized[idx] = max(normalized[idx], prior)

    return normalized, {
        "scores": scores,
        "prior_targets": prior_targets,
        "delta": normalized - before,
        "v_spread": _v_pose_spread_score(hand_data),
        "strength": strength,
        "threshold": threshold,
    }


def _env_csv_set(name, default):
    raw = os.getenv(name)
    if raw is None:
        return set(default)
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def _apply_open_recovery(values, hand_data, side):
    normalized = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0).copy()
    if not _env_flag("RH5DG2_ENABLE_OPEN_RECOVERY", True):
        return normalized, {}

    enabled_hands = _env_csv_set("RH5DG2_OPEN_RECOVERY_HANDS", ("right",))
    if side.lower() not in enabled_hands and "both" not in enabled_hands:
        return normalized, {}

    enabled_fingers = _env_csv_set(
        "RH5DG2_OPEN_RECOVERY_FINGERS",
        ("index", "middle", "little"),
    )
    threshold = float(np.clip(float(os.getenv("RH5DG2_OPEN_RECOVERY_THRESHOLD", "0.62")), 0.0, 0.98))
    strength = float(np.clip(float(os.getenv("RH5DG2_OPEN_RECOVERY_STRENGTH", "1.0")), 0.0, 1.0))
    open_value = float(np.clip(float(os.getenv("RH5DG2_OPEN_RECOVERY_VALUE", "1.0")), 0.0, 1.0))
    hard_set = _env_flag("RH5DG2_OPEN_RECOVERY_HARD_SET", True)
    scores = _human_finger_shape_scores(hand_data)
    before = normalized.copy()

    activations = {}
    for finger in enabled_fingers:
        if finger not in _RH5DG2_FINGER_FLEX_JOINTS:
            continue
        extension = float(scores[finger]["extension_score"])
        activation = float(np.clip((extension - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0))
        activation *= strength
        activations[finger] = activation
        if activation <= 0.0:
            continue
        for idx in _RH5DG2_FINGER_FLEX_JOINTS[finger]:
            if hard_set:
                normalized[idx] = max(normalized[idx], open_value)
            else:
                normalized[idx] = max(
                    normalized[idx],
                    normalized[idx] * (1.0 - activation) + open_value * activation,
                )

    if not any(value > 0.0 for value in activations.values()):
        return normalized, {}

    return normalized, {
        "side": side,
        "scores": scores,
        "activations": activations,
        "delta": normalized - before,
        "threshold": threshold,
        "strength": strength,
        "open_value": open_value,
        "hard_set": hard_set,
    }


def _fmt_finger_scores(scores):
    return [
        (
            f"{finger}:dist={data['distance']:.4f},"
            f"raw={data['raw']:.4f},cal={data['calibrated']:.4f}"
        )
        for finger, data in scores.items()
    ]


def _vector_weight_summary(side, retargeting):
    optimizer = retargeting.optimizer
    weights = np.asarray(getattr(optimizer, "target_link_weights", []), dtype=np.float64)
    origins = list(getattr(optimizer, "origin_link_names", []))
    tasks = list(getattr(optimizer, "task_link_names", []))
    rows = []
    for idx, (origin, task, weight) in enumerate(zip(origins, tasks, weights)):
        finger = "other"
        for name in ("thumb", "index", "middle", "ring", "pinky"):
            if name in origin or name in task:
                finger = name
                break
        rows.append(f"{side}:{idx}:{finger}:{origin}->{task}:w={weight:.2f}")
    return rows


def _prepare_position_reference(hand_data, indices):
    data = np.asarray(hand_data, dtype=np.float64)
    idx = np.asarray(indices, dtype=np.int64)
    wrist = data[0].copy()
    palm_center = np.mean(data[_RH5DG2_PALM_INDICES], axis=0)
    selected = data[idx]
    wrist_relative = selected - wrist
    palm_relative = selected - palm_center
    rh5dg2_reference = wrist_relative @ _RH5DG2_POSITION_AXES.T
    return rh5dg2_reference, wrist, palm_center, selected, wrist_relative, palm_relative


def _prepare_vector_reference(hand_data, indices):
    data = np.asarray(hand_data, dtype=np.float64)
    idx = np.asarray(indices, dtype=np.int64)
    source = data[idx[0, :]]
    target = data[idx[1, :]]
    input_vectors = target - source
    rh5dg2_vectors = input_vectors @ _RH5DG2_POSITION_AXES.T
    return rh5dg2_vectors, source, target, input_vectors


def _postprocess_sim_command(raw_q, joint_limits, joint_names, hand_data, curl_scale=1.0, finger_curl_scales=None, side="right"):
    clamped, unclamped = _normalize_to_unit_interval(raw_q, joint_limits, return_unclamped=True)
    gained = _apply_teleop_close_gain(clamped)
    prior, prior_debug = _apply_open_hand_prior(gained, hand_data)
    calibrated, finger_scores = _apply_finger_open_calibration(prior, hand_data)
    curl_target, curl_debug = _apply_human_curl_command(
        calibrated,
        hand_data,
        curl_scale=curl_scale,
        finger_curl_scales=finger_curl_scales,
    )
    safe_floor = _apply_safe_close_floor(curl_target)
    target, open_recovery_debug = _apply_open_recovery(safe_floor, hand_data, side)
    debug = {
        "unclamped": unclamped,
        "clamped": clamped,
        "gain": gained,
        "prior": prior,
        "prior_debug": prior_debug,
        "calibrated": calibrated,
        "curl": curl_target,
        "curl_debug": curl_debug,
        "safe_floor": safe_floor,
        "open_recovery_debug": open_recovery_debug,
        "finger_command_delta": _finger_command_delta({"curl_debug": curl_debug}),
        "finger_scores": finger_scores,
        "target": target,
        "denorm_rad": _denormalize_from_unit_interval(target, joint_limits),
        "saturation": _saturation_summary(raw_q, target, joint_limits),
        "mapping": _fmt_named_mapping(joint_names, joint_limits),
    }
    return target, debug


def _set_cmd_active(cmd, active):
    active = bool(active)
    cmd.mode = 0b0001 if active else 0
    reserve_value = [1, 0, 0] if active else [0, 0, 0]
    try:
        cmd.reserve = reserve_value
    except Exception:
        try:
            cmd.reserve = tuple(reserve_value)
        except Exception:
            try:
                cmd.reserve[0] = reserve_value[0]
                cmd.reserve[1] = reserve_value[1]
                cmd.reserve[2] = reserve_value[2]
            except Exception:
                pass
    try:
        if list(cmd.reserve)[:3] != reserve_value:
            cmd.reserve[0] = reserve_value[0]
            cmd.reserve[1] = reserve_value[1]
            cmd.reserve[2] = reserve_value[2]
    except Exception:
        pass


def _cmd_is_active(cmd):
    try:
        reserve0 = int(list(getattr(cmd, "reserve", (0, 0, 0)))[0])
    except Exception:
        reserve0 = 0
    return int(getattr(cmd, "mode", 0)) == 1 and reserve0 == 1


def _cmd_active_indices(cmds, offset=0, count=RH5DG2_Num_Motors):
    active = []
    for local_idx in range(count):
        if _cmd_is_active(cmds[offset + local_idx]):
            active.append(local_idx)
    return active


def _fmt_safe_enabled(indices, values):
    arr = np.asarray(values, dtype=np.float64)
    return {int(idx): float(np.round(arr[int(idx)], 4)) for idx in sorted(indices)}


def _thumb_raw_values(raw_q, joint_names, side):
    raw = np.asarray(raw_q, dtype=np.float64).reshape(-1)
    values = {}
    for short_name in ("yaw", "mcp", "dip"):
        joint_name = f"{side}_thumb_{short_name}_joint"
        try:
            idx = joint_names.index(joint_name)
            values[joint_name] = float(raw[idx])
        except ValueError:
            values[joint_name] = 0.0
    return values


def _shape_grasp(values):
    # Bias mid-range values toward closure so the robot grabs earlier.
    return np.clip(np.power(np.asarray(values, dtype=np.float64), RH5DG2_GRASP_SHARPNESS), 0.0, 1.0)


class _RH5DG2Retargeting:
    def __init__(self, fast_mode=False, retarget_mode=None):
        RetargetingConfig.set_default_urdf_dir(str(_ASSETS_ROOT))
        self.fast_mode = bool(fast_mode or _env_flag("RH5DG2_FAST_MODE", False))
        self.retarget_mode = _normalize_rh5dg2_retarget_mode(retarget_mode)
        cfg = self._load_config()
        if self.retarget_mode is not None:
            self._apply_retarget_mode(cfg)
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
        self.left_retargeting_type = self.left_retargeting.optimizer.retargeting_type.lower()
        self.right_retargeting_type = self.right_retargeting.optimizer.retargeting_type.lower()
        logger_mp.info(
            "[RH5DG2 retargeting] left=%s right=%s override=%s fast_mode=%s left_indices_shape=%s right_indices_shape=%s",
            self.left_retargeting_type,
            self.right_retargeting_type,
            self.retarget_mode or "config",
            self.fast_mode,
            np.asarray(self.left_indices).shape,
            np.asarray(self.right_indices).shape,
        )

        self.left_retargeting_to_hardware = [
            self.left_retargeting.joint_names.index(name) for name in self.left_joint_names
        ]
        self.right_retargeting_to_hardware = [
            self.right_retargeting.joint_names.index(name) for name in self.right_joint_names
        ]

        self.left_joint_limits = self._load_joint_limits(
            Path(cfg["left"]["urdf_path"]), self.left_joint_names
        )
        self.right_joint_limits = self._load_joint_limits(
            Path(cfg["right"]["urdf_path"]), self.right_joint_names
        )
        self._set_open_initial_qpos(self.left_retargeting, self.left_joint_names, self.left_joint_limits)
        self._set_open_initial_qpos(self.right_retargeting, self.right_joint_names, self.right_joint_limits)

    def _apply_fast_config(self, cfg):
        normal_delta = float(os.getenv("RH5DG2_FAST_NORMAL_DELTA", "0.0015"))
        low_pass_alpha = float(os.getenv("RH5DG2_FAST_LOW_PASS_ALPHA", "1.0"))
        for side in ("left", "right"):
            cfg[side]["normal_delta"] = normal_delta
            cfg[side]["low_pass_alpha"] = low_pass_alpha

    def _apply_retarget_mode(self, cfg):
        for side in ("left", "right"):
            cfg[side]["type"] = self.retarget_mode

    def _configure_fast_optimizer(self, retargeting):
        opt = retargeting.optimizer.opt
        maxeval = int(os.getenv("RH5DG2_FAST_MAXEVAL", "14"))
        ftol_abs = float(os.getenv("RH5DG2_FAST_FTOL_ABS", "1e-4"))
        xtol_abs = float(os.getenv("RH5DG2_FAST_XTOL_ABS", "1e-4"))
        maxtime = float(os.getenv("RH5DG2_FAST_MAXTIME", "0.006"))
        if maxeval > 0 and hasattr(opt, "set_maxeval"):
            opt.set_maxeval(maxeval)
        if ftol_abs > 0.0 and hasattr(opt, "set_ftol_abs"):
            opt.set_ftol_abs(ftol_abs)
        if xtol_abs > 0.0 and hasattr(opt, "set_xtol_abs"):
            opt.set_xtol_abs(xtol_abs)
        if maxtime > 0.0 and hasattr(opt, "set_maxtime"):
            opt.set_maxtime(maxtime)
        if retargeting.filter is not None and float(os.getenv("RH5DG2_FAST_LOW_PASS_ALPHA", "1.0")) >= 1.0:
            retargeting.filter = None
        logger_mp.info(
            "[RH5DG2 fast optimizer] maxeval=%s ftol_abs=%s xtol_abs=%s maxtime=%s normal_delta=%s filter=%s",
            maxeval,
            ftol_abs,
            xtol_abs,
            maxtime,
            getattr(retargeting.optimizer, "norm_delta", None),
            retargeting.filter is not None,
        )

    def _set_open_initial_qpos(self, retargeting, joint_names, joint_limits):
        robot_qpos = np.zeros(retargeting.optimizer.robot.dof, dtype=np.float32)
        name_to_lower = {
            name: float(limit[0])
            for name, limit in zip(joint_names, joint_limits)
        }
        for idx, joint_name in zip(retargeting.optimizer.idx_pin2target, retargeting.optimizer.target_joint_names):
            robot_qpos[idx] = name_to_lower.get(joint_name, 0.0)
        retargeting.set_qpos(robot_qpos)
        if retargeting.filter is not None:
            retargeting.filter.reset()

    def _load_config(self):
        with _RH5DG2_CONFIG_PATH.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        fixed_cfg = {}
        for side in ("left", "right"):
            side_cfg = dict(cfg[side])
            side_cfg["target_joint_names"] = [
                joint_name.replace("_pitch_joint", "_yaw_joint")
                for joint_name in side_cfg["target_joint_names"]
            ]
            side_cfg["urdf_path"] = str(self._resolve_urdf_path(side_cfg["urdf_path"]))
            fixed_cfg[side] = side_cfg
        return fixed_cfg

    def _resolve_urdf_path(self, urdf_path_str):
        urdf_path = Path(urdf_path_str)
        if (_ASSETS_ROOT / urdf_path).exists():
            source_path = _ASSETS_ROOT / urdf_path
        else:
            # The right-hand URDF file is still named RH56DG2_R.urdf in the repo.
            fallback_name = urdf_path.name.replace("RH5DG2_R", "RH56DG2_R")
            fallback_path = urdf_path.with_name(fallback_name)
            source_path = _ASSETS_ROOT / fallback_path
            if source_path.exists():
                logger_mp.warning(
                    "[RH5DG2] Using fallback URDF path %s for missing %s",
                    fallback_path,
                    urdf_path,
                )
                urdf_path = fallback_path
            else:
                raise FileNotFoundError(f"RH5DG2 URDF not found: {urdf_path}")

        urdf_text = source_path.read_text(encoding="utf-8")
        mesh_uri = (_RH5DG2_ASSET_DIR / "meshes").resolve().as_uri() + "/"
        rewritten_text = urdf_text
        for package_prefix in (
            "package://RH5DG2_R/meshes/",
            "package://RH5DG2_L/meshes/",
            "package://RH5DG2/meshes/",
        ):
            rewritten_text = rewritten_text.replace(package_prefix, mesh_uri)

        if rewritten_text == urdf_text:
            return urdf_path

        _RH5DG2_URDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        rewritten_path = _RH5DG2_URDF_CACHE_DIR / urdf_path.name
        rewritten_path.write_text(rewritten_text, encoding="utf-8")
        logger_mp.warning(
            "[RH5DG2] Rewrote URDF mesh package paths for %s -> %s mesh_uri=%s",
            source_path,
            rewritten_path,
            mesh_uri,
        )
        return rewritten_path

    def _load_joint_limits(self, urdf_path, joint_names):
        xml_path = Path(urdf_path)
        if not xml_path.is_absolute():
            xml_path = _ASSETS_ROOT / xml_path
        xml_root = ET.parse(xml_path).getroot()
        joint_limits = {}
        for joint in xml_root.findall("joint"):
            joint_name = joint.get("name")
            limit = joint.find("limit")
            if joint_name is None or limit is None:
                continue
            lower = float(limit.get("lower", "0.0"))
            upper = float(limit.get("upper", "0.0"))
            joint_limits[joint_name] = (lower, upper)

        missing = [joint_name for joint_name in joint_names if joint_name not in joint_limits]
        if missing:
            raise ValueError(f"Missing RH5DG2 joint limits for: {missing}")

        return [joint_limits[joint_name] for joint_name in joint_names]


class RH5DG2_Controller_DFX:
    def __init__(
        self,
        left_hand_array,
        right_hand_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
        fps=100.0,
        Unit_Test=False,
        simulation_mode=False,
        network_interface=None,
        input_timestamp_value=None,
        log_throttle_s=1.0,
        fast_mode=False,
        safe_mode=False,
        enabled_indices=None,
        pitch_only=False,
        safe_gain=0.2,
        raw_close_direction=1.0,
        safe_active_hand=RH5DG2_SAFE_ACTIVE_HAND_DEFAULT,
        safe_baseline=None,
        restore_repeat=80,
        restore_interval_s=0.1,
        restore_settle_s=0.75,
        curl_scale=1.0,
        index_curl_scale=1.0,
        enable_thumb=False,
        thumb_source="raw",
        thumb10_scale=0.3,
        thumb11_scale=1.0,
        thumb12_scale=1.0,
        retarget_mode=None,
    ):
        logger_mp.info("Initialize RH5DG2_Controller_DFX...")

        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode
        self.network_interface = network_interface
        self.input_timestamp_value = input_timestamp_value
        self.log_throttle_s = max(float(log_throttle_s), 0.0)
        self._hand_input_ready_for_debug = True
        self.fast_mode = bool(fast_mode)
        self.safe_mode = bool(safe_mode)
        self.safe_gain = float(np.clip(safe_gain, 0.0, 1.0))
        self.raw_close_direction = 1.0 if float(raw_close_direction) >= 0.0 else -1.0
        self.safe_active_hand = str(safe_active_hand or RH5DG2_SAFE_ACTIVE_HAND_DEFAULT).strip().lower()
        if self.safe_active_hand not in ("right", "left", "both"):
            logger_mp.warning("[RH5DG2 safe mode] invalid safe_active_hand=%r; using right", safe_active_hand)
            self.safe_active_hand = RH5DG2_SAFE_ACTIVE_HAND_DEFAULT
        self.safe_baseline_left, self.safe_baseline_right = _coerce_rh5dg2_safe_baseline(safe_baseline)
        self.restore_repeat = max(int(restore_repeat), 1)
        self.restore_interval_s = max(float(restore_interval_s), 0.0)
        self.restore_settle_s = max(float(restore_settle_s), 0.0)
        self.curl_scale = float(np.clip(curl_scale, 0.0, 10.0))
        self.finger_curl_scales = {
            "index": float(np.clip(index_curl_scale, 0.0, 10.0)),
        }
        self.enable_thumb = bool(enable_thumb)
        self.thumb_source = str(thumb_source or "raw").strip().lower()
        if self.thumb_source not in ("raw", "curl"):
            logger_mp.warning("[RH5DG2 safe mode] invalid thumb_source=%r; using raw", thumb_source)
            self.thumb_source = "raw"
        self.thumb_scales = {
            10: float(np.clip(thumb10_scale, 0.0, 10.0)),
            11: float(np.clip(thumb11_scale, 0.0, 10.0)),
            12: float(np.clip(thumb12_scale, 0.0, 10.0)),
        }
        self.pitch_only = bool(pitch_only)
        if self.pitch_only:
            enabled_indices = RH5DG2_RAW_PITCH_INDICES
        if self.enable_thumb:
            if enabled_indices is None:
                enabled_indices = RH5DG2_RAW_PITCH_INDICES
            enabled_indices = sorted(set(enabled_indices) | set(RH5DG2_RAW_THUMB_INDICES))
        if enabled_indices is None:
            enabled_indices = RH5DG2_RAW_PITCH_INDICES if self.safe_mode else range(RH5DG2_Num_Motors)
        self.safe_enabled_indices = {
            int(idx)
            for idx in enabled_indices
            if 0 <= int(idx) < RH5DG2_Num_Motors
        }
        if self.safe_mode:
            self.safe_enabled_indices &= set(RH5DG2_SAFE_DELTA_LIMIT)
        self.safe_disabled_indices = set(range(RH5DG2_Num_Motors)) - self.safe_enabled_indices
        self._safe_baseline_left = None if self.safe_baseline_left is None else self.safe_baseline_left.copy()
        self._safe_baseline_right = None if self.safe_baseline_right is None else self.safe_baseline_right.copy()
        self._init_pose_left = None
        self._init_pose_right = None
        self._hand_control_thread = None
        self._last_left_active_mask = np.ones(RH5DG2_Num_Motors, dtype=bool)
        self._last_right_active_mask = np.ones(RH5DG2_Num_Motors, dtype=bool)
        self._last_left_command = None
        self._last_right_command = None
        self._last_debug_ts = 0.0
        self._loop_rate_start_ts = time.time()
        self._publish_rate_start_ts = time.time()
        self.dds_domain_id = 1 if simulation_mode else 0
        self.hand_retargeting = _RH5DG2Retargeting(fast_mode=self.fast_mode, retarget_mode=retarget_mode)
        self.retarget_mode = self.hand_retargeting.retarget_mode
        self.left_state_ready = False
        self.right_state_ready = False

        if self.safe_mode:
            logger_mp.warning("RH5DG2 SAFE MODE")
            logger_mp.warning("enabled actuators: %s", " ".join(str(i) for i in sorted(self.safe_enabled_indices)))
            logger_mp.warning("disabled actuators: %s", " ".join(str(i) for i in sorted(self.safe_disabled_indices)))
            logger_mp.warning("gain: %.3f", self.safe_gain)
            logger_mp.warning("safe_delta_limit: %s", RH5DG2_SAFE_DELTA_LIMIT)
            logger_mp.warning("safe_abs_min: %s", np.round(RH5DG2_SAFE_ABS_MIN, 4).tolist())
            logger_mp.warning("safe_abs_max: %s", np.round(RH5DG2_SAFE_ABS_MAX, 4).tolist())
            logger_mp.warning("raw_close_direction: %.1f", self.raw_close_direction)
            logger_mp.warning("safe_active_hand: %s", self.safe_active_hand)
            logger_mp.warning(
                "enable_thumb: %s thumb_source: %s thumb_scales: %s",
                self.enable_thumb,
                self.thumb_source,
                self.thumb_scales,
            )
            logger_mp.warning(
                "curl_scale: %.3f index_curl_scale: %.3f",
                self.curl_scale,
                self.finger_curl_scales["index"],
            )
            logger_mp.warning(
                "restore: repeat=%s interval_s=%.3f settle_s=%.3f",
                self.restore_repeat,
                self.restore_interval_s,
                self.restore_settle_s,
            )
            logger_mp.warning(
                "left_safe_baseline: %s",
                "startup current_state" if self.safe_baseline_left is None else np.round(self.safe_baseline_left, 4).tolist(),
            )
            logger_mp.warning(
                "right_safe_baseline: %s",
                "startup current_state" if self.safe_baseline_right is None else np.round(self.safe_baseline_right, 4).tolist(),
            )

        # teleop_hand_and_arm initializes DDS first. This is idempotent and makes
        # direct/controller-level use match test_keyboard_rh5dg2.py.
        ChannelFactoryInitialize(self.dds_domain_id, networkInterface=self.network_interface)

        self.HandCmd_publisher = ChannelPublisher(kTopicRH5DG2DFXCommand, MotorCmds_)
        self.HandCmd_publisher.Init()

        self.HandState_subscriber = ChannelSubscriber(kTopicRH5DG2DFXState, MotorStates_)
        self.HandState_subscriber.Init()

        self.left_hand_state_array = Array("d", RH5DG2_Num_Motors, lock=True)
        self.right_hand_state_array = Array("d", RH5DG2_Num_Motors, lock=True)

        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state, daemon=True)
        self.subscribe_state_thread.start()

        wait_count = 0
        while not (self.left_state_ready and self.right_state_ready):
            if wait_count % 100 == 0:
                logger_mp.info("[RH5DG2_Controller_DFX] Waiting to subscribe DDS hand states...")
            time.sleep(0.01)
            wait_count += 1
            if wait_count > 500:
                logger_mp.warning("[RH5DG2_Controller_DFX] Timeout waiting for initial hand states. Proceeding anyway.")
                break
        logger_mp.info("[RH5DG2_Controller_DFX] Initial hand states received or timeout.")
        self._capture_initial_pose()

        hand_control_process = threading.Thread(
            target=self.control_process,
            args=(
                left_hand_array,
                right_hand_array,
                self.left_hand_state_array,
                self.right_hand_state_array,
                dual_hand_data_lock,
                dual_hand_state_array,
                dual_hand_action_array,
                self.input_timestamp_value,
            ),
            daemon=True,
        )
        self._hand_control_thread = hand_control_process
        self._hand_control_thread.start()

        logger_mp.info("Initialize RH5DG2_Controller_DFX OK!")

    def _debug_print(self, *args, **kwargs):
        if self.simulation_mode:
            print(*args, **kwargs)
            return True
        try:
            print(*args, **kwargs)
            return True
        except BlockingIOError:
            self._blocked_print_count = getattr(self, "_blocked_print_count", 0) + 1
            return False

    def _subscribe_hand_state(self):
        while True:
            hand_msg = self.HandState_subscriber.Read()
            if hand_msg is not None:
                with self.left_hand_state_array.get_lock():
                    for idx, joint_id in enumerate(RH5DG2_Left_Hand_JointIndex):
                        self.left_hand_state_array[idx] = hand_msg.states[joint_id].q
                with self.right_hand_state_array.get_lock():
                    for idx, joint_id in enumerate(RH5DG2_Right_Hand_JointIndex):
                        self.right_hand_state_array[idx] = hand_msg.states[joint_id].q
                self.left_state_ready = True
                self.right_state_ready = True
            time.sleep(0.002)

    def _capture_initial_pose(self):
        with self.left_hand_state_array.get_lock():
            left = np.array(self.left_hand_state_array[:], dtype=np.float64)
        with self.right_hand_state_array.get_lock():
            right = np.array(self.right_hand_state_array[:], dtype=np.float64)
        if self.safe_mode and self.safe_baseline_left is not None and self.safe_baseline_right is not None:
            self._init_pose_left = self.safe_baseline_left.copy()
            self._init_pose_right = self.safe_baseline_right.copy()
            logger_mp.warning(
                "[RH5DG2 init pose captured] using safe baseline for restore left=%s right=%s",
                np.round(self._init_pose_left, 4).tolist(),
                np.round(self._init_pose_right, 4).tolist(),
            )
            return
        if left.size == RH5DG2_Num_Motors and right.size == RH5DG2_Num_Motors and np.isfinite(left).all() and np.isfinite(right).all():
            self._init_pose_left = left.copy()
            self._init_pose_right = right.copy()
            logger_mp.warning(
                "[RH5DG2 init pose captured] left=%s right=%s",
                np.round(self._init_pose_left, 4).tolist(),
                np.round(self._init_pose_right, 4).tolist(),
            )
        else:
            logger_mp.warning("[RH5DG2 init pose captured] unavailable; left=%s right=%s", left.tolist(), right.tolist())

    def _safe_active_masks_for_init(self):
        left_mask = np.ones(RH5DG2_Num_Motors, dtype=bool)
        right_mask = np.ones(RH5DG2_Num_Motors, dtype=bool)
        if self.safe_mode:
            left_mask[:] = False
            right_mask[:] = False
            if self.safe_active_hand in ("left", "both"):
                left_mask[list(self.safe_enabled_indices)] = True
            if self.safe_active_hand in ("right", "both"):
                right_mask[list(self.safe_enabled_indices)] = True
        return left_mask, right_mask

    def _safe_enabled_indices_for_side(self, side):
        if not self.safe_mode:
            return list(range(RH5DG2_Num_Motors))
        if side == "left" and self.safe_active_hand in ("left", "both"):
            return _fmt_index_list(self.safe_enabled_indices)
        if side == "right" and self.safe_active_hand in ("right", "both"):
            return _fmt_index_list(self.safe_enabled_indices)
        return []

    def _valid_restore_pose(self, pose, active_mask):
        if pose is None:
            return False
        try:
            arr = np.asarray(pose, dtype=np.float64).reshape(-1)
        except Exception:
            return False
        if arr.size != RH5DG2_Num_Motors or not np.isfinite(arr).all():
            return False
        mask = np.asarray(active_mask, dtype=bool).reshape(RH5DG2_Num_Motors)
        if not np.any(mask):
            return True
        return not np.allclose(arr[mask], 0.0, atol=1e-6)

    def _restore_start_pose(self, side, active_mask):
        last_attr = "_last_left_command" if side == "left" else "_last_right_command"
        state_array = self.left_hand_state_array if side == "left" else self.right_hand_state_array
        init_pose = self._init_pose_left if side == "left" else self._init_pose_right

        last_command = getattr(self, last_attr, None)
        if self._valid_restore_pose(last_command, active_mask):
            return np.asarray(last_command, dtype=np.float64).copy(), "last_command"

        with state_array.get_lock():
            current_state = np.array(state_array[:], dtype=np.float64)
        if self._valid_restore_pose(current_state, active_mask):
            return current_state.copy(), "dds_state"

        if self._valid_restore_pose(init_pose, active_mask):
            return np.asarray(init_pose, dtype=np.float64).copy(), "init_pose"

        return None, "unavailable"

    def restore_initial_pose(self, repeat=None, interval_s=None, settle_s=None):
        if self._init_pose_left is None or self._init_pose_right is None:
            logger_mp.warning("[RH5DG2 restore init pose] skipped: no captured init pose")
            return False
        repeat = self.restore_repeat if repeat is None else max(int(repeat), 1)
        interval_s = self.restore_interval_s if interval_s is None else max(float(interval_s), 0.0)
        settle_s = self.restore_settle_s if settle_s is None else max(float(settle_s), 0.0)
        self.running = False
        if self._hand_control_thread is not None and self._hand_control_thread.is_alive():
            self._hand_control_thread.join(timeout=0.5)
        left_mask, right_mask = self._safe_active_masks_for_init()
        left_start, left_start_source = self._restore_start_pose("left", left_mask)
        right_start, right_start_source = self._restore_start_pose("right", right_mask)
        if left_start is None or right_start is None:
            logger_mp.warning(
                "[RH5DG2 restore init pose] aborted: invalid zero/unavailable ramp start left_source=%s right_source=%s left_start=%s right_start=%s",
                left_start_source,
                right_start_source,
                None if left_start is None else np.round(left_start, 4).tolist(),
                None if right_start is None else np.round(right_start, 4).tolist(),
            )
            return False
        logger_mp.warning(
            "[RH5DG2 restore init pose] ramp repeat=%s interval_s=%.3f duration_s=%.3f left_active=%s right_active=%s left_start_source=%s right_start_source=%s left_start=%s right_start=%s left_target=%s right_target=%s",
            repeat,
            interval_s,
            repeat * interval_s,
            np.where(left_mask)[0].tolist(),
            np.where(right_mask)[0].tolist(),
            left_start_source,
            right_start_source,
            np.round(left_start, 4).tolist(),
            np.round(right_start, 4).tolist(),
            np.round(self._init_pose_left, 4).tolist(),
            np.round(self._init_pose_right, 4).tolist(),
        )
        for step in range(1, repeat + 1):
            alpha = step / repeat
            left_target = left_start + alpha * (self._init_pose_left - left_start)
            right_target = right_start + alpha * (self._init_pose_right - right_start)
            self.ctrl_dual_hand(
                left_target,
                right_target,
                left_active_mask=left_mask,
                right_active_mask=right_mask,
            )
            time.sleep(interval_s)
        if settle_s > 0.0:
            logger_mp.warning("[RH5DG2 restore init pose] settle_s=%.3f", settle_s)
            time.sleep(settle_s)
        logger_mp.warning("[RH5DG2 restore init pose] done publish_count=%s", repeat)
        return True

    def _ensure_hand_msg(self):
        expected = len(RH5DG2_Left_Hand_JointIndex) + len(RH5DG2_Right_Hand_JointIndex)
        if hasattr(self, "hand_msg") and len(getattr(self.hand_msg, "cmds", [])) == expected:
            return
        self.hand_msg = MotorCmds_()
        self.hand_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(expected)]
        for idx in range(RH5DG2_Num_Motors):
            self.hand_msg.cmds[idx].q = 1.0
            self.hand_msg.cmds[RH5DG2_Num_Motors + idx].q = 1.0

    def ctrl_dual_hand(self, left_q_target, right_q_target, left_active_mask=None, right_active_mask=None):
        self._ensure_hand_msg()
        publish_start = time.time()
        if left_active_mask is None:
            left_active_mask = np.ones(RH5DG2_Num_Motors, dtype=bool)
        if right_active_mask is None:
            right_active_mask = np.ones(RH5DG2_Num_Motors, dtype=bool)
        left_active_mask = np.asarray(left_active_mask, dtype=bool).reshape(RH5DG2_Num_Motors)
        right_active_mask = np.asarray(right_active_mask, dtype=bool).reshape(RH5DG2_Num_Motors)
        left_q_target = np.asarray(left_q_target, dtype=np.float64).reshape(RH5DG2_Num_Motors)
        right_q_target = np.asarray(right_q_target, dtype=np.float64).reshape(RH5DG2_Num_Motors)
        # 수정됨: 0~12 인덱스(오른손)에 right_q_target을 할당, 13~25에 left_q_target 할당
        for idx in range(RH5DG2_Num_Motors):
            self.hand_msg.cmds[idx].q = right_q_target[idx]
            _set_cmd_active(self.hand_msg.cmds[idx], right_active_mask[idx])
        for idx in range(RH5DG2_Num_Motors):
            self.hand_msg.cmds[RH5DG2_Num_Motors + idx].q = left_q_target[idx]
            _set_cmd_active(self.hand_msg.cmds[RH5DG2_Num_Motors + idx], left_active_mask[idx])
        for cmd in self.hand_msg.cmds:
            cmd.dq = 0.0
            cmd.tau = 0.0
            cmd.kp = 1.0
            cmd.kd = 0.05
        if self._valid_restore_pose(left_q_target, left_active_mask):
            self._last_left_command = left_q_target.copy()
        if self._valid_restore_pose(right_q_target, right_active_mask):
            self._last_right_command = right_q_target.copy()
        if not hasattr(self, "_publish_debug_count"):
            self._publish_debug_count = 0
        self._publish_debug_count += 1
        if self.simulation_mode:
            should_debug = (
                self._publish_debug_count <= 5
                or (self.log_throttle_s > 0 and publish_start - self._last_debug_ts >= self.log_throttle_s)
            )
        else:
            real_log_interval = max(self.log_throttle_s, 2.0)
            should_debug = self.log_throttle_s > 0 and publish_start - self._last_debug_ts >= real_log_interval
        should_debug = should_debug and self._hand_input_ready_for_debug
        write_ok = self.HandCmd_publisher.Write(self.hand_msg)
        publish_done = time.time()
        publish_latency_ms = (publish_done - publish_start) * 1000.0
        if not write_ok:
            self._debug_print(
                "[RH5DG2 skip publish: DDS writer unavailable] "
                f"write_ok={write_ok} topic={kTopicRH5DG2DFXCommand} domain={self.dds_domain_id} "
                f"{_publisher_debug_status(self.HandCmd_publisher)}"
            )
        if should_debug:
            payload = np.asarray([cmd.q for cmd in self.hand_msg.cmds], dtype=np.float64)
            right_active_indices = _cmd_active_indices(self.hand_msg.cmds, 0, RH5DG2_Num_Motors)
            left_active_indices = _cmd_active_indices(self.hand_msg.cmds, RH5DG2_Num_Motors, RH5DG2_Num_Motors)
            left_enabled_indices = self._safe_enabled_indices_for_side("left")
            right_enabled_indices = self._safe_enabled_indices_for_side("right")
            left_fields = _fmt_motor_fields_for_indices(self.hand_msg.cmds, left_active_indices, RH5DG2_Num_Motors)
            right_fields = _fmt_motor_fields_for_indices(self.hand_msg.cmds, right_active_indices, 0)
            if self.simulation_mode:
                self._debug_print(
                    f"[RH5DG2 teleop publish payload] topic={kTopicRH5DG2DFXCommand} domain={self.dds_domain_id} "
                    f"write_ok={write_ok} "
                    f"{_publisher_debug_status(self.HandCmd_publisher)} "
                    f"safe_active_hand={self.safe_active_hand if self.safe_mode else 'all'} "
                    f"left_enabled={left_enabled_indices} "
                    f"right_enabled={right_enabled_indices} "
                    f"len={payload.size} finite={np.isfinite(payload).all()} "
                    f"min={payload.min():.4f} max={payload.max():.4f} "
                    f"right0_12={np.round(payload[:RH5DG2_Num_Motors], 4).tolist()} "
                    f"left13_25={np.round(payload[RH5DG2_Num_Motors:], 4).tolist()} "
                    f"right_active={right_active_indices} "
                    f"left_active={left_active_indices} "
                    f"right_active_mask_requested={np.where(right_active_mask)[0].tolist()} "
                    f"left_active_mask_requested={np.where(left_active_mask)[0].tolist()} "
                    f"right_fields={right_fields} "
                    f"left_fields={left_fields} "
                    f"left={_fmt_debug(left_q_target)} right={_fmt_debug(right_q_target)} "
                    f"left_fingers={_finger_command_values(left_q_target)} "
                    f"right_fingers={_finger_command_values(right_q_target)}"
                )
                debug_field_count = RH5DG2_Num_Motors if self.safe_mode else 5
                self._debug_print(f"[RH5DG2 teleop publish fields right] {_fmt_motor_fields(self.hand_msg.cmds, 0, debug_field_count)}")
                self._debug_print(f"[RH5DG2 teleop publish fields left] {_fmt_motor_fields(self.hand_msg.cmds, RH5DG2_Num_Motors, debug_field_count)}")
            else:
                self._debug_print(
                    f"[RH5DG2 teleop publish] topic={kTopicRH5DG2DFXCommand} domain={self.dds_domain_id} "
                    f"write_ok={write_ok} len={payload.size} finite={np.isfinite(payload).all()} "
                    f"safe_active_hand={self.safe_active_hand if self.safe_mode else 'all'} "
                    f"left_enabled={left_enabled_indices} right_enabled={right_enabled_indices} "
                    f"right_active={right_active_indices} left_active={left_active_indices} "
                    f"right_fields={right_fields} "
                    f"left_fields={left_fields} "
                    f"blocked_print_count={getattr(self, '_blocked_print_count', 0)}"
                )
            self._debug_print(
                f"[RH5DG2 publish hz] hz={_rate_hz(self._publish_debug_count, self._publish_rate_start_ts):.2f} "
                f"count={self._publish_debug_count}"
            )
            self._debug_print(f"[RH5DG2 publish latency ms] latency_ms={publish_latency_ms:.3f}")
            self._last_debug_ts = publish_start
        return publish_done, write_ok, publish_latency_ms

    def _postprocess_real_safe_command(
        self,
        raw_q,
        joint_limits,
        joint_names,
        hand_data,
        current_raw_state,
        side,
        should_debug=False,
    ):
        normalized, debug = _postprocess_sim_command(
            raw_q,
            joint_limits,
            joint_names,
            hand_data,
            curl_scale=self.curl_scale,
            finger_curl_scales=self.finger_curl_scales,
            side=side,
        )
        current = np.asarray(current_raw_state, dtype=np.float64).reshape(RH5DG2_Num_Motors)
        current = np.where(np.isfinite(current), current, 0.0)
        baseline_attr = "_safe_baseline_left" if side == "left" else "_safe_baseline_right"
        baseline = getattr(self, baseline_attr)
        if baseline is None or np.asarray(baseline).size != RH5DG2_Num_Motors:
            baseline = current.copy()
            setattr(self, baseline_attr, baseline)
            logger_mp.warning(
                "[RH5DG2 safe baseline] side=%s baseline=%s",
                side,
                np.round(baseline, 4).tolist(),
            )

        before_gain = current.copy()
        after_gain = current.copy()
        after_clamp = current.copy()
        active_mask = np.zeros(RH5DG2_Num_Motors, dtype=bool)
        thumb_debug = None
        spread_score = float(np.clip(debug["prior_debug"]["v_spread"].get("score", 0.5), 0.0, 1.0))
        spread_centered = 2.0 * spread_score - 1.0
        for raw_idx in sorted(self.safe_enabled_indices):
            semantic_idx = RH5DG2_SAFE_RAW_FROM_NORMALIZED.get(raw_idx)
            if semantic_idx is None:
                if raw_idx not in RH5DG2_RAW_SPREAD_INDICES:
                    continue
                limit = float(RH5DG2_SAFE_DELTA_LIMIT.get(raw_idx, 0.0))
                direction = float(RH5DG2_SAFE_SPREAD_DIRECTION.get(raw_idx, 1.0))
                before = baseline[raw_idx] + direction * spread_centered * limit
                before_gain[raw_idx] = before
                after_gain[raw_idx] = before
                clamp_lo = max(float(RH5DG2_SAFE_ABS_MIN[raw_idx]), float(baseline[raw_idx] - limit))
                clamp_hi = min(float(RH5DG2_SAFE_ABS_MAX[raw_idx]), float(baseline[raw_idx] + limit))
                if clamp_lo > clamp_hi:
                    logger_mp.warning(
                        "[RH5DG2 safe spread clamp] invalid clamp window raw_idx=%s lo=%.4f hi=%.4f baseline=%.4f limit=%.4f abs_min=%.4f abs_max=%.4f",
                        raw_idx,
                        clamp_lo,
                        clamp_hi,
                        float(baseline[raw_idx]),
                        limit,
                        float(RH5DG2_SAFE_ABS_MIN[raw_idx]),
                        float(RH5DG2_SAFE_ABS_MAX[raw_idx]),
                    )
                    fallback = float(np.clip(baseline[raw_idx], RH5DG2_SAFE_ABS_MIN[raw_idx], RH5DG2_SAFE_ABS_MAX[raw_idx]))
                    clamp_lo = clamp_hi = fallback
                after_clamp[raw_idx] = np.clip(before, clamp_lo, clamp_hi)
                active_mask[raw_idx] = True
                continue
            limit = float(RH5DG2_SAFE_DELTA_LIMIT.get(raw_idx, 0.0))
            closure = float(np.clip(1.0 - normalized[semantic_idx], 0.0, 1.0))
            retarget_target = baseline[raw_idx] + self.raw_close_direction * closure * limit
            before_gain[raw_idx] = retarget_target
            before = baseline[raw_idx] + self.safe_gain * (retarget_target - baseline[raw_idx])
            after_gain[raw_idx] = before
            clamp_lo = max(float(RH5DG2_SAFE_ABS_MIN[raw_idx]), float(baseline[raw_idx] - limit))
            clamp_hi = min(float(RH5DG2_SAFE_ABS_MAX[raw_idx]), float(baseline[raw_idx] + limit))
            if clamp_lo > clamp_hi:
                logger_mp.warning(
                    "[RH5DG2 safe clamp] invalid clamp window raw_idx=%s lo=%.4f hi=%.4f baseline=%.4f limit=%.4f abs_min=%.4f abs_max=%.4f",
                    raw_idx,
                    clamp_lo,
                    clamp_hi,
                    float(baseline[raw_idx]),
                    limit,
                    float(RH5DG2_SAFE_ABS_MIN[raw_idx]),
                    float(RH5DG2_SAFE_ABS_MAX[raw_idx]),
                )
                clamp_lo = clamp_hi = float(baseline[raw_idx])
            after_clamp[raw_idx] = np.clip(before, clamp_lo, clamp_hi)
            active_mask[raw_idx] = True

        if self.enable_thumb:
            thumb_score = debug["curl_debug"]["scores"].get("thumb", {})
            thumb_landmark_curl = float(np.clip(thumb_score.get("curl", 0.0), 0.0, 1.0))
            thumb_raw_named = _thumb_raw_values(raw_q, joint_names, side)
            thumb_raw_close_ratio = float(
                np.clip(
                    max(
                        abs(thumb_raw_named.get(f"{side}_thumb_mcp_joint", 0.0)),
                        abs(thumb_raw_named.get(f"{side}_thumb_dip_joint", 0.0)),
                    ),
                    0.0,
                    1.0,
                )
            )
            thumb_close_ratio = thumb_raw_close_ratio if self.thumb_source == "raw" else thumb_landmark_curl
            thumb_targets = {}
            thumb_current = {}
            thumb_limits = {}
            for raw_idx in RH5DG2_RAW_THUMB_INDICES:
                if raw_idx not in self.safe_enabled_indices:
                    continue
                close_value = float(RH5DG2_SAFE_THUMB_CLOSE[raw_idx])
                open_value = float(RH5DG2_SAFE_THUMB_OPEN[raw_idx])
                scale = float(self.thumb_scales[raw_idx])
                before = open_value - thumb_close_ratio * scale * (open_value - close_value)
                before_gain[raw_idx] = before
                after_gain[raw_idx] = before
                after_clamp[raw_idx] = np.clip(before, RH5DG2_SAFE_ABS_MIN[raw_idx], RH5DG2_SAFE_ABS_MAX[raw_idx])
                active_mask[raw_idx] = True
                thumb_targets[int(raw_idx)] = float(np.round(after_clamp[raw_idx], 4))
                thumb_current[int(raw_idx)] = float(np.round(current[raw_idx], 4))
                thumb_limits[int(raw_idx)] = {
                    "close": close_value,
                    "open": open_value,
                    "abs_min": float(RH5DG2_SAFE_ABS_MIN[raw_idx]),
                    "abs_max": float(RH5DG2_SAFE_ABS_MAX[raw_idx]),
                    "scale": scale,
                }
            thumb_debug = {
                "thumb_source": self.thumb_source,
                "thumb_close_ratio": thumb_close_ratio,
                "thumb_curl": thumb_landmark_curl,
                "thumb_raw_close_ratio": thumb_raw_close_ratio,
                "thumb_raw_named": thumb_raw_named,
                "current": thumb_current,
                "target": thumb_targets,
                "limits": thumb_limits,
                "score": thumb_score,
            }

        enabled_list = sorted(self.safe_enabled_indices)
        delta_enabled = after_clamp[enabled_list] - baseline[enabled_list] if enabled_list else np.array([])
        max_abs_delta = float(np.max(np.abs(delta_enabled))) if delta_enabled.size else 0.0
        delta_small_threshold = float(os.getenv("RH5DG2_SAFE_TARGET_DELTA_WARN", "0.5"))
        safe_debug = {
            "normalized": normalized,
            "before_gain": before_gain,
            "after_gain": after_gain,
            "after_clamp": after_clamp,
            "delta_from_baseline": after_clamp - baseline,
            "delta_from_current": after_clamp - current,
            "max_abs_delta_enabled": max_abs_delta,
            "delta_small_threshold": delta_small_threshold,
            "baseline": baseline.copy(),
            "current": current.copy(),
            "active_mask": active_mask.copy(),
            "enabled_values": _fmt_safe_enabled(self.safe_enabled_indices, after_clamp),
            "enabled_abs_min": _fmt_safe_enabled(self.safe_enabled_indices, RH5DG2_SAFE_ABS_MIN),
            "enabled_abs_max": _fmt_safe_enabled(self.safe_enabled_indices, RH5DG2_SAFE_ABS_MAX),
            "spread_debug": {
                "score": spread_score,
                "centered": spread_centered,
                "direction": RH5DG2_SAFE_SPREAD_DIRECTION,
                "active_indices": [
                    int(idx)
                    for idx in RH5DG2_RAW_SPREAD_INDICES
                    if idx in self.safe_enabled_indices
                ],
            },
            "thumb_debug": thumb_debug,
            "sim_debug": debug,
        }
        now = time.time()
        debug_attr = f"_last_safe_target_debug_{side}"
        last_debug = getattr(self, debug_attr, 0.0)
        safe_debug_interval = 1.0 if self.simulation_mode else 2.0
        should_safe_debug = should_debug or now - last_debug >= safe_debug_interval
        if should_safe_debug:
            setattr(self, debug_attr, now)
            self._debug_print(
                "[RH5DG2 safe raw command] "
                f"hand_tracking_ready=True "
                f"side={side} gain={self.safe_gain:.3f} "
                f"enabled={sorted(self.safe_enabled_indices)} "
                f"disabled={sorted(self.safe_disabled_indices)} "
                f"baseline={np.round(baseline, 4).tolist()} "
                f"current_state={np.round(current, 4).tolist()} "
                f"raw_target_before_gain={np.round(before_gain, 4).tolist()} "
                f"raw_target_after_gain={np.round(after_gain, 4).tolist()} "
                f"raw_target_after_clamp={np.round(after_clamp, 4).tolist()} "
                f"delta_from_baseline={np.round(safe_debug['delta_from_baseline'], 4).tolist()} "
                f"delta_from_current={np.round(safe_debug['delta_from_current'], 4).tolist()} "
                f"max_abs_delta_enabled={max_abs_delta:.4f} "
                f"enabled actuator values={safe_debug['enabled_values']} "
                f"enabled_abs_min={safe_debug['enabled_abs_min']} "
                f"enabled_abs_max={safe_debug['enabled_abs_max']} "
                f"spread={safe_debug['spread_debug']}"
            )
            if max_abs_delta < delta_small_threshold:
                self._debug_print(
                    "[RH5DG2 safe raw command] target_delta too small after gain/clamp "
                    f"max_abs_delta_enabled={max_abs_delta:.4f} "
                    f"threshold={delta_small_threshold:.4f} "
                    f"gain={self.safe_gain:.3f} safe_delta_limit={RH5DG2_SAFE_DELTA_LIMIT}"
                )
            self._debug_print(
                "[RH5DG2 safe curl source] "
                f"side={side} curl_scores={_fmt_curl_scores(debug['curl_debug']['scores'])} "
                f"command_delta={debug['finger_command_delta']} "
                f"curl_scale={debug['curl_debug']['curl_scale']:.3f} "
                f"finger_curl_scales={debug['curl_debug']['finger_curl_scales']} "
                f"normalized={np.round(normalized, 4).tolist()}"
            )
            if side == "right":
                if thumb_debug is not None:
                    self._debug_print(
                        "[RH5DG2 safe thumb detail] "
                        f"thumb_source={thumb_debug['thumb_source']} "
                        f"thumb_close_ratio={thumb_debug['thumb_close_ratio']:.4f} "
                        f"thumb_curl={thumb_debug['thumb_curl']:.4f} "
                        f"thumb_raw_close_ratio={thumb_debug['thumb_raw_close_ratio']:.4f} "
                        f"thumb_raw_named={thumb_debug['thumb_raw_named']} "
                        f"current={thumb_debug['current']} "
                        f"target={thumb_debug['target']} "
                        f"limits={thumb_debug['limits']} "
                        f"score={thumb_debug['score']} "
                        f"active_indices={np.where(active_mask)[0].tolist()}"
                    )
                self._debug_print(
                    "[RH5DG2 safe index detail] "
                    f"landmarks={_finger_landmark_debug(hand_data, 'index')} "
                    f"score={debug['curl_debug']['scores']['index']} "
                    f"semantic_index_values={{4: {float(normalized[4]):.4f}, 5: {float(normalized[5]):.4f}}} "
                    f"raw_target_index4={float(after_clamp[4]):.4f} "
                    f"raw_target_index9={float(after_clamp[9]):.4f} "
                    f"baseline_index4={float(baseline[4]):.4f} "
                    f"baseline_index9={float(baseline[9]):.4f} "
                    f"delta_index4={float(after_clamp[4] - baseline[4]):.4f} "
                    f"delta_index9={float(after_clamp[9] - baseline[9]):.4f}"
                )
        return after_clamp, active_mask, safe_debug

    def _retarget(self, left_hand_data, right_hand_data, left_state_raw=None, right_state_raw=None):
        # [핵심 수정] DexPilot 모드(상대 벡터)와 Position 모드(절대 좌표) 자동 호환 처리
        left_idx = np.array(self.hand_retargeting.left_indices)
        right_idx = np.array(self.hand_retargeting.right_indices)

        if not hasattr(self, "_retarget_input_debug_count"):
            self._retarget_input_debug_count = 0
        self._retarget_input_debug_count += 1
        if self.simulation_mode:
            should_debug = self._retarget_input_debug_count <= 5 or self._retarget_input_debug_count % 50 == 0
        else:
            now = time.time()
            real_log_interval = max(self.log_throttle_s, 2.0)
            last_retarget_debug = getattr(self, "_last_retarget_debug_ts", 0.0)
            should_debug = self.log_throttle_s > 0 and now - last_retarget_debug >= real_log_interval
            if should_debug:
                self._last_retarget_debug_ts = now
        if should_debug:
            self._debug_print(
                "[RH5DG2 teleop retarget input] "
                f"left={_fmt_hand_input(left_hand_data, left_idx)} "
                f"right={_fmt_hand_input(right_hand_data, right_idx)}"
            )

        if left_idx.ndim == 2:
            # Vector/DexPilot modes expect task-space vectors in the RH5DG2 URDF hand frame.
            ref_left_value, left_source, left_target, left_input_vectors = _prepare_vector_reference(
                left_hand_data, left_idx
            )
            ref_right_value, right_source, right_target, right_input_vectors = _prepare_vector_reference(
                right_hand_data, right_idx
            )
            if should_debug:
                self._debug_print(
                    "[RH5DG2 teleop vector frame] "
                    f"left_source={_fmt_matrix(left_source)} left_target={_fmt_matrix(left_target)} "
                    f"left_input_vectors={_fmt_matrix(left_input_vectors)} "
                    f"left_ref_rh5dg2={_fmt_matrix(ref_left_value)} "
                    f"right_source={_fmt_matrix(right_source)} right_target={_fmt_matrix(right_target)} "
                    f"right_input_vectors={_fmt_matrix(right_input_vectors)} "
                    f"right_ref_rh5dg2={_fmt_matrix(ref_right_value)}"
                )
        else:
            # Position mode expects fingertip points in the RH5DG2 URDF hand frame.
            (
                ref_left_value,
                left_wrist,
                left_palm,
                left_selected,
                left_wrist_rel,
                left_palm_rel,
            ) = _prepare_position_reference(left_hand_data, left_idx)
            (
                ref_right_value,
                right_wrist,
                right_palm,
                right_selected,
                right_wrist_rel,
                right_palm_rel,
            ) = _prepare_position_reference(right_hand_data, right_idx)
            if should_debug:
                self._debug_print(
                    "[RH5DG2 teleop position frame] "
                    f"left_wrist={_fmt_matrix(left_wrist)} left_palm={_fmt_matrix(left_palm)} "
                    f"left_selected={_fmt_matrix(left_selected)} "
                    f"left_wrist_rel={_fmt_matrix(left_wrist_rel)} "
                    f"left_palm_rel={_fmt_matrix(left_palm_rel)} "
                    f"left_ref_rh5dg2={_fmt_matrix(ref_left_value)} "
                    f"right_wrist={_fmt_matrix(right_wrist)} right_palm={_fmt_matrix(right_palm)} "
                    f"right_selected={_fmt_matrix(right_selected)} "
                    f"right_wrist_rel={_fmt_matrix(right_wrist_rel)} "
                    f"right_palm_rel={_fmt_matrix(right_palm_rel)} "
                    f"right_ref_rh5dg2={_fmt_matrix(ref_right_value)}"
                )

        # 계산된 ref_value를 바탕으로 역기구학(IK) 수행
        left_q_target = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.left_retargeting_to_hardware]
        right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[self.hand_retargeting.right_retargeting_to_hardware]
        left_q_raw = np.asarray(left_q_target, dtype=np.float64).copy()
        right_q_raw = np.asarray(right_q_target, dtype=np.float64).copy()

        if should_debug:
            self._debug_print(
                f"[RH5DG2 teleop retarget raw] left={_fmt_debug(left_q_raw)} right={_fmt_debug(right_q_raw)} "
                f"left_named={_fmt_named_values(self.hand_retargeting.left_joint_names, left_q_raw)} "
                f"right_named={_fmt_named_values(self.hand_retargeting.right_joint_names, right_q_raw)}"
            )

        # 시뮬레이션 모드 정규화(Normalization) 적용 (이전에 적용했던 수정사항)
        if self.simulation_mode:
            left_q_target, left_debug = _postprocess_sim_command(
                left_q_target,
                self.hand_retargeting.left_joint_limits,
                self.hand_retargeting.left_joint_names,
                left_hand_data,
                side="left",
            )
            right_q_target, right_debug = _postprocess_sim_command(
                right_q_target,
                self.hand_retargeting.right_joint_limits,
                self.hand_retargeting.right_joint_names,
                right_hand_data,
                side="right",
            )
            if should_debug:
                self._debug_print(
                    "[RH5DG2 teleop normalize detail] "
                    f"left_limits={_fmt_named_limits(self.hand_retargeting.left_joint_names, self.hand_retargeting.left_joint_limits)} "
                    f"right_limits={_fmt_named_limits(self.hand_retargeting.right_joint_names, self.hand_retargeting.right_joint_limits)} "
                    f"left_unclamped={_fmt_named_values(self.hand_retargeting.left_joint_names, left_debug['unclamped'])} "
                    f"right_unclamped={_fmt_named_values(self.hand_retargeting.right_joint_names, right_debug['unclamped'])} "
                    f"left_clamped={_fmt_named_values(self.hand_retargeting.left_joint_names, left_debug['clamped'])} "
                    f"right_clamped={_fmt_named_values(self.hand_retargeting.right_joint_names, right_debug['clamped'])} "
                    f"left_gain={_fmt_named_values(self.hand_retargeting.left_joint_names, left_debug['gain'])} "
                    f"right_gain={_fmt_named_values(self.hand_retargeting.right_joint_names, right_debug['gain'])} "
                    f"left_open_prior={_fmt_named_values(self.hand_retargeting.left_joint_names, left_debug['prior'])} "
                    f"right_open_prior={_fmt_named_values(self.hand_retargeting.right_joint_names, right_debug['prior'])} "
                    f"open_calibration_enabled={_env_flag('RH5DG2_ENABLE_OPEN_CALIBRATION')} "
                    f"teleop_safe_close_enabled={_env_flag('RH5DG2_ENABLE_TELEOP_SAFE_CLOSE')} "
                    f"left_calibrated={_fmt_named_values(self.hand_retargeting.left_joint_names, left_debug['calibrated'])} "
                    f"right_calibrated={_fmt_named_values(self.hand_retargeting.right_joint_names, right_debug['calibrated'])} "
                    f"left_curl_command={_fmt_named_values(self.hand_retargeting.left_joint_names, left_debug['curl'])} "
                    f"right_curl_command={_fmt_named_values(self.hand_retargeting.right_joint_names, right_debug['curl'])} "
                    f"left_safe_floor={_fmt_named_values(self.hand_retargeting.left_joint_names, left_q_target)} "
                    f"right_safe_floor={_fmt_named_values(self.hand_retargeting.right_joint_names, right_q_target)}"
                )
                self._debug_print(
                    "[RH5DG2 teleop open prior] "
                    f"left_shape={_fmt_shape_scores(left_debug['prior_debug']['scores'])} "
                    f"right_shape={_fmt_shape_scores(right_debug['prior_debug']['scores'])} "
                    f"left_prior_delta={_fmt_named_values(self.hand_retargeting.left_joint_names, left_debug['prior_debug']['delta'])} "
                    f"right_prior_delta={_fmt_named_values(self.hand_retargeting.right_joint_names, right_debug['prior_debug']['delta'])} "
                    f"left_v_spread={left_debug['prior_debug']['v_spread']} "
                    f"right_v_spread={right_debug['prior_debug']['v_spread']} "
                    f"strength={left_debug['prior_debug']['strength']:.3f} "
                    f"threshold={left_debug['prior_debug']['threshold']:.3f}"
                )
                self._debug_print(
                    "[RH5DG2 teleop curl command] "
                    f"left_curl_scores={_fmt_curl_scores(left_debug['curl_debug']['scores'])} "
                    f"right_curl_scores={_fmt_curl_scores(right_debug['curl_debug']['scores'])} "
                    f"left_delta={left_debug['finger_command_delta']} "
                    f"right_delta={right_debug['finger_command_delta']} "
                    f"left_decouple_delta={_fmt_named_values(self.hand_retargeting.left_joint_names, left_debug['curl_debug']['decouple_delta'])} "
                    f"right_decouple_delta={_fmt_named_values(self.hand_retargeting.right_joint_names, right_debug['curl_debug']['decouple_delta'])} "
                    f"left_targets={_fmt_named_values(self.hand_retargeting.left_joint_names, left_debug['curl_debug']['targets'])} "
                    f"right_targets={_fmt_named_values(self.hand_retargeting.right_joint_names, right_debug['curl_debug']['targets'])} "
                    f"decouple_enabled={left_debug['curl_debug']['decouple_enabled']} "
                    f"decouple_strength={left_debug['curl_debug']['decouple_strength']:.3f} "
                    f"decouple_threshold={left_debug['curl_debug']['decouple_threshold']:.3f} "
                    f"strength={left_debug['curl_debug']['strength']:.3f} "
                    f"threshold={left_debug['curl_debug']['threshold']:.3f}"
                )
                self._debug_print(
                    "[RH5DG2 teleop command mapping] "
                    f"left_mapping={left_debug['mapping']} "
                    f"right_mapping={right_debug['mapping']} "
                    f"left_denorm_rad={_fmt_named_values(self.hand_retargeting.left_joint_names, left_debug['denorm_rad'])} "
                    f"right_denorm_rad={_fmt_named_values(self.hand_retargeting.right_joint_names, right_debug['denorm_rad'])} "
                    f"left_saturation={left_debug['saturation']} "
                    f"right_saturation={right_debug['saturation']}"
                )
                self._debug_print(
                    "[RH5DG2 teleop calibrated] "
                    f"left_scores={_fmt_finger_scores(left_debug['finger_scores'])} "
                    f"right_scores={_fmt_finger_scores(right_debug['finger_scores'])} "
                    f"left={_fmt_named_values(self.hand_retargeting.left_joint_names, left_q_target)} "
                    f"right={_fmt_named_values(self.hand_retargeting.right_joint_names, right_q_target)}"
                )
        elif self.safe_mode:
            left_state_raw = np.zeros(RH5DG2_Num_Motors, dtype=np.float64) if left_state_raw is None else left_state_raw
            right_state_raw = np.zeros(RH5DG2_Num_Motors, dtype=np.float64) if right_state_raw is None else right_state_raw
            left_q_target, self._last_left_active_mask, left_debug = self._postprocess_real_safe_command(
                left_q_target,
                self.hand_retargeting.left_joint_limits,
                self.hand_retargeting.left_joint_names,
                left_hand_data,
                left_state_raw,
                "left",
                should_debug=should_debug,
            )
            right_q_target, self._last_right_active_mask, right_debug = self._postprocess_real_safe_command(
                right_q_target,
                self.hand_retargeting.right_joint_limits,
                self.hand_retargeting.right_joint_names,
                right_hand_data,
                right_state_raw,
                "right",
                should_debug=should_debug,
            )
            self._last_left_active_mask, self._last_right_active_mask = self._safe_active_masks_for_init()
            if should_debug:
                self._debug_print(
                    "[RH5DG2 teleop safe mode detail] "
                    f"left_enabled={left_debug['enabled_values']} "
                    f"right_enabled={right_debug['enabled_values']} "
                    f"safe_active_hand={self.safe_active_hand} "
                    f"left_active={np.where(self._last_left_active_mask)[0].tolist()} "
                    f"right_active={np.where(self._last_right_active_mask)[0].tolist()}"
                )
        else:
            self._last_left_active_mask = np.ones(RH5DG2_Num_Motors, dtype=bool)
            self._last_right_active_mask = np.ones(RH5DG2_Num_Motors, dtype=bool)
            left_q_target = _clip_to_joint_limits(left_q_target, self.hand_retargeting.left_joint_limits)
            right_q_target = _clip_to_joint_limits(right_q_target, self.hand_retargeting.right_joint_limits)
            if should_debug:
                self._debug_print(
                    "[RH5DG2 teleop clip detail] "
                    f"left_limits={_fmt_named_limits(self.hand_retargeting.left_joint_names, self.hand_retargeting.left_joint_limits)} "
                    f"right_limits={_fmt_named_limits(self.hand_retargeting.right_joint_names, self.hand_retargeting.right_joint_limits)} "
                    f"left_clipped={_fmt_named_values(self.hand_retargeting.left_joint_names, left_q_target)} "
                    f"right_clipped={_fmt_named_values(self.hand_retargeting.right_joint_names, right_q_target)}"
                )

        if should_debug:
            self._debug_print(
                f"[RH5DG2 teleop retarget] sim={self.simulation_mode} "
                f"left={_fmt_debug(left_q_target)} right={_fmt_debug(right_q_target)}"
            )

        return left_q_target, right_q_target

    def control_process(
        self,
        left_hand_array,
        right_hand_array,
        left_hand_state_array,
        right_hand_state_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
        input_timestamp_value=None,
    ):
        self.running = True

        # Run DFX hand control in a thread so the DDS writer shares the already
        # initialized participant instead of creating CycloneDDS entities after fork.
        ChannelFactoryInitialize(self.dds_domain_id, networkInterface=self.network_interface)
        self.HandCmd_publisher = ChannelPublisher(kTopicRH5DG2DFXCommand, MotorCmds_)
        self.HandCmd_publisher.Init()
        logger_mp.info(
            f"[RH5DG2_Controller_DFX] Control DDS publisher initialized "
            f"topic={kTopicRH5DG2DFXCommand} domain={self.dds_domain_id} "
            f"network_interface={self.network_interface} "
            f"{_publisher_debug_status(self.HandCmd_publisher)}"
        )

        left_q_target = np.full(RH5DG2_Num_Motors, 1.0)
        right_q_target = np.full(RH5DG2_Num_Motors, 1.0)
        left_active_mask = np.ones(RH5DG2_Num_Motors, dtype=bool)
        right_active_mask = np.ones(RH5DG2_Num_Motors, dtype=bool)
        if self.safe_mode:
            left_active_mask[:] = False
            right_active_mask[:] = False

        self._ensure_hand_msg()

        try:
            loop_count = 0
            previous_hand_ready = None
            while self.running:
                try:
                    loop_count += 1
                    start_time = time.time()
                    with left_hand_array.get_lock():
                        left_hand_data = np.array(left_hand_array[:]).reshape(25, 3).copy()
                    with right_hand_array.get_lock():
                        right_hand_data = np.array(right_hand_array[:]).reshape(25, 3).copy()
                    input_timestamp = 0.0
                    if input_timestamp_value is not None:
                        with input_timestamp_value.get_lock():
                            input_timestamp = float(input_timestamp_value.value)

                    left_state_raw = np.array(left_hand_state_array[:], dtype=np.float64)
                    right_state_raw = np.array(right_hand_state_array[:], dtype=np.float64)
                    state_data = np.concatenate((left_state_raw, right_state_raw))
                    if self.safe_mode and np.isfinite(left_state_raw).all() and np.isfinite(right_state_raw).all():
                        if self._safe_baseline_left is None:
                            self._safe_baseline_left = left_state_raw.copy()
                        if self._safe_baseline_right is None:
                            self._safe_baseline_right = right_state_raw.copy()

                    hand_status = _hand_landmark_status(left_hand_data, right_hand_data)
                    if self.simulation_mode and not self.safe_mode:
                        hand_ready = _is_legacy_sim_hand_tracking_ready(left_hand_data, right_hand_data)
                        hand_status["hand_tracking_ready"] = bool(hand_ready)
                        hand_status["ready_policy"] = "legacy_sim"
                    else:
                        hand_ready = hand_status["hand_tracking_ready"]
                        hand_status["ready_policy"] = "safe_valid_points"
                    self._hand_input_ready_for_debug = bool(hand_ready)
                    if previous_hand_ready is None or hand_ready != previous_hand_ready:
                        if hand_ready:
                            self._debug_print("[RH5DG2 hand input valid] PICO hand tracking resumed.")
                        else:
                            self._debug_print(
                                "[RH5DG2 hand input invalid] PICO hand landmarks unavailable; "
                                "suppress periodic debug logs until tracking resumes."
                            )
                        previous_hand_ready = bool(hand_ready)
                    if self.simulation_mode:
                        should_debug = (
                            loop_count <= 5
                            or (self.log_throttle_s > 0 and start_time - self._last_debug_ts >= self.log_throttle_s)
                        )
                    else:
                        real_log_interval = max(self.log_throttle_s, 2.0)
                        should_debug = self.log_throttle_s > 0 and start_time - self._last_debug_ts >= real_log_interval
                    should_debug = should_debug and hand_ready
                    retarget_start = time.time()
                    if hand_ready:
                        left_q_target, right_q_target = self._retarget(
                            left_hand_data,
                            right_hand_data,
                            left_state_raw=left_state_raw,
                            right_state_raw=right_state_raw,
                        )
                        left_active_mask = self._last_left_active_mask.copy()
                        right_active_mask = self._last_right_active_mask.copy()
                        retarget_done = time.time()
                        retarget_latency_ms = (retarget_done - retarget_start) * 1000.0
                        if should_debug:
                            self._debug_print(
                                "[RH5DG2 hand tracking status] "
                                f"hand_tracking_ready={hand_status['hand_tracking_ready']} "
                                f"left_finite={hand_status['left_finite']} "
                                f"right_finite={hand_status['right_finite']} "
                                f"left_allzero={hand_status['left_allzero']} "
                                f"right_allzero={hand_status['right_allzero']} "
                                f"left_valid_points={hand_status['left_valid_points']} "
                                f"right_valid_points={hand_status['right_valid_points']} "
                                f"min_valid_points={hand_status['min_valid_points']} "
                                f"ready_policy={hand_status['ready_policy']} "
                                f"left_sentinel={hand_status['left_sentinel']}"
                            )
                            self._debug_print(
                                f"[RH5DG2 DFX control after retarget] ready={hand_ready} "
                                f"left={_fmt_debug(left_q_target)} right={_fmt_debug(right_q_target)}"
                            )
                            self._debug_print(f"[RH5DG2 retarget latency ms] latency_ms={retarget_latency_ms:.3f}")
                    else:
                        if self.safe_mode:
                            left_q_target = left_state_raw.copy()
                            right_q_target = right_state_raw.copy()
                            left_active_mask = np.zeros(RH5DG2_Num_Motors, dtype=bool)
                            right_active_mask = np.zeros(RH5DG2_Num_Motors, dtype=bool)
                        retarget_done = time.time()
                        retarget_latency_ms = 0.0
                        if should_debug:
                            reasons = []
                            if not hand_status["left_finite"] or not hand_status["right_finite"]:
                                reasons.append("non_finite")
                            if hand_status["left_allzero"] or hand_status["right_allzero"]:
                                reasons.append("allzero")
                                self._debug_print(
                                    "[RH5DG2 hand input invalid] PICO hand landmarks are all zero; "
                                    "skip retarget and hold baseline."
                                )
                            if hand_status["left_valid_points"] < hand_status["min_valid_points"] or hand_status["right_valid_points"] < hand_status["min_valid_points"]:
                                reasons.append("valid_points_below_min")
                            if not reasons:
                                reasons.append("ready_gate_false")
                            self._debug_print(
                                "[RH5DG2 skip retarget: ready gate false] "
                                f"reasons={reasons}"
                            )
                            self._debug_print(
                                "[RH5DG2 hand tracking status] "
                                f"hand_tracking_ready={hand_status['hand_tracking_ready']} "
                                f"left_finite={hand_status['left_finite']} "
                                f"right_finite={hand_status['right_finite']} "
                                f"left_allzero={hand_status['left_allzero']} "
                                f"right_allzero={hand_status['right_allzero']} "
                                f"left_valid_points={hand_status['left_valid_points']} "
                                f"right_valid_points={hand_status['right_valid_points']} "
                                f"min_valid_points={hand_status['min_valid_points']} "
                                f"ready_policy={hand_status['ready_policy']} "
                                f"left_sentinel={hand_status['left_sentinel']}"
                            )
                            self._debug_print(
                                f"[RH5DG2 DFX control no retarget] ready={hand_ready} "
                                f"safe_mode={self.safe_mode} "
                                f"right_hand_zero={hand_status['right_allzero']} "
                                f"left_sentinel={hand_status['left_sentinel']} "
                                f"left={_fmt_debug(left_q_target)} right={_fmt_debug(right_q_target)} "
                                f"left_active={np.where(left_active_mask)[0].tolist()} "
                                f"right_active={np.where(right_active_mask)[0].tolist()}"
                            )
                    action_data = np.concatenate((left_q_target, right_q_target))
                    if dual_hand_state_array is not None and dual_hand_action_array is not None:
                        with dual_hand_data_lock:
                            dual_hand_state_array[:] = state_data
                            dual_hand_action_array[:] = action_data

                    if should_debug:
                        self._debug_print(
                            f"[RH5DG2 DFX control pre publish] ready={hand_ready} "
                            f"left={_fmt_debug(left_q_target)} right={_fmt_debug(right_q_target)} "
                            f"action={_fmt_debug(action_data)}"
                        )
                    publish_done, write_ok, publish_latency_ms = self.ctrl_dual_hand(
                        left_q_target,
                        right_q_target,
                        left_active_mask=left_active_mask,
                        right_active_mask=right_active_mask,
                    )
                    time_elapsed = publish_done - start_time
                    sleep_s = max(0.0, (1.0 / self.fps) - time_elapsed)
                    if should_debug:
                        self._debug_print(f"[RH5DG2 retarget hz] hz={_rate_hz(loop_count, self._loop_rate_start_ts):.2f} count={loop_count}")
                        self._debug_print(f"[RH5DG2 control loop sleep] sleep_ms={sleep_s * 1000.0:.3f} fps_target={self.fps}")
                        if input_timestamp > 0.0:
                            self._debug_print(
                                f"[RH5DG2 latency breakdown] fast_mode={self.fast_mode} ready={hand_ready} "
                                f"input_timestamp={input_timestamp:.6f} loop_start={start_time:.6f} "
                                f"retarget_start={retarget_start:.6f} retarget_done={retarget_done:.6f} "
                                f"publish_done={publish_done:.6f} write_ok={write_ok} "
                                f"input_to_loop_ms={(start_time - input_timestamp) * 1000.0:.2f} "
                                f"loop_to_retarget_start_ms={(retarget_start - start_time) * 1000.0:.2f} "
                                f"retarget_ms={retarget_latency_ms:.2f} publish_ms={publish_latency_ms:.2f} "
                                f"input_to_publish_ms={(publish_done - input_timestamp) * 1000.0:.2f} "
                                f"sleep_ms={sleep_s * 1000.0:.2f}"
                            )
                        self._last_debug_ts = start_time
                    time.sleep(sleep_s)
                except BlockingIOError:
                    self._blocked_print_count = getattr(self, "_blocked_print_count", 0) + 1
                    time.sleep(max(0.001, min(0.02, 1.0 / max(float(self.fps), 1.0))))
                    continue
        finally:
            logger_mp.info("RH5DG2_Controller_DFX has been closed.")


class RH5DG2_Controller_FTP:
    def __init__(
        self,
        left_hand_array,
        right_hand_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
        fps=100.0,
        Unit_Test=False,
        simulation_mode=False,
        network_interface=None,
        input_timestamp_value=None,
        log_throttle_s=1.0,
        fast_mode=False,
        retarget_mode=None,
    ):
        logger_mp.info("Initialize RH5DG2_Controller_FTP...")

        from inspire_sdkpy import inspire_dds
        import inspire_sdkpy.inspire_hand_defaut as inspire_hand_default

        self.inspire_dds = inspire_dds
        self.inspire_hand_default = inspire_hand_default
        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode
        self.network_interface = network_interface
        self.input_timestamp_value = input_timestamp_value
        self.log_throttle_s = max(float(log_throttle_s), 0.0)
        self.fast_mode = bool(fast_mode)
        self._last_debug_ts = 0.0
        self._loop_rate_start_ts = time.time()
        self._publish_rate_start_ts = time.time()
        self.dds_domain_id = 1 if simulation_mode else 0
        ChannelFactoryInitialize(self.dds_domain_id, networkInterface=self.network_interface)
        self.hand_retargeting = _RH5DG2Retargeting(fast_mode=self.fast_mode, retarget_mode=retarget_mode)
        self.retarget_mode = self.hand_retargeting.retarget_mode
        self.left_state_ready = False
        self.right_state_ready = False

        self.LeftHandCmd_publisher = ChannelPublisher(
            kTopicRH5DG2FTPLeftCommand, self.inspire_dds.inspire_hand_ctrl
        )
        self.LeftHandCmd_publisher.Init()
        self.RightHandCmd_publisher = ChannelPublisher(
            kTopicRH5DG2FTPRightCommand, self.inspire_dds.inspire_hand_ctrl
        )
        self.RightHandCmd_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber(
            kTopicRH5DG2FTPLeftState, self.inspire_dds.inspire_hand_state
        )
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(
            kTopicRH5DG2FTPRightState, self.inspire_dds.inspire_hand_state
        )
        self.RightHandState_subscriber.Init()

        self.left_hand_state_array = Array("d", RH5DG2_Num_Motors, lock=True)
        self.right_hand_state_array = Array("d", RH5DG2_Num_Motors, lock=True)

        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state, daemon=True)
        self.subscribe_state_thread.start()

        wait_count = 0
        while not (self.left_state_ready and self.right_state_ready):
            if wait_count % 100 == 0:
                logger_mp.info("[RH5DG2_Controller_FTP] Waiting to subscribe DDS hand states...")
            time.sleep(0.01)
            wait_count += 1
            if wait_count > 500:
                logger_mp.warning("[RH5DG2_Controller_FTP] Timeout waiting for initial hand states. Proceeding anyway.")
                break
        logger_mp.info("[RH5DG2_Controller_FTP] Initial hand states received or timeout.")

        hand_control_process = Process(
            target=self.control_process,
            args=(
                left_hand_array,
                right_hand_array,
                self.left_hand_state_array,
                self.right_hand_state_array,
                dual_hand_data_lock,
                dual_hand_state_array,
                dual_hand_action_array,
                self.input_timestamp_value,
            ),
        )
        hand_control_process.daemon = True
        hand_control_process.start()

        logger_mp.info("Initialize RH5DG2_Controller_FTP OK!")

    def _subscribe_hand_state(self):
        while True:
            left_state_msg = self.LeftHandState_subscriber.Read()
            if left_state_msg is not None and hasattr(left_state_msg, "angle_act"):
                if len(left_state_msg.angle_act) >= RH5DG2_Num_Motors:
                    with self.left_hand_state_array.get_lock():
                        for idx in range(RH5DG2_Num_Motors):
                            self.left_hand_state_array[idx] = left_state_msg.angle_act[idx] / 1000.0
                    self.left_state_ready = True

            right_state_msg = self.RightHandState_subscriber.Read()
            if right_state_msg is not None and hasattr(right_state_msg, "angle_act"):
                if len(right_state_msg.angle_act) >= RH5DG2_Num_Motors:
                    with self.right_hand_state_array.get_lock():
                        for idx in range(RH5DG2_Num_Motors):
                            self.right_hand_state_array[idx] = right_state_msg.angle_act[idx] / 1000.0
                    self.right_state_ready = True

            time.sleep(0.002)

    def _send_hand_command(self, left_angle_cmd_scaled, right_angle_cmd_scaled):
        publish_start = time.time()
        left_cmd_msg = self.inspire_hand_default.get_inspire_hand_ctrl()
        left_cmd_msg.angle_set = left_angle_cmd_scaled
        left_cmd_msg.mode = 0b0001
        left_write_ok = self.LeftHandCmd_publisher.Write(left_cmd_msg)

        right_cmd_msg = self.inspire_hand_default.get_inspire_hand_ctrl()
        right_cmd_msg.angle_set = right_angle_cmd_scaled
        right_cmd_msg.mode = 0b0001
        right_write_ok = self.RightHandCmd_publisher.Write(right_cmd_msg)
        publish_done = time.time()
        publish_latency_ms = (publish_done - publish_start) * 1000.0
        if not hasattr(self, "_publish_debug_count"):
            self._publish_debug_count = 0
        self._publish_debug_count += 1
        if self._publish_debug_count <= 5 or (self.log_throttle_s > 0 and publish_start - self._last_debug_ts >= self.log_throttle_s):
            print(
                f"[RH5DG2 teleop publish] topics=({kTopicRH5DG2FTPLeftCommand},{kTopicRH5DG2FTPRightCommand}) "
                f"domain={self.dds_domain_id} left_write_ok={left_write_ok} right_write_ok={right_write_ok} "
                f"left={_fmt_debug(left_angle_cmd_scaled)} right={_fmt_debug(right_angle_cmd_scaled)}"
            )
            print(
                f"[RH5DG2 publish hz] hz={_rate_hz(self._publish_debug_count, self._publish_rate_start_ts):.2f} "
                f"count={self._publish_debug_count}"
            )
            print(f"[RH5DG2 publish latency ms] latency_ms={publish_latency_ms:.3f}")
            self._last_debug_ts = publish_start
        return publish_done, (left_write_ok, right_write_ok), publish_latency_ms

    def _retarget(self, left_hand_data, right_hand_data):
        # [핵심 수정] DexPilot 모드(상대 벡터)와 Position 모드(절대 좌표) 자동 호환 처리
        left_idx = np.array(self.hand_retargeting.left_indices)
        right_idx = np.array(self.hand_retargeting.right_indices)

        if not hasattr(self, "_retarget_input_debug_count"):
            self._retarget_input_debug_count = 0
        self._retarget_input_debug_count += 1
        should_debug = self._retarget_input_debug_count % 50 == 0
        if should_debug:
            print(
                "[RH5DG2 teleop retarget input] "
                f"left={_fmt_hand_input(left_hand_data, left_idx)} "
                f"right={_fmt_hand_input(right_hand_data, right_idx)}"
            )

        if left_idx.ndim == 2:
            # Vector/DexPilot modes expect task-space vectors in the RH5DG2 URDF hand frame.
            ref_left_value, left_source, left_target, left_input_vectors = _prepare_vector_reference(
                left_hand_data, left_idx
            )
            ref_right_value, right_source, right_target, right_input_vectors = _prepare_vector_reference(
                right_hand_data, right_idx
            )
            if should_debug:
                print(
                    "[RH5DG2 teleop vector frame] "
                    f"left_source={_fmt_matrix(left_source)} left_target={_fmt_matrix(left_target)} "
                    f"left_input_vectors={_fmt_matrix(left_input_vectors)} "
                    f"left_ref_rh5dg2={_fmt_matrix(ref_left_value)} "
                    f"right_source={_fmt_matrix(right_source)} right_target={_fmt_matrix(right_target)} "
                    f"right_input_vectors={_fmt_matrix(right_input_vectors)} "
                    f"right_ref_rh5dg2={_fmt_matrix(ref_right_value)}"
                )
        else:
            # Position mode expects fingertip points in the RH5DG2 URDF hand frame.
            (
                ref_left_value,
                left_wrist,
                left_palm,
                left_selected,
                left_wrist_rel,
                left_palm_rel,
            ) = _prepare_position_reference(left_hand_data, left_idx)
            (
                ref_right_value,
                right_wrist,
                right_palm,
                right_selected,
                right_wrist_rel,
                right_palm_rel,
            ) = _prepare_position_reference(right_hand_data, right_idx)
            if should_debug:
                print(
                    "[RH5DG2 teleop position frame] "
                    f"left_wrist={_fmt_matrix(left_wrist)} left_palm={_fmt_matrix(left_palm)} "
                    f"left_selected={_fmt_matrix(left_selected)} "
                    f"left_wrist_rel={_fmt_matrix(left_wrist_rel)} "
                    f"left_palm_rel={_fmt_matrix(left_palm_rel)} "
                    f"left_ref_rh5dg2={_fmt_matrix(ref_left_value)} "
                    f"right_wrist={_fmt_matrix(right_wrist)} right_palm={_fmt_matrix(right_palm)} "
                    f"right_selected={_fmt_matrix(right_selected)} "
                    f"right_wrist_rel={_fmt_matrix(right_wrist_rel)} "
                    f"right_palm_rel={_fmt_matrix(right_palm_rel)} "
                    f"right_ref_rh5dg2={_fmt_matrix(ref_right_value)}"
                )

        # 계산된 ref_value를 바탕으로 역기구학(IK) 수행
        left_q_target = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.left_retargeting_to_hardware]
        right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[self.hand_retargeting.right_retargeting_to_hardware]
        left_q_raw = np.asarray(left_q_target, dtype=np.float64).copy()
        right_q_raw = np.asarray(right_q_target, dtype=np.float64).copy()

        if should_debug:
            print(
                f"[RH5DG2 teleop retarget raw] left={_fmt_debug(left_q_raw)} right={_fmt_debug(right_q_raw)} "
                f"left_named={_fmt_named_values(self.hand_retargeting.left_joint_names, left_q_raw)} "
                f"right_named={_fmt_named_values(self.hand_retargeting.right_joint_names, right_q_raw)}"
            )

        # 시뮬레이션 모드 정규화(Normalization) 적용 (이전에 적용했던 수정사항)
        if self.simulation_mode:
            left_q_target, left_debug = _postprocess_sim_command(
                left_q_target,
                self.hand_retargeting.left_joint_limits,
                self.hand_retargeting.left_joint_names,
                left_hand_data,
                side="left",
            )
            right_q_target, right_debug = _postprocess_sim_command(
                right_q_target,
                self.hand_retargeting.right_joint_limits,
                self.hand_retargeting.right_joint_names,
                right_hand_data,
                side="right",
            )
            if should_debug:
                print(
                    "[RH5DG2 teleop normalize detail] "
                    f"left_limits={_fmt_named_limits(self.hand_retargeting.left_joint_names, self.hand_retargeting.left_joint_limits)} "
                    f"right_limits={_fmt_named_limits(self.hand_retargeting.right_joint_names, self.hand_retargeting.right_joint_limits)} "
                    f"left_unclamped={_fmt_named_values(self.hand_retargeting.left_joint_names, left_debug['unclamped'])} "
                    f"right_unclamped={_fmt_named_values(self.hand_retargeting.right_joint_names, right_debug['unclamped'])} "
                    f"left_clamped={_fmt_named_values(self.hand_retargeting.left_joint_names, left_debug['clamped'])} "
                    f"right_clamped={_fmt_named_values(self.hand_retargeting.right_joint_names, right_debug['clamped'])} "
                    f"left_gain={_fmt_named_values(self.hand_retargeting.left_joint_names, left_debug['gain'])} "
                    f"right_gain={_fmt_named_values(self.hand_retargeting.right_joint_names, right_debug['gain'])} "
                    f"left_open_prior={_fmt_named_values(self.hand_retargeting.left_joint_names, left_debug['prior'])} "
                    f"right_open_prior={_fmt_named_values(self.hand_retargeting.right_joint_names, right_debug['prior'])} "
                    f"open_calibration_enabled={_env_flag('RH5DG2_ENABLE_OPEN_CALIBRATION')} "
                    f"teleop_safe_close_enabled={_env_flag('RH5DG2_ENABLE_TELEOP_SAFE_CLOSE')} "
                    f"left_calibrated={_fmt_named_values(self.hand_retargeting.left_joint_names, left_debug['calibrated'])} "
                    f"right_calibrated={_fmt_named_values(self.hand_retargeting.right_joint_names, right_debug['calibrated'])} "
                    f"left_safe_floor={_fmt_named_values(self.hand_retargeting.left_joint_names, left_q_target)} "
                    f"right_safe_floor={_fmt_named_values(self.hand_retargeting.right_joint_names, right_q_target)}"
                )
                print(
                    "[RH5DG2 teleop open prior] "
                    f"left_shape={_fmt_shape_scores(left_debug['prior_debug']['scores'])} "
                    f"right_shape={_fmt_shape_scores(right_debug['prior_debug']['scores'])} "
                    f"left_prior_delta={_fmt_named_values(self.hand_retargeting.left_joint_names, left_debug['prior_debug']['delta'])} "
                    f"right_prior_delta={_fmt_named_values(self.hand_retargeting.right_joint_names, right_debug['prior_debug']['delta'])} "
                    f"left_v_spread={left_debug['prior_debug']['v_spread']} "
                    f"right_v_spread={right_debug['prior_debug']['v_spread']} "
                    f"strength={left_debug['prior_debug']['strength']:.3f} "
                    f"threshold={left_debug['prior_debug']['threshold']:.3f}"
                )
                print(
                    "[RH5DG2 teleop curl command] "
                    f"left_curl_scores={_fmt_curl_scores(left_debug['curl_debug']['scores'])} "
                    f"right_curl_scores={_fmt_curl_scores(right_debug['curl_debug']['scores'])} "
                    f"left_delta={left_debug['finger_command_delta']} "
                    f"right_delta={right_debug['finger_command_delta']} "
                    f"left_decouple_delta={_fmt_named_values(self.hand_retargeting.left_joint_names, left_debug['curl_debug']['decouple_delta'])} "
                    f"right_decouple_delta={_fmt_named_values(self.hand_retargeting.right_joint_names, right_debug['curl_debug']['decouple_delta'])} "
                    f"left_targets={_fmt_named_values(self.hand_retargeting.left_joint_names, left_debug['curl_debug']['targets'])} "
                    f"right_targets={_fmt_named_values(self.hand_retargeting.right_joint_names, right_debug['curl_debug']['targets'])} "
                    f"decouple_enabled={left_debug['curl_debug']['decouple_enabled']} "
                    f"decouple_strength={left_debug['curl_debug']['decouple_strength']:.3f} "
                    f"decouple_threshold={left_debug['curl_debug']['decouple_threshold']:.3f} "
                    f"strength={left_debug['curl_debug']['strength']:.3f} "
                    f"threshold={left_debug['curl_debug']['threshold']:.3f}"
                )
                print(
                    "[RH5DG2 teleop command mapping] "
                    f"left_mapping={left_debug['mapping']} "
                    f"right_mapping={right_debug['mapping']} "
                    f"left_denorm_rad={_fmt_named_values(self.hand_retargeting.left_joint_names, left_debug['denorm_rad'])} "
                    f"right_denorm_rad={_fmt_named_values(self.hand_retargeting.right_joint_names, right_debug['denorm_rad'])} "
                    f"left_saturation={left_debug['saturation']} "
                    f"right_saturation={right_debug['saturation']}"
                )
                print(
                    "[RH5DG2 teleop calibrated] "
                    f"left_scores={_fmt_finger_scores(left_debug['finger_scores'])} "
                    f"right_scores={_fmt_finger_scores(right_debug['finger_scores'])} "
                    f"left={_fmt_named_values(self.hand_retargeting.left_joint_names, left_q_target)} "
                    f"right={_fmt_named_values(self.hand_retargeting.right_joint_names, right_q_target)}"
                )
        else:
            left_q_target = _clip_to_joint_limits(left_q_target, self.hand_retargeting.left_joint_limits)
            right_q_target = _clip_to_joint_limits(right_q_target, self.hand_retargeting.right_joint_limits)
            if should_debug:
                print(
                    "[RH5DG2 teleop clip detail] "
                    f"left_limits={_fmt_named_limits(self.hand_retargeting.left_joint_names, self.hand_retargeting.left_joint_limits)} "
                    f"right_limits={_fmt_named_limits(self.hand_retargeting.right_joint_names, self.hand_retargeting.right_joint_limits)} "
                    f"left_clipped={_fmt_named_values(self.hand_retargeting.left_joint_names, left_q_target)} "
                    f"right_clipped={_fmt_named_values(self.hand_retargeting.right_joint_names, right_q_target)}"
                )

        if should_debug:
            print(
                f"[RH5DG2 teleop retarget] sim={self.simulation_mode} "
                f"left={_fmt_debug(left_q_target)} right={_fmt_debug(right_q_target)}"
            )

        return left_q_target, right_q_target

    def control_process(
        self,
        left_hand_array,
        right_hand_array,
        left_hand_state_array,
        right_hand_state_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
        input_timestamp_value=None,
    ):
        logger_mp.info("[RH5DG2_Controller_FTP] Control process started.")
        self.running = True
        _reset_channel_factory_after_fork()
        ChannelFactoryInitialize(self.dds_domain_id, networkInterface=self.network_interface)
        self.LeftHandCmd_publisher = ChannelPublisher(
            kTopicRH5DG2FTPLeftCommand, self.inspire_dds.inspire_hand_ctrl
        )
        self.LeftHandCmd_publisher.Init()
        self.RightHandCmd_publisher = ChannelPublisher(
            kTopicRH5DG2FTPRightCommand, self.inspire_dds.inspire_hand_ctrl
        )
        self.RightHandCmd_publisher.Init()
        logger_mp.info(
            f"[RH5DG2_Controller_FTP] Child DDS publishers initialized "
            f"topics=({kTopicRH5DG2FTPLeftCommand},{kTopicRH5DG2FTPRightCommand}) "
            f"domain={self.dds_domain_id} network_interface={self.network_interface} "
            f"left_status={_publisher_debug_status(self.LeftHandCmd_publisher)} "
            f"right_status={_publisher_debug_status(self.RightHandCmd_publisher)}"
        )

        left_q_target = np.full(RH5DG2_Num_Motors, 1.0)
        right_q_target = np.full(RH5DG2_Num_Motors, 1.0)

        try:
            loop_count = 0
            while self.running:
                loop_count += 1
                start_time = time.time()
                with left_hand_array.get_lock():
                    left_hand_data = np.array(left_hand_array[:]).reshape(25, 3).copy()
                with right_hand_array.get_lock():
                    right_hand_data = np.array(right_hand_array[:]).reshape(25, 3).copy()
                input_timestamp = 0.0
                if input_timestamp_value is not None:
                    with input_timestamp_value.get_lock():
                        input_timestamp = float(input_timestamp_value.value)

                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))

                if self.simulation_mode:
                    hand_ready = _is_legacy_sim_hand_tracking_ready(left_hand_data, right_hand_data)
                else:
                    hand_ready = _is_hand_tracking_ready(left_hand_data, right_hand_data)
                should_debug = (
                    loop_count <= 5
                    or (self.log_throttle_s > 0 and start_time - self._last_debug_ts >= self.log_throttle_s)
                )
                retarget_start = time.time()
                if hand_ready:
                    left_q_target, right_q_target = self._retarget(left_hand_data, right_hand_data)
                    retarget_done = time.time()
                    retarget_latency_ms = (retarget_done - retarget_start) * 1000.0
                    if should_debug:
                        print(
                            f"[RH5DG2 FTP control after retarget] ready={hand_ready} "
                            f"left={_fmt_debug(left_q_target)} right={_fmt_debug(right_q_target)}"
                        )
                        print(f"[RH5DG2 retarget latency ms] latency_ms={retarget_latency_ms:.3f}")
                elif should_debug:
                    retarget_done = time.time()
                    retarget_latency_ms = 0.0
                    print(
                        f"[RH5DG2 FTP control no retarget] ready={hand_ready} "
                        f"right_hand_zero={np.allclose(right_hand_data, 0.0, atol=1e-5)} "
                        f"left_sentinel={np.allclose(left_hand_data[4], np.array([-1.13, 0.3, 0.15]), atol=1e-3)} "
                        f"left={_fmt_debug(left_q_target)} right={_fmt_debug(right_q_target)}"
                    )

                scaled_left_cmd = [int(np.clip(value * 1000.0, 0.0, 1000.0)) for value in left_q_target]
                scaled_right_cmd = [int(np.clip(value * 1000.0, 0.0, 1000.0)) for value in right_q_target]

                action_data = np.concatenate((left_q_target, right_q_target))
                if dual_hand_state_array is not None and dual_hand_action_array is not None:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                if should_debug:
                    print(
                        f"[RH5DG2 FTP control pre publish] ready={hand_ready} "
                        f"left={_fmt_debug(left_q_target)} right={_fmt_debug(right_q_target)} "
                        f"scaled_left={scaled_left_cmd[:5]} scaled_right={scaled_right_cmd[:5]}"
                    )
                publish_done, write_ok, publish_latency_ms = self._send_hand_command(scaled_left_cmd, scaled_right_cmd)
                time_elapsed = publish_done - start_time
                sleep_s = max(0.0, (1.0 / self.fps) - time_elapsed)
                if should_debug:
                    print(f"[RH5DG2 retarget hz] hz={_rate_hz(loop_count, self._loop_rate_start_ts):.2f} count={loop_count}")
                    print(f"[RH5DG2 control loop sleep] sleep_ms={sleep_s * 1000.0:.3f} fps_target={self.fps}")
                    if input_timestamp > 0.0:
                        print(
                            f"[RH5DG2 latency breakdown] fast_mode={self.fast_mode} ready={hand_ready} "
                            f"input_timestamp={input_timestamp:.6f} loop_start={start_time:.6f} "
                            f"retarget_start={retarget_start:.6f} retarget_done={retarget_done:.6f} "
                            f"publish_done={publish_done:.6f} write_ok={write_ok} "
                            f"input_to_loop_ms={(start_time - input_timestamp) * 1000.0:.2f} "
                            f"loop_to_retarget_start_ms={(retarget_start - start_time) * 1000.0:.2f} "
                            f"retarget_ms={retarget_latency_ms:.2f} publish_ms={publish_latency_ms:.2f} "
                            f"input_to_publish_ms={(publish_done - input_timestamp) * 1000.0:.2f} "
                            f"sleep_ms={sleep_s * 1000.0:.2f}"
                        )
                    self._last_debug_ts = start_time
                time.sleep(sleep_s)
        finally:
            logger_mp.info("RH5DG2_Controller_FTP has been closed.")


# Backward-compatible class names.
Inspire_Controller_DFX = RH5DG2_Controller_DFX
Inspire_Controller_FTP = RH5DG2_Controller_FTP


__all__ = [
    "RH5DG2_Controller_DFX",
    "RH5DG2_Controller_FTP",
    "RH5DG2_Left_Hand_JointIndex",
    "RH5DG2_Num_Motors",
    "RH5DG2_Right_Hand_JointIndex",
    "Inspire_Controller_DFX",
    "Inspire_Controller_FTP",
    "Inspire_Left_Hand_JointIndex",
    "Inspire_Num_Motors",
    "Inspire_Right_Hand_JointIndex",
    "kTopicRH5DG2DFXCommand",
    "kTopicRH5DG2DFXState",
    "kTopicRH5DG2FTPLeftCommand",
    "kTopicRH5DG2FTPLeftState",
    "kTopicRH5DG2FTPRightCommand",
    "kTopicRH5DG2FTPRightState",
    "kTopicInspireDFXCommand",
    "kTopicInspireDFXState",
    "kTopicInspireFTPLeftCommand",
    "kTopicInspireFTPLeftState",
    "kTopicInspireFTPRightCommand",
    "kTopicInspireFTPRightState",
]
