from enum import IntEnum
from multiprocessing import Array, Process
import os
from pathlib import Path
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
import yaml

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
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ASSETS_ROOT = _REPO_ROOT / "assets"
_RH5DG2_ASSET_DIR = _ASSETS_ROOT / "RH5DG2"
_RH5DG2_CONFIG_PATH = _RH5DG2_ASSET_DIR / "RH5DG2.yml"
_RH5DG2_URDF_CACHE_DIR = Path("/tmp/opencode/rh5dg2_urdf")

kTopicRH5DG2DFXCommand = "rt/rh5dg2/cmd"
kTopicRH5DG2DFXState = "rt/rh5dg2/state"
kTopicRH5DG2FTPLeftCommand = "rt/rh5dg2_hand/ctrl/l"
kTopicRH5DG2FTPRightCommand = "rt/rh5dg2_hand/ctrl/r"
kTopicRH5DG2FTPLeftState = "rt/rh5dg2_hand/state/l"
kTopicRH5DG2FTPRightState = "rt/rh5dg2_hand/state/r"


def _env_flag(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


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
        rows.append(
            f"i={idx} q={getattr(motor, 'q', None)} dq={getattr(motor, 'dq', None)} "
            f"tau={getattr(motor, 'tau', None)} kp={getattr(motor, 'kp', None)} "
            f"kd={getattr(motor, 'kd', None)} mode={getattr(motor, 'mode', None)}"
        )
    return " | ".join(rows)


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


def _fmt_named_limits(names, joint_limits):
    rows = []
    for name, (lower, upper) in zip(names, joint_limits):
        rows.append(f"{name}=[{lower:.4f},{upper:.4f}] range={upper - lower:.4f}")
    return rows


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
    # 수정됨: np.allclose를 사용하여 미세한 소수점 오차 무시
    return (
        not np.allclose(right_hand_data, 0.0, atol=1e-5)
        and not np.allclose(left_hand_data[4], np.array([-1.13, 0.3, 0.15]), atol=1e-3)
    )


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


def _fmt_finger_scores(scores):
    return [
        (
            f"{finger}:dist={data['distance']:.4f},"
            f"raw={data['raw']:.4f},cal={data['calibrated']:.4f}"
        )
        for finger, data in scores.items()
    ]


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

def _shape_grasp(values):
    # Bias mid-range values toward closure so the robot grabs earlier.
    return np.clip(np.power(np.asarray(values, dtype=np.float64), RH5DG2_GRASP_SHARPNESS), 0.0, 1.0)


class _RH5DG2Retargeting:
    def __init__(self):
        RetargetingConfig.set_default_urdf_dir(str(_ASSETS_ROOT))
        cfg = self._load_config()

        self.left_retargeting = RetargetingConfig.from_dict(cfg["left"]).build()
        self.right_retargeting = RetargetingConfig.from_dict(cfg["right"]).build()

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

        self.left_joint_limits = self._load_joint_limits(
            Path(cfg["left"]["urdf_path"]), self.left_joint_names
        )
        self.right_joint_limits = self._load_joint_limits(
            Path(cfg["right"]["urdf_path"]), self.right_joint_names
        )
        self._set_open_initial_qpos(self.left_retargeting, self.left_joint_names, self.left_joint_limits)
        self._set_open_initial_qpos(self.right_retargeting, self.right_joint_names, self.right_joint_limits)

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
    ):
        logger_mp.info("Initialize RH5DG2_Controller_DFX...")

        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode
        self.network_interface = network_interface
        self.input_timestamp_value = input_timestamp_value
        self.log_throttle_s = max(float(log_throttle_s), 0.0)
        self._last_debug_ts = 0.0
        self._loop_rate_start_ts = time.time()
        self._publish_rate_start_ts = time.time()
        self.dds_domain_id = 1 if simulation_mode else 0
        self.hand_retargeting = _RH5DG2Retargeting()
        self.left_state_ready = False
        self.right_state_ready = False

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
        hand_control_process.start()

        logger_mp.info("Initialize RH5DG2_Controller_DFX OK!")

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

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        publish_start = time.time()
        # 수정됨: 0~12 인덱스(오른손)에 right_q_target을 할당, 13~25에 left_q_target 할당
        for idx in range(RH5DG2_Num_Motors):
            self.hand_msg.cmds[idx].q = right_q_target[idx]
        for idx in range(RH5DG2_Num_Motors):
            self.hand_msg.cmds[RH5DG2_Num_Motors + idx].q = left_q_target[idx]
        for cmd in self.hand_msg.cmds:
            cmd.dq = 0.0
            cmd.tau = 0.0
            cmd.kp = 1.0
            cmd.kd = 0.05
            cmd.mode = 0b0001
        if not hasattr(self, "_publish_debug_count"):
            self._publish_debug_count = 0
        self._publish_debug_count += 1
        should_debug = (
            self._publish_debug_count <= 5
            or (self.log_throttle_s > 0 and publish_start - self._last_debug_ts >= self.log_throttle_s)
        )
        write_ok = self.HandCmd_publisher.Write(self.hand_msg)
        publish_latency_ms = (time.time() - publish_start) * 1000.0
        if should_debug:
            payload = np.asarray([cmd.q for cmd in self.hand_msg.cmds], dtype=np.float64)
            print(
                f"[RH5DG2 teleop publish payload] topic={kTopicRH5DG2DFXCommand} domain={self.dds_domain_id} "
                f"write_ok={write_ok} "
                f"{_publisher_debug_status(self.HandCmd_publisher)} "
                f"len={payload.size} finite={np.isfinite(payload).all()} "
                f"min={payload.min():.4f} max={payload.max():.4f} "
                f"right0_12={np.round(payload[:RH5DG2_Num_Motors], 4).tolist()} "
                f"left13_25={np.round(payload[RH5DG2_Num_Motors:], 4).tolist()} "
                f"left={_fmt_debug(left_q_target)} right={_fmt_debug(right_q_target)}"
            )
            print(
                f"[RH5DG2 publish hz] hz={_rate_hz(self._publish_debug_count, self._publish_rate_start_ts):.2f} "
                f"count={self._publish_debug_count}"
            )
            print(f"[RH5DG2 publish latency ms] latency_ms={publish_latency_ms:.3f}")
            print(f"[RH5DG2 teleop publish fields right] {_fmt_motor_fields(self.hand_msg.cmds, 0, 5)}")
            print(f"[RH5DG2 teleop publish fields left] {_fmt_motor_fields(self.hand_msg.cmds, RH5DG2_Num_Motors, 5)}")
            self._last_debug_ts = publish_start

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
            left_q_clamped, left_q_unclamped = _normalize_to_unit_interval(
                left_q_target, self.hand_retargeting.left_joint_limits, return_unclamped=True
            )
            right_q_clamped, right_q_unclamped = _normalize_to_unit_interval(
                right_q_target, self.hand_retargeting.right_joint_limits, return_unclamped=True
            )
            left_q_gain = _apply_teleop_close_gain(left_q_clamped)
            right_q_gain = _apply_teleop_close_gain(right_q_clamped)
            left_q_calibrated, left_finger_scores = _apply_finger_open_calibration(left_q_gain, left_hand_data)
            right_q_calibrated, right_finger_scores = _apply_finger_open_calibration(right_q_gain, right_hand_data)
            left_q_target = _apply_safe_close_floor(left_q_calibrated)
            right_q_target = _apply_safe_close_floor(right_q_calibrated)
            if should_debug:
                print(
                    "[RH5DG2 teleop normalize detail] "
                    f"left_limits={_fmt_named_limits(self.hand_retargeting.left_joint_names, self.hand_retargeting.left_joint_limits)} "
                    f"right_limits={_fmt_named_limits(self.hand_retargeting.right_joint_names, self.hand_retargeting.right_joint_limits)} "
                    f"left_unclamped={_fmt_named_values(self.hand_retargeting.left_joint_names, left_q_unclamped)} "
                    f"right_unclamped={_fmt_named_values(self.hand_retargeting.right_joint_names, right_q_unclamped)} "
                    f"left_clamped={_fmt_named_values(self.hand_retargeting.left_joint_names, left_q_clamped)} "
                    f"right_clamped={_fmt_named_values(self.hand_retargeting.right_joint_names, right_q_clamped)} "
                    f"left_gain={_fmt_named_values(self.hand_retargeting.left_joint_names, left_q_gain)} "
                    f"right_gain={_fmt_named_values(self.hand_retargeting.right_joint_names, right_q_gain)} "
                    f"open_calibration_enabled={_env_flag('RH5DG2_ENABLE_OPEN_CALIBRATION')} "
                    f"teleop_safe_close_enabled={_env_flag('RH5DG2_ENABLE_TELEOP_SAFE_CLOSE')} "
                    f"left_calibrated={_fmt_named_values(self.hand_retargeting.left_joint_names, left_q_calibrated)} "
                    f"right_calibrated={_fmt_named_values(self.hand_retargeting.right_joint_names, right_q_calibrated)} "
                    f"left_safe_floor={_fmt_named_values(self.hand_retargeting.left_joint_names, left_q_target)} "
                    f"right_safe_floor={_fmt_named_values(self.hand_retargeting.right_joint_names, right_q_target)}"
                )
                print(
                    "[RH5DG2 teleop calibrated] "
                    f"left_scores={_fmt_finger_scores(left_finger_scores)} "
                    f"right_scores={_fmt_finger_scores(right_finger_scores)} "
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

        self.hand_msg = MotorCmds_()
        self.hand_msg.cmds = [
            unitree_go_msg_dds__MotorCmd_()
            for _ in range(len(RH5DG2_Left_Hand_JointIndex) + len(RH5DG2_Right_Hand_JointIndex))
        ]
        for idx in range(RH5DG2_Num_Motors):
            self.hand_msg.cmds[idx].q = 1.0
        for idx in range(RH5DG2_Num_Motors):
            self.hand_msg.cmds[RH5DG2_Num_Motors + idx].q = 1.0

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

                hand_ready = _is_hand_tracking_ready(left_hand_data, right_hand_data)
                should_debug = (
                    loop_count <= 5
                    or (self.log_throttle_s > 0 and start_time - self._last_debug_ts >= self.log_throttle_s)
                )
                retarget_start = time.time()
                if hand_ready:
                    left_q_target, right_q_target = self._retarget(left_hand_data, right_hand_data)
                    retarget_latency_ms = (time.time() - retarget_start) * 1000.0
                    if should_debug:
                        print(
                            f"[RH5DG2 DFX control after retarget] ready={hand_ready} "
                            f"left={_fmt_debug(left_q_target)} right={_fmt_debug(right_q_target)}"
                        )
                        print(f"[RH5DG2 retarget latency ms] latency_ms={retarget_latency_ms:.3f}")
                elif should_debug:
                    retarget_latency_ms = 0.0
                    print(
                        f"[RH5DG2 DFX control no retarget] ready={hand_ready} "
                        f"right_hand_zero={np.allclose(right_hand_data, 0.0, atol=1e-5)} "
                        f"left_sentinel={np.allclose(left_hand_data[4], np.array([-1.13, 0.3, 0.15]), atol=1e-3)} "
                        f"left={_fmt_debug(left_q_target)} right={_fmt_debug(right_q_target)}"
                    )
                action_data = np.concatenate((left_q_target, right_q_target))
                if dual_hand_state_array is not None and dual_hand_action_array is not None:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                if should_debug:
                    print(
                        f"[RH5DG2 DFX control pre publish] ready={hand_ready} "
                        f"left={_fmt_debug(left_q_target)} right={_fmt_debug(right_q_target)} "
                        f"action={_fmt_debug(action_data)}"
                    )
                self.ctrl_dual_hand(left_q_target, right_q_target)
                time_elapsed = time.time() - start_time
                sleep_s = max(0.0, (1.0 / self.fps) - time_elapsed)
                if should_debug:
                    print(f"[RH5DG2 retarget hz] hz={_rate_hz(loop_count, self._loop_rate_start_ts):.2f} count={loop_count}")
                    print(f"[RH5DG2 control loop sleep] sleep_ms={sleep_s * 1000.0:.3f} fps_target={self.fps}")
                    if input_timestamp > 0.0:
                        print(
                            f"[RH5DG2 end-to-end latency] "
                            f"input_timestamp={input_timestamp:.6f} publish_timestamp={time.time():.6f} "
                            f"latency_ms={(time.time() - input_timestamp) * 1000.0:.2f}"
                        )
                    self._last_debug_ts = start_time
                time.sleep(sleep_s)
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
        self._last_debug_ts = 0.0
        self._loop_rate_start_ts = time.time()
        self._publish_rate_start_ts = time.time()
        self.dds_domain_id = 1 if simulation_mode else 0
        ChannelFactoryInitialize(self.dds_domain_id, networkInterface=self.network_interface)
        self.hand_retargeting = _RH5DG2Retargeting()
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
            print(f"[RH5DG2 publish latency ms] latency_ms={(time.time() - publish_start) * 1000.0:.3f}")
            self._last_debug_ts = publish_start

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
            left_q_clamped, left_q_unclamped = _normalize_to_unit_interval(
                left_q_target, self.hand_retargeting.left_joint_limits, return_unclamped=True
            )
            right_q_clamped, right_q_unclamped = _normalize_to_unit_interval(
                right_q_target, self.hand_retargeting.right_joint_limits, return_unclamped=True
            )
            left_q_gain = _apply_teleop_close_gain(left_q_clamped)
            right_q_gain = _apply_teleop_close_gain(right_q_clamped)
            left_q_calibrated, left_finger_scores = _apply_finger_open_calibration(left_q_gain, left_hand_data)
            right_q_calibrated, right_finger_scores = _apply_finger_open_calibration(right_q_gain, right_hand_data)
            left_q_target = _apply_safe_close_floor(left_q_calibrated)
            right_q_target = _apply_safe_close_floor(right_q_calibrated)
            if should_debug:
                print(
                    "[RH5DG2 teleop normalize detail] "
                    f"left_limits={_fmt_named_limits(self.hand_retargeting.left_joint_names, self.hand_retargeting.left_joint_limits)} "
                    f"right_limits={_fmt_named_limits(self.hand_retargeting.right_joint_names, self.hand_retargeting.right_joint_limits)} "
                    f"left_unclamped={_fmt_named_values(self.hand_retargeting.left_joint_names, left_q_unclamped)} "
                    f"right_unclamped={_fmt_named_values(self.hand_retargeting.right_joint_names, right_q_unclamped)} "
                    f"left_clamped={_fmt_named_values(self.hand_retargeting.left_joint_names, left_q_clamped)} "
                    f"right_clamped={_fmt_named_values(self.hand_retargeting.right_joint_names, right_q_clamped)} "
                    f"left_gain={_fmt_named_values(self.hand_retargeting.left_joint_names, left_q_gain)} "
                    f"right_gain={_fmt_named_values(self.hand_retargeting.right_joint_names, right_q_gain)} "
                    f"open_calibration_enabled={_env_flag('RH5DG2_ENABLE_OPEN_CALIBRATION')} "
                    f"teleop_safe_close_enabled={_env_flag('RH5DG2_ENABLE_TELEOP_SAFE_CLOSE')} "
                    f"left_calibrated={_fmt_named_values(self.hand_retargeting.left_joint_names, left_q_calibrated)} "
                    f"right_calibrated={_fmt_named_values(self.hand_retargeting.right_joint_names, right_q_calibrated)} "
                    f"left_safe_floor={_fmt_named_values(self.hand_retargeting.left_joint_names, left_q_target)} "
                    f"right_safe_floor={_fmt_named_values(self.hand_retargeting.right_joint_names, right_q_target)}"
                )
                print(
                    "[RH5DG2 teleop calibrated] "
                    f"left_scores={_fmt_finger_scores(left_finger_scores)} "
                    f"right_scores={_fmt_finger_scores(right_finger_scores)} "
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

                hand_ready = _is_hand_tracking_ready(left_hand_data, right_hand_data)
                should_debug = (
                    loop_count <= 5
                    or (self.log_throttle_s > 0 and start_time - self._last_debug_ts >= self.log_throttle_s)
                )
                retarget_start = time.time()
                if hand_ready:
                    left_q_target, right_q_target = self._retarget(left_hand_data, right_hand_data)
                    retarget_latency_ms = (time.time() - retarget_start) * 1000.0
                    if should_debug:
                        print(
                            f"[RH5DG2 FTP control after retarget] ready={hand_ready} "
                            f"left={_fmt_debug(left_q_target)} right={_fmt_debug(right_q_target)}"
                        )
                        print(f"[RH5DG2 retarget latency ms] latency_ms={retarget_latency_ms:.3f}")
                elif should_debug:
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
                self._send_hand_command(scaled_left_cmd, scaled_right_cmd)
                time_elapsed = time.time() - start_time
                sleep_s = max(0.0, (1.0 / self.fps) - time_elapsed)
                if should_debug:
                    print(f"[RH5DG2 retarget hz] hz={_rate_hz(loop_count, self._loop_rate_start_ts):.2f} count={loop_count}")
                    print(f"[RH5DG2 control loop sleep] sleep_ms={sleep_s * 1000.0:.3f} fps_target={self.fps}")
                    if input_timestamp > 0.0:
                        print(
                            f"[RH5DG2 end-to-end latency] "
                            f"input_timestamp={input_timestamp:.6f} publish_timestamp={time.time():.6f} "
                            f"latency_ms={(time.time() - input_timestamp) * 1000.0:.2f}"
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
