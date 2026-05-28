from enum import IntEnum
from multiprocessing import Array, Process
from pathlib import Path
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from dex_retargeting import RetargetingConfig
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

import logging_mp

logger_mp = logging_mp.getLogger(__name__)


RH5DG2_Num_Motors = 13
RH5DG2_GRASP_SHARPNESS = 1.35
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
    return (
        not np.all(right_hand_data == 0.0)
        and not np.all(left_hand_data[4] == np.array([-1.13, 0.3, 0.15]))
    )


def _normalize_to_unit_interval(values, joint_limits):
    normalized = np.empty(len(values), dtype=np.float64)
    for idx, (value, (lower, upper)) in enumerate(zip(values, joint_limits)):
        if np.isclose(upper, lower):
            normalized[idx] = 0.5
            continue
        normalized[idx] = np.clip((upper - value) / (upper - lower), 0.0, 1.0)
    return normalized


def _clip_to_joint_limits(values, joint_limits):
    clipped = np.empty(len(values), dtype=np.float64)
    for idx, (value, (lower, upper)) in enumerate(zip(values, joint_limits)):
        clipped[idx] = np.clip(value, lower, upper)
    return clipped


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
        if "package://RH5DG2_R/meshes/" not in urdf_text:
            return urdf_path

        _RH5DG2_URDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        rewritten_path = _RH5DG2_URDF_CACHE_DIR / urdf_path.name
        rewritten_text = urdf_text.replace("package://RH5DG2_R/meshes/", "package://RH5DG2/meshes/")
        rewritten_path.write_text(rewritten_text, encoding="utf-8")
        logger_mp.warning(
            "[RH5DG2] Rewrote URDF mesh package paths for %s -> %s",
            source_path,
            rewritten_path,
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
    ):
        logger_mp.info("Initialize RH5DG2_Controller_DFX...")

        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode
        self.hand_retargeting = _RH5DG2Retargeting()
        self.left_state_ready = False
        self.right_state_ready = False

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
            ),
        )
        hand_control_process.daemon = True
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
        for idx, joint_id in enumerate(RH5DG2_Left_Hand_JointIndex):
            self.hand_msg.cmds[joint_id].q = left_q_target[idx]
        for idx, joint_id in enumerate(RH5DG2_Right_Hand_JointIndex):
            self.hand_msg.cmds[joint_id].q = right_q_target[idx]
        self.HandCmd_publisher.Write(self.hand_msg)

    def _retarget(self, left_hand_data, right_hand_data):
        ref_left_value = (
            left_hand_data[self.hand_retargeting.left_indices[1, :]]
            - left_hand_data[self.hand_retargeting.left_indices[0, :]]
        )
        ref_right_value = (
            right_hand_data[self.hand_retargeting.right_indices[1, :]]
            - right_hand_data[self.hand_retargeting.right_indices[0, :]]
        )

        left_q_target = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[
            self.hand_retargeting.left_retargeting_to_hardware
        ]
        right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[
            self.hand_retargeting.right_retargeting_to_hardware
        ]

        # DFX uses the DDS motor q directly, so keep targets in joint-angle space.
        left_q_target = _clip_to_joint_limits(left_q_target, self.hand_retargeting.left_joint_limits)
        right_q_target = _clip_to_joint_limits(right_q_target, self.hand_retargeting.right_joint_limits)
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
    ):
        self.running = True
        left_q_target = np.full(RH5DG2_Num_Motors, 1.0)
        right_q_target = np.full(RH5DG2_Num_Motors, 1.0)

        self.hand_msg = MotorCmds_()
        self.hand_msg.cmds = [
            unitree_go_msg_dds__MotorCmd_()
            for _ in range(len(RH5DG2_Left_Hand_JointIndex) + len(RH5DG2_Right_Hand_JointIndex))
        ]
        for joint_id in list(RH5DG2_Left_Hand_JointIndex) + list(RH5DG2_Right_Hand_JointIndex):
            self.hand_msg.cmds[joint_id].q = 1.0

        try:
            while self.running:
                start_time = time.time()
                with left_hand_array.get_lock():
                    left_hand_data = np.array(left_hand_array[:]).reshape(25, 3).copy()
                with right_hand_array.get_lock():
                    right_hand_data = np.array(right_hand_array[:]).reshape(25, 3).copy()

                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))

                if _is_hand_tracking_ready(left_hand_data, right_hand_data):
                    left_q_target, right_q_target = self._retarget(left_hand_data, right_hand_data)

                action_data = np.concatenate((left_q_target, right_q_target))
                if dual_hand_state_array is not None and dual_hand_action_array is not None:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                self.ctrl_dual_hand(left_q_target, right_q_target)
                time_elapsed = time.time() - start_time
                time.sleep(max(0.0, (1.0 / self.fps) - time_elapsed))
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
    ):
        logger_mp.info("Initialize RH5DG2_Controller_FTP...")

        from inspire_sdkpy import inspire_dds
        import inspire_sdkpy.inspire_hand_defaut as inspire_hand_default

        self.inspire_dds = inspire_dds
        self.inspire_hand_default = inspire_hand_default
        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode
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
        left_cmd_msg = self.inspire_hand_default.get_inspire_hand_ctrl()
        left_cmd_msg.angle_set = left_angle_cmd_scaled
        left_cmd_msg.mode = 0b0001
        self.LeftHandCmd_publisher.Write(left_cmd_msg)

        right_cmd_msg = self.inspire_hand_default.get_inspire_hand_ctrl()
        right_cmd_msg.angle_set = right_angle_cmd_scaled
        right_cmd_msg.mode = 0b0001
        self.RightHandCmd_publisher.Write(right_cmd_msg)

    def _retarget(self, left_hand_data, right_hand_data):
        ref_left_value = (
            left_hand_data[self.hand_retargeting.left_indices[1, :]]
            - left_hand_data[self.hand_retargeting.left_indices[0, :]]
        )
        ref_right_value = (
            right_hand_data[self.hand_retargeting.right_indices[1, :]]
            - right_hand_data[self.hand_retargeting.right_indices[0, :]]
        )

        left_q_target = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[
            self.hand_retargeting.left_retargeting_to_hardware
        ]
        right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[
            self.hand_retargeting.right_retargeting_to_hardware
        ]

        left_q_target = _normalize_to_unit_interval(left_q_target, self.hand_retargeting.left_joint_limits)
        right_q_target = _normalize_to_unit_interval(right_q_target, self.hand_retargeting.right_joint_limits)
        left_q_target = _shape_grasp(left_q_target)
        right_q_target = _shape_grasp(right_q_target)
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
    ):
        logger_mp.info("[RH5DG2_Controller_FTP] Control process started.")
        self.running = True

        left_q_target = np.full(RH5DG2_Num_Motors, 1.0)
        right_q_target = np.full(RH5DG2_Num_Motors, 1.0)

        try:
            while self.running:
                start_time = time.time()
                with left_hand_array.get_lock():
                    left_hand_data = np.array(left_hand_array[:]).reshape(25, 3).copy()
                with right_hand_array.get_lock():
                    right_hand_data = np.array(right_hand_array[:]).reshape(25, 3).copy()

                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))

                if _is_hand_tracking_ready(left_hand_data, right_hand_data):
                    left_q_target, right_q_target = self._retarget(left_hand_data, right_hand_data)

                scaled_left_cmd = [int(np.clip(value * 1000.0, 0.0, 1000.0)) for value in left_q_target]
                scaled_right_cmd = [int(np.clip(value * 1000.0, 0.0, 1000.0)) for value in right_q_target]

                action_data = np.concatenate((left_q_target, right_q_target))
                if dual_hand_state_array is not None and dual_hand_action_array is not None:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                self._send_hand_command(scaled_left_cmd, scaled_right_cmd)
                time_elapsed = time.time() - start_time
                time.sleep(max(0.0, (1.0 / self.fps) - time_elapsed))
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
