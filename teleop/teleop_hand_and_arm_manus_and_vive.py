import time
import argparse
import copy
import importlib
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
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK, H2_ArmIK
from teleimager.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.comm_state_logger import CommStateLogger
from teleop.utils.audio_recorder import BackgroundAudioRecorder, AudioRecorderError
from teleop.utils.ipc import IPC_Server
from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
from teleop.utils.manus_haptics import ManusHapticUDPSender, ManusNormalForceMapper
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


def _fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret


T_MANUS_TO_UNITREE_HAND_LEFT = np.array([[0, 1, 0, 0],
                                         [0, 0, -1, 0],
                                         [1, 0, 0, 0],
                                         [0, 0, 0, 1]], dtype=float)
T_MANUS_TO_UNITREE_HAND_RIGHT = np.array([[0, 1, 0, 0],
                                          [0, 0, -1, 0],
                                          [1, 0, 0, 0],
                                          [0, 0, 0, 1]], dtype=float)

TRACKER_WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=float)
TRACKER_TRANSLATION_SCALE = 1.0
CALIBRATION_ORIENTATION_MAX_ERROR_DEG = 45.0

# Tracker-device -> arm EE axis correction. The tracker rotation is first
# expressed in the calibrated tracker frame, matching the translation conversion.
R_TRACKER_TO_EE_LEFT = np.array([[0, 0, 1],
                                 [-1, 0, 0],
                                 [0, -1, 0]], dtype=float)
R_TRACKER_TO_EE_RIGHT = np.array([[0, 0, -1],
                                  [-1, 0, 0],
                                  [0, 1, 0]], dtype=float)
R_TRACKER_TO_EE_ABS = {
    "left": R_TRACKER_TO_EE_LEFT,
    "right": R_TRACKER_TO_EE_RIGHT,
}


def _normalize_ros_msg_type(msg_type):
    if "/msg/" in msg_type:
        return msg_type
    if "." in msg_type:
        parts = msg_type.split(".")
        if len(parts) >= 3 and parts[-2] == "msg":
            return f"{parts[0]}/msg/{parts[-1]}"
    if "/" in msg_type:
        pkg, name = msg_type.split("/", 1)
        return f"{pkg}/msg/{name}"
    raise ValueError(f"ROS2 message type must look like 'pkg/msg/Type': {msg_type}")


def _load_ros_msg_type(msg_type):
    normalized = _normalize_ros_msg_type(msg_type)
    try:
        from rosidl_runtime_py.utilities import get_message
        return get_message(normalized)
    except Exception:
        pkg, _, name = normalized.split("/")
        module = importlib.import_module(f"{pkg}.msg")
        return getattr(module, name)


def _quat_to_rot(qx, qy, qz, qw):
    quat = np.array([qx, qy, qz, qw], dtype=float)
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm < 1e-8:
        return None
    x, y, z, w = quat / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


def _pose_to_mat(position, orientation):
    rot = _quat_to_rot(orientation.x, orientation.y, orientation.z, orientation.w)
    if rot is None:
        return None
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = [position.x, position.y, position.z]
    return mat


def _point_like_to_xyz(point):
    if isinstance(point, dict):
        if all(key in point for key in ("x", "y", "z")):
            return [point["x"], point["y"], point["z"]]
        if "pose" in point:
            return _point_like_to_xyz(point["pose"])
        if "position" in point:
            return _point_like_to_xyz(point["position"])
        if "pos" in point:
            return _point_like_to_xyz(point["pos"])
    if hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
        return [point.x, point.y, point.z]
    if hasattr(point, "pose"):
        return _point_like_to_xyz(point.pose)
    if hasattr(point, "position"):
        return _point_like_to_xyz(point.position)
    if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 3:
        return [point[0], point[1], point[2]]
    return None


def _extract_manus_glove_side(msg, topic=None):
    side = getattr(msg, "side", None)
    if isinstance(side, str) and side.lower() in ("left", "right"):
        return side.lower()
    if topic:
        topic_lower = topic.lower()
        if "left" in topic_lower or topic_lower.endswith("_l"):
            return "left"
        if "right" in topic_lower or topic_lower.endswith("_r"):
            return "right"
    return None


def _extract_manus_glove_positions(msg):
    raw_nodes = getattr(msg, "raw_nodes", None)
    if not raw_nodes:
        return None, None

    positions = np.zeros((25, 3), dtype=float)
    wrist_mat = None
    filled = np.zeros(25, dtype=bool)
    for node in raw_nodes:
        node_id = getattr(node, "node_id", None)
        if node_id is None or not (0 <= node_id < 25):
            continue
        pose = getattr(node, "pose", None)
        if pose is None:
            continue
        position = getattr(pose, "position", None)
        if position is None:
            continue
        xyz = _point_like_to_xyz(position)
        if xyz is None:
            continue
        positions[node_id] = xyz
        filled[node_id] = True
        if node_id == 0 and hasattr(pose, "orientation"):
            wrist_mat = _pose_to_mat(pose.position, pose.orientation)

    if not filled[0]:
        return None, None
    return positions, wrist_mat


def _pose_dict_to_mat(pose):
    if not isinstance(pose, dict):
        return None
    position = pose.get("position", pose.get("pos"))
    orientation = pose.get("orientation", pose.get("quat", pose.get("rotation")))
    if position is None or orientation is None:
        return None

    def pick_xyz(value):
        if isinstance(value, dict):
            return [value.get("x"), value.get("y"), value.get("z")]
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return [value[0], value[1], value[2]]
        return None

    xyz = pick_xyz(position)
    if isinstance(orientation, dict):
        quat = [orientation.get("x"), orientation.get("y"), orientation.get("z"), orientation.get("w")]
    elif isinstance(orientation, (list, tuple)) and len(orientation) >= 4:
        quat = list(orientation[:4])
    else:
        quat = None
    if xyz is None or quat is None or any(value is None for value in xyz + quat):
        return None
    rot = _quat_to_rot(quat[0], quat[1], quat[2], quat[3])
    if rot is None:
        return None
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = np.asarray(xyz, dtype=float)
    return mat


def _extract_manus_json_packet(packet):
    if not isinstance(packet, dict):
        return []

    updates = []
    for side in ("left", "right"):
        side_packet = packet.get(side)
        if side_packet is None:
            continue
        if isinstance(side_packet, dict):
            positions = side_packet.get("positions", side_packet.get("raw_nodes", side_packet.get("nodes")))
            wrist_pose = side_packet.get("wrist_pose", side_packet.get("wrist"))
        else:
            positions = side_packet
            wrist_pose = None
        updates.extend(_extract_manus_json_positions(side, positions, wrist_pose))

    side = packet.get("side")
    if isinstance(side, str) and side.lower() in ("left", "right"):
        positions = packet.get("positions", packet.get("raw_nodes", packet.get("nodes")))
        wrist_pose = packet.get("wrist_pose", packet.get("wrist"))
        updates.extend(_extract_manus_json_positions(side.lower(), positions, wrist_pose))
    return updates


def _extract_manus_json_positions(side, positions_payload, wrist_pose_payload=None):
    if positions_payload is None:
        return []

    positions = np.zeros((25, 3), dtype=float)
    wrist_mat = _pose_dict_to_mat(wrist_pose_payload)
    filled = 0

    if isinstance(positions_payload, list) and any(
        isinstance(item, dict) and ("node_id" in item or "pose" in item)
        for item in positions_payload
    ):
        for node in positions_payload:
            if not isinstance(node, dict):
                continue
            node_id = node.get("node_id", node.get("id", node.get("index")))
            if node_id is None or not (0 <= int(node_id) < 25):
                continue
            pose = node.get("pose")
            position = node.get("position", node.get("pos"))
            if position is None and isinstance(pose, dict):
                position = pose.get("position", pose.get("pos"))
            xyz = _point_like_to_xyz(position)
            if xyz is not None:
                positions[int(node_id)] = xyz
                filled += 1
            if wrist_mat is None and int(node_id) == 0:
                wrist_mat = _pose_dict_to_mat(pose)
    elif isinstance(positions_payload, list) and len(positions_payload) == 25 and all(
        isinstance(item, (list, tuple, dict)) for item in positions_payload
    ):
        for idx, item in enumerate(positions_payload):
            if isinstance(item, dict):
                pose = item.get("pose")
                xyz = _point_like_to_xyz(item.get("position", item.get("pos", pose)))
                if wrist_mat is None and idx == 0 and pose is not None:
                    wrist_mat = _pose_dict_to_mat(pose)
            else:
                xyz = _point_like_to_xyz(item)
            if xyz is not None:
                positions[idx] = xyz
                filled += 1

    if filled < 8 or not np.all(np.isfinite(positions)):
        return []
    return [(side, positions, wrist_mat)]


def _vive_tracker_key_to_side(key):
    key = str(key).lower()
    if key in ("left", "right", "head"):
        return key
    if key.startswith("left") or key.startswith("l_"):
        return "left"
    if key.startswith("right") or key.startswith("r_"):
        return "right"
    if key.startswith("head") or key.startswith("h_"):
        return "head"
    return None


def _extract_vive_json_packet(packet):
    if not isinstance(packet, dict):
        return []

    updates = []
    trackers = packet.get("trackers")
    if isinstance(trackers, dict):
        items = trackers.items()
    else:
        items = (
            (key, packet.get(key))
            for key in ("left", "right", "head", "left_tracker", "right_tracker", "head_tracker")
            if key in packet
        )

    for key, item in items:
        side = _vive_tracker_key_to_side(key)
        if side is None or not isinstance(item, dict):
            continue
        if item.get("ok", True) is False:
            continue
        pose_payload = item.get("pose", item.get("transform", item))
        pose = _pose_dict_to_mat(pose_payload)
        if pose is None or not np.all(np.isfinite(pose)):
            continue
        updates.append((side, pose))
    return updates


def _normalize_vec(vec, eps=1e-6):
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if not np.isfinite(norm) or norm < eps:
        return None
    return vec / norm


def _project_rotation(rot):
    if rot is None or not np.all(np.isfinite(rot)):
        return None
    u, _, vt = np.linalg.svd(rot)
    projected = u @ vt
    if np.linalg.det(projected) < 0:
        u[:, -1] *= -1.0
        projected = u @ vt
    return projected


def _rot_to_euler_xyz(rot):
    rot = _project_rotation(rot)
    if rot is None:
        return None
    sy = np.sqrt(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(rot[2, 1], rot[2, 2])
        y = np.arctan2(-rot[2, 0], sy)
        z = np.arctan2(rot[1, 0], rot[0, 0])
    else:
        x = np.arctan2(-rot[1, 2], rot[1, 1])
        y = np.arctan2(-rot[2, 0], sy)
        z = 0.0
    return np.array([x, y, z], dtype=float)


def _rot_to_euler_xyz_deg(rot):
    euler_xyz = _rot_to_euler_xyz(rot)
    if euler_xyz is None:
        return None
    return np.degrees(euler_xyz)


def _rotation_error_deg(rot_a, rot_b):
    rot_a = _project_rotation(rot_a)
    rot_b = _project_rotation(rot_b)
    if rot_a is None or rot_b is None:
        return None
    rel = _project_rotation(rot_a.T @ rot_b)
    if rel is None:
        return None
    cos_angle = (np.trace(rel) - 1.0) * 0.5
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(angle))


def _fmt_vec(vec):
    return f"[{vec[0]: .3f}, {vec[1]: .3f}, {vec[2]: .3f}]"


def _fmt_mat(mat):
    mat = np.asarray(mat, dtype=float)
    return np.array2string(mat, precision=3, suppress_small=True)


def _rot_summary_text(rot):
    rot = _project_rotation(rot)
    if rot is None:
        return "invalid"
    euler = _rot_to_euler_xyz_deg(rot)
    angle = _rotation_error_deg(np.eye(3), rot)
    euler_part = "xyz_deg=invalid" if euler is None else f"xyz_deg=[{euler[0]: .1f}, {euler[1]: .1f}, {euler[2]: .1f}]"
    angle_part = "angle_deg=invalid" if angle is None else f"angle_deg={angle:.1f}"
    return f"{euler_part} {angle_part}"


def _tracker_rot_to_calib_frame(tracker_rot, tracker_basis):
    tracker_rot = _project_rotation(tracker_rot)
    if tracker_rot is None or tracker_basis is None:
        return None
    return _project_rotation(tracker_basis.T @ tracker_rot)


def _tracker_abs_rot_to_ee_rot(tracker_rot, side, tracker_basis):
    tracker_rot = _tracker_rot_to_calib_frame(tracker_rot, tracker_basis)
    if tracker_rot is None:
        return None
    return _project_rotation(tracker_rot @ R_TRACKER_TO_EE_ABS[side])


def _wrap_angle_rad(angle):
    return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi


def _pose_from_yaw_pitch(yaw_pitch):
    yaw, pitch = np.asarray(yaw_pitch, dtype=np.float64).reshape(2)
    yaw = _wrap_angle_rad(yaw)
    pitch = float(np.clip(pitch, -np.pi * 0.5 + 1e-6, np.pi * 0.5 - 1e-6))
    cp = np.cos(pitch)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = [cp * np.cos(yaw), cp * np.sin(yaw), np.sin(pitch)]
    return pose


def _vive_head_tracker_rot_yaw_pitch(rot, yaw_sign=1.0, pitch_sign=1.0):
    euler_xyz = _rot_to_euler_xyz(rot)
    if euler_xyz is None:
        return None
    # Vive head tracker axis mapping for the camera neck:
    # Y rotation drives camera yaw, X rotation drives camera pitch, Z/roll is ignored.
    yaw = float(yaw_sign) * float(euler_xyz[1])
    pitch = float(pitch_sign) * float(euler_xyz[0])
    return np.array([yaw, pitch], dtype=np.float64)


def _vive_head_tracker_yaw_pitch(head_pose, yaw_sign=1.0, pitch_sign=1.0):
    pose = np.asarray(head_pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return None
    return _vive_head_tracker_rot_yaw_pitch(
        pose[:3, :3],
        yaw_sign=yaw_sign,
        pitch_sign=pitch_sign,
    )


class LibsurviveTFReader:
    def __init__(
        self,
        node,
        left_name,
        right_name,
        head_name=None,
        tracking_frame="libsurvive_world",
        stale_timeout=0.5,
    ):
        import tf2_ros

        self._buffer = tf2_ros.Buffer()
        self._listener = tf2_ros.TransformListener(self._buffer, node)
        self._names = {"left": left_name, "right": right_name}
        if head_name:
            self._names["head"] = head_name
        self._tracking_frame = tracking_frame
        self._stale_timeout = stale_timeout
        self._last_ok = {key: 0.0 for key in self._names}
        self._last_poses = {key: None for key in self._names}
        logger_mp.info(
            f"LibsurviveTFReader: left='{left_name}' right='{right_name}' "
            f"head='{head_name}' frame='{tracking_frame}'"
        )

    def _lookup_one(self, side):
        name = self._names.get(side)
        if not name:
            return None, False
        try:
            from rclpy.time import Time as RclpyTime

            transform = self._buffer.lookup_transform(self._tracking_frame, name, RclpyTime())
            pose = _pose_to_mat(transform.transform.translation, transform.transform.rotation)
            if pose is not None:
                self._last_ok[side] = time.monotonic()
                self._last_poses[side] = pose
        except Exception as e:
            now = time.monotonic()
            if now - getattr(self, f"_last_err_{side}", 0.0) > 3.0:
                logger_mp.warning(f"[Vive/TF] {side} lookup failed: {e}")
                setattr(self, f"_last_err_{side}", now)

        now = time.monotonic()
        ok = (now - self._last_ok[side]) <= self._stale_timeout
        return self._last_poses[side], ok

    def read(self):
        left_pose, left_ok = self._lookup_one("left")
        right_pose, right_ok = self._lookup_one("right")
        return left_pose, right_pose, left_ok, right_ok

    def read_head(self):
        if "head" not in self._names:
            return None, False
        return self._lookup_one("head")


class ManusHandReader:
    def __init__(self, node, topics, msg_type, stale_timeout):
        self._lock = threading.Lock()
        self._positions = {"left": None, "right": None}
        self._wrist_mats = {"left": None, "right": None}
        self._stamp = {"left": 0.0, "right": 0.0}
        self._last_warn = {"left": 0.0, "right": 0.0}
        self._stale_timeout = stale_timeout
        ros_msg_type = _load_ros_msg_type(msg_type)
        self._subs = []
        for topic in topics:
            self._subs.append(
                node.create_subscription(
                    ros_msg_type,
                    topic,
                    lambda msg, topic=topic: self._callback(msg, topic),
                    5,
                )
            )
        logger_mp.info(f"ManusHandReader subscribed: topics={topics}, type={msg_type}")

    def _callback(self, msg, topic):
        side = _extract_manus_glove_side(msg, topic)
        if side is None:
            now = time.monotonic()
            if now - getattr(self, "_last_side_warn", 0.0) > 3.0:
                logger_mp.warning("Cannot determine Manus glove side. Set msg.side or use left/right in topic names.")
                self._last_side_warn = now
            return

        positions, wrist_mat = _extract_manus_glove_positions(msg)
        now = time.monotonic()
        if positions is None or positions.shape != (25, 3) or not np.all(np.isfinite(positions)):
            if now - self._last_warn[side] > 2.0:
                logger_mp.warning(f"Cannot parse {side} Manus hand positions as 25 xyz joints.")
                self._last_warn[side] = now
            return

        with self._lock:
            self._positions[side] = positions
            self._wrist_mats[side] = wrist_mat
            self._stamp[side] = now

    def read(self):
        now = time.monotonic()
        with self._lock:
            left_pos = None if self._positions["left"] is None else self._positions["left"].copy()
            right_pos = None if self._positions["right"] is None else self._positions["right"].copy()
            left_wrist = None if self._wrist_mats["left"] is None else self._wrist_mats["left"].copy()
            right_wrist = None if self._wrist_mats["right"] is None else self._wrist_mats["right"].copy()
            left_ok = left_pos is not None and (now - self._stamp["left"]) <= self._stale_timeout
            right_ok = right_pos is not None and (now - self._stamp["right"]) <= self._stale_timeout
        return left_pos, right_pos, left_wrist, right_wrist, left_ok, right_ok


class ManusUDPJsonReader:
    def __init__(self, host, port, stale_timeout, recv_size=65535):
        self.host = host
        self.port = int(port)
        self._stale_timeout = float(stale_timeout)
        self._lock = threading.Lock()
        self._positions = {"left": None, "right": None}
        self._wrist_mats = {"left": None, "right": None}
        self._stamp = {"left": 0.0, "right": 0.0}
        self._last_warn = 0.0
        self._stop_event = threading.Event()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.host, self.port))
        self._socket.settimeout(0.1)
        self._recv_size = int(recv_size)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger_mp.info("[manus udp] listening udp=%s:%s stale_timeout=%.3f", self.host, self.port, self._stale_timeout)

    def _loop(self):
        while not self._stop_event.is_set():
            try:
                raw, _ = self._socket.recvfrom(self._recv_size)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                packet = json.loads(raw.decode("utf-8"))
                updates = _extract_manus_json_packet(packet)
                now = time.monotonic()
                if not updates:
                    raise ValueError("packet does not contain usable Manus hand positions")
                with self._lock:
                    for side, positions, wrist_mat in updates:
                        self._positions[side] = positions
                        self._wrist_mats[side] = wrist_mat
                        self._stamp[side] = now
            except Exception as exc:
                now = time.monotonic()
                if now - self._last_warn > 2.0:
                    logger_mp.warning("[manus udp] malformed packet: %s", exc)
                    self._last_warn = now

    def read(self):
        now = time.monotonic()
        with self._lock:
            left_pos = None if self._positions["left"] is None else self._positions["left"].copy()
            right_pos = None if self._positions["right"] is None else self._positions["right"].copy()
            left_wrist = None if self._wrist_mats["left"] is None else self._wrist_mats["left"].copy()
            right_wrist = None if self._wrist_mats["right"] is None else self._wrist_mats["right"].copy()
            left_ok = left_pos is not None and (now - self._stamp["left"]) <= self._stale_timeout
            right_ok = right_pos is not None and (now - self._stamp["right"]) <= self._stale_timeout
        return left_pos, right_pos, left_wrist, right_wrist, left_ok, right_ok

    def close(self):
        self._stop_event.set()
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        self._socket.close()


class ViveUDPJsonReader:
    def __init__(self, host, port, stale_timeout, recv_size=65535):
        self.host = host
        self.port = int(port)
        self._stale_timeout = float(stale_timeout)
        self._lock = threading.Lock()
        self._poses = {"left": None, "right": None, "head": None}
        self._stamp = {"left": 0.0, "right": 0.0, "head": 0.0}
        self._last_warn = 0.0
        self._stop_event = threading.Event()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.host, self.port))
        self._socket.settimeout(0.1)
        self._recv_size = int(recv_size)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger_mp.info("[vive udp] listening udp=%s:%s stale_timeout=%.3f", self.host, self.port, self._stale_timeout)

    def _loop(self):
        while not self._stop_event.is_set():
            try:
                raw, _ = self._socket.recvfrom(self._recv_size)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                packet = json.loads(raw.decode("utf-8"))
                updates = _extract_vive_json_packet(packet)
                now = time.monotonic()
                if not updates:
                    if packet.get("source") == "vive_tf_to_udp" and not packet.get("trackers"):
                        continue
                    raise ValueError("packet does not contain usable Vive tracker poses")
                with self._lock:
                    for side, pose in updates:
                        self._poses[side] = pose
                        self._stamp[side] = now
            except Exception as exc:
                now = time.monotonic()
                if now - self._last_warn > 2.0:
                    logger_mp.warning("[vive udp] malformed packet: %s", exc)
                    self._last_warn = now

    def _read_one(self, side):
        now = time.monotonic()
        with self._lock:
            pose = None if self._poses[side] is None else self._poses[side].copy()
            ok = pose is not None and (now - self._stamp[side]) <= self._stale_timeout
        return pose, ok

    def read(self):
        left_pose, left_ok = self._read_one("left")
        right_pose, right_ok = self._read_one("right")
        return left_pose, right_pose, left_ok, right_ok

    def read_head(self):
        return self._read_one("head")

    def close(self):
        self._stop_event.set()
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        self._socket.close()


class ViveManusInfoReader:
    def __init__(
        self,
        left_tracker_name,
        right_tracker_name,
        head_tracker_name,
        manus_topics,
        manus_msg_type,
        manus_transport="ros2",
        manus_udp_host="0.0.0.0",
        manus_udp_port=56120,
        vive_transport="tf",
        vive_udp_host="0.0.0.0",
        vive_udp_port=56130,
        libsurvive_tracking_frame="libsurvive_world",
        stale_timeout=0.5,
        manus_hand_transform="legacy",
    ):
        self.manus_hand_transform = str(manus_hand_transform or "legacy").strip().lower()
        if self.manus_hand_transform not in ("legacy", "televuer"):
            raise ValueError(
                f"Unsupported Manus hand transform: {manus_hand_transform}. "
                "Use legacy or televuer."
            )
        if self.manus_hand_transform == "televuer":
            logger_mp.warning(
                "[Vive/Manus] --manus-hand-transform=televuer is not valid for Manus raw skeleton axes; "
                "using legacy Manus->Unitree transform instead."
            )
            self.manus_hand_transform = "legacy"
        self._rclpy = None
        self._owns_rclpy = False
        self.node = None
        self.executor = None
        self.spin_thread = None

        needs_ros = manus_transport == "ros2" or vive_transport == "tf"
        if needs_ros:
            import rclpy
            from rclpy.executors import MultiThreadedExecutor

            self._rclpy = rclpy
            self._owns_rclpy = not rclpy.ok()
            if self._owns_rclpy:
                rclpy.init(args=None)

            self.node = rclpy.create_node("vive_manus_info_reader")
            self.executor = MultiThreadedExecutor()
            self.executor.add_node(self.node)
            self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
            self.spin_thread.start()

        if vive_transport == "tf":
            if self.node is None:
                raise RuntimeError("Vive TF transport requires an rclpy node.")
            self.vive_reader = LibsurviveTFReader(
                self.node,
                left_name=left_tracker_name,
                right_name=right_tracker_name,
                head_name=head_tracker_name,
                tracking_frame=libsurvive_tracking_frame,
                stale_timeout=stale_timeout,
            )
        elif vive_transport == "udp_json":
            self.vive_reader = ViveUDPJsonReader(vive_udp_host, vive_udp_port, stale_timeout)
        else:
            raise ValueError(f"Unsupported Vive transport: {vive_transport}")

        if manus_transport == "ros2":
            if self.node is None:
                raise RuntimeError("Manus ROS2 transport requires an rclpy node.")
            self.manus_reader = ManusHandReader(self.node, manus_topics, manus_msg_type, stale_timeout)
        elif manus_transport == "udp_json":
            self.manus_reader = ManusUDPJsonReader(manus_udp_host, manus_udp_port, stale_timeout)
        else:
            raise ValueError(f"Unsupported Manus transport: {manus_transport}")

        logger_mp.info(
            "[Vive/Manus] transports vive=%s manus=%s stale_timeout=%.3f hand_transform=%s",
            vive_transport,
            manus_transport,
            float(stale_timeout),
            self.manus_hand_transform,
        )
        self._tracker_ref = {"left": None, "right": None}
        self._tracker_rot_ref = {"left": None, "right": None}
        self._head_tracker_rot_ref = None
        self._head_tracker_rot_ref_frame = None
        self._wrist_ref = {"left": None, "right": None}
        self._last_wrist_pose = {"left": None, "right": None}
        self._tracker_origin = None
        self._tracker_basis = None
        self._relative_ref_ready = False

    @property
    def calibrated(self):
        return self._tracker_basis is not None and self._tracker_origin is not None

    @property
    def relative_reference_ready(self):
        return bool(self._relative_ref_ready)

    def reset_calibration(self):
        self._tracker_ref = {"left": None, "right": None}
        self._tracker_rot_ref = {"left": None, "right": None}
        self._head_tracker_rot_ref = None
        self._head_tracker_rot_ref_frame = None
        self._wrist_ref = {"left": None, "right": None}
        self._last_wrist_pose = {"left": None, "right": None}
        self._tracker_origin = None
        self._tracker_basis = None
        self._relative_ref_ready = False

    def close(self):
        for reader in (getattr(self, "vive_reader", None), getattr(self, "manus_reader", None)):
            if reader is not None and hasattr(reader, "close"):
                try:
                    reader.close()
                except Exception:
                    pass
        try:
            if self.executor is not None:
                self.executor.shutdown()
        except Exception:
            pass
        try:
            if self.node is not None:
                self.node.destroy_node()
        except Exception:
            pass
        if self._owns_rclpy and self._rclpy is not None:
            try:
                self._rclpy.shutdown()
            except Exception:
                pass

    def read(self):
        left_tracker, right_tracker, left_tracker_ok, right_tracker_ok = self.vive_reader.read()
        head_tracker, head_tracker_ok = self.vive_reader.read_head()
        left_hand, right_hand, left_wrist, right_wrist, left_hand_ok, right_hand_ok = self.manus_reader.read()
        return {
            "left_tracker": left_tracker,
            "right_tracker": right_tracker,
            "head_tracker": head_tracker,
            "left_tracker_ok": left_tracker_ok,
            "right_tracker_ok": right_tracker_ok,
            "head_tracker_ok": head_tracker_ok,
            "left_hand": left_hand,
            "right_hand": right_hand,
            "left_wrist": left_wrist,
            "right_wrist": right_wrist,
            "left_hand_ok": left_hand_ok,
            "right_hand_ok": right_hand_ok,
        }

    def read_head_tracker(self):
        return self.vive_reader.read_head()

    def reset_head_tracker_neck_ref(self):
        self._head_tracker_rot_ref = None
        self._head_tracker_rot_ref_frame = None

    def reset_relative_motion_reference(self, left_wrist_ref, right_wrist_ref):
        if not self.calibrated:
            logger_mp.warning("[Vive/Relative] Cannot reset sync reference before calibration.")
            return False
        wrist_refs = {
            "left": np.asarray(left_wrist_ref, dtype=np.float64),
            "right": np.asarray(right_wrist_ref, dtype=np.float64),
        }
        for side, wrist_ref in wrist_refs.items():
            if wrist_ref.shape != (4, 4) or not np.isfinite(wrist_ref).all():
                logger_mp.warning("[Vive/Relative] Cannot reset %s sync reference: invalid robot EE pose.", side)
                return False

        info = self.read()
        for side in ("left", "right"):
            tracker_pose = info[f"{side}_tracker"]
            if not info[f"{side}_tracker_ok"] or tracker_pose is None:
                logger_mp.warning("[Vive/Relative] Cannot reset %s sync reference: tracker is invalid or stale.", side)
                return False
            tracker_rot = _tracker_abs_rot_to_ee_rot(tracker_pose[:3, :3], side, self._tracker_basis)
            if tracker_rot is None:
                logger_mp.warning("[Vive/Relative] Cannot reset %s sync reference: tracker rotation is invalid.", side)
                return False

            self._tracker_ref[side] = self._tracker_basis.T @ (tracker_pose[:3, 3] - self._tracker_origin)
            self._tracker_rot_ref[side] = tracker_rot.copy()
            self._wrist_ref[side] = wrist_refs[side].copy()

        self._last_wrist_pose = {"left": None, "right": None}
        self._relative_ref_ready = True
        logger_mp.info(
            "[Vive/Relative] sync reference reset at [r] from current robot EE pose. "
            "left_wrist_ref=%s right_wrist_ref=%s left_tracker_ref=%s right_tracker_ref=%s",
            np.round(self._wrist_ref["left"][:3, 3], 3).tolist(),
            np.round(self._wrist_ref["right"][:3, 3], 3).tolist(),
            np.round(self._tracker_ref["left"], 3).tolist(),
            np.round(self._tracker_ref["right"], 3).tolist(),
        )
        return True

    def read_head_tracker_yaw_pitch_for_neck(self, yaw_sign=1.0, pitch_sign=1.0):
        head_tracker_pose, head_tracker_ok = self.read_head_tracker()
        if not head_tracker_ok or head_tracker_pose is None:
            return None, False
        if self._tracker_basis is not None:
            head_tracker_rot = _tracker_rot_to_calib_frame(head_tracker_pose[:3, :3], self._tracker_basis)
            ref_frame = "calibrated"
        else:
            head_tracker_rot = _project_rotation(head_tracker_pose[:3, :3])
            ref_frame = "raw"
        if head_tracker_rot is None:
            return None, False
        if self._head_tracker_rot_ref is None or self._head_tracker_rot_ref_frame != ref_frame:
            self._head_tracker_rot_ref = head_tracker_rot.copy()
            self._head_tracker_rot_ref_frame = ref_frame
            logger_mp.info(
                "[Vive/Head] neck tracker_rot_ref set from current head tracker orientation "
                f"(frame={ref_frame})."
            )
        relative_head_rot = _project_rotation(self._head_tracker_rot_ref.T @ head_tracker_rot)
        yaw_pitch = _vive_head_tracker_rot_yaw_pitch(
            relative_head_rot,
            yaw_sign=yaw_sign,
            pitch_sign=pitch_sign,
        )
        return yaw_pitch, yaw_pitch is not None

    def calibrate_reference_frame(self, tele_data=None):
        info = self.read()
        if not (
            info["left_tracker_ok"]
            and info["right_tracker_ok"]
            and info["left_tracker"] is not None
            and info["right_tracker"] is not None
        ):
            logger_mp.warning("[Vive/Frame] Cannot calibrate. Need valid left/right trackers.")
            return False

        if tele_data is None:
            logger_mp.warning("[Vive/Frame] Cannot calibrate orientation. Robot EE pose is not available yet.")
            return False

        left_xyz = info["left_tracker"][:3, 3].copy()
        right_xyz = info["right_tracker"][:3, 3].copy()
        origin = 0.5 * (left_xyz + right_xyz)

        z_axis = TRACKER_WORLD_UP.copy()
        side_axis = _normalize_vec(right_xyz - left_xyz)
        if z_axis is None or side_axis is None:
            logger_mp.warning("[Vive/Frame] Cannot calibrate. Tracker geometry is degenerate.")
            return False

        side_axis = _normalize_vec(side_axis - z_axis * float(side_axis @ z_axis))
        if side_axis is None:
            logger_mp.warning("[Vive/Frame] Cannot calibrate. side and z axes are nearly parallel.")
            return False

        x_axis = _normalize_vec(np.cross(z_axis, side_axis))
        if x_axis is None:
            logger_mp.warning("[Vive/Frame] Cannot calibrate. Failed to derive forward x-axis.")
            return False
        y_axis = _normalize_vec(np.cross(z_axis, x_axis))
        if y_axis is None:
            logger_mp.warning("[Vive/Frame] Cannot calibrate. Failed to orthogonalize y-axis.")
            return False

        tracker_basis = np.column_stack([x_axis, y_axis, z_axis])
        orientation_errors = {}
        orientation_debug = {}
        for side in ("left", "right"):
            tracker_raw_rot = _project_rotation(info[f"{side}_tracker"][:3, :3])
            tracker_calib_rot = _tracker_rot_to_calib_frame(info[f"{side}_tracker"][:3, :3], tracker_basis)
            tracker_rot = _tracker_abs_rot_to_ee_rot(info[f"{side}_tracker"][:3, :3], side, tracker_basis)
            wrist_pose = getattr(tele_data, f"{side}_wrist_pose", None)
            wrist_rot = None if wrist_pose is None else wrist_pose[:3, :3]
            error_deg = _rotation_error_deg(tracker_rot, wrist_rot)
            if error_deg is None:
                logger_mp.warning(f"[Vive/Frame] Cannot calibrate. {side} orientation is invalid.")
                return False
            orientation_errors[side] = error_deg
            orientation_debug[side] = (
                tracker_raw_rot,
                tracker_calib_rot,
                tracker_rot,
                _project_rotation(wrist_rot),
            )

        print("[Vive/Frame orientation check]", flush=True)
        for side in ("left", "right"):
            tracker_raw_rot, tracker_calib_rot, tracker_rot, wrist_rot = orientation_debug[side]
            print(
                f"{side} tracker_raw_rot=\n{_fmt_mat(tracker_raw_rot)}\n"
                f"{side} tracker_calib_rot=\n{_fmt_mat(tracker_calib_rot)}\n"
                f"{side} tracker_rot_ee=\n{_fmt_mat(tracker_rot)}\n"
                f"{side} ee_rot=\n{_fmt_mat(wrist_rot)}\n"
                f"{side} rot_error_deg={orientation_errors[side]:.1f}",
                flush=True,
            )

        max_error = max(orientation_errors.values())
        if max_error > CALIBRATION_ORIENTATION_MAX_ERROR_DEG:
            logger_mp.warning(
                "[Vive/Frame] Cannot calibrate. Tracker/EE orientation mismatch is too large: "
                f"left={orientation_errors['left']:.1f}deg "
                f"right={orientation_errors['right']:.1f}deg "
                f"limit={CALIBRATION_ORIENTATION_MAX_ERROR_DEG:.1f}deg"
            )
            return False

        self._tracker_origin = origin
        self._tracker_basis = tracker_basis
        self._tracker_ref["left"] = self._tracker_basis.T @ (left_xyz - origin)
        self._tracker_ref["right"] = self._tracker_basis.T @ (right_xyz - origin)
        self._tracker_rot_ref = {
            "left": orientation_debug["left"][2].copy(),
            "right": orientation_debug["right"][2].copy(),
        }
        self._head_tracker_rot_ref = None
        self._head_tracker_rot_ref_frame = None
        if info["head_tracker_ok"] and info["head_tracker"] is not None:
            head_tracker_rot = _tracker_rot_to_calib_frame(info["head_tracker"][:3, :3], self._tracker_basis)
            if head_tracker_rot is not None:
                self._head_tracker_rot_ref = head_tracker_rot.copy()
                self._head_tracker_rot_ref_frame = "calibrated"
        self._last_wrist_pose = {"left": None, "right": None}
        self._wrist_ref = {
            "left": tele_data.left_wrist_pose.copy(),
            "right": tele_data.right_wrist_pose.copy(),
        }
        self._relative_ref_ready = False

        logger_mp.info(
            "[Vive/Frame] Calibrated from attention pose. "
            f"origin(wrist_mid)={origin.round(3)} "
            f"x(forward)={x_axis.round(3)} "
            f"y(right_to_left)={y_axis.round(3)} "
            f"z(world_up)={z_axis.round(3)} "
            f"orientation_error(left/right)={orientation_errors['left']:.1f}/{orientation_errors['right']:.1f}deg "
            f"head_ref={'set' if self._head_tracker_rot_ref is not None else 'waiting'}"
        )
        return True

    def _apply_side_relative_motion(self, tele_data, info, side):
        if self._tracker_basis is None or self._tracker_origin is None:
            return False
        if not self._relative_ref_ready:
            return False

        tracker_pose = info[f"{side}_tracker"]
        tracker_ok = info[f"{side}_tracker_ok"]
        attr = f"{side}_wrist_pose"

        if not tracker_ok or tracker_pose is None:
            if self._last_wrist_pose[side] is not None:
                setattr(tele_data, attr, self._last_wrist_pose[side].copy())
            return False

        current_tracker_xyz = self._tracker_basis.T @ (tracker_pose[:3, 3] - self._tracker_origin)
        if self._wrist_ref[side] is None or self._tracker_ref[side] is None:
            self._relative_ref_ready = False
            return False

        target_wrist_pose = self._wrist_ref[side].copy()
        target_wrist_pose[:3, 3] = (
            self._wrist_ref[side][:3, 3]
            + (current_tracker_xyz - self._tracker_ref[side]) * TRACKER_TRANSLATION_SCALE
        )
        current_tracker_rot = _tracker_abs_rot_to_ee_rot(tracker_pose[:3, :3], side, self._tracker_basis)
        if current_tracker_rot is not None:
            if self._tracker_rot_ref[side] is None:
                self._tracker_rot_ref[side] = current_tracker_rot.copy()
                logger_mp.info(f"[Vive/Relative] {side} tracker_rot_ref set.")
            relative_tracker_rot = self._tracker_rot_ref[side].T @ current_tracker_rot
            target_wrist_rot = _project_rotation(self._wrist_ref[side][:3, :3] @ relative_tracker_rot)
            if target_wrist_rot is not None:
                target_wrist_pose[:3, :3] = target_wrist_rot
        setattr(tele_data, attr, target_wrist_pose)
        self._last_wrist_pose[side] = target_wrist_pose.copy()
        return True

    def apply_relative_wrist_motion(self, tele_data):
        info = self.read()
        left_ready = self._apply_side_relative_motion(tele_data, info, "left")
        right_ready = self._apply_side_relative_motion(tele_data, info, "right")
        return left_ready and right_ready

    def _hand_pos_for_retargeting(self, hand_positions, wrist_mat, valid, side):
        if not valid or hand_positions is None:
            return np.zeros((25, 3))
        if wrist_mat is not None:
            arm = _fast_mat_inv(wrist_mat)
        else:
            wrist = hand_positions[0].copy()
            arm = np.eye(4)
            arm[:3, 3] = -wrist

        t_manus_to_unitree = (
            T_MANUS_TO_UNITREE_HAND_LEFT
            if side == "left"
            else T_MANUS_TO_UNITREE_HAND_RIGHT
        )
        hom = np.concatenate([hand_positions.T, np.ones((1, hand_positions.shape[0]))])
        local = arm @ hom
        return (t_manus_to_unitree @ local)[0:3, :].T

    def apply_manus_hand_data(self, tele_data):
        (
            left_hand_raw,
            right_hand_raw,
            left_wrist_mat,
            right_wrist_mat,
            left_hand_ok,
            right_hand_ok,
        ) = self.manus_reader.read()

        left_hand_pos = self._hand_pos_for_retargeting(
            left_hand_raw, left_wrist_mat, left_hand_ok, "left"
        )
        right_hand_pos = self._hand_pos_for_retargeting(
            right_hand_raw, right_wrist_mat, right_hand_ok, "right"
        )

        hands_ready = left_hand_ok and right_hand_ok
        tele_data.left_hand_pos = left_hand_pos
        tele_data.right_hand_pos = right_hand_pos
        tele_data.left_hand_pinchValue = (
            float(np.linalg.norm(left_hand_pos[4] - left_hand_pos[9]) * 100.0)
            if left_hand_ok
            else 0.0
        )
        tele_data.right_hand_pinchValue = (
            float(np.linalg.norm(right_hand_pos[4] - right_hand_pos[9]) * 100.0)
            if right_hand_ok
            else 0.0
        )
        tele_data.hand_motion_data_ready = hands_ready
        return hands_ready

    def print_tracker_positions(self, tele_data=None):
        info = self.read()
        samples = [
            ("left", info["left_tracker"], info["left_tracker_ok"]),
            ("right", info["right_tracker"], info["right_tracker_ok"]),
            ("head", info["head_tracker"], info["head_tracker_ok"]),
        ]

        frame_xyz_by_side = {}
        for side, pose, valid in samples:
            if pose is None:
                continue
            raw_xyz = pose[:3, 3]
            if self._tracker_basis is not None and self._tracker_origin is not None:
                frame_xyz = self._tracker_basis.T @ (raw_xyz - self._tracker_origin)
                frame_xyz_by_side[side] = frame_xyz

        if self._tracker_basis is None or self._tracker_origin is None:
            print("[tracker calib pos] not calibrated. Press [c] in attention pose.", flush=True)
        else:
            tracker_parts = []
            for side in ("left", "right", "head"):
                valid = info[f"{side}_tracker_ok"]
                frame_xyz = frame_xyz_by_side.get(side)
                if frame_xyz is None:
                    tracker_parts.append(f"{side}: invalid")
                else:
                    status = "" if valid else " stale"
                    tracker_parts.append(f"{side}: {_fmt_vec(frame_xyz)}{status}")
            print("[tracker calib pos] " + " | ".join(tracker_parts), flush=True)

            if "left" in frame_xyz_by_side and "right" in frame_xyz_by_side:
                center = 0.5 * (frame_xyz_by_side["left"] + frame_xyz_by_side["right"])
                span = frame_xyz_by_side["right"] - frame_xyz_by_side["left"]
                print(f"[tracker calib delta] center: {_fmt_vec(center)} | right-left: {_fmt_vec(span)}", flush=True)

        ee_parts = []
        for side in ("left", "right"):
            attr = f"{side}_wrist_pose"
            pose = None
            if tele_data is not None and hasattr(tele_data, attr):
                pose = getattr(tele_data, attr)
            elif self._last_wrist_pose[side] is not None:
                pose = self._last_wrist_pose[side]
            if pose is None:
                ee_parts.append(f"{side}: invalid")
            else:
                ee_parts.append(f"{side}: {_fmt_vec(pose[:3, 3])}")
        print("[robot ee pos] " + " | ".join(ee_parts), flush=True)

        if self._tracker_basis is None:
            print("[tracker rot debug] not calibrated. Press [c] in attention pose.", flush=True)
            return

        for side in ("left", "right"):
            tracker_pose = info[f"{side}_tracker"]
            tracker_ok = info[f"{side}_tracker_ok"]
            if not tracker_ok or tracker_pose is None:
                print(f"[tracker rot debug] {side}: invalid tracker", flush=True)
                continue

            tracker_rot_ee = _tracker_abs_rot_to_ee_rot(tracker_pose[:3, :3], side, self._tracker_basis)
            if tracker_rot_ee is None:
                print(f"[tracker rot debug] {side}: invalid tracker_rot_ee", flush=True)
                continue

            relative_tracker_rot = None
            if self._tracker_rot_ref.get(side) is not None:
                relative_tracker_rot = _project_rotation(self._tracker_rot_ref[side].T @ tracker_rot_ee)

            target_wrist_rot = None
            if relative_tracker_rot is not None and self._wrist_ref.get(side) is not None:
                target_wrist_rot = _project_rotation(self._wrist_ref[side][:3, :3] @ relative_tracker_rot)

            attr = f"{side}_wrist_pose"
            current_wrist_pose = None
            if tele_data is not None and hasattr(tele_data, attr):
                current_wrist_pose = getattr(tele_data, attr)
            elif self._last_wrist_pose[side] is not None:
                current_wrist_pose = self._last_wrist_pose[side]
            current_wrist_rot = None if current_wrist_pose is None else _project_rotation(current_wrist_pose[:3, :3])

            print(
                f"[tracker rot debug] {side}\n"
                f"tracker_rot_ee ({_rot_summary_text(tracker_rot_ee)})=\n{_fmt_mat(tracker_rot_ee)}\n"
                f"relative_tracker_rot ({_rot_summary_text(relative_tracker_rot)})=\n{_fmt_mat(relative_tracker_rot) if relative_tracker_rot is not None else 'invalid'}\n"
                f"target_wrist_rot_calc ({_rot_summary_text(target_wrist_rot)})=\n{_fmt_mat(target_wrist_rot) if target_wrist_rot is not None else 'invalid'}\n"
                f"current_wrist_rot ({_rot_summary_text(current_wrist_rot)})=\n{_fmt_mat(current_wrist_rot) if current_wrist_rot is not None else 'invalid'}",
                flush=True,
            )


VIVE_MANUS_READER = None
PRINT_TRACKER_POSITION = False
CALIBRATE_TRACKER_FRAME = False
CALIBRATE_TRACKER_FRAME_AT = None
CALIBRATE_TRACKER_FRAME_DELAY = 1.0
CALIBRATE_WAIT_WARN_AT = 0.0
START_SYNC_DELAY = 1.0
START_SYNC_AT = None
NECK_NEUTRAL_RESET_REQUEST = False
NECK_CAMERA_SYNC_ACTIVE = False

# Guards the [r]/[p]/[c] state transitions shared between the sshkeyboard listener
# thread (on_press) and the main teleop loop (delayed sync / deferred calibration).
STATE_LOCK = threading.Lock()


def maybe_print_tracker_position(tele_data=None):
    global PRINT_TRACKER_POSITION
    if not PRINT_TRACKER_POSITION:
        return
    PRINT_TRACKER_POSITION = False
    if VIVE_MANUS_READER is None:
        logger_mp.warning("[Vive tracker position] reader is not ready yet.")
        return
    try:
        VIVE_MANUS_READER.print_tracker_positions(tele_data=tele_data)
    except Exception as e:
        logger_mp.warning(f"[Vive tracker position] print failed: {e}")


def maybe_calibrate_tracker_frame(tele_data=None):
    global CALIBRATE_TRACKER_FRAME, CALIBRATE_TRACKER_FRAME_AT, CALIBRATE_WAIT_WARN_AT
    if not CALIBRATE_TRACKER_FRAME:
        return False
    if CALIBRATE_TRACKER_FRAME_AT is not None and time.monotonic() < CALIBRATE_TRACKER_FRAME_AT:
        return False
    if tele_data is None:
        now = time.monotonic()
        if now - CALIBRATE_WAIT_WARN_AT > 1.0:
            logger_mp.warning("[Vive/Frame] Waiting for robot EE pose before calibration.")
            CALIBRATE_WAIT_WARN_AT = now
        return False
    calibrated = False
    if VIVE_MANUS_READER is None:
        logger_mp.warning("[Vive/Frame] reader is not ready yet.")
    else:
        try:
            calibrated = bool(VIVE_MANUS_READER.calibrate_reference_frame(tele_data=tele_data))
        except Exception as e:
            logger_mp.warning(f"[Vive/Frame] calibration failed: {e}")
    if calibrated:
        CALIBRATE_TRACKER_FRAME = False
        CALIBRATE_TRACKER_FRAME_AT = None
        return True
    # Keep the [c] request pending and retry instead of silently dropping it —
    # otherwise the next [r] starts teleop uncalibrated and the arms never follow.
    CALIBRATE_TRACKER_FRAME_AT = time.monotonic() + CALIBRATE_TRACKER_FRAME_DELAY
    logger_mp.warning(
        "[Vive/Frame] Calibration attempt failed; retrying in %.1fs. Hold the attention pose. "
        "Press [c] to restart or [q] to quit.",
        CALIBRATE_TRACKER_FRAME_DELAY,
    )
    return False


def _maybe_calibrate_tracker_frame_and_pause_locked(tele_data=None):
    global START, START_SYNC_AT, NECK_NEUTRAL_RESET_REQUEST, NECK_CAMERA_SYNC_ACTIVE
    calibrated = maybe_calibrate_tracker_frame(tele_data=tele_data)
    if calibrated:
        START = False
        NECK_NEUTRAL_RESET_REQUEST = True
        NECK_CAMERA_SYNC_ACTIVE = False
        if START_SYNC_AT is not None:
            # A queued [r] is waiting on this calibration: re-arm the sync delay so the
            # reference is captured a moment after calibration, not instantly.
            START_SYNC_AT = time.monotonic() + START_SYNC_DELAY
            logger_mp.info(
                "[Vive/Frame] Calibration complete. Queued [r] resumes: sync starts in %.1fs — hold the desired pose.",
                START_SYNC_DELAY,
            )
        else:
            logger_mp.info("[Vive/Frame] Calibration complete. Press [r] to sync arm, Manus hand, and neck camera motion.")
    return calibrated


def maybe_calibrate_tracker_frame_and_pause(tele_data=None):
    with STATE_LOCK:
        return _maybe_calibrate_tracker_frame_and_pause_locked(tele_data=tele_data)


def _maybe_start_delayed_sync_locked(arm_ik=None, arm_ctrl=None):
    global START, START_SYNC_AT, PRINT_TRACKER_POSITION, PAUSE_TO_READY
    global NECK_CAMERA_SYNC_ACTIVE, NECK_NEUTRAL_RESET_REQUEST
    if START_SYNC_AT is None:
        return False
    if STOP:
        START_SYNC_AT = None
        return False
    if CALIBRATE_TRACKER_FRAME:
        # Calibration is still pending; keep the queued [r] and fire after it completes.
        return False
    if time.monotonic() < START_SYNC_AT:
        return False
    if START:
        # A stale [r] press raced with an already-running teleop; do not re-fire the sync mid-run.
        START_SYNC_AT = None
        return False

    sync_ready = VIVE_MANUS_READER is None or VIVE_MANUS_READER.calibrated
    if sync_ready and VIVE_MANUS_READER is not None:
        # Capture the sync reference BEFORE flipping START: starting with a broken
        # reference leaves teleop "running" while the arms silently never follow.
        left_wrist_ref, right_wrist_ref = _current_robot_ee_poses_from_fk(arm_ik, arm_ctrl)
        relative_ref_ready = False
        if left_wrist_ref is not None and right_wrist_ref is not None:
            relative_ref_ready = VIVE_MANUS_READER.reset_relative_motion_reference(
                left_wrist_ref,
                right_wrist_ref,
            )
        if not relative_ref_ready:
            START_SYNC_AT = time.monotonic() + START_SYNC_DELAY
            logger_mp.warning(
                "[teleop sync] sync reference capture failed (tracker or robot EE pose invalid); "
                "start deferred, retrying in %.1fs. Hold the pose. Press [c] to recalibrate or [q] to quit.",
                START_SYNC_DELAY,
            )
            return False
        try:
            VIVE_MANUS_READER.reset_head_tracker_neck_ref()
        except Exception as exc:
            logger_mp.debug("[Vive/Head] neck ref reset skipped: %s", exc)

    START_SYNC_AT = None
    START = True
    PAUSE_TO_READY = False
    PRINT_TRACKER_POSITION = True
    NECK_NEUTRAL_RESET_REQUEST = True
    NECK_CAMERA_SYNC_ACTIVE = bool(sync_ready)
    if not sync_ready:
        logger_mp.warning(
            "[teleop start] started without Vive/Manus calibration. "
            "Robot start/home behavior is preserved, but arm/Manus/neck camera sync stays disabled. "
            "Press [c] to calibrate, then press [r] to sync."
        )
        return True
    logger_mp.info(
        "[teleop sync] robot, Manus hand, and neck camera sync started after %.1fs delay. "
        "vive_arm_ref_ready=%s",
        START_SYNC_DELAY,
        True if VIVE_MANUS_READER is None else VIVE_MANUS_READER.relative_reference_ready,
    )
    return True


def maybe_start_delayed_sync(arm_ik=None, arm_ctrl=None):
    with STATE_LOCK:
        return _maybe_start_delayed_sync_locked(arm_ik=arm_ik, arm_ctrl=arm_ctrl)


def _apply_vive_manus_input(tele_data):
    if VIVE_MANUS_READER is None:
        return

    maybe_calibrate_tracker_frame_and_pause(tele_data=tele_data)
    if not START or CALIBRATE_TRACKER_FRAME:
        tele_data.arm_motion_data_ready = False
        tele_data.hand_motion_data_ready = False
        tele_data.motion_data_ready = False
        maybe_print_tracker_position(tele_data=tele_data)
        return
    if not VIVE_MANUS_READER.calibrated:
        tele_data.arm_motion_data_ready = False
        tele_data.hand_motion_data_ready = False
        tele_data.motion_data_ready = False
        now = time.monotonic()
        if now - getattr(VIVE_MANUS_READER, "_last_calibration_required_log", 0.0) > 2.0:
            logger_mp.warning("[Vive/Manus] Not calibrated. Press [c] in attention pose, then press [r] to sync.")
            VIVE_MANUS_READER._last_calibration_required_log = now
        maybe_print_tracker_position(tele_data=tele_data)
        return

    trackers_ready = VIVE_MANUS_READER.apply_relative_wrist_motion(tele_data)
    hands_ready = VIVE_MANUS_READER.apply_manus_hand_data(tele_data)
    tele_data.arm_motion_data_ready = trackers_ready
    tele_data.motion_data_ready = trackers_ready or hands_ready
    maybe_print_tracker_position(tele_data=tele_data)

    if not trackers_ready:
        now = time.monotonic()
        if now - getattr(VIVE_MANUS_READER, "_last_tracker_wait_log", 0.0) > 2.0:
            if not VIVE_MANUS_READER.relative_reference_ready:
                logger_mp.warning("[Vive/Relative] Waiting for [r] sync reference from current robot EE pose.")
            else:
                logger_mp.warning("[Vive/Relative] Keep left/right/head trackers valid.")
            VIVE_MANUS_READER._last_tracker_wait_log = now
    if not hands_ready:
        now = time.monotonic()
        if now - getattr(VIVE_MANUS_READER, "_last_manus_wait_log", 0.0) > 2.0:
            logger_mp.warning("[Manus] Waiting for valid left/right glove data.")
            VIVE_MANUS_READER._last_manus_wait_log = now


def _update_neck_control(
    args,
    neck_ctrl,
    neck_feedback,
    tele_data,
    arm_ctrl,
    loop_count,
    neck_log_last_ts,
    neck_log_interval,
    allow_waist=True,
):
    global NECK_NEUTRAL_RESET_REQUEST, NECK_CAMERA_SYNC_ACTIVE
    if neck_ctrl is None:
        return None, neck_log_last_ts
    try:
        if args.neck_input_source == "vive_head":
            if VIVE_MANUS_READER is None:
                raise ValueError("Vive/Manus reader is not ready")
            if not VIVE_MANUS_READER.calibrated:
                NECK_CAMERA_SYNC_ACTIVE = False
                raise ValueError("Vive/Manus frame is not calibrated. Press [c] first, then press [r].")
            neck_measured, neck_pose_valid = VIVE_MANUS_READER.read_head_tracker_yaw_pitch_for_neck(
                yaw_sign=args.vive_head_neck_yaw_sign,
                pitch_sign=args.vive_head_neck_pitch_sign,
            )
            if not neck_pose_valid or neck_measured is None:
                raise ValueError("Vive head tracker is not calibrated, invalid, or stale")
            neck_pose = _pose_from_yaw_pitch(neck_measured)
        else:
            if tele_data is None or not getattr(tele_data, "head_pose_is_valid", True):
                raise ValueError("Vision Pro head pose is invalid")
            neck_pose = tele_data.head_pose
            neck_measured = neck_ctrl._extract_yaw_pitch(neck_pose)

        if NECK_NEUTRAL_RESET_REQUEST:
            neck_ctrl.reset_neutral()
            NECK_NEUTRAL_RESET_REQUEST = False
            logger_mp.info("[teleop neck] neutral reset after Vive/Frame calibration.")

        neck_command, neck_target = neck_ctrl.update(neck_pose)
        neck_actual = neck_feedback.read_latest() if neck_feedback is not None else None
        neck_record = {
            "raw_head_yaw_pitch": neck_measured.tolist(),
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
        should_log_neck = neck_log_interval is not None and now - neck_log_last_ts >= neck_log_interval
        if should_log_neck:
            neck_log_last_ts = now
            waist_actual = None
            waist_error = None
            if waist_command is not None:
                waist_actual = arm_ctrl.get_waist_yaw_relative_position()
                waist_error = waist_command - waist_actual
            logger_mp.info(
                f"[teleop neck] source={args.neck_input_source} "
                f"raw_head={np.round(neck_measured, 4).tolist()} "
                f"target={np.round(neck_target, 4).tolist()} "
                f"command={np.round(neck_command, 4).tolist()} "
                f"actual={None if neck_actual is None else np.round(neck_actual.get('yaw_pitch'), 4).tolist()} "
                f"waist_yaw={None if waist_command is None else round(waist_command, 4)} "
                f"waist_actual={None if waist_actual is None else round(waist_actual, 4)} "
                f"waist_error={None if waist_error is None else round(waist_error, 4)}"
            )
        return neck_record, neck_log_last_ts
    except (ValueError, OSError) as e:
        if loop_count % 30 == 0:
            logger_mp.warning(f"[teleop neck] command skipped: {e}")
        return None, neck_log_last_ts

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
PAUSE_TO_READY = False  # True while teleop is paused by [p]; arms are driven back to the ready pose
TACTILE_VR_OVERLAY_VISIBLE = True  # Toggle RH5DG2 tactile overlay visibility in the XR viewer
# waist keyboard control (H1_2 only): [j]/[k] nudge the waist yaw left/right during teleop
WAIST_YAW_REL     = 0.0     # current relative waist-yaw target (rad, relative to startup home)
WAIST_KEY_STEP    = 0.05    # rad added/removed per [j]/[k] press; overwritten from args in main
WAIST_KEY_LIMIT   = 0.1745  # +/- clamp for the accumulator (rad); overwritten from args in main
WAIST_KEY_ENABLED = False   # set True when --enable-waist-keyboard is passed
WAIST_KEY_INVERT  = False   # swap [j]/[k] direction; overwritten from args in main
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
    global PRINT_TRACKER_POSITION, CALIBRATE_TRACKER_FRAME, CALIBRATE_TRACKER_FRAME_AT
    global START_SYNC_AT, PAUSE_TO_READY
    global NECK_CAMERA_SYNC_ACTIVE, NECK_NEUTRAL_RESET_REQUEST
    if key == 'r':
        if START:
            logger_mp.warning("[teleop] ignored [r]: teleop is already running. Press [p] to pause, or [q] to quit.")
            return
        START = False
        PAUSE_TO_READY = False
        PRINT_TRACKER_POSITION = False
        START_SYNC_AT = time.monotonic() + START_SYNC_DELAY
        NECK_CAMERA_SYNC_ACTIVE = False
        NECK_NEUTRAL_RESET_REQUEST = True
        if CALIBRATE_TRACKER_FRAME:
            # Do not drop the [r]: keep it queued so the sync fires automatically
            # (with a fresh delay) once the pending calibration completes.
            logger_mp.info(
                "[teleop sync] [r] queued while calibration is pending. "
                "Sync will start %.1fs after calibration completes.",
                START_SYNC_DELAY,
            )
        elif VIVE_MANUS_READER is not None and not VIVE_MANUS_READER.calibrated:
            logger_mp.warning(
                "[teleop start] [r] pressed before calibration. "
                "Robot start/home behavior will run in %.1fs, but arm/Manus/neck camera sync requires [c] first.",
                START_SYNC_DELAY,
            )
        else:
            logger_mp.info("[teleop sync] [r] pressed. Sync will start in %.1fs; hold the desired pose.", START_SYNC_DELAY)
    elif key == 'p':
        if START:
            # Pause teleop: drive the arms back to the ready pose and wait for [r] to resume.
            START = False
            START_SYNC_AT = None
            PAUSE_TO_READY = True
            PRINT_TRACKER_POSITION = False
            NECK_CAMERA_SYNC_ACTIVE = False
            NECK_NEUTRAL_RESET_REQUEST = True
            logger_mp.info(
                "[teleop pause] [p] pressed during teleop. Arms return to the ready pose and stay paused; "
                "press [r] to re-sync and resume."
            )
        else:
            logger_mp.warning("[teleop pause] ignored [p] because teleop is not running. Press [r] to start.")
    elif key == 'q':
        START = False
        START_SYNC_AT = None
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
    elif key == 'c':
        START = False
        START_SYNC_AT = None
        NECK_CAMERA_SYNC_ACTIVE = False
        NECK_NEUTRAL_RESET_REQUEST = True
        if VIVE_MANUS_READER is not None:
            try:
                VIVE_MANUS_READER.reset_calibration()
                VIVE_MANUS_READER.reset_head_tracker_neck_ref()
            except Exception as exc:
                logger_mp.debug("[Vive/Frame] calibration reset skipped: %s", exc)
        CALIBRATE_TRACKER_FRAME = True
        CALIBRATE_TRACKER_FRAME_AT = time.monotonic() + CALIBRATE_TRACKER_FRAME_DELAY
        logger_mp.info(
            f"[Vive/Frame] Calibration scheduled in {CALIBRATE_TRACKER_FRAME_DELAY:.1f}s. "
            "Motion and neck camera sync are paused until [r]."
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

def _make_arm_ready_q(current_q):
    q = np.zeros_like(np.asarray(current_q, dtype=np.float64).reshape(-1))
    if q.size == 0:
        return q
    half = q.size // 2
    if q.size >= 8 and half + 3 < q.size:
        q[3] = -0.3
        q[half + 3] = -0.3
    return q

def _se3_to_mat(se3):
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = np.asarray(se3.rotation, dtype=np.float64)
    mat[:3, 3] = np.asarray(se3.translation, dtype=np.float64).reshape(3)
    return mat

def _current_robot_ee_poses_from_fk(arm_ik, arm_ctrl):
    if arm_ik is None or arm_ctrl is None or not hasattr(arm_ctrl, "get_current_dual_arm_q"):
        return None, None
    try:
        q = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=np.float64).reshape(-1)
        model = arm_ik.reduced_robot.model
        data = arm_ik.reduced_robot.data
        if q.size != model.nq or not np.isfinite(q).all():
            raise ValueError(f"invalid q size={q.size}, expected={model.nq}")
        import pinocchio as pin
        pin.framesForwardKinematics(model, data, q)
        left_id = getattr(arm_ik, "L_hand_id", model.getFrameId("L_ee"))
        right_id = getattr(arm_ik, "R_hand_id", model.getFrameId("R_ee"))
        return _se3_to_mat(data.oMf[left_id]), _se3_to_mat(data.oMf[right_id])
    except Exception as exc:
        logger_mp.warning("[teleop sync] failed to compute current robot EE pose for Vive reference: %s", exc)
        return None, None

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

def _safe_enter_hand_standby_open(hand_ctrl):
    if hand_ctrl is not None and hasattr(hand_ctrl, "enter_standby_open"):
        try:
            hand_ctrl.enter_standby_open()
            return True
        except Exception as exc:
            logger_mp.debug("[teleop hand standby] enter_standby_open failed: %s", exc)
    return False


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

def _safe_enter_hand_auto(hand_ctrl):
    if hand_ctrl is not None and hasattr(hand_ctrl, "enter_auto"):
        try:
            hand_ctrl.enter_auto()
            return True
        except Exception as exc:
            logger_mp.debug("[teleop hand standby] enter_auto failed: %s", exc)
    return False

def _apply_pose_jump_filter(left_pose, right_pose, last_left, last_right, max_frame_jump):
    left = np.asarray(left_pose, dtype=np.float64).copy()
    right = np.asarray(right_pose, dtype=np.float64).copy()
    if left.shape != (4, 4) or right.shape != (4, 4):
        return left_pose, right_pose, last_left, last_right, None, None
    if last_left is None or last_right is None:
        return left, right, left.copy(), right.copy(), 0.0, 0.0

    left_jump = float(np.linalg.norm(left[:3, 3] - last_left[:3, 3]))
    right_jump = float(np.linalg.norm(right[:3, 3] - last_right[:3, 3]))
    if left_jump > max_frame_jump:
        left = last_left.copy()
    else:
        last_left = left.copy()
    if right_jump > max_frame_jump:
        right = last_right.copy()
    else:
        last_right = right.copy()
    return left, right, last_left, last_right, left_jump, right_jump

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 20.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1', 'H2'], default='H1_2', help='Select arm controller')
    parser.add_argument('--joint-temperature-interval', type=float, default=0.0, help='Seconds between arm-joint temperature logs from rt/lowstate. Set a positive value to enable (default: disabled).')
    parser.add_argument('--comm-log', action=argparse.BooleanOptionalAction, default=True, help='Write a per-run JSONL log of arm/hand commands, states, DDS health, and motor temperatures (default: enabled).')
    parser.add_argument('--comm-log-dir', type=str, default=None, help='Directory for communication/state logs (default: <teleop>/logs/comm_state).')
    parser.add_argument('--comm-log-temp-interval', type=float, default=5.0, help='Seconds between motor-temperature records in the comm log (default: 5.0; 0 disables).')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'inspire_dg2', 'rh5dg2_ftp', 'rh5dg2_dfx', 'rh56f1', 'brainco'], default='rh5dg2_dfx', help='Select end effector controller')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--viewer-host-ip', type=str, default=None, help='Host IP advertised to the XR browser for the HTTPS/WSS viewer. If omitted, infer it from the route to --img-server-ip.')
    parser.add_argument('--network-interface', type=str, default='enp44s0', help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    # Vive Tracker / Manus ROS2 input parameters
    parser.add_argument('--left-tracker-name', type=str, default='LHR-36B992CB', help='libsurvive TF child frame for the left wrist tracker')
    parser.add_argument('--right-tracker-name', type=str, default='LHR-0FAD1369', help='libsurvive TF child frame for the right wrist tracker')
    parser.add_argument('--head-tracker-name', type=str, default='LHR-1F621773', help='libsurvive TF child frame for the head tracker; empty string disables head tracking')
    parser.add_argument('--libsurvive-tracking-frame', type=str, default='libsurvive_world', help='libsurvive TF parent frame')
    parser.add_argument('--ros-stale-timeout', type=float, default=0.5, help='Seconds before ROS2 tracker/glove data is considered stale')
    parser.add_argument('--vive-transport', type=str, choices=['tf', 'udp_json'], default='udp_json', help='How to receive Vive tracker poses. udp_json reads JSON packets from vive_tf_to_udp.py.')
    parser.add_argument('--vive-udp-host', type=str, default='0.0.0.0', help='UDP bind host for --vive-transport udp_json.')
    parser.add_argument('--vive-udp-port', type=int, default=56130, help='UDP bind port for --vive-transport udp_json.')
    parser.add_argument('--manus-transport', type=str, choices=['ros2', 'udp_json'], default='udp_json', help='How to receive Manus hand landmarks. udp_json reads JSON packets from manus_ros2_to_udp.py.')
    parser.add_argument('--manus-udp-host', type=str, default='0.0.0.0', help='UDP bind host for --manus-transport udp_json.')
    parser.add_argument('--manus-udp-port', type=int, default=56120, help='UDP bind port for --manus-transport udp_json.')
    parser.add_argument('--manus-topics', type=str, nargs='+', default=['manus_glove_0', 'manus_glove_1'], help='ROS2 Manus glove topics; msg.side or topic name must identify left/right')
    parser.add_argument('--manus-msg-type', type=str, default='manus_ros2_msgs/msg/ManusGlove', help='ROS2 message type for Manus glove data')
    parser.add_argument('--manus-hand-transform', type=str, choices=['legacy', 'televuer'], default='legacy', help='Manus wrist-local hand axis transform before retargeting. legacy is the correct Manus raw-skeleton transform; televuer is accepted only as a deprecated alias.')
    parser.add_argument('--enable-manus-haptics', action=argparse.BooleanOptionalAction, default=True, help='Map RH5DG2 fingertip normal force to MANUS finger vibration through the UDP bridge (default: enabled).')
    parser.add_argument('--manus-haptic-host', type=str, default=os.getenv('MANUS_HAPTIC_HOST', '192.168.123.54'), help='Host running manus_ros2_to_udp.py with its haptic listener enabled. Defaults to MANUS_HAPTIC_HOST or 192.168.123.54.')
    parser.add_argument('--manus-haptic-port', type=int, default=56121, help='UDP haptic command port on manus_ros2_to_udp.py.')
    parser.add_argument('--manus-haptic-hz', type=float, default=20.0, help='MANUS haptic command heartbeat frequency.')
    parser.add_argument('--manus-haptic-baseline-seconds', type=float, default=0.5, help='Quiet normal-force baseline duration after haptics become active.')
    parser.add_argument('--manus-haptic-ema-alpha', type=float, default=0.9, help='EMA alpha for fingertip normal force before vibration mapping.')
    parser.add_argument('--manus-haptic-deadband', type=float, default=1.0, help='Baseline-corrected normal-force deadband.')
    parser.add_argument('--manus-haptic-normal-max', type=float, default=1000.0, help='Normal force value mapped to maximum MANUS vibration.')
    parser.add_argument('--manus-haptic-gamma', type=float, default=1.0, help='Exponent applied after normalizing MANUS vibration power.')
    parser.add_argument('--manus-haptic-debug-rate', type=float, default=0.0, help='Periodic raw fingertip normal force log rate in Hz. Set 0 to disable.')
    parser.add_argument('--start-sync-delay', type=float, default=START_SYNC_DELAY, help='Seconds to wait after pressing [r] before enabling robot/Manus/neck sync.')
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
    parser.add_argument('--arm-standby-on-tracking-loss', action=argparse.BooleanOptionalAction, default=True, help='Enter arm standby when XR head/wrist tracking becomes invalid.')
    parser.add_argument('--arm-lost-timeout', type=float, default=0.5, help='Seconds of invalid XR arm tracking before entering standby.')
    parser.add_argument('--arm-found-confirm', type=float, default=0.5, help='Seconds of valid XR arm tracking required before leaving standby.')
    parser.add_argument('--arm-standby-action', type=str, choices=['ready', 'hold'], default='ready', help='Arm target while XR tracking is lost: ready pose or current-pose hold.')
    parser.add_argument('--arm-max-frame-jump', type=float, default=0.15, help='Reject valid XR wrist pose frames whose translation jumps more than this many meters from the last good pose. Set 0 to disable.')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--headless', action=argparse.BooleanOptionalAction, default=True, help='Enable headless mode and disable Rerun recording visualization by default. Use --no-headless to enable Rerun.')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--disable-arm', action='store_true', help='Disable arm IK/control while keeping XR and hand paths alive.')
    parser.add_argument('--disable-arm-tracking', action='store_true', help='Initialize and hold the arms at their start pose while keeping the normal Vive calibration/sync and Manus hand-control flow.')
    parser.add_argument('--disable-body', action=argparse.BooleanOptionalAction, default=True, help='Disable high-level body/loco command publishing.')
    parser.add_argument('--hand-only', action='store_true', help='Run XR input and end-effector hand control only; arm and body control stay off.')
    parser.add_argument('--enable-neck', action=argparse.BooleanOptionalAction, default=True, help='Send Vision Pro head yaw/pitch to the external UDP neck controller.')
    parser.add_argument('--neck-host', type=str, default=None, help='External neck controller host. Defaults to --img-server-ip.')
    parser.add_argument('--neck-port', type=int, default=9091, help='External neck controller UDP command port.')
    parser.add_argument('--neck-yaw-limit', type=float, default=1.2, help='Absolute relative neck yaw command limit in radians.')
    parser.add_argument('--neck-pitch-limit', type=float, default=0.8, help='Absolute relative neck pitch command limit in radians.')
    parser.add_argument('--neck-smoothing-alpha', type=float, default=0.25, help='Neck command low-pass alpha from 0 to 1.')
    parser.add_argument('--neck-max-step', type=float, default=0.08, help='Maximum neck command change per control frame in radians.')
    parser.add_argument('--neck-command-deadband', type=float, default=0.04, help='Minimum yaw/pitch command change in radians before sending a new neck UDP command. Set 0 to send every frame.')
    parser.add_argument('--neck-feedback-port', type=int, default=9093, help='UDP port that receives actual neck yaw,pitch feedback from the pan/tilt process.')
    parser.add_argument('--neck-log-rate', type=float, default=0.0, help='Neck debug log rate in Hz. Set 0 to disable periodic neck logs.')
    parser.add_argument('--neck-input-source', type=str, choices=['visionpro', 'vive_head'], default='vive_head', help='Head orientation source for neck/camera control. vive_head maps calibrated Vive head tracker relative Y->yaw and X->pitch, ignoring roll.')
    parser.add_argument('--vive-head-neck-yaw-sign', type=float, choices=[-1.0, 1.0], default=-1.0, help='Sign for Vive head tracker Y-axis yaw when --neck-input-source vive_head.')
    parser.add_argument('--vive-head-neck-pitch-sign', type=float, choices=[-1.0, 1.0], default=-1.0, help='Sign for Vive head tracker X-axis pitch when --neck-input-source vive_head.')
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
    parser.add_argument('--loop', action='store_true', default=True, help='Stream robot state + cameras to the Config Loop sidecar (default: on)')
    parser.add_argument('--no-loop', dest='loop', action='store_false', help='Disable Config Loop streaming')
    parser.add_argument('--loop-addr', type=str, default='127.0.0.1:5590', help='Config Loop sidecar TCP host:port')
    parser.add_argument('--loop-hand-name', type=str, default='rh5dg2', help='Config Loop source name for the separate RH5DG2 tactile stream.')
    # record mode and task info
    parser.add_argument('--record', action=argparse.BooleanOptionalAction, default=True, help='Enable data recording mode')
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
    if args.ros_stale_timeout <= 0.0:
        raise ValueError("--ros-stale-timeout must be greater than zero.")
    if args.vive_transport == "udp_json" and args.vive_udp_port <= 0:
        raise ValueError("--vive-udp-port must be greater than zero.")
    if args.manus_transport == "udp_json" and args.manus_udp_port <= 0:
        raise ValueError("--manus-udp-port must be greater than zero.")
    if args.enable_manus_haptics:
        if not args.manus_haptic_host:
            raise ValueError("--manus-haptic-host must not be empty.")
        if not 1 <= args.manus_haptic_port <= 65535:
            raise ValueError("--manus-haptic-port must be between 1 and 65535.")
        if args.manus_haptic_hz <= 0.0:
            raise ValueError("--manus-haptic-hz must be greater than zero.")
        if args.manus_haptic_baseline_seconds < 0.0:
            raise ValueError("--manus-haptic-baseline-seconds must be zero or greater.")
        if not 0.0 < args.manus_haptic_ema_alpha <= 1.0:
            raise ValueError("--manus-haptic-ema-alpha must be greater than 0 and at most 1.")
        if args.manus_haptic_deadband < 0.0:
            raise ValueError("--manus-haptic-deadband must be zero or greater.")
        if args.manus_haptic_normal_max <= 0.0:
            raise ValueError("--manus-haptic-normal-max must be greater than zero.")
        if args.manus_haptic_gamma <= 0.0:
            raise ValueError("--manus-haptic-gamma must be greater than zero.")
        if args.manus_haptic_debug_rate < 0.0:
            raise ValueError("--manus-haptic-debug-rate must be zero or greater.")
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
    if args.neck_input_source == "vive_head" and not args.enable_neck:
        logger_mp.warning("[teleop neck] --neck-input-source vive_head is set but neck control is disabled.")
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
    if args.ee == "rh56f1" and args.arm != "H1_2":
        raise ValueError("--ee rh56f1 currently supports the H1_2 arm path only; use --arm H1_2.")
    rh5dg2_safe_baseline = _resolve_rh5dg2_safe_baseline(args.rh5dg2_safe_baseline)
    if args.hand_only:
        args.disable_arm = True
        args.disable_body = True
    if args.disable_arm_tracking and args.disable_arm:
        raise ValueError("--disable-arm-tracking requires the arm controller for the c/r sync flow; do not combine it with --disable-arm or --hand-only.")
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
    if args.hand_only or args.disable_arm or args.disable_arm_tracking or args.disable_body:
        logger_mp.warning(
            "[teleop safety mode] hand_only=%s disable_arm=%s disable_arm_tracking=%s disable_body=%s motion_requested=%s",
            args.hand_only,
            args.disable_arm,
            args.disable_arm_tracking,
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
    arm_ik = None
    hand_ctrl = None
    comm_logger = None
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
    manus_haptic_mapper = None
    manus_haptic_sender = None
    manus_haptics_active = False
    manus_haptic_last_log = 0.0
    manus_haptic_last_warning = 0.0
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
                f"[teleop neck] enabled target={args.neck_host or args.img_server_ip}:{args.neck_port} "
                f"yaw_limit={args.neck_yaw_limit:.3f} pitch_limit={args.neck_pitch_limit:.3f} "
                f"alpha={args.neck_smoothing_alpha:.3f} max_step={args.neck_max_step:.3f} "
                f"deadband={args.neck_command_deadband:.3f} "
                f"input_source={args.neck_input_source} "
                f"vive_yaw_sign={args.vive_head_neck_yaw_sign:.0f} "
                f"vive_pitch_sign={args.vive_head_neck_pitch_sign:.0f}"
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

        if args.enable_manus_haptics:
            manus_haptic_mapper = ManusNormalForceMapper(
                baseline_seconds=args.manus_haptic_baseline_seconds,
                ema_alpha=args.manus_haptic_ema_alpha,
                deadband=args.manus_haptic_deadband,
                normal_max=args.manus_haptic_normal_max,
                gamma=args.manus_haptic_gamma,
            )
            manus_haptic_sender = ManusHapticUDPSender(
                host=args.manus_haptic_host,
                port=args.manus_haptic_port,
                send_hz=args.manus_haptic_hz,
            )
            manus_haptic_sender.stop(force=True)
            logger_mp.info(
                "[MANUS haptics] enabled target=udp://%s:%s hz=%.2f "
                "baseline=%.3fs alpha=%.3f deadband=%.3f normal_max=%.3f gamma=%.3f",
                args.manus_haptic_host,
                args.manus_haptic_port,
                args.manus_haptic_hz,
                args.manus_haptic_baseline_seconds,
                args.manus_haptic_ema_alpha,
                args.manus_haptic_deadband,
                args.manus_haptic_normal_max,
                args.manus_haptic_gamma,
            )

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
                                     tracking_timeout=args.arm_lost_timeout,
                                     session_timeout=max(2.0, args.arm_lost_timeout * 4.0),
                                     )

        VIVE_MANUS_READER = ViveManusInfoReader(
            left_tracker_name=args.left_tracker_name,
            right_tracker_name=args.right_tracker_name,
            head_tracker_name=args.head_tracker_name,
            libsurvive_tracking_frame=args.libsurvive_tracking_frame,
            manus_topics=args.manus_topics,
            manus_msg_type=args.manus_msg_type,
            manus_transport=args.manus_transport,
            manus_udp_host=args.manus_udp_host,
            manus_udp_port=args.manus_udp_port,
            vive_transport=args.vive_transport,
            vive_udp_host=args.vive_udp_host,
            vive_udp_port=args.vive_udp_port,
            stale_timeout=args.ros_stale_timeout,
            manus_hand_transform=args.manus_hand_transform,
        )
        logger_mp.info(
            "[Vive/Manus] reader initialized: left_tracker=%s right_tracker=%s head_tracker=%s "
            "vive_transport=%s manus_transport=%s manus_topics=%s",
            args.left_tracker_name,
            args.right_tracker_name,
            args.head_tracker_name,
            args.vive_transport,
            args.manus_transport,
            args.manus_topics,
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
            arm_ik = H1_2_ArmIK(scale_input_poses=False)
            arm_ctrl = H1_2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1":
            arm_ik = H1_ArmIK(scale_input_poses=False)
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
                thumb_curl_gain=args.inspire_dg2_thumb_curl_gain,
                right_thumb_curl_gain=args.inspire_dg2_right_thumb_curl_gain,
                thumb_curl_threshold=args.inspire_dg2_thumb_curl_threshold,
                thumb_curl_strength=args.inspire_dg2_thumb_curl_strength,
                thumb_curl_log_rate=args.inspire_dg2_thumb_curl_log_rate,
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
            logger_mp.info(
                "[teleop InspireDG2 thumb] curl_gain=%.3f right_gain=%.3f threshold=%.3f strength=%.3f log_rate=%.3f",
                args.inspire_dg2_thumb_curl_gain,
                args.inspire_dg2_right_thumb_curl_gain,
                args.inspire_dg2_thumb_curl_threshold,
                args.inspire_dg2_thumb_curl_strength,
                args.inspire_dg2_thumb_curl_log_rate,
            )
        if args.ee == "rh56f1":
            logger_mp.info(
                f"[teleop hand side mapping] ee={args.ee} "
                f"rh56f1_retarget_mode={args.rh56f1_retarget_mode} "
                "scope=hand_landmarks_only arm_wrist_pose_uses_arm_sensitivity=True"
            )

        # Unified EE handles for Config Loop streaming: point at whichever
        # state/action arrays this --ee selection created (left+right concatenated),
        # so the main loop reads the EE once regardless of hand/gripper type.
        loop_ee_state_array = None
        loop_ee_action_array = None
        loop_ee_lock = None
        if args.ee == "dex1":
            loop_ee_state_array = dual_gripper_state_array
            loop_ee_action_array = dual_gripper_action_array
            loop_ee_lock = dual_gripper_data_lock
        elif args.ee in ("dex3", "inspire_dfx", "inspire_ftp", "inspire_dg2",
                         "rh5dg2_dfx", "rh5dg2_ftp", "rh56f1", "brainco"):
            loop_ee_state_array = dual_hand_state_array
            loop_ee_action_array = dual_hand_action_array
            loop_ee_lock = dual_hand_data_lock

        # per-run communication / action / state log (JSONL)
        if args.comm_log:
            comm_log_dir = args.comm_log_dir or os.path.join(current_dir, "logs", "comm_state")
            try:
                comm_logger = CommStateLogger(
                    comm_log_dir,
                    meta={
                        "script": os.path.basename(__file__),
                        "arm": args.arm,
                        "ee": args.ee,
                        "frequency": args.frequency,
                        "sim": bool(args.sim),
                        "motion": bool(args.motion),
                        "cmd_topic": "rt/arm_sdk" if args.motion else "rt/lowcmd",
                        "hostname": socket.gethostname(),
                        "argv": sys.argv[1:],
                    },
                    arm_ctrl=arm_ctrl,
                    temperature_interval=args.comm_log_temp_interval,
                )
            except Exception as exc:
                logger_mp.error("[comm log] failed to open log, continuing without it: %s", exc)
                comm_logger = None

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
                "manus_haptics": {
                    "enabled": args.enable_manus_haptics,
                    "target_host": args.manus_haptic_host,
                    "target_port": args.manus_haptic_port,
                    "hz": args.manus_haptic_hz,
                    "baseline_seconds": args.manus_haptic_baseline_seconds,
                    "ema_alpha": args.manus_haptic_ema_alpha,
                    "deadband": args.manus_haptic_deadband,
                    "normal_max": args.manus_haptic_normal_max,
                    "gamma": args.manus_haptic_gamma,
                    "finger_order": ["thumb", "index", "middle", "ring", "little"],
                },
                "neck": {
                    "enabled": args.enable_neck,
                    "input_source": args.neck_input_source,
                    "command_host": args.neck_host or args.img_server_ip,
                    "command_port": args.neck_port,
                    "feedback_port": args.neck_feedback_port,
                    "command_deadband": args.neck_command_deadband,
                    "vive_head_axis_mapping": {
                        "yaw": "tracker_euler_y",
                        "pitch": "tracker_euler_x",
                        "roll": "ignored",
                        "yaw_sign": args.vive_head_neck_yaw_sign,
                        "pitch_sign": args.vive_head_neck_pitch_sign,
                    },
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
        logger_mp.info(f"🟢  Press [r] to start syncing the robot with your movements after {START_SYNC_DELAY:.1f}s.")
        logger_mp.info("🟣  Press [p] while running to PAUSE teleop and return the arms to the ready pose; press [r] to re-sync and resume.")
        logger_mp.info("🟠  Press [c] in attention pose to calibrate Vive tracker frame; [r] pressed during calibration is queued and syncs automatically once calibration completes.")
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
        prestart_neck_loop_count = 0
        neck_log_last_ts = 0.0
        neck_log_interval = 1.0 / args.neck_log_rate if args.neck_log_rate > 0 else None
        while not START and not STOP: # wait for start or stop signal.
            prestart_neck_loop_count += 1
            time.sleep(0.033)
            joint_temperature_last_log = _maybe_log_arm_joint_temperatures(
                args, arm_ctrl, joint_temperature_last_log
            )
            tele_data = tv_wrapper.get_tele_data()
            maybe_print_tracker_position(tele_data=tele_data)
            maybe_calibrate_tracker_frame_and_pause(tele_data=tele_data)
            maybe_start_delayed_sync(arm_ik=arm_ik, arm_ctrl=arm_ctrl)
            if NECK_CAMERA_SYNC_ACTIVE:
                _, neck_log_last_ts = _update_neck_control(
                    args,
                    neck_ctrl,
                    neck_feedback,
                    tele_data,
                    arm_ctrl,
                    prestart_neck_loop_count,
                    neck_log_last_ts,
                    neck_log_interval,
                    allow_waist=False,
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

        maybe_print_tracker_position(tele_data=tv_wrapper.get_tele_data())
        logger_mp.info("---------------------🚀start Tracking🚀-------------------------")
        logger_mp.info(
            f"[teleop ready state] READY={READY} START={START} STOP={STOP} "
            f"disable_arm={args.disable_arm} disable_arm_tracking={args.disable_arm_tracking}"
        )
        arm_tracking_hold_q = None
        if args.disable_arm:
            logger_mp.warning("[teleop arm disabled reason] --disable-arm set; IK/control publish will be skipped.")
        else:
            arm_ctrl.speed_gradual_max()
            # Seed the arm target with the current (ready) pose so START eases up from
            # where the arms already are, instead of the arms first snapping toward the
            # zeros/home target and dropping "as if torque off" before rising to the IK pose.
            try:
                _seed_q = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=np.float64).reshape(-1)
                if _seed_q.size and np.isfinite(_seed_q).all():
                    arm_ctrl.ctrl_dual_arm(_seed_q, np.zeros_like(_seed_q))
                    if args.disable_arm_tracking:
                        arm_tracking_hold_q = _seed_q.copy()
            except Exception as _seed_exc:
                logger_mp.debug("[teleop arm startup] q_target seed skipped: %s", _seed_exc)
            logger_mp.info(
                f"[teleop arm startup safety] reset wrist base at START; "
                f"startup_duration={args.arm_startup_duration:.3f}s "
                f"startup_max_step={args.arm_startup_max_step:.4f}rad"
            )
            if args.disable_arm_tracking:
                logger_mp.warning(
                    "[teleop arm tracking] OFF: arms will hold the start pose; "
                    "Vive c/r calibration, Manus hand control, and tactile feedback remain active."
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
        arm_fsm = "ACTIVE"
        arm_lost_since = None
        arm_found_since = None
        arm_last_good_left_pose = None
        arm_last_good_right_pose = None
        arm_standby_logged = False
        logger_mp.info(
            f"[teleop arm sensitivity config] pos_gain={np.round(arm_sensitivity_config['pos_gain'], 4).tolist()} "
            f"rot_gain={arm_sensitivity_config['rot_gain']:.3f} "
            f"max_delta={arm_sensitivity_config['max_delta']:.3f} "
            f"smoothing_alpha={arm_sensitivity_config['smoothing_alpha']:.3f} "
            f"enabled={arm_sensitivity_config.get('enabled', False)} sim={args.sim}"
        )
        logger_mp.info(
            f"[teleop arm tracking fsm] enabled={args.arm_standby_on_tracking_loss and not args.disable_arm_tracking} "
            f"lost_timeout={args.arm_lost_timeout:.3f}s found_confirm={args.arm_found_confirm:.3f}s "
            f"standby_action={args.arm_standby_action} max_frame_jump={args.arm_max_frame_jump:.3f}m"
        )

        # main loop. robot start to follow VR user's motion
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
            # get image ( --loop also needs frames every tick, independent of record )
            if camera_config['head_camera']['enable_zmq']:
                if args.record or args.screen_record or args.loop or (xr_need_local_img and selected_camera_name == "head"):
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
                if args.record or args.loop:
                    left_wrist_img = img_client.get_left_wrist_frame()
                    if left_wrist_img is not None and left_wrist_img.bgr is not None and cv2 is not None:
                        # Store the camera frame exactly as received from the image server.
                        left_wrist_bgr = left_wrist_img.bgr

            if camera_config['right_wrist_camera']['enable_zmq']:
                if args.record or args.loop:
                    right_wrist_img = img_client.get_right_wrist_frame()
                    if right_wrist_img is not None and right_wrist_img.bgr is not None and cv2 is not None:
                        # Store the camera frame exactly as received from the image server.
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
            tele_data = tv_wrapper.get_tele_data()
            maybe_start_delayed_sync(arm_ik=arm_ik, arm_ctrl=arm_ctrl)
            _apply_vive_manus_input(tele_data)
            if not START or CALIBRATE_TRACKER_FRAME:
                if manus_haptic_sender is not None:
                    try:
                        manus_haptic_sender.stop()
                    except OSError as exc:
                        now = time.monotonic()
                        if now - manus_haptic_last_warning >= 2.0:
                            logger_mp.warning("[MANUS haptics] failed to send stop command: %s", exc)
                            manus_haptic_last_warning = now
                    manus_haptics_active = False
                arm_last_good_left_pose = None
                arm_last_good_right_pose = None
                arm_sensitivity_state.clear()
                arm_tracking_hold_q = None
                tracking_start_time = time.time()
                if PAUSE_TO_READY and not args.disable_arm and arm_ctrl is not None:
                    # [p] pause: ease the arms back to the ready pose while teleop is suspended.
                    try:
                        pause_current_q = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=np.float64).reshape(-1)
                        if pause_current_q.size and np.isfinite(pause_current_q).all():
                            pause_ready_q = _make_arm_ready_q(pause_current_q)
                            pause_max_step = max(0.0, args.arm_startup_max_step)
                            pause_target_q = pause_current_q + np.clip(
                                pause_ready_q - pause_current_q, -pause_max_step, pause_max_step
                            )
                            arm_ctrl.ctrl_dual_arm(pause_target_q, np.zeros_like(pause_target_q))
                    except Exception as exc:
                        if loop_count % 30 == 0:
                            logger_mp.warning("[teleop pause] ready pose command skipped: %s", exc)
                    _safe_enter_hand_standby_open(hand_ctrl)
                    if loop_count % 100 == 0:
                        logger_mp.info("[teleop pause] paused at ready pose. Press [r] to re-sync and resume.")
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / args.frequency) - time_elapsed)
                time.sleep(sleep_time)
                logger_mp.debug(f"main process sleep: {sleep_time}")
                continue
            if VIVE_MANUS_READER is not None:
                arm_tracking_ready = bool(getattr(tele_data, "arm_motion_data_ready", False))
            else:
                arm_tracking_ready = bool(
                    getattr(tele_data, "tracking_active", True)
                    and getattr(tele_data, "head_pose_is_valid", True)
                    and getattr(tele_data, "left_arm_is_valid", True)
                    and getattr(tele_data, "right_arm_is_valid", True)
                )
            raw_arm_tracking_ready = arm_tracking_ready
            now = time.time()
            if (
                args.arm_standby_on_tracking_loss
                and not args.disable_arm
                and not args.disable_arm_tracking
                and arm_ctrl is not None
            ):
                if raw_arm_tracking_ready:
                    arm_lost_since = None
                    if arm_fsm == "STANDBY":
                        if arm_found_since is None:
                            arm_found_since = now
                        elif now - arm_found_since >= args.arm_found_confirm:
                            arm_fsm = "ACTIVE"
                            arm_found_since = None
                            arm_last_good_left_pose = None
                            arm_last_good_right_pose = None
                            arm_sensitivity_state.clear()
                            tracking_start_time = now
                            arm_standby_logged = False
                            arm_ctrl.speed_gradual_max()
                            _safe_enter_hand_auto(hand_ctrl)
                            logger_mp.warning("[teleop arm tracking fsm] Tracking restored -> ACTIVE")
                            if comm_logger is not None:
                                comm_logger.log_event("arm_fsm", state="ACTIVE", loop=loop_count)
                    else:
                        arm_found_since = None
                else:
                    arm_found_since = None
                    if arm_lost_since is None:
                        arm_lost_since = now
                    elif arm_fsm == "ACTIVE" and now - arm_lost_since >= args.arm_lost_timeout:
                        arm_fsm = "STANDBY"
                        arm_last_good_left_pose = None
                        arm_last_good_right_pose = None
                        arm_sensitivity_state.clear()
                        _safe_enter_hand_standby_open(hand_ctrl)
                        if comm_logger is not None:
                            comm_logger.log_event("arm_fsm", state="STANDBY", loop=loop_count)
                        logger_mp.warning(
                            "[teleop arm tracking fsm] Tracking lost -> STANDBY "
                            "head_valid=%s left_arm_valid=%s right_arm_valid=%s",
                            getattr(tele_data, "head_pose_is_valid", None),
                            getattr(tele_data, "left_arm_is_valid", None),
                            getattr(tele_data, "right_arm_is_valid", None),
                        )
                arm_tracking_ready = raw_arm_tracking_ready and arm_fsm == "ACTIVE"
            latest_tactiles = None
            if rh5dg2_tactile_udp is not None:
                latest_tactiles = rh5dg2_tactile_udp.read_latest()
            if not latest_tactiles and hand_ctrl is not None and hasattr(hand_ctrl, "read_latest_tactile"):
                latest_tactiles = hand_ctrl.read_latest_tactile()
            if manus_haptic_sender is not None and manus_haptic_mapper is not None:
                haptic_motion_ready = (
                    bool(getattr(tele_data, "hand_motion_data_ready", False))
                    if args.disable_arm_tracking
                    else arm_tracking_ready
                )
                haptic_ready = bool(
                    haptic_motion_ready
                    and latest_tactiles
                    and not latest_tactiles.get("_stale", False)
                )
                try:
                    if haptic_ready:
                        if not manus_haptics_active:
                            manus_haptic_mapper.reset()
                            manus_haptics_active = True
                        stale_sides = latest_tactiles.get("_stale_sides", ())
                        for stale_side in stale_sides:
                            manus_haptic_mapper.reset(str(stale_side).removesuffix("_ee"))
                        haptic_powers = manus_haptic_mapper.update(
                            latest_tactiles,
                            stale_sides=stale_sides,
                        )
                        manus_haptic_sender.send(haptic_powers)
                        if args.manus_haptic_debug_rate > 0.0:
                            now = time.monotonic()
                            if now - manus_haptic_last_log >= 1.0 / args.manus_haptic_debug_rate:
                                manus_haptic_last_log = now
                                force_debug = manus_haptic_mapper.debug_snapshot()
                                logger_mp.info(
                                    "[MANUS raw normal force] finger_order=%s left=%s right=%s stale_sides=%s",
                                    ["thumb", "index", "middle", "ring", "little"],
                                    np.round(force_debug["left"]["raw_normal"], 1).tolist(),
                                    np.round(force_debug["right"]["raw_normal"], 1).tolist(),
                                    list(stale_sides),
                                )
                    else:
                        manus_haptic_sender.stop()
                        manus_haptics_active = False
                except (OSError, ValueError) as exc:
                    now = time.monotonic()
                    if now - manus_haptic_last_warning >= 2.0:
                        logger_mp.warning("[MANUS haptics] update skipped: %s", exc)
                        manus_haptic_last_warning = now
                    try:
                        manus_haptic_sender.stop()
                    except OSError:
                        pass
                    manus_haptics_active = False
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
            neck_record, neck_log_last_ts = _update_neck_control(
                args,
                neck_ctrl,
                neck_feedback,
                tele_data,
                arm_ctrl,
                loop_count,
                neck_log_last_ts,
                neck_log_interval,
                allow_waist=True,
            )

            # keyboard-driven H1_2 waist yaw ([j]/[k]); independent of neck tracking
            if WAIST_KEY_ENABLED and arm_ctrl is not None and hasattr(arm_ctrl, "ctrl_waist_yaw"):
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

            # [수정 부분: 강제 Swap 로직 제거하고 있는 그대로(Left->Left, Right->Right) 할당]
            left_hand_pos = tele_data.left_hand_pos
            right_hand_pos = tele_data.right_hand_pos
            if args.ee in ("rh5dg2_dfx", "rh5dg2_ftp") and args.rh5dg2_hand_swap:
                left_hand_pos, right_hand_pos = right_hand_pos, left_hand_pos

            left_wrist_pose = tele_data.left_wrist_pose
            right_wrist_pose = tele_data.right_wrist_pose
            if (
                args.arm_max_frame_jump > 0.0
                and raw_arm_tracking_ready
                and arm_fsm == "ACTIVE"
                and not args.disable_arm
                and not args.disable_arm_tracking
            ):
                (
                    left_wrist_pose,
                    right_wrist_pose,
                    arm_last_good_left_pose,
                    arm_last_good_right_pose,
                    left_jump,
                    right_jump,
                ) = _apply_pose_jump_filter(
                    left_wrist_pose,
                    right_wrist_pose,
                    arm_last_good_left_pose,
                    arm_last_good_right_pose,
                    args.arm_max_frame_jump,
                )
                if left_jump is not None and (
                    left_jump > args.arm_max_frame_jump or right_jump > args.arm_max_frame_jump
                ):
                    logger_mp.warning(
                        "[teleop arm pose filter] rejected wrist jump "
                        "left=%.3fm right=%.3fm max=%.3fm",
                        left_jump,
                        right_jump,
                        args.arm_max_frame_jump,
                    )
            left_wrist_pose, right_wrist_pose, left_arm_sens_debug, right_arm_sens_debug = _apply_arm_sensitivity(
                left_wrist_pose,
                right_wrist_pose,
                arm_sensitivity_state,
                arm_sensitivity_config,
                enabled=not args.disable_arm and not args.disable_arm_tracking and arm_tracking_ready,
            )
            if loop_count % 50 == 0:
                logger_mp.debug(
                    f"[teleop arm input] ready={READY} start={START} "
                    f"tracking_ready={arm_tracking_ready} "
                    f"tracking_active={getattr(tele_data, 'tracking_active', None)} "
                    f"session_alive={getattr(tele_data, 'session_alive', None)} "
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
                raw_hand_tracking_ready = raw_arm_tracking_ready
                if VIVE_MANUS_READER is not None:
                    raw_hand_tracking_ready = bool(getattr(tele_data, "hand_motion_data_ready", False))
                if should_hand_debug:
                    logger_mp.info(
                        f"[teleop hand input before write] ee={args.ee} input={args.input_mode} "
                        f"left={_fmt_hand_debug(left_hand_pos)} right={_fmt_hand_debug(right_hand_pos)} "
                        f"timestamp={now:.6f}"
                    )
                    logger_mp.info(
                        "[teleop hand tracking status] "
                        f"hand_tracking_ready={hand_status['hand_tracking_ready']} "
                        f"raw_hand_tracking_ready={raw_hand_tracking_ready} "
                        f"raw_arm_tracking_ready={raw_arm_tracking_ready} "
                        f"left_allzero={hand_status['left_allzero']} "
                        f"right_allzero={hand_status['right_allzero']} "
                        f"left_valid_points={hand_status['left_valid_points']} "
                        f"right_valid_points={hand_status['right_valid_points']}"
                    )
                hand_tracking_ready = hand_status["hand_tracking_ready"] and raw_hand_tracking_ready
                if hand_tracking_ready:
                    _safe_enter_hand_auto(hand_ctrl)
                    with left_hand_pos_array.get_lock():
                        left_hand_pos_array[:] = left_hand_pos.flatten()
                        left_shared_debug = np.array(left_hand_pos_array[:]).reshape(25, 3).copy() if should_hand_debug else None
                    with right_hand_pos_array.get_lock():
                        right_hand_pos_array[:] = right_hand_pos.flatten()
                        right_shared_debug = np.array(right_hand_pos_array[:]).reshape(25, 3).copy() if should_hand_debug else None
                    if args.ee in ("rh5dg2_dfx", "rh5dg2_ftp"):
                        with hand_input_timestamp.get_lock():
                            hand_input_timestamp.value = now
                else:
                    left_shared_debug = None
                    right_shared_debug = None
                    _safe_enter_hand_standby_open(hand_ctrl)
                    if loop_count % 10 == 0:
                        logger_mp.warning(
                            "[teleop hand tracking hold] skipped invalid hand frame "
                            "raw_hand_tracking_ready=%s raw_arm_tracking_ready=%s hand_tracking_ready=%s",
                            raw_hand_tracking_ready,
                            raw_arm_tracking_ready,
                            hand_status["hand_tracking_ready"],
                        )
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
            elif args.disable_arm_tracking:
                current_q = np.asarray(current_lr_arm_q, dtype=np.float64).reshape(-1)
                if (
                    arm_tracking_hold_q is None
                    or arm_tracking_hold_q.shape != current_q.shape
                    or not np.isfinite(arm_tracking_hold_q).all()
                ):
                    arm_tracking_hold_q = current_q.copy()
                    logger_mp.warning(
                        "[teleop arm tracking hold] captured fixed start pose target=%s",
                        _fmt_vec_debug(arm_tracking_hold_q),
                    )
                sol_q = arm_tracking_hold_q.copy()
                sol_tauff = np.zeros_like(sol_q)
                if loop_count % 50 == 0:
                    logger_mp.info(
                        "[teleop arm tracking hold] Vive wrist IK skipped; fixed_target=%s current_q=%s",
                        _fmt_vec_debug(sol_q),
                        _fmt_vec_debug(current_q),
                    )
            elif arm_fsm == "STANDBY":
                if args.arm_standby_action == "ready":
                    sol_q = _make_arm_ready_q(current_lr_arm_q)
                else:
                    sol_q = np.asarray(current_lr_arm_q, dtype=np.float64).copy()
                sol_tauff = np.zeros_like(sol_q)
                if not arm_standby_logged or loop_count % 50 == 0:
                    arm_standby_logged = True
                    logger_mp.warning(
                        f"[teleop arm tracking standby] loop={loop_count} action={args.arm_standby_action} "
                        f"head_valid={getattr(tele_data, 'head_pose_is_valid', None)} "
                        f"left_arm_valid={getattr(tele_data, 'left_arm_is_valid', None)} "
                        f"right_arm_valid={getattr(tele_data, 'right_arm_is_valid', None)} "
                        f"target={_fmt_vec_debug(sol_q)}"
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

            # per-cycle communication/action/state log (never breaks teleop)
            if comm_logger is not None:
                try:
                    if loop_ee_lock is not None:
                        with loop_ee_lock:
                            _cl_hand_state = list(loop_ee_state_array)
                            _cl_hand_action = list(loop_ee_action_array)
                    else:
                        _cl_hand_state = None
                        _cl_hand_action = None
                    comm_logger.log_cycle(
                        loop_count,
                        arm_q=current_lr_arm_q,
                        arm_dq=current_lr_arm_dq,
                        arm_action_q=sol_q,
                        arm_action_tauff=sol_tauff,
                        hand_state=_cl_hand_state,
                        hand_action=_cl_hand_action,
                        mode=arm_fsm,
                    )
                except Exception as exc:
                    if loop_count % 200 == 0:
                        logger_mp.warning("[comm log] cycle log failed: %s", exc)

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
            if comm_logger is not None:
                comm_logger.close()
                comm_logger = None
        except Exception as e:
            logger_mp.error(f"Failed to close comm/state logger: {e}")

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
            if manus_haptic_sender is not None:
                manus_haptic_sender.close()
        except Exception as e:
            logger_mp.error(f"Failed to stop MANUS haptics: {e}")

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
            if VIVE_MANUS_READER is not None:
                VIVE_MANUS_READER.close()
        except Exception as e:
            logger_mp.error(f"Failed to close Vive/Manus reader: {e}")

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
