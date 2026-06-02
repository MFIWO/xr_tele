import time
import argparse
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
from teleop.utils.ipc import IPC_Server
from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
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
    global STOP, START, RECORD_TOGGLE
    if key == 'r':
        START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and START == True:
        RECORD_TOGGLE = True
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

def _apply_camera_orientation(image, camera_name, args):
    if image is None:
        return None
    oriented = image
    if camera_name == "left_wrist" and args.left_wrist_camera_vflip:
        oriented = np.flipud(oriented).copy()
    if camera_name == "right_wrist" and args.right_wrist_camera_vflip:
        oriented = np.flipud(oriented).copy()
    return oriented

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1', 'H2'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'rh5dg2_ftp', 'rh5dg2_dfx', 'brainco'], help='Select end effector controller')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    parser.add_argument('--camera', type=str, choices=['head', 'left_wrist', 'right_wrist', 'all'], default='head', help='Camera stream shown in the 8012 XR viewer.')
    parser.add_argument('--viewer-camera-mode', type=str, choices=['auto', 'webrtc', 'zmq', 'none'], default='auto', help='Select how the 8012 XR viewer receives the head camera.')
    parser.add_argument('--no-left-wrist-camera-vflip', dest='left_wrist_camera_vflip', action='store_false', help='Disable vertical flip correction for the left wrist camera.')
    parser.add_argument('--right-wrist-camera-vflip', action='store_true', help='Enable vertical flip correction for the right wrist camera.')
    parser.add_argument('--hand-control-hz', type=float, default=50.0, help='RH5DG2 hand retarget/publish loop frequency.')
    parser.add_argument('--hand-debug-rate', type=float, default=1.0, help='Teleop hand input debug log rate in Hz.')
    parser.add_argument('--rh5dg2-log-throttle', type=float, default=1.0, help='RH5DG2 controller debug log rate in Hz.')
    parser.add_argument('--rh5dg2-hand-swap', action='store_true', help='Enable RH5DG2-only left/right hand input swap for devices that report swapped hand labels.')
    parser.add_argument('--disable-hand-smoothing', action='store_true', help='Reserved flag for RH5DG2 hand path; current RH5DG2 path has no smoothing enabled.')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--disable-arm', action='store_true', help='Disable arm IK/control while keeping XR and hand paths alive.')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--affinity', action = 'store_true', help = 'Enable high priority and set CPU affinity mode')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument('--task-dir', type = str, default = './utils/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')

    args = parser.parse_args()
    logger_mp.debug(f"args: {args}")

    try:
        # setup dds communication domains id
        if args.sim:
            ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        else:
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)

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
        selected_camera_key = _camera_config_key(selected_camera_name)
        selected_camera_config = camera_config[selected_camera_key]
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
        viewer_host_ip = _local_ip_for_remote(args.img_server_ip)
        viewer_url = f"https://{viewer_host_ip}:8012/?ws=wss://{viewer_host_ip}:8012"
        selected_webrtc_port = selected_camera_config.get("webrtc_port")
        selected_zmq_port = selected_camera_config.get("zmq_port")
        webrtc_offer_url = f"https://{args.img_server_ip}:{selected_webrtc_port}/offer" if selected_webrtc_port else None
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
            logger_mp.warning("[teleop camera selected] --camera=all logs all streams but displays head camera in the current 8012 viewer.")
        logger_mp.info(
            f"[teleop viewer 8012] url={viewer_url} bind=0.0.0.0:8012 "
            f"display_mode={args.display_mode} requested_camera_mode={args.viewer_camera_mode} "
            f"selected_camera_mode={viewer_route} selected_camera={selected_camera_name}"
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
                logger_mp.warning("[teleop camera route] immersive/ego requested but head camera has no ZMQ/WebRTC enabled.")
            logger_mp.info(
                f"[teleop viewer stream bind] selected_camera={selected_camera_name} mode={viewer_route} "
                f"webrtc_url={webrtc_offer_url if viewer_webrtc else None} "
                f"zmq={args.img_server_ip}:{selected_zmq_port if viewer_zmq else None}"
            )

        # televuer_wrapper: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand", 
                                     binocular=selected_camera_config['binocular'],
                                     img_shape=selected_camera_config['image_shape'],
                                     # maybe should decrease fps for better performance?
                                     # https://github.com/unitreerobotics/xr_teleoperate/issues/172
                                     # display_fps=camera_config['head_camera']['fps'] ? args.frequency? 30.0?
                                     display_mode=args.display_mode,
                                     zmq=viewer_zmq,
                                     webrtc=viewer_webrtc,
                                     webrtc_url=webrtc_offer_url,
                                     )
        
        # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        if args.motion:
            if args.input_mode == "controller":
                loco_wrapper = LocoClientWrapper()
        else:
            motion_switcher = MotionSwitcher()
            status, result = motion_switcher.Enter_Debug_Mode()
            logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

        # arm
        if args.arm == "G1_29":
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
                                              log_throttle_s=args.rh5dg2_log_throttle)
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
                                               log_throttle_s=args.rh5dg2_log_throttle)
        else:
            pass

        if args.ee in ["dex3", "inspire_dfx", "inspire_ftp", "rh5dg2_dfx", "rh5dg2_ftp", "brainco"]:
            logger_mp.info(f"[teleop ee] ee={args.ee} hand_controller={hand_ctrl.__class__.__name__}")
        if args.ee in ("rh5dg2_dfx", "rh5dg2_ftp"):
            logger_mp.info(
                f"[teleop hand side mapping] ee={args.ee} "
                f"rh5dg2_hand_swap={args.rh5dg2_hand_swap} "
                "scope=hand_landmarks_only arm_wrist_pose_unchanged=True"
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
            recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),
                                     task_goal = args.task_goal,
                                     task_desc = args.task_desc,
                                     task_steps = args.task_steps,
                                     frequency = args.frequency, 
                                     rerun_log = not args.headless)

        logger_mp.info("----------------------------------------------------------------")
        logger_mp.info("🟢  Press [r] to start syncing the robot with your movements.")
        if args.record:
            logger_mp.info("🟡  Press [s] to START or SAVE recording (toggle cycle).")
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        READY = True                  # now ready to (1) enter START state
        logger_mp.info(f"[teleop ready state] READY={READY} START={START} STOP={STOP} disable_arm={args.disable_arm}")
        while not START and not STOP: # wait for start or stop signal.
            time.sleep(0.033)
            if selected_camera_config.get('enable_zmq') and xr_need_local_img:
                prestart_img = _get_camera_frame(img_client, selected_camera_name)
                if prestart_img is not None and prestart_img.bgr is not None:
                    prestart_bgr = _apply_camera_orientation(prestart_img.bgr, selected_camera_name, args)
                    _safe_render_to_xr(tv_wrapper, prestart_bgr, "[teleop camera prestart]")
                else:
                    logger_mp.warning(f"[teleop camera prestart] no {selected_camera_name} frame received for XR display.")

        logger_mp.info("---------------------🚀start Tracking🚀-------------------------")
        logger_mp.info(f"[teleop ready state] READY={READY} START={START} STOP={STOP} disable_arm={args.disable_arm}")
        if args.disable_arm:
            logger_mp.warning("[teleop arm disabled reason] --disable-arm set; IK/control publish will be skipped.")
        else:
            arm_ctrl.speed_gradual_max()

        head_img = None
        left_wrist_img = None
        right_wrist_img = None
        viewer_frame_count = 0
        hand_input_count = 0
        hand_input_rate_start = time.time()
        hand_debug_last_ts = 0.0
        hand_debug_interval = 1.0 / args.hand_debug_rate if args.hand_debug_rate > 0 else None

        # main loop. robot start to follow VR user's motion
        loop_count = 0
        while not STOP:
            loop_count += 1
            start_time = time.time()
            # get image
            if camera_config['head_camera']['enable_zmq']:
                if args.record or (xr_need_local_img and selected_camera_name == "head"):
                    head_img = img_client.get_head_frame()
            if xr_need_local_img:
                viewer_frame = head_img if selected_camera_name == "head" else _get_camera_frame(img_client, selected_camera_name)
                viewer_frame_count += 1
                frame_timestamp = time.time()
                if viewer_frame is not None and viewer_frame.bgr is not None:
                    viewer_bgr = _apply_camera_orientation(viewer_frame.bgr, selected_camera_name, args)
                    _safe_render_to_xr(tv_wrapper, viewer_bgr, "[teleop camera loop]")
                    if loop_count % 50 == 0:
                        logger_mp.info(
                            f"[teleop viewer frame] camera_name={selected_camera_name} "
                            f"received_frame_count={viewer_frame_count} frame_timestamp={frame_timestamp:.6f} "
                            f"fps={getattr(viewer_frame, 'fps', None)}"
                        )
                        logger_mp.info(
                            f"[teleop viewer latency] camera_name={selected_camera_name} "
                            f"latency_ms={(time.time() - frame_timestamp) * 1000.0:.2f} "
                            "source=local_receive_timestamp"
                        )
                elif loop_count % 50 == 0:
                    logger_mp.warning(
                        f"[teleop camera loop] no frame received for XR display camera_name={selected_camera_name}"
                    )
            if camera_config['left_wrist_camera']['enable_zmq']:
                if args.record:
                    left_wrist_img = img_client.get_left_wrist_frame()
                    if left_wrist_img is not None and left_wrist_img.bgr is not None and cv2 is not None:
                        # 화면 누움 방향에 따라 ROTATE_90_CLOCKWISE 또는 ROTATE_90_COUNTERCLOCKWISE 선택
                        left_wrist_img.bgr = cv2.rotate(left_wrist_img.bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        left_wrist_img.bgr = _apply_camera_orientation(left_wrist_img.bgr, "left_wrist", args)
            
            # ---- [수정 부분: 오른쪽 손목 카메라 회전] ----
            if camera_config['right_wrist_camera']['enable_zmq']:
                if args.record:
                    right_wrist_img = img_client.get_right_wrist_frame()
                    if right_wrist_img is not None and right_wrist_img.bgr is not None and cv2 is not None:
                        right_wrist_img.bgr = cv2.rotate(right_wrist_img.bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        right_wrist_img.bgr = _apply_camera_orientation(right_wrist_img.bgr, "right_wrist", args)

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if recorder.create_episode():
                        RECORD_RUNNING = True
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    recorder.save_episode()
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)

            # get xr's tele data
            tele_data = tv_wrapper.get_tele_data()

            # [수정 부분: 강제 Swap 로직 제거하고 있는 그대로(Left->Left, Right->Right) 할당]
            left_hand_pos = tele_data.left_hand_pos
            right_hand_pos = tele_data.right_hand_pos
            if args.ee in ("rh5dg2_dfx", "rh5dg2_ftp") and args.rh5dg2_hand_swap:
                left_hand_pos, right_hand_pos = right_hand_pos, left_hand_pos

            left_wrist_pose = tele_data.left_wrist_pose
            right_wrist_pose = tele_data.right_wrist_pose
            if loop_count % 50 == 0:
                logger_mp.info(
                    f"[teleop arm input] ready={READY} start={START} "
                    f"left_wrist={_fmt_pose_debug(left_wrist_pose)} "
                    f"right_wrist={_fmt_pose_debug(right_wrist_pose)} "
                    f"head={_fmt_pose_debug(getattr(tele_data, 'head_pose', []))}"
                )

            left_hand_pinchValue = tele_data.left_hand_pinchValue
            right_hand_pinchValue = tele_data.right_hand_pinchValue

            left_hand_pinch = tele_data.left_hand_pinch
            right_hand_pinch = tele_data.right_hand_pinch

            left_hand_squeeze = tele_data.left_hand_squeeze
            right_hand_squeeze = tele_data.right_hand_squeeze

            left_hand_squeezeValue = tele_data.left_hand_squeezeValue
            right_hand_squeezeValue = tele_data.right_hand_squeezeValue

            if (args.ee == "dex3" or args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "rh5dg2_dfx" or args.ee == "rh5dg2_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                hand_input_count += 1
                now = time.time()
                should_hand_debug = hand_debug_interval is not None and now - hand_debug_last_ts >= hand_debug_interval
                if should_hand_debug:
                    hand_debug_last_ts = now
                if should_hand_debug:
                    logger_mp.info(
                        f"[teleop hand input before write] ee={args.ee} input={args.input_mode} "
                        f"left={_fmt_hand_debug(left_hand_pos)} right={_fmt_hand_debug(right_hand_pos)} "
                        f"timestamp={now:.6f}"
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
            if args.input_mode == "controller" and args.motion:
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
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

            # solve ik using motor data and wrist pose, then use ik results to control arms.
            if args.disable_arm:
                sol_q = np.asarray(current_lr_arm_q, dtype=np.float64).copy()
                sol_tauff = np.zeros_like(sol_q)
                if loop_count % 50 == 0:
                    logger_mp.warning(
                        f"[teleop arm disabled reason] loop={loop_count} reason=--disable-arm "
                        f"current_q={_fmt_vec_debug(current_lr_arm_q)}"
                    )
            else:
                time_ik_start = time.time()
                if loop_count % 50 == 0:
                    logger_mp.info(
                        f"[teleop arm ik enter] current_q={_fmt_vec_debug(current_lr_arm_q)} "
                        f"current_dq={_fmt_vec_debug(current_lr_arm_dq)} "
                        f"left_wrist={_fmt_pose_debug(left_wrist_pose)} "
                        f"right_wrist={_fmt_pose_debug(right_wrist_pose)}"
                    )
                sol_q, sol_tauff  = arm_ik.solve_ik(left_wrist_pose, right_wrist_pose, current_lr_arm_q, current_lr_arm_dq)
                time_ik_end = time.time()
                logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
                if loop_count % 50 == 0:
                    logger_mp.info(
                        f"[teleop arm ik result] dt={time_ik_end - time_ik_start:.6f} "
                        f"current_q={_fmt_vec_debug(current_lr_arm_q)} sol_q={_fmt_vec_debug(sol_q)} "
                        f"tauff={_fmt_vec_debug(sol_tauff)}"
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
            if not args.disable_arm:
                if loop_count % 50 == 0:
                    logger_mp.info(
                        f"[teleop arm publish enter] controller={arm_ctrl.__class__.__name__} "
                        f"sim={args.sim} target={_fmt_vec_debug(sol_q)}"
                    )
                arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
                if loop_count % 50 == 0:
                    arm_write_ok = arm_ctrl.get_last_write_ok() if hasattr(arm_ctrl, "get_last_write_ok") else None
                    logger_mp.info(
                        f"[teleop arm publish] controller={arm_ctrl.__class__.__name__} "
                        f"sim={args.sim} write_ok={arm_write_ok} topic=rt/lowcmd domain={1 if args.sim else 0} "
                        f"target={_fmt_vec_debug(sol_q)} tauff={_fmt_vec_debug(sol_tauff)}"
                    )

            # record data
            if args.record:
                READY = recorder.is_ready() # now ready to (2) enter RECORD_RUNNING state
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
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
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

                elif (args.ee == "rh5dg2_dfx" or args.ee == "rh5dg2_ftp") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:13]
                        right_ee_state = dual_hand_state_array[-13:]
                        left_hand_action = dual_hand_action_array[:13]
                        right_hand_action = dual_hand_action_array[-13:]
                        current_body_state = []
                        current_body_action = []
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []

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
                            colors[f"color_{0}"] = head_img.bgr[:, :camera_config['head_camera']['image_shape'][1]//2]
                            colors[f"color_{1}"] = head_img.bgr[:, camera_config['head_camera']['image_shape'][1]//2:]
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{2}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{3}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    else:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{1}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{2}"] = right_wrist_img.bgr
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
                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

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
            arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")
        
        try:
            if args.ipc:
                ipc_server.stop()
            else:
                stop_listening()
                listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")
        
        try:
            if img_client is not None:
                img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
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
            if args.sim:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")
        
        try:
            if args.record:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)
