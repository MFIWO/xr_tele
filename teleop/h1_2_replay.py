import argparse
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time

import numpy as np

current_dir = Path(__file__).resolve().parent
repo_dir = current_dir.parent
if str(repo_dir) not in sys.path:
    sys.path.append(str(repo_dir))

try:
    import logging_mp
except ImportError:
    import logging as logging_mp

logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

try:
    from sshkeyboard import listen_keyboard, stop_listening
except ImportError:
    listen_keyboard = None
    stop_listening = None


INSPIRE_DG2_REPLAY_JOINTS = 13
running = True
replay_event = threading.Event()
replay_event.set()


def _resolve_episode_json(path):
    episode_path = Path(path).expanduser()
    if not episode_path.is_absolute():
        episode_path = (Path.cwd() / episode_path).resolve()
    if episode_path.is_dir():
        episode_path = episode_path / "data.json"
    if not episode_path.exists():
        raise FileNotFoundError(f"episode json not found: {episode_path}")
    return episode_path


def load_episode(path):
    episode_path = _resolve_episode_json(path)
    with episode_path.open("r", encoding="utf-8") as f:
        text = f.read()
    try:
        episode = json.loads(text)
    except json.JSONDecodeError as exc:
        stripped = text.rstrip()
        repaired = None
        # EpisodeWriter writes the file as {"info":..., "data": [ ...frames... ]}.
        # If recording is interrupted after the last frame, the trailing data-array
        # and root-object close can be missing while every frame object is intact.
        if exc.pos >= len(stripped) - 2:
            for suffix in ("\n]\n}\n", "\n}\n]\n}\n"):
                try:
                    repaired = json.loads(stripped + suffix)
                    logger_mp.warning(
                        "episode json was missing final closing bracket(s); using in-memory repair for %s",
                        episode_path,
                    )
                    break
                except json.JSONDecodeError:
                    pass
        if repaired is None:
            raise
        episode = repaired
    info = episode.get("info", {})
    fps = (info.get("image", {}) or {}).get("fps") or 30.0
    frames = episode.get("data", [])
    if not frames:
        raise ValueError(f"episode contains no frames: {episode_path}")
    return float(fps), frames, episode_path, info


def get_qpos(frame, key, default_len=None):
    qpos = ((frame.get("actions", {}).get(key, {}) or {}).get("qpos"))
    if not qpos:
        qpos = ((frame.get("states", {}).get(key, {}) or {}).get("qpos"))
    if qpos is None:
        return None
    qpos = list(qpos)
    if default_len is not None and len(qpos) != default_len:
        raise ValueError(f"{key}.qpos length {len(qpos)} != {default_len}")
    return qpos


def _publish_hand_frame(frame, hand_pub):
    left_ee = get_qpos(frame, "left_ee", default_len=INSPIRE_DG2_REPLAY_JOINTS)
    right_ee = get_qpos(frame, "right_ee", default_len=INSPIRE_DG2_REPLAY_JOINTS)
    if left_ee is not None and right_ee is not None:
        hand_pub.write(left_ee, right_ee)


def get_neck_yaw_pitch(frame, source="command"):
    source_map = {
        "command": ("actions", "command_yaw_pitch"),
        "target": ("actions", "target_yaw_pitch"),
        "raw": ("states", "raw_head_yaw_pitch"),
        "actual": ("states", "actual_yaw_pitch"),
    }
    root_key, value_key = source_map[source]
    neck = ((frame.get(root_key, {}) or {}).get("neck", {}) or {})
    value = neck.get(value_key)
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 2 or not np.isfinite(arr).all():
        return None
    return arr


class NeckReplayPublisher:
    def __init__(self, host, port, dry_run=False):
        if not host:
            raise ValueError("neck host is required")
        self.address = (str(host), int(port))
        self.dry_run = bool(dry_run)
        self.socket = None if self.dry_run else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.last_command = None
        self.commands_sent = 0

    def write(self, yaw_pitch):
        arr = np.asarray(yaw_pitch, dtype=np.float64).reshape(-1)
        if arr.size != 2 or not np.isfinite(arr).all():
            return False
        self.last_command = arr.copy()
        self.commands_sent += 1
        if self.dry_run:
            return True
        payload = f"{arr[0]:.5f},{arr[1]:.5f}".encode("ascii")
        self.socket.sendto(payload, self.address)
        return True

    def zero(self):
        return self.write([0.0, 0.0])

    def close(self):
        if self.socket is not None:
            self.socket.close()


def _on_press(key):
    global running
    if key == "q":
        running = False
        replay_event.set()
        if stop_listening is not None:
            stop_listening()
    elif key == "c":
        replay_event.set()
    else:
        logger_mp.info("%s pressed (q=quit, c=replay)", key)


def _start_keyboard_thread():
    if listen_keyboard is None:
        logger_mp.warning("sshkeyboard is not installed; use Ctrl-C to quit.")
        return None
    thread = threading.Thread(
        target=listen_keyboard,
        kwargs={"on_press": _on_press, "until": None, "sequential": False},
        daemon=True,
    )
    thread.start()
    return thread


class InspireDG2ReplayPublisher:
    def __init__(self, dry_run=False, lock_spread_joints=False):
        self.dry_run = bool(dry_run)
        self.lock_spread_joints = bool(lock_spread_joints)
        self.publisher = None
        if not self.dry_run:
            from teleop.robot_control.robot_hand_inspire_dg2 import kTopicInspireDG2Command
            from unitree_sdk2py.core.channel import ChannelPublisher
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_

            self.publisher = ChannelPublisher(kTopicInspireDG2Command, MotorCmds_)
            self.publisher.Init()

    @staticmethod
    def _validate_targets(right_target, left_target):
        right_target = np.asarray(right_target, dtype=np.float64).reshape(-1)
        left_target = np.asarray(left_target, dtype=np.float64).reshape(-1)
        if right_target.size != INSPIRE_DG2_REPLAY_JOINTS:
            raise ValueError(f"right_ee expected {INSPIRE_DG2_REPLAY_JOINTS} values, got {right_target.size}")
        if left_target.size != INSPIRE_DG2_REPLAY_JOINTS:
            raise ValueError(f"left_ee expected {INSPIRE_DG2_REPLAY_JOINTS} values, got {left_target.size}")
        return right_target, left_target

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

    @staticmethod
    def _command_message(right_target, left_target, lock_spread_joints=False):
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_

        right_target, left_target = InspireDG2ReplayPublisher._validate_targets(right_target, left_target)
        if lock_spread_joints:
            from teleop.robot_control.robot_hand_inspire_dg2 import _lock_dg2_spread_joints

            right_target = _lock_dg2_spread_joints(right_target, side="right")
            left_target = _lock_dg2_spread_joints(left_target, side="left")

        msg = MotorCmds_()
        msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(INSPIRE_DG2_REPLAY_JOINTS * 2)]
        # DDS bridge order follows robot_hand_inspire_dg2.py: right hand, then left hand.
        for idx, value in enumerate(np.concatenate((right_target, left_target))):
            InspireDG2ReplayPublisher._set_dds_cmd_active(msg.cmds[idx], True)
            msg.cmds[idx].q = float(value)
            msg.cmds[idx].dq = 0.0
            msg.cmds[idx].tau = 0.0
            msg.cmds[idx].kp = 1.0
            msg.cmds[idx].kd = 0.05
        return msg

    def write(self, left_target, right_target):
        if self.dry_run:
            self._validate_targets(right_target, left_target)
            return None
        return self.publisher.Write(
            self._command_message(
                right_target,
                left_target,
                lock_spread_joints=self.lock_spread_joints,
            )
        )


def _compute_arm_tau(
    arm_ctrl,
    q_target,
    arm_feedforward=None,
    torque_feedback=True,
    torque_kp=25.0,
    torque_kd=1.5,
    torque_limit=8.0,
):
    q_target = np.asarray(q_target, dtype=np.float64).reshape(-1)
    tau = np.zeros_like(q_target)
    if arm_feedforward is not None:
        tau = np.asarray(arm_feedforward.compute(q_target), dtype=np.float64).reshape(-1)

    if not torque_feedback or arm_ctrl is None:
        return tau

    current_q = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=np.float64).reshape(-1)
    current_dq = np.asarray(arm_ctrl.get_current_dual_arm_dq(), dtype=np.float64).reshape(-1)
    if current_q.shape != q_target.shape or current_dq.shape != q_target.shape:
        return tau

    feedback_tau = float(torque_kp) * (q_target - current_q) - float(torque_kd) * current_dq
    if torque_limit > 0.0:
        feedback_tau = np.clip(feedback_tau, -float(torque_limit), float(torque_limit))
    return tau + feedback_tau


def _send_arm_command(
    arm_ctrl,
    q_target,
    arm_feedforward=None,
    torque_feedback=True,
    torque_kp=25.0,
    torque_kd=1.5,
    torque_limit=8.0,
):
    tau_cmd = _compute_arm_tau(
        arm_ctrl,
        q_target,
        arm_feedforward=arm_feedforward,
        torque_feedback=torque_feedback,
        torque_kp=torque_kp,
        torque_kd=torque_kd,
        torque_limit=torque_limit,
    )
    arm_ctrl.ctrl_dual_arm(q_target, tau_cmd)
    return tau_cmd


def _servo_arm_until(
    arm_ctrl,
    q_target,
    deadline,
    servo_hz,
    arm_feedforward=None,
    torque_feedback=True,
    torque_kp=25.0,
    torque_kd=1.5,
    torque_limit=8.0,
):
    if arm_ctrl is None:
        return None
    q_target = np.asarray(q_target, dtype=np.float64).reshape(-1)
    servo_dt = 1.0 / max(float(servo_hz), 1.0)
    last_error = None

    while running:
        _send_arm_command(
            arm_ctrl,
            q_target,
            arm_feedforward=arm_feedforward,
            torque_feedback=torque_feedback,
            torque_kp=torque_kp,
            torque_kd=torque_kd,
            torque_limit=torque_limit,
        )
        current_q = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=np.float64).reshape(-1)
        if current_q.shape == q_target.shape:
            last_error = float(np.max(np.abs(q_target - current_q)))

        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        time.sleep(min(servo_dt, remaining))

    return last_error


def _configure_h1_2_arm_for_replay(arm_ctrl, speed_gradual_max=True, velocity_limit=0.0, kp_scale=1.0, kd_scale=1.0):
    if arm_ctrl is None:
        return
    if velocity_limit > 0.0:
        arm_ctrl.arm_velocity_limit = float(velocity_limit)
        if hasattr(arm_ctrl, "_speed_gradual_max"):
            arm_ctrl._speed_gradual_max = False
        logger_mp.info("[H1-2 Replay] arm velocity limit override: %.3f", arm_ctrl.arm_velocity_limit)
    elif speed_gradual_max and hasattr(arm_ctrl, "speed_gradual_max"):
        arm_ctrl.speed_gradual_max()

    if abs(kp_scale - 1.0) < 1e-9 and abs(kd_scale - 1.0) < 1e-9:
        return

    try:
        from teleop.robot_control.robot_arm import H1_2_JointArmIndex
    except Exception as exc:
        logger_mp.warning("Failed to import H1_2_JointArmIndex for replay gain scaling: %s", exc)
        return

    lock = getattr(arm_ctrl, "ctrl_lock", None)
    if lock is None:
        return
    with lock:
        for motor_id in H1_2_JointArmIndex:
            cmd = arm_ctrl.msg.motor_cmd[motor_id]
            cmd.kp *= float(kp_scale)
            cmd.kd *= float(kd_scale)
    logger_mp.info(
        "[H1-2 Replay] scaled arm PD gains: kp_scale=%.3f kd_scale=%.3f",
        kp_scale,
        kd_scale,
    )


def _blend_to_first_frame(
    arm_ctrl,
    first_q,
    duration,
    hz,
    arm_feedforward=None,
    torque_feedback=True,
    torque_kp=25.0,
    torque_kd=1.5,
    torque_limit=8.0,
):
    if duration <= 0.0:
        return
    first_q = np.asarray(first_q, dtype=np.float64)
    start_q = arm_ctrl.get_current_dual_arm_q()
    steps = max(1, int(duration * hz))
    zero_tau = np.zeros_like(first_q)
    for i in range(steps):
        if not running:
            break
        alpha = (i + 1) / steps
        q = (1.0 - alpha) * start_q + alpha * first_q
        if torque_feedback:
            _send_arm_command(
                arm_ctrl,
                q,
                arm_feedforward=arm_feedforward,
                torque_feedback=torque_feedback,
                torque_kp=torque_kp,
                torque_kd=torque_kd,
                torque_limit=torque_limit,
            )
        else:
            tau_ff = arm_feedforward.compute(q) if arm_feedforward is not None else zero_tau
            arm_ctrl.ctrl_dual_arm(q, tau_ff)
        time.sleep(1.0 / hz)


def _wait_for_arm_target(
    arm_ctrl,
    q_target,
    tolerance,
    timeout,
    check_hz,
    arm_feedforward=None,
    torque_feedback=True,
    torque_kp=25.0,
    torque_kd=1.5,
    torque_limit=8.0,
):
    if arm_ctrl is None or timeout <= 0.0 or tolerance <= 0.0:
        return 0.0, None, False
    q_target = np.asarray(q_target, dtype=np.float64).reshape(-1)
    check_dt = 1.0 / max(float(check_hz), 1.0)
    start = time.monotonic()
    last_error = None
    reached = False

    while running:
        _send_arm_command(
            arm_ctrl,
            q_target,
            arm_feedforward=arm_feedforward,
            torque_feedback=torque_feedback,
            torque_kp=torque_kp,
            torque_kd=torque_kd,
            torque_limit=torque_limit,
        )
        current_q = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=np.float64).reshape(-1)
        if current_q.shape == q_target.shape:
            last_error = float(np.max(np.abs(q_target - current_q)))
            if last_error <= tolerance:
                reached = True
                break
        if time.monotonic() - start >= timeout:
            break
        time.sleep(check_dt)

    return time.monotonic() - start, last_error, reached


class H1_2RneaFeedForward:
    def __init__(self):
        import pinocchio as pin
        from teleop.robot_control.robot_arm_ik import H1_2_ArmIK

        self.pin = pin
        self.arm_ik = H1_2_ArmIK()
        self.data = self.arm_ik.reduced_robot.model.createData()

    def compute(self, q_cmd):
        q_cmd = np.asarray(q_cmd, dtype=np.float64)
        v = np.zeros(self.arm_ik.reduced_robot.model.nv)
        return self.pin.rnea(self.arm_ik.reduced_robot.model, self.data, q_cmd, v, v)


def main():
    global running

    parser = argparse.ArgumentParser(description="Replay H1-2 arm + InspireDG2 hand episode data.")
    parser.add_argument("episode_path", nargs="?", help="episode directory or data.json path")
    parser.add_argument("--episode", dest="episode_opt", help="episode directory or data.json path")
    parser.add_argument("--network-interface", "--iface", dest="network_interface", default=None)
    parser.add_argument("--hz", type=float, default=0.0, help="override replay frequency; 0 uses episode fps")
    parser.add_argument("--sim", action="store_true", help="use DDS domain 1")
    parser.add_argument("--motion", action="store_true", help="publish arm command to rt/arm_sdk")
    parser.add_argument("--skip-debug-mode", action="store_true", help="do not call MotionSwitcher.Enter_Debug_Mode before arm replay")
    parser.add_argument("--no-arm", action="store_true", help="do not publish arm commands")
    parser.add_argument("--no-hand", action="store_true", help="do not publish InspireDG2 hand commands")
    parser.add_argument("--neck", action=argparse.BooleanOptionalAction, default=True, help="replay neck commands from episode actions.neck")
    parser.add_argument("--neck-host", default=None, help="neck controller UDP host; default uses episode info.recording.neck.command_host")
    parser.add_argument("--neck-port", type=int, default=None, help="neck controller UDP port; default uses episode info.recording.neck.command_port or 9091")
    parser.add_argument("--neck-source", choices=("command", "target", "raw", "actual"), default="command", help="which recorded neck yaw/pitch to replay")
    parser.add_argument("--neck-zero-on-exit", action=argparse.BooleanOptionalAction, default=True, help="send 0,0 to the neck controller on replay exit")
    parser.add_argument("--neck-log-rate", type=float, default=1.0, help="max rate for replay neck logs; 0 disables")
    parser.add_argument("--dry-run", action="store_true", help="parse and time replay without DDS publishers")
    parser.add_argument("--once", action="store_true", help="exit after one playback")
    parser.add_argument("--lock-dg2-spread-joints", action="store_true", help="force DG2 spread joints to neutral before publishing")
    parser.add_argument("--no-rnea-feedforward", action="store_true", help="disable Pinocchio RNEA arm feed-forward torque")
    parser.add_argument("--use-rnea-feedforward", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arm-servo-hz", type=float, default=250.0, help="rate for replay-side arm command/tau refresh inside each frame")
    parser.add_argument("--arm-torque-feedback", action=argparse.BooleanOptionalAction, default=True, help="add bounded replay-side torque from q/dq tracking error")
    parser.add_argument("--arm-torque-kp", type=float, default=25.0, help="outer-loop torque gain from q target error")
    parser.add_argument("--arm-torque-kd", type=float, default=1.5, help="outer-loop torque damping gain from measured dq")
    parser.add_argument("--arm-torque-limit", type=float, default=8.0, help="per-joint absolute limit for replay-side feedback torque; 0 disables clipping")
    parser.add_argument("--arm-speed-gradual-max", action=argparse.BooleanOptionalAction, default=True, help="match teleop startup by ramping arm velocity limit toward max")
    parser.add_argument("--arm-velocity-limit", type=float, default=0.0, help="override controller arm velocity limit; 0 keeps controller/ramp behavior")
    parser.add_argument("--arm-kp-scale", type=float, default=1.0, help="optional replay-only scale for H1-2 arm motor kp")
    parser.add_argument("--arm-kd-scale", type=float, default=1.0, help="optional replay-only scale for H1-2 arm motor kd")
    parser.add_argument("--hand-timing", choices=("after-arm", "immediate"), default="after-arm", help="publish hand after the frame arm servo window or immediately at frame start")
    parser.add_argument("--arm-tracking-wait", action=argparse.BooleanOptionalAction, default=False, help="extra wait after each frame until arm qpos approaches target")
    parser.add_argument("--arm-wait-tolerance", type=float, default=0.08, help="max joint error in radians accepted by optional tracking wait")
    parser.add_argument("--arm-wait-timeout", type=float, default=0.25, help="max seconds to wait for each arm target")
    parser.add_argument("--arm-wait-check-hz", type=float, default=100.0, help="arm qpos polling rate while waiting for target tracking")
    parser.add_argument("--arm-wait-log-rate", type=float, default=1.0, help="max rate for arm tracking wait logs; 0 disables")
    parser.add_argument("--startup-blend", type=float, default=2.0, help="seconds to blend current arm pose to first frame")
    parser.add_argument("--skip-arm-go-home-on-exit", action="store_true")
    args = parser.parse_args()

    episode_arg = args.episode_opt or args.episode_path
    if not episode_arg:
        parser.error("episode path is required")

    fps_file, frames, episode_json, episode_info = load_episode(episode_arg)
    os.chdir(current_dir)
    play_hz = args.hz if args.hz > 0.0 else fps_file
    dt = 1.0 / play_hz
    episode_neck_info = (((episode_info.get("recording", {}) or {}).get("neck", {}) or {}))
    neck_enabled_in_episode = bool(episode_neck_info.get("enabled", False))
    neck_host = args.neck_host or episode_neck_info.get("command_host")
    neck_port = args.neck_port or episode_neck_info.get("command_port") or 9091
    replay_neck = bool(args.neck and (neck_enabled_in_episode or args.neck_host is not None or args.neck_port is not None))

    logger_mp.info("[H1-2 Replay] file=%s frames=%s hz=%.2f sim=%s dry_run=%s", episode_json, len(frames), play_hz, args.sim, args.dry_run)
    logger_mp.info(
        "[H1-2 Replay] arm_servo_hz=%.1f torque_feedback=%s torque_kp=%.2f torque_kd=%.2f torque_limit=%.2f hand_timing=%s",
        args.arm_servo_hz,
        args.arm_torque_feedback,
        args.arm_torque_kp,
        args.arm_torque_kd,
        args.arm_torque_limit,
        args.hand_timing,
    )
    logger_mp.info(
        "[H1-2 Replay] arm_tracking_wait=%s tolerance=%.4f timeout=%.3fs check_hz=%.1f",
        args.arm_tracking_wait,
        args.arm_wait_tolerance,
        args.arm_wait_timeout,
        args.arm_wait_check_hz,
    )
    logger_mp.info(
        "[H1-2 Replay] neck=%s source=%s target=%s:%s episode_enabled=%s",
        replay_neck,
        args.neck_source,
        neck_host,
        neck_port,
        neck_enabled_in_episode,
    )
    logger_mp.info("Press 'q' to quit. After each run, press 'c' to replay.")

    keyboard_thread = _start_keyboard_thread()
    arm_ctrl = None
    hand_pub = None
    neck_pub = None
    motion_switcher = None

    try:
        if not args.dry_run:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize

            ChannelFactoryInitialize(1 if args.sim else 0, networkInterface=args.network_interface)

        arm_feedforward = None
        if not args.no_arm:
            if not args.dry_run:
                from teleop.robot_control.robot_arm import H1_2_ArmController

                if not args.motion and not args.sim and not args.skip_debug_mode:
                    from teleop.utils.motion_switcher import MotionSwitcher

                    motion_switcher = MotionSwitcher()
                    status, result = motion_switcher.Enter_Debug_Mode()
                    logger_mp.info("Enter debug mode: %s", "Success" if status == 0 else f"Failed status={status} result={result}")

                arm_ctrl = H1_2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
                _configure_h1_2_arm_for_replay(
                    arm_ctrl,
                    speed_gradual_max=args.arm_speed_gradual_max,
                    velocity_limit=args.arm_velocity_limit,
                    kp_scale=args.arm_kp_scale,
                    kd_scale=args.arm_kd_scale,
                )
                if not args.no_rnea_feedforward or args.use_rnea_feedforward:
                    arm_feedforward = H1_2RneaFeedForward()

        if not args.no_hand:
            hand_pub = InspireDG2ReplayPublisher(
                dry_run=args.dry_run,
                lock_spread_joints=args.lock_dg2_spread_joints,
            )

        if replay_neck:
            if not neck_host:
                logger_mp.warning(
                    "[H1-2 Replay neck] disabled: no --neck-host and no episode info.recording.neck.command_host"
                )
            else:
                neck_pub = NeckReplayPublisher(neck_host, neck_port, dry_run=args.dry_run)
                logger_mp.info("[H1-2 Replay neck] publishing UDP commands to %s:%s", neck_host, neck_port)

        first_left = get_qpos(frames[0], "left_arm", default_len=7) or [0.0] * 7
        first_right = get_qpos(frames[0], "right_arm", default_len=7) or [0.0] * 7
        first_q = np.array(first_left + first_right, dtype=np.float64)
        if arm_ctrl is not None:
            _blend_to_first_frame(
                arm_ctrl,
                first_q,
                args.startup_blend,
                play_hz,
                arm_feedforward=arm_feedforward,
                torque_feedback=args.arm_torque_feedback,
                torque_kp=args.arm_torque_kp,
                torque_kd=args.arm_torque_kd,
                torque_limit=args.arm_torque_limit,
            )

        while running:
            replay_event.wait()
            replay_event.clear()
            if not running:
                break

            next_frame_time = time.monotonic()
            last_arm_wait_log = 0.0
            last_neck_log = 0.0
            for i, frame in enumerate(frames):
                if not running:
                    break

                next_frame_time += dt
                q_cmd = None
                if not args.no_arm:
                    left_arm = get_qpos(frame, "left_arm", default_len=7) or [0.0] * 7
                    right_arm = get_qpos(frame, "right_arm", default_len=7) or [0.0] * 7
                    q_cmd = np.array(left_arm + right_arm, dtype=np.float64)

                if args.hand_timing == "immediate" and not args.no_hand and hand_pub is not None:
                    _publish_hand_frame(frame, hand_pub)

                if neck_pub is not None:
                    neck_cmd = get_neck_yaw_pitch(frame, source=args.neck_source)
                    if neck_cmd is not None:
                        neck_pub.write(neck_cmd)
                        now = time.monotonic()
                        if args.neck_log_rate > 0.0 and now - last_neck_log >= 1.0 / args.neck_log_rate:
                            last_neck_log = now
                            logger_mp.info(
                                "[H1-2 Replay neck] idx=%s source=%s yaw_pitch=%s",
                                i,
                                args.neck_source,
                                np.round(neck_cmd, 5).tolist(),
                            )

                if q_cmd is not None and arm_ctrl is not None:
                    _servo_arm_until(
                        arm_ctrl,
                        q_cmd,
                        next_frame_time,
                        args.arm_servo_hz,
                        arm_feedforward=arm_feedforward,
                        torque_feedback=args.arm_torque_feedback,
                        torque_kp=args.arm_torque_kp,
                        torque_kd=args.arm_torque_kd,
                        torque_limit=args.arm_torque_limit,
                    )

                if (
                    args.arm_tracking_wait
                    and q_cmd is not None
                    and arm_ctrl is not None
                    and not args.dry_run
                ):
                    waited, arm_error, reached = _wait_for_arm_target(
                        arm_ctrl,
                        q_cmd,
                        args.arm_wait_tolerance,
                        args.arm_wait_timeout,
                        args.arm_wait_check_hz,
                        arm_feedforward=arm_feedforward,
                        torque_feedback=args.arm_torque_feedback,
                        torque_kp=args.arm_torque_kp,
                        torque_kd=args.arm_torque_kd,
                        torque_limit=args.arm_torque_limit,
                    )
                    now = time.monotonic()
                    should_log = (
                        args.arm_wait_log_rate > 0.0
                        and (not reached or waited > dt)
                        and now - last_arm_wait_log >= 1.0 / args.arm_wait_log_rate
                    )
                    if should_log:
                        last_arm_wait_log = now
                        logger_mp.info(
                            "[H1-2 Replay arm wait] idx=%s reached=%s waited=%.3fs error=%s tolerance=%.4f",
                            i,
                            reached,
                            waited,
                            None if arm_error is None else round(arm_error, 5),
                            args.arm_wait_tolerance,
                        )

                if args.hand_timing == "after-arm" and not args.no_hand and hand_pub is not None:
                    _publish_hand_frame(frame, hand_pub)

                sleep_time = next_frame_time - time.monotonic()
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
                else:
                    next_frame_time = time.monotonic()

            if running:
                logger_mp.info("Playback finished. Press 'c' to replay, or 'q' to quit.")
            if args.once:
                break

    except KeyboardInterrupt:
        running = False
        logger_mp.info("KeyboardInterrupt, exiting.")
    finally:
        if neck_pub is not None:
            try:
                if args.neck_zero_on_exit:
                    neck_pub.zero()
                neck_pub.close()
            except Exception as exc:
                logger_mp.warning("Failed to close neck replay publisher: %s", exc)
        if arm_ctrl is not None and not args.skip_arm_go_home_on_exit:
            try:
                arm_ctrl.ctrl_dual_arm_go_home()
            except Exception as exc:
                logger_mp.warning("Failed to send arm go-home: %s", exc)
        if keyboard_thread is not None:
            try:
                keyboard_thread.join(timeout=0.2)
            except Exception:
                pass
        if motion_switcher is not None:
            logger_mp.info("Leaving debug mode active, matching teleop_hand_and_arm.py shutdown behavior.")
        logger_mp.info("Replay exit.")


if __name__ == "__main__":
    main()
