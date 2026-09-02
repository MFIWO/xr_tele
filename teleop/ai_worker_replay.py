#!/usr/bin/env python3
"""Safely replay recorded AI Worker arm and HX5 qpos with live diagnostics.

The episode's ``actions.*.qpos`` values are replay targets.  Recorded state,
the post-limiter command, and fresh ``/joint_states`` feedback are kept as
separate signals in the Rerun view; no recorded state is ever substituted for
a missing action.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path
import socket
import sys
import threading
import time

import numpy as np


_TELEOP_DIR = Path(__file__).resolve().parent
_REPO_DIR = _TELEOP_DIR.parent
if str(_REPO_DIR) not in sys.path:
    sys.path.append(str(_REPO_DIR))

from teleop.utils.ai_worker_episode import (  # noqa: E402
    AIWorkerReplayFrame,
    load_episode,
    preflight_ai_worker_episode,
)
from teleop.utils.replay_visualizer import ReplayVisualizer  # noqa: E402


logger = logging.getLogger(__name__)


class PedalEstopReceiver:
    """Receive the pedal terminal's latched ``ESTOP 0|1`` UDP state."""

    def __init__(self, host="127.0.0.1", port=8765):
        self._active = False
        self._running = True
        self._message_received = threading.Event()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((str(host), int(port)))
        self._socket.settimeout(0.2)
        self._thread = threading.Thread(
            target=self._run,
            name="ai-worker-replay-estop",
            daemon=True,
        )
        self._thread.start()

    def _run(self):
        while self._running:
            try:
                payload, _ = self._socket.recvfrom(64)
            except socket.timeout:
                continue
            except OSError:
                break
            fields = payload.decode("ascii", errors="ignore").strip().split()
            if len(fields) == 2 and fields[0] == "ESTOP" and fields[1] in ("0", "1"):
                self._active = fields[1] == "1"
                self._message_received.set()

    @property
    def active(self):
        return self._active

    def wait_for_state(self, timeout):
        return self._message_received.wait(max(0.0, float(timeout)))

    def close(self):
        if not self._running:
            return
        self._running = False
        self._socket.close()
        self._thread.join(timeout=1.0)


class PedalMotionInhibitor:
    """Continuously keep pedal base/lift motion disabled during replay."""

    def __init__(self, host="127.0.0.1", port=8766):
        self._target = (str(host), int(port))
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def publish(self):
        self._socket.sendto(b"MOTION 0", self._target)

    def close(self):
        self._socket.close()


def _qpos_group(arm_q=None, hand_q=None):
    group = {}
    if arm_q is not None:
        arm_q = np.asarray(arm_q, dtype=np.float64).reshape(14)
        group["left_arm"] = {"qpos": arm_q[:7].tolist()}
        group["right_arm"] = {"qpos": arm_q[7:].tolist()}
    if hand_q is not None:
        hand_q = np.asarray(hand_q, dtype=np.float64).reshape(40)
        group["left_ee"] = {"qpos": hand_q[:20].tolist()}
        group["right_ee"] = {"qpos": hand_q[20:].tolist()}
    return group


def _targets_for_frame(frame: AIWorkerReplayFrame):
    return frame.arm_action, frame.hand_action


def _live_group(arm_ctrl, hand_pub):
    arm_q = arm_ctrl.get_current_dual_arm_q() if arm_ctrl is not None else None
    hand_q = hand_pub.get_current_dual_hand_q() if hand_pub is not None else None
    return _qpos_group(arm_q=arm_q, hand_q=hand_q)


def _sent_group(arm_ctrl, hand_pub):
    arm_q = (
        arm_ctrl.get_last_commanded_dual_arm_q()
        if arm_ctrl is not None
        else None
    )
    hand_q = (
        hand_pub.get_last_commanded_dual_hand_q()
        if hand_pub is not None
        else None
    )
    return _qpos_group(arm_q=arm_q, hand_q=hand_q)


def _send_target(arm_ctrl, hand_pub, arm_target, hand_target):
    if arm_ctrl is not None and arm_target is not None:
        arm_ctrl.ctrl_dual_arm(arm_target)
    if hand_pub is not None and hand_target is not None:
        hand_pub.write(hand_target[:20], hand_target[20:])
    return _sent_group(arm_ctrl, hand_pub)


def _safe_hold(arm_ctrl, hand_pub):
    """Overwrite active trajectories with the latest measured-pose hold."""
    errors = []
    if arm_ctrl is not None:
        try:
            hold_q = arm_ctrl.sync_arm_command_to_measured()
            arm_ctrl.ctrl_dual_arm(hold_q)
        except Exception as exc:  # A hold failure must be surfaced, not hidden.
            errors.append(f"arm hold failed: {exc}")
    if hand_pub is not None:
        try:
            hand_pub.hold_current()
        except Exception as exc:
            # A measured joint can be microscopically outside the nominal URDF
            # limit. Repeating the last validated command is the safe fallback.
            try:
                last_q = hand_pub.get_last_commanded_dual_hand_q()
                hand_pub.write(last_q[:20], last_q[20:])
            except Exception as fallback_exc:
                errors.append(f"hand hold failed: {exc}; fallback failed: {fallback_exc}")
    if errors:
        raise RuntimeError("; ".join(errors))


def _publish_inhibit(inhibitor):
    if inhibitor is not None:
        try:
            inhibitor.publish()
        except OSError as exc:
            logger.warning("Failed to publish pedal motion inhibit: %s", exc)


def _wait_while_estopped(estop, inhibitor, arm_ctrl, hand_pub, hold_hz=20.0):
    if estop is None or not estop.active:
        return False
    logger.critical(
        "Pedal U E-stop is latched: replay timeline paused; arm/HX5 measured pose is held."
    )
    period = 1.0 / max(1.0, float(hold_hz))
    while estop.active:
        _publish_inhibit(inhibitor)
        _safe_hold(arm_ctrl, hand_pub)
        time.sleep(period)
    _safe_hold(arm_ctrl, hand_pub)
    logger.warning("Pedal U E-stop cleared; safely blending to the pending replay target.")
    return True


def _quintic_blend(alpha):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return 10.0 * alpha**3 - 15.0 * alpha**4 + 6.0 * alpha**5


def _blend_to_target(
    arm_ctrl,
    hand_pub,
    arm_target,
    hand_target,
    duration,
    hz,
    estop,
    inhibitor,
):
    """Blend from measured feedback, restarting the blend after every E-stop."""
    if arm_ctrl is None and hand_pub is None:
        return
    duration = max(0.0, float(duration))
    hz = max(1.0, float(hz))

    while True:
        _wait_while_estopped(estop, inhibitor, arm_ctrl, hand_pub)
        if arm_ctrl is not None:
            arm_start = arm_ctrl.sync_arm_command_to_measured()
        else:
            arm_start = None
        hand_start = (
            hand_pub.get_current_dual_hand_q()
            if hand_pub is not None
            else None
        )

        steps = max(1, int(math.ceil(duration * hz)))
        interrupted = False
        for step in range(1, steps + 1):
            if estop is not None and estop.active:
                interrupted = True
                break
            alpha = 1.0 if duration == 0.0 else _quintic_blend(step / steps)
            arm_command = None
            hand_command = None
            if arm_target is not None:
                arm_command = arm_start + alpha * (arm_target - arm_start)
            if hand_target is not None:
                hand_command = hand_start + alpha * (hand_target - hand_start)
            _send_target(arm_ctrl, hand_pub, arm_command, hand_command)
            _publish_inhibit(inhibitor)
            if step != steps:
                time.sleep(1.0 / hz)
        if not interrupted:
            return


def _wait_until(deadline, estop, inhibitor):
    """Wait for a replay deadline while detecting U without a long blind sleep."""
    while True:
        _publish_inhibit(inhibitor)
        if estop is not None and estop.active:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return False
        time.sleep(min(0.01, remaining))


def _episode_domain(info):
    recording = info.get("recording", {}) if isinstance(info, dict) else {}
    ai_worker = recording.get("ai_worker", {}) if isinstance(recording, dict) else {}
    value = ai_worker.get("dds_domain_id") if isinstance(ai_worker, dict) else None
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid recording.ai_worker.dds_domain_id: {value!r}") from exc
    if value < 0:
        raise ValueError(f"invalid recording.ai_worker.dds_domain_id: {value}")
    return value


def _arm_home_q(model):
    """Return the explicit model home pose for the selected robot."""
    from teleop.robot_control.robotis_ai_worker import (
        AI_WORKER_SG2_HOME_Q,
        AI_WORKER_SH5_HOME_Q,
    )

    homes = {
        "sg2": AI_WORKER_SG2_HOME_Q,
        "sh5": AI_WORKER_SH5_HOME_Q,
    }
    try:
        return np.asarray(homes[str(model).lower()], dtype=np.float64).copy()
    except KeyError as exc:
        raise ValueError(f"unsupported AI Worker model: {model!r}") from exc


def _read_post_replay_choice(input_stream=None, output_stream=None):
    """Read a single R/Q choice, using immediate key input on an interactive TTY."""
    input_stream = sys.stdin if input_stream is None else input_stream
    output_stream = sys.stdout if output_stream is None else output_stream
    prompt = "Replay finished: [R] replay again, [Q] move to home and exit: "

    while True:
        print(prompt, end="", file=output_stream, flush=True)
        if input_stream.isatty():
            import termios
            import tty

            file_descriptor = input_stream.fileno()
            previous_settings = termios.tcgetattr(file_descriptor)
            try:
                tty.setcbreak(file_descriptor)
                choice = input_stream.read(1)
            finally:
                termios.tcsetattr(
                    file_descriptor,
                    termios.TCSADRAIN,
                    previous_settings,
                )
            print(file=output_stream, flush=True)
        else:
            choice = input_stream.readline()
            if choice == "":
                raise RuntimeError(
                    "post-replay prompt requires an interactive terminal or R/Q input"
                )

        choice = choice.strip().lower()[:1]
        if choice in ("r", "q"):
            return choice
        print("Please press R or Q.", file=output_stream, flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Replay AI Worker arm/HX5 actions and show recorded target, sent "
            "command, live measured state, signed error, and episode cameras."
        )
    )
    parser.add_argument("episode_path", nargs="?", help="episode directory or data.json")
    parser.add_argument("--episode", dest="episode_opt", help="episode directory or data.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="enable DDS commands; without this flag replay is visualization-only",
    )
    mode.add_argument("--dry-run", action="store_true", help="explicit visualization-only alias")
    parser.add_argument("--domain-id", type=int, default=None, help="DDS domain; defaults to episode metadata, then 30")
    parser.add_argument("--hz", type=float, default=0.0, help="override episode replay rate")
    parser.add_argument("--no-arm", action="store_true")
    parser.add_argument("--no-hand", action="store_true")
    parser.add_argument(
        "--allow-metadata-mismatch",
        action="store_true",
        help="allow a manually verified legacy/non-AI metadata header",
    )
    parser.add_argument("--joint-state-timeout", type=float, default=5.0)
    parser.add_argument("--command-duration", type=float, default=0.0, help="0 uses two replay periods")
    parser.add_argument("--arm-velocity-limit", type=float, default=3.0)
    parser.add_argument("--startup-blend", type=float, default=2.0)
    parser.add_argument("--resume-blend", type=float, default=0.5)
    parser.add_argument("--blend-hz", type=float, default=50.0)
    parser.add_argument(
        "--post-replay-prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="after each executed replay, accept R to replay or Q to move the arms home and exit",
    )
    parser.add_argument(
        "--ai-worker-model",
        choices=("sg2", "sh5"),
        default="sg2",
        help="robot model used to select the Q-key home pose",
    )
    parser.add_argument(
        "--home-duration",
        type=float,
        default=3.0,
        help="seconds used for the Q-key smooth move to the model home pose",
    )
    parser.add_argument(
        "--pedal-estop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require and honor ESTOP state from ai_worker_pedal_teleop.py",
    )
    parser.add_argument("--pedal-estop-host", default="127.0.0.1")
    parser.add_argument("--pedal-estop-port", type=int, default=8765)
    parser.add_argument("--pedal-state-timeout", type=float, default=2.0)
    parser.add_argument("--pedal-safety-host", default="127.0.0.1")
    parser.add_argument("--pedal-safety-port", type=int, default=8766)
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show replay curves and saved cameras in Rerun",
    )
    parser.add_argument("--rerun-viewer", choices=("web", "native", "connect", "auto", "off"), default=None)
    parser.add_argument("--rerun-prefix", default="ai_worker_replay/")
    parser.add_argument("--rerun-memory-limit", default="300MB")
    parser.add_argument("--rerun-idx-window", type=int, default=300)
    parser.add_argument("--visualization-queue-size", type=int, default=4)
    parser.add_argument("--viewer-hold-seconds", type=float, default=0.0)
    args = parser.parse_args(argv)
    args.episode_path = args.episode_opt or args.episode_path
    if not args.episode_path:
        parser.error("episode path is required")
    if args.no_arm and args.no_hand:
        parser.error("--no-arm and --no-hand cannot both be set")
    if args.post_replay_prompt and not args.execute:
        parser.error("--post-replay-prompt requires --execute")
    if args.post_replay_prompt and args.no_arm:
        parser.error("--post-replay-prompt cannot be combined with --no-arm")
    for name in (
        "joint_state_timeout",
        "startup_blend",
        "resume_blend",
        "viewer_hold_seconds",
        "pedal_state_timeout",
        "hz",
        "command_duration",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and zero or greater")
    for name in ("arm_velocity_limit", "blend_hz", "home_duration"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and greater than zero")
    if args.domain_id is not None and args.domain_id < 0:
        parser.error("--domain-id must be zero or greater")
    return args


def main(argv=None):
    args = parse_args(argv)
    fps, source_frames, episode_json, info = load_episode(args.episode_path)
    replay_frames = preflight_ai_worker_episode(
        info,
        source_frames,
        replay_arm=not args.no_arm,
        replay_hand=not args.no_hand,
        allow_metadata_mismatch=args.allow_metadata_mismatch,
    )

    replay_hz = args.hz or fps
    if not math.isfinite(replay_hz) or replay_hz <= 0.0:
        raise ValueError("replay frequency must be a finite number greater than zero")
    period = 1.0 / replay_hz
    command_duration = args.command_duration or max(0.02, 2.0 * period)
    metadata_domain = _episode_domain(info)
    domain_id = args.domain_id if args.domain_id is not None else (metadata_domain if metadata_domain is not None else 30)
    if (
        args.domain_id is not None
        and metadata_domain is not None
        and args.domain_id != metadata_domain
        and not args.allow_metadata_mismatch
    ):
        raise ValueError(
            f"requested DDS domain {args.domain_id} differs from episode domain "
            f"{metadata_domain}; pass --allow-metadata-mismatch after verification"
        )

    execute = bool(args.execute)
    logger.info(
        "AI Worker replay preflight passed: file=%s frames=%d hz=%.2f mode=%s arm=%s hand=%s domain=%d",
        episode_json,
        len(replay_frames),
        replay_hz,
        "DDS EXECUTE" if execute else "VISUALIZATION ONLY",
        not args.no_arm,
        not args.no_hand,
        domain_id,
    )
    if not execute:
        logger.warning("No DDS commands will be sent. Add --execute only after reviewing this episode.")

    visualizer = None
    arm_ctrl = None
    hand_pub = None
    estop = None
    inhibitor = None
    replay_completed = False
    try:
        if execute:
            # The transport reads this environment variable during construction.
            os.environ["ROS_DOMAIN_ID"] = str(domain_id)
            inhibitor = PedalMotionInhibitor(args.pedal_safety_host, args.pedal_safety_port)
            _publish_inhibit(inhibitor)
            if args.pedal_estop:
                estop = PedalEstopReceiver(args.pedal_estop_host, args.pedal_estop_port)
                if not estop.wait_for_state(args.pedal_state_timeout):
                    raise RuntimeError(
                        "No pedal/keyboard ESTOP heartbeat was received. Start "
                        "ai_worker_pedal_teleop.py (including its --keyboard mode) "
                        "or explicitly use --no-pedal-estop for an isolated test."
                    )
                logger.info(
                    "Pedal/keyboard ESTOP heartbeat received: state=%s",
                    "ON" if estop.active else "OFF",
                )

            if not args.no_arm:
                from teleop.robot_control.robotis_ai_worker import AIWorkerArmController

                arm_ctrl = AIWorkerArmController(
                    command_duration=command_duration,
                    home_q=_arm_home_q(args.ai_worker_model),
                )
                arm_ctrl.arm_velocity_limit = float(args.arm_velocity_limit)
                if not arm_ctrl.wait_for_joint_state(args.joint_state_timeout):
                    raise RuntimeError("All 14 AI Worker arm joints were not received on /joint_states.")
                arm_ctrl.sync_arm_command_to_measured()

            if not args.no_hand:
                from teleop.robot_control.robot_hand_hx5_replay import AIWorkerHX5ReplayPublisher

                hand_pub = AIWorkerHX5ReplayPublisher(command_duration=command_duration)
                if not hand_pub.wait_for_joint_state(args.joint_state_timeout):
                    raise RuntimeError("All 40 HX5 joints were not received on /joint_states.")

            first_arm, first_hand = _targets_for_frame(replay_frames[0])
            _blend_to_target(
                arm_ctrl,
                hand_pub,
                first_arm,
                first_hand,
                args.startup_blend,
                args.blend_hz,
                estop,
                inhibitor,
            )

        # In execute mode, do not open a blank viewer until all safety and DDS
        # prerequisites have passed. Visualization-only mode still opens it
        # immediately because it has no hardware prerequisites.
        if args.visualize:
            visualizer = ReplayVisualizer(
                episode_dir=episode_json.parent,
                info=info,
                prefix=args.rerun_prefix,
                idx_window=args.rerun_idx_window,
                memory_limit=args.rerun_memory_limit,
                viewer=args.rerun_viewer,
                queue_size=args.visualization_queue_size,
            )

        cycle_index = 0
        timeline_index = 0
        while True:
            if cycle_index > 0:
                first_arm, first_hand = _targets_for_frame(replay_frames[0])
                logger.info(
                    "Blending to the first frame before replay cycle %d.",
                    cycle_index + 1,
                )
                _blend_to_target(
                    arm_ctrl,
                    hand_pub,
                    first_arm,
                    first_hand,
                    args.startup_blend,
                    args.blend_hz,
                    estop,
                    inhibitor,
                )

            next_deadline = time.monotonic()
            for sequence_index, replay_frame in enumerate(replay_frames):
                arm_target, hand_target = _targets_for_frame(replay_frame)
                resumed = False
                if execute:
                    resumed = _wait_while_estopped(estop, inhibitor, arm_ctrl, hand_pub)
                    if resumed:
                        _blend_to_target(
                            arm_ctrl,
                            hand_pub,
                            arm_target,
                            hand_target,
                            args.resume_blend,
                            args.blend_hz,
                            estop,
                            inhibitor,
                        )
                        next_deadline = time.monotonic()
                    sent_actions = _send_target(
                        arm_ctrl,
                        hand_pub,
                        arm_target,
                        hand_target,
                    )
                else:
                    # Do not fabricate a DDS-sent command in visualization-only mode.
                    sent_actions = None

                next_deadline += period
                estopped_during_frame = _wait_until(next_deadline, estop, inhibitor)
                if next_deadline < time.monotonic() - period:
                    next_deadline = time.monotonic()

                if execute and estopped_during_frame:
                    _safe_hold(arm_ctrl, hand_pub)
                    # The measured-pose overwrite is now the most recent command;
                    # do not leave the graph labelled with the pre-E-stop target.
                    sent_actions = _sent_group(arm_ctrl, hand_pub)
                live_states = _live_group(arm_ctrl, hand_pub) if execute else None
                if visualizer is not None:
                    visualizer.submit(
                        replay_frame.source_frame,
                        sent_actions=sent_actions,
                        live_states=live_states,
                        replay_time_s=timeline_index / replay_hz,
                        timeline_idx=timeline_index,
                        status={
                            "execute": execute,
                            "estop": bool(estop is not None and estop.active),
                            "resumed": resumed,
                            "cycle_index": cycle_index,
                            "sequence_index": sequence_index,
                            "source_frame_idx": replay_frame.idx,
                        },
                    )
                timeline_index += 1

            logger.info(
                "AI Worker replay cycle %d completed %d frames.",
                cycle_index + 1,
                len(replay_frames),
            )
            if not args.post_replay_prompt:
                break

            choice = _read_post_replay_choice()
            if choice == "r":
                cycle_index += 1
                continue

            logger.info(
                "Q selected: moving %s arms to home over %.2fs.",
                args.ai_worker_model.upper(),
                args.home_duration,
            )
            _blend_to_target(
                arm_ctrl,
                hand_pub,
                arm_ctrl.home_q,
                None,
                args.home_duration,
                args.blend_hz,
                estop,
                inhibitor,
            )
            logger.info("Home pose reached; replay is exiting.")
            break

        replay_completed = True
        return 0
    except KeyboardInterrupt:
        logger.warning("Replay interrupted by operator.")
        return 130
    finally:
        _publish_inhibit(inhibitor)
        if execute and (arm_ctrl is not None or hand_pub is not None):
            for _ in range(3):
                try:
                    _safe_hold(arm_ctrl, hand_pub)
                except Exception as exc:
                    logger.error("Replay exit hold failed: %s", exc)
                    break
                time.sleep(0.02)
        if hand_pub is not None:
            try:
                hand_pub.close()
            except Exception as exc:
                logger.error("Failed to close HX5 replay transport: %s", exc)
        if arm_ctrl is not None:
            try:
                arm_ctrl.close()
            except Exception as exc:
                logger.error("Failed to close AI Worker arm transport: %s", exc)
        if replay_completed and args.viewer_hold_seconds > 0.0 and visualizer is not None:
            logger.info("Keeping replay viewer available for %.1f seconds.", args.viewer_hold_seconds)
            time.sleep(args.viewer_hold_seconds)
        if visualizer is not None:
            try:
                visualizer.close(drain=True)
                if visualizer.dropped_frames:
                    logger.info("Replay viewer dropped %d stale display frames.", visualizer.dropped_frames)
            except Exception as exc:
                logger.error("Failed to close replay visualization: %s", exc)
        if estop is not None:
            try:
                estop.close()
            except Exception as exc:
                logger.error("Failed to close replay E-stop receiver: %s", exc)
        if inhibitor is not None:
            _publish_inhibit(inhibitor)
            try:
                inhibitor.close()
            except Exception as exc:
                logger.error("Failed to close pedal motion inhibitor: %s", exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
