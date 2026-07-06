#!/usr/bin/env python3
"""
RH5DG2 tiny raw delta tester.

This script is for the Host PC. It first subscribes to rt/rh5dg2/state,
captures the current 26 raw q values, then builds a MotorCmds_ command in raw
angleSet units. Default mode is dry-run: no command topic is published unless
--enable-publish is provided.
"""

import argparse
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from cyclonedds.domain import DomainParticipant
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import array, float32, sequence, uint8, uint32
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic


JOINT_INDEX: Dict[str, int] = {
    "j0": 0,
    "j1": 1,
    "j2": 2,
    "j3": 3,
    "j4": 4,
    "j5": 5,
    "j6": 6,
    "j7": 7,
    "j8": 8,
    "j9": 9,
    "j10": 10,
    "j11": 11,
    "j12": 12,
    # Friendly aliases. These are not guaranteed anatomical mappings yet.
    "index_pip": 1,
    "middle_pip": 2,
    "ring_pip": 7,
    "little_pip": 8,
    "thumb": 10,
}


@dataclass
class MotorState_(IdlStruct, typename="unitree_go.msg.dds_.MotorState_"):
    mode: uint8 = 0
    q: float32 = 0.0
    dq: float32 = 0.0
    ddq: float32 = 0.0
    tau_est: float32 = 0.0
    q_raw: float32 = 0.0
    dq_raw: float32 = 0.0
    ddq_raw: float32 = 0.0
    temperature: uint8 = 0
    lost: uint32 = 0
    reserve: array[uint32, 2] = (0, 0)


@dataclass
class MotorStates_(IdlStruct, typename="unitree_go.msg.dds_.MotorStates_"):
    states: sequence[MotorState_] = ()


@dataclass
class MotorCmd_(IdlStruct, typename="unitree_go.msg.dds_.MotorCmd_"):
    mode: uint8 = 0
    q: float32 = 0.0
    dq: float32 = 0.0
    tau: float32 = 0.0
    kp: float32 = 0.0
    kd: float32 = 0.0
    reserve: array[uint32, 3] = (0, 0, 0)


@dataclass
class MotorCmds_(IdlStruct, typename="unitree_go.msg.dds_.MotorCmds_"):
    cmds: sequence[MotorCmd_] = ()


def log(msg: str) -> None:
    print(f"[tiny-raw] {msg}", flush=True)


def configure_cyclonedds_interface(interface: Optional[str]) -> Optional[str]:
    if not interface:
        return None
    xml = f"""<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS>
  <Domain>
    <General>
      <Interfaces>
        <NetworkInterface name="{interface}"/>
      </Interfaces>
      <AllowMulticast>true</AllowMulticast>
    </General>
  </Domain>
</CycloneDDS>
"""
    f = tempfile.NamedTemporaryFile("w", prefix="rh5dg2_tiny_raw_", suffix=".xml", delete=False)
    f.write(xml)
    f.close()
    os.environ["CYCLONEDDS_URI"] = f"file://{f.name}"
    return f.name


def get_joint_offset(hand: str, joint: str, raw_index: Optional[int]) -> int:
    if hand not in {"right", "left"}:
        raise ValueError("--hand must be right or left")
    if raw_index is not None:
        if raw_index < 0 or raw_index > 12:
            raise ValueError("--raw-index must be in 0..12")
        base = 0 if hand == "right" else 13
        return base + raw_index
    if joint not in JOINT_INDEX:
        raise ValueError(f"unknown --joint {joint!r}; use one of: {', '.join(sorted(JOINT_INDEX))}")
    base = 0 if hand == "right" else 13
    return base + JOINT_INDEX[joint]


def local_index_from_global(global_index: int) -> int:
    return global_index if global_index < 13 else global_index - 13


def finite_raw_values(states: Sequence[MotorState_]) -> List[float]:
    vals: List[float] = []
    for s in states[:26]:
        q = float(getattr(s, "q_raw", 0.0))
        if q == 0.0:
            q = float(getattr(s, "q", 0.0))
        vals.append(q)
    return vals


def sample_to_raw_values(sample: object) -> Optional[List[float]]:
    states_obj = getattr(sample, "states", None)
    if states_obj is None:
        log(f"skip invalid state sample type={type(sample).__name__}")
        return None
    states = list(states_obj)
    if len(states) < 26:
        log(f"skip short state sample states_len={len(states)}")
        return None
    vals = finite_raw_values(states)
    if len(vals) != 26:
        return None
    if any(not math.isfinite(v) for v in vals):
        log(f"skip non-finite state q={vals}")
        return None
    return vals


def wait_for_state(reader: DataReader, timeout: float) -> List[float]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        samples = reader.take(10)
        for sample in samples:
            vals = sample_to_raw_values(sample)
            if vals is None:
                continue
            return vals
        time.sleep(0.02)
    raise TimeoutError(f"no valid {26}-state sample received within {timeout:.1f}s")


def wait_for_latest_state(reader: DataReader, timeout: float) -> List[float]:
    deadline = time.monotonic() + timeout
    latest: Optional[List[float]] = None
    while time.monotonic() < deadline:
        samples = reader.take(50)
        for sample in samples:
            vals = sample_to_raw_values(sample)
            if vals is not None:
                latest = vals
        if latest is not None:
            return latest
        time.sleep(0.02)
    raise TimeoutError(f"no valid readback state sample received within {timeout:.1f}s")


def drain_reader(reader: DataReader) -> None:
    while reader.take(50):
        pass


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def make_cmds(
    current: Sequence[float],
    target: Sequence[float],
    active_hand: Optional[str] = None,
    inactive_hand_nan: bool = False,
) -> MotorCmds_:
    cmds = []
    for idx, q in enumerate(target):
        hand = "right" if idx < 13 else "left"
        is_active = active_hand is None or hand == active_hand
        q_value = float(q)
        mode = 1 if is_active else 0
        reserve = (1, 0, 0) if is_active else (0, 0, 0)
        if not is_active and inactive_hand_nan:
            q_value = float("nan")
        cmds.append(MotorCmd_(mode=mode, q=q_value, dq=0.0, tau=0.0, kp=0.0, kd=0.0, reserve=reserve))
    return MotorCmds_(cmds=cmds)


def hand_slice(hand: str) -> slice:
    return slice(0, 13) if hand == "right" else slice(13, 26)


def max_changed(observed: Sequence[float]) -> tuple:
    if not observed:
        return -1, 0.0
    idx = max(range(len(observed)), key=lambda i: abs(observed[i]))
    return idx, observed[idx]


def publish_once(writer: Optional[DataWriter], cmd: MotorCmds_, enabled: bool) -> None:
    if not enabled:
        return
    if writer is None:
        raise RuntimeError("internal error: publish enabled without writer")
    writer.write(cmd)


def command_hand_filter(args: argparse.Namespace, default_hand: str) -> Optional[str]:
    if args.right_only_command and args.left_only_command:
        raise ValueError("choose only one of --right-only-command or --left-only-command")
    if args.right_only_command:
        return "right"
    if args.left_only_command:
        return "left"
    if args.target_hand_only_command:
        return default_hand
    return None


def command_summary(cmd: MotorCmds_) -> str:
    q = [float(getattr(c, "q", 0.0)) for c in cmd.cmds]
    mode = [int(getattr(c, "mode", 0)) for c in cmd.cmds]
    finite = [math.isfinite(v) for v in q]
    return (
        f"right_q={q[:13]} left_q={q[13:26]} "
        f"right_mode={mode[:13]} left_mode={mode[13:26]} "
        f"right_finite={finite[:13]} left_finite={finite[13:26]}"
    )


def run_scan(args: argparse.Namespace, reader: DataReader, writer: Optional[DataWriter]) -> int:
    hand = "right" if args.scan_right else "left"
    if args.start_index < 0 or args.start_index > 12 or args.end_index < 0 or args.end_index > 12:
        log("REFUSE: --start-index and --end-index must be in 0..12")
        return 2
    if args.end_index < args.start_index:
        log("REFUSE: --end-index must be >= --start-index")
        return 2

    log(
        f"scan mode hand={hand} local_range={args.start_index}..{args.end_index} "
        f"delta_raw={args.delta_raw} publish={args.enable_publish}"
    )
    log("csv_header=test_index,target_delta,observed_deltas[13],max_changed_index,max_changed_delta")

    for local_idx in range(args.start_index, args.end_index + 1):
        current = wait_for_state(reader, args.state_timeout)
        before_hand = current[hand_slice(hand)]
        target = list(current)
        global_idx = (0 if hand == "right" else 13) + local_idx
        before = current[global_idx]
        after_target = clamp(before + args.delta_raw, args.raw_min, args.raw_max)
        target[global_idx] = after_target

        log(
            f"scan prepare hand={hand} local_index={local_idx} global_index={global_idx} "
            f"current={before} target={after_target} delta={after_target - before}"
        )

        drain_reader(reader)
        if args.enable_publish:
            active_hand = command_hand_filter(args, hand)
            cmd = make_cmds(current, target, active_hand=active_hand, inactive_hand_nan=args.inactive_hand_nan)
            log(f"scan command active_hand={active_hand} inactive_hand_nan={args.inactive_hand_nan} {command_summary(cmd)}")
            publish_once(writer, cmd, True)
            log(f"scan published local_index={local_idx}")
            time.sleep(args.readback_delay)
            readback = wait_for_latest_state(reader, args.readback_timeout)
        else:
            log(f"scan dry-run local_index={local_idx}: not publishing")
            readback = current

        after_hand = readback[hand_slice(hand)]
        observed = [round(a - b, 3) for a, b in zip(after_hand, before_hand)]
        max_idx, max_delta = max_changed(observed)
        obs_text = "[" + ";".join(str(v) for v in observed) + "]"
        print(f"{local_idx},{after_target - before},{obs_text},{max_idx},{max_delta}", flush=True)

        if local_idx != args.end_index:
            time.sleep(args.scan_sleep)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tiny raw RH5DG2 command tester")
    p.add_argument("--network-interface", default=None)
    p.add_argument("--domain", type=int, default=0)
    p.add_argument("--state-topic", default="rt/rh5dg2/state")
    p.add_argument("--cmd-topic", default="rt/rh5dg2/cmd")
    p.add_argument("--state-timeout", type=float, default=3.0)
    p.add_argument("--hand", choices=["right", "left"], default="right")
    p.add_argument("--joint", default="index_pip")
    p.add_argument("--raw-index", type=int, default=None, help="Hand-local raw actuator index 0..12; overrides --joint")
    p.add_argument("--delta-raw", type=float, default=5.0)
    p.add_argument("--raw-min", type=float, default=0.0)
    p.add_argument("--raw-max", type=float, default=2500.0)
    p.add_argument("--publish-count", type=int, default=3)
    p.add_argument("--hz", type=float, default=5.0)
    p.add_argument("--enable-publish", action="store_true")
    p.add_argument("--allow-negative-delta", action="store_true")
    p.add_argument(
        "--target-hand-only-command",
        action="store_true",
        help="Mark only --hand/scan hand as active command; inactive hand mode/reserve are cleared.",
    )
    p.add_argument("--right-only-command", action="store_true", help="Mark only right-hand command entries active.")
    p.add_argument("--left-only-command", action="store_true", help="Mark only left-hand command entries active.")
    p.add_argument(
        "--inactive-hand-nan",
        action="store_true",
        help="Set inactive-hand q values to NaN. Requires bridge support; useful to prove inactive hand must not be written.",
    )
    p.add_argument("--scan-right", action="store_true")
    p.add_argument("--scan-left", action="store_true")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--end-index", type=int, default=12)
    p.add_argument("--scan-sleep", type=float, default=1.5)
    p.add_argument("--readback-delay", type=float, default=0.5)
    p.add_argument("--readback-timeout", type=float, default=2.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config_path = configure_cyclonedds_interface(args.network_interface)
    if config_path:
        log(f"CycloneDDS interface={args.network_interface} config={config_path}")

    if args.delta_raw < 0 and not args.allow_negative_delta:
        log("REFUSE: negative delta is disabled by default; pass --allow-negative-delta for that dry-run/publish")
        return 2
    if args.delta_raw > 10:
        log("REFUSE: safety limit is --delta-raw <= 10")
        return 2
    if args.scan_right and args.scan_left:
        log("REFUSE: choose only one of --scan-right or --scan-left")
        return 2
    if args.right_only_command and args.left_only_command:
        log("REFUSE: choose only one of --right-only-command or --left-only-command")
        return 2

    dp = DomainParticipant(args.domain)
    state_topic = Topic(dp, args.state_topic, MotorStates_)
    state_reader = DataReader(dp, state_topic)
    cmd_topic = Topic(dp, args.cmd_topic, MotorCmds_)
    cmd_writer = DataWriter(dp, cmd_topic) if args.enable_publish else None

    log(
        f"mode={'PUBLISH' if args.enable_publish else 'DRY-RUN'} domain={args.domain} "
        f"state_topic={args.state_topic} cmd_topic={args.cmd_topic}"
    )
    if args.scan_right or args.scan_left:
        return run_scan(args, state_reader, cmd_writer)

    selector = f"raw_index={args.raw_index}" if args.raw_index is not None else f"joint={args.joint}"
    log(f"waiting for current raw state hand={args.hand} {selector} delta_raw={args.delta_raw}")

    current = wait_for_state(state_reader, args.state_timeout)
    target = list(current)
    idx = get_joint_offset(args.hand, args.joint, args.raw_index)
    local_idx = local_index_from_global(idx)
    before = current[idx]
    after = clamp(before + args.delta_raw, args.raw_min, args.raw_max)
    target[idx] = after

    log(f"current_raw={current}")
    log(
        f"selected global_index={idx} local_raw_index={local_idx} hand={args.hand} "
        f"joint={args.joint if args.raw_index is None else 'ignored'} "
        f"current={before} target={after} delta={after - before}"
    )
    log(f"target_raw={target}")

    active_hand = command_hand_filter(args, args.hand)
    cmd = make_cmds(current, target, active_hand=active_hand, inactive_hand_nan=args.inactive_hand_nan)
    log(f"command active_hand={active_hand} inactive_hand_nan={args.inactive_hand_nan} {command_summary(cmd)}")
    if not args.enable_publish:
        log("dry-run only: not publishing. Add --enable-publish only after PC2 bridge dry-run is running.")
        return 0

    period = 1.0 / max(0.1, args.hz)
    for i in range(args.publish_count):
        cmd_writer.write(cmd)
        log(f"published raw command count={i + 1}/{args.publish_count}")
        time.sleep(period)
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
