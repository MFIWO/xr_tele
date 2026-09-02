#!/usr/bin/env python3
"""
Safe RH5DG2 serial-to-DDS bridge.

Default mode is state-only: read serial angleAct/forceAct and publish
rt/rh5dg2/state as unitree_go::msg::dds_::MotorStates_.

Command mode is intentionally locked down. It is disabled unless
--enable-command is passed, dry-run unless --write-serial is passed, and real
serial writes are still refused unless the operator explicitly accepts raw
angleSet risk with --allow-raw-angle-write.
"""

import argparse
import fcntl
import json
import math
import os
import socket
import struct
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import serial
from cyclonedds.domain import DomainParticipant
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import array, float32, sequence, uint8, uint32
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic


REG = {
    "ID": 1000,
    "baudrate": 1001,
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
    "actionSeq": 2160,
    "actionRun": 2162,
}

FINGERS = ("little", "ring", "middle", "index", "thumb")
TACTILE_ADDR = 0x0BB8
TACTILE_BYTES = 0x44
TACTILE_SOURCE = "rh5dg2_serial_dds_bridge"


def parse_tactile_payload(payload: Sequence[int]) -> Tuple[Dict[str, List[int]], List[int]]:
    need = len(FINGERS) * 10 + 9 * 2
    if len(payload) < need:
        raise ValueError(f"short tactile payload: got {len(payload)} bytes, need {need}")

    data = bytes(int(v) & 0xFF for v in payload)
    fingers: Dict[str, List[int]] = {}
    for index, finger in enumerate(FINGERS):
        base = index * 10
        fingers[finger] = list(struct.unpack("<4H", data[base : base + 8]))

    palm_base = len(FINGERS) * 10
    palm = list(struct.unpack("<9H", data[palm_base : palm_base + 18]))
    return fingers, palm


def make_tactile_hand_payload(side: str, fingers: Dict[str, List[int]], palm: List[int]) -> Dict[str, object]:
    return {
        side: {
            "timestamp": time.time(),
            "source": TACTILE_SOURCE,
            "fingers": fingers,
            "palm": palm,
        }
    }


def parse_udp_target(text: str, default_host: str) -> Tuple[str, int]:
    if ":" not in text:
        raise ValueError(f"UDP target must be HOST:PORT, got {text!r}")
    host, port_text = text.rsplit(":", 1)
    host = host or default_host
    port = int(port_text)
    if not 1 <= port <= 65535:
        raise ValueError(f"UDP target port out of range: {port}")
    return host, port


def build_tactile_udp_targets(args: argparse.Namespace) -> List[Tuple[str, int]]:
    ports: Sequence[int] = args.tactile_target_port if args.tactile_target_port else [9105, 56011]
    targets = [(args.tactile_target_host, int(port)) for port in ports]
    for target in args.tactile_target or []:
        targets.append(parse_udp_target(target, args.tactile_target_host))
    for _host, port in targets:
        if not 1 <= int(port) <= 65535:
            raise ValueError(f"UDP target port out of range: {port}")
    return list(dict.fromkeys(targets))


def optional_int(text: str) -> Optional[int]:
    if text.lower() in {"none", "null", "off", ""}:
        return None
    return int(text)


def parse_index_list(text: str) -> Tuple[int, ...]:
    if text.lower() in {"none", "null", "off", ""}:
        return ()
    indices: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 0 or value >= 13:
            raise argparse.ArgumentTypeError(f"hand-local index out of range [0, 12]: {value}")
        indices.append(value)
    return tuple(indices)


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
    print(f"[rh5dg2-bridge] {msg}", flush=True)


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
    f = tempfile.NamedTemporaryFile("w", prefix="rh5dg2_cyclonedds_", suffix=".xml", delete=False)
    f.write(xml)
    f.close()
    os.environ["CYCLONEDDS_URI"] = f"file://{f.name}"
    return f.name


class InspireSerialBus:
    def __init__(
        self,
        port: str,
        baudrate: int,
        timeout: float,
        io_delay: float,
        verbose: bool,
        read_retries: int,
        reconnect_attempts: int,
        reconnect_delay: float,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.io_delay = io_delay
        self.verbose = verbose
        self.read_retries = max(0, read_retries)
        self.reconnect_attempts = max(0, reconnect_attempts)
        self.reconnect_delay = max(0.0, reconnect_delay)
        self.ser: Optional[serial.Serial] = None
        self._open_serial()

    def _open_serial(self) -> None:
        self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        try:
            fcntl.flock(self.ser.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.ser.close()
            self.ser = None
            raise RuntimeError(
                f"serial port {self.port} is already in use; stop the RH5DG2 bridge or other diagnostic first"
            ) from exc

    def close(self) -> None:
        if self.ser is not None and self.ser.is_open:
            try:
                fcntl.flock(self.ser.fileno(), fcntl.LOCK_UN)
            finally:
                self.ser.close()

    def reconnect(self) -> bool:
        self.close()
        for attempt in range(1, self.reconnect_attempts + 1):
            if self.reconnect_delay > 0:
                time.sleep(self.reconnect_delay)
            try:
                self._open_serial()
                log(f"serial reconnect ok port={self.port} attempt={attempt}")
                return True
            except Exception as exc:
                log(f"WARN serial reconnect failed port={self.port} attempt={attempt}: {exc}")
        return False

    def _write_register(self, hand_id: int, address: int, payload: Sequence[int]) -> None:
        if self.ser is None or not self.ser.is_open:
            if not self.reconnect():
                raise RuntimeError(f"serial port {self.port} is not open and reconnect failed")
        frame = [0xEB, 0x90, hand_id, len(payload) + 3, 0x12, address & 0xFF, (address >> 8) & 0xFF]
        frame.extend(int(v) & 0xFF for v in payload)
        frame.append(sum(frame[2:]) & 0xFF)
        if self.verbose:
            log(f"serial write id={hand_id} addr={address} bytes={[hex(b) for b in frame]}")
        self.ser.write(bytes(frame))
        time.sleep(self.io_delay)
        self.ser.reset_input_buffer()

    def _read_register_bytes(self, hand_id: int, address: int, count: int) -> List[int]:
        last_exc: Optional[Exception] = None
        for attempt in range(self.read_retries + 1):
            try:
                return self._read_register_bytes_once(hand_id, address, count)
            except (TimeoutError, serial.SerialException, OSError) as exc:
                last_exc = exc
                if attempt >= self.read_retries:
                    break
                log(
                    f"WARN serial read retry id={hand_id} addr={address} "
                    f"attempt={attempt + 1}/{self.read_retries}: {exc}"
                )
                if isinstance(exc, (serial.SerialException, OSError)):
                    self.reconnect()
                elif self.io_delay > 0:
                    time.sleep(self.io_delay)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("serial read failed without an exception")

    def _read_register_bytes_once(self, hand_id: int, address: int, count: int) -> List[int]:
        if self.ser is None or not self.ser.is_open:
            if not self.reconnect():
                raise RuntimeError(f"serial port {self.port} is not open and reconnect failed")
        self.ser.reset_input_buffer()
        frame = [0xEB, 0x90, hand_id, 0x04, 0x11, address & 0xFF, (address >> 8) & 0xFF, count]
        frame.append(sum(frame[2:]) & 0xFF)
        if self.verbose:
            log(f"serial read id={hand_id} addr={address} count={count} bytes={[hex(b) for b in frame]}")
        self.ser.write(bytes(frame))
        time.sleep(self.io_delay)

        deadline = time.monotonic() + max(0.2, self.timeout)
        data = bytearray()
        while time.monotonic() < deadline:
            chunk = self.ser.read_all()
            if chunk:
                data.extend(chunk)
                if len(data) >= 7 + count:
                    break
            time.sleep(0.005)

        if self.verbose:
            log(f"serial response id={hand_id} addr={address} len={len(data)} raw={[hex(b) for b in data]}")
        if len(data) < 7 + count:
            raise TimeoutError(f"id={hand_id} addr={address}: short response len={len(data)} expected>={7 + count}")

        start = self._find_payload_start(data, address)
        if start is None or start + count > len(data):
            # The vendor examples index payload at byte 7. Keep this fallback for
            # compatibility with their simple parser.
            start = 7
        return [int(v) for v in data[start : start + count]]

    @staticmethod
    def _find_payload_start(data: bytearray, address: int) -> Optional[int]:
        lo = address & 0xFF
        hi = (address >> 8) & 0xFF
        for i in range(0, max(0, len(data) - 2)):
            if data[i] == lo and data[i + 1] == hi:
                return i + 2
        return None

    def read13(self, hand_id: int, name: str) -> List[int]:
        if name not in {"angleSet", "forceSet", "speedSet", "angleAct", "forceAct", "temp", "errCode", "statusCode"}:
            raise ValueError(f"unsupported register read: {name}")
        vals = self._read_register_bytes(hand_id, REG[name], 28)
        out: List[int] = []
        for i in range(13):
            v = (vals[2 * i] & 0xFF) | ((vals[2 * i + 1] & 0xFF) << 8)
            if v > 32767:
                v -= 65536
            out.append(v)
        return out

    def read_tactile(self, hand_id: int) -> Tuple[Dict[str, List[int]], List[int]]:
        vals = self._read_register_bytes(hand_id, TACTILE_ADDR, TACTILE_BYTES)
        return parse_tactile_payload(vals)

    def write13(self, hand_id: int, name: str, values: Sequence[int]) -> None:
        if name not in {"angleSet", "forceSet", "speedSet", "mode"}:
            raise ValueError(f"unsupported register write: {name}")
        if len(values) != 13:
            raise ValueError(f"{name} requires 13 values, got {len(values)}")
        payload: List[int] = []
        for v in values:
            iv = int(v)
            payload.append(iv & 0xFF)
            payload.append((iv >> 8) & 0xFF)
        self._write_register(hand_id, REG[name], payload)


@dataclass
class HandConfig:
    label: str
    hand_id: Optional[int]


class Bridge:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.hands = [HandConfig("right", args.right_id), HandConfig("left", args.left_id)]
        self.bus = InspireSerialBus(
            args.serial_port,
            args.baudrate,
            args.serial_timeout,
            args.serial_delay,
            args.verbose,
            args.serial_read_retries,
            args.serial_reconnect_attempts,
            args.serial_reconnect_delay,
        )

        config_path = configure_cyclonedds_interface(args.network_interface)
        if config_path:
            log(f"CycloneDDS interface={args.network_interface} config={config_path}")

        self.dp = DomainParticipant(args.domain)
        self.state_topic = Topic(self.dp, args.state_topic, MotorStates_)
        self.state_writer = DataWriter(self.dp, self.state_topic)

        self.cmd_reader = None
        if args.enable_command:
            self.cmd_topic = Topic(self.dp, args.cmd_topic, MotorCmds_)
            self.cmd_reader = DataReader(self.dp, self.cmd_topic)

        self.last_angle: List[Optional[List[int]]] = [None, None]
        self.last_force: List[Optional[List[int]]] = [None, None]
        self.last_state_time = 0.0
        self.last_report_time = 0.0
        self.last_tactile_time = 0.0
        self.last_command_write_time = 0.0
        self.command_inhibit_until = 0.0
        self.read_count = 0
        self.publish_count = 0
        self.tactile_read_count = 0
        self.tactile_sent_count = 0
        self.tactile_error_count = 0
        self.last_tactile_keys: List[str] = []
        self.start_time = time.monotonic()
        self.tactile_sock = None
        self.tactile_targets: List[Tuple[str, int]] = []
        if args.enable_tactile_udp:
            self.tactile_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.tactile_targets = build_tactile_udp_targets(args)

    def run(self) -> None:
        log("starting safe bridge")
        log(f"serial port={self.args.serial_port} baudrate={self.args.baudrate} timeout={self.args.serial_timeout}")
        log(
            f"serial recovery read_retries={self.args.serial_read_retries} "
            f"reconnect_attempts={self.args.serial_reconnect_attempts} "
            f"reconnect_delay={self.args.serial_reconnect_delay}"
        )
        log(f"hand IDs: right={self.args.right_id} left={self.args.left_id}")
        log(f"DDS domain={self.args.domain} state_topic={self.args.state_topic} type=unitree_go::msg::dds_::MotorStates_")
        log("state q conversion: raw angleAct register value is published unchanged; units are not proven")
        log(
            f"loop defaults rate_hz={self.args.rate_hz} command_rate_hz={self.args.command_rate_hz} "
            f"post_write_readback_delay={self.args.post_write_readback_delay} "
            f"max_delta_raw={self.args.max_delta_raw} raw_range=[{self.args.raw_min}, {self.args.raw_max}]"
        )
        if self.args.enable_command:
            log(f"COMMAND MODE enabled: cmd_topic={self.args.cmd_topic} dry_run={not self.args.write_serial}")
            if self.args.zero_roll_indices:
                log(f"roll zero clamp enabled: hand-local indices={list(self.args.zero_roll_indices)}")
            else:
                log("roll zero clamp disabled: passing roll/spread commands through")
            if self.args.write_serial and not self.args.allow_raw_angle_write:
                log("serial command writes refused: --allow-raw-angle-write was not provided")
        else:
            log("command mode disabled: no DDS command subscriber is created")
        if self.args.enable_tactile_udp:
            target_text = ", ".join(f"{host}:{port}" for host, port in self.tactile_targets)
            log(
                f"tactile UDP enabled: targets={target_text} "
                f"rate={self.args.tactile_hz}Hz"
            )

        period = 1.0 / self.args.rate_hz
        try:
            while True:
                loop_start = time.monotonic()
                self.read_state_once()
                self.publish_state_once()
                if self.cmd_reader is not None:
                    self.process_commands()
                self.publish_tactile_udp_once()
                self.report_periodically()
                elapsed = time.monotonic() - loop_start
                time.sleep(max(0.0, period - elapsed))
        finally:
            if self.tactile_sock is not None:
                self.tactile_sock.close()
            self.bus.close()

    def read_state_once(self) -> None:
        any_ok = False
        for idx, hand in enumerate(self.hands):
            if hand.hand_id is None:
                self.last_angle[idx] = None
                self.last_force[idx] = None
                continue
            try:
                angle = self.bus.read13(hand.hand_id, "angleAct")
                force = self.bus.read13(hand.hand_id, "forceAct") if self.args.read_force else None
                self.last_angle[idx] = angle
                self.last_force[idx] = force
                any_ok = True
                if self.args.verbose:
                    log(f"{hand.label} id={hand.hand_id} angleAct={angle} forceAct={force}")
            except Exception as exc:
                log(f"WARN state read failed for {hand.label} id={hand.hand_id}: {exc}")
                self.last_angle[idx] = None
                self.last_force[idx] = None
        if any_ok:
            self.last_state_time = time.monotonic()
            self.read_count += 1

    def publish_state_once(self) -> None:
        states: List[MotorState_] = []
        for idx in range(2):
            angle = self.last_angle[idx]
            force = self.last_force[idx]
            for j in range(13):
                if angle is None:
                    states.append(MotorState_(q=0.0, q_raw=0.0, tau_est=0.0, lost=1, reserve=(0, 0)))
                else:
                    q = float(angle[j])
                    tau_est = float(force[j]) if force is not None else 0.0
                    states.append(MotorState_(q=q, q_raw=q, tau_est=tau_est, lost=0, reserve=(0, 0)))
        self.state_writer.write(MotorStates_(states=states))
        self.publish_count += 1

    def publish_tactile_udp_once(self) -> None:
        if self.tactile_sock is None or not self.tactile_targets:
            return
        now = time.monotonic()
        if now - self.last_tactile_time < 1.0 / max(0.1, self.args.tactile_hz):
            return
        self.last_tactile_time = now

        payload: Dict[str, object] = {}
        for hand in self.hands:
            if hand.hand_id is None:
                continue
            side = "right_ee" if hand.label == "right" else "left_ee"
            try:
                fingers, palm = self.bus.read_tactile(hand.hand_id)
                payload.update(make_tactile_hand_payload(side, fingers, palm))
                self.tactile_read_count += 1
            except Exception as exc:
                self.tactile_error_count += 1
                if self.args.verbose:
                    log(f"WARN tactile read failed for {hand.label} id={hand.hand_id}: {exc}")

        if not payload:
            self.last_tactile_keys = []
            return
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        for target in self.tactile_targets:
            try:
                self.tactile_sock.sendto(encoded, target)
                self.tactile_sent_count += 1
            except OSError as exc:
                self.tactile_error_count += 1
                if self.args.verbose:
                    log(f"WARN tactile UDP send failed target={target[0]}:{target[1]}: {exc}")
        self.last_tactile_keys = sorted(payload.keys())

    def process_commands(self) -> None:
        samples = self.cmd_reader.take(10)
        if not samples:
            return
        for sample in samples:
            cmds_obj = getattr(sample, "cmds", None)
            if cmds_obj is None:
                log(f"skip invalid DDS sample type={type(sample).__name__}")
                continue

            cmds = list(cmds_obj)
            vals = [float(c.q) for c in cmds[:26]]
            finite_vals = [v for v in vals if math.isfinite(v)]
            log(
                f"COMMAND RECEIVED len={len(cmds)} "
                f"q_min={min(finite_vals) if finite_vals else None} "
                f"q_max={max(finite_vals) if finite_vals else None} q={vals}"
            )
            if len(cmds) != 26:
                log(f"REJECT command: expected 26 cmds, got {len(cmds)}")
                continue

            self.zero_forced_roll(vals)

            right_active = self.command_hand_active(cmds, 0)
            left_active = self.command_hand_active(cmds, 13)
            if self.args.write_right_only:
                left_active = False
            if self.args.write_left_only:
                right_active = False

            selected = []
            if right_active:
                ok, reason = self.validate_hand_command(vals[:13], "right")
                if not ok:
                    log(f"REJECT right command: {reason}")
                    continue
                cur = self.current_hand_angles(0)
                target = self.limit_hand_command(vals[:13], cur)
                selected.append(("right", self.args.right_id, cur, target))
            else:
                cur = self.current_hand_angles(0)
                target = [int(round(v)) for v in vals[:13] if math.isfinite(v)]
                log(f"SKIP right reason=inactive_command current={cur} target_hint={target}")

            if left_active:
                ok, reason = self.validate_hand_command(vals[13:26], "left")
                if not ok:
                    log(f"REJECT left command: {reason}")
                    continue
                cur = self.current_hand_angles(1)
                target = self.limit_hand_command(vals[13:26], cur)
                selected.append(("left", self.args.left_id, cur, target))
            else:
                cur = self.current_hand_angles(1)
                target = [int(round(v)) for v in vals[13:26] if math.isfinite(v)]
                log(f"SKIP left reason=inactive_command current={cur} target_hint={target}")

            self.log_command_selection(selected, right_active, left_active)
            if not self.args.write_serial:
                log(f"DRY-RUN command accepted but not written: {[(s[0], s[3]) for s in selected]}")
                continue
            if not self.args.allow_raw_angle_write:
                log("REJECT serial write: raw angle mapping/unit conversion is not proven")
                continue
            if time.monotonic() < self.command_inhibit_until:
                remaining = self.command_inhibit_until - time.monotonic()
                log(f"REJECT serial write: command fault cooldown active remaining={remaining:.2f}s")
                continue
            self.write_selected_commands(selected)

    @staticmethod
    def command_hand_active(cmds: Sequence[MotorCmd_], start: int) -> bool:
        hand_cmds = cmds[start : start + 13]
        finite_any = any(math.isfinite(float(c.q)) for c in hand_cmds)
        explicit_any = False
        for c in hand_cmds:
            reserve = getattr(c, "reserve", (0, 0, 0))
            reserve0 = int(reserve[0]) if len(reserve) > 0 else 0
            if int(getattr(c, "mode", 0)) != 0 or reserve0 == 1:
                explicit_any = True
                break
        return finite_any and explicit_any

    def zero_forced_roll(self, vals: List[float]) -> None:
        if not self.args.zero_roll_indices:
            return
        changed: List[Tuple[int, float]] = []
        for base in (0, 13):
            for local_index in self.args.zero_roll_indices:
                index = base + local_index
                if index >= len(vals) or not math.isfinite(vals[index]):
                    continue
                if vals[index] != 0.0:
                    changed.append((index, vals[index]))
                vals[index] = 0.0
        if changed:
            log(f"ZERO roll command indices={self.args.zero_roll_indices} changed={changed}")

    def validate_hand_command(self, vals: Sequence[float], label: str) -> Tuple[bool, str]:
        if len(vals) != 13:
            return False, f"expected 13 q values for {label}, got {len(vals)}"
        if time.monotonic() - self.last_state_time > self.args.state_timeout:
            return False, "state is stale or has never been read"
        if any(not math.isfinite(v) for v in vals):
            return False, "non-finite q value"
        for v in vals:
            if v < self.args.raw_min or v > self.args.raw_max:
                return False, f"q outside raw safety range [{self.args.raw_min}, {self.args.raw_max}]"
        return True, "ok"

    def limit_hand_command(self, vals: Sequence[float], current: Sequence[int]) -> List[int]:
        out: List[int] = []
        now = time.monotonic()
        min_interval = 1.0 / max(0.1, self.args.command_rate_hz)
        if now - self.last_command_write_time < min_interval:
            log("rate limit: holding previous state because command arrived too fast")
            return [int(v) for v in current]

        for target, cur in zip(vals, current):
            target_i = int(round(max(self.args.raw_min, min(self.args.raw_max, target))))
            delta = target_i - cur
            if not self.args.allow_close and delta < 0:
                target_i = cur
            if abs(target_i - cur) > self.args.max_delta_raw:
                target_i = cur + (self.args.max_delta_raw if target_i > cur else -self.args.max_delta_raw)
            out.append(int(target_i))
        return out

    def current_hand_angles(self, hand_index: int) -> List[int]:
        angle = self.last_angle[hand_index]
        if angle is None:
            return [0] * 13
        return [int(v) for v in angle]

    def current_angles_flat(self) -> List[int]:
        flat: List[int] = []
        for angle in self.last_angle:
            if angle is None:
                flat.extend([0] * 13)
            else:
                flat.extend(int(v) for v in angle)
        return flat

    def log_command_selection(
        self,
        selected: Sequence[Tuple[str, Optional[int], List[int], List[int]]],
        right_active: bool,
        left_active: bool,
    ) -> None:
        by_label = {label: (hand_id, cur, target) for label, hand_id, cur, target in selected}
        for label, active in (("right", right_active), ("left", left_active)):
            if label in by_label:
                _hand_id, cur, target = by_label[label]
                deltas = [t - c for t, c in zip(target, cur)]
                log(f"{label.upper()} ACTIVE={active} TARGET={target} DELTA={deltas}")
            else:
                log(f"{label.upper()} ACTIVE={active} TARGET=[] DELTA=[]")

    def write_selected_commands(self, selected: Sequence[Tuple[str, Optional[int], List[int], List[int]]]) -> None:
        changed = [
            (label, hand_id, cur, target)
            for label, hand_id, cur, target in selected
            if hand_id is not None and any(t != c for t, c in zip(target, cur))
        ]

        if not changed:
            log("WRITE skipped: command equals current angleAct after safety limits")
            return

        try:
            for label, hand_id, cur, target in changed:
                deltas = [t - c for t, c in zip(target, cur)]
                log(f"WRITE {label} id={hand_id} current_angleAct={cur} target_angleSet={target} delta={deltas}")
                changed_indices = [i for i, d in enumerate(deltas) if d != 0]
                for i in changed_indices:
                    log(f"WRITE {label} id={hand_id} target_index{i}={target[i]} current_index{i}={cur[i]}")
                self.bus.write13(hand_id, "angleSet", target)
            self.last_command_write_time = time.monotonic()
            self.log_post_write_readback(changed)
        except Exception as exc:
            log(f"WRITE/READBACK ERROR: {exc}; attempting restore to pre-command angleAct")
            self.restore_changed_hands(changed)
            self.command_inhibit_until = time.monotonic() + max(0.0, self.args.command_fault_cooldown)
            self.last_state_time = 0.0
            self.last_angle = [None, None]
            self.last_force = [None, None]
            if self.bus.reconnect():
                log(
                    "serial recovered after write/readback error; "
                    f"commands inhibited for {self.args.command_fault_cooldown:.2f}s"
                )
            else:
                log("WARN serial reconnect failed after write/readback error")
            if self.args.exit_on_write_error:
                raise

    def log_post_write_readback(self, changed: Sequence[Tuple[str, int, List[int], List[int]]]) -> None:
        if self.args.post_write_readback_delay > 0:
            time.sleep(self.args.post_write_readback_delay)
        for label, hand_id, cur, target in changed:
            after = self.bus.read13(hand_id, "angleAct")
            observed = [a - c for a, c in zip(after, cur)]
            target_error = [a - t for a, t in zip(after, target)]
            log(
                f"READBACK {label} id={hand_id} after_angleAct={after} "
                f"observed_delta={observed} target_error={target_error}"
            )
            if label == "right":
                self.last_angle[0] = after
            elif label == "left":
                self.last_angle[1] = after

    def restore_changed_hands(self, changed: Sequence[Tuple[str, int, List[int], List[int]]]) -> None:
        for label, hand_id, cur, _target in changed:
            try:
                log(f"RESTORE {label} id={hand_id} angleSet={cur}")
                self.bus.write13(hand_id, "angleSet", cur)
            except Exception as restore_exc:
                log(f"RESTORE FAILED {label} id={hand_id}: {restore_exc}")

    def report_periodically(self) -> None:
        now = time.monotonic()
        if now - self.last_report_time < self.args.log_interval:
            return
        self.last_report_time = now
        uptime = max(0.001, now - self.start_time)
        right = self.last_angle[0] if self.last_angle[0] is not None else ["lost"] * 13
        if self.args.left_id is None:
            left = ["disabled"] * 13
        else:
            left = self.last_angle[1] if self.last_angle[1] is not None else ["lost"] * 13
        log(
            f"freq read={self.read_count / uptime:.2f}Hz publish={self.publish_count / uptime:.2f}Hz "
            f"tactile_sent={self.tactile_sent_count} tactile_errors={self.tactile_error_count} "
            f"tactile_keys={self.last_tactile_keys} "
            f"right_q={right} left_q={left}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Safe RH5DG2 serial-to-DDS bridge")
    p.add_argument("--serial-port", default="/dev/ttyUSB0")
    p.add_argument("--baudrate", type=int, default=115200)
    p.add_argument("--serial-timeout", type=float, default=0.2)
    p.add_argument("--serial-delay", type=float, default=0.02)
    p.add_argument("--serial-read-retries", type=int, default=3)
    p.add_argument("--serial-reconnect-attempts", type=int, default=5)
    p.add_argument("--serial-reconnect-delay", type=float, default=0.1)
    p.add_argument("--right-id", type=optional_int, default=1)
    p.add_argument("--left-id", type=optional_int, default=2, help="Set to the left hand serial ID when known, or None to disable")
    p.add_argument("--domain", type=int, default=0)
    p.add_argument("--network-interface", default="eth0")
    p.add_argument("--state-topic", default="rt/rh5dg2/state")
    p.add_argument("--cmd-topic", default="rt/rh5dg2/cmd")
    p.add_argument("--rate-hz", type=float, default=30.0)
    p.add_argument("--read-force", action="store_true")
    p.add_argument("--log-interval", type=float, default=1.0)
    p.add_argument("--verbose", action="store_true")

    p.add_argument("--enable-tactile-udp", action=argparse.BooleanOptionalAction, default=True, help="Read tactile data in this serial owner and send UDP JSON")
    p.add_argument("--tactile-target-host", default="192.168.123.6")
    p.add_argument(
        "--tactile-target-port",
        type=int,
        action="append",
        default=None,
        help="UDP destination port on --tactile-target-host; repeat to send identical JSON to multiple ports.",
    )
    p.add_argument(
        "--tactile-target",
        action="append",
        default=None,
        help="Additional tactile UDP destination as HOST:PORT; repeatable.",
    )
    p.add_argument("--tactile-hz", type=float, default=10.0)

    p.add_argument("--enable-command", action="store_true")
    p.add_argument("--write-serial", action="store_true", help="Actually write accepted commands to serial")
    p.add_argument("--write-right-only", action="store_true", help="Ignore left-hand DDS commands")
    p.add_argument("--write-left-only", action="store_true", help="Ignore right-hand DDS commands")
    p.add_argument("--allow-close", action="store_true", help="Allow decreasing raw angle targets")
    p.add_argument("--allow-raw-angle-write", action="store_true", help="Operator accepts that q is raw angleSet")
    p.add_argument("--state-timeout", type=float, default=0.5)
    p.add_argument("--raw-min", type=int, default=-200)
    p.add_argument("--raw-max", type=int, default=2500)
    p.add_argument(
        "--zero-roll-indices",
        type=parse_index_list,
        default=(),
        help="Hand-local roll indices forced to 0 before command validation/write; disabled by default, pass 3,5 to clamp",
    )
    p.add_argument("--max-delta-raw", type=int, default=400)
    p.add_argument("--command-rate-hz", type=float, default=20.0)
    p.add_argument("--post-write-readback-delay", type=float, default=0.0)
    p.add_argument("--command-fault-cooldown", type=float, default=1.0)
    p.add_argument("--exit-on-write-error", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.write_serial and not args.enable_command:
        log("--write-serial ignored because --enable-command is not set")
    if args.write_serial and not args.allow_raw_angle_write:
        log("SAFETY: serial writes will be refused until --allow-raw-angle-write is also supplied")
    if args.allow_close and not args.enable_command:
        log("--allow-close ignored because command mode is disabled")
    if args.write_right_only and args.write_left_only:
        log("ERROR: choose only one of --write-right-only or --write-left-only")
        return 2

    try:
        Bridge(args).run()
    except KeyboardInterrupt:
        log("stopped")
        return 0
    except Exception as exc:
        log(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
