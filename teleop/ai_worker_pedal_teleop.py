#!/usr/bin/env python3
"""Drive the AI Worker base/lift from a PCsensor pedal or system keyboard."""

import argparse
import math
import os
from pathlib import Path
import selectors
import socket
import struct
import sys
import threading
import time


EV_KEY = 0x01
KEY_TO_MOTION = {
    17: "w",  # KEY_W
    31: "s",  # KEY_S
    30: "a",  # KEY_A
    32: "d",  # KEY_D
}
KEY_O = 24
KEY_P = 25
KEY_U = 22
CONTROL_KEY_CODES = set(KEY_TO_MOTION) | {KEY_O, KEY_P, KEY_U}
INPUT_EVENT = struct.Struct("@llHHI")


def discover_pedal_keyboards(proc_devices="/proc/bus/input/devices"):
    """Return event-device paths for PCsensor FootSwitch keyboard interfaces."""
    text = Path(proc_devices).read_text(encoding="utf-8", errors="replace")
    devices = []
    for block in text.split("\n\n"):
        if 'N: Name="PCsensor FootSwitch Keyboard"' not in block:
            continue
        for line in block.splitlines():
            if not line.startswith("H: Handlers="):
                continue
            for handler in line.split("=", 1)[1].split():
                if handler.startswith("event"):
                    devices.append(f"/dev/input/{handler}")
    return sorted(set(devices))


def discover_system_keyboards(proc_devices="/proc/bus/input/devices"):
    """Return non-pedal keyboard event devices that provide all control keys."""
    text = Path(proc_devices).read_text(encoding="utf-8", errors="replace")
    devices = []
    for block in text.split("\n\n"):
        if 'N: Name="PCsensor FootSwitch Keyboard"' in block:
            continue
        handlers = []
        key_bits = None
        for line in block.splitlines():
            if line.startswith("H: Handlers="):
                handlers = line.split("=", 1)[1].split()
            elif line.startswith("B: KEY="):
                try:
                    key_bits = int("".join(line.split("=", 1)[1].split()), 16)
                except ValueError:
                    key_bits = None
        if "kbd" not in handlers or key_bits is None:
            continue
        if not all(key_bits & (1 << code) for code in CONTROL_KEY_CODES):
            continue
        devices.extend(
            f"/dev/input/{item}" for item in handlers if item.startswith("event")
        )
    return sorted(set(devices))


class PedalState:
    def __init__(self):
        self._pressed = []

    def update(self, device, code, value):
        motion = KEY_TO_MOTION.get(code)
        if motion is None:
            return
        item = (device, motion)
        if value == 1:  # key down
            if item in self._pressed:
                self._pressed.remove(item)
            self._pressed.append(item)
        elif value == 0:  # key up
            if item in self._pressed:
                self._pressed.remove(item)
        # Ignore value 2 (OS key repeat); physical key state is already known.

    def remove_device(self, device):
        self._pressed = [item for item in self._pressed if item[0] != device]

    @property
    def active_motion(self):
        return self._pressed[-1][1] if self._pressed else None


class PedalAuxState:
    def __init__(self):
        self._lift_keys = set()
        self.estop = False

    def update(self, device, code, value):
        if code in (KEY_O, KEY_P):
            item = (device, code)
            if value == 1:
                self._lift_keys.add(item)
            elif value == 0:
                self._lift_keys.discard(item)
        elif code == KEY_U and value == 1:
            self.estop = not self.estop
            return True
        return False

    def remove_device(self, device):
        self._lift_keys = {item for item in self._lift_keys if item[0] != device}

    @property
    def lift_direction(self):
        codes = {code for _, code in self._lift_keys}
        if KEY_O in codes and KEY_P not in codes:
            return 1
        if KEY_P in codes and KEY_O not in codes:
            return -1
        return 0


class EstopNotifier:
    def __init__(self, host, port):
        self.target = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def publish(self, active):
        self.sock.sendto(f"ESTOP {int(bool(active))}".encode("ascii"), self.target)

    def close(self):
        self.sock.close()


class TeleopSafetyReceiver:
    """Fail closed unless teleop reports valid tracking and no E-stop."""

    def __init__(self, host, port, timeout):
        self.timeout = max(0.05, float(timeout))
        self._allowed = False
        self._last_message = 0.0
        self._running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, int(port)))
        self.sock.settimeout(0.2)
        self.thread = threading.Thread(target=self._run, name="teleop-safety", daemon=True)
        self.thread.start()

    def _run(self):
        while self._running:
            try:
                payload, _ = self.sock.recvfrom(64)
            except socket.timeout:
                continue
            except OSError:
                break
            fields = payload.decode("ascii", errors="ignore").strip().split()
            if len(fields) == 2 and fields[0] == "MOTION" and fields[1] in ("0", "1"):
                self._allowed = fields[1] == "1"
                self._last_message = time.monotonic()

    @property
    def allowed(self):
        return self._allowed and time.monotonic() - self._last_message <= self.timeout

    def close(self):
        self._running = False
        self.sock.close()
        self.thread.join(timeout=1.0)


def motion_is_allowed(estop, tracking_heartbeat_allowed):
    """Safety priority: a latched E-stop always wins over tracking recovery."""
    return bool(tracking_heartbeat_allowed) and not bool(estop)


class CmdVelPublisher:
    def __init__(self, domain_id, linear_speed, angular_speed):
        from robotis_dds_python.idl.geometry_msgs.msg import Twist_, Vector3_
        from robotis_dds_python.tools.topic_manager import TopicManager

        self.Twist = Twist_
        self.Vector3 = Vector3_
        self.linear_speed = float(linear_speed)
        self.angular_speed = float(angular_speed)
        self.manager = TopicManager(domain_id=int(domain_id))
        self.writer = self.manager.topic_writer(topic_name="/cmd_vel", topic_type=Twist_)

    def publish(self, motion):
        linear_x = 0.0
        angular_z = 0.0
        if motion == "w":
            linear_x = self.linear_speed
        elif motion == "s":
            linear_x = -self.linear_speed
        elif motion == "a":
            angular_z = self.angular_speed
        elif motion == "d":
            angular_z = -self.angular_speed
        self.writer.write(
            self.Twist(
                linear=self.Vector3(x=linear_x, y=0.0, z=0.0),
                angular=self.Vector3(x=0.0, y=0.0, z=angular_z),
            )
        )

    def stop(self):
        self.publish(None)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Control the AI Worker base/lift from a PCsensor pedal or system keyboard."
    )
    parser.add_argument(
        "--keyboard",
        action="store_true",
        help=(
            "Read W/A/S/D, O/P, and U from an auto-detected system keyboard "
            "instead of a PCsensor pedal."
        ),
    )
    parser.add_argument(
        "--device",
        action="append",
        default=[],
        help="Explicit /dev/input/eventN pedal keyboard; may be repeated.",
    )
    parser.add_argument("--domain-id", type=int, default=30)
    parser.add_argument("--linear-speed", type=float, default=0.4)
    parser.add_argument("--angular-speed", type=float, default=0.8)
    parser.add_argument("--publish-rate", type=float, default=30.0)
    parser.add_argument("--lift-speed", type=float, default=0.08, help="Lift speed in metres/second while O or P is held.")
    parser.add_argument("--estop-host", default="127.0.0.1", help="Host running teleop_hand_and_arm.py.")
    parser.add_argument("--estop-port", type=int, default=8765, help="UDP port used to latch upper-body E-stop.")
    parser.add_argument("--safety-host", default="127.0.0.1", help="Local address receiving teleop safety heartbeats.")
    parser.add_argument("--safety-port", type=int, default=8766, help="UDP port receiving teleop safety heartbeats.")
    parser.add_argument("--safety-timeout", type=float, default=0.5, help="Stop base/lift when the teleop heartbeat expires.")
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print auto-detected pedal keyboard devices and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and display pedal state without publishing DDS commands.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.keyboard and args.device:
        raise ValueError("--keyboard cannot be combined with --device")
    devices = (
        discover_system_keyboards()
        if args.keyboard
        else (args.device or discover_pedal_keyboards())
    )
    if args.list_devices:
        print("\n".join(devices))
        return 0
    if not devices:
        if args.keyboard:
            raise RuntimeError(
                "No system keyboard event devices were found. "
                "Use --device /dev/input/eventN to select one explicitly."
            )
        else:
            raise RuntimeError(
                "No PCsensor FootSwitch keyboard event devices were found. "
                "Use --keyboard for system keyboard control."
            )
    if not math.isfinite(args.publish_rate) or args.publish_rate <= 0.0:
        raise ValueError("--publish-rate must be finite and greater than zero")
    if not math.isfinite(args.lift_speed) or args.lift_speed < 0.0:
        raise ValueError("--lift-speed must be finite and zero or greater")
    if not math.isfinite(args.linear_speed) or args.linear_speed < 0.0:
        raise ValueError("--linear-speed must be finite and zero or greater")
    if not math.isfinite(args.angular_speed) or args.angular_speed < 0.0:
        raise ValueError("--angular-speed must be finite and zero or greater")
    if not math.isfinite(args.safety_timeout) or args.safety_timeout <= 0.0:
        raise ValueError("--safety-timeout must be finite and greater than zero")
    if args.domain_id < 0:
        raise ValueError("--domain-id must be zero or greater")

    # Both /cmd_vel and JointTrajectory must use the same DDS domain.  The
    # shared Robotis transport reads ROS_DOMAIN_ID when it is constructed.
    os.environ["ROS_DOMAIN_ID"] = str(args.domain_id)

    selector = selectors.DefaultSelector()
    buffers = {}
    for device in devices:
        try:
            fd = os.open(device, os.O_RDONLY | os.O_NONBLOCK)
        except PermissionError as exc:
            raise PermissionError(
                f"Cannot read {device}. Grant the current user read access to the pedal "
                "event devices or add the user to the input group."
            ) from exc
        selector.register(fd, selectors.EVENT_READ, device)
        buffers[fd] = bytearray()

    period = 1.0 / args.publish_rate
    publisher = None if args.dry_run else CmdVelPublisher(
        args.domain_id, args.linear_speed, args.angular_speed
    )
    lift = None
    if not args.dry_run:
        from teleop.robot_control.robotis_ai_worker_lift import AIWorkerLiftController
        lift = AIWorkerLiftController(command_duration=max(period * 2.0, 0.02))
        if not lift.wait_for_joint_state(timeout=5.0):
            lift.close()
            raise RuntimeError("No lift_joint was received on /joint_states within 5 seconds.")
    estop_notifier = EstopNotifier(args.estop_host, args.estop_port)
    safety = TeleopSafetyReceiver(args.safety_host, args.safety_port, args.safety_timeout)
    state = PedalState()
    aux = PedalAuxState()
    next_publish = time.monotonic()
    last_displayed = object()

    if args.keyboard:
        print(f"Keyboard devices: {', '.join(devices)}")
    else:
        print(f"Pedal devices: {', '.join(devices)}")
    print(
        "W=forward, S=backward, A=counter-clockwise, D=clockwise; "
        "O=lift up, P=lift down, U=toggle latched upper-body E-stop; release=stop. Ctrl+C exits."
    )
    if not args.dry_run:
        print(f"Publishing /cmd_vel on DDS domain {args.domain_id} at {args.publish_rate:.1f} Hz")

    try:
        while selector.get_map():
            now = time.monotonic()
            timeout = max(0.0, next_publish - now)
            for key, _ in selector.select(timeout):
                fd = key.fd
                device = key.data
                try:
                    data = os.read(fd, INPUT_EVENT.size * 64)
                except BlockingIOError:
                    continue
                if not data:
                    selector.unregister(fd)
                    os.close(fd)
                    buffers.pop(fd, None)
                    state.remove_device(device)
                    aux.remove_device(device)
                    continue
                buffer = buffers[fd]
                buffer.extend(data)
                while len(buffer) >= INPUT_EVENT.size:
                    packet = bytes(buffer[:INPUT_EVENT.size])
                    del buffer[:INPUT_EVENT.size]
                    _, _, event_type, code, value = INPUT_EVENT.unpack(packet)
                    if event_type == EV_KEY:
                        state.update(device, code, value)
                        if aux.update(device, code, value):
                            estop_notifier.publish(aux.estop)
                            print(f"upper_body_estop={'ON' if aux.estop else 'OFF'}", flush=True)

            tracking_allowed = safety.allowed
            motion_allowed = motion_is_allowed(aux.estop, tracking_allowed)
            motion = state.active_motion if motion_allowed else None
            lift_label = {1: "UP", -1: "DOWN", 0: "HOLD"}[aux.lift_direction]
            display_state = (
                state.active_motion,
                motion,
                lift_label,
                aux.estop,
                tracking_allowed,
            )
            if display_state != last_displayed:
                print(
                    f"held_wasd={state.active_motion or 'NONE'} "
                    f"motion={motion or 'STOP'} lift={lift_label} "
                    f"upper_body_estop={'ON' if aux.estop else 'OFF'} "
                    f"tracking_allow={tracking_allowed}",
                    flush=True,
                )
                last_displayed = display_state

            now = time.monotonic()
            if now >= next_publish:
                if publisher is not None:
                    publisher.publish(motion)
                    if not motion_allowed or aux.lift_direction == 0:
                        lift.hold()
                    else:
                        lift.nudge(aux.lift_direction, args.lift_speed * period)
                    estop_notifier.publish(aux.estop)
                next_publish = now + period
    except KeyboardInterrupt:
        pass
    finally:
        if publisher is not None:
            for _ in range(3):
                publisher.stop()
                time.sleep(0.02)
        if lift is not None:
            try:
                lift.hold()
            finally:
                lift.close()
        estop_notifier.close()
        safety.close()
        for key in list(selector.get_map().values()):
            selector.unregister(key.fd)
            os.close(key.fd)
        selector.close()
        print("Pedal teleop stopped.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
