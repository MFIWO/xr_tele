#!/usr/bin/env python3
"""Drive the AI Worker base from PCsensor USB pedal key press/release events."""

import argparse
import os
from pathlib import Path
import selectors
import struct
import sys
import time


EV_KEY = 0x01
KEY_TO_MOTION = {
    17: "w",  # KEY_W
    31: "s",  # KEY_S
    30: "a",  # KEY_A
    32: "d",  # KEY_D
}
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
        description="Publish /cmd_vel only while a PCsensor foot pedal is physically held."
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
    devices = args.device or discover_pedal_keyboards()
    if args.list_devices:
        print("\n".join(devices))
        return 0
    if not devices:
        raise RuntimeError("No PCsensor FootSwitch keyboard event devices were found.")
    if args.publish_rate <= 0.0:
        raise ValueError("--publish-rate must be greater than zero")

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

    publisher = None if args.dry_run else CmdVelPublisher(
        args.domain_id, args.linear_speed, args.angular_speed
    )
    state = PedalState()
    period = 1.0 / args.publish_rate
    next_publish = time.monotonic()
    last_displayed = object()

    print(f"Pedal devices: {', '.join(devices)}")
    print(
        "W=forward, S=backward, A=counter-clockwise, D=clockwise; "
        "release=stop. Ctrl+C exits."
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
                    continue
                buffer = buffers[fd]
                buffer.extend(data)
                while len(buffer) >= INPUT_EVENT.size:
                    packet = bytes(buffer[:INPUT_EVENT.size])
                    del buffer[:INPUT_EVENT.size]
                    _, _, event_type, code, value = INPUT_EVENT.unpack(packet)
                    if event_type == EV_KEY:
                        state.update(device, code, value)

            motion = state.active_motion
            if motion != last_displayed:
                print(f"motion={motion or 'STOP'}", flush=True)
                last_displayed = motion

            now = time.monotonic()
            if publisher is not None and now >= next_publish:
                publisher.publish(motion)
                next_publish = now + period
    except KeyboardInterrupt:
        pass
    finally:
        if publisher is not None:
            for _ in range(3):
                publisher.stop()
                time.sleep(0.02)
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
