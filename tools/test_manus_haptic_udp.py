#!/usr/bin/env python3
"""Send standalone MANUS haptic test commands to the UDP-to-ROS bridge."""

import argparse
import json
import os
import socket
import time


FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")
ZERO_POWERS = [0.0] * len(FINGER_ORDER)


def _build_powers(finger, intensity, explicit_powers):
    if explicit_powers is not None:
        powers = [float(value) for value in explicit_powers]
    elif finger == "all":
        powers = [float(intensity)] * len(FINGER_ORDER)
    else:
        powers = list(ZERO_POWERS)
        powers[FINGER_ORDER.index(finger)] = float(intensity)
    if any(not 0.0 <= value <= 1.0 for value in powers):
        raise ValueError("All vibration powers must be between 0 and 1.")
    return powers


def _make_packet(sequence, left, right):
    return {
        "source": "test_manus_haptic_udp",
        "type": "manus_haptics",
        "timestamp": time.time(),
        "sequence": int(sequence),
        "left": list(left),
        "right": list(right),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Send MANUS finger vibration commands using UDP only."
    )
    parser.add_argument(
        "--host",
        default=os.getenv("MANUS_HAPTIC_HOST", "192.168.123.54"),
        help="Host running manus_ros2_to_udp.py. Defaults to MANUS_HAPTIC_HOST or 192.168.123.54.",
    )
    parser.add_argument("--port", type=int, default=56121)
    parser.add_argument("--side", choices=["left", "right", "both"], default="right")
    parser.add_argument(
        "--finger",
        choices=["all", *FINGER_ORDER],
        default="all",
        help="Finger to vibrate. Ignored when --powers is provided.",
    )
    parser.add_argument("--intensity", type=float, default=0.3)
    parser.add_argument(
        "--powers",
        type=float,
        nargs=5,
        metavar=("THUMB", "INDEX", "MIDDLE", "RING", "LITTLE"),
    )
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--hz", type=float, default=20.0)
    args = parser.parse_args()

    if not args.host:
        raise ValueError("--host must not be empty.")
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535.")
    if args.duration <= 0.0:
        raise ValueError("--duration must be greater than zero.")
    if args.hz <= 0.0:
        raise ValueError("--hz must be greater than zero.")

    selected = _build_powers(args.finger, args.intensity, args.powers)
    left = selected if args.side in ("left", "both") else ZERO_POWERS
    right = selected if args.side in ("right", "both") else ZERO_POWERS
    target = (args.host, args.port)
    period = 1.0 / args.hz
    sequence = 0
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    last_send_error = None

    def send(left_powers, right_powers):
        nonlocal sequence, last_send_error
        packet = _make_packet(sequence, left_powers, right_powers)
        raw = json.dumps(packet, separators=(",", ":")).encode("utf-8")
        try:
            sock.sendto(raw, target)
        except OSError as exc:
            if str(exc) != last_send_error:
                print(f"UDP send failed: {exc}")
                last_send_error = str(exc)
            return False
        sequence += 1
        last_send_error = None
        return True

    try:
        print(
            f"Sending UDP -> {args.host}:{args.port} "
            f"side={args.side} left={left} right={right}"
        )
        end_time = time.monotonic() + args.duration
        while time.monotonic() < end_time:
            if not send(left, right):
                break
            time.sleep(period)
    except KeyboardInterrupt:
        print("Interrupted; stopping vibration.")
    finally:
        zero_packets_sent = 0
        for _ in range(5):
            if send(ZERO_POWERS, ZERO_POWERS):
                zero_packets_sent += 1
            time.sleep(0.02)
        sock.close()
        if zero_packets_sent:
            print(f"Sent {zero_packets_sent} zero packets; vibration stop requested.")
        else:
            print("Could not send zero packets; verify that the UDP bridge is running.")


if __name__ == "__main__":
    main()
