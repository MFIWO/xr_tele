#!/usr/bin/env python3
"""Publish a standalone MANUS finger-vibration command for hardware testing."""

import argparse
import importlib
import time


FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")
ZERO_POWERS = [0.0] * len(FINGER_ORDER)


def _load_ros_msg_type(msg_type):
    normalized = msg_type
    if "/msg/" not in normalized:
        if "." in normalized:
            parts = normalized.split(".")
            if len(parts) >= 3 and parts[-2] == "msg":
                normalized = f"{parts[0]}/msg/{parts[-1]}"
        elif "/" in normalized:
            package, name = normalized.split("/", 1)
            normalized = f"{package}/msg/{name}"
    package, _, name = normalized.split("/")
    try:
        from rosidl_runtime_py.utilities import get_message

        return get_message(normalized)
    except Exception:
        module = importlib.import_module(f"{package}.msg")
        return getattr(module, name)


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


def main():
    parser = argparse.ArgumentParser(
        description="Publish only MANUS vibration_cmd topics without starting teleoperation."
    )
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
        help="Explicit five-finger powers in MANUS order.",
    )
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--discovery-seconds", type=float, default=1.0)
    parser.add_argument(
        "--msg-type",
        default="manus_ros2_msgs/msg/ManusVibrationCommand",
    )
    parser.add_argument(
        "--left-topic",
        default="manus_glove_1/vibration_cmd",
    )
    parser.add_argument(
        "--right-topic",
        default="manus_glove_0/vibration_cmd",
    )
    args = parser.parse_args()

    if args.duration <= 0.0:
        raise ValueError("--duration must be greater than zero.")
    if args.hz <= 0.0:
        raise ValueError("--hz must be greater than zero.")
    if args.discovery_seconds < 0.0:
        raise ValueError("--discovery-seconds must be zero or greater.")
    powers = _build_powers(args.finger, args.intensity, args.powers)

    import rclpy

    msg_type = _load_ros_msg_type(args.msg_type)
    rclpy.init(args=None)
    node = rclpy.create_node("test_manus_haptic_topic")
    topics = {
        "left": args.left_topic,
        "right": args.right_topic,
    }
    selected_sides = ("left", "right") if args.side == "both" else (args.side,)
    publishers = {
        side: node.create_publisher(msg_type, topics[side], 5)
        for side in selected_sides
    }

    def publish(values):
        for publisher in publishers.values():
            msg = msg_type()
            msg.intensities = list(values)
            publisher.publish(msg)

    try:
        discovery_end = time.monotonic() + args.discovery_seconds
        while time.monotonic() < discovery_end:
            rclpy.spin_once(node, timeout_sec=0.05)

        for side in selected_sides:
            subscriptions = node.count_subscribers(topics[side])
            node.get_logger().info(
                f"side={side} topic={topics[side]} subscribers={subscriptions}"
            )
            if subscriptions == 0:
                node.get_logger().warning(
                    f"No subscriber found for {topics[side]}; check the glove index/topic."
                )

        node.get_logger().info(
            f"Publishing side={args.side} powers={powers} for {args.duration:.3f}s"
        )
        period = 1.0 / args.hz
        end_time = time.monotonic() + args.duration
        while time.monotonic() < end_time:
            publish(powers)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(period)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted; stopping vibration.")
    finally:
        # MANUS keeps the last vibration power until a new command changes it.
        # Repeat zero briefly so DDS discovery/timing does not leave vibration on.
        for _ in range(5):
            publish(ZERO_POWERS)
            rclpy.spin_once(node, timeout_sec=0.02)
            time.sleep(0.02)
        node.get_logger().info("Published zero powers; vibration stopped.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
