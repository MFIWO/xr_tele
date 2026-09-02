#!/usr/bin/env python3
import argparse
import inspect
import importlib
import json
import math
import socket
import sys
import time


MANUS_HAPTIC_SIDES = ("left", "right")
MANUS_HAPTIC_FINGERS = 5
ZERO_HAPTIC_POWERS = [0.0] * MANUS_HAPTIC_FINGERS


def _normalize_manus_side(side):
    if isinstance(side, str) and side.lower() in MANUS_HAPTIC_SIDES:
        return side.lower()
    return None


def _haptic_topic_for_glove_topic(glove_topic):
    return f"{glove_topic.rstrip('/')}/vibration_cmd"


def _assign_haptic_route(routes, side, haptic_topic):
    """Assign a MANUS side to a topic while keeping the route one-to-one."""
    normalized_side = _normalize_manus_side(side)
    if normalized_side is None:
        return False

    changed = routes.get(normalized_side) != haptic_topic
    for routed_side, routed_topic in list(routes.items()):
        if routed_side != normalized_side and routed_topic == haptic_topic:
            del routes[routed_side]
            changed = True
    routes[normalized_side] = haptic_topic
    return changed


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
    except Exception as get_message_error:
        pkg, _, name = normalized.split("/")
        try:
            module = importlib.import_module(f"{pkg}.msg")
            return getattr(module, name)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"Cannot import {normalized}. Source the ROS installation and the "
                "MANUS workspace that built manus_ros2_msgs before starting this bridge. "
                "For this project that is typically:\n"
                "  source /opt/ros/jazzy/setup.bash\n"
                "  source /workspace/tmp/manus_ws/install/setup.bash"
            ) from get_message_error


def _message_source_path(msg_type):
    try:
        return inspect.getfile(msg_type)
    except Exception:
        return "<unknown>"


def _check_msg_type_support(msg_type, requested_type):
    try:
        msg_type.__class__.__import_type_support__()
    except Exception as exc:
        source_path = _message_source_path(msg_type)
        print(
            "\n[manus_ros2_to_udp] Cannot load ROS2 type support for "
            f"{requested_type}.\n"
            "This usually means the Manus message package was built for a "
            "different ROS/Python environment.\n\n"
            f"Current Python: {sys.version.split()[0]}\n"
            f"Message module: {source_path}\n"
            f"Original error: {exc}\n\n"
            "Fix one of these before running this bridge:\n"
            "  1. Run tools/manus_ros2_to_udp.py in the ROS environment that "
            "built manus_ros2_msgs, then send UDP to this container.\n"
            "  2. Rebuild manus_ros2_msgs inside this container with the same "
            "ROS2/Python version, then source that install/setup.bash.\n\n"
            "If the module path contains python3.12 but Current Python is 3.10, "
            "do not source that install space from this container.\n",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc


def _point_to_dict(point):
    return {
        "x": float(getattr(point, "x")),
        "y": float(getattr(point, "y")),
        "z": float(getattr(point, "z")),
    }


def _quat_to_dict(quat):
    return {
        "x": float(getattr(quat, "x")),
        "y": float(getattr(quat, "y")),
        "z": float(getattr(quat, "z")),
        "w": float(getattr(quat, "w")),
    }


def _msg_to_packet(msg):
    side = getattr(msg, "side", None)
    raw_nodes = []
    for node in getattr(msg, "raw_nodes", []) or []:
        node_id = int(getattr(node, "node_id"))
        pose = getattr(node, "pose", None)
        if pose is None:
            continue
        position = getattr(pose, "position", None)
        orientation = getattr(pose, "orientation", None)
        if position is None:
            continue
        item = {
            "node_id": node_id,
            "pose": {
                "position": _point_to_dict(position),
            },
        }
        if orientation is not None:
            item["pose"]["orientation"] = _quat_to_dict(orientation)
        raw_nodes.append(item)
    return {
        "side": side,
        "raw_nodes": raw_nodes,
        "timestamp": time.time(),
    }


def _parse_haptic_packet(packet):
    if not isinstance(packet, dict) or packet.get("type") != "manus_haptics":
        raise ValueError("packet type must be 'manus_haptics'")
    parsed = {}
    for side in MANUS_HAPTIC_SIDES:
        values = packet.get(side)
        if not isinstance(values, list) or len(values) != MANUS_HAPTIC_FINGERS:
            raise ValueError(f"{side} must contain {MANUS_HAPTIC_FINGERS} intensities")
        powers = []
        for value in values:
            power = float(value)
            if not math.isfinite(power):
                raise ValueError(f"{side} intensities must be finite")
            powers.append(max(0.0, min(1.0, power)))
        parsed[side] = powers
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", nargs="+", default=["manus_glove_0", "manus_glove_1"])
    parser.add_argument("--msg-type", default="manus_ros2_msgs/msg/ManusGlove")
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=56120)
    parser.add_argument(
        "--haptic-only",
        action="store_true",
        help="Run only the UDP-to-MANUS vibration bridge; do not subscribe to ManusGlove topics.",
    )
    parser.add_argument(
        "--haptic-listen-host",
        default="0.0.0.0",
        help="UDP bind host for haptic commands.",
    )
    parser.add_argument(
        "--haptic-listen-port",
        type=int,
        default=56121,
        help="UDP port for haptic commands from teleop. Set 0 to disable.",
    )
    parser.add_argument(
        "--haptic-msg-type",
        default="manus_ros2_msgs/msg/ManusVibrationCommand",
    )
    parser.add_argument(
        "--left-haptic-topic",
        default="manus_glove_1/vibration_cmd",
        help="Left vibration topic used only by --haptic-only (manual fallback).",
    )
    parser.add_argument(
        "--right-haptic-topic",
        default="manus_glove_0/vibration_cmd",
        help="Right vibration topic used only by --haptic-only (manual fallback).",
    )
    parser.add_argument(
        "--haptic-timeout",
        type=float,
        default=0.3,
        help="Seconds without a haptic UDP heartbeat before publishing zero.",
    )
    args = parser.parse_args()
    if not 0 <= args.haptic_listen_port <= 65535:
        raise ValueError("--haptic-listen-port must be between 0 and 65535.")
    if args.haptic_only and args.haptic_listen_port == 0:
        raise ValueError("--haptic-only requires --haptic-listen-port greater than zero.")
    if args.haptic_listen_port > 0 and args.haptic_timeout <= 0.0:
        raise ValueError("--haptic-timeout must be greater than zero.")

    # Auto routing learns glove side from ManusGlove.side. Haptic-only mode has
    # no glove messages to learn from, so it intentionally retains the manual
    # topic arguments as a compatibility fallback.
    auto_haptic_routing = not args.haptic_only

    import rclpy

    msg_type = None
    sock = None
    if not args.haptic_only:
        msg_type = _load_ros_msg_type(args.msg_type)
        _check_msg_type_support(msg_type, args.msg_type)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    haptic_sock = None
    haptic_publishers = {}
    haptic_routes = {}
    haptic_state = None
    haptic_timer = None
    rclpy.init(args=None)
    node_name = "manus_haptic_udp_bridge" if args.haptic_only else "manus_ros2_to_udp"
    node = rclpy.create_node(node_name)

    def stop_all_haptic_topics():
        if not haptic_publishers:
            return
        for publisher in haptic_publishers.values():
            msg = haptic_msg_type()
            msg.intensities = list(ZERO_HAPTIC_POWERS)
            publisher.publish(msg)

    def callback(msg, topic):
        packet = _msg_to_packet(msg)
        raw = json.dumps(packet, separators=(",", ":")).encode("utf-8")
        sock.sendto(raw, (args.udp_host, args.udp_port))

        if auto_haptic_routing and haptic_publishers:
            side = _normalize_manus_side(getattr(msg, "side", None))
            if side is not None:
                haptic_topic = _haptic_topic_for_glove_topic(topic)
                if _assign_haptic_route(haptic_routes, side, haptic_topic):
                    # Clear every output before applying a new or changed route,
                    # so a formerly assigned glove cannot keep vibrating.
                    stop_all_haptic_topics()
                    if haptic_state is not None:
                        haptic_state["last_published"] = {
                            route_side: list(ZERO_HAPTIC_POWERS)
                            for route_side in MANUS_HAPTIC_SIDES
                        }
                    node.get_logger().info(
                        f"Learned MANUS haptic route: {side} -> {haptic_topic}"
                    )

    subs = []
    if not args.haptic_only:
        subs = [
            node.create_subscription(
                msg_type,
                topic,
                lambda msg, topic=topic: callback(msg, topic),
                5,
            )
            for topic in args.topics
        ]
        node.get_logger().info(
            f"Forwarding {args.topics} ({args.msg_type}) to udp://{args.udp_host}:{args.udp_port}"
        )
    else:
        node.get_logger().info("Haptic-only mode: ManusGlove subscriptions are disabled.")

    def publish_haptics(powers_by_side):
        published_sides = set()
        for side in MANUS_HAPTIC_SIDES:
            haptic_topic = haptic_routes.get(side)
            publisher = haptic_publishers.get(haptic_topic)
            if publisher is None:
                continue
            msg = haptic_msg_type()
            msg.intensities = list(powers_by_side.get(side, ZERO_HAPTIC_POWERS))
            publisher.publish(msg)
            published_sides.add(side)
        return published_sides

    if args.haptic_listen_port > 0:
        haptic_msg_type = _load_ros_msg_type(args.haptic_msg_type)
        _check_msg_type_support(haptic_msg_type, args.haptic_msg_type)
        if auto_haptic_routing:
            haptic_topics = [_haptic_topic_for_glove_topic(topic) for topic in args.topics]
        else:
            haptic_topics = [args.left_haptic_topic, args.right_haptic_topic]
            _assign_haptic_route(haptic_routes, "left", args.left_haptic_topic)
            _assign_haptic_route(haptic_routes, "right", args.right_haptic_topic)
        haptic_publishers = {
            topic: node.create_publisher(haptic_msg_type, topic, 5)
            for topic in dict.fromkeys(haptic_topics)
        }
        haptic_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        haptic_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        haptic_sock.bind((args.haptic_listen_host, args.haptic_listen_port))
        haptic_sock.setblocking(False)
        haptic_state = {
            "last_rx": 0.0,
            "zeroed": True,
            "last_warning": 0.0,
            "last_route_warning": 0.0,
            "last_published": {
                side: list(ZERO_HAPTIC_POWERS)
                for side in MANUS_HAPTIC_SIDES
            },
        }

        def haptic_values_changed(powers_by_side, epsilon=1e-4):
            return any(
                abs(
                    powers_by_side[side][finger_index]
                    - haptic_state["last_published"][side][finger_index]
                ) > epsilon
                for side in MANUS_HAPTIC_SIDES
                if side in haptic_routes
                for finger_index in range(MANUS_HAPTIC_FINGERS)
            )

        def poll_haptics():
            received = None
            while True:
                try:
                    raw, _ = haptic_sock.recvfrom(4096)
                except BlockingIOError:
                    break
                except OSError:
                    return
                try:
                    packet = json.loads(raw.decode("utf-8"))
                    received = _parse_haptic_packet(packet)
                except Exception as exc:
                    now = time.monotonic()
                    if now - haptic_state["last_warning"] >= 2.0:
                        node.get_logger().warning(f"Ignoring malformed haptic UDP packet: {exc}")
                        haptic_state["last_warning"] = now

            now = time.monotonic()
            if received is not None:
                haptic_state["last_rx"] = now
                haptic_state["zeroed"] = all(
                    all(power == 0.0 for power in received[side])
                    for side in MANUS_HAPTIC_SIDES
                )
                missing_routes = [
                    side
                    for side in MANUS_HAPTIC_SIDES
                    if side not in haptic_routes and any(received[side])
                ]
                if (
                    missing_routes
                    and now - haptic_state["last_route_warning"] >= 2.0
                ):
                    node.get_logger().warning(
                        "Waiting for ManusGlove.side before routing haptics for: "
                        + ", ".join(missing_routes)
                    )
                    haptic_state["last_route_warning"] = now
                if haptic_values_changed(received):
                    published_sides = publish_haptics(received)
                    for side in published_sides:
                        haptic_state["last_published"][side] = list(received[side])
            elif (
                haptic_state["last_rx"] > 0.0
                and not haptic_state["zeroed"]
                and now - haptic_state["last_rx"] > args.haptic_timeout
            ):
                publish_haptics({
                    side: list(ZERO_HAPTIC_POWERS)
                    for side in MANUS_HAPTIC_SIDES
                })
                haptic_state["zeroed"] = True
                haptic_state["last_published"] = {
                    side: list(ZERO_HAPTIC_POWERS)
                    for side in MANUS_HAPTIC_SIDES
                }
                node.get_logger().warning(
                    f"Haptic UDP timeout ({args.haptic_timeout:.3f}s); vibration stopped."
                )

        haptic_timer = node.create_timer(0.005, poll_haptics)
        stop_all_haptic_topics()
        routing_description = (
            f"auto from ManusGlove.side; candidates={haptic_topics}"
            if auto_haptic_routing
            else f"manual left={args.left_haptic_topic} right={args.right_haptic_topic}"
        )
        node.get_logger().info(
            "Haptic bridge listening on "
            f"udp://{args.haptic_listen_host}:{args.haptic_listen_port}; "
            f"routing={routing_description} "
            f"timeout={args.haptic_timeout:.3f}s"
        )

    try:
        rclpy.spin(node)
    finally:
        if haptic_publishers:
            try:
                publish_haptics({
                    side: list(ZERO_HAPTIC_POWERS)
                    for side in MANUS_HAPTIC_SIDES
                })
                rclpy.spin_once(node, timeout_sec=0.05)
            except Exception as exc:
                node.get_logger().warning(f"Could not publish final haptic stop command: {exc}")
        if haptic_timer is not None:
            node.destroy_timer(haptic_timer)
        if haptic_sock is not None:
            haptic_sock.close()
        for sub in subs:
            node.destroy_subscription(sub)
        node.destroy_node()
        rclpy.shutdown()
        if sock is not None:
            sock.close()


if __name__ == "__main__":
    main()
