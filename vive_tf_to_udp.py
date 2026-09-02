#!/usr/bin/env python3
import argparse
import json
import math
import socket
import time


def _safe_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _vec3_to_dict(value):
    return {
        "x": _safe_float(value.x),
        "y": _safe_float(value.y),
        "z": _safe_float(value.z),
    }


def _quat_to_dict(value):
    return {
        "x": _safe_float(value.x),
        "y": _safe_float(value.y),
        "z": _safe_float(value.z),
        "w": _safe_float(value.w),
    }


def _transform_to_pose(transform):
    return {
        "position": _vec3_to_dict(transform.transform.translation),
        "orientation": _quat_to_dict(transform.transform.rotation),
    }


def _stamp_to_float(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _frame_id(value):
    return str(value).strip().lstrip("/")


def _pose_is_finite(transform):
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    return all(
        math.isfinite(float(value))
        for value in (
            translation.x,
            translation.y,
            translation.z,
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        )
    )


def _transform_to_dict(transform):
    return {
        "parent_frame": str(transform.header.frame_id),
        "child_frame": str(transform.child_frame_id),
        "stamp": _stamp_to_float(transform.header.stamp),
        "valid": _pose_is_finite(transform),
        "pose": _transform_to_pose(transform),
    }


def _tracker_specs(args):
    specs = {
        "left_tracker": args.left_tracker_name,
        "right_tracker": args.right_tracker_name,
        "head_tracker": args.head_tracker_name,
    }
    return {key: name for key, name in specs.items() if name}


def _tracker_aliases_from_transforms(transforms, trackers):
    aliases = {}
    key_by_child = {_frame_id(name): key for key, name in trackers.items()}

    for transform in transforms:
        key = key_by_child.get(_frame_id(transform.child_frame_id))
        if key is None:
            continue
        tracker_name = trackers[key]
        if _pose_is_finite(transform):
            aliases[key] = {
                "name": tracker_name,
                "ok": True,
                "parent_frame": str(transform.header.frame_id),
                "pose": _transform_to_pose(transform),
            }
        else:
            aliases[key] = {
                "name": tracker_name,
                "ok": False,
                "parent_frame": str(transform.header.frame_id),
                "error": "non-finite transform",
            }

    return aliases


def _packet_from_tf_message(msg, topic, trackers, args):
    packet = {
        "source": "vive_tf_to_udp",
        "topic": topic,
        "tracking_frame": args.tracking_frame,
        "timestamp": time.time(),
        "transforms": [_transform_to_dict(transform) for transform in msg.transforms],
        "trackers": _tracker_aliases_from_transforms(msg.transforms, trackers),
    }
    return packet

def main():
    parser = argparse.ArgumentParser(description="Forward ROS2 TF messages to UDP JSON.")
    parser.add_argument("--udp-host", default="127.0.0.1", help="Destination host or container IP")
    parser.add_argument("--udp-port", type=int, default=56130, help="Destination UDP port")
    parser.add_argument("--tracking-frame", default="libsurvive_world", help="TF parent frame")
    parser.add_argument("--left-tracker-name", default="LHR-36B992CB")
    parser.add_argument("--right-tracker-name", default="LHR-0FAD1369")
    parser.add_argument("--head-tracker-name", default="LHR-1F621773")
    parser.add_argument("--tf-topic", default="/tf", help="Dynamic TF topic to forward")
    parser.add_argument("--tf-static-topic", default="/tf_static", help="Static TF topic to forward")
    parser.add_argument("--no-tf-static", action="store_true", help="Do not forward /tf_static")
    parser.add_argument("--max-packet-bytes", type=int, default=65000, help="Drop UDP packets larger than this")
    parser.add_argument("--rate", type=float, default=None, help="Deprecated; TF messages are forwarded as they arrive")
    parser.add_argument("--warn-period", type=float, default=2.0, help="Seconds between repeated lookup warnings")
    args = parser.parse_args()

    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from tf2_msgs.msg import TFMessage

    trackers = _tracker_specs(args)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rclpy.init(args=None)
    node = rclpy.create_node("vive_tf_to_udp")
    dropped = {"count": 0, "last_warn": 0.0}

    def send_packet(packet):
        raw = json.dumps(packet, separators=(",", ":")).encode("utf-8")
        if len(raw) > args.max_packet_bytes:
            dropped["count"] += 1
            now = time.monotonic()
            if now - dropped["last_warn"] >= args.warn_period:
                node.get_logger().warning(
                    f"Dropping oversized TF UDP packet bytes={len(raw)} "
                    f"limit={args.max_packet_bytes} dropped={dropped['count']}"
                )
                dropped["last_warn"] = now
            return
        sock.sendto(raw, (args.udp_host, args.udp_port))

    def make_callback(topic):
        def callback(msg):
            packet = _packet_from_tf_message(msg, topic, trackers, args)
            send_packet(packet)

        return callback

    tf_sub = node.create_subscription(TFMessage, args.tf_topic, make_callback(args.tf_topic), 100)
    subs = [tf_sub]
    if not args.no_tf_static:
        static_qos = QoSProfile(depth=100)
        static_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        static_qos.reliability = ReliabilityPolicy.RELIABLE
        subs.append(
            node.create_subscription(
                TFMessage,
                args.tf_static_topic,
                make_callback(args.tf_static_topic),
                static_qos,
            )
        )

    node.get_logger().info(
        f"Forwarding all TF messages from {args.tf_topic}"
        f"{'' if args.no_tf_static else f' and {args.tf_static_topic}'} "
        f"to udp://{args.udp_host}:{args.udp_port}; tracker aliases={trackers}"
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for sub in subs:
            node.destroy_subscription(sub)
        node.destroy_node()
        rclpy.shutdown()
        sock.close()


if __name__ == "__main__":
    main()
