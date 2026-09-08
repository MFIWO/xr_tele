#!/usr/bin/env python3
"""Forward ROBOTIS WebXR ROS 2 topics to xr_tele over a local Unix socket.

This helper intentionally uses only the Python standard library and ROS 2
packages so ROS 2 Jazzy's system Python can run it independently of xr_tele's
Python 3.10 environment.
"""

from __future__ import annotations

import argparse
from multiprocessing.connection import Listener
import os
import signal


STATE_TOPICS = {
    "leader_left": "/leader/joint_trajectory_command_broadcaster_left/joint_trajectory",
    "leader_right": "/leader/joint_trajectory_command_broadcaster_right/joint_trajectory",
    "leader_head": "/leader/joystick_controller_left/joint_trajectory",
    "leader_lift": "/leader/joystick_controller_right/joint_trajectory",
    "leader_mobile": "/cmd_vel",
    "follower": "/joint_states",
    "follower_mobile": "/odom",
}

CAMERA_TOPICS = {
    "head": "/zed/zed_node/left/image_rect_color/compressed",
    "left_wrist": "/camera_left/camera_left/color/image_rect_raw/compressed",
    "right_wrist": "/camera_right/camera_right/color/image_rect_raw/compressed",
}


def _stamp_ns(message):
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    if stamp is None:
        return None
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _trajectory_event(source, message):
    if not message.points:
        return None
    return {
        "kind": "trajectory",
        "source": source,
        "joint_names": list(message.joint_names),
        "positions": list(message.points[0].positions),
    }


def _joint_state_event(message):
    return {
        "kind": "joint_state",
        "source": "follower",
        "name": list(message.name),
        "position": list(message.position),
        "velocity": list(message.velocity),
    }


def _twist_event(message):
    return {
        "kind": "twist",
        "source": "leader_mobile",
        "linear": [float(message.linear.x), float(message.linear.y)],
        "angular_z": float(message.angular.z),
    }


def _odometry_event(message):
    twist = message.twist.twist
    return {
        "kind": "odometry",
        "source": "follower_mobile",
        "linear": [float(twist.linear.x), float(twist.linear.y)],
        "angular_z": float(twist.angular.z),
    }


def _image_event(name, message):
    return {
        "kind": "image",
        "name": name,
        "data": bytes(message.data),
        "source_time_ns": _stamp_ns(message),
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--domain-id", type=int, default=30)
    return parser.parse_args()


def main():
    args = _parse_args()
    authkey_hex = os.environ.get("XR_TELE_ROS_BRIDGE_AUTHKEY", "")
    if not authkey_hex:
        raise RuntimeError("XR_TELE_ROS_BRIDGE_AUTHKEY is required")

    listener = Listener(
        args.socket,
        family="AF_UNIX",
        authkey=bytes.fromhex(authkey_hex),
    )
    connection = listener.accept()
    listener.close()

    stop = False

    def request_stop(*_args):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    try:
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from sensor_msgs.msg import CompressedImage, JointState
        from trajectory_msgs.msg import JointTrajectory

        rclpy.init(args=None, domain_id=args.domain_id)
        node = rclpy.create_node("xr_tele_robotis_webxr_bridge")
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        def send_event(factory, *factory_args):
            def callback(message):
                nonlocal stop
                if stop:
                    return
                try:
                    event = factory(*factory_args, message)
                    if event is not None:
                        connection.send(event)
                except (BrokenPipeError, EOFError, OSError):
                    stop = True

            return callback

        for source in ("leader_left", "leader_right", "leader_head", "leader_lift"):
            node.create_subscription(
                JointTrajectory,
                STATE_TOPICS[source],
                send_event(_trajectory_event, source),
                qos,
            )
        node.create_subscription(
            Twist,
            STATE_TOPICS["leader_mobile"],
            send_event(_twist_event),
            qos,
        )
        node.create_subscription(
            JointState,
            STATE_TOPICS["follower"],
            send_event(_joint_state_event),
            qos,
        )
        node.create_subscription(
            Odometry,
            STATE_TOPICS["follower_mobile"],
            send_event(_odometry_event),
            qos,
        )
        for name, topic in CAMERA_TOPICS.items():
            node.create_subscription(
                CompressedImage,
                topic,
                send_event(_image_event, name),
                qos,
            )

        connection.send(
            {
                "kind": "bridge_ready",
                "rmw": os.environ.get("RMW_IMPLEMENTATION", "process default"),
                "domain_id": args.domain_id,
            }
        )

        while not stop:
            if connection.poll():
                command = connection.recv()
                if command == "close":
                    break
            executor.spin_once(timeout_sec=0.05)
    except Exception as exc:
        try:
            connection.send({"kind": "bridge_error", "message": repr(exc)})
        except Exception:
            pass
        raise
    finally:
        try:
            executor.remove_node(node)
            executor.shutdown(timeout_sec=0.5)
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
        connection.close()


if __name__ == "__main__":
    main()
