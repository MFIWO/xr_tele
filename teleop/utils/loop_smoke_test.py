"""Smoke-test the Config Loop streaming path WITHOUT a robot or cameras.

Runs the real LoopRobotStreamer + LoopCameraStreamer (same code the teleop loop
uses), feeding synthetic joint state and a moving test-pattern image. Use it to
verify, before touching the robot:

  1. The loop_porting_kit sidecar is running and Loop accepts the declared channels.
  2. The robot-state channels show up in the Loop UI.
  3. The camera decodes as a real image (NOT green noise) -- the #1 camera risk.

Run (with Config Loop already running on the same host):

    python -m teleop.utils.loop_smoke_test --loop-addr localhost:50051 --seconds 60

Then open the Loop UI and confirm the "camera-head" source shows a moving
colour-bar image and "robot-step" shows changing joints. You can also point VLC
at rtsp://127.0.0.1:8554/head to confirm the RTSP side independently of Loop.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from teleop.utils.loop_streamer import LoopCameraStreamer, LoopRobotStreamer


def _synthetic_camera_config(width: int, height: int, fps: float) -> dict:
    return {
        "head_camera": {"enable_zmq": True, "binocular": False,
                        "image_shape": [height, width], "fps": fps},
        "left_wrist_camera": {"enable_zmq": False},
        "right_wrist_camera": {"enable_zmq": False},
    }


def _test_pattern(width: int, height: int, phase: float) -> np.ndarray:
    """A moving colour-bar BGR frame so motion/decoding is obvious in the UI."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    bars = 8
    shift = int((phase * width) % width)
    for i in range(bars):
        x0 = (i * width) // bars
        x1 = ((i + 1) * width) // bars
        colour = [(i * 32) % 256, (i * 64) % 256, (i * 96) % 256]
        img[:, x0:x1] = colour
    img = np.roll(img, shift, axis=1)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop-addr", default="127.0.0.1:5590", help="Config Loop sidecar TCP host:port")
    parser.add_argument("--ee", default="dex3", help="EE type, to size the synthetic hand channels")
    parser.add_argument("--arm", default="H1_2", help="Arm type used to choose the Loop robot root key")
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--loop-source-key", default="robot-step",
                        help="robot source_key; match to the Cell Config robot source role")
    parser.add_argument("--loop-action-space", default="target_joint_position")
    parser.add_argument("--loop-robot-type", default=None, help="default: arm root key; '' to omit")
    parser.add_argument("--loop-gripper-type", default=None, help="default: derived from --ee; '' to omit")
    parser.add_argument("--loop-finger-type", default=None, help="default: derived from --ee; '' to omit")
    args = parser.parse_args()

    robot = LoopRobotStreamer(
        args.loop_addr,
        args.ee,
        args.frequency,
        arm=args.arm,
        source_key=args.loop_source_key,
        action_space=args.loop_action_space,
        robot_type=args.loop_robot_type,
        gripper_type=args.loop_gripper_type,
        finger_type=args.loop_finger_type,
    )
    robot.connect()
    camera = LoopCameraStreamer(args.loop_addr,
                                _synthetic_camera_config(args.width, args.height, args.frequency))
    camera.connect()
    print(f"[smoke] streaming synthetic data to {args.loop_addr} for {args.seconds:g}s; "
          f"check the Loop UI (or VLC rtsp://127.0.0.1:8554/head).")

    from teleop.utils.loop_streamer import ee_dim_per_hand
    ee_dim = ee_dim_per_hand(args.ee)
    period = 1.0 / args.frequency
    t0 = time.time()
    try:
        step = 0
        while time.time() - t0 < args.seconds:
            phase = step * 0.02
            arm_q = np.array([0.3 * math.sin(phase + i) for i in range(14)])
            arm_dq = np.zeros(14)
            sol_q = arm_q + 0.01
            ee_state = [0.5 + 0.5 * math.sin(phase + i) for i in range(2 * ee_dim)]
            ee_action = list(ee_state)
            robot.send(time.time_ns() // 1000, arm_q, arm_dq, sol_q, ee_state, ee_action)
            camera.set_head(_test_pattern(args.width, args.height, phase))
            step += 1
            time.sleep(period)
        print(f"[smoke] done. robot stats: {robot._sender.stats() if robot._sender else None}")
    finally:
        camera.close()
        robot.close()


if __name__ == "__main__":
    main()
