"""Kinematic playback of recorded H1-2 actions in MuJoCo.

Replays the arm (and approximately the hand) joint targets from a recorded episode
parquet (robot-h1_2.parquet) on the assets/h1_2/scene.xml model, so an episode's
actions can be inspected without the real robot. No physics: each frame writes qpos
directly and calls mj_forward, so the robot floats at the model's default base pose
and legs stay at zero.

Layout notes:
  * arm columns are the 7-dim URDF-order joint angles and map 1:1 onto the MJCF
    joints (same names as the URDF).
  * hand columns are RH5DG2 raw motor units (bigger = more open); the MJCF carries
    the Inspire hand, so fingers are driven by an *approximate* per-motor closure
    fraction mapped onto each Inspire joint range. Good enough to see grasps
    open/close, not a calibrated hand model. Disable with --no-hands.
  * head [40:42] has no counterpart joint in the MJCF (torso_joint only) and is
    ignored.

Usage (host, outside the docker container):
    ~/.venvs/mujoco/bin/python teleop/utils/h1_2_mujoco_action_viewer.py \
        --data test_data --source action --loop
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    sys.exit("mujoco is not importable; run with ~/.venvs/mujoco/bin/python")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_XML = REPO_ROOT / "assets" / "h1_2" / "scene.xml"

ARM_JOINTS_LEFT = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]
ARM_JOINTS_RIGHT = [j.replace("left_", "right_") for j in ARM_JOINTS_LEFT]

# RH5DG2 raw motor index -> the Inspire joints it should close, per hand side prefix.
# Raw units: RH5DG2_SAFE_ABS_MAX ~= open, RH5DG2_SAFE_ABS_MIN ~= closed (see
# teleop/robot_control/robot_hand_RH5DG2.py).
DG2_RAW_OPEN = np.array([1650, 1650, 1650, 200, 1650, 0, 1900, 1900, 1900, 1870, 1950, 1800, 2040], float)
DG2_RAW_CLOSED = np.array([1030, 1030, 1030, 0, 1030, -200, 1150, 1150, 1150, 1150, 940, 1250, 1550], float)
DG2_MOTOR_TO_JOINTS = {
    4: ["index_proximal_joint"],
    9: ["index_intermediate_joint"],
    2: ["middle_proximal_joint"],
    8: ["middle_intermediate_joint"],
    1: ["ring_proximal_joint"],
    7: ["ring_intermediate_joint"],
    0: ["pinky_proximal_joint"],
    6: ["pinky_intermediate_joint"],
    10: ["thumb_proximal_yaw_joint"],
    11: ["thumb_proximal_pitch_joint"],
    12: ["thumb_intermediate_joint", "thumb_distal_joint"],
}


def load_frames(data_path: Path, source: str):
    """-> (timestamps_s, left_arm (n,7), right_arm (n,7), left_hand (n,13), right_hand (n,13))"""
    if data_path.is_dir():
        candidates = sorted(data_path.rglob("robot-h1_2.parquet"))
        if not candidates:
            sys.exit(f"no robot-h1_2.parquet under {data_path}")
        data_path = candidates[0]
    import pyarrow.parquet as pq
    table = pq.read_table(str(data_path))
    cols = set(table.column_names)

    def col(group):
        name = f"h1_2.{source}.{group}.joint_position"
        if name not in cols:
            sys.exit(f"{data_path} has no column {name!r}; available: {sorted(cols)}")
        return np.asarray(table[name].to_pylist(), dtype=np.float64)

    ts = np.asarray(table["timestamp_us"].to_pylist(), dtype=np.float64) * 1e-6 \
        if "timestamp_us" in cols else None
    hands_group = ("left_ee", "right_ee")
    return data_path, ts, col("left_arm"), col("right_arm"), col(hands_group[0]), col(hands_group[1])


def hand_qpos_targets(model, raw13, prefix):
    """Approximate Inspire joint angles for one hand from 13 DG2 raw motor values."""
    closure = np.clip((DG2_RAW_OPEN - raw13) / (DG2_RAW_OPEN - DG2_RAW_CLOSED), 0.0, 1.0)
    out = []
    for motor, joints in DG2_MOTOR_TO_JOINTS.items():
        for jname in joints:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_{jname}")
            if jid < 0:
                continue
            lo, hi = model.jnt_range[jid]
            # open ~= the range end closest to zero, closed ~= the other end
            r_open, r_close = (lo, hi) if abs(lo) <= abs(hi) else (hi, lo)
            out.append((model.jnt_qposadr[jid], r_open + closure[motor] * (r_close - r_open)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "test_data",
                        help="episode dir or robot-h1_2.parquet path (default: test_data/)")
    parser.add_argument("--source", choices=("action", "observation"), default="action",
                        help="replay the commanded actions or the measured observations")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed factor")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="frame rate when the parquet has no usable timestamps")
    parser.add_argument("--loop", action="store_true", help="repeat the episode until the viewer closes")
    parser.add_argument("--no-hands", action="store_true", help="leave the fingers at the model default")
    args = parser.parse_args()

    path, ts, la, ra, lh, rh = load_frames(args.data, args.source)
    n = len(la)
    if ts is not None and n >= 2 and np.all(np.diff(ts) >= 0) and ts[-1] > ts[0]:
        dts = np.diff(ts, append=ts[-1] + np.median(np.diff(ts)))
    else:
        dts = np.full(n, 1.0 / max(args.fps, 1e-3))
    print(f"{path}: {n} frames of h1_2.{args.source}.*, {dts.sum():.1f}s at 1x "
          f"(playing at {args.speed:g}x)")

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    arm_adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
               for j in ARM_JOINTS_LEFT + ARM_JOINTS_RIGHT]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for i in range(n):
                if not viewer.is_running():
                    break
                frame_start = time.time()
                arm = np.concatenate((la[i], ra[i]))
                for adr, q in zip(arm_adr, arm):
                    data.qpos[adr] = q
                if not args.no_hands:
                    for adr, q in hand_qpos_targets(model, lh[i], "L") + \
                                  hand_qpos_targets(model, rh[i], "R"):
                        data.qpos[adr] = q
                mujoco.mj_forward(model, data)
                viewer.sync()
                if i % 30 == 0:
                    print(f"\rframe {i + 1}/{n}", end="", flush=True)
                time.sleep(max(0.0, dts[i] / max(args.speed, 1e-3) - (time.time() - frame_start)))
            print()
            if not args.loop:
                break
    print("done.")


if __name__ == "__main__":
    main()
