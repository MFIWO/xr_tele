"""Offline validation for the H1_2 + RH56F1 SIM retargeting path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_control.robot_hand_RH56F1 import RH56F1Retargeting


def _synthetic_hand(side, closed=False):
    sign = 1.0 if side == "left" else -1.0
    hand = np.zeros((25, 3), dtype=np.float64)
    starts = (1, 5, 10, 15, 20)
    for finger, start in enumerate(starts):
        base = np.array([0.025 - 0.014 * finger, -0.015, sign * (0.025 - 0.012 * finger)])
        for offset in range(4 if start == 1 else 5):
            if closed:
                hand[start + offset] = base + np.array([0.003 * offset, -0.012 * offset, 0.002 * offset])
            else:
                hand[start + offset] = base + np.array([0.0, -0.035 * offset, 0.0])
    return hand


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retarget-mode",
        choices=("config", "vector", "dexpilot"),
        default="config",
        help="RH56F1 retargeting mode. config follows assets/RH56F1/RH56F1.yml.",
    )
    args = parser.parse_args()

    retargeting = RH56F1Retargeting(retarget_mode=args.retarget_mode)
    print(
        "Retargeting mode:",
        retargeting.left_retargeting_type,
        retargeting.right_retargeting_type,
        "indices:",
        np.asarray(retargeting.left_indices).shape,
        np.asarray(retargeting.right_indices).shape,
    )
    for pose, closed in (("open", False), ("closed", True)):
        left_hand = _synthetic_hand("left", closed)
        right_hand = _synthetic_hand("right", closed)
        left, right = retargeting.retarget_abs(left_hand, right_hand)
        left_sim, right_sim = retargeting.retarget_sim(left_hand, right_hand)
        print(f"{pose} left abs :", np.round(left, 4).tolist())
        print(f"{pose} right abs:", np.round(right, 4).tolist())
        print(f"{pose} left sim :", np.round(left_sim, 4).tolist())
        print(f"{pose} right sim:", np.round(right_sim, 4).tolist())
        if left.shape != (6,) or right.shape != (6,):
            raise RuntimeError("RH56F1 command shape must be 6 actuators per hand")
        if left_sim.shape != (12,) or right_sim.shape != (12,):
            raise RuntimeError("RH56F1 sim command shape must be 12 joints per hand")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise RuntimeError("RH56F1 abs retargeting produced non-finite values")
        if not np.isfinite(left_sim).all() or not np.isfinite(right_sim).all():
            raise RuntimeError("RH56F1 sim retargeting produced non-finite values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
