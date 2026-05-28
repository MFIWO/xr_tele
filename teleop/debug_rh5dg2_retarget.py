# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Offline RH5DG2 retargeting debug helper.

This script validates the RH5DG2 YAML/URDF mapping and prints the computed
normalized hand commands for a synthetic or user-provided 25x3 landmark set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_control.robot_hand_RH5DG2 import (  # noqa: E402
    _RH5DG2Retargeting,
    _normalize_to_unit_interval,
)


def _load_hand(path: str | None, side: str) -> np.ndarray:
    if path is None:
        base = np.zeros((25, 3), dtype=np.float64)
        for idx in range(25):
            base[idx] = (idx * 0.015, (0.02 if side == "left" else -0.02) + idx * 0.002, 0.01 * (idx % 5))
        return base

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".npy":
        data = np.load(p)
    elif p.suffix.lower() == ".json":
        data = np.asarray(json.loads(p.read_text(encoding="utf-8")), dtype=np.float64)
    else:
        raise ValueError("hand input must be .npy or .json")

    data = np.asarray(data, dtype=np.float64)
    if data.shape == (75,):
        data = data.reshape(25, 3)
    if data.shape != (25, 3):
        raise ValueError(f"expected hand data shape (25, 3) or flat 75, got {data.shape}")
    return data


def _load_yaml_joint_names() -> dict[str, list[str]]:
    yaml_path = ROOT.parent / "assets" / "RH5DG2" / "RH5DG2.yml"
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    return {
        "left": list(cfg["left"]["target_joint_names"]),
        "right": list(cfg["right"]["target_joint_names"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug RH5DG2 retargeting")
    parser.add_argument("--left-hand-npy", type=str, default=None)
    parser.add_argument("--right-hand-npy", type=str, default=None)
    parser.add_argument("--left-hand-json", type=str, default=None)
    parser.add_argument("--right-hand-json", type=str, default=None)
    parser.add_argument("--dump-limits", action="store_true")
    args = parser.parse_args()

    yaml_joint_names = _load_yaml_joint_names()
    print("YAML joint names:")
    print(" left:", yaml_joint_names["left"])
    print(" right:", yaml_joint_names["right"])

    retarget = _RH5DG2Retargeting()
    print("\nRetargeting joint names:")
    print(" left:", retarget.left_joint_names)
    print(" right:", retarget.right_joint_names)
    print("\nHardware remap indices:")
    print(" left:", retarget.left_retargeting_to_hardware)
    print(" right:", retarget.right_retargeting_to_hardware)
    print("\nHuman landmark indices shape:")
    print(" left:", retarget.left_indices.shape)
    print(" right:", retarget.right_indices.shape)

    if args.dump_limits:
        print("\nLeft joint limits:")
        print(retarget.left_joint_limits)
        print("\nRight joint limits:")
        print(retarget.right_joint_limits)

    left_path = args.left_hand_npy or args.left_hand_json
    right_path = args.right_hand_npy or args.right_hand_json
    left = _load_hand(left_path, "left")
    right = _load_hand(right_path, "right")

    left_ref = left[retarget.left_indices[1, :]] - left[retarget.left_indices[0, :]]
    right_ref = right[retarget.right_indices[1, :]] - right[retarget.right_indices[0, :]]

    left_q = retarget.left_retargeting.retarget(left_ref)[retarget.left_retargeting_to_hardware]
    right_q = retarget.right_retargeting.retarget(right_ref)[retarget.right_retargeting_to_hardware]
    left_norm = _normalize_to_unit_interval(left_q, retarget.left_joint_limits)
    right_norm = _normalize_to_unit_interval(right_q, retarget.right_joint_limits)

    print("\nSynthetic / input hand data retarget result:")
    print(" left raw:", np.array2string(left_q, precision=4, suppress_small=True))
    print(" right raw:", np.array2string(right_q, precision=4, suppress_small=True))
    print(" left normalized:", np.array2string(left_norm, precision=4, suppress_small=True))
    print(" right normalized:", np.array2string(right_norm, precision=4, suppress_small=True))

    if np.allclose(left_norm, 0.5) and np.allclose(right_norm, 0.5):
        print("\nWARNING: retarget output is still neutral (0.5). Check the hand landmark input or YAML mapping.")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
