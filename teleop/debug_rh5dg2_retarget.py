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
    _apply_finger_open_calibration,
    _apply_safe_close_floor,
    _apply_teleop_close_gain,
    _fmt_finger_scores,
    _fmt_curl_scores,
    _normalize_to_unit_interval,
    _prepare_position_reference,
    _prepare_vector_reference,
    _postprocess_sim_command,
)


def _synthetic_hand(side: str, pose: str) -> np.ndarray:
    sign = 1.0 if side == "left" else -1.0
    data = np.zeros((25, 3), dtype=np.float64)
    finger_bases = {
        "thumb": (1, np.array([0.020, -0.025, sign * 0.025]), np.array([0.018, -0.025, sign * 0.030])),
        "index": (5, np.array([0.020, -0.045, sign * 0.012]), np.array([0.006, -0.035, sign * 0.004])),
        "middle": (10, np.array([0.000, -0.050, 0.000]), np.array([0.000, -0.040, 0.000])),
        "ring": (15, np.array([-0.018, -0.045, -sign * 0.012]), np.array([-0.005, -0.035, -sign * 0.004])),
        "little": (20, np.array([-0.034, -0.035, -sign * 0.025]), np.array([-0.010, -0.030, -sign * 0.006])),
    }
    for _, (start, open_step, close_step) in finger_bases.items():
        step = open_step if pose == "open" else close_step
        base = step * 0.6
        for offset in range(4):
            data[start + offset] = base + step * (offset + 1)
    return data


def _load_hand(path: str | None, side: str, synthetic: str) -> np.ndarray:
    if path is None:
        return _synthetic_hand(side, synthetic)

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
    parser.add_argument("--synthetic", choices=("open", "close"), default="open")
    parser.add_argument("--retarget-mode", choices=("config", "vector", "dexpilot"), default="config")
    parser.add_argument("--dump-limits", action="store_true")
    args = parser.parse_args()

    yaml_joint_names = _load_yaml_joint_names()
    print("YAML joint names:")
    print(" left:", yaml_joint_names["left"])
    print(" right:", yaml_joint_names["right"])

    retarget = _RH5DG2Retargeting(retarget_mode=args.retarget_mode)
    print("\nRetargeting joint names:")
    print(" left:", retarget.left_joint_names)
    print(" right:", retarget.right_joint_names)
    print("\nRetargeting mode:")
    print(" left:", retarget.left_retargeting_type)
    print(" right:", retarget.right_retargeting_type)
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
    left = _load_hand(left_path, "left", args.synthetic)
    right = _load_hand(right_path, "right", args.synthetic)

    left_indices = np.asarray(retarget.left_indices)
    right_indices = np.asarray(retarget.right_indices)
    if left_indices.ndim == 2:
        left_ref, *_ = _prepare_vector_reference(left, left_indices)
        right_ref, *_ = _prepare_vector_reference(right, right_indices)
    else:
        left_ref, *_ = _prepare_position_reference(left, left_indices)
        right_ref, *_ = _prepare_position_reference(right, right_indices)

    left_q = retarget.left_retargeting.retarget(left_ref)[retarget.left_retargeting_to_hardware]
    right_q = retarget.right_retargeting.retarget(right_ref)[retarget.right_retargeting_to_hardware]
    left_norm = _normalize_to_unit_interval(left_q, retarget.left_joint_limits)
    right_norm = _normalize_to_unit_interval(right_q, retarget.right_joint_limits)
    left_gain = _apply_teleop_close_gain(left_norm)
    right_gain = _apply_teleop_close_gain(right_norm)
    left_calibrated, left_scores = _apply_finger_open_calibration(left_gain, left)
    right_calibrated, right_scores = _apply_finger_open_calibration(right_gain, right)
    left_safe = _apply_safe_close_floor(left_calibrated)
    right_safe = _apply_safe_close_floor(right_calibrated)
    left_post, left_debug = _postprocess_sim_command(
        left_q, retarget.left_joint_limits, retarget.left_joint_names, left
    )
    right_post, right_debug = _postprocess_sim_command(
        right_q, retarget.right_joint_limits, retarget.right_joint_names, right
    )

    print("\nSynthetic / input hand data retarget result:")
    print(" left raw:", np.array2string(left_q, precision=4, suppress_small=True))
    print(" right raw:", np.array2string(right_q, precision=4, suppress_small=True))
    print(" left normalized:", np.array2string(left_norm, precision=4, suppress_small=True))
    print(" right normalized:", np.array2string(right_norm, precision=4, suppress_small=True))
    print(" left gain:", np.array2string(left_gain, precision=4, suppress_small=True))
    print(" right gain:", np.array2string(right_gain, precision=4, suppress_small=True))
    print(" left finger scores:", _fmt_finger_scores(left_scores))
    print(" right finger scores:", _fmt_finger_scores(right_scores))
    print(" left calibrated:", np.array2string(left_calibrated, precision=4, suppress_small=True))
    print(" right calibrated:", np.array2string(right_calibrated, precision=4, suppress_small=True))
    print(" left safe:", np.array2string(left_safe, precision=4, suppress_small=True))
    print(" right safe:", np.array2string(right_safe, precision=4, suppress_small=True))
    print(" left postprocess:", np.array2string(left_post, precision=4, suppress_small=True))
    print(" right postprocess:", np.array2string(right_post, precision=4, suppress_small=True))
    print(" left shape scores:", left_debug["prior_debug"]["scores"])
    print(" right shape scores:", right_debug["prior_debug"]["scores"])
    print(" left curl scores:", _fmt_curl_scores(left_debug["curl_debug"]["scores"]))
    print(" right curl scores:", _fmt_curl_scores(right_debug["curl_debug"]["scores"]))
    print(" left finger command delta:", left_debug["finger_command_delta"])
    print(" right finger command delta:", right_debug["finger_command_delta"])
    print(" left saturation:", left_debug["saturation"])
    print(" right saturation:", right_debug["saturation"])
    print(" left denorm rad:", np.array2string(left_debug["denorm_rad"], precision=4, suppress_small=True))
    print(" right denorm rad:", np.array2string(right_debug["denorm_rad"], precision=4, suppress_small=True))

    if np.allclose(left_norm, 0.5) and np.allclose(right_norm, 0.5):
        print("\nWARNING: retarget output is still neutral (0.5). Check the hand landmark input or YAML mapping.")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
