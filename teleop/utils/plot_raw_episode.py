#!/usr/bin/env python3
"""Plot raw xr_teleoperate EpisodeWriter JSON values.

Examples:
  python utils/plot_raw_episode.py --episode "utils/data/pick cube/episode_0013"
  python utils/plot_raw_episode.py --json "utils/data/pick cube/episode_0013/data.json"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "numpy is required for plotting. Install it in the active environment with: "
        "pip install numpy matplotlib"
    ) from exc

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it in the active environment with: "
        "pip install numpy matplotlib"
    ) from exc


QPOS_PARTS = ("left_arm", "right_arm", "left_ee", "right_ee", "body")
TACTILE_SIDES = ("left_ee", "right_ee")
TACTILE_FINGERS = ("thumb", "index", "middle", "ring", "little")


def _as_data_json(path: Path) -> Path:
    if path.is_dir():
        path = path / "data.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _flatten_numeric(value: Any) -> list[float]:
    out: list[float] = []
    if value is None:
        return out
    if isinstance(value, bool):
        return [float(value)]
    if isinstance(value, (int, float)):
        return [float(value)] if np.isfinite(value) else []
    if isinstance(value, list):
        for item in value:
            out.extend(_flatten_numeric(item))
        return out
    if isinstance(value, dict):
        for item in value.values():
            out.extend(_flatten_numeric(item))
        return out
    return out


def _get_qpos(frame: dict[str, Any], group: str, part: str) -> list[float]:
    value = frame.get(group, {}).get(part, {}).get("qpos", [])
    return _flatten_numeric(value)


def _get_neck(frame: dict[str, Any], group: str, key: str) -> list[float]:
    value = frame.get(group, {}).get("neck", {}).get(key, [])
    return _flatten_numeric(value)


def _collect_matrix(frames: list[dict[str, Any]], group: str, part: str) -> np.ndarray | None:
    rows = [_get_qpos(frame, group, part) for frame in frames]
    width = max((len(row) for row in rows), default=0)
    if width == 0:
        return None
    arr = np.full((len(rows), width), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        arr[i, : len(row)] = row
    return arr


def _collect_neck(frames: list[dict[str, Any]], group: str, key: str) -> np.ndarray | None:
    rows = [_get_neck(frame, group, key) for frame in frames]
    width = max((len(row) for row in rows), default=0)
    if width == 0:
        return None
    arr = np.full((len(rows), width), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        arr[i, : len(row)] = row
    return arr


def _plot_matrix(ax, arr: np.ndarray | None, title: str, ylabel: str = "value") -> None:
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if arr is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
        return
    for i in range(arr.shape[1]):
        ax.plot(arr[:, i], linewidth=0.8, label=str(i))
    if arr.shape[1] <= 14:
        ax.legend(ncol=min(arr.shape[1], 7), fontsize=7, loc="upper right")


def _plot_state_action(frames: list[dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(len(QPOS_PARTS), 2, figsize=(16, 3.0 * len(QPOS_PARTS)), sharex=True)
    fig.suptitle("Raw qpos from xr_teleoperate data.json", fontsize=14)
    for row, part in enumerate(QPOS_PARTS):
        state = _collect_matrix(frames, "states", part)
        action = _collect_matrix(frames, "actions", part)
        _plot_matrix(axes[row, 0], state, f"states.{part}.qpos")
        _plot_matrix(axes[row, 1], action, f"actions.{part}.qpos")
    axes[-1, 0].set_xlabel("frame")
    axes[-1, 1].set_xlabel("frame")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_dir / "qpos_state_action.png", dpi=160)
    plt.close(fig)


def _plot_neck(frames: list[dict[str, Any]], output_dir: Path) -> None:
    specs = [
        ("states", "raw_head_yaw_pitch"),
        ("states", "actual_yaw_pitch"),
        ("actions", "target_yaw_pitch"),
        ("actions", "command_yaw_pitch"),
    ]
    fig, axes = plt.subplots(len(specs), 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Neck fields", fontsize=14)
    for ax, (group, key) in zip(axes, specs):
        _plot_matrix(ax, _collect_neck(frames, group, key), f"{group}.neck.{key}", ylabel="rad")
    axes[-1].set_xlabel("frame")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_dir / "neck.png", dpi=160)
    plt.close(fig)


def _tactile_array(frame: dict[str, Any], side: str) -> list[float]:
    return _flatten_numeric(frame.get("tactiles", {}).get(side, {}))


def _tactile_component(frame: dict[str, Any], side: str, component: str) -> list[float]:
    tactile = frame.get("tactiles", {}).get(side, {})
    if not isinstance(tactile, dict):
        return []
    if component == "palm":
        return _flatten_numeric(tactile.get("palm", []))
    fingers = tactile.get("fingers", {})
    if not isinstance(fingers, dict):
        return []
    return _flatten_numeric(fingers.get(component, []))


def _collect_tactile_component(frames: list[dict[str, Any]], side: str, component: str) -> np.ndarray | None:
    rows = [_tactile_component(frame, side, component) for frame in frames]
    width = max((len(row) for row in rows), default=0)
    if width == 0:
        return None
    arr = np.full((len(rows), width), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        arr[i, : len(row)] = row
    return arr


def _plot_tactile(frames: list[dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Tactile flattened summary", fontsize=14)
    for ax, side in zip(axes, TACTILE_SIDES):
        rows = [_tactile_array(frame, side) for frame in frames]
        max_width = max((len(row) for row in rows), default=0)
        if max_width == 0:
            ax.text(0.5, 0.5, f"{side} missing", ha="center", va="center", transform=ax.transAxes)
            continue
        arr = np.full((len(rows), max_width), np.nan, dtype=np.float64)
        for i, row in enumerate(rows):
            arr[i, : len(row)] = row
        ax.plot(np.nanmean(arr, axis=1), label="mean", linewidth=1.0)
        ax.plot(np.nanmax(arr, axis=1), label="max", linewidth=1.0)
        ax.plot(np.nanmin(arr, axis=1), label="min", linewidth=1.0)
        ax.set_title(f"tactiles.{side} ({max_width} flattened numeric values)")
        ax.set_ylabel("raw")
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[-1].set_xlabel("frame")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "tactile_summary.png", dpi=160)
    plt.close(fig)


def _plot_tactile_detail(frames: list[dict[str, Any]], output_dir: Path) -> None:
    components = (*TACTILE_FINGERS, "palm")
    for side in TACTILE_SIDES:
        fig, axes = plt.subplots(len(components), 2, figsize=(18, 3.0 * len(components)), sharex="col")
        fig.suptitle(f"Tactile detail: {side}", fontsize=14)

        for row, component in enumerate(components):
            arr = _collect_tactile_component(frames, side, component)
            line_ax = axes[row, 0]
            heat_ax = axes[row, 1]

            line_ax.set_title(f"{component} channels")
            line_ax.set_ylabel("raw")
            line_ax.grid(True, alpha=0.25)
            heat_ax.set_title(f"{component} heatmap")

            if arr is None:
                line_ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=line_ax.transAxes)
                heat_ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=heat_ax.transAxes)
                continue

            for ch in range(arr.shape[1]):
                line_ax.plot(arr[:, ch], linewidth=0.8, label=f"ch{ch}")
            line_ax.legend(ncol=min(arr.shape[1], 6), fontsize=7, loc="upper right")

            image = heat_ax.imshow(arr.T, aspect="auto", interpolation="nearest", origin="lower")
            heat_ax.set_ylabel("channel")
            heat_ax.set_yticks(range(arr.shape[1]))
            heat_ax.set_yticklabels([f"ch{i}" for i in range(arr.shape[1])])
            fig.colorbar(image, ax=heat_ax, fraction=0.02, pad=0.01)

        axes[-1, 0].set_xlabel("frame")
        axes[-1, 1].set_xlabel("frame")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(output_dir / f"tactile_{side}_detail.png", dpi=160)
        plt.close(fig)


def _write_tactile_csv(frames: list[dict[str, Any]], output_dir: Path) -> None:
    rows: list[dict[str, float | int]] = []
    field_order: list[str] = ["frame_idx"]

    def add_values(row: dict[str, float | int], prefix: str, values: list[float]) -> None:
        for i, value in enumerate(values):
            key = f"{prefix}.ch{i}"
            row[key] = value
            if key not in field_order:
                field_order.append(key)

    for fallback_idx, frame in enumerate(frames):
        row: dict[str, float | int] = {"frame_idx": int(frame.get("idx", fallback_idx))}
        for side in TACTILE_SIDES:
            for finger in TACTILE_FINGERS:
                add_values(row, f"tactiles.{side}.fingers.{finger}", _tactile_component(frame, side, finger))
            add_values(row, f"tactiles.{side}.palm", _tactile_component(frame, side, "palm"))
        rows.append(row)

    with open(output_dir / "tactile_values.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        writer.writeheader()
        writer.writerows(rows)


def _availability_summary(frames: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_frames": len(frames),
        "qpos_dims": {},
        "neck_dims": {},
        "tactile_dims": {},
        "camera_keys": {},
        "audio_frames": 0,
    }

    for group in ("states", "actions"):
        for part in QPOS_PARTS:
            dims = [len(_get_qpos(frame, group, part)) for frame in frames]
            summary["qpos_dims"][f"{group}.{part}.qpos"] = {
                "min": int(min(dims, default=0)),
                "max": int(max(dims, default=0)),
                "nonempty_frames": int(sum(dim > 0 for dim in dims)),
            }

    for group, keys in {
        "states": ("raw_head_yaw_pitch", "actual_yaw_pitch"),
        "actions": ("target_yaw_pitch", "command_yaw_pitch"),
    }.items():
        for key in keys:
            dims = [len(_get_neck(frame, group, key)) for frame in frames]
            summary["neck_dims"][f"{group}.neck.{key}"] = {
                "min": int(min(dims, default=0)),
                "max": int(max(dims, default=0)),
                "nonempty_frames": int(sum(dim > 0 for dim in dims)),
            }

    for side in TACTILE_SIDES:
        for component in (*TACTILE_FINGERS, "palm"):
            dims = [len(_tactile_component(frame, side, component)) for frame in frames]
            summary["tactile_dims"][f"tactiles.{side}.{component}"] = {
                "min": int(min(dims, default=0)),
                "max": int(max(dims, default=0)),
                "nonempty_frames": int(sum(dim > 0 for dim in dims)),
            }

    camera_counts: dict[str, int] = {}
    for frame in frames:
        for key in frame.get("colors", {}).keys():
            camera_counts[key] = camera_counts.get(key, 0) + 1
        if frame.get("audios"):
            summary["audio_frames"] += 1
    summary["camera_keys"] = camera_counts
    return summary


def _plot_availability(summary: dict[str, Any], output_dir: Path) -> None:
    labels = []
    values = []
    for section in ("qpos_dims", "neck_dims", "tactile_dims"):
        for key, item in summary[section].items():
            labels.append(key)
            values.append(item["max"])
    fig, ax = plt.subplots(figsize=(14, max(5, 0.35 * len(labels))))
    ax.barh(labels, values)
    ax.set_title("Max numeric dimension per raw field")
    ax.set_xlabel("dimension")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "field_dimensions.png", dpi=160)
    plt.close(fig)


def _write_flat_csv(frames: list[dict[str, Any]], output_dir: Path) -> None:
    rows: list[dict[str, float | int]] = []
    field_order: list[str] = ["frame_idx"]

    def add_values(row: dict[str, float | int], prefix: str, values: list[float]) -> None:
        for i, value in enumerate(values):
            key = f"{prefix}.{i}"
            row[key] = value
            if key not in field_order:
                field_order.append(key)

    for fallback_idx, frame in enumerate(frames):
        row: dict[str, float | int] = {"frame_idx": int(frame.get("idx", fallback_idx))}
        for group in ("states", "actions"):
            for part in QPOS_PARTS:
                add_values(row, f"{group}.{part}.qpos", _get_qpos(frame, group, part))
        for group, keys in {
            "states": ("raw_head_yaw_pitch", "actual_yaw_pitch"),
            "actions": ("target_yaw_pitch", "command_yaw_pitch"),
        }.items():
            for key in keys:
                add_values(row, f"{group}.neck.{key}", _get_neck(frame, group, key))
        rows.append(row)

    with open(output_dir / "flat_values.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, default=None, help="Episode directory containing data.json.")
    parser.add_argument("--json", type=Path, default=None, help="Path to data.json.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for PNG/CSV outputs.")
    args = parser.parse_args()

    if args.json is None and args.episode is None:
        raise SystemExit("Provide --episode or --json")

    json_path = _as_data_json(args.json or args.episode)
    episode_dir = json_path.parent
    output_dir = args.output_dir or (episode_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    episode = json.load(open(json_path, encoding="utf-8"))
    frames = episode.get("data", [])
    if not frames:
        raise ValueError(f"No frames found in {json_path}")

    _plot_state_action(frames, output_dir)
    _plot_neck(frames, output_dir)
    _plot_tactile(frames, output_dir)
    _plot_tactile_detail(frames, output_dir)

    summary = _availability_summary(frames)
    _plot_availability(summary, output_dir)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _write_flat_csv(frames, output_dir)
    _write_tactile_csv(frames, output_dir)

    print(f"Wrote plots and raw table to: {output_dir}")
    print(f"Frames: {summary['num_frames']}")
    print("Camera frames:", summary["camera_keys"])
    print("Audio frames:", summary["audio_frames"])


if __name__ == "__main__":
    main()
