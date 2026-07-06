#!/usr/bin/env python3
"""Realtime plotter for teleop episode data and optional WAV audio.

This script is intentionally separate from data collection. It tails the
partially-written ``data.json`` and the growing ``audios/audio.wav`` file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import numpy as np
except ImportError:
    np = None


QPOS_PARTS = ("left_arm", "right_arm", "body")
HAND_PARTS = ("left_ee", "right_ee")
STATE_FIELDS = ("qpos", "qvel", "torque")


def flatten_numeric(value: Any) -> list[float]:
    out: list[float] = []
    if value is None:
        return out
    if isinstance(value, bool):
        return [float(value)]
    if isinstance(value, (int, float)):
        value = float(value)
        return [value] if math.isfinite(value) else []
    if isinstance(value, list):
        for item in value:
            out.extend(flatten_numeric(item))
    elif isinstance(value, dict):
        for item in value.values():
            out.extend(flatten_numeric(item))
    return out


def parse_partial_episode(text: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        episode = json.loads(text)
        frames = episode.get("data", [])
        return episode.get("info", {}), frames if isinstance(frames, list) else []
    except json.JSONDecodeError:
        pass

    info: dict[str, Any] = {}
    info_key = text.find('"info"')
    data_key = text.find('"data"')
    if 0 <= info_key < data_key:
        info_start = text.find("{", info_key)
        decoder = json.JSONDecoder()
        if info_start >= 0:
            try:
                info_obj, _ = decoder.raw_decode(text, info_start)
                if isinstance(info_obj, dict):
                    info = info_obj
            except json.JSONDecodeError:
                info = {}

    array_start = text.find("[", data_key)
    if array_start < 0:
        return info, []

    decoder = json.JSONDecoder()
    pos = array_start + 1
    frames: list[dict[str, Any]] = []
    while pos < len(text):
        while pos < len(text) and text[pos] in " \t\r\n,":
            pos += 1
        if pos >= len(text) or text[pos] == "]":
            break
        try:
            item, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            break
        if isinstance(item, dict):
            frames.append(item)
        pos = end
    return info, frames


def load_frames(data_json: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not data_json.exists():
        return {}, []
    try:
        text = data_json.read_text(encoding="utf-8")
    except OSError:
        return {}, []
    return parse_partial_episode(text)


def append_series(
    series: dict[str, list[float]],
    present: set[str],
    frame_idx: int,
    prefix: str,
    values: list[float],
    max_channels: int,
) -> None:
    for channel, value in enumerate(values[:max_channels]):
        label = f"{prefix}[{channel}]"
        if label not in series:
            series[label] = [float("nan")] * frame_idx
        series[label].append(float(value))
        present.add(label)


def collect_series(frames: list[dict[str, Any]], kind: str, max_channels: int) -> dict[str, np.ndarray]:
    series: dict[str, list[float]] = {}
    for frame_idx, frame in enumerate(frames):
        present: set[str] = set()
        if kind == "robot":
            for group in ("states", "actions"):
                for part in QPOS_PARTS:
                    for field in STATE_FIELDS:
                        values = flatten_numeric((frame.get(group) or {}).get(part, {}).get(field, []))
                        append_series(series, present, frame_idx, f"{group}.{part}.{field}", values, max_channels)
        elif kind == "hands":
            for group in ("states", "actions"):
                for part in HAND_PARTS:
                    for field in STATE_FIELDS:
                        values = flatten_numeric((frame.get(group) or {}).get(part, {}).get(field, []))
                        append_series(series, present, frame_idx, f"{group}.{part}.{field}", values, max_channels)
        elif kind == "neck":
            specs = (
                ("states.neck.raw", (frame.get("states") or {}).get("neck", {}).get("raw_head_yaw_pitch", [])),
                ("states.neck.actual", (frame.get("states") or {}).get("neck", {}).get("actual_yaw_pitch", [])),
                ("actions.neck.target", (frame.get("actions") or {}).get("neck", {}).get("target_yaw_pitch", [])),
                ("actions.neck.command", (frame.get("actions") or {}).get("neck", {}).get("command_yaw_pitch", [])),
            )
            for prefix, raw_values in specs:
                append_series(series, present, frame_idx, prefix, flatten_numeric(raw_values), 2)
        for label, values in series.items():
            if label not in present:
                values.append(float("nan"))
    return {label: np.asarray(values, dtype=np.float64) for label, values in series.items()}


def read_audio_samples(path: Path, sample_rate: int, channels: int, window_seconds: float) -> np.ndarray:
    if not path.exists():
        return np.asarray([], dtype=np.int16)
    try:
        raw = path.read_bytes()
    except OSError:
        return np.asarray([], dtype=np.int16)
    if len(raw) <= 44:
        return np.asarray([], dtype=np.int16)
    samples = np.frombuffer(raw[44:], dtype="<i2")
    if channels > 1 and samples.size >= channels:
        samples = samples[: samples.size - (samples.size % channels)]
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
    max_samples = int(max(window_seconds, 0.1) * max(sample_rate, 1))
    return samples[-max_samples:].copy()


def audio_rms_series(samples: np.ndarray, chunk_size: int) -> np.ndarray:
    if samples.size == 0:
        return np.asarray([], dtype=np.float64)
    chunk_size = max(int(chunk_size), 1)
    usable = samples.size - (samples.size % chunk_size)
    if usable <= 0:
        data = samples.astype(np.float64)
        return np.asarray([float(np.sqrt(np.mean(data * data)) / 32768.0)])
    chunks = samples[:usable].reshape(-1, chunk_size).astype(np.float64)
    return np.sqrt(np.mean(chunks * chunks, axis=1)) / 32768.0


def plot_series(ax, title: str, series: dict[str, np.ndarray], window: int, max_lines: int) -> None:
    ax.clear()
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if not series:
        ax.text(0.5, 0.5, "waiting for data", ha="center", va="center", transform=ax.transAxes)
        return
    for label, values in list(series.items())[:max_lines]:
        y = values[-window:]
        x = np.arange(values.size - y.size, values.size)
        ax.plot(x, y, linewidth=1.0, label=label)
    ax.legend(loc="upper left", fontsize=7, ncol=2)


def plot_audio(ax, args, info: dict[str, Any]) -> None:
    ax.clear()
    ax.set_title("audio")
    ax.grid(True, alpha=0.25)
    audio_info = info.get("audio", {}) if isinstance(info, dict) else {}
    sample_rate = int(audio_info.get("sample_rate") or args.audio_sample_rate)
    channels = int(audio_info.get("channels") or args.audio_channels)
    chunk_size = int(audio_info.get("chunk_size") or args.audio_chunk_size)
    samples = read_audio_samples(args.audio, sample_rate, channels, args.audio_window_seconds)
    if samples.size == 0:
        ax.text(0.5, 0.5, "waiting for audio", ha="center", va="center", transform=ax.transAxes)
        return
    if args.audio_view == "waveform":
        y = samples.astype(np.float64) / 32768.0
        if y.size > args.audio_max_points:
            step = int(math.ceil(y.size / args.audio_max_points))
            y = y[::step]
        x = np.linspace(-len(samples) / max(sample_rate, 1), 0.0, y.size)
        ax.plot(x, y, linewidth=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("seconds")
        ax.set_ylabel("amplitude")
    else:
        rms = audio_rms_series(samples, chunk_size)
        x = np.arange(rms.size)
        ax.plot(x, rms, linewidth=1.0)
        ax.set_ylim(0.0, max(0.05, float(np.nanmax(rms)) * 1.2))
        ax.set_ylabel("RMS")


def resolve_paths(args) -> None:
    if args.episode is not None:
        if args.data_json is None:
            args.data_json = args.episode / "data.json"
        if args.audio is None:
            args.audio = args.episode / "audios" / "audio.wav"
    if args.data_json is None:
        raise ValueError("--episode or --data-json is required")
    if args.audio is None:
        args.audio = args.data_json.parent / "audios" / "audio.wav"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, help="Episode directory to monitor.")
    parser.add_argument("--data-json", type=Path, help="Path to data.json. Can be partially written.")
    parser.add_argument("--audio", type=Path, help="Path to audios/audio.wav.")
    parser.add_argument("--refresh", type=float, default=0.25, help="Plot refresh period in seconds.")
    parser.add_argument("--window-frames", type=int, default=300, help="Recent data.json frames to display.")
    parser.add_argument("--max-lines", type=int, default=16, help="Maximum lines per subplot.")
    parser.add_argument("--max-channels", type=int, default=8, help="Maximum channels per qpos/qvel/torque vector.")
    parser.add_argument("--audio-view", choices=["rms", "waveform"], default="rms")
    parser.add_argument("--audio-window-seconds", type=float, default=5.0)
    parser.add_argument("--audio-sample-rate", type=int, default=48000)
    parser.add_argument("--audio-channels", type=int, default=1)
    parser.add_argument("--audio-chunk-size", type=int, default=1024)
    parser.add_argument("--audio-max-points", type=int, default=4000)
    args = parser.parse_args()
    if plt is None or np is None:
        print("realtime_data_plot.py requires matplotlib and numpy in this Python environment.", file=sys.stderr)
        return 1
    resolve_paths(args)

    plt.ion()
    fig, axes = plt.subplots(5, 1, figsize=(13, 11), constrained_layout=True)
    fig.canvas.manager.set_window_title("teleop realtime data plot")

    while plt.fignum_exists(fig.number):
        info, frames = load_frames(args.data_json)
        recent = frames[-max(args.window_frames, 1) :]
        fig.suptitle(f"{args.data_json} | frames={len(frames)}", fontsize=10)
        plot_series(axes[0], "robot states/actions qpos/qvel/torque", collect_series(recent, "robot", args.max_channels), args.window_frames, args.max_lines)
        plot_series(axes[1], "hand command/state", collect_series(recent, "hands", args.max_channels), args.window_frames, args.max_lines)
        plot_series(axes[2], "neck yaw/pitch target/command/actual", collect_series(recent, "neck", args.max_channels), args.window_frames, args.max_lines)
        plot_audio(axes[3], args, info)
        axes[4].clear()
        axes[4].axis("off")
        audio_meta = info.get("audio", {}) if isinstance(info, dict) else {}
        axes[4].text(
            0.0,
            0.8,
            f"audio={args.audio}\n"
            f"device={audio_meta.get('device')} backend={audio_meta.get('backend')} "
            f"chunks={audio_meta.get('total_chunks')} dropped={audio_meta.get('dropped_chunks')}\n"
            f"refresh={args.refresh}s",
            va="top",
            family="monospace",
        )
        plt.pause(max(args.refresh, 0.05))
        time.sleep(0.001)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
