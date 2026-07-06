#!/usr/bin/env python3
"""Create plots and camera previews for recorded xr_teleoperate episodes.

This script intentionally uses only stdlib, numpy, and OpenCV so it can run in
minimal robot environments where matplotlib is not installed.

Examples:
  python teleop/utils/visualize_recorded_episodes.py --episodes 23 25
  python teleop/utils/visualize_recorded_episodes.py --base-dir "teleop/utils/data/pick cube" --episodes 23 25
  python teleop/utils/visualize_recorded_episodes.py --output-root ../xr_episode_visualizations
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


QPOS_PARTS = ("left_arm", "right_arm", "left_ee", "right_ee", "body")
TACTILE_SIDES = ("left_ee", "right_ee")
TACTILE_COMPONENTS = ("thumb", "index", "middle", "ring", "little", "palm")
AUDIO_DEFAULT_SAMPLE_RATE_HZ = 16000.0
COLORS = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
    (57, 59, 121),
    (82, 84, 163),
    (107, 110, 207),
    (156, 158, 222),
    (99, 121, 57),
    (140, 162, 82),
)


def flatten_numeric(value: Any) -> list[float]:
    out: list[float] = []
    if value is None:
        return out
    if isinstance(value, bool):
        return [float(value)]
    if isinstance(value, (int, float)):
        return [float(value)] if math.isfinite(float(value)) else []
    if isinstance(value, list):
        for item in value:
            out.extend(flatten_numeric(item))
    elif isinstance(value, dict):
        for item in value.values():
            out.extend(flatten_numeric(item))
    return out


def collect_rows(frames: list[dict[str, Any]], getter) -> np.ndarray | None:
    rows = [getter(frame) for frame in frames]
    width = max((len(row) for row in rows), default=0)
    if width == 0:
        return None
    arr = np.full((len(rows), width), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        arr[i, : len(row)] = row
    return arr


def qpos_getter(group: str, part: str):
    def get(frame: dict[str, Any]) -> list[float]:
        return flatten_numeric(frame.get(group, {}).get(part, {}).get("qpos", []))

    return get


def neck_getter(group: str, key: str):
    def get(frame: dict[str, Any]) -> list[float]:
        return flatten_numeric(frame.get(group, {}).get("neck", {}).get(key, []))

    return get


def tactile_getter(side: str, component: str):
    def get(frame: dict[str, Any]) -> list[float]:
        tactile = frame.get("tactiles", {}).get(side, {})
        if not isinstance(tactile, dict):
            return []
        if component == "palm":
            return flatten_numeric(tactile.get("palm", []))
        fingers = tactile.get("fingers", {})
        if not isinstance(fingers, dict):
            return []
        return flatten_numeric(fingers.get(component, []))

    return get


def storage_contract() -> dict[str, Any]:
    return {
        "writer": "teleop.utils.episode_writer.EpisodeWriter",
        "episode_json": "data.json",
        "top_level_keys": ["info", "text", "data"],
        "frame_keys": [
            "idx",
            "colors",
            "depths",
            "states",
            "actions",
            "tactiles",
            "audios",
            "sim_state",
        ],
        "colors": {
            "path_pattern": "colors/{idx:06d}_{camera_key}.jpg",
            "writer": "cv2.imwrite",
            "json_reference": "data[*].colors[camera_key]",
        },
        "depths": {
            "path_pattern": "depths/{idx:06d}_{depth_key}.jpg",
            "writer": "cv2.imwrite",
            "json_reference": "data[*].depths[depth_key]",
        },
        "audios": {
            "path_pattern": "audios/audio_{idx:06d}_{mic_key}.npy",
            "writer": "numpy.save",
            "dtype": "int16",
            "json_reference": "data[*].audios[mic_key]",
            "input": "teleop_hand_and_arm.AudioUDPReceiver latest UDP int16 PCM packet",
        },
        "continuous_audio": {
            "path": "audios/audio.wav",
            "writer": "teleop.utils.audio_recorder.BackgroundAudioRecorder",
            "format": "WAV/S16_LE",
            "metadata": "info.audio when --enable-audio is set",
            "chunk_timestamps": "info.audio_chunks when available",
        },
        "online_rerun_logging": {
            "logger": "teleop.utils.rerun_visualizer.RerunLogger",
            "currently_logs": [
                "states/actions qpos/qvel/torque scalar series",
                "neck scalar series",
                "colors/depths images",
                "tactile scalar series and summaries",
                "frame audio sample_count/rms/peak_abs when .npy audio chunks are present",
            ],
            "note": "continuous audios/audio.wav is summarized by realtime_data_plot.py, not streamed as raw audio into Rerun",
        },
    }


def put_label(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.6) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (35, 35, 35), 1, cv2.LINE_AA)


def line_plot(
    arr: np.ndarray | None,
    title: str,
    *,
    width: int = 1120,
    height: int = 320,
    ylabel: str = "value",
    max_channels: int | None = None,
) -> np.ndarray:
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    margin_l, margin_r, margin_t, margin_b = 72, 20, 44, 42
    x0, y0 = margin_l, height - margin_b
    x1, y1 = width - margin_r, margin_t
    cv2.rectangle(img, (x0, y1), (x1, y0), (218, 218, 218), 1)
    put_label(img, title, (16, 28), 0.62)
    cv2.putText(img, ylabel, (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(img, "frame", (width // 2 - 24, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA)

    if arr is None or arr.size == 0:
        put_label(img, "missing", (width // 2 - 45, height // 2), 0.7)
        return img

    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        put_label(img, "no finite values", (width // 2 - 80, height // 2), 0.7)
        return img
    vmin, vmax = float(np.nanmin(valid)), float(np.nanmax(valid))
    if math.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.1, 1.0)
        vmin -= pad
        vmax += pad
    else:
        pad = (vmax - vmin) * 0.04
        vmin -= pad
        vmax += pad

    for frac in np.linspace(0.0, 1.0, 5):
        y = int(y0 - frac * (y0 - y1))
        cv2.line(img, (x0, y), (x1, y), (235, 235, 235), 1)
        val = vmin + frac * (vmax - vmin)
        cv2.putText(img, f"{val:.3g}", (8, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 70, 70), 1, cv2.LINE_AA)

    n = arr.shape[0]
    channels = arr.shape[1] if max_channels is None else min(arr.shape[1], max_channels)
    xs = np.linspace(x0, x1, n).astype(np.int32)
    for ch in range(channels):
        series = arr[:, ch]
        mask = np.isfinite(series)
        if not np.any(mask):
            continue
        ys = (y0 - (series - vmin) / (vmax - vmin) * (y0 - y1)).astype(np.float64)
        pts = np.column_stack([xs[mask], ys[mask].astype(np.int32)])
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, COLORS[ch % len(COLORS)], 1, cv2.LINE_AA)

    legend_x, legend_y = x1 - 185, y1 + 18
    legend_count = min(channels, 12)
    for ch in range(legend_count):
        y = legend_y + ch * 17
        cv2.line(img, (legend_x, y - 4), (legend_x + 22, y - 4), COLORS[ch % len(COLORS)], 2, cv2.LINE_AA)
        cv2.putText(img, f"ch{ch}", (legend_x + 28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (45, 45, 45), 1, cv2.LINE_AA)
    if channels > legend_count:
        cv2.putText(img, f"+{channels - legend_count} more", (legend_x, legend_y + legend_count * 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (45, 45, 45), 1, cv2.LINE_AA)
    return img


def heatmap(arr: np.ndarray | None, title: str, *, width: int = 1120, height: int = 240) -> np.ndarray:
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    put_label(img, title, (16, 28), 0.62)
    if arr is None or arr.size == 0:
        put_label(img, "missing", (width // 2 - 45, height // 2), 0.7)
        return img
    data = arr.T
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        put_label(img, "no finite values", (width // 2 - 80, height // 2), 0.7)
        return img
    vmin, vmax = np.nanpercentile(valid, [1, 99])
    if math.isclose(float(vmin), float(vmax)):
        vmin, vmax = np.nanmin(valid), np.nanmax(valid) + 1.0
    norm = np.clip((np.nan_to_num(data, nan=vmin) - vmin) / (vmax - vmin), 0.0, 1.0)
    gray = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
    plot = cv2.resize(colored, (width - 92, height - 70), interpolation=cv2.INTER_NEAREST)
    img[44 : 44 + plot.shape[0], 72 : 72 + plot.shape[1]] = plot
    cv2.rectangle(img, (72, 44), (72 + plot.shape[1], 44 + plot.shape[0]), (218, 218, 218), 1)
    cv2.putText(img, "channel", (7, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(img, "frame", (width // 2 - 24, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA)
    return img


def stack_grid(images: list[list[np.ndarray]], gap: int = 10) -> np.ndarray:
    if not images:
        raise ValueError("No images to stack")
    row_imgs = []
    for row in images:
        h = max(img.shape[0] for img in row)
        padded = []
        for img in row:
            if img.shape[0] < h:
                img = cv2.copyMakeBorder(img, 0, h - img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            padded.append(img)
        row_img = cv2.hconcat([padded[0]] + [cv2.copyMakeBorder(i, 0, 0, gap, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for i in padded[1:]])
        row_imgs.append(row_img)
    w = max(img.shape[1] for img in row_imgs)
    padded_rows = []
    for img in row_imgs:
        if img.shape[1] < w:
            img = cv2.copyMakeBorder(img, 0, 0, 0, w - img.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
        padded_rows.append(img)
    return cv2.vconcat([padded_rows[0]] + [cv2.copyMakeBorder(i, gap, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for i in padded_rows[1:]])


def plot_qpos(frames: list[dict[str, Any]], out_dir: Path) -> None:
    rows: list[list[np.ndarray]] = []
    for part in QPOS_PARTS:
        rows.append(
            [
                line_plot(collect_rows(frames, qpos_getter("states", part)), f"states.{part}.qpos"),
                line_plot(collect_rows(frames, qpos_getter("actions", part)), f"actions.{part}.qpos"),
            ]
        )
    cv2.imwrite(str(out_dir / "qpos_state_action.png"), stack_grid(rows))


def plot_neck(frames: list[dict[str, Any]], out_dir: Path) -> None:
    specs = (
        ("states", "raw_head_yaw_pitch"),
        ("states", "actual_yaw_pitch"),
        ("actions", "target_yaw_pitch"),
        ("actions", "command_yaw_pitch"),
    )
    rows = [[line_plot(collect_rows(frames, neck_getter(group, key)), f"{group}.neck.{key}", ylabel="rad")] for group, key in specs]
    cv2.imwrite(str(out_dir / "neck.png"), stack_grid(rows))


def plot_tactile(frames: list[dict[str, Any]], out_dir: Path) -> None:
    summary_rows: list[list[np.ndarray]] = []
    for side in TACTILE_SIDES:
        arrays = [collect_rows(frames, tactile_getter(side, component)) for component in TACTILE_COMPONENTS]
        valid_arrays = [arr for arr in arrays if arr is not None]
        if not valid_arrays:
            summary_rows.append([[line_plot(None, f"tactiles.{side}.summary")][0]])
            continue
        flat = np.concatenate(valid_arrays, axis=1)
        finite = np.isfinite(flat)
        summary = np.full((flat.shape[0], 3), np.nan, dtype=np.float64)
        for i in range(flat.shape[0]):
            row = flat[i, finite[i]]
            if row.size:
                summary[i] = (float(np.mean(row)), float(np.max(row)), float(np.min(row)))
        summary_rows.append([line_plot(summary, f"tactiles.{side}.summary (mean/max/min)", ylabel="raw")])
    cv2.imwrite(str(out_dir / "tactile_summary.png"), stack_grid(summary_rows))

    for side in TACTILE_SIDES:
        rows: list[list[np.ndarray]] = []
        for component in TACTILE_COMPONENTS:
            arr = collect_rows(frames, tactile_getter(side, component))
            rows.append(
                [
                    line_plot(arr, f"tactiles.{side}.{component}", ylabel="raw"),
                    heatmap(arr, f"tactiles.{side}.{component}.heatmap"),
                ]
            )
        cv2.imwrite(str(out_dir / f"tactile_{side}_detail.png"), stack_grid(rows))


def summarize_audio(episode_dir: Path, episode: dict[str, Any], frames: list[dict[str, Any]]) -> dict[str, Any]:
    info_audio = episode.get("info", {}).get("audio", {})
    sample_rate = float(info_audio.get("sample_rate", AUDIO_DEFAULT_SAMPLE_RATE_HZ) or AUDIO_DEFAULT_SAMPLE_RATE_HZ)
    channels = int(info_audio.get("channels", 1) or 1)
    summary: dict[str, Any] = {
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "frames_with_audio_refs": 0,
        "continuous_wav": {},
        "mics": {},
        "missing_files": [],
        "load_errors": [],
    }
    if str(info_audio.get("format", "")).lower() == "wav" and info_audio.get("path"):
        wav_path = episode_dir / str(info_audio["path"])
        summary["continuous_wav"] = {
            "path": str(info_audio["path"]),
            "exists": wav_path.exists(),
            "size_bytes": wav_path.stat().st_size if wav_path.exists() else 0,
            "start_timestamp": info_audio.get("start_timestamp"),
            "end_timestamp": info_audio.get("end_timestamp"),
            "chunk_size": info_audio.get("chunk_size"),
            "chunks": len((episode.get("info") or {}).get("audio_chunks", []) or []),
        }

    for frame in frames:
        frame_idx = frame.get("idx")
        audios = frame.get("audios") or {}
        if not isinstance(audios, dict) or not audios:
            continue
        summary["frames_with_audio_refs"] += 1
        for mic, rel_path in audios.items():
            mic_stats = summary["mics"].setdefault(
                mic,
                {
                    "frames": 0,
                    "files": 0,
                    "missing_files": 0,
                    "load_errors": 0,
                    "sample_frames": 0,
                    "sample_values": 0,
                    "min_sample_frames_per_file": None,
                    "max_sample_frames_per_file": 0,
                    "dtypes": {},
                    "shapes": {},
                    "first_path": None,
                    "last_path": None,
                },
            )
            mic_stats["frames"] += 1
            if not rel_path:
                mic_stats["missing_files"] += 1
                summary["missing_files"].append({"idx": frame_idx, "mic": mic, "path": rel_path})
                continue

            audio_path = episode_dir / str(rel_path)
            if not audio_path.exists():
                mic_stats["missing_files"] += 1
                summary["missing_files"].append({"idx": frame_idx, "mic": mic, "path": str(rel_path)})
                continue

            try:
                audio = np.load(audio_path, mmap_mode="r", allow_pickle=False)
            except Exception as exc:
                mic_stats["load_errors"] += 1
                summary["load_errors"].append({"idx": frame_idx, "mic": mic, "path": str(rel_path), "error": repr(exc)})
                continue

            shape = tuple(int(dim) for dim in audio.shape)
            shape_key = "x".join(str(dim) for dim in shape) if shape else "scalar"
            dtype_key = str(audio.dtype)
            sample_frames = int(shape[0]) if shape else int(audio.size)
            sample_values = int(audio.size)

            mic_stats["files"] += 1
            mic_stats["sample_frames"] += sample_frames
            mic_stats["sample_values"] += sample_values
            mic_stats["min_sample_frames_per_file"] = (
                sample_frames
                if mic_stats["min_sample_frames_per_file"] is None
                else min(mic_stats["min_sample_frames_per_file"], sample_frames)
            )
            mic_stats["max_sample_frames_per_file"] = max(mic_stats["max_sample_frames_per_file"], sample_frames)
            mic_stats["dtypes"][dtype_key] = mic_stats["dtypes"].get(dtype_key, 0) + 1
            mic_stats["shapes"][shape_key] = mic_stats["shapes"].get(shape_key, 0) + 1
            mic_stats["first_path"] = mic_stats["first_path"] or str(rel_path)
            mic_stats["last_path"] = str(rel_path)

    for mic_stats in summary["mics"].values():
        mic_stats["approx_saved_chunk_duration_s"] = (
            float(mic_stats["sample_frames"]) / sample_rate if sample_rate > 0.0 else None
        )
    return summary


def write_summary(episode_dir: Path, episode: dict[str, Any], frames: list[dict[str, Any]], out_dir: Path) -> None:
    summary: dict[str, Any] = {
        "num_frames": len(frames),
        "info": episode.get("info", {}),
        "text": episode.get("text", {}),
        "camera_keys": {},
        "qpos_dims": {},
        "tactile_dims": {},
        "audio": {},
        "storage_contract": storage_contract(),
    }
    for frame in frames:
        for key in frame.get("colors", {}).keys():
            summary["camera_keys"][key] = summary["camera_keys"].get(key, 0) + 1
    for group in ("states", "actions"):
        for part in QPOS_PARTS:
            dims = [len(qpos_getter(group, part)(frame)) for frame in frames]
            summary["qpos_dims"][f"{group}.{part}.qpos"] = {"min": min(dims), "max": max(dims), "nonempty_frames": sum(dim > 0 for dim in dims)}
    for side in TACTILE_SIDES:
        for component in TACTILE_COMPONENTS:
            dims = [len(tactile_getter(side, component)(frame)) for frame in frames]
            summary["tactile_dims"][f"tactiles.{side}.{component}"] = {"min": min(dims), "max": max(dims), "nonempty_frames": sum(dim > 0 for dim in dims)}
    summary["audio"] = summarize_audio(episode_dir, episode, frames)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def load_image(episode_dir: Path, rel_path: str | None, size: tuple[int, int]) -> np.ndarray:
    if not rel_path:
        return np.full((size[1], size[0], 3), 230, dtype=np.uint8)
    img = cv2.imread(str(episode_dir / rel_path))
    if img is None:
        return np.full((size[1], size[0], 3), 230, dtype=np.uint8)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def make_camera_sheet(episode_dir: Path, frames: list[dict[str, Any]], out_dir: Path, samples: int = 8) -> None:
    keys = sorted({key for frame in frames for key in frame.get("colors", {}).keys()})
    idxs = np.linspace(0, len(frames) - 1, min(samples, len(frames))).astype(int)
    rows = []
    for key in keys:
        row = []
        for i in idxs:
            frame = frames[int(i)]
            img = load_image(episode_dir, frame.get("colors", {}).get(key), (240, 135))
            put_label(img, f"{key} idx {frame.get('idx', i)}", (8, 20), 0.45)
            row.append(img)
        rows.append(row)
    cv2.imwrite(str(out_dir / "camera_contact_sheet.jpg"), stack_grid(rows, gap=6), [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def make_camera_video(episode_dir: Path, frames: list[dict[str, Any]], out_dir: Path, stride: int, fps: float) -> None:
    keys = sorted({key for frame in frames for key in frame.get("colors", {}).keys()})
    if not keys:
        return
    tile_size = (320, 180)
    cols = 2 if len(keys) > 1 else 1
    rows = math.ceil(len(keys) / cols)
    frame_size = (tile_size[0] * cols, tile_size[1] * rows)
    video_path = out_dir / "camera_preview.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), max(fps / stride, 1.0), frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {video_path}")
    blank = np.full((tile_size[1], tile_size[0], 3), 230, dtype=np.uint8)
    for frame in frames[::stride]:
        tiles = []
        for key in keys:
            img = load_image(episode_dir, frame.get("colors", {}).get(key), tile_size)
            put_label(img, f"{key} idx {frame.get('idx', 0)}", (8, 22), 0.55)
            tiles.append(img)
        while len(tiles) < rows * cols:
            tiles.append(blank.copy())
        grid_rows = [cv2.hconcat(tiles[r * cols : (r + 1) * cols]) for r in range(rows)]
        writer.write(cv2.vconcat(grid_rows))
    writer.release()


def process_episode(episode_dir: Path, out_dir: Path, video_stride: int, video_fps: float) -> None:
    json_path = episode_dir / "data.json"
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    with open(json_path, encoding="utf-8") as f:
        episode = json.load(f)
    frames = episode.get("data", [])
    if not frames:
        raise ValueError(f"No frames in {json_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_qpos(frames, out_dir)
    plot_neck(frames, out_dir)
    plot_tactile(frames, out_dir)
    make_camera_sheet(episode_dir, frames, out_dir)
    make_camera_video(episode_dir, frames, out_dir, video_stride, video_fps)
    write_summary(episode_dir, episode, frames, out_dir)
    print(f"Wrote {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path("teleop/utils/data/pick cube"))
    parser.add_argument("--episodes", type=int, nargs="+", default=[23, 25])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("../xr_episode_visualizations"),
        help="Directory where per-episode visualization folders are written.",
    )
    parser.add_argument("--video-stride", type=int, default=3, help="Use every Nth frame in camera_preview.mp4.")
    parser.add_argument("--video-fps", type=float, default=30.0, help="Source episode FPS.")
    args = parser.parse_args()

    base_dir = args.base_dir
    if not base_dir.is_absolute():
        base_dir = Path.cwd() / base_dir
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    for episode_idx in args.episodes:
        process_episode(
            base_dir / f"episode_{episode_idx:04d}",
            output_root / f"episode_{episode_idx:04d}",
            max(args.video_stride, 1),
            args.video_fps,
        )


if __name__ == "__main__":
    main()
