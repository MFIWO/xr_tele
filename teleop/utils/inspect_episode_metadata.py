#!/usr/bin/env python3
"""Inspect xr_teleoperate EpisodeWriter output schema and saved metadata.

Pass either an episode directory or a data.json path. The script prints a
compact summary and can write a machine-readable JSON report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None


QPOS_PARTS = ("left_arm", "right_arm", "left_ee", "right_ee", "body", "neck")
TACTILE_SIDES = ("left_ee", "right_ee")
TACTILE_COMPONENTS = ("thumb", "index", "middle", "ring", "little", "palm")
DEFAULT_AUDIO_SAMPLE_RATE_HZ = 16000.0


def flatten_numeric(value: Any) -> list[float]:
    values: list[float] = []
    if value is None:
        return values
    if isinstance(value, bool):
        return [float(value)]
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        for item in value:
            values.extend(flatten_numeric(item))
    elif isinstance(value, dict):
        for item in value.values():
            values.extend(flatten_numeric(item))
    return values


def load_episode(path: Path) -> tuple[Path, dict[str, Any]]:
    data_json = path / "data.json" if path.is_dir() else path
    if not data_json.exists():
        raise FileNotFoundError(data_json)
    with data_json.open(encoding="utf-8") as f:
        episode = json.load(f)
    return data_json.parent, episode


def storage_contract() -> dict[str, Any]:
    return {
        "writer": "teleop.utils.episode_writer.EpisodeWriter",
        "write_flow": [
            "teleop_hand_and_arm.on_press('s') sets RECORD_TOGGLE",
            "main loop calls EpisodeWriter.create_episode() on first toggle",
            "main loop calls EpisodeWriter.add_item(...) while RECORD_RUNNING",
            "EpisodeWriter worker thread writes files and appends data.json frames",
            "second toggle calls EpisodeWriter.save_episode(), closing the JSON array",
        ],
        "episode_json": {
            "path": "data.json",
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
        },
        "colors": {
            "path_pattern": "colors/{idx:06d}_{camera_key}.jpg",
            "tool": "cv2.imwrite",
            "json_reference": "data[*].colors[camera_key]",
        },
        "depths": {
            "path_pattern": "depths/{idx:06d}_{depth_key}.jpg",
            "tool": "cv2.imwrite",
            "json_reference": "data[*].depths[depth_key]",
        },
        "audios": {
            "path_pattern": "audios/audio_{idx:06d}_{mic_key}.npy",
            "tool": "numpy.save",
            "format": "int16 PCM samples in .npy",
            "json_reference": "data[*].audios[mic_key]",
            "source": "--record-audio-udp-port via AudioUDPReceiver",
        },
        "continuous_audio": {
            "path": "audios/audio.wav",
            "tool": "teleop.utils.audio_recorder.BackgroundAudioRecorder",
            "format": "WAV/S16_LE",
            "metadata": "info.audio when --enable-audio is set",
            "chunk_timestamps": "info.audio_chunks when available",
        },
        "rerun_live_logging": {
            "tool": "teleop.utils.rerun_visualizer.RerunLogger",
            "enabled_when": "EpisodeWriter(rerun_log=True), normally not --headless",
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


def summarize_refs(episode_dir: Path, frames: list[dict[str, Any]], field: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "frames_with_refs": 0,
        "total_refs": 0,
        "keys": {},
        "missing": [],
    }
    for frame in frames:
        refs = frame.get(field) or {}
        if not isinstance(refs, dict) or not refs:
            continue
        summary["frames_with_refs"] += 1
        for key, rel_path in refs.items():
            stats = summary["keys"].setdefault(
                key,
                {
                    "refs": 0,
                    "existing_files": 0,
                    "missing_files": 0,
                    "extensions": {},
                    "first_path": None,
                    "last_path": None,
                },
            )
            summary["total_refs"] += 1
            stats["refs"] += 1
            stats["first_path"] = stats["first_path"] or rel_path
            stats["last_path"] = rel_path
            suffix = Path(str(rel_path)).suffix.lower() if rel_path else ""
            stats["extensions"][suffix] = stats["extensions"].get(suffix, 0) + 1

            if rel_path and (episode_dir / str(rel_path)).exists():
                stats["existing_files"] += 1
            else:
                stats["missing_files"] += 1
                summary["missing"].append({"idx": frame.get("idx"), "key": key, "path": rel_path})
    return summary


def qpos_dims(frames: list[dict[str, Any]]) -> dict[str, Any]:
    dims: dict[str, Any] = {}
    for group in ("states", "actions"):
        for part in QPOS_PARTS:
            key = f"{group}.{part}.qpos"
            lengths = [
                len(flatten_numeric((frame.get(group) or {}).get(part, {}).get("qpos", [])))
                for frame in frames
            ]
            dims[key] = {
                "min": min(lengths, default=0),
                "max": max(lengths, default=0),
                "nonempty_frames": sum(length > 0 for length in lengths),
            }
    return dims


def tactile_dims(frames: list[dict[str, Any]]) -> dict[str, Any]:
    dims: dict[str, Any] = {}
    for side in TACTILE_SIDES:
        for component in TACTILE_COMPONENTS:
            key = f"tactiles.{side}.{component}"
            lengths = []
            for frame in frames:
                tactile = (frame.get("tactiles") or {}).get(side, {})
                if not isinstance(tactile, dict):
                    lengths.append(0)
                    continue
                if component == "palm":
                    values = tactile.get("palm", [])
                else:
                    values = (tactile.get("fingers") or {}).get(component, [])
                lengths.append(len(flatten_numeric(values)))
            dims[key] = {
                "min": min(lengths, default=0),
                "max": max(lengths, default=0),
                "nonempty_frames": sum(length > 0 for length in lengths),
            }
    return dims


def summarize_audio(episode_dir: Path, episode: dict[str, Any], frames: list[dict[str, Any]]) -> dict[str, Any]:
    info_audio = episode.get("info", {}).get("audio", {})
    sample_rate = float(info_audio.get("sample_rate", DEFAULT_AUDIO_SAMPLE_RATE_HZ) or DEFAULT_AUDIO_SAMPLE_RATE_HZ)
    summary: dict[str, Any] = summarize_refs(episode_dir, frames, "audios")
    summary.update(
        {
            "sample_rate_hz": sample_rate,
            "numpy_available": np is not None,
            "continuous_wav": {},
            "mics": {},
            "load_errors": [],
        }
    )
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
    if np is None:
        return summary

    for frame in frames:
        audios = frame.get("audios") or {}
        if not isinstance(audios, dict):
            continue
        for mic, rel_path in audios.items():
            mic_stats = summary["mics"].setdefault(
                mic,
                {
                    "files_loaded": 0,
                    "sample_frames": 0,
                    "sample_values": 0,
                    "min_sample_frames_per_file": None,
                    "max_sample_frames_per_file": 0,
                    "dtypes": {},
                    "shapes": {},
                },
            )
            if not rel_path:
                continue
            audio_path = episode_dir / str(rel_path)
            if not audio_path.exists():
                continue
            try:
                audio = np.load(audio_path, mmap_mode="r", allow_pickle=False)
            except Exception as exc:
                summary["load_errors"].append({"idx": frame.get("idx"), "mic": mic, "path": rel_path, "error": repr(exc)})
                continue

            shape = tuple(int(dim) for dim in audio.shape)
            shape_key = "x".join(str(dim) for dim in shape) if shape else "scalar"
            dtype_key = str(audio.dtype)
            sample_frames = int(shape[0]) if shape else int(audio.size)
            sample_values = int(audio.size)

            mic_stats["files_loaded"] += 1
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

    for mic_stats in summary["mics"].values():
        mic_stats["approx_saved_chunk_duration_s"] = (
            float(mic_stats["sample_frames"]) / sample_rate if sample_rate > 0.0 else None
        )
    return summary


def summarize_episode(episode_dir: Path, episode: dict[str, Any]) -> dict[str, Any]:
    frames = episode.get("data", [])
    if not isinstance(frames, list):
        raise ValueError("episode data key must be a list")

    frame_keys = sorted({key for frame in frames if isinstance(frame, dict) for key in frame.keys()})
    return {
        "episode_dir": str(episode_dir),
        "data_json": str(episode_dir / "data.json"),
        "num_frames": len(frames),
        "top_level_keys": sorted(episode.keys()),
        "frame_keys_seen": frame_keys,
        "info": episode.get("info", {}),
        "text": episode.get("text", {}),
        "recording_metadata": (episode.get("info") or {}).get("recording", {}),
        "storage_contract": storage_contract(),
        "streams": {
            "colors": summarize_refs(episode_dir, frames, "colors"),
            "depths": summarize_refs(episode_dir, frames, "depths"),
            "audios": summarize_refs(episode_dir, frames, "audios"),
        },
        "qpos_dims": qpos_dims(frames),
        "tactile_dims": tactile_dims(frames),
        "audio": summarize_audio(episode_dir, episode, frames),
    }


def print_human_summary(summary: dict[str, Any]) -> None:
    print(f"episode_dir: {summary['episode_dir']}")
    print(f"frames: {summary['num_frames']}")
    print("storage: data.json + colors/*.jpg + depths/*.jpg + audios/*.npy + optional audios/audio.wav")
    recording = summary.get("recording_metadata") or {}
    if recording:
        audio_meta = recording.get("audio", {})
        print(f"recording.audio: {audio_meta}")
    for field in ("colors", "depths", "audios"):
        stream = summary["streams"][field]
        keys = ", ".join(sorted(stream["keys"].keys())) or "none"
        print(f"{field}: frames_with_refs={stream['frames_with_refs']} total_refs={stream['total_refs']} keys={keys}")
    audio = summary["audio"]
    continuous_wav = audio.get("continuous_wav") or {}
    if continuous_wav:
        print(
            f"audio.wav: exists={continuous_wav['exists']} "
            f"size_bytes={continuous_wav['size_bytes']} chunks={continuous_wav['chunks']}"
        )
    for mic, stats in audio.get("mics", {}).items():
        duration = stats.get("approx_saved_chunk_duration_s")
        duration_text = f"{duration:.3f}s" if isinstance(duration, (int, float)) else "unknown"
        print(
            f"audio.{mic}: files_loaded={stats['files_loaded']} "
            f"sample_frames={stats['sample_frames']} duration={duration_text} "
            f"dtypes={stats['dtypes']} shapes={stats['shapes']}"
        )
    if not audio.get("mics"):
        print("audio: no loadable .npy audio files referenced")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path, help="Episode directory or data.json path")
    parser.add_argument("--output-json", type=Path, help="Optional report JSON path")
    args = parser.parse_args()

    episode_dir, episode = load_episode(args.episode)
    summary = summarize_episode(episode_dir, episode)
    print_human_summary(summary)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
