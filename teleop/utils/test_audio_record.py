#!/usr/bin/env python3
"""Record a short WAV file with the teleop background audio recorder."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teleop.utils.audio_recorder import AudioRecorderError, BackgroundAudioRecorder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="plughw:2,0")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--dtype", choices=["int16"], default="int16")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=Path("/tmp/test_audio.wav"))
    args = parser.parse_args()

    recorder = BackgroundAudioRecorder(
        str(args.output),
        device=args.device,
        sample_rate=args.sample_rate,
        channels=args.channels,
        dtype=args.dtype,
        chunk_size=args.chunk_size,
        rel_path=str(args.output),
    )

    try:
        recorder.start()
    except AudioRecorderError as exc:
        print(f"failed to start audio recording: {exc}", file=sys.stderr)
        return 1

    print(
        f"recording {args.duration:.2f}s to {args.output} "
        f"device={args.device} sample_rate={args.sample_rate} channels={args.channels}"
    )
    try:
        deadline = time.time() + max(args.duration, 0.0)
        while time.time() < deadline:
            time.sleep(min(0.1, max(deadline - time.time(), 0.0)))
    except KeyboardInterrupt:
        print("stopping early after KeyboardInterrupt")
    finally:
        metadata = recorder.stop()

    print(json.dumps(metadata, indent=2))
    print(f"chunk timestamps: {len(recorder.chunk_timestamps)} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
