#!/usr/bin/env python3
"""Analyze and attenuate stationary fan noise in teleop WAV recordings.

The default preset is based on episode_0025's fan spectrum. It avoids scipy so
it can run in the lightweight teleop container.
"""

from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np


EPISODE25_FAN_NOTCHES = [
    (129.0, 35.0, 0.08),
    (205.0, 30.0, 0.25),
    (240.0, 35.0, 0.25),
    (258.0, 35.0, 0.25),
    (346.0, 45.0, 0.18),
    (387.0, 45.0, 0.08),
    (498.0, 45.0, 0.08),
    (516.0, 45.0, 0.08),
    (803.0, 45.0, 0.12),
    (820.0, 45.0, 0.12),
    (844.0, 45.0, 0.12),
    (873.0, 45.0, 0.10),
    (1266.0, 55.0, 0.20),
    (1670.0, 60.0, 0.18),
    (1764.0, 60.0, 0.22),
    (1840.0, 60.0, 0.20),
    (2725.0, 160.0, 0.16),
    (3088.0, 70.0, 0.14),
    (4550.0, 520.0, 0.08),
    (4758.0, 220.0, 0.08),
    (5500.0, 170.0, 0.22),
]


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.getnframes()
        raw = wf.readframes(frames)
    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported, got sample_width={sample_width}")
    data = np.frombuffer(raw, dtype="<i2")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return sample_rate, data.astype(np.float64)


def write_wav(path: Path, sample_rate: int, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 32700.0:
        data = data * (32700.0 / peak)
    out = np.clip(np.round(data), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(out.tobytes())


def smooth_lowpass(freq: np.ndarray, cutoff: float, transition: float) -> np.ndarray:
    mask = np.ones_like(freq)
    hi = cutoff + max(transition, 1.0)
    taper = (freq > cutoff) & (freq < hi)
    mask[freq >= hi] = 0.0
    mask[taper] = 0.5 * (1.0 + np.cos(np.pi * (freq[taper] - cutoff) / (hi - cutoff)))
    return mask


def smooth_highpass(freq: np.ndarray, cutoff: float, transition: float) -> np.ndarray:
    mask = np.ones_like(freq)
    lo = max(0.0, cutoff - max(transition, 1.0))
    taper = (freq > lo) & (freq < cutoff)
    mask[freq <= lo] = 0.0
    mask[taper] = 0.5 * (1.0 - np.cos(np.pi * (freq[taper] - lo) / max(cutoff - lo, 1.0)))
    return mask


def notch_mask(freq: np.ndarray, notches: list[tuple[float, float, float]]) -> np.ndarray:
    mask = np.ones_like(freq)
    for center, width, gain in notches:
        sigma = max(width, 1e-6) / 2.355
        notch = 1.0 - (1.0 - gain) * np.exp(-0.5 * ((freq - center) / sigma) ** 2)
        mask *= notch
    return mask


def parse_notch(raw: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in raw.split(",")]
    if len(parts) == 2:
        return parts[0], parts[1], 0.1
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    raise argparse.ArgumentTypeError("--notch must be center,width or center,width,gain")


def analyze(sample_rate: int, data: np.ndarray, nfft: int = 8192) -> dict:
    centered = data - float(np.mean(data))
    window = np.hanning(nfft)
    hop = nfft // 2
    spectra = []
    for start in range(0, max(1, len(centered) - nfft + 1), hop):
        segment = centered[start : start + nfft]
        if len(segment) < nfft:
            break
        segment = segment - float(np.mean(segment))
        spectra.append(np.abs(np.fft.rfft(segment * window)) ** 2)
    psd = np.mean(spectra, axis=0) if spectra else np.abs(np.fft.rfft(centered[:nfft] * window)) ** 2
    freq = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    db = 10.0 * np.log10(psd + 1e-12)
    usable = (freq >= 20.0) & (freq <= 12000.0)
    median_db = float(np.median(db[usable]))
    local = []
    usable_idxs = np.where(usable)[0]
    for idx in usable_idxs[1:-1]:
        if db[idx] > db[idx - 1] and db[idx] > db[idx + 1]:
            local.append(idx)
    peaks = sorted(local, key=lambda idx: db[idx], reverse=True)[:30]
    return {
        "sample_rate": sample_rate,
        "duration_s": len(data) / sample_rate,
        "rms_dbfs": 20.0 * math.log10(float(np.sqrt(np.mean(centered * centered))) / 32768.0),
        "median_spectrum_db": median_db,
        "strong_peaks": [
            {
                "freq_hz": round(float(freq[idx]), 3),
                "db": round(float(db[idx]), 3),
                "above_median_db": round(float(db[idx] - median_db), 3),
            }
            for idx in peaks
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input WAV path")
    parser.add_argument("--output", type=Path, help="Filtered WAV output path")
    parser.add_argument("--preset", choices=["episode25_fan", "none"], default="episode25_fan")
    parser.add_argument("--notch", action="append", type=parse_notch, default=[], help="Extra notch as center,width[,gain]")
    parser.add_argument("--lowpass", type=float, help="Optional smooth low-pass cutoff Hz")
    parser.add_argument("--highpass", type=float, help="Optional smooth high-pass cutoff Hz")
    parser.add_argument("--transition", type=float, default=700.0, help="Low/high-pass transition width Hz")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    sample_rate, data = read_wav(args.input)
    report = analyze(sample_rate, data)
    print(json.dumps(report, indent=2))
    if args.analyze_only:
        return 0

    output = args.output or args.input.with_name(args.input.stem + "_fan_filtered.wav")
    mean = float(np.mean(data))
    centered = data - mean
    freq = np.fft.rfftfreq(len(centered), 1.0 / sample_rate)
    spectrum = np.fft.rfft(centered)
    mask = np.ones_like(freq)
    notches = list(EPISODE25_FAN_NOTCHES if args.preset == "episode25_fan" else [])
    notches.extend(args.notch)
    if notches:
        mask *= notch_mask(freq, notches)
    if args.lowpass is not None:
        mask *= smooth_lowpass(freq, args.lowpass, args.transition)
    if args.highpass is not None:
        mask *= smooth_highpass(freq, args.highpass, args.transition)
    filtered = np.fft.irfft(spectrum * mask, n=len(centered)) + mean
    write_wav(output, sample_rate, filtered)
    print(f"wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
