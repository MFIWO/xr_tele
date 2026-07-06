#!/usr/bin/env python3
"""Inspect host audio inputs for teleop audio recording."""

from __future__ import annotations

import argparse
import shutil
import subprocess


def run_command(cmd: list[str], timeout: float = 4.0) -> tuple[int | None, str]:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return None, f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        return None, output.strip() or f"{cmd[0]}: timed out"
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode, output.strip()


def print_sounddevice_info() -> None:
    print("== sounddevice query_devices() ==")
    try:
        import sounddevice as sd
    except ImportError as exc:
        print(f"sounddevice unavailable: {exc}")
        return

    try:
        print(sd.query_devices())
        print()
        print("== sounddevice defaults ==")
        print(f"default.device: {sd.default.device}")
        input_device = sd.default.device[0]
        if input_device is not None and int(input_device) >= 0:
            info = sd.query_devices(input_device, "input")
            print(f"default input index: {input_device}")
            print(f"default samplerate: {info.get('default_samplerate')}")
            print(f"max input channels: {info.get('max_input_channels')}")
        else:
            print("default input index: none")
    except Exception as exc:
        print(f"sounddevice query failed: {exc}")


def print_arecord_info(device: str) -> None:
    print()
    print("== ALSA arecord -l ==")
    if shutil.which("arecord") is None:
        print("arecord unavailable: executable not found")
        return

    _, output = run_command(["arecord", "-l"])
    print(output or "(no output)")

    print()
    print("== ALSA recommended device check ==")
    print(f"recommended device string: {device}")
    cmd = [
        "arecord",
        "-D",
        device,
        "-f",
        "S16_LE",
        "-r",
        "48000",
        "-c",
        "1",
        "-d",
        "1",
        "-t",
        "raw",
        "--dump-hw-params",
        "/dev/null",
    ]
    code, params = run_command(cmd, timeout=5.0)
    if code == 0:
        print("recommended format check: ok")
    elif code is None:
        print("recommended format check: unavailable")
    else:
        print(f"recommended format check: failed with exit code {code}")
    if params:
        print(params)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recommended-device", default="plughw:2,0", help="Device string to print and probe.")
    args = parser.parse_args()

    print_sounddevice_info()
    print_arecord_info(args.recommended_device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
