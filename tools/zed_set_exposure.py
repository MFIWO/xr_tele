#!/usr/bin/env python3
"""Set / disable exposure (and related image settings) on a ZED camera via the ZED SDK.

Run this ON the machine that has the ZED connected (e.g. the robot's PC2).

Examples
--------
# Show current settings and exit:
    python zed_set_exposure.py --show

# Turn OFF auto exposure/gain and set a fixed exposure (0-100 %) and gain (0-100 %):
    python zed_set_exposure.py --exposure 20 --gain 30

# Turn auto exposure back ON:
    python zed_set_exposure.py --auto-exposure on

# Also fix white balance (2800-6500 K) and brightness (0-8):
    python zed_set_exposure.py --exposure 20 --wb 4600 --brightness 4

# Keep the camera open and streaming so the settings stay applied (Ctrl-C to stop):
    python zed_set_exposure.py --exposure 20 --hold

IMPORTANT
---------
* The ZED SDK opens the camera EXCLUSIVELY. teleimager grabs the ZED as a plain
  UVC/opencv device, so you cannot run teleimager and this script on the SAME
  camera at the same time (one will get CAMERA_NOT_DETECTED / device busy).
* SDK image settings are applied to the running camera session. When this script
  closes the camera and teleimager reopens it as UVC, the V4L2 defaults apply
  again. Use --hold to keep them applied, or use teleimager's own `controls:`
  (V4L2) block for the teleop pipeline.
"""

from __future__ import annotations

import argparse
import time
from typing import Any


# config-key -> ZED VIDEO_SETTINGS enum name, with valid range for the help text.
SETTING_INFO = {
    "exposure":   ("EXPOSURE", "0-100 (%)"),
    "gain":       ("GAIN", "0-100 (%)"),
    "brightness": ("BRIGHTNESS", "0-8"),
    "contrast":   ("CONTRAST", "0-8"),
    "hue":        ("HUE", "0-11"),
    "saturation": ("SATURATION", "0-8"),
    "sharpness":  ("SHARPNESS", "0-8"),
    "gamma":      ("GAMMA", "1-9"),
    "wb":         ("WHITEBALANCE_TEMPERATURE", "2800-6500 (K)"),
}


def _get_setting(cam: Any, sl: Any, enum_name: str):
    """Read a VIDEO_SETTINGS value, tolerant of SDK 3.x (value) vs 4.x ((err, value))."""
    setting = getattr(sl.VIDEO_SETTINGS, enum_name, None)
    if setting is None:
        return None
    res = cam.get_camera_settings(setting)
    if isinstance(res, tuple):  # SDK 4.x -> (ERROR_CODE, value)
        err, val = res
        if err != sl.ERROR_CODE.SUCCESS:
            return None
        return val
    return res  # SDK 3.x -> value


def _set_setting(cam: Any, sl: Any, enum_name: str, val: int) -> str:
    setting = getattr(sl.VIDEO_SETTINGS, enum_name, None)
    if setting is None:
        return f"unsupported ({enum_name} not in this SDK)"
    res = cam.set_camera_settings(setting, int(val))
    # SDK 4.x returns an ERROR_CODE; SDK 3.x returns None.
    if res is None or res == sl.ERROR_CODE.SUCCESS:
        return "ok"
    return str(getattr(res, "name", res))


def show_settings(cam: Any, sl: Any) -> None:
    aec = _get_setting(cam, sl, "AEC_AGC")
    print(f"  AEC_AGC (auto exposure/gain): {aec}  (1=auto ON, 0=manual)")
    wb_auto = _get_setting(cam, sl, "WHITEBALANCE_AUTO")
    print(f"  WHITEBALANCE_AUTO           : {wb_auto}  (1=auto ON, 0=manual)")
    for key, (enum_name, rng) in SETTING_INFO.items():
        val = _get_setting(cam, sl, enum_name)
        print(f"  {key:<11} ({enum_name:<26}) [{rng:<14}] = {val}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--serial", type=int, help="Open a specific ZED serial number")
    p.add_argument("--camera-id", type=int, help="Open a specific SDK camera id")
    p.add_argument("--timeout", type=float, default=5.0, help="open_timeout_sec")

    p.add_argument("--show", action="store_true", help="Print current settings and exit")
    p.add_argument(
        "--auto-exposure",
        choices=("on", "off"),
        help="Turn AEC_AGC (auto exposure+gain) on/off. "
        "Passing --exposure or --gain implies 'off'.",
    )
    p.add_argument("--exposure", type=int, help="Manual exposure 0-100 %% (implies auto off)")
    p.add_argument("--gain", type=int, help="Manual gain 0-100 %% (implies auto off)")
    p.add_argument("--brightness", type=int, help="0-8")
    p.add_argument("--contrast", type=int, help="0-8")
    p.add_argument("--hue", type=int, help="0-11")
    p.add_argument("--saturation", type=int, help="0-8")
    p.add_argument("--sharpness", type=int, help="0-8")
    p.add_argument("--gamma", type=int, help="1-9")
    p.add_argument(
        "--auto-wb", choices=("on", "off"), help="Turn auto white balance on/off"
    )
    p.add_argument("--wb", type=int, help="White balance temperature 2800-6500 K (implies auto-wb off)")

    p.add_argument(
        "--hold",
        action="store_true",
        help="Keep the camera open and grabbing so settings stay applied (Ctrl-C to exit)",
    )
    return p.parse_args()


def main() -> int:
    try:
        import pyzed.sl as sl
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to import pyzed.sl: {exc}")
        print("Install the ZED SDK + Python API on this machine first.")
        return 1

    args = parse_args()

    init = sl.InitParameters()
    init.sdk_verbose = 0
    init.open_timeout_sec = args.timeout
    depth_none = getattr(getattr(sl, "DEPTH_MODE", None), "NONE", None)
    if depth_none is not None:
        init.depth_mode = depth_none
    if args.serial is not None:
        init.set_from_serial_number(args.serial)
    elif args.camera_id is not None:
        init.set_from_camera_id(args.camera_id)

    cam = sl.Camera()
    err = cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED: {getattr(err, 'name', err)}")
        print("If teleimager is running, stop it first — it holds the camera (UVC).")
        return 2

    try:
        print(f"ZED SDK: {sl.Camera.get_sdk_version()}")
        info = cam.get_camera_information()
        print(f"Opened model={getattr(info, 'camera_model', '?')} "
              f"sn={getattr(info, 'serial_number', '?')}\n")

        if args.show:
            print("Current settings:")
            show_settings(cam, sl)
            return 0

        # --- decide auto-exposure state ---
        manual_exposure = args.exposure is not None or args.gain is not None
        if args.auto_exposure == "off" or manual_exposure:
            print(_report("AEC_AGC(auto exposure/gain) -> OFF",
                          _set_setting(cam, sl, "AEC_AGC", 0)))
        elif args.auto_exposure == "on":
            print(_report("AEC_AGC(auto exposure/gain) -> ON",
                          _set_setting(cam, sl, "AEC_AGC", 1)))

        # --- white balance auto state ---
        if args.auto_wb == "off" or args.wb is not None:
            print(_report("WHITEBALANCE_AUTO -> OFF",
                          _set_setting(cam, sl, "WHITEBALANCE_AUTO", 0)))
        elif args.auto_wb == "on":
            print(_report("WHITEBALANCE_AUTO -> ON",
                          _set_setting(cam, sl, "WHITEBALANCE_AUTO", 1)))

        # --- manual values ---
        for key, (enum_name, _rng) in SETTING_INFO.items():
            val = getattr(args, key)
            if val is not None:
                print(_report(f"{key} ({enum_name}) -> {val}",
                              _set_setting(cam, sl, enum_name, val)))

        print("\nSettings applied. Current state:")
        # A grab lets the SDK push settings to the sensor before we read them back.
        runtime = sl.RuntimeParameters()
        cam.grab(runtime)
        show_settings(cam, sl)

        if args.hold:
            print("\n--hold: keeping camera open so settings stay applied. Ctrl-C to stop.")
            try:
                while True:
                    cam.grab(runtime)
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nStopping.")
        else:
            print("\nNote: closing the camera now. If teleimager reopens it as UVC, "
                  "these SDK settings are replaced by the V4L2 defaults. Use --hold "
                  "to keep them, or teleimager's `controls:` block for the teleop stream.")
        return 0
    finally:
        cam.close()


def _report(action: str, status: str) -> str:
    return f"  {action}: {status}"


if __name__ == "__main__":
    raise SystemExit(main())
