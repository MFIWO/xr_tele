#!/usr/bin/env python3
"""Probe ZED SDK modes on the machine that has a connected ZED camera."""

from __future__ import annotations

import argparse
from typing import Any


DEFAULT_MODES = ("HD2K", "HD1200", "HD1080", "HD720", "SVGA", "VGA")
DEFAULT_FPS = (0, 15, 30, 60, 100, 120)


def value(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    try:
        attr = getattr(obj, name)
    except Exception:
        return default
    if callable(attr):
        try:
            return attr()
        except TypeError:
            return attr
    return attr


def enum_text(obj: Any) -> str:
    if obj is None:
        return "unknown"
    name = value(obj, "name")
    if name:
        return str(name)
    return str(obj)


def resolution_tuple(res: Any) -> tuple[int, int] | None:
    width = value(res, "width")
    height = value(res, "height")
    if width is None or height is None:
        return None
    try:
        return int(width), int(height)
    except (TypeError, ValueError):
        return None


def mat_shape(mat: Any) -> tuple[int, int] | None:
    width = value(mat, "get_width")
    height = value(mat, "get_height")
    if width is None or height is None:
        return None
    try:
        return int(width), int(height)
    except (TypeError, ValueError):
        return None


def configure_input(sl: Any, init: Any, args: argparse.Namespace) -> None:
    if args.serial is not None:
        init.set_from_serial_number(args.serial)
    elif args.camera_id is not None:
        init.set_from_camera_id(args.camera_id)


def configure_depth_none(sl: Any, init: Any) -> None:
    depth_none = getattr(getattr(sl, "DEPTH_MODE", None), "NONE", None)
    if depth_none is not None:
        init.depth_mode = depth_none


def print_devices(sl: Any) -> None:
    print(f"ZED SDK: {sl.Camera.get_sdk_version()}")
    devices = sl.Camera.get_device_list()
    if not devices:
        print("No ZED devices found by sl.Camera.get_device_list().")
        return

    print("Devices:")
    for idx, dev in enumerate(devices):
        fields = []
        for name in (
            "id",
            "serial_number",
            "camera_model",
            "input_type",
            "camera_state",
            "path",
        ):
            item = value(dev, name)
            if item is not None:
                fields.append(f"{name}={enum_text(item)}")
        print(f"  [{idx}] " + " ".join(fields))


def probe_once(sl: Any, mode_name: str, fps: int, args: argparse.Namespace) -> dict[str, Any]:
    mode = getattr(sl.RESOLUTION, mode_name, None)
    if mode is None:
        return {"mode": mode_name, "fps": fps, "status": "missing enum"}

    init = sl.InitParameters()
    init.camera_resolution = mode
    init.camera_fps = fps
    init.sdk_verbose = 0
    init.open_timeout_sec = args.timeout
    configure_input(sl, init, args)
    configure_depth_none(sl, init)

    cam = sl.Camera()
    err = cam.open(init)
    result: dict[str, Any] = {
        "mode": mode_name,
        "fps": fps,
        "status": enum_text(err),
    }

    if err == sl.ERROR_CODE.SUCCESS:
        try:
            info = cam.get_camera_information()
            cfg = value(info, "camera_configuration")
            res = resolution_tuple(value(cfg, "resolution"))
            actual_fps = value(cfg, "fps")
            result["model"] = enum_text(value(info, "camera_model"))
            result["serial"] = value(info, "serial_number")
            result["sdk_res"] = res
            result["actual_fps"] = actual_fps
            if res is not None:
                width, height = res
                result["side_by_side_guess"] = (width * 2, height)

            if args.grab:
                runtime = sl.RuntimeParameters()
                grab_err = cam.grab(runtime)
                result["grab"] = enum_text(grab_err)
                if grab_err == sl.ERROR_CODE.SUCCESS:
                    views: dict[str, tuple[int, int]] = {}
                    for view_name in ("LEFT", "RIGHT", "SIDE_BY_SIDE"):
                        view = getattr(sl.VIEW, view_name, None)
                        if view is None:
                            continue
                        mat = sl.Mat()
                        retrieve_err = cam.retrieve_image(mat, view)
                        if retrieve_err == sl.ERROR_CODE.SUCCESS:
                            shape = mat_shape(mat)
                            if shape is not None:
                                views[view_name] = shape
                    result["views"] = views
        finally:
            cam.close()

    return result


def format_result(row: dict[str, Any]) -> str:
    requested = f"{row['mode']:>6} fps={row['fps']:>3}"
    if row["status"] != "SUCCESS":
        return f"{requested} -> {row['status']}"

    parts = [f"{requested} -> OK"]
    if row.get("sdk_res"):
        width, height = row["sdk_res"]
        parts.append(f"sdk={width}x{height}")
    if row.get("side_by_side_guess"):
        width, height = row["side_by_side_guess"]
        parts.append(f"teleimager_sbs~={height}x{width}")
    if row.get("actual_fps") is not None:
        parts.append(f"actual_fps={row['actual_fps']}")
    if row.get("model"):
        parts.append(f"model={row['model']}")
    if row.get("serial") is not None:
        parts.append(f"sn={row['serial']}")
    if row.get("grab"):
        parts.append(f"grab={row['grab']}")
    if row.get("views"):
        view_text = ",".join(
            f"{name}:{width}x{height}"
            for name, (width, height) in sorted(row["views"].items())
        )
        parts.append(f"views={view_text}")
    return " ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", type=int, help="Open a specific ZED serial number")
    parser.add_argument("--camera-id", type=int, help="Open a specific SDK camera id")
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument("--fps", nargs="+", type=int, default=list(DEFAULT_FPS))
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--grab", action="store_true", help="Grab one frame and print retrieved view sizes")
    return parser.parse_args()


def main() -> int:
    try:
        import pyzed.sl as sl
    except Exception as exc:
        print(f"Failed to import pyzed.sl: {exc}")
        return 1

    args = parse_args()
    print_devices(sl)
    print()
    print("Mode probe:")
    for mode_name in args.modes:
        for fps in args.fps:
            print(format_result(probe_once(sl, mode_name, fps, args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
