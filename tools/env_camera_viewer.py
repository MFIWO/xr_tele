#!/usr/bin/env python3
import argparse
import os
import signal
import threading
import time

cv2 = None


def _normalize_device(device):
    if isinstance(device, str):
        if device.isdigit():
            return f"/dev/video{device}"
        if device.startswith("video") and device[5:].isdigit():
            return f"/dev/{device}"
    return device


def _is_rgb_capture(device):
    """True if the node yields a 3-channel frame, not a metadata node."""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return False
    ok, frame = cap.read()
    cap.release()
    return bool(ok and frame is not None and getattr(frame, "ndim", 0) == 3)


def _resolve_capture_node(physical_path):
    """Return the RGB-capable /dev/videoN bound to a fixed USB sysfs path."""
    base = "/sys/class/video4linux"

    def _idx(name):
        tail = name.replace("video", "")
        return int(tail) if tail.isdigit() else 1_000_000

    candidates = []
    for name in sorted(os.listdir(base), key=_idx):
        dev = os.path.join(base, name, "device")
        try:
            if os.path.realpath(dev) == physical_path:
                candidates.append(f"/dev/{name}")
        except OSError:
            continue
    if not candidates:
        raise RuntimeError(f"no /dev/video* node found for physical_path {physical_path}")
    for node in candidates:
        if _is_rgb_capture(node):
            return node
    return candidates[0]


def _resolve_device(device, physical_path, label):
    """physical_path wins; otherwise accept a by-path symlink or a plain node."""
    if physical_path:
        node = _resolve_capture_node(physical_path)
        print(f"[env_camera_viewer] {label} physical_path {physical_path} -> {node}", flush=True)
        return node

    device = _normalize_device(device)
    if device and os.path.islink(device):
        real = os.path.realpath(device)
        print(f"[env_camera_viewer] {label} {device} -> {real}", flush=True)
        return real
    return device


def _open_capture(device, width, height, fps):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(f"failed to open {device}")

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f"opened {device}, but failed to read first frame")

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4))
    print(
        f"[env_camera_viewer] opened {device} requested={width}x{height}@{fps} MJPG "
        f"negotiated={actual_width}x{actual_height}@{actual_fps:.2f} {fourcc}",
        flush=True,
    )
    return cap


def _open_capture_with_retry(label, device, physical_path, width, height, fps, stop_event):
    attempt = 0
    while not stop_event.is_set():
        attempt += 1
        try:
            resolved = _resolve_device(device, physical_path, label)
            cap = _open_capture(resolved, width, height, fps)
            if attempt > 1:
                print(f"[env_camera_viewer] {label} reconnected (attempt {attempt}) on {resolved}", flush=True)
            return cap, resolved
        except Exception as exc:
            delay = min(0.5 * attempt, 5.0)
            print(
                f"[env_camera_viewer] {label} not available (attempt {attempt}): {exc}; "
                f"retry in {delay:.1f}s",
                flush=True,
            )
            stop_event.wait(delay)
    return None, None


def _put_lines(frame, lines, origin=(12, 24), line_height=22, color=(255, 255, 255)):
    x, y = origin
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )

def _fit_height(frame, target_height):
    if target_height <= 0 or frame.shape[0] == target_height:
        return frame
    scale = target_height / float(frame.shape[0])
    target_width = max(1, int(round(frame.shape[1] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, (target_width, target_height), interpolation=interpolation)


def _ensure_bgr(frame):
    if getattr(frame, "ndim", 0) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if getattr(frame, "ndim", 0) == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame

def _draw_overlay(frame, label, resolved_device, read_fps, dropped):
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 58), (0, 0, 0), -1)
    _put_lines(
        out,
        [
            f"{label} {resolved_device}",
            f"fps={read_fps:.1f} dropped={dropped}",
        ],
        origin=(10, 21),
        line_height=24,
    )
    return out


def main():
    parser = argparse.ArgumentParser(description="Open /dev/video5 and display it live.")
    parser.add_argument("--device", default="/dev/video5")
    parser.add_argument(
        "--physical-path",
        default=None,
        help="sysfs device path for the camera USB port. Overrides --device.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--display-height", type=int, default=480)
    parser.add_argument("--no-overlay", action="store_true")
    args = parser.parse_args()

    global cv2
    import cv2 as _cv2

    cv2 = _cv2

    stop_event = threading.Event()

    def _stop(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print("[env_camera_viewer] press q or Esc to quit", flush=True)
    cv2.namedWindow("video5", cv2.WINDOW_NORMAL)

    period = 1.0 / max(float(args.fps), 1.0)
    reconnect_after_s = 1.5
    cap = None
    resolved = None

    try:
        while not stop_event.is_set():
            if cap is None:
                cap, resolved = _open_capture_with_retry(
                    "video5",
                    args.device,
                    args.physical_path,
                    args.width,
                    args.height,
                    args.fps,
                    stop_event,
                )
                if cap is None:
                    break
                shown = 0
                dropped = 0
                log_start = time.monotonic()
                next_time = time.monotonic()
                last_ok = time.monotonic()

            ok, frame = cap.read()
            if not ok or frame is None:
                dropped += 1
                if time.monotonic() - last_ok > reconnect_after_s:
                    print(
                        f"[env_camera_viewer] video5 lost frames for >{reconnect_after_s:.1f}s; "
                        "reconnecting...",
                        flush=True,
                    )
                    cap.release()
                    cap = None
                    continue
                time.sleep(0.02)
                continue

            last_ok = time.monotonic()
            shown += 1

            now = time.monotonic()
            elapsed = max(now - log_start, 1e-9)
            read_fps = shown / elapsed
            frame = _ensure_bgr(frame)
            if not args.no_overlay:
                frame = _draw_overlay(frame, "video5", resolved, read_fps, dropped)
            frame = _fit_height(frame, args.display_height)
            cv2.imshow("video5", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                stop_event.set()

            if now - log_start >= 1.0:
                print(
                    f"[env_camera_viewer] video5 fps={read_fps:.1f} dropped={dropped}",
                    flush=True,
                )
                shown = 0
                dropped = 0
                log_start = now

            next_time += period
            sleep_time = next_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.monotonic()
    finally:
        stop_event.set()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("[env_camera_viewer] video5 stopped", flush=True)


if __name__ == "__main__":
    main()
