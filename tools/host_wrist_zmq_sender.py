#!/usr/bin/env python3
import argparse
import json
import os
import signal
import threading
import time

cv2 = None
zmq = None


def _is_rgb_capture(device):
    """True if the node yields a 3-channel frame (i.e. a real capture node, not a metadata node)."""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return False
    ok, frame = cap.read()
    cap.release()
    return bool(ok and frame is not None and getattr(frame, "ndim", 0) == 3)


def _resolve_capture_node(physical_path):
    """Return the RGB-capable /dev/videoN bound to a fixed USB port (sysfs device path).
    Stable across reboot/replug, unlike /dev/videoN numbering. Mirrors the image_server
    cam_config_server.yaml `physical_path` resolution."""
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
    # probe inconclusive: fall back to the lowest-numbered node (usually the capture node)
    return candidates[0]


def _resolve_device(device, physical_path, label):
    """physical_path (USB-port-fixed) wins; otherwise accept a by-path symlink or a plain node."""
    if physical_path:
        node = _resolve_capture_node(physical_path)
        print(f"[host_wrist_sender] {label} physical_path {physical_path} -> {node}", flush=True)
        return node
    if device and os.path.islink(device):
        real = os.path.realpath(device)
        print(f"[host_wrist_sender] {label} {device} -> {real}", flush=True)
        return real
    return device


def _open_capture(device, width, height, fps):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
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
        f"[host_wrist_sender] opened {device} requested={width}x{height}@{fps} MJPG "
        f"negotiated={actual_width}x{actual_height}@{actual_fps:.2f} {fourcc}",
        flush=True,
    )
    return cap


def _open_capture_with_retry(label, device, physical_path, width, height, fps, stop_event):
    """Resolve + open the capture, retrying on every failure until it succeeds or stop_event
    is set. Re-resolves the device each attempt, so a camera that comes back on a DIFFERENT
    /dev/videoN (after replug/re-enumeration) is still found via its USB-port physical_path.
    Returns (cap, resolved_device) or (None, None) if stopped."""
    attempt = 0
    while not stop_event.is_set():
        attempt += 1
        try:
            resolved = _resolve_device(device, physical_path, label)
            cap = _open_capture(resolved, width, height, fps)
            if attempt > 1:
                print(f"[host_wrist_sender] {label} reconnected (attempt {attempt}) on {resolved}", flush=True)
            return cap, resolved
        except Exception as exc:
            delay = min(0.5 * attempt, 5.0)
            print(
                f"[host_wrist_sender] {label} not available (attempt {attempt}): {exc}; retry in {delay:.1f}s",
                flush=True,
            )
            stop_event.wait(delay)
    return None, None


def _camera_loop(name, device, physical_path, endpoint, width, height, fps, jpeg_quality, stop_event):
    context = zmq.Context.instance()
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.setsockopt(zmq.LINGER, 0)
    if hasattr(zmq, "IMMEDIATE"):
        socket.setsockopt(zmq.IMMEDIATE, 1)
    socket.connect(endpoint)
    print(f"[host_wrist_sender] {name} connect {endpoint}", flush=True)

    period = 1.0 / max(float(fps), 1.0)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    reconnect_after_s = 1.5  # no good frame for this long => treat as disconnect and reopen

    cap = None
    frame_id = 0
    try:
        while not stop_event.is_set():
            if cap is None:
                cap, _resolved = _open_capture_with_retry(
                    name, device, physical_path, width, height, fps, stop_event
                )
                if cap is None:
                    break  # stop requested during retry
                sent = 0
                dropped = 0
                log_start = time.monotonic()
                next_time = time.monotonic()
                last_ok = time.monotonic()

            ok, frame = cap.read()
            if not ok or frame is None:
                dropped += 1
                if time.monotonic() - last_ok > reconnect_after_s:
                    print(
                        f"[host_wrist_sender] {name} lost frames for >{reconnect_after_s:.1f}s; "
                        "camera likely dropped off USB. Reconnecting...",
                        flush=True,
                    )
                    cap.release()
                    cap = None
                    continue
                time.sleep(0.02)
                continue
            last_ok = time.monotonic()

            timestamp_ns = time.time_ns()
            ok, jpg = cv2.imencode(".jpg", frame, encode_param)
            if not ok:
                dropped += 1
                continue

            meta = {
                "camera_name": name,
                "timestamp_ns": timestamp_ns,
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
                "encoding": "jpg",
                "frame_id": frame_id,
            }
            try:
                socket.send_multipart(
                    [json.dumps(meta, separators=(",", ":")).encode("utf-8"), jpg.tobytes()],
                    flags=zmq.NOBLOCK,
                )
                sent += 1
                frame_id += 1
            except zmq.Again:
                dropped += 1

            now = time.monotonic()
            if now - log_start >= 1.0:
                elapsed = now - log_start
                print(
                    f"[host_wrist_sender] {name} fps={sent / elapsed:.1f} "
                    f"send_ok={sent > 0} dropped={dropped}",
                    flush=True,
                )
                sent = 0
                dropped = 0
                log_start = now

            next_time += period
            sleep_time = next_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.monotonic()
    finally:
        if cap is not None:
            cap.release()
        socket.close()
        print(f"[host_wrist_sender] {name} stopped", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-device", default="/dev/video0")
    parser.add_argument("--right-device", default="/dev/video2")
    parser.add_argument(
        "--left-physical-path",
        default=None,
        help="sysfs device path of the LEFT wrist camera's USB port. Overrides --left-device and is "
             "STABLE across reboot/replug (unlike /dev/videoN). "
             "e.g. /sys/devices/pci0000:80/0000:80:14.0/usb3/3-5/3-5.1/3-5.1.2/3-5.1.2:1.0",
    )
    parser.add_argument(
        "--right-physical-path",
        default=None,
        help="sysfs device path of the RIGHT wrist camera's USB port. Overrides --right-device.",
    )
    parser.add_argument("--robot-ip", default="192.168.123.164")
    parser.add_argument("--left-port", type=int, default=55602)
    parser.add_argument("--right-port", type=int, default=55604)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    args = parser.parse_args()

    global cv2, zmq
    import cv2 as _cv2
    import zmq as _zmq
    cv2 = _cv2
    zmq = _zmq

    stop_event = threading.Event()

    def _stop(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    left_endpoint = f"tcp://{args.robot_ip}:{args.left_port}"
    right_endpoint = f"tcp://{args.robot_ip}:{args.right_port}"
    threads = [
        threading.Thread(
            target=_camera_loop,
            args=("left_wrist", args.left_device, args.left_physical_path, left_endpoint, args.width, args.height, args.fps, args.jpeg_quality, stop_event),
            daemon=True,
        ),
        threading.Thread(
            target=_camera_loop,
            args=("right_wrist", args.right_device, args.right_physical_path, right_endpoint, args.width, args.height, args.fps, args.jpeg_quality, stop_event),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    while not stop_event.is_set() and all(thread.is_alive() for thread in threads):
        time.sleep(0.2)
    stop_event.set()
    for thread in threads:
        thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
