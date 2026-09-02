#!/usr/bin/env python3
import argparse
import json
import os
import signal
import subprocess
import threading
import time

cv2 = None
zmq = None

_LOG_LOCK = threading.Lock()
_LOG_PATH = None  # --log-file 로 정해진다. None 이면 stdout 만.

_STATUS_LOCK = threading.Lock()
_STATUS = {}  # camera label -> 상태 dict (상태파일로 1초마다 나간다)

_CLAIM_LOCK = threading.Lock()
_CLAIMED_NODES = {}  # camera label -> /dev/videoN realpath currently opened by that thread
_PINNED_LINKS = set()  # all pinned by-path devices; the fallback must never take another camera's pinned port


def _log(msg):
    """stdout 과 로그파일에 함께 남긴다.

    터미널로만 내보내면 세션이 닫히는 순간 사라진다 — 실제로 재접속 때 무슨 해상도로
    붙었는지 사후에 확인할 수 없어 원인 조사가 막혔다. 그래서 파일에도 남긴다."""
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    if _LOG_PATH:
        try:
            with _LOG_LOCK:
                with open(_LOG_PATH, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except OSError:
            pass  # 로그를 못 써도 송신은 계속한다


def _status(label, **fields):
    with _STATUS_LOCK:
        _STATUS.setdefault(label, {}).update(fields)


def _status_writer(path, stop_event, stale_after=3.0, interval=1.0):
    """카메라 스레드와 **별개 스레드**에서 상태를 파일로 쓴다.

    별개 스레드인 것이 핵심이다. cap.read() 가 반환하지 않고 멈춰 버리면 그 카메라
    스레드는 아무것도 못 하지만, 이 스레드는 계속 돌면서 last_frame 이 갱신되지 않는 것을
    보고 state=stalled 로 찍어 준다. '장치는 잡고 있는데 프레임이 안 나오는' 상태가
    밖에서 보이게 된다."""
    while not stop_event.is_set():
        now = time.monotonic()
        with _STATUS_LOCK:
            cams = {}
            for label, st in _STATUS.items():
                d = dict(st)
                last = d.pop("_last_frame_mono", None)
                if last is not None:
                    d["stale_s"] = round(now - last, 1)
                    if d.get("state") == "streaming" and now - last > stale_after:
                        d["state"] = "stalled"
                cams[label] = d
        payload = {"ts": int(time.time()), "pid": os.getpid(), "cameras": cams}
        try:
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
            os.replace(tmp, path)  # 원자적 교체 — 읽는 쪽이 반쪽 파일을 보지 않는다
        except OSError:
            pass
        stop_event.wait(interval)


def _set_claim(label, node):
    with _CLAIM_LOCK:
        if node is None:
            _CLAIMED_NODES.pop(label, None)
        else:
            _CLAIMED_NODES[label] = os.path.realpath(node)


def _claimed_by_others(label):
    with _CLAIM_LOCK:
        return {node for other, node in _CLAIMED_NODES.items() if other != label}


def _usb_id_of_video_node(node):
    """Return 'vid:pid' of the USB device behind /dev/videoN, or None."""
    name = os.path.basename(os.path.realpath(node))
    iface = os.path.realpath(os.path.join("/sys/class/video4linux", name, "device"))
    usb_dev = os.path.dirname(iface)
    try:
        with open(os.path.join(usb_dev, "idVendor")) as f:
            vid = f.read().strip()
        with open(os.path.join(usb_dev, "idProduct")) as f:
            pid = f.read().strip()
    except OSError:
        return None
    return f"{vid}:{pid}"


def _fallback_same_model_node(label, usb_id):
    """The pinned USB port is empty (cable replugged into a different port?). If exactly ONE
    unclaimed camera with the expected USB id is present, use it. Ambiguous cases (0 or 2+
    candidates) fail instead, so left/right can never silently swap."""
    if not usb_id:
        return None
    by_path = "/dev/v4l/by-path"
    claimed = _claimed_by_others(label)
    candidates = []
    try:
        entries = sorted(os.listdir(by_path))
    except OSError:
        entries = []
    for entry in entries:
        # one entry per camera: capture node only, and skip the duplicate usbv2-* aliases
        if not entry.endswith("-video-index0") or "usbv2" in entry:
            continue
        link = os.path.join(by_path, entry)
        if link in _PINNED_LINKS:
            # A camera on a pinned port belongs to that port's label, even if its
            # thread has not claimed it yet (startup race) — never steal it.
            continue
        node = os.path.realpath(link)
        if node in claimed:
            continue
        if _usb_id_of_video_node(node) != usb_id:
            continue
        candidates.append((link, node))
    if len(candidates) == 1:
        return candidates[0]
    return None


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


def _resolve_device(device, physical_path, label, usb_id=None):
    """physical_path (USB-port-fixed) wins; otherwise accept a by-path symlink or a plain node.
    If a pinned by-path device is missing, fall back to the only unclaimed same-model camera."""
    if physical_path:
        node = _resolve_capture_node(physical_path)
        _log(f"[host_wrist_sender] {label} physical_path {physical_path} -> {node}")
        return node
    if device and os.path.islink(device):
        real = os.path.realpath(device)
        _log(f"[host_wrist_sender] {label} {device} -> {real}")
        return real
    if device and device.startswith("/dev/v4l/") and not os.path.exists(device):
        fallback = _fallback_same_model_node(label, usb_id)
        if fallback is not None:
            link, node = fallback
            _log(
                f"[host_wrist_sender] WARNING: {label} pinned port is empty ({device}); "
                f"falling back to the only other unclaimed same-model camera {link} -> {node}. "
                "VERIFY the left/right image mapping and re-pin the port in run_wrist_sender.sh!",
            )
            return node
        raise RuntimeError(
            f"{label} pinned device {device} is missing and no unambiguous same-model fallback camera was found"
        )
    return device


def _apply_v4l2_stream_controls(device):
    """재연결할 때마다 다시 건다 — 카메라를 다시 꽂으면 v4l2 컨트롤이 초기화된다.

    exposure_dynamic_framerate=0 은 **자동노출이 어두운 곳에서 프레임률을 떨어뜨리는 것**을
    막는다. 이게 풀리면 조명이 어두울 때 fps 가 조용히 절반으로 떨어진다.

    예전에는 set 만 하고 결과를 보지 않았다. v4l2-ctl 이 성공을 반환해도 카메라가
    무시할 수 있으므로 **되읽어 확인한다.** (성공 여부, 실제값) 을 돌려준다.
    """
    try:
        subprocess.run(
            ["v4l2-ctl", "-d", device, "--set-ctrl", "exposure_dynamic_framerate=0"],
            check=True, capture_output=True, timeout=2.0,
        )
    except Exception as exc:
        _log(f"[host_wrist_sender] WARNING: {device} exposure_dynamic_framerate 설정 실패: {exc}")
        return False, f"설정 실패: {exc}"
    try:
        r = subprocess.run(
            ["v4l2-ctl", "-d", device, "--get-ctrl", "exposure_dynamic_framerate"],
            capture_output=True, timeout=2.0, text=True, check=True,
        )
        val = r.stdout.strip().rsplit(":", 1)[-1].strip()
    except Exception as exc:
        _log(f"[host_wrist_sender] WARNING: {device} exposure_dynamic_framerate 확인 실패: {exc}")
        return False, f"확인 실패: {exc}"
    if val == "0":
        _log(f"[host_wrist_sender] {device} exposure_dynamic_framerate=0 (fixed fps) 확인됨")
        return True, val
    _log(f"[host_wrist_sender] WARNING: {device} exposure_dynamic_framerate 가 0 이 아니다 "
         f"(실제 {val}) — 어두운 곳에서 프레임률이 떨어질 수 있다")
    return False, val


def _open_capture(device, width, height, fps):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {device}")
    fixed_fps_ok, fixed_fps_val = _apply_v4l2_stream_controls(device)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f"opened {device}, but failed to read first frame")
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4))
    _log(
        f"[host_wrist_sender] opened {device} requested={width}x{height}@{fps} MJPG "
        f"negotiated={actual_width}x{actual_height}@{actual_fps:.2f} {fourcc}",
    )

    # 요청한 해상도로 열리지 않았으면 그대로 쓰지 않는다.
    # USB 재열거 직후 UVC probe control 이 -71 로 실패하면 드라이버가 협상 없이
    # 기본값(160x120)으로 스트림을 연다. 그대로 두면 Loop 저장물까지 흘러간다.
    # 드라이버 보고값과 실제로 들어온 첫 프레임을 모두 본다 — 둘이 다를 수 있다.
    frame_h, frame_w = frame.shape[:2]
    if (frame_w, frame_h) != (width, height) or (actual_width, actual_height) != (width, height):
        cap.release()
        raise RuntimeError(
            f"{device}: resolution negotiation failed - requested {width}x{height}, "
            f"got frame {frame_w}x{frame_h} (driver reports {actual_width}x{actual_height}, {fourcc})"
        )
    return cap, {"res": f"{actual_width}x{actual_height}", "fourcc": fourcc,
                 "negotiated_fps": round(actual_fps, 2),
                 "fixed_fps": fixed_fps_ok, "fixed_fps_val": fixed_fps_val}


def _open_capture_with_retry(label, device, physical_path, width, height, fps, stop_event, usb_id=None):
    """Resolve + open the capture, retrying on every failure until it succeeds or stop_event
    is set. Re-resolves the device each attempt, so a camera that comes back on a DIFFERENT
    /dev/videoN (after replug/re-enumeration) is still found via its USB-port physical_path.
    Returns (cap, resolved_device) or (None, None) if stopped."""
    attempt = 0
    while not stop_event.is_set():
        attempt += 1
        try:
            resolved = _resolve_device(device, physical_path, label, usb_id)
            cap, info = _open_capture(resolved, width, height, fps)
            if attempt > 1:
                _log(f"[host_wrist_sender] {label} reconnected (attempt {attempt}) on {resolved}")
            _status(label, state="streaming", device=resolved, fail_streak=0,
                    last_error=None, **info)
            return cap, resolved
        except Exception as exc:
            delay = min(0.5 * attempt, 5.0)
            _status(label, state="reopening", device=None, res=None, fourcc=None,
                    fps=0.0, fail_streak=attempt, last_error=str(exc))
            _log(
                f"[host_wrist_sender] {label} not available (attempt {attempt}): {exc}; retry in {delay:.1f}s",
            )
            stop_event.wait(delay)
    return None, None


def _camera_loop(name, device, physical_path, endpoint, width, height, fps, jpeg_quality, usb_id, stop_event):
    context = zmq.Context.instance()
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.setsockopt(zmq.LINGER, 0)
    if hasattr(zmq, "IMMEDIATE"):
        socket.setsockopt(zmq.IMMEDIATE, 1)
    socket.connect(endpoint)
    _log(f"[host_wrist_sender] {name} connect {endpoint}")

    period = 1.0 / max(float(fps), 1.0)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    reconnect_after_s = 1.5  # no good frame for this long => treat as disconnect and reopen

    cap = None
    frame_id = 0
    try:
        while not stop_event.is_set():
            if cap is None:
                cap, resolved = _open_capture_with_retry(
                    name, device, physical_path, width, height, fps, stop_event, usb_id
                )
                if cap is None:
                    break  # stop requested during retry
                _set_claim(name, resolved)
                sent = 0
                dropped = 0
                log_start = time.monotonic()
                next_time = time.monotonic()
                last_ok = time.monotonic()

            ok, frame = cap.read()
            if not ok or frame is None:
                dropped += 1
                if time.monotonic() - last_ok > reconnect_after_s:
                    _log(
                        f"[host_wrist_sender] {name} lost frames for >{reconnect_after_s:.1f}s; "
                        "camera likely dropped off USB. Reconnecting...",
                    )
                    cap.release()
                    cap = None
                    _set_claim(name, None)
                    continue
                time.sleep(0.02)
                continue
            last_ok = time.monotonic()
            _status(name, _last_frame_mono=last_ok)

            # 열린 뒤 재열거로 크기가 바뀌는 경우 — 열 때 검증만으로는 못 막는다.
            if frame.shape[1] != width or frame.shape[0] != height:
                dropped += 1
                _log(
                    f"[host_wrist_sender] {name} frame size changed to "
                    f"{frame.shape[1]}x{frame.shape[0]} (expected {width}x{height}); reopening",
                )
                _status(name, state="reopening", res=f"{frame.shape[1]}x{frame.shape[0]}",
                        last_error="frame size changed mid-stream")
                cap.release()
                cap = None
                _set_claim(name, None)
                continue

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
                _log(
                    f"[host_wrist_sender] {name} fps={sent / elapsed:.1f} "
                    f"send_ok={sent > 0} dropped={dropped}",
                )
                _status(name, fps=round(sent / elapsed, 1), dropped=dropped,
                        frames_total=frame_id)
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
        _set_claim(name, None)
        socket.close()
        _status(name, state="stopped", fps=0.0)
        _log(f"[host_wrist_sender] {name} stopped")


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
    parser.add_argument(
        "--camera-usb-id",
        default="0edc:2b10",
        help="vid:pid of the wrist cameras. When a pinned port is empty, the sender may fall back "
             "to the ONLY other unclaimed camera with this id (replugged into a different port). "
             "Pass an empty string to disable the fallback.",
    )
    parser.add_argument(
        "--log-file",
        default=os.path.expanduser("~/dfws/logs/wrist_sender_%Y%m%d.log"),
        help="로그를 남길 파일(strftime 서식 사용 가능). 빈 문자열이면 stdout 만 쓴다. "
             "터미널로만 내보내면 세션이 닫힐 때 사라져 사후 원인 조사가 불가능하다.",
    )
    parser.add_argument(
        "--state-file",
        default=os.path.expanduser("~/dfws/logs/wrist_sender.state"),
        help="카메라별 상태를 1초마다 쓰는 JSON 파일. 빈 문자열이면 쓰지 않는다. "
             "수집 시작 전에 두 손목캠이 정상인지 이걸로 확인한다.",
    )
    parser.add_argument(
        "--stale-after", type=float, default=3.0,
        help="이 시간(초) 넘게 새 프레임이 없으면 상태를 stalled 로 표시한다.",
    )
    args = parser.parse_args()

    global _LOG_PATH
    if args.log_file:
        _LOG_PATH = time.strftime(args.log_file)
        try:
            os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
        except OSError:
            _LOG_PATH = None

    global cv2, zmq
    import cv2 as _cv2
    import zmq as _zmq
    cv2 = _cv2
    zmq = _zmq

    for pinned in (args.left_device, args.right_device):
        if pinned and pinned.startswith("/dev/v4l/"):
            _PINNED_LINKS.add(pinned)

    stop_event = threading.Event()

    def _stop(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    _log(f"[host_wrist_sender] start pid={os.getpid()} "
         f"target={args.width}x{args.height}@{args.fps} "
         f"fallback={'on' if args.camera_usb_id else 'OFF'}")

    if args.state_file:
        try:
            os.makedirs(os.path.dirname(args.state_file), exist_ok=True)
            threading.Thread(
                target=_status_writer,
                args=(args.state_file, stop_event, args.stale_after),
                daemon=True,
            ).start()
            _log(f"[host_wrist_sender] state file {args.state_file}")
        except OSError as exc:
            _log(f"[host_wrist_sender] WARNING: state file disabled: {exc}")

    left_endpoint = f"tcp://{args.robot_ip}:{args.left_port}"
    right_endpoint = f"tcp://{args.robot_ip}:{args.right_port}"
    threads = [
        threading.Thread(
            target=_camera_loop,
            args=("left_wrist", args.left_device, args.left_physical_path, left_endpoint, args.width, args.height, args.fps, args.jpeg_quality, args.camera_usb_id, stop_event),
            daemon=True,
        ),
        threading.Thread(
            target=_camera_loop,
            args=("right_wrist", args.right_device, args.right_physical_path, right_endpoint, args.width, args.height, args.fps, args.jpeg_quality, args.camera_usb_id, stop_event),
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
