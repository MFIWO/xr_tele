"""Thin policy-inference client: ship observations to the LeRobot policy server.

The robot PC has no GPU, so nothing about torch/lerobot lives here. This module only
serializes the robot observation (state vector + camera frames) onto ONE outgoing TCP
connection and hands back the action chunks the server predicts::

    teleop_hand_and_arm_policy_infer.py  --TCP:5601-->  policy_bridge_server.py (GPU PC)
                                                        └─ lerobot ACT / GR00T policy

Design constraints, same as ``loop_streamer.py``:
  * ``submit``/``pop_action`` never block or raise into the 20Hz control loop — the
    observation goes into a latest-wins slot and a background thread does all socket
    I/O (JPEG encoding of the camera buffers included).
  * A dead/absent server only means the action queue drains; the caller decides what
    to do about that (hold pose). The link auto-reconnects and re-sends the handshake.

Wire format (must match ``policy_bridge_server.py``, VERSION 1)::

    b"PB" | ver u8 | meta_len u32 | meta json | payload_len u32 | payload

Frames, all carrying a json ``meta`` with a ``type`` field:
  * ``hello``  client -> server: run config (task, camera keys, vector dims, fps)
  * ``ready``  server -> client: image size the policy wants, chunk size, dims
  * ``obs``    client -> server: {seq, t, state[N], cameras[...]}, payload = frames
  * ``act``    server -> client: {seq, n, dim, infer_ms}, payload = float32 n*dim
"""

from __future__ import annotations

import json
import os
import socket
import struct
import threading
import time

import numpy as np

try:
    import cv2
except ImportError:  # keep the module importable without OpenCV (raw encoding only)
    cv2 = None

try:
    import logging_mp as _logging
except ImportError:  # keep the module importable outside the teleop env
    import logging as _logging

logger_mp = _logging.getLogger(__name__)

WIRE_MAGIC = b"PB"
WIRE_VERSION = 1

AGGREGATE_FUNCTIONS = {
    # Same registry lerobot's async_inference client uses, so a chunk that overlaps
    # actions already queued blends instead of snapping.
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}


def encode_frame(sock, meta: dict, payload: bytes = b"") -> None:
    meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    sock.sendall(
        WIRE_MAGIC
        + bytes([WIRE_VERSION])
        + struct.pack(">I", len(meta_bytes))
        + meta_bytes
        + struct.pack(">I", len(payload))
    )
    if payload:
        sock.sendall(payload)


def _recv_exactly(sock, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError("peer closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def decode_frame(sock) -> tuple[dict, bytes]:
    header = _recv_exactly(sock, 3)
    if header[:2] != WIRE_MAGIC:
        raise ConnectionError(f"bad frame magic {header[:2]!r}")
    if header[2] != WIRE_VERSION:
        raise ConnectionError(f"unsupported wire version {header[2]}")
    (meta_len,) = struct.unpack(">I", _recv_exactly(sock, 4))
    meta = json.loads(_recv_exactly(sock, meta_len).decode("utf-8"))
    (payload_len,) = struct.unpack(">I", _recv_exactly(sock, 4))
    payload = _recv_exactly(sock, payload_len) if payload_len else b""
    return meta, payload


def parse_addr(addr: str, default_port: int = 5601) -> tuple[str, int]:
    if ":" in addr:
        host, _, port = addr.rpartition(":")
        return host or "127.0.0.1", int(port)
    return addr, default_port


class PolicyBridgeClient:
    """Latest-wins observation sender + action-chunk queue for one policy server."""

    def __init__(
        self,
        addr: str,
        task: str,
        cameras: list[str],
        state_dim: int,
        action_dim: int,
        fps: float = 20.0,
        actions_per_chunk: int = 0,
        chunk_size_threshold: float = 0.5,
        aggregate_fn_name: str = "weighted_average",
        image_encoding: str = "jpg",
        jpeg_quality: int = 95,
        connect_timeout: float = 2.0,
        reconnect_interval: float = 1.0,
    ):
        self.host, self.port = parse_addr(addr)
        self.task = task
        self.cameras = list(cameras)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.fps = float(fps)
        self.actions_per_chunk = int(actions_per_chunk)
        self.chunk_size_threshold = float(chunk_size_threshold)
        if aggregate_fn_name not in AGGREGATE_FUNCTIONS:
            raise ValueError(
                f"unknown aggregate function {aggregate_fn_name!r}; "
                f"available: {sorted(AGGREGATE_FUNCTIONS)}"
            )
        self.aggregate_fn = AGGREGATE_FUNCTIONS[aggregate_fn_name]
        self.aggregate_fn_name = aggregate_fn_name
        if image_encoding not in ("jpg", "raw"):
            raise ValueError("image_encoding must be 'jpg' or 'raw'")
        if image_encoding == "jpg" and cv2 is None:
            raise RuntimeError("image_encoding='jpg' needs OpenCV; install cv2 or use 'raw'")
        self.image_encoding = image_encoding
        self.jpeg_quality = int(jpeg_quality)
        self.connect_timeout = float(connect_timeout)
        self.reconnect_interval = float(reconnect_interval)

        # XR_POLICY_OBS_DUMP=<dir>: save every transmitted camera frame (post-resize,
        # exactly what goes on the wire) for visual inspection.
        self._dump_dir = os.environ.get("XR_POLICY_OBS_DUMP") or None
        if self._dump_dir:
            if cv2 is None:
                logger_mp.warning("[policy bridge] XR_POLICY_OBS_DUMP set but OpenCV is missing; dump disabled.")
                self._dump_dir = None
            else:
                os.makedirs(self._dump_dir, exist_ok=True)
                logger_mp.info("[policy bridge] dumping transmitted camera frames to %s", self._dump_dir)

        self._lock = threading.Lock()
        self._obs_slot = None            # (state ndarray, {cam: bgr ndarray}, capture_time)
        self._obs_event = threading.Event()
        self._stop = threading.Event()
        self._thread = None

        self._server_info = None         # 'ready' meta from the server
        self._image_size = None          # (height, width) the policy wants
        self._chunk_size = max(self.actions_per_chunk, 1)

        self._actions = {}               # timestep -> action ndarray
        self._step = 0                   # actions consumed so far
        self._inflight = {}              # obs seq -> base timestep
        self._seq = 0

        # telemetry
        self._sent = 0
        self._received = 0
        self._dropped = 0
        self._last_action_time = 0.0
        self._last_rtt_ms = 0.0
        self._last_infer_ms = 0.0
        self._connected = False

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="policy-bridge", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._obs_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def ready(self) -> bool:
        """True once the server accepted the handshake and announced its policy."""
        return self._server_info is not None

    @property
    def server_info(self) -> dict | None:
        return self._server_info

    @property
    def image_size(self) -> tuple[int, int] | None:
        return self._image_size

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    # -- control-loop API (never blocks) --------------------------------------

    def submit(self, state, images: dict) -> None:
        """Stage the freshest observation. Only references are copied here."""
        with self._lock:
            self._obs_slot = (np.asarray(state, dtype=np.float32).reshape(-1), images, time.time())
        self._obs_event.set()

    def hungry(self) -> bool:
        """True when the action queue ran low enough to warrant a new observation."""
        with self._lock:
            queued = len(self._actions)
        return queued / float(self._chunk_size) <= self.chunk_size_threshold

    def pop_action(self):
        """Return the next action to execute, or None when the queue ran dry."""
        with self._lock:
            if not self._actions:
                return None
            step = min(k for k in self._actions if k >= self._step)  # never replay the past
            action = self._actions.pop(step)
            for stale in [k for k in self._actions if k < step]:
                del self._actions[stale]
                self._dropped += 1
            self._step = step + 1
            return action

    def reset(self) -> None:
        """Drop every queued action; use on pause/resume so a stale chunk cannot fire."""
        with self._lock:
            self._actions.clear()
            self._inflight.clear()
            self._obs_slot = None

    def stats(self) -> dict:
        with self._lock:
            queued = len(self._actions)
        return {
            "connected": self._connected,
            "ready": self.ready,
            "queued": queued,
            "chunk_size": self._chunk_size,
            "sent": self._sent,
            "received": self._received,
            "dropped": self._dropped,
            "step": self._step,
            "action_age_s": (time.time() - self._last_action_time) if self._last_action_time else float("inf"),
            "rtt_ms": self._last_rtt_ms,
            "infer_ms": self._last_infer_ms,
        }

    # -- link ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            sock = None
            reader = None
            try:
                sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
                sock.settimeout(None)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                encode_frame(sock, {
                    "type": "hello",
                    "task": self.task,
                    "cameras": self.cameras,
                    "state_dim": self.state_dim,
                    "action_dim": self.action_dim,
                    "fps": self.fps,
                    "actions_per_chunk": self.actions_per_chunk,
                    "image_encoding": self.image_encoding,
                })
                meta, _ = decode_frame(sock)
                if meta.get("type") != "ready":
                    raise ConnectionError(f"expected a 'ready' frame, got {meta.get('type')!r}")
                self._apply_ready(meta)
                self._connected = True
                logger_mp.info(
                    "[policy bridge] connected to %s:%s policy=%s device=%s image=%s chunk=%s",
                    self.host, self.port, meta.get("policy"), meta.get("device"),
                    self._image_size, self._chunk_size,
                )
                reader = threading.Thread(target=self._read_loop, args=(sock,), daemon=True)
                reader.start()
                self._send_loop(sock)
            except (OSError, ConnectionError, ValueError, json.JSONDecodeError) as exc:
                if self._connected or self._sent == 0:
                    logger_mp.warning("[policy bridge] link to %s:%s down: %s", self.host, self.port, exc)
            finally:
                self._connected = False
                self._server_info = None
                if sock is not None:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    sock.close()
                if reader is not None:
                    reader.join(timeout=0.5)
                self.reset()
            if not self._stop.is_set():
                time.sleep(self.reconnect_interval)

    def _apply_ready(self, meta: dict) -> None:
        size = meta.get("image_size")
        self._image_size = (int(size[0]), int(size[1])) if size else None
        self._chunk_size = max(int(meta.get("chunk_size", self._chunk_size)), 1)
        server_dim = int(meta.get("action_dim", self.action_dim))
        if server_dim != self.action_dim:
            raise ConnectionError(
                f"server action_dim={server_dim} but this robot expects {self.action_dim}; "
                "the checkpoint does not match the recorded action layout"
            )
        server_state = int(meta.get("state_dim", self.state_dim))
        if server_state != self.state_dim:
            raise ConnectionError(
                f"server state_dim={server_state} but this robot builds {self.state_dim}"
            )
        server_cameras = list(meta.get("cameras", self.cameras))
        missing = [c for c in server_cameras if c not in self.cameras]
        if missing:
            raise ConnectionError(f"server wants camera keys this run does not produce: {missing}")
        # Only ship what the checkpoint consumes: the server drops any extra frame on
        # arrival, so encoding and sending the rest is pure bandwidth waste.
        dropped = [c for c in self.cameras if c not in server_cameras]
        if dropped:
            logger_mp.info(
                "[policy bridge] server only wants cameras %s; not sending %s", server_cameras, dropped
            )
            self.cameras = server_cameras
        self._server_info = meta

    def _send_loop(self, sock) -> None:
        while not self._stop.is_set():
            self._obs_event.wait(timeout=0.1)
            self._obs_event.clear()
            if self._stop.is_set():
                return
            if not self.hungry():
                continue
            with self._lock:
                if len(self._inflight) >= 2:
                    # The server is still working on the last two observations; sending
                    # more only queues staler inputs behind them.
                    continue
                slot = self._obs_slot
                self._obs_slot = None
                base_step = self._step
                self._seq += 1
                seq = self._seq
                self._inflight[seq] = base_step
            if slot is None:
                with self._lock:
                    self._inflight.pop(seq, None)
                continue
            state, images, capture_time = slot
            try:
                meta, payload = self._encode_observation(seq, state, images, capture_time)
            except (ValueError, KeyError) as exc:
                with self._lock:
                    self._inflight.pop(seq, None)
                logger_mp.warning("[policy bridge] observation dropped: %s", exc)
                continue
            encode_frame(sock, meta, payload)
            self._sent += 1

    def _encode_observation(self, seq, state, images, capture_time):
        if state.size != self.state_dim:
            raise ValueError(f"state has {state.size} dims, expected {self.state_dim}")
        parts = []
        cam_meta = []
        for name in self.cameras:
            frame = images.get(name)
            if frame is None:
                raise KeyError(f"camera {name!r} produced no frame")
            blob, shape, rgb = self._encode_image(frame)
            parts.append(blob)
            cam_meta.append({"name": name, "len": len(blob), "shape": list(shape)})
            if self._dump_dir:
                bgr_out = rgb[:, :, ::-1]
                cv2.imwrite(os.path.join(self._dump_dir, f"{name}_latest.jpg"), bgr_out)
                cv2.imwrite(os.path.join(self._dump_dir, f"{int(seq):05d}_{name}.jpg"), bgr_out)
        meta = {
            "type": "obs",
            "seq": int(seq),
            "t": float(capture_time),
            "enc": self.image_encoding,
            "state": [float(v) for v in state],
            "task": self.task,
            "cameras": cam_meta,
        }
        return meta, b"".join(parts)

    def _encode_image(self, bgr):
        """BGR frame -> wire bytes. The dataset stores RGB, so the flip happens here.

        Both encodings carry RGB: cv2.imencode/imdecode is channel-agnostic as long as
        both ends treat the array the same way, and the server decodes straight to RGB.
        """
        image = bgr
        if self._image_size is not None and image.shape[:2] != self._image_size:
            image = cv2.resize(
                image,
                (self._image_size[1], self._image_size[0]),
                interpolation=cv2.INTER_AREA,
            )
        rgb = np.ascontiguousarray(image[:, :, ::-1], dtype=np.uint8)
        if self.image_encoding == "raw":
            return rgb.tobytes(), rgb.shape, rgb
        ok, buf = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise ValueError("cv2.imencode failed")
        return buf.tobytes(), rgb.shape, rgb

    def _read_loop(self, sock) -> None:
        while not self._stop.is_set():
            try:
                meta, payload = decode_frame(sock)
            except (OSError, ConnectionError, ValueError, json.JSONDecodeError):
                return
            if meta.get("type") != "act":
                logger_mp.debug("[policy bridge] ignoring frame type=%s", meta.get("type"))
                continue
            self._on_actions(meta, payload)

    def _on_actions(self, meta: dict, payload: bytes) -> None:
        n = int(meta.get("n", 0))
        dim = int(meta.get("dim", self.action_dim))
        if n <= 0 or dim != self.action_dim:
            logger_mp.warning("[policy bridge] rejected chunk n=%s dim=%s", n, dim)
            return
        chunk = np.frombuffer(payload, dtype="<f4", count=n * dim).reshape(n, dim).astype(np.float64)
        seq = int(meta.get("seq", 0))
        now = time.time()
        with self._lock:
            base = self._inflight.pop(seq, None)
            if base is None:
                # Chunk for an observation from a previous connection / after a reset.
                return
            self._chunk_size = max(self._chunk_size, n)
            for i in range(n):
                step = base + i
                if step < self._step:
                    continue  # already executed; never rewind
                existing = self._actions.get(step)
                self._actions[step] = chunk[i] if existing is None else self.aggregate_fn(existing, chunk[i])
            self._received += 1
            self._last_action_time = now
            self._last_infer_ms = float(meta.get("infer_ms", 0.0))
            sent_at = float(meta.get("t_obs", 0.0))
            if sent_at:
                self._last_rtt_ms = (now - sent_at) * 1000.0
