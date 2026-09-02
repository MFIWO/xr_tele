"""Thin Config Loop client: ship teleop data to the loop_porting_kit sidecar.

This file REPLACES the original in-process Loop streamer (loop-sdk + RTSP
servers inside the teleop process). The heavy lifting now lives in the
standalone sidecar (``loop_porting_kit/loop_bridge/run_loop_streamer.py``,
normally on the host next to the Loop desktop app); this module only
serializes robot state / tactile / camera frames onto ONE outgoing TCP
connection. The repo therefore no longer depends on loop-sdk, grpc, or
anything beyond the Python stdlib for its ``--loop`` feature.

    teleop (--loop)  --TCP:5590-->  loop_porting_kit sidecar  --gRPC/RTSP-->  Loop

Public API is unchanged (`LoopRobotStreamer`, `LoopHandStreamer`,
`LoopCameraStreamer`, `ee_dim_per_hand`, ...), so ``teleop_hand_and_arm.py``
and ``loop_smoke_test.py`` run as before. The sidecar address comes from the
``LOOP_BRIDGE_ADDR`` env var (default ``127.0.0.1:5590``; the sidecar runs on
the host and the container uses host networking, so localhost is right).

Design constraints kept from the original module:
  * ``send``/``set_*`` never block or raise into the 30Hz control loop —
    frames go into latest-wins slots and a background thread does all socket
    I/O (serialization of big camera buffers included).
  * A dead/absent sidecar only means dropped frames; teleop is unaffected and
    the link auto-reconnects, re-sending the config frames first.

Wire format (must match ``loop_porting_kit/loop_bridge/wire.py``, VERSION 1):
    b"LB" | ver u8 | topic_len u8 | topic | meta_len u32 | meta json | payload_len u32 | payload
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import struct
import threading
import time

try:
    import logging_mp as _logging
except ImportError:  # keep the module importable outside the teleop env
    import logging as _logging

logger_mp = _logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# End-effector dimensions (per hand) — unchanged from the original module; the
# repo-side split of the concatenated left+right EE arrays still happens here.
# ----------------------------------------------------------------------------
_EE_DIM_PER_HAND = {
    "dex3": 7,
    "dex1": 1,
    "inspire_dfx": 6,
    "inspire_ftp": 6,
    "brainco": 6,
    "rh5dg2_dfx": 13,
    "rh5dg2_ftp": 13,
    "inspire_dg2": 13,
    "rh56f1": 12,
}


def ee_dim_per_hand(ee: str | None) -> int:
    """Per-hand end-effector channel count for the given ``--ee`` selection."""
    return _EE_DIM_PER_HAND.get(ee or "", 0)


_EE_GRIPPER = {"dex1"}


def ee_config_axes(ee: str | None) -> dict[str, tuple[str, ...]]:
    """Optional Loop robot-config axes to advertise for the given ``--ee``."""
    if not ee:
        return {}
    if ee in _EE_GRIPPER:
        return {"gripper_type": (ee,)}
    return {"finger_type": (ee,)}


_ARM_ROOT_KEYS = {
    "G1_29": "g1",
    "G1_23": "g1",
    "H1_2": "h1_2",
    "H1": "h1",
    "H2": "h2",
}


def arm_root_key(arm: str | None) -> str:
    """Top-level robot key used in the Config Loop robot step."""
    return _ARM_ROOT_KEYS.get(arm or "", "g1")


def _floats(values) -> list[float]:
    return [float(v) for v in values] if values is not None else []


# ============================================================================
# Bridge link (wire format V1 — keep in sync with loop_bridge/wire.py)
# ============================================================================
_MAGIC = b"LB"
_VERSION = 1
_HEADER = struct.Struct("!2sBB")
_U32 = struct.Struct("!I")

_DEFAULT_BRIDGE_ADDR = "127.0.0.1:5590"


def _encode_frame(topic: str, meta: dict, payload: bytes = b"") -> bytes:
    topic_b = topic.encode("ascii")
    meta_b = json.dumps(meta, separators=(",", ":"), default=float).encode("utf-8")
    return b"".join((
        _HEADER.pack(_MAGIC, _VERSION, len(topic_b)),
        topic_b,
        _U32.pack(len(meta_b)),
        meta_b,
        _U32.pack(len(payload)),
        payload,
    ))


class _BridgeLink:
    """One shared TCP connection to the sidecar, fed by latest-wins slots.

    ``publish`` only swaps references under a lock; the sender thread encodes
    (including numpy ``.tobytes()`` for camera frames) and writes the socket.
    Slots are keyed by topic, so a stalled sidecar costs at most one pending
    frame per topic — never unbounded memory, never a blocked control loop.
    """

    _instance: "_BridgeLink | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def acquire(cls, addr: str | None = None) -> "_BridgeLink":
        with cls._instance_lock:
            requested_addr = os.environ.get("LOOP_BRIDGE_ADDR", addr or _DEFAULT_BRIDGE_ADDR)
            if cls._instance is None or not cls._instance._running:
                cls._instance = cls(requested_addr)
            elif requested_addr != f"{cls._instance._addr[0]}:{cls._instance._addr[1]}":
                logger_mp.warning(
                    "[loop] bridge already connected to %s:%s; ignoring requested address %s",
                    cls._instance._addr[0],
                    cls._instance._addr[1],
                    requested_addr,
                )
            cls._instance._refs += 1
            return cls._instance

    def __init__(self, addr: str) -> None:
        host, _, port = addr.rpartition(":")
        self._addr = (host or "127.0.0.1", int(port))
        self._cond = threading.Condition()
        self._configs: dict[str, dict] = {}     # topic -> meta, re-sent on reconnect
        self._configs_dirty = True
        self._slots: dict[str, tuple] = {}      # topic -> (meta, payload_source)
        self._refs = 0
        self._running = True
        self._sock: socket.socket | None = None
        self._sent: dict[str, int] = {}
        self._connected_once = False
        self._warned_down = False
        self._thread = threading.Thread(target=self._run, name="loop-bridge-tx", daemon=True)
        self._thread.start()

    # ---------------------------------------------------------------- API
    def register_config(self, topic: str, meta: dict) -> None:
        with self._cond:
            self._configs[topic] = meta
            self._configs_dirty = True
            self._cond.notify()

    def publish(self, topic: str, meta: dict, payload_source=None) -> None:
        """payload_source: None | bytes | ("bgr", ndarray) | ("jpg", buffer)."""
        with self._cond:
            self._slots[topic] = (meta, payload_source)
            self._cond.notify()

    def stats(self) -> dict:
        with self._cond:
            return {"bridge": f"{self._addr[0]}:{self._addr[1]}",
                    "connected": self._sock is not None, "sent": dict(self._sent)}

    def release(self) -> None:
        with self._instance_lock:
            self._refs -= 1
            if self._refs > 0:
                return
            _BridgeLink._instance = None
        with self._cond:
            self._running = False
            self._cond.notify()
        self._thread.join(timeout=3.0)

    # ------------------------------------------------------------- sender
    def _run(self) -> None:
        backoff = 0.5
        while True:
            with self._cond:
                while self._running and not self._slots and not self._configs_dirty:
                    self._cond.wait()
                if not self._running:
                    break
                batch = self._slots
                self._slots = {}
                configs = dict(self._configs) if self._configs_dirty else None
                self._configs_dirty = False
            try:
                if self._sock is None:
                    self._sock = socket.create_connection(self._addr, timeout=3.0)
                    self._sock.settimeout(10.0)
                    with self._cond:  # (re)send configs on connect, incl. any registered
                        configs = dict(self._configs)  # while the connection was being set up
                        self._configs_dirty = False
                    if not self._connected_once:
                        self._connected_once = True
                        logger_mp.info(f"[loop] bridge link up ({self._addr[0]}:{self._addr[1]})")
                    self._warned_down = False
                    backoff = 0.5
                if configs:
                    for topic, meta in configs.items():
                        self._send_frame(topic, meta, None)
                for topic, (meta, payload_source) in batch.items():
                    self._send_frame(topic, meta, payload_source)
            except Exception as exc:
                if self._sock is not None:
                    with contextlib.suppress(Exception):
                        self._sock.close()
                    self._sock = None
                with self._cond:
                    self._configs_dirty = True  # resend on next connect
                if not self._warned_down:
                    logger_mp.warning(
                        f"[loop] bridge unreachable at {self._addr[0]}:{self._addr[1]} "
                        f"(dropping frames, retrying quietly): {exc}"
                    )
                    self._warned_down = True
                # simple backoff so a downed sidecar isn't hammered
                time.sleep(min(backoff, 5.0))
                backoff = min(backoff * 2, 5.0)
        if self._sock is not None:
            with contextlib.suppress(Exception):
                self._sock.close()
            self._sock = None

    def _send_frame(self, topic: str, meta: dict, payload_source) -> None:
        payload = b""
        if payload_source is not None:
            kind, buf = payload_source
            if kind == "jpg":
                payload = buf.tobytes() if hasattr(buf, "tobytes") else bytes(buf)
            else:  # "bgr": ndarray -> raw bytes (C-order), shape already in meta
                payload = buf.tobytes()
        self._sock.sendall(_encode_frame(topic, meta, payload))
        self._sent[topic] = self._sent.get(topic, 0) + 1


# ============================================================================
# Public streamers (same API as the original in-process implementations)
# ============================================================================
class LoopRobotStreamer:
    """Same call surface as the original; forwards ticks to the sidecar."""

    def __init__(
        self,
        loop_addr: str,
        ee: str | None,
        frequency: float,
        *,
        arm: str | None = None,
        source_key: str = "robot-step",
        action_space: str = "target_joint_position",
        robot_type: str | None = None,
        gripper_type: str | None = None,
        finger_type: str | None = None,
        head_dim: int = 0,
        body_dim: int = 0,
        raw_head_dim: int = 0,
        connect_timeout_s: float = 5.0,
    ) -> None:
        self._config = {
            "loop_addr": loop_addr,
            "ee": ee,
            "frequency": frequency,
            "arm": arm,
            "source_key": source_key,
            "action_space": action_space,
            "robot_type": robot_type,
            "gripper_type": gripper_type,
            "finger_type": finger_type,
            "head_dim": int(head_dim or 0),
            "body_dim": int(body_dim or 0),
            "raw_head_dim": int(raw_head_dim or 0),
        }
        self._link: _BridgeLink | None = None
        self._sender = None  # kept for loop_smoke_test's stats probe

    def connect(self) -> None:
        self._link = _BridgeLink.acquire(self._config["loop_addr"])
        self._link.register_config("config/robot", self._config)
        self._sender = self._link
        logger_mp.info(f"[loop] robot-state forwarding to sidecar (source={self._config['source_key']})")

    def set_session_metadata(self, metadata: dict) -> None:
        """Ship a free-form session/CLI snapshot to the sidecar, which writes it
        as ``teleop_session.json`` next to each Loop-saved episode. Registered
        like a config, so it is re-sent on every reconnect. Call after connect().
        Values are JSON-sanitized here (non-serializable -> str)."""
        if self._link is None:
            return
        try:
            clean = json.loads(json.dumps(metadata, default=str))
            self._link.register_config("meta/session", clean)
            logger_mp.info(f"[loop] session metadata registered ({len(clean)} top-level keys)")
        except Exception as exc:
            logger_mp.warning(f"[loop] session metadata rejected: {exc}")

    def send(self, timestamp_us: int, arm_q, arm_dq, arm_action,
             ee_state_full=None, ee_action_full=None,
             head_state=None, head_action=None,
             body_state=None, raw_head=None) -> None:
        if self._link is None:
            return
        try:
            self._link.publish("robot", {
                "t": int(timestamp_us),
                "q": _floats(arm_q),
                "dq": _floats(arm_dq),
                "a": _floats(arm_action),
                "es": _floats(ee_state_full),
                "ea": _floats(ee_action_full),
                "hq": _floats(head_state),
                "ha": _floats(head_action),
                "bq": _floats(body_state),
                "rh": _floats(raw_head),
            })
        except Exception as exc:  # never let streaming break teleop
            logger_mp.warning(f"[loop] robot publish failed: {exc}")

    def close(self) -> None:
        if self._link is not None:
            self._link.release()
            self._link = None
            self._sender = None


class LoopHandStreamer:
    """Same call surface as the original tactile streamer."""

    def __init__(self, loop_addr: str, frequency: float, *,
                 hand_key: str = "rh5dg2", source_key: str = "rh5dg2") -> None:
        self._config = {
            "loop_addr": loop_addr,
            "frequency": frequency,
            "hand_key": hand_key,
            "source_key": source_key,
        }
        self._link: _BridgeLink | None = None

    def connect(self) -> None:
        self._link = _BridgeLink.acquire(self._config["loop_addr"])
        self._link.register_config("config/hand", self._config)
        logger_mp.info(f"[loop] tactile forwarding to sidecar (source={self._config['source_key']})")

    def send(self, timestamp_us: int, hand_state_full, hand_action_full, tactiles=None) -> None:
        if self._link is None:
            return
        try:
            self._link.publish("tactile", {
                "t": int(timestamp_us),
                "es": _floats(hand_state_full),
                "ea": _floats(hand_action_full),
                "tactiles": _jsonable_tactiles(tactiles),
            })
        except Exception as exc:
            logger_mp.warning(f"[loop] tactile publish failed: {exc}")

    def close(self) -> None:
        if self._link is not None:
            self._link.release()
            self._link = None


def _jsonable_tactiles(tactiles):
    """Tactile dicts may hold numpy arrays/scalars; make them JSON-clean."""
    if not isinstance(tactiles, dict):
        return None
    out = {}
    for side, data in tactiles.items():
        if not isinstance(data, dict):
            continue
        side_out = {}
        fingers = data.get("fingers")
        if isinstance(fingers, dict):
            side_out["fingers"] = {k: _floats(v) for k, v in fingers.items() if v is not None}
        palm = data.get("palm")
        if palm is not None:
            side_out["palm"] = _floats(palm)
        out[str(side)] = side_out
    return out


class LoopCameraStreamer:
    """Same call surface as the original camera streamer.

    ``set_*`` ships the teleimager JPEG bytes when available (the sidecar
    JPEG-bypasses or decodes as needed); otherwise the raw BGR frame goes over
    localhost, which is cheap. Reference-swap semantics are preserved: the
    control thread never encodes or touches the network.
    """

    _SETTER_KEYS = ("head_camera", "left_wrist_camera", "right_wrist_camera")

    def __init__(self, loop_addr: str, camera_config: dict, *, connect_timeout_s: float = 5.0) -> None:
        self._config = {"loop_addr": loop_addr, "camera_config": _jsonable_camera_config(camera_config)}
        self._link: _BridgeLink | None = None
        self._warned_dtype = False

    def connect(self) -> None:
        self._link = _BridgeLink.acquire(self._config["loop_addr"])
        self._link.register_config("config/camera", self._config)
        enabled = [k for k in self._SETTER_KEYS
                   if (self._config["camera_config"].get(k) or {}).get("enable_zmq")]
        logger_mp.info(f"[loop] camera forwarding to sidecar (cameras={enabled or 'none'})")

    def _set(self, cfg_key: str, bgr, jpg) -> None:
        if self._link is None or (bgr is None and jpg is None):
            return
        try:
            meta: dict = {"t": time.time_ns() // 1000}
            if jpg is not None:
                meta["enc"] = "jpg"
                payload = ("jpg", jpg)
            else:
                if getattr(bgr, "dtype", None) is not None and str(bgr.dtype) != "uint8":
                    if not self._warned_dtype:
                        logger_mp.warning(f"[loop] {cfg_key}: unsupported dtype {bgr.dtype}; frame dropped")
                        self._warned_dtype = True
                    return
                meta["enc"] = "bgr"
                meta["shape"] = list(bgr.shape)
                payload = ("bgr", bgr)
            self._link.publish(f"cam/{cfg_key}", meta, payload)
        except Exception as exc:
            logger_mp.warning(f"[loop] camera publish failed for {cfg_key}: {exc}")

    def set_head(self, bgr, jpg=None) -> None:
        self._set("head_camera", bgr, jpg)

    def set_left_wrist(self, bgr, jpg=None) -> None:
        self._set("left_wrist_camera", bgr, jpg)

    def set_right_wrist(self, bgr, jpg=None) -> None:
        self._set("right_wrist_camera", bgr, jpg)

    def close(self) -> None:
        if self._link is not None:
            self._link.release()
            self._link = None


def _jsonable_camera_config(camera_config: dict) -> dict:
    """Keep only the JSON-safe fields the sidecar needs to build its channels."""
    out = {}
    for key, cam in (camera_config or {}).items():
        if not isinstance(cam, dict):
            continue
        cleaned = {}
        for field in ("enable_zmq", "binocular", "fps"):
            if field in cam:
                cleaned[field] = cam[field]
        shape = cam.get("image_shape")
        if shape is not None:
            cleaned["image_shape"] = [int(v) for v in list(shape)[:2]]
        out[str(key)] = cleaned
    return out
