"""Stream robot state and camera frames from xr_teleoperate into Config Loop.

This module is an OPTIONAL sink, enabled with ``--loop`` on
``teleop_hand_and_arm.py``. With the flag off, importing this module has zero
effect on teleoperation; with it on, two independent streams start:

  1. Robot state  -> loop_sdk.RobotStepSender (gRPC, non-blocking send)
  2. Camera media -> an in-process RTSP/RTP-JPEG server that Loop PULLS from,
     advertised over gRPC via loop_sdk.SourceProducer.

Why an RTSP server at all? loop-sdk has no ``send_camera_frame()`` API: Loop
pulls camera media from an RTSP/RTP endpoint and the SDK only advertises where.
We therefore stand up the server ourselves. The RTP/JPEG packetization
(``_parse_jpeg_for_rtp`` / ``_mjpeg_rtp_payloads`` / ``_rtp_header`` / the RTSP
request handling) is VENDORED VERBATIM from the loop-sdk e2e fixture
(``loop-sdk/e2e/send_simulated_sources.py``) because that is the only camera
transport proven to interop with Loop (see loop-sdk/docs/camera.md "green
noise" troubleshooting for why hand-rolling this is risky). The only thing we
changed is the frame *source*: the fixture loops a fixed list of frames from a
video file; we serve the latest live frame handed in by the control loop.

Threading / data flow (cameras)
-------------------------------

    control loop (30Hz, sync)                 loop bg thread (asyncio)
    -------------------------                 ------------------------
    head_img.bgr  --set_frame()-->  [latest raw BGR slot, lock]
                                          |  (per-camera encoder thread)
                                          v  cv2.imencode + _parse_jpeg_for_rtp
                                    [latest JpegRtpFrame slot, lock]
                                          |
                                          v  RtspLiveServer._stream (camera fps)
                                    RTP/JPEG over RTSP/TCP  --> Loop pulls

The control thread only swaps a numpy reference (set_frame). JPEG encoding and
all network I/O happen off the control thread, so a slow or dead Loop consumer
can never stall the 30Hz robot control loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging_mp
import random
import struct
import threading
import time
from dataclasses import dataclass

logger_mp = logging_mp.getLogger(__name__)

# ----------------------------------------------------------------------------
# End-effector dimensions (per hand). The dual_hand_state/action arrays in
# teleop_hand_and_arm.py concatenate left+right, so per-hand width is half.
#   dex3            -> Array('d', 14) => 7 per hand
#   dex1 (gripper)  -> Array('d', 2)  => 1 per hand
#   inspire_dfx/inspire_ftp/brainco -> Array('d', 12) => 6 per hand
#   rh5dg2_*/inspire_dg2 -> Array('d', RH5DG2_Num_Motors * 2) => 13 per hand
#   rh56f1          -> Array('d', RH56F1_Num_Retarget_Joints * 2) => 12 per hand
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
    "hx5_d20": 20,
}

_EMBODIMENT_ROOT = {
    "AI_WORKER": "ai_worker",
}


def ee_dim_per_hand(ee: str | None) -> int:
    """Per-hand end-effector channel count for the given ``--ee`` selection."""
    return _EE_DIM_PER_HAND.get(ee or "", 0)


def _as_float_list(values) -> list[float]:
    """Coerce a numpy array / list / Array slice into a clean ``list[float]``.

    The SDK locks channel layout on the first send and treats a list of all
    numbers as ONE vector channel; a stray None flips it to indexed scalars and
    silently changes the layout. So we always emit clean float lists.
    """
    return [float(v) for v in values]


# ============================================================================
# Robot state streaming
# ============================================================================
class LoopRobotStreamer:
    """Wrap loop_sdk.RobotStepSender for a dual 7-DOF-arm + optional EE loop.

    Channel layout (fixed at first send, complete from tick 0):

        <embodiment>.observation.left_arm.joint_position   [7]
        <embodiment>.observation.left_arm.joint_velocity   [7]
        <embodiment>.observation.right_arm.joint_position  [7]
        <embodiment>.observation.right_arm.joint_velocity  [7]
        <embodiment>.observation.left_ee.joint_position    [n]
        <embodiment>.observation.right_ee.joint_position   [n]
        <embodiment>.action.left_arm.joint_position        [7]
        <embodiment>.action.right_arm.joint_position       [7]
        <embodiment>.action.left_ee.joint_position         [n]
        <embodiment>.action.right_ee.joint_position        [n]

    Existing embodiments retain ``g1`` for compatibility; AI Worker uses
    ``ai_worker`` and HX5 supplies ``n=20`` without altering its joint order.
    """

    def __init__(
        self,
        loop_addr: str,
        ee: str | None,
        frequency: float,
        *,
        arm: str | None = None,
        source_key: str = "robot-step",
        connect_timeout_s: float = 5.0,
    ) -> None:
        self._loop_addr = loop_addr
        self._ee = ee
        self._ee_dim = ee_dim_per_hand(ee)
        self._arm = arm
        self._root = _EMBODIMENT_ROOT.get(arm or "", "g1")
        self._frequency = int(round(frequency))
        self._source_key = source_key
        self._connect_timeout_s = connect_timeout_s
        self._sender = None
        self._sequence = 0
        self._send_failed_once = False

    def connect(self) -> None:
        """Connect and FAIL FAST if Loop is unreachable at startup.

        RobotStepSender.connect()/declare() are non-blocking: the gRPC channel
        is lazy and never raises on an unreachable address. So we declare the
        layout, then POLL stats().connected with a timeout and raise if the
        channel never comes up. This confirms the channel opened (a strong
        startup check) but not that Loop accepted registration.
        """
        from loop_sdk import RobotConfigOptions, RobotStepSender

        self._sender = RobotStepSender(
            self._loop_addr,
            self._source_key,
            name=self._source_key,
            options=RobotConfigOptions(control_hz=(self._frequency,)),
        )
        self._sender.connect()
        self._sender.declare(self._channel_layout())

        deadline = time.time() + self._connect_timeout_s
        while time.time() < deadline:
            try:
                if getattr(self._sender.stats(), "connected", False):
                    logger_mp.info(f"[loop] robot-state connected to {self._loop_addr}")
                    return
            except Exception:
                pass
            time.sleep(0.1)
        raise ConnectionError(
            f"[loop] could not reach Config Loop at {self._loop_addr} within "
            f"{self._connect_timeout_s:g}s (robot-state). Is Loop running?"
        )

    def _channel_layout(self) -> tuple[str, ...]:
        """Flattened dotted channel keys, derived from a zero-filled step so the
        declared layout exactly matches what send() will emit."""
        from loop_sdk import flatten_step

        zeros_arm = [0.0] * 7
        zeros_ee = [0.0] * self._ee_dim
        step = self._build_step(
            zeros_arm, zeros_arm, zeros_arm, zeros_arm,
            zeros_ee, zeros_ee, zeros_ee, zeros_ee,
        )
        return tuple(flatten_step(step).keys())

    def _build_step(
        self,
        left_arm_q, right_arm_q, left_arm_dq, right_arm_dq,
        left_ee_state, right_ee_state, left_ee_action, right_ee_action,
    ) -> dict:
        observation: dict = {
            "left_arm": {"joint_position": _as_float_list(left_arm_q),
                         "joint_velocity": _as_float_list(left_arm_dq)},
            "right_arm": {"joint_position": _as_float_list(right_arm_q),
                          "joint_velocity": _as_float_list(right_arm_dq)},
        }
        action: dict = {
            "left_arm": {"joint_position": _as_float_list(left_arm_q)},
            "right_arm": {"joint_position": _as_float_list(right_arm_q)},
        }
        # NOTE: action arms are filled by send() with sol_q; placeholders above
        # are replaced there. _build_step is also used for the layout probe with
        # zeros, so keep keys present regardless.
        if self._ee_dim > 0:
            observation["left_ee"] = {"joint_position": _as_float_list(left_ee_state)}
            observation["right_ee"] = {"joint_position": _as_float_list(right_ee_state)}
            action["left_ee"] = {"joint_position": _as_float_list(left_ee_action)}
            action["right_ee"] = {"joint_position": _as_float_list(right_ee_action)}
        return {self._root: {"observation": observation, "action": action}}

    def send(
        self,
        timestamp_us: int,
        arm_q,
        arm_dq,
        arm_action,
        ee_state_full=None,
        ee_action_full=None,
    ) -> None:
        """Send one tick. Never raises into the control loop.

        ``arm_q``/``arm_dq``/``arm_action`` are 14-dim (left[:7] + right[-7:]).
        ``ee_state_full``/``ee_action_full`` are the LEFT+RIGHT concatenated EE
        arrays (length ``2 * ee_dim_per_hand``); split here by the known per-hand
        width. Pass empty/None when there is no end effector.
        """
        if self._sender is None:
            return
        try:
            n = self._ee_dim
            es = list(ee_state_full or [])
            ea = list(ee_action_full or [])
            left_ee_state, right_ee_state = es[:n], es[n : 2 * n]
            left_ee_action, right_ee_action = ea[:n], ea[n : 2 * n]
            step = self._build_step(
                arm_q[:7], arm_q[-7:], arm_dq[:7], arm_dq[-7:],
                left_ee_state, right_ee_state,
                left_ee_action, right_ee_action,
            )
            # overwrite action arms with the actual IK target
            root = step[self._root]
            root["action"]["left_arm"]["joint_position"] = _as_float_list(arm_action[:7])
            root["action"]["right_arm"]["joint_position"] = _as_float_list(arm_action[-7:])
            self._sender.send(timestamp_us, step, sequence=self._sequence)
            self._sequence += 1
        except Exception as exc:  # never let streaming break teleop
            if not self._send_failed_once:
                logger_mp.warning(f"[loop] robot-state send failed (will keep trying silently): {exc}")
                self._send_failed_once = True

    def close(self) -> None:
        if self._sender is not None:
            with contextlib.suppress(Exception):
                self._sender.disconnect()
            self._sender = None


# ============================================================================
# Camera streaming
# ============================================================================
# --- VENDORED from loop-sdk/e2e/send_simulated_sources.py (RTP/JPEG core) -----
# Keep these byte-for-byte compatible with the SDK; only the frame source below
# is xr_teleoperate-specific.
_RTP_CLOCK_RATE = 90_000
_RTP_PAYLOAD_TYPE = 26
_MAX_RTP_PAYLOAD = 1_200
_RTSP_SESSION_ID = "xr-teleoperate-loop"
_SHUTDOWN_TIMEOUT_SECONDS = 3.0
# RTP/JPEG dimension limit: width/height are expressed in 8px blocks in one byte.
_MAX_RTP_JPEG_DIM = 255 * 8  # 2040 px


@dataclass(frozen=True)
class JpegRtpFrame:
    width_blocks: int
    height_blocks: int
    jpeg_type: int
    restart_interval: int
    quantization_tables: bytes
    scan_data: bytes


def _parse_jpeg_for_rtp(jpeg: bytes) -> JpegRtpFrame:
    if len(jpeg) < 4 or jpeg[:2] != b"\xff\xd8":
        raise ValueError("MJPEG frame must start with SOI")
    offset = 2
    width = height = 0
    jpeg_type = 1
    restart_interval = 0
    quantization_tables: dict[int, bytes] = {}
    scan_start = 0
    scan_end = len(jpeg)
    while offset < len(jpeg):
        if jpeg[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(jpeg) and jpeg[offset] == 0xFF:
            offset += 1
        if offset >= len(jpeg):
            break
        marker = jpeg[offset]
        offset += 1
        if marker == 0xD9:
            break
        if marker in {0x01, *range(0xD0, 0xD8)}:
            continue
        if offset + 2 > len(jpeg):
            raise ValueError("Malformed JPEG marker length")
        segment_length = int.from_bytes(jpeg[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(jpeg):
            raise ValueError("Malformed JPEG segment")
        segment = jpeg[offset + 2 : offset + segment_length]
        offset += segment_length
        if marker == 0xDB:
            quantization_tables.update(_jpeg_quantization_tables(segment))
        elif marker == 0xDD:
            restart_interval = _jpeg_restart_interval(segment)
        elif marker == 0xC0:
            width, height, jpeg_type = _jpeg_frame_header(segment)
        elif marker == 0xDA:
            scan_start = offset
            eoi = jpeg.rfind(b"\xff\xd9")
            scan_end = eoi if eoi > scan_start else len(jpeg)
            break
    if width <= 0 or height <= 0:
        raise ValueError("MJPEG frame is missing baseline dimensions")
    if scan_start <= 0:
        raise ValueError("MJPEG frame is missing scan data")
    width_blocks = (width + 7) // 8
    height_blocks = (height + 7) // 8
    if width_blocks > 255 or height_blocks > 255:
        raise ValueError("MJPEG frame is too large for RTP/JPEG dimensions")
    luma_table = quantization_tables.get(0)
    chroma_table = quantization_tables.get(1, luma_table)
    if luma_table is None or chroma_table is None:
        raise ValueError("MJPEG frame is missing quantization tables")
    if restart_interval > 0:
        jpeg_type += 64
    return JpegRtpFrame(
        width_blocks=width_blocks,
        height_blocks=height_blocks,
        jpeg_type=jpeg_type,
        restart_interval=restart_interval,
        quantization_tables=luma_table + chroma_table,
        scan_data=jpeg[scan_start:scan_end],
    )


def _jpeg_quantization_tables(segment: bytes) -> dict[int, bytes]:
    tables: dict[int, bytes] = {}
    offset = 0
    while offset < len(segment):
        table_spec = segment[offset]
        offset += 1
        precision = table_spec >> 4
        table_id = table_spec & 0x0F
        table_size = 64 * (2 if precision else 1)
        if precision != 0:
            raise ValueError("Only 8-bit MJPEG quantization tables are supported")
        if offset + table_size > len(segment):
            raise ValueError("Malformed MJPEG quantization table")
        tables[table_id] = segment[offset : offset + table_size]
        offset += table_size
    return tables


def _jpeg_restart_interval(segment: bytes) -> int:
    if len(segment) != 2:
        raise ValueError("Malformed MJPEG restart interval")
    return int.from_bytes(segment, "big")


def _jpeg_frame_header(segment: bytes) -> tuple[int, int, int]:
    if len(segment) < 9:
        raise ValueError("Malformed MJPEG frame header")
    height = int.from_bytes(segment[1:3], "big")
    width = int.from_bytes(segment[3:5], "big")
    sampling = segment[7]
    jpeg_type = 0 if sampling == 0x21 else 1
    return width, height, jpeg_type


def _rtp_header(sequence: int, timestamp: int, ssrc: int, marker: bool) -> bytes:
    return struct.pack("!BBHII", 0x80, (0x80 if marker else 0) | _RTP_PAYLOAD_TYPE, sequence, timestamp, ssrc)


def _mjpeg_rtp_payloads(frame: JpegRtpFrame) -> list[bytes]:
    fragments: list[bytes] = []
    offset = 0
    while offset < len(frame.scan_data):
        quantization_header = b""
        if offset == 0:
            quantization_header = b"\x00\x00" + len(frame.quantization_tables).to_bytes(2, "big")
            quantization_header += frame.quantization_tables
        restart_header = b""
        if frame.restart_interval > 0:
            restart_header = frame.restart_interval.to_bytes(2, "big") + b"\xff\xff"
        chunk_size = _MAX_RTP_PAYLOAD - 8 - len(restart_header) - len(quantization_header)
        if chunk_size <= 0:
            raise ValueError("MJPEG quantization tables are too large for RTP payload")
        chunk = frame.scan_data[offset : offset + chunk_size]
        header = (
            b"\x00" + offset.to_bytes(3, "big") + bytes([frame.jpeg_type, 255, frame.width_blocks, frame.height_blocks])
        )
        fragments.append(header + restart_header + quantization_header + chunk)
        offset += len(chunk)
    return fragments


async def _write_mjpeg_rtp_frame(writer, channel, sequence, timestamp, ssrc, frame: JpegRtpFrame) -> int:
    payloads = _mjpeg_rtp_payloads(frame)
    for index, payload in enumerate(payloads):
        marker = index == len(payloads) - 1
        rtp = _rtp_header(sequence, timestamp, ssrc, marker) + payload
        writer.write(b"$" + bytes([channel]) + struct.pack("!H", len(rtp)) + rtp)
        await writer.drain()
        sequence = (sequence + 1) % 2**16
    return sequence


async def _read_rtsp_request(reader):
    header = await _read_until(reader, b"\r\n\r\n")
    if not header:
        return None
    lines = header.decode("iso-8859-1").split("\r\n")
    request_line = lines[0].split(" ")
    if len(request_line) < 2:
        return None
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line or ":" not in line:
            continue
        key, value = line.split(":", maxsplit=1)
        headers[key.strip().lower()] = value.strip()
    content_length = int(headers.get("content-length", "0") or "0")
    if content_length > 0:
        await reader.readexactly(content_length)
    return request_line[0].upper(), request_line[1], headers


async def _read_until(reader, separator: bytes) -> bytes:
    try:
        return await reader.readuntil(separator)
    except asyncio.IncompleteReadError as error:
        return error.partial


async def _write_rtsp_response(writer, cseq, *, status_code=200, status_text="OK", headers=None, body=b"") -> None:
    response_headers = {"CSeq": cseq, "Server": "xr-teleoperate-loop", "Content-Length": str(len(body)), **(headers or {})}
    writer.write(f"RTSP/1.0 {status_code} {status_text}\r\n".encode())
    for key, value in response_headers.items():
        writer.write(f"{key}: {value}\r\n".encode())
    writer.write(b"\r\n")
    writer.write(body)
    await writer.drain()


def _content_base(uri: str) -> str:
    return uri if uri.endswith("/") else f"{uri}/"


def _interleaved_channel(transport: str) -> int:
    marker = "interleaved="
    if marker not in transport:
        return 0
    value = transport.split(marker, maxsplit=1)[1].split(";", maxsplit=1)[0]
    first = value.split("-", maxsplit=1)[0]
    return int(first) if first.isdecimal() else 0


# --- END vendored core -------------------------------------------------------


class _LatestFrame:
    """Thread-safe holder for the most recent parsed JpegRtpFrame of a camera."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: JpegRtpFrame | None = None

    def set(self, frame: JpegRtpFrame) -> None:
        with self._lock:
            self._frame = frame

    def get(self) -> JpegRtpFrame | None:
        with self._lock:
            return self._frame


class _RawSlot:
    """Thread-safe single-slot mailbox for the latest frame.

    Holds BOTH the decoded BGR array and (when available) the original JPEG
    bytes from teleimager. The control thread drops references (free); an
    encoder thread pulls them. ``seq`` lets the encoder skip an unchanged frame.

    Preferring ``jpg`` lets us BYPASS a decode->re-encode round trip: teleimager
    publishes ``cv2.imencode('.jpg', ...)`` bytes (baseline + standard Huffman,
    RTP/JPEG-compatible), so for monocular cameras we can packetize those bytes
    directly. BGR is still needed for the binocular head (crop the left eye) and
    as a fallback when no JPEG is available (e.g. the smoke test).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bgr = None
        self._jpg = None
        self._seq = 0

    def set(self, bgr, jpg) -> None:
        with self._lock:
            self._bgr = bgr
            self._jpg = jpg
            self._seq += 1

    def get(self) -> tuple[object, object, int]:
        with self._lock:
            return self._bgr, self._jpg, self._seq


@dataclass
class _CameraChannel:
    source_key: str
    name: str
    rtsp_path: str
    port: int
    width: int
    height: int
    fps: float
    # which slice of the source frame this channel serves:
    #   None   -> full frame (monocular: JPEG bypass possible)
    #   "left" -> left half  of a side-by-side binocular frame
    #   "right"-> right half of a side-by-side binocular frame
    # (a binocular head fans out into one "left" + one "right" channel)
    crop: str | None = None
    raw: _RawSlot | None = None
    latest: _LatestFrame | None = None
    producer: object = None  # SourceProducer

    def __post_init__(self) -> None:
        self.raw = _RawSlot()
        self.latest = _LatestFrame()

    @property
    def rtsp_uri(self) -> str:
        return f"rtsp://{_RTSP_ADVERTISE_HOST}:{self.port}/{self.rtsp_path.strip('/')}"


# Loop and the RTSP server run on the same host as xr_teleoperate (LOOP_ADDR
# default localhost), so 127.0.0.1 is correct for both listen and advertise.
_RTSP_ADVERTISE_HOST = "127.0.0.1"
_RTSP_LISTEN_HOST = "0.0.0.0"
_RTSP_BASE_PORT = 8554


class _RtspLiveServer:
    """RTSP server that streams a camera's LATEST live frame as RTP/JPEG/TCP.

    Adapted from loop-sdk e2e RtspMjpegFixtureServer: identical RTSP handshake
    and RTP/JPEG writing; the only change is ``_stream`` reads the live
    ``_LatestFrame`` instead of indexing a fixed frame list.
    """

    def __init__(self, channel: _CameraChannel) -> None:
        self._channel = channel
        self._server: asyncio.AbstractServer | None = None
        self._client_tasks: set = set()
        self._client_writers: set = set()
        self._ssrc = random.getrandbits(32)
        self._stopping = False

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle_client, _RTSP_LISTEN_HOST, self._channel.port)

    async def stop(self) -> None:
        self._stopping = True
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        for writer in list(self._client_writers):
            writer.close()
        current = asyncio.current_task()
        tasks = {t for t in self._client_tasks if t is not current and not t.done()}
        for t in tasks:
            t.cancel()
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=_SHUTDOWN_TIMEOUT_SECONDS)
            for t in pending:
                t.cancel()

    async def serve_forever(self) -> None:
        if self._server is None:
            raise RuntimeError("RTSP server has not been started")
        async with self._server:
            await self._server.serve_forever()

    async def _handle_client(self, reader, writer) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._client_tasks.add(task)
        self._client_writers.add(writer)
        client = writer.get_extra_info("peername")
        logger_mp.info(f"[loop] rtsp {self._channel.source_key} client connected {client}")
        interleaved_channel = 0
        try:
            while not self._stopping and not reader.at_eof():
                request = await _read_rtsp_request(reader)
                if request is None:
                    return
                method, uri, headers = request
                cseq = headers.get("cseq", "1")
                if method == "OPTIONS":
                    await _write_rtsp_response(writer, cseq, headers={"Public": "OPTIONS, DESCRIBE, SETUP, PLAY, TEARDOWN"})
                elif method == "DESCRIBE":
                    await _write_rtsp_response(
                        writer, cseq,
                        headers={"Content-Base": _content_base(uri), "Content-Type": "application/sdp"},
                        body=self._sdp().encode(),
                    )
                elif method == "SETUP":
                    interleaved_channel = _interleaved_channel(headers.get("transport", ""))
                    await _write_rtsp_response(
                        writer, cseq,
                        headers={
                            "Session": _RTSP_SESSION_ID,
                            "Transport": (
                                "RTP/AVP/TCP;unicast;"
                                f"interleaved={interleaved_channel}-{interleaved_channel + 1};"
                                f"ssrc={self._ssrc:08X}"
                            ),
                        },
                    )
                elif method == "PLAY":
                    await _write_rtsp_response(writer, cseq, headers={"Session": _RTSP_SESSION_ID, "Range": "npt=0.000-"})
                    await self._stream(writer, interleaved_channel)
                    return
                elif method == "TEARDOWN":
                    await _write_rtsp_response(writer, cseq, headers={"Session": _RTSP_SESSION_ID})
                    return
                else:
                    await _write_rtsp_response(writer, cseq, status_code=405, status_text="Method Not Allowed")
        except (ConnectionError, BrokenPipeError, asyncio.IncompleteReadError):
            return
        finally:
            self._client_writers.discard(writer)
            if task is not None:
                self._client_tasks.discard(task)
            writer.close()
            with contextlib.suppress(ConnectionError, BrokenPipeError):
                await writer.wait_closed()
            logger_mp.info(f"[loop] rtsp {self._channel.source_key} client disconnected {client}")

    def _sdp(self) -> str:
        return "\r\n".join([
            "v=0",
            f"o=- 0 0 IN IP4 {_RTSP_ADVERTISE_HOST}",
            f"s={self._channel.name}",
            "t=0 0",
            "a=control:*",
            f"m=video 0 RTP/AVP/TCP {_RTP_PAYLOAD_TYPE}",
            f"a=rtpmap:{_RTP_PAYLOAD_TYPE} JPEG/{_RTP_CLOCK_RATE}",
            "a=control:trackID=0",
            "",
        ])

    async def _stream(self, writer, channel: int) -> None:
        sequence = random.randrange(0, 2**16)
        frame_interval = 1.0 / self._channel.fps
        started = time.monotonic()
        next_send = started
        while not self._stopping and not writer.is_closing():
            delay = next_send - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            send_at = time.monotonic()
            frame = self._channel.latest.get()
            if frame is not None:
                timestamp = int(max(0.0, send_at - started) * _RTP_CLOCK_RATE) % 2**32
                sequence = await _write_mjpeg_rtp_frame(writer, channel, sequence, timestamp, self._ssrc, frame)
            next_send += frame_interval
            if next_send <= send_at:
                next_send = send_at + frame_interval


class LoopCameraStreamer:
    """Serve live head/wrist frames to Config Loop over RTSP/RTP-JPEG.

    Lifecycle:
      streamer = LoopCameraStreamer(loop_addr, camera_config)
      streamer.connect()              # fail-fast: bind RTSP + advertise via gRPC
      streamer.set_head(bgr)          # called from the control loop each tick
      streamer.set_left_wrist(bgr)
      streamer.set_right_wrist(bgr)
      streamer.close()                # in finally
    """

    def __init__(self, loop_addr: str, camera_config: dict, *, connect_timeout_s: float = 5.0) -> None:
        self._loop_addr = loop_addr
        self._camera_config = camera_config
        self._connect_timeout_s = connect_timeout_s
        self._channels: dict[str, _CameraChannel] = {}        # source_key -> channel
        self._by_cfg: dict[str, list[_CameraChannel]] = {}    # cam_config key -> channels fed from it
        self._servers: list[_RtspLiveServer] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._encoder_threads: list[threading.Thread] = []
        self._stop_encoders = threading.Event()
        self._jpeg_quality = 80

    # --- channel set-up -----------------------------------------------------
    def _build_channels(self) -> None:
        """One Loop source per enabled camera, EXCEPT a binocular head, which is a
        single side-by-side JPEG on the wire -> fanned out into two sources
        (left eye + right eye). Each eye is served as its own RTSP source."""
        cfg = self._camera_config
        self._by_cfg = {}
        port = _RTSP_BASE_PORT
        specs = [
            ("head_camera", "head", "head camera", True),
            ("left_wrist_camera", "left_wrist", "left wrist camera", False),
            ("right_wrist_camera", "right_wrist", "right wrist camera", False),
        ]
        for cfg_key, base_path, base_name, is_head in specs:
            cam = cfg.get(cfg_key)
            if not cam or not cam.get("enable_zmq"):
                continue
            h, w = cam["image_shape"][0], cam["image_shape"][1]
            binocular = bool(cam.get("binocular")) if is_head else False
            if binocular:
                # side-by-side stereo: split into left/right, each half-width
                stream_w = w // 2
                variants = [("left", f"{base_path}-left", f"{base_name} (left)"),
                            ("right", f"{base_path}-right", f"{base_name} (right)")]
            else:
                stream_w = w
                variants = [(None, base_path, base_name)]
            if stream_w > _MAX_RTP_JPEG_DIM or h > _MAX_RTP_JPEG_DIM:
                logger_mp.warning(
                    f"[loop] {cfg_key} {stream_w}x{h} exceeds RTP/JPEG limit "
                    f"{_MAX_RTP_JPEG_DIM}px; skipping this camera."
                )
                continue
            chans: list[_CameraChannel] = []
            for crop, rtsp_path, name in variants:
                ch = _CameraChannel(
                    source_key=f"camera-{rtsp_path}",
                    name=name,
                    rtsp_path=rtsp_path,
                    port=port,
                    width=stream_w,
                    height=h,
                    fps=float(cam.get("fps", 30.0)),
                    crop=crop,
                )
                self._channels[ch.source_key] = ch
                chans.append(ch)
                port += 1
            self._by_cfg[cfg_key] = chans

    def connect(self) -> None:
        from loop_sdk import (
            CameraOpenResult,
            CameraPixelFormat,
            CameraResolution,
            CameraSetting,
            CameraStreamSchema,
            RtspCameraEndpoint,
            SourceProducer,
            SourceSchema,
        )

        self._build_channels()
        if not self._channels:
            logger_mp.warning("[loop] no ZMQ-enabled cameras found; camera streaming disabled.")
            return

        # 1) start the asyncio RTSP servers on a background thread.
        ready = threading.Event()

        def _run_loop() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            async def _start_servers() -> None:
                for ch in self._channels.values():
                    server = _RtspLiveServer(ch)
                    await server.start()
                    self._servers.append(server)
                    asyncio.ensure_future(server.serve_forever())

            self._loop.run_until_complete(_start_servers())
            ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=_run_loop, name="loop-rtsp", daemon=True)
        self._thread.start()
        if not ready.wait(timeout=self._connect_timeout_s):
            raise ConnectionError("[loop] RTSP servers failed to start within timeout")

        # 2) start per-camera encoder threads (cv2.imencode off the control thread).
        for ch in self._channels.values():
            t = threading.Thread(target=self._encode_loop, args=(ch,), name=f"loop-enc-{ch.rtsp_path}", daemon=True)
            t.start()
            self._encoder_threads.append(t)

        # 3) advertise each camera over gRPC; SourceProducer.connect does the
        #    handshake. on_camera_open returns our fixed setting + rtsp uri.
        def _make_open_cb(setting, uri):
            def _cb(_requested):
                return CameraOpenResult(applied_setting=setting, rtsp=RtspCameraEndpoint(uri=uri))
            return _cb

        for ch in self._channels.values():
            setting = CameraSetting(
                resolution=CameraResolution(width=ch.width, height=ch.height),
                fps=ch.fps,
                pixel_format=CameraPixelFormat.MJPEG,
            )
            ch.producer = SourceProducer.connect(
                loop_addr=self._loop_addr,
                source=SourceSchema(
                    camera=CameraStreamSchema(
                        source_key=ch.source_key,
                        name=ch.name,
                        rtsp=RtspCameraEndpoint(uri=ch.rtsp_uri),
                        available_settings=(setting,),
                    ),
                ),
                client_id=f"xr-teleoperate-{ch.rtsp_path}",
                on_camera_open=_make_open_cb(setting, ch.rtsp_uri),
            )
            logger_mp.info(f"[loop] camera {ch.source_key} advertised, serving {ch.rtsp_uri}")

    # --- frame ingestion (called from the 30Hz control thread) --------------
    def _set(self, cfg_key: str, bgr, jpg) -> None:
        # one source frame may feed >1 channel (binocular head -> left + right);
        # each channel's encoder slices its own half.
        if bgr is None and jpg is None:
            return
        for ch in self._by_cfg.get(cfg_key, ()):
            ch.raw.set(bgr, jpg)

    def set_head(self, bgr, jpg=None) -> None:
        self._set("head_camera", bgr, jpg)

    def set_left_wrist(self, bgr, jpg=None) -> None:
        self._set("left_wrist_camera", bgr, jpg)

    def set_right_wrist(self, bgr, jpg=None) -> None:
        self._set("right_wrist_camera", bgr, jpg)

    # --- encoder thread -----------------------------------------------------
    def _encode_loop(self, ch: _CameraChannel) -> None:
        """Turn the latest frame into an RTP-ready JpegRtpFrame.

        Fast path (full-frame, i.e. monocular cams): teleimager's JPEG bytes are
        already RTP/JPEG compatible, so parse them directly -- no decode, no
        re-encode, no quality loss. Slow path (a cropped eye of a binocular head,
        or no JPEG, or an incompatible/oversize JPEG): decode BGR, slice the
        configured half, and cv2-encode a fresh compatible JPEG.
        """
        cv2 = None
        try:
            import cv2  # only needed for the BGR fallback / binocular slice
        except Exception as exc:
            logger_mp.warning(f"[loop] cv2 unavailable for {ch.source_key}; JPEG-bypass only: {exc}")
        last_seq = -1
        period = 1.0 / max(ch.fps, 1.0)
        warned_fallback = False
        while not self._stop_encoders.is_set():
            bgr, jpg, seq = ch.raw.get()
            if seq == last_seq or (bgr is None and jpg is None):
                time.sleep(period)
                continue
            last_seq = seq
            frame = None
            # fast path: direct JPEG bypass (only for full-frame channels; a
            # cropped eye can't be sliced out of the combined JPEG without decoding)
            if jpg is not None and ch.crop is None:
                try:
                    frame = _parse_jpeg_for_rtp(bytes(jpg))
                except Exception as exc:
                    if not warned_fallback:
                        logger_mp.warning(f"[loop] {ch.source_key} JPEG not RTP-compatible, "
                                          f"falling back to re-encode: {exc}")
                        warned_fallback = True
            # slow path: decode / slice the configured half / re-encode via cv2
            if frame is None and bgr is not None and cv2 is not None:
                try:
                    half = bgr.shape[1] // 2
                    if ch.crop == "left":
                        img = bgr[:, :half]
                    elif ch.crop == "right":
                        img = bgr[:, half:]
                    else:
                        img = bgr
                    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
                    if ok:
                        frame = _parse_jpeg_for_rtp(buf.tobytes())
                except Exception as exc:
                    logger_mp.warning(f"[loop] encode failed for {ch.source_key}: {exc}")
            if frame is not None:
                ch.latest.set(frame)
            time.sleep(period)

    # --- teardown -----------------------------------------------------------
    def close(self) -> None:
        self._stop_encoders.set()
        for t in self._encoder_threads:
            t.join(timeout=2.0)
        for ch in self._channels.values():
            if ch.producer is not None:
                with contextlib.suppress(Exception):
                    ch.producer.close()
        if self._loop is not None:
            async def _stop_all() -> None:
                for server in self._servers:
                    with contextlib.suppress(Exception):
                        await server.stop()
            with contextlib.suppress(Exception):
                fut = asyncio.run_coroutine_threadsafe(_stop_all(), self._loop)
                fut.result(timeout=_SHUTDOWN_TIMEOUT_SECONDS + 1.0)
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=3.0)
