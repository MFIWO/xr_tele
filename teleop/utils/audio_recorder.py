"""Background WAV recorder for teleop episodes.

The recorder keeps audio I/O off the teleop loop. In ``auto`` mode it first
tries a sounddevice callback stream, then falls back to ALSA ``arecord`` with
raw PCM streamed through stdout into Python's WAV writer.
"""

from __future__ import annotations

import math
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
import wave
from dataclasses import dataclass
from typing import Any


class AudioRecorderError(RuntimeError):
    """Raised when an audio backend cannot start."""


@dataclass
class _AudioChunk:
    index: int
    timestamp: float
    data: bytes
    num_samples: int


class BackgroundAudioRecorder:
    """Record int16 mono/stereo audio to WAV from a background backend."""

    SUPPORTED_DTYPES = {"int16": {"sample_width": 2, "alsa_format": "S16_LE"}}

    def __init__(
        self,
        output_path: str,
        *,
        device: str = "plughw:2,0",
        sample_rate: int = 48000,
        channels: int = 1,
        dtype: str = "int16",
        chunk_size: int = 1024,
        rel_path: str = "audios/audio.wav",
        backend: str = "auto",
        queue_max_chunks: int = 512,
    ) -> None:
        self.output_path = os.path.abspath(output_path)
        self.rel_path = rel_path
        self.device = device
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.dtype = str(dtype)
        self.chunk_size = int(chunk_size)
        self.backend = backend
        self.queue_max_chunks = int(queue_max_chunks)

        if self.dtype not in self.SUPPORTED_DTYPES:
            raise ValueError(f"unsupported audio dtype {self.dtype!r}; supported: {sorted(self.SUPPORTED_DTYPES)}")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")
        if self.channels <= 0:
            raise ValueError("channels must be greater than zero")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if self.backend not in ("auto", "sounddevice", "arecord"):
            raise ValueError("backend must be one of: auto, sounddevice, arecord")

        self.sample_width = self.SUPPORTED_DTYPES[self.dtype]["sample_width"]
        self.alsa_format = self.SUPPORTED_DTYPES[self.dtype]["alsa_format"]
        self.bytes_per_chunk = self.chunk_size * self.channels * self.sample_width

        self._queue: queue.Queue[_AudioChunk | None] = queue.Queue(maxsize=self.queue_max_chunks)
        self._stop_event = threading.Event()
        self._writer_thread: threading.Thread | None = None
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._wave_file: wave.Wave_write | None = None
        self._sd_stream: Any = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._stderr_tail: list[str] = []

        self.active_backend: str | None = None
        self.started = False
        self.start_timestamp: float | None = None
        self.end_timestamp: float | None = None
        self.total_chunks = 0
        self.total_samples = 0
        self.dropped_chunks = 0
        self.chunk_timestamps: list[dict[str, Any]] = []
        self.last_error: str | None = None

    def start(self) -> None:
        """Start recording without blocking the caller's control loop."""

        if self.started:
            raise AudioRecorderError("audio recorder is already started")

        errors: list[str] = []
        backends = ["sounddevice", "arecord"] if self.backend == "auto" else [self.backend]
        for backend in backends:
            try:
                self._start_with_backend(backend)
                return
            except Exception as exc:
                errors.append(f"{backend}: {exc}")
                self._cleanup_after_failed_start()

        self.last_error = "; ".join(errors)
        raise AudioRecorderError(self.last_error)

    def stop(self, timeout: float = 2.0) -> dict[str, Any]:
        """Stop capture, drain queued chunks, and finalize the WAV header."""

        if not self.started:
            self.end_timestamp = self.end_timestamp or time.time()
            return self.metadata()

        self.end_timestamp = time.time()
        self._stop_event.set()

        if self._sd_stream is not None:
            try:
                self._sd_stream.stop()
            except Exception as exc:
                self.last_error = repr(exc)
            try:
                self._sd_stream.close()
            except Exception as exc:
                self.last_error = repr(exc)
            self._sd_stream = None

        if self._proc is not None:
            try:
                if self._proc.poll() is None:
                    self._proc.send_signal(signal.SIGINT)
                    self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            except Exception as exc:
                self.last_error = repr(exc)

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=timeout)

        self._put_sentinel()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=timeout)

        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=0.2)

        self._close_wave()
        self.started = False
        return self.metadata()

    def metadata(self) -> dict[str, Any]:
        """Return data.json-compatible audio metadata."""

        return {
            "enabled": self.start_timestamp is not None and self.active_backend is not None,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "dtype": self.dtype,
            "format": "wav",
            "path": self.rel_path,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "chunk_size": self.chunk_size,
            "backend": self.active_backend,
            "total_chunks": self.total_chunks,
            "total_samples": self.total_samples,
            "dropped_chunks": self.dropped_chunks,
            "error": self.last_error,
        }

    def _start_with_backend(self, backend: str) -> None:
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self._open_wave()
        self._stop_event.clear()
        self._writer_thread = threading.Thread(target=self._writer_loop, name="teleop_audio_wav_writer", daemon=True)
        self._writer_thread.start()
        self.start_timestamp = time.time()
        self.active_backend = backend

        if backend == "sounddevice":
            self._start_sounddevice()
        elif backend == "arecord":
            self._start_arecord()
        else:
            raise AudioRecorderError(f"unknown backend {backend!r}")

        self.started = True

    def _start_sounddevice(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioRecorderError("sounddevice is not installed") from exc

        def callback(indata, frames, time_info, status) -> None:
            del time_info
            if status:
                self.last_error = str(status)
            if self._stop_event.is_set():
                return
            self._enqueue_chunk(bytes(indata), int(frames), time.time())

        try:
            self._sd_stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                device=self.device,
                channels=self.channels,
                dtype=self.dtype,
                callback=callback,
            )
            self._sd_stream.start()
        except Exception as exc:
            raise AudioRecorderError(f"could not open sounddevice input {self.device!r}: {exc}") from exc

    def _start_arecord(self) -> None:
        arecord_path = shutil.which("arecord")
        if arecord_path is None:
            raise AudioRecorderError("arecord executable was not found")

        cmd = [
            arecord_path,
            "-D",
            self.device,
            "-f",
            self.alsa_format,
            "-r",
            str(self.sample_rate),
            "-c",
            str(self.channels),
            "-t",
            "raw",
            "-q",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="teleop_audio_arecord_stderr",
            daemon=True,
        )
        self._stderr_thread.start()

        time.sleep(0.15)
        if self._proc.poll() is not None:
            stderr = self._stderr_text()
            raise AudioRecorderError(stderr or f"arecord exited with code {self._proc.returncode}")

        self._reader_thread = threading.Thread(
            target=self._arecord_reader_loop,
            name="teleop_audio_arecord_reader",
            daemon=True,
        )
        self._reader_thread.start()

    def _open_wave(self) -> None:
        self._wave_file = wave.open(self.output_path, "wb")
        self._wave_file.setnchannels(self.channels)
        self._wave_file.setsampwidth(self.sample_width)
        self._wave_file.setframerate(self.sample_rate)

    def _close_wave(self) -> None:
        if self._wave_file is not None:
            try:
                self._wave_file.close()
            finally:
                self._wave_file = None

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                if self._wave_file is not None:
                    self._wave_file.writeframes(item.data)
                    self.total_chunks += 1
                    self.total_samples += item.num_samples
                    self.chunk_timestamps.append(
                        {
                            "index": item.index,
                            "timestamp": item.timestamp,
                            "num_samples": item.num_samples,
                        }
                    )
            except Exception as exc:
                self.last_error = repr(exc)
            finally:
                self._queue.task_done()

    def _arecord_reader_loop(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None
        while not self._stop_event.is_set():
            try:
                data = self._proc.stdout.read(self.bytes_per_chunk)
            except Exception as exc:
                self.last_error = repr(exc)
                break
            if not data:
                break
            samples = len(data) // (self.channels * self.sample_width)
            self._enqueue_chunk(data, samples, time.time())

    def _drain_stderr(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            return
        for raw in iter(self._proc.stderr.readline, b""):
            if not raw:
                break
            text = raw.decode("utf-8", errors="replace").strip()
            if text:
                self._stderr_tail.append(text)
                del self._stderr_tail[:-20]

    def _enqueue_chunk(self, data: bytes, frames: int, timestamp: float) -> None:
        index = self.total_chunks + self._queue.qsize() + self.dropped_chunks
        chunk = _AudioChunk(index=index, timestamp=timestamp, data=data, num_samples=int(frames))
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            self.dropped_chunks += 1

    def _put_sentinel(self) -> None:
        while True:
            try:
                self._queue.put_nowait(None)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                    self.dropped_chunks += 1
                except queue.Empty:
                    continue

    def _stderr_text(self) -> str:
        return "\n".join(self._stderr_tail[-10:]).strip()

    def _cleanup_after_failed_start(self) -> None:
        self._stop_event.set()
        if self._sd_stream is not None:
            try:
                self._sd_stream.close()
            except Exception:
                pass
            self._sd_stream = None
        if self._proc is not None:
            try:
                if self._proc.poll() is None:
                    self._proc.terminate()
                    self._proc.wait(timeout=0.5)
            except Exception:
                pass
            self._proc = None
        self._put_sentinel()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=0.5)
            self._writer_thread = None
        self._close_wave()
        self.active_backend = None
        self.started = False
        try:
            if os.path.exists(self.output_path) and os.path.getsize(self.output_path) <= 44:
                os.remove(self.output_path)
        except OSError:
            pass


def audio_rms_int16(raw: bytes) -> float:
    """Compute RMS for little-endian int16 PCM bytes without requiring numpy."""

    if not raw:
        return 0.0
    total = 0.0
    count = 0
    for i in range(0, len(raw) - 1, 2):
        value = int.from_bytes(raw[i : i + 2], byteorder="little", signed=True)
        total += value * value
        count += 1
    if count == 0:
        return 0.0
    return math.sqrt(total / count)
