"""Best-effort Rerun visualization for recorded episode replay.

The replay control loop only calls :meth:`ReplayVisualizer.submit`.  Rendering,
JPEG file access, and Rerun logging happen on a bounded worker queue so a slow
viewer cannot delay robot commands.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from queue import Empty, Full, Queue
import re
import threading
from typing import Any

import numpy as np

from . import rerun_visualizer as _rerun


logger = logging.getLogger(__name__)

REPLAY_PARTS = ("left_arm", "right_arm", "left_ee", "right_ee")
_VALUE_SERIES = (
    "recorded_state",
    "target_action",
    "sent_command",
    "live_measured",
)
_DIFFERENCE_SERIES = (
    "target_minus_recorded",
    "target_minus_sent",
    "target_minus_live",
    "sent_minus_live",
)
_STOP = object()


def _numeric_vector(value: Any) -> np.ndarray | None:
    """Return a finite, one-dimensional float vector or ``None``."""
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("qpos")
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        return None
    return vector


def _part_vector(group: Any, part: str) -> np.ndarray | None:
    if not isinstance(group, dict):
        return None
    return _numeric_vector(group.get(part))


def _difference(left: np.ndarray | None, right: np.ndarray | None) -> np.ndarray | None:
    if left is None or right is None or left.shape != right.shape:
        return None
    return left - right


def build_replay_series(
    frame: dict[str, Any],
    sent_actions: dict[str, Any] | None = None,
    live_states: dict[str, Any] | None = None,
) -> dict[str, dict[str, list[float]]]:
    """Build qpos and signed-error series for one replay frame.

    ``frame`` uses the EpisodeWriter schema.  ``sent_actions`` and
    ``live_states`` use the corresponding group schema, for example
    ``{"left_arm": {"qpos": [...]}}``.  Missing or shape-incompatible values
    are omitted instead of being replaced by misleading zeros.  A sent
    command is the post-limiter value published by the controller; a live
    measurement is the robot response read from ``/joint_states``.
    """
    recorded_states = frame.get("states", {}) if isinstance(frame, dict) else {}
    target_actions = frame.get("actions", {}) if isinstance(frame, dict) else {}
    sent_actions = sent_actions or {}
    live_states = live_states or {}

    result: dict[str, dict[str, list[float]]] = {}
    for part in REPLAY_PARTS:
        recorded = _part_vector(recorded_states, part)
        target = _part_vector(target_actions, part)
        sent = _part_vector(sent_actions, part)
        live = _part_vector(live_states, part)

        vectors = {
            "recorded_state": recorded,
            "target_action": target,
            "sent_command": sent,
            "live_measured": live,
            "target_minus_recorded": _difference(target, recorded),
            "target_minus_sent": _difference(target, sent),
            "target_minus_live": _difference(target, live),
            "sent_minus_live": _difference(sent, live),
        }
        part_result = {
            name: vector.tolist()
            for name, vector in vectors.items()
            if vector is not None
        }
        if part_result:
            result[part] = part_result
    return result


def _camera_mapping(value: Any) -> dict[str, str]:
    """Normalize either ``color_key -> label`` or ``label -> color_key``."""
    if isinstance(value, (list, tuple)):
        return {
            f"color_{index}": str(label)
            for index, label in enumerate(value)
            if isinstance(label, str) and label
        }
    if not isinstance(value, dict):
        return {}

    result = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, str):
            continue
        key = raw_key.strip()
        label = raw_value.strip()
        if not key or not label:
            continue
        if key.startswith("color_"):
            result[key] = label
        elif label.startswith("color_"):
            result[label] = key
    return result


def camera_labels_from_info(info: dict[str, Any] | None) -> dict[str, str]:
    """Return recorded color-key labels, preferring explicit episode metadata.

    Supported explicit keys are intentionally permissive so older experiments
    can provide either mapping direction.  AI Worker recordings made before a
    mapping was stored use the known monocular head/left-wrist/right-wrist
    ordering as a compatibility fallback.
    """
    info = info if isinstance(info, dict) else {}
    recording = info.get("recording", {})
    recording = recording if isinstance(recording, dict) else {}
    camera = recording.get("camera", {})
    camera = camera if isinstance(camera, dict) else {}
    ai_worker = recording.get("ai_worker", {})
    ai_worker = ai_worker if isinstance(ai_worker, dict) else {}

    explicit = {}
    # Apply broader legacy locations first; the dedicated camera metadata is
    # the authoritative source if the same key appears more than once.
    for container in (info, recording, ai_worker, camera):
        for key in (
            "color_keys",
            "color_key_labels",
            "color_key_map",
            "camera_key_labels",
            "camera_keys",
            "recorded_color_keys",
        ):
            mapping = _camera_mapping(container.get(key))
            if mapping:
                explicit.update(mapping)
    if explicit:
        return explicit

    robot = recording.get("robot", {})
    robot = robot if isinstance(robot, dict) else {}
    arm_name = str(robot.get("arm", "")).upper()
    if arm_name == "AI_WORKER" or bool(ai_worker):
        return {
            "color_0": "head",
            "color_1": "left_wrist",
            "color_2": "right_wrist",
        }
    return {}


def _entity_segment(value: str) -> str:
    segment = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return segment.strip("_") or "unknown"


def _status_items(status: Any, prefix: str = ""):
    if not isinstance(status, dict):
        return
    for key, value in status.items():
        path = f"{prefix}/{_entity_segment(key)}" if prefix else _entity_segment(key)
        if isinstance(value, dict):
            yield from _status_items(value, path)
        elif isinstance(value, (bool, int, float, np.number)) and math.isfinite(float(value)):
            yield path, float(value)


class ReplayVisualizer:
    """Asynchronously show replay telemetry and recorded cameras in Rerun."""

    def __init__(
        self,
        episode_dir,
        info,
        prefix="ai_worker_replay/",
        idx_window=300,
        memory_limit="300MB",
        viewer=None,
        queue_size=4,
    ):
        self.episode_dir = Path(episode_dir).expanduser().resolve()
        self.info = info if isinstance(info, dict) else {}
        normalized_prefix = str(prefix).strip("/")
        self.prefix = f"{normalized_prefix}/" if normalized_prefix else ""
        self.idx_window = max(0, int(idx_window))
        self.camera_labels = camera_labels_from_info(self.info)
        self.queue = Queue(maxsize=max(1, int(queue_size)))
        self.dropped_frames = 0
        self.logged_frames = 0
        self.last_error = None
        self._closed = False
        self._state_lock = threading.Lock()
        self._worker = None
        self._rerun_logger = None
        self.enabled = False

        try:
            self._rerun_logger = _rerun.RerunLogger(
                prefix=self.prefix,
                IdxRangeBoundary=self.idx_window,
                memory_limit=memory_limit,
                viewer=viewer,
            )
        except Exception as exc:
            self.last_error = repr(exc)
            logger.warning(
                "Replay visualization unavailable; replay will continue without it: %s",
                exc,
            )
            return
        try:
            self._send_blueprint()
        except Exception as exc:
            self.last_error = repr(exc)
            logger.warning(
                "Replay visualization blueprint failed; using the default Rerun layout: %s",
                exc,
            )

        self.enabled = True
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="replay-rerun-logger",
            daemon=True,
        )
        self._worker.start()

    def _time_ranges(self):
        if not self.idx_window:
            return None
        rrb = _rerun.rrb
        return [
            rrb.VisibleTimeRange(
                "idx",
                start=rrb.TimeRangeBoundary.cursor_relative(seq=-self.idx_window),
                end=rrb.TimeRangeBoundary.cursor_relative(),
            )
        ]

    def _send_blueprint(self):
        rr = _rerun.rr
        rrb = _rerun.rrb
        time_ranges = self._time_ranges()
        views = []
        for part in REPLAY_PARTS:
            label = part.replace("_", " ").title()
            views.append(
                rrb.TimeSeriesView(
                    origin=f"{self.prefix}{part}/qpos",
                    name=f"{label}: qpos",
                    time_ranges=time_ranges,
                    plot_legend=rrb.PlotLegend(visible=True),
                )
            )
            views.append(
                rrb.TimeSeriesView(
                    origin=f"{self.prefix}{part}/difference",
                    name=f"{label}: signed differences",
                    time_ranges=time_ranges,
                    plot_legend=rrb.PlotLegend(visible=True),
                )
            )

        views.append(
            rrb.TimeSeriesView(
                origin=f"{self.prefix}status",
                name="Replay status",
                time_ranges=time_ranges,
                plot_legend=rrb.PlotLegend(visible=True),
            )
        )

        camera_keys = list(self.camera_labels)
        if not camera_keys:
            camera_keys = [f"color_{index}" for index in range(4)]
        for camera_key in camera_keys:
            label = self.camera_labels.get(camera_key, camera_key)
            views.append(
                rrb.Spatial2DView(
                    origin=f"{self.prefix}colors/{_entity_segment(camera_key)}",
                    name=f"Camera: {label}",
                    time_ranges=time_ranges,
                )
            )
        rr.send_blueprint(rrb.Grid(contents=views, grid_columns=3))

    @staticmethod
    def _normalize_group(
        group: dict[str, Any] | None,
        wrapper_key: str,
    ) -> dict[str, Any]:
        if not isinstance(group, dict):
            return {}
        # Accept a complete frame-like wrapper in addition to the documented
        # direct state/action group for convenient replay call sites.
        if isinstance(group.get(wrapper_key), dict):
            return group[wrapper_key]
        return group

    def submit(
        self,
        frame,
        sent_actions=None,
        live_states=None,
        replay_time_s=None,
        status=None,
    ):
        """Queue one display update without waiting for the Rerun worker.

        Returns ``True`` when visualization is enabled and the newest frame was
        queued.  When full, the oldest pending display update is discarded.
        """
        if not self.enabled or self._closed or not isinstance(frame, dict):
            return False

        try:
            frame_idx = int(frame.get("idx", 0))
        except (TypeError, ValueError):
            frame_idx = 0
        frame_colors = frame.get("colors", {})
        if not isinstance(frame_colors, dict):
            frame_colors = {}
        payload = {
            "idx": frame_idx,
            "series": build_replay_series(
                frame,
                sent_actions=self._normalize_group(sent_actions, "actions"),
                live_states=self._normalize_group(live_states, "states"),
            ),
            "colors": dict(frame_colors),
            "replay_time_s": replay_time_s,
            "status": dict(status) if isinstance(status, dict) else {},
        }

        with self._state_lock:
            if self._closed:
                return False
            try:
                self.queue.put_nowait(payload)
            except Full:
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                    self.dropped_frames += 1
                except Empty:
                    pass
                try:
                    self.queue.put_nowait(payload)
                except Full:
                    self.dropped_frames += 1
                    return False
        return True

    def _log_vector(self, part: str, series_name: str, values: list[float]):
        rr = _rerun.rr
        if series_name in _VALUE_SERIES:
            category = "qpos"
        elif series_name in _DIFFERENCE_SERIES:
            category = "difference"
        else:
            return
        for index, value in enumerate(values):
            rr.log(
                f"{self.prefix}{part}/{category}/{series_name}/joint_{index}",
                rr.Scalar(float(value)),
            )

    def _resolve_color_path(self, relative_path: Any) -> Path | None:
        if not isinstance(relative_path, (str, Path)):
            return None
        path = Path(relative_path)
        candidate = path.resolve() if path.is_absolute() else (self.episode_dir / path).resolve()
        try:
            candidate.relative_to(self.episode_dir)
        except ValueError:
            logger.warning("Skipping episode image outside episode directory: %s", candidate)
            return None
        return candidate if candidate.is_file() else None

    def _log_payload(self, payload):
        rr = _rerun.rr
        rr.set_time_sequence("idx", payload["idx"])
        replay_time_s = payload.get("replay_time_s")
        if replay_time_s is not None:
            try:
                replay_time_s = float(replay_time_s)
            except (TypeError, ValueError):
                replay_time_s = None
            if replay_time_s is not None and math.isfinite(replay_time_s):
                rr.set_time_seconds("replay_time", replay_time_s)

        for part, part_series in payload["series"].items():
            for series_name, values in part_series.items():
                self._log_vector(part, series_name, values)

        for camera_key, relative_path in payload["colors"].items():
            image_path = self._resolve_color_path(relative_path)
            if image_path is not None:
                rr.log(
                    f"{self.prefix}colors/{_entity_segment(camera_key)}",
                    rr.EncodedImage(path=image_path),
                )

        for status_path, value in _status_items(payload.get("status", {})):
            rr.log(f"{self.prefix}status/{status_path}", rr.Scalar(value))

    def _worker_loop(self):
        while True:
            payload = self.queue.get()
            try:
                if payload is _STOP:
                    return
                try:
                    self._log_payload(payload)
                    self.logged_frames += 1
                except Exception as exc:
                    self.last_error = repr(exc)
                    logger.warning("Replay visualization frame failed: %s", exc)
            finally:
                self.queue.task_done()

    def close(self, drain=True):
        """Stop the display worker, optionally logging every pending update."""
        if not self.enabled:
            self._closed = True
            return
        with self._state_lock:
            if self._closed:
                return
            self._closed = True

        if drain:
            self.queue.join()
        else:
            while True:
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except Empty:
                    break
        self.queue.put(_STOP)
        if self._worker is not None:
            self._worker.join(timeout=5.0)
            if self._worker.is_alive():
                logger.warning("Replay visualization worker did not stop within 5 seconds.")
        try:
            flush = getattr(_rerun.rr, "flush", None)
            if flush is not None:
                flush()
        except Exception as exc:
            logger.warning("Failed to flush Rerun replay visualization: %s", exc)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close(drain=True)
