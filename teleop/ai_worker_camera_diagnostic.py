"""Read-only health check for the three AI Worker camera DDS topics.

This utility constructs only :class:`RobotisDDSImageClient`, whose DDS entities
are camera readers.  It never imports or constructs an arm, hand, lift, base,
trajectory, or other motor-command publisher.
"""

import argparse
from dataclasses import dataclass
import math
import sys
import time

import numpy as np

from teleop.robot_control.robotis_image_client import (
    AI_WORKER_CAMERA_TOPICS,
    RobotisDDSImageClient,
)


_CAMERAS = (
    ("head", "head_camera", "get_head_frame"),
    ("left_wrist", "left_wrist_camera", "get_left_wrist_frame"),
    ("right_wrist", "right_wrist_camera", "get_right_wrist_frame"),
)


@dataclass(frozen=True)
class CameraDiagnosticResult:
    name: str
    topic: str
    configured_shape: tuple
    frame_shape: tuple | None
    fps: float | None
    age_seconds: float | None
    status: str

    @property
    def healthy(self):
        return self.status == "OK"


def _shape_tuple(value):
    if value is None:
        return None
    shape = tuple(int(dimension) for dimension in np.asarray(value).shape)
    if len(shape) != 3 or any(dimension <= 0 for dimension in shape):
        return None
    return shape


def collect_camera_diagnostics(client, freshness, now_ns=None):
    """Return one immutable health snapshot from an image-client-compatible object."""
    freshness = float(freshness)
    if not math.isfinite(freshness) or freshness <= 0.0:
        raise ValueError("freshness must be a finite value greater than zero")
    if now_ns is None:
        now_ns = time.time_ns()

    config = client.get_cam_config()
    timestamps = client.get_frame_timestamps()
    results = []
    for name, config_key, getter_name in _CAMERAS:
        camera_config = config.get(config_key, {})
        configured_shape = tuple(camera_config.get("image_shape", ()))
        frame = getattr(client, getter_name)()
        bgr = None if frame is None else getattr(frame, "bgr", None)
        frame_shape = _shape_tuple(bgr)
        raw_fps = None if frame is None else getattr(frame, "fps", None)
        try:
            fps = float(raw_fps) if raw_fps is not None else None
        except (TypeError, ValueError):
            fps = None

        receive_time_ns = timestamps.get(name, {}).get("receive_time_ns")
        age_seconds = None
        if receive_time_ns is not None:
            try:
                age_seconds = max(0.0, (int(now_ns) - int(receive_time_ns)) / 1e9)
            except (TypeError, ValueError, OverflowError):
                age_seconds = None

        if frame_shape is None or age_seconds is None:
            status = "MISSING"
        elif fps is None or not math.isfinite(fps) or fps <= 0.0:
            status = "INVALID"
        elif age_seconds > freshness:
            status = "STALE"
        else:
            status = "OK"

        results.append(
            CameraDiagnosticResult(
                name=name,
                topic=AI_WORKER_CAMERA_TOPICS[name],
                configured_shape=configured_shape,
                frame_shape=frame_shape,
                fps=fps,
                age_seconds=age_seconds,
                status=status,
            )
        )
    return tuple(results)


def format_camera_result(result):
    frame_shape = "missing" if result.frame_shape is None else "x".join(
        str(dimension) for dimension in result.frame_shape
    )
    configured_shape = "missing" if not result.configured_shape else "x".join(
        str(dimension) for dimension in result.configured_shape
    )
    fps = "missing" if result.fps is None else f"{result.fps:.2f}"
    age = "missing" if result.age_seconds is None else f"{result.age_seconds:.3f}s"
    return (
        f"{result.name}: status={result.status} topic={result.topic} "
        f"configured_shape={configured_shape} frame_shape={frame_shape} "
        f"fps={fps} age={age}"
    )


def run_camera_diagnostic(
    client,
    duration,
    freshness,
    output=print,
    monotonic=time.monotonic,
    sleep=time.sleep,
):
    """Observe for ``duration`` seconds, print a final snapshot, and return an exit code."""
    duration = float(duration)
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError("duration must be a finite value greater than or equal to zero")
    deadline = monotonic() + duration
    while True:
        remaining = deadline - monotonic()
        if remaining <= 0.0:
            break
        sleep(min(0.05, remaining))

    results = collect_camera_diagnostics(client, freshness=freshness)
    for result in results:
        output(format_camera_result(result))
    healthy = all(result.healthy for result in results)
    output(
        "AI Worker camera diagnostic: PASS (all three frames are fresh)"
        if healthy
        else "AI Worker camera diagnostic: FAIL (missing, invalid, or stale frame)"
    )
    return 0 if healthy else 1


def _nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _positive_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite value greater than zero")
    return parsed


def _parser():
    parser = argparse.ArgumentParser(
        description="Read-only head/wrist camera DDS freshness diagnostic for AI Worker."
    )
    parser.add_argument(
        "--domain",
        "--domain-id",
        dest="domain_id",
        type=_nonnegative_int,
        default=None,
        help="CycloneDDS/ROS domain ID (robotis_lab commonly uses 30).",
    )
    parser.add_argument(
        "--duration",
        type=_positive_float,
        default=5.0,
        help="Seconds to collect frames before reporting (default: 5).",
    )
    parser.add_argument(
        "--freshness",
        type=_positive_float,
        default=0.5,
        help="Maximum allowed host receive age in seconds (default: 0.5).",
    )
    return parser


def main(argv=None, client_factory=RobotisDDSImageClient, output=print):
    args = _parser().parse_args(argv)
    output(
        f"AI Worker camera diagnostic: domain={args.domain_id} "
        f"duration={args.duration:.2f}s freshness={args.freshness:.3f}s"
    )
    output("DDS mode: camera readers only; no motor command publisher is created.")
    try:
        client = client_factory(domain_id=args.domain_id)
    except Exception as exc:
        output(f"AI Worker camera diagnostic: SETUP_ERROR {type(exc).__name__}: {exc}")
        return 2

    try:
        return run_camera_diagnostic(
            client,
            duration=args.duration,
            freshness=args.freshness,
            output=output,
        )
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
