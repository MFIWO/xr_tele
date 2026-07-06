#!/usr/bin/env python3
"""Estimate head-camera and robot-base pose from an AprilTag board.

Frame and transform convention:
  T_a_b maps points expressed in frame b into frame a.

  T_camera_tag:
    Pose returned by AprilTag PnP. It maps tag-frame points into the
    head RGB camera frame.

  T_board_tag:
    Known board layout pose. The board is on z=0, x points right across the
    printed board, y points down the printed board, and +z points into the
    printed board. This assumes tags are printed upright in row-major order.

  T_board_camera:
    Camera pose in board coordinates, computed per detected tag as:
      T_board_camera = T_board_tag @ inv(T_camera_tag)

  T_camera_robot_base:
    Extrinsic mapping robot-base-frame points into the head-camera frame.
    The default is identity, so robot_base == camera unless configured.

Dependencies:
  pip install opencv-python numpy scipy pupil-apriltags

The script prefers pupil-apriltags because it ships prebuilt wheels on common
platforms. If pupil-apriltags is unavailable, it falls back to apriltag when
that package is installed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from scipy.spatial.transform import Rotation
except ImportError:
    Rotation = None


TAG_FAMILY = "tag36h11"


def require_runtime_dependencies() -> None:
    missing = []
    if cv2 is None:
        missing.append("opencv-python")
    if np is None:
        missing.append("numpy")
    if Rotation is None:
        missing.append("scipy")
    if missing:
        raise RuntimeError(
            "missing Python dependencies: "
            + ", ".join(missing)
            + "\nInstall with: pip install opencv-python numpy scipy pupil-apriltags"
        )


@dataclass
class TagPoseObservation:
    tag_id: int
    corners_px: np.ndarray
    center_px: np.ndarray
    T_camera_tag: np.ndarray
    decision_margin: float | None = None
    hamming: int | None = None
    pose_error: float | None = None


class AprilTagBackend:
    def __init__(self) -> None:
        self.name = ""
        self.detector: Any = None
        self._mode = ""

        try:
            from pupil_apriltags import Detector as PupilDetector

            self.detector = PupilDetector(
                families=TAG_FAMILY,
                nthreads=1,
                quad_decimate=1.0,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
                debug=0,
            )
            self.name = "pupil_apriltags"
            self._mode = "pupil"
            return
        except ImportError as pupil_error:
            self._pupil_error = pupil_error

        try:
            import apriltag

            options = apriltag.DetectorOptions(families=TAG_FAMILY)
            self.detector = apriltag.Detector(options)
            self.name = "apriltag"
            self._mode = "apriltag"
            return
        except ImportError as apriltag_error:
            raise RuntimeError(
                "No AprilTag backend is installed. Install one of:\n"
                "  pip install pupil-apriltags\n"
                "  pip install apriltag"
            ) from apriltag_error

    def detect(
        self,
        gray: np.ndarray,
        camera_params: tuple[float, float, float, float],
        tag_size_m: float,
    ) -> list[TagPoseObservation]:
        if self._mode == "pupil":
            return self._detect_pupil(gray, camera_params, tag_size_m)
        if self._mode == "apriltag":
            return self._detect_apriltag(gray, camera_params, tag_size_m)
        raise RuntimeError("AprilTag backend is not initialized")

    def _detect_pupil(
        self,
        gray: np.ndarray,
        camera_params: tuple[float, float, float, float],
        tag_size_m: float,
    ) -> list[TagPoseObservation]:
        detections = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=tag_size_m,
        )

        observations: list[TagPoseObservation] = []
        for det in detections:
            T_camera_tag = np.eye(4, dtype=np.float64)
            T_camera_tag[:3, :3] = np.asarray(det.pose_R, dtype=np.float64)
            T_camera_tag[:3, 3] = np.asarray(det.pose_t, dtype=np.float64).reshape(3)
            observations.append(
                TagPoseObservation(
                    tag_id=int(det.tag_id),
                    corners_px=np.asarray(det.corners, dtype=np.float64),
                    center_px=np.asarray(det.center, dtype=np.float64),
                    T_camera_tag=T_camera_tag,
                    decision_margin=_optional_float(det, "decision_margin"),
                    hamming=_optional_int(det, "hamming"),
                    pose_error=_optional_float(det, "pose_err"),
                )
            )
        return observations

    def _detect_apriltag(
        self,
        gray: np.ndarray,
        camera_params: tuple[float, float, float, float],
        tag_size_m: float,
    ) -> list[TagPoseObservation]:
        detections = self.detector.detect(gray)

        observations: list[TagPoseObservation] = []
        for det in detections:
            pose_result = self.detector.detection_pose(
                det,
                camera_params=list(camera_params),
                tag_size=tag_size_m,
            )
            if isinstance(pose_result, tuple):
                T_camera_tag = np.asarray(pose_result[0], dtype=np.float64)
                pose_error = float(pose_result[-1]) if len(pose_result) > 1 else None
            else:
                T_camera_tag = np.asarray(pose_result, dtype=np.float64)
                pose_error = None
            if T_camera_tag.shape != (4, 4):
                raise RuntimeError(
                    f"apriltag.detection_pose returned shape {T_camera_tag.shape}, expected 4x4"
                )
            observations.append(
                TagPoseObservation(
                    tag_id=int(det.tag_id),
                    corners_px=np.asarray(det.corners, dtype=np.float64),
                    center_px=np.asarray(det.center, dtype=np.float64),
                    T_camera_tag=T_camera_tag,
                    decision_margin=_optional_float(det, "decision_margin"),
                    hamming=_optional_int(det, "hamming"),
                    pose_error=pose_error,
                )
            )
        return observations


def _optional_float(obj: Any, name: str) -> float | None:
    value = getattr(obj, name, None)
    if value is None:
        return None
    return float(value)


def _optional_int(obj: Any, name: str) -> int | None:
    value = getattr(obj, name, None)
    if value is None:
        return None
    return int(value)


def parse_layout(text: str) -> tuple[int, int]:
    parts = text.lower().replace(" ", "").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("tag layout must look like 3x2")
    try:
        cols, rows = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tag layout must contain integers, e.g. 3x2") from exc
    if cols <= 0 or rows <= 0:
        raise argparse.ArgumentTypeError("tag layout dimensions must be positive")
    return cols, rows


def parse_tag_ids(text: str) -> list[int]:
    try:
        ids = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tag ids must be comma-separated integers") from exc
    if not ids:
        raise argparse.ArgumentTypeError("at least one tag id is required")
    if len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError("tag ids must be unique")
    return ids


def parse_matrix_4x4(text: str) -> np.ndarray:
    if np is None:
        raise argparse.ArgumentTypeError("numpy is required to parse a 4x4 matrix")
    if text.strip().lower() in {"i", "identity"}:
        return np.eye(4, dtype=np.float64)
    normalized = text.replace(",", " ")
    try:
        values = [float(part) for part in normalized.split()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("matrix values must be numbers") from exc
    if len(values) != 16:
        raise argparse.ArgumentTypeError(
            "matrix must contain 16 row-major numbers, e.g. '1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1'"
        )
    T = np.asarray(values, dtype=np.float64).reshape(4, 4)
    if not np.allclose(T[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
        raise argparse.ArgumentTypeError("last row of a homogeneous transform must be 0,0,0,1")
    return T


def inverse_transform(T_a_b: np.ndarray) -> np.ndarray:
    R_a_b = T_a_b[:3, :3]
    t_a_b = T_a_b[:3, 3]
    T_b_a = np.eye(4, dtype=np.float64)
    T_b_a[:3, :3] = R_a_b.T
    T_b_a[:3, 3] = -R_a_b.T @ t_a_b
    return T_b_a


def build_board_tag_transforms(
    cols: int,
    rows: int,
    tag_ids: list[int],
    tag_size_m: float,
    gap_x_m: float,
    gap_y_m: float,
    board_origin: str,
) -> dict[int, np.ndarray]:
    expected_count = cols * rows
    if len(tag_ids) != expected_count:
        raise ValueError(
            f"tag layout {cols}x{rows} expects {expected_count} ids, got {len(tag_ids)}"
        )

    pitch_x_m = tag_size_m + gap_x_m
    pitch_y_m = tag_size_m + gap_y_m
    if board_origin == "center":
        x0 = -0.5 * (cols - 1) * pitch_x_m
        y0 = -0.5 * (rows - 1) * pitch_y_m
    elif board_origin == "tag0":
        x0 = 0.0
        y0 = 0.0
    else:
        raise ValueError(f"unsupported board origin: {board_origin}")

    transforms: dict[int, np.ndarray] = {}
    for index, tag_id in enumerate(tag_ids):
        row = index // cols
        col = index % cols
        T_board_tag = np.eye(4, dtype=np.float64)
        T_board_tag[:3, 3] = np.array(
            [
                x0 + col * pitch_x_m,
                y0 + row * pitch_y_m,
                0.0,
            ],
            dtype=np.float64,
        )
        transforms[tag_id] = T_board_tag
    return transforms


def resolve_effective_tag_ids(
    args: argparse.Namespace,
    observations: list[TagPoseObservation],
    cols: int,
    rows: int,
) -> tuple[list[int], str]:
    expected_count = cols * rows

    cached_tag_ids = getattr(args, "_effective_tag_ids", None)
    if cached_tag_ids is not None:
        return list(cached_tag_ids), getattr(args, "_effective_tag_id_source", "auto_cached")

    configured_tag_ids = list(args.tag_ids)
    configured_has_expected_count = len(configured_tag_ids) == expected_count

    if args.auto_tag_ids or not configured_has_expected_count:
        reason = "auto_detected" if args.auto_tag_ids else "auto_detected_tag_id_count_mismatch"
        inferred_tag_ids = infer_board_tag_ids_from_observations(observations, cols, rows)
        cache_effective_tag_ids(args, inferred_tag_ids, reason)
        return inferred_tag_ids, reason

    configured_ids = set(configured_tag_ids)
    detected_ids = {obs.tag_id for obs in observations}
    if (
        detected_ids
        and detected_ids.isdisjoint(configured_ids)
        and not args.strict_tag_ids
        and not args.ignore_unknown_tags
    ):
        inferred_tag_ids = infer_board_tag_ids_from_observations(observations, cols, rows)
        reason = "auto_detected_no_configured_matches"
        cache_effective_tag_ids(args, inferred_tag_ids, reason)
        return inferred_tag_ids, reason

    return configured_tag_ids, "configured"


def cache_effective_tag_ids(args: argparse.Namespace, tag_ids: list[int], source: str) -> None:
    args._effective_tag_ids = list(tag_ids)
    args._effective_tag_id_source = source


def infer_board_tag_ids_from_observations(
    observations: list[TagPoseObservation],
    cols: int,
    rows: int,
) -> list[int]:
    expected_count = cols * rows
    unique_observations = best_observation_per_tag_id(observations)
    if len(unique_observations) < expected_count:
        detected_ids = sorted(obs.tag_id for obs in unique_observations)
        raise RuntimeError(
            f"need {expected_count} visible tags to auto-infer board ids for {cols}x{rows}, "
            f"but only detected {len(unique_observations)} ids: {detected_ids}"
        )

    selected = sorted(unique_observations, key=tag_corner_area_px, reverse=True)[:expected_count]
    ordered = order_observations_row_major(selected, cols, rows)
    return [obs.tag_id for obs in ordered]


def best_observation_per_tag_id(observations: list[TagPoseObservation]) -> list[TagPoseObservation]:
    best_by_id: dict[int, TagPoseObservation] = {}
    best_area_by_id: dict[int, float] = {}
    for obs in observations:
        area = tag_corner_area_px(obs)
        if obs.tag_id not in best_by_id or area > best_area_by_id[obs.tag_id]:
            best_by_id[obs.tag_id] = obs
            best_area_by_id[obs.tag_id] = area
    return list(best_by_id.values())


def tag_corner_area_px(obs: TagPoseObservation) -> float:
    corners = np.asarray(obs.corners_px, dtype=np.float64)
    x = corners[:, 0]
    y = corners[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def order_observations_row_major(
    observations: list[TagPoseObservation],
    cols: int,
    rows: int,
) -> list[TagPoseObservation]:
    sorted_by_y = sorted(observations, key=lambda obs: (float(obs.center_px[1]), float(obs.center_px[0])))
    ordered: list[TagPoseObservation] = []
    for row in range(rows):
        row_observations = sorted_by_y[row * cols : (row + 1) * cols]
        ordered.extend(sorted(row_observations, key=lambda obs: float(obs.center_px[0])))
    return ordered


def fuse_T_board_camera(candidates: list[np.ndarray]) -> np.ndarray:
    if not candidates:
        raise ValueError("cannot fuse an empty pose list")
    translations = np.stack([T[:3, 3] for T in candidates], axis=0)
    rotations = Rotation.from_matrix(np.stack([T[:3, :3] for T in candidates], axis=0))

    T_board_camera = np.eye(4, dtype=np.float64)
    T_board_camera[:3, 3] = translations.mean(axis=0)
    try:
        T_board_camera[:3, :3] = rotations.mean().as_matrix()
    except Exception:
        # Fallback for older SciPy builds: align quaternion signs, then average.
        quats = rotations.as_quat()
        for i in range(1, len(quats)):
            if np.dot(quats[0], quats[i]) < 0.0:
                quats[i] *= -1.0
        q_mean = quats.mean(axis=0)
        q_mean /= np.linalg.norm(q_mean)
        T_board_camera[:3, :3] = Rotation.from_quat(q_mean).as_matrix()
    return T_board_camera


def make_pose_json(T_a_b: np.ndarray) -> dict[str, Any]:
    rotation = Rotation.from_matrix(T_a_b[:3, :3])
    return {
        "matrix": matrix_to_list(T_a_b),
        "translation_m": vector_to_list(T_a_b[:3, 3]),
        "quaternion_xyzw": vector_to_list(rotation.as_quat()),
        "euler_xyz_deg": vector_to_list(rotation.as_euler("xyz", degrees=True)),
    }


def matrix_to_list(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def vector_to_list(vector: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(vector).reshape(-1)]


def camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def draw_debug_image(
    image_bgr: np.ndarray,
    observations: list[TagPoseObservation],
    K: np.ndarray,
    tag_size_m: float,
) -> np.ndarray:
    debug = image_bgr.copy()
    for obs in observations:
        corners = np.round(obs.corners_px).astype(np.int32)
        distance_m = float(np.linalg.norm(obs.T_camera_tag[:3, 3]))

        cv2.polylines(debug, [corners], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        for index, point in enumerate(corners):
            cv2.circle(debug, tuple(point), 4, (0, 128 + index * 30, 255 - index * 40), -1, cv2.LINE_AA)

        label_xy = tuple(np.round(obs.center_px + np.array([8.0, -8.0])).astype(np.int32))
        put_text_with_outline(debug, f"id {obs.tag_id}  {distance_m:.3f} m", label_xy)
        draw_tag_axes(debug, obs.T_camera_tag, K, axis_len_m=0.65 * tag_size_m)
    return debug


def draw_tag_axes(
    image_bgr: np.ndarray,
    T_camera_tag: np.ndarray,
    K: np.ndarray,
    axis_len_m: float,
) -> None:
    points_tag = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_len_m, 0.0, 0.0],
            [0.0, axis_len_m, 0.0],
            [0.0, 0.0, axis_len_m],
        ],
        dtype=np.float64,
    )
    R_camera_tag = T_camera_tag[:3, :3]
    t_camera_tag = T_camera_tag[:3, 3].reshape(3, 1)

    if np.any((R_camera_tag @ points_tag.T + t_camera_tag)[2, :] <= 0.0):
        return

    rvec, _ = cv2.Rodrigues(R_camera_tag)
    image_points, _ = cv2.projectPoints(
        points_tag,
        rvec,
        t_camera_tag,
        K,
        distCoeffs=np.zeros(5),
    )
    pts = np.round(image_points.reshape(-1, 2)).astype(np.int32)
    origin = tuple(pts[0])
    x_pt = tuple(pts[1])
    y_pt = tuple(pts[2])
    z_pt = tuple(pts[3])

    cv2.line(image_bgr, origin, x_pt, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(image_bgr, origin, y_pt, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(image_bgr, origin, z_pt, (255, 0, 0), 2, cv2.LINE_AA)
    put_text_with_outline(image_bgr, "x", x_pt, color=(0, 0, 255), scale=0.45)
    put_text_with_outline(image_bgr, "y", y_pt, color=(0, 160, 0), scale=0.45)
    put_text_with_outline(image_bgr, "z", z_pt, color=(255, 0, 0), scale=0.45)


def put_text_with_outline(
    image_bgr: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.55,
) -> None:
    cv2.putText(image_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate head-camera and robot-base pose from a tag36h11 AprilTag board.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=Path, help="Path to head RGB camera JPG/PNG image")
    parser.add_argument("--teleimager-host", type=str, help="Read live head frames from teleimager ImageClient")
    parser.add_argument("--teleimager-request-port", type=int, default=60000, help="Teleimager camera config request port")
    parser.add_argument(
        "--teleimager-head-only",
        action="store_true",
        help="In live mode, subscribe only to head_camera instead of ImageClient's enabled camera set",
    )
    parser.add_argument(
        "--teleimager-head-port",
        type=int,
        default=None,
        help="Head camera ZMQ port for --teleimager-head-only; if omitted, request teleimager config",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Live mode frame limit; 0 means run until Ctrl-C")
    parser.add_argument("--print-every-s", type=nonnegative_float, default=0.25, help="Minimum live terminal print interval")
    parser.add_argument("--live-write-every-s", type=nonnegative_float, default=1.0, help="Minimum live JSON/debug write interval")
    parser.add_argument("--debug-display", action="store_true", help="Show live debug image with cv2.imshow")
    parser.add_argument("--fx", required=True, type=positive_float, help="RGB camera focal length fx in pixels")
    parser.add_argument("--fy", required=True, type=positive_float, help="RGB camera focal length fy in pixels")
    parser.add_argument("--cx", required=True, type=float, help="RGB camera principal point cx in pixels")
    parser.add_argument("--cy", required=True, type=float, help="RGB camera principal point cy in pixels")
    parser.add_argument(
        "--crop-eye",
        choices=("none", "left", "right"),
        default="none",
        help="Crop a side-by-side binocular image to one eye before detection",
    )
    parser.add_argument("--tag-size-m", type=positive_float, default=0.1, help="Measured black-square tag side length in meters")
    parser.add_argument("--tag-layout", type=parse_layout, default=parse_layout("3x2"), help="Tag grid as COLSxROWS")
    parser.add_argument("--tag-ids", type=parse_tag_ids, default=parse_tag_ids("0,1,2,3,4,5"), help="Row-major tag ids, e.g. 0,1,2,3,4,5")
    parser.add_argument(
        "--auto-tag-ids",
        action="store_true",
        help="Infer row-major board ids from detected tags and cache the first full-board live detection",
    )
    parser.add_argument(
        "--strict-tag-ids",
        action="store_true",
        help="Fail instead of auto-inferring when detected ids do not match --tag-ids",
    )
    parser.add_argument("--gap-x-m", type=nonnegative_float, default=0.0, help="Horizontal gap between black tag squares in meters")
    parser.add_argument("--gap-y-m", type=nonnegative_float, default=0.0, help="Vertical gap between black tag squares in meters")
    parser.add_argument("--board-origin", choices=("center", "tag0"), default="center", help="Board frame origin")
    parser.add_argument("--output-json", type=Path, default=Path("result.json"), help="Output JSON path")
    parser.add_argument("--debug-image", type=Path, default=Path("debug.jpg"), help="Debug image path")
    parser.add_argument(
        "--T-camera-robot-base",
        "--t-camera-robot-base",
        dest="T_camera_robot_base",
        type=parse_matrix_4x4,
        default=None,
        help="16 row-major numbers for extrinsic mapping robot_base -> camera",
    )
    parser.add_argument(
        "--T-robot-base-camera",
        "--t-robot-base-camera",
        dest="T_robot_base_camera",
        type=parse_matrix_4x4,
        default=None,
        help="16 row-major numbers for extrinsic mapping camera -> robot_base",
    )
    parser.add_argument(
        "--ignore-unknown-tags",
        action="store_true",
        help="Ignore detected tags that are not listed in --tag-ids",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.image is None and args.teleimager_host is None:
        raise ValueError("provide either --image or --teleimager-host")
    if args.image is not None and args.teleimager_host is not None:
        raise ValueError("provide only one of --image or --teleimager-host")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be non-negative")
    if args.T_camera_robot_base is not None and args.T_robot_base_camera is not None:
        raise ValueError("provide only one of --T-camera-robot-base or --T-robot-base-camera")
    if args.auto_tag_ids and args.strict_tag_ids:
        raise ValueError("provide only one of --auto-tag-ids or --strict-tag-ids")
    cols, rows = args.tag_layout
    expected_count = cols * rows
    if args.strict_tag_ids and len(args.tag_ids) != expected_count:
        raise ValueError(
            f"--tag-layout {cols}x{rows} needs {expected_count} ids, got {len(args.tag_ids)}"
        )


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        validate_args(args)
        if args.teleimager_host is not None:
            return run_live(args)
        result = estimate(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    write_result_files(args, result)
    print_result_summary(result)
    print(f"Wrote JSON: {args.output_json}")
    print(f"Wrote debug image: {args.debug_image}")
    return 0


def estimate(args: argparse.Namespace) -> dict[str, Any]:
    image_path: Path = args.image
    if not image_path.exists():
        raise FileNotFoundError(f"image does not exist: {image_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"image path is not a file: {image_path}")

    require_runtime_dependencies()

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"OpenCV could not read image: {image_path}")

    backend = AprilTagBackend()
    result, debug = estimate_from_bgr(args, image_bgr, image_label=str(image_path), backend=backend)
    write_debug_image(args.debug_image, debug)
    return result


def estimate_from_bgr(
    args: argparse.Namespace,
    image_bgr: np.ndarray,
    *,
    image_label: str,
    backend: AprilTagBackend,
) -> tuple[dict[str, Any], np.ndarray]:
    original_shape = image_bgr.shape
    image_bgr = crop_eye_image(image_bgr, args.crop_eye)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(gray)

    fx, fy, cx, cy = float(args.fx), float(args.fy), float(args.cx), float(args.cy)
    K = camera_matrix(fx, fy, cx, cy)
    camera_params = (fx, fy, cx, cy)

    observations = backend.detect(gray, camera_params, float(args.tag_size_m))
    if not observations:
        raise RuntimeError(f"no {TAG_FAMILY} AprilTags detected in {image_label}")

    cols, rows = args.tag_layout
    effective_tag_ids, tag_id_source = resolve_effective_tag_ids(args, observations, cols, rows)
    T_board_tag_by_id = build_board_tag_transforms(
        cols=cols,
        rows=rows,
        tag_ids=effective_tag_ids,
        tag_size_m=float(args.tag_size_m),
        gap_x_m=float(args.gap_x_m),
        gap_y_m=float(args.gap_y_m),
        board_origin=args.board_origin,
    )

    allowed_ids = set(T_board_tag_by_id)
    unknown_ids = sorted({obs.tag_id for obs in observations if obs.tag_id not in allowed_ids})
    auto_tag_ids_active = tag_id_source.startswith("auto")
    if unknown_ids and not args.ignore_unknown_tags and not auto_tag_ids_active:
        raise ValueError(
            f"detected tag ids not present in effective board ids: {unknown_ids}; "
            "fix --tag-ids/--tag-layout, pass --auto-tag-ids, or pass --ignore-unknown-tags"
        )

    used_observations = [obs for obs in observations if obs.tag_id in allowed_ids]
    if not used_observations:
        raise RuntimeError("AprilTags were detected, but none matched the effective board ids")
    used_observations.sort(key=lambda obs: effective_tag_ids.index(obs.tag_id))

    per_tag_json: list[dict[str, Any]] = []
    T_board_camera_candidates: list[np.ndarray] = []
    for obs in used_observations:
        T_camera_tag = obs.T_camera_tag
        T_board_tag = T_board_tag_by_id[obs.tag_id]
        T_board_camera_from_tag = T_board_tag @ inverse_transform(T_camera_tag)
        T_board_camera_candidates.append(T_board_camera_from_tag)
        per_tag_json.append(
            {
                "tag_id": obs.tag_id,
                "corners_px": matrix_to_list(obs.corners_px),
                "center_px": vector_to_list(obs.center_px),
                "camera_tag_distance_m": float(np.linalg.norm(T_camera_tag[:3, 3])),
                "T_camera_tag": make_pose_json(T_camera_tag),
                "T_board_tag": make_pose_json(T_board_tag),
                "T_board_camera_from_this_tag": make_pose_json(T_board_camera_from_tag),
                "decision_margin": obs.decision_margin,
                "hamming": obs.hamming,
                "pose_error": obs.pose_error,
            }
        )

    T_board_camera = fuse_T_board_camera(T_board_camera_candidates)
    T_camera_robot_base = get_T_camera_robot_base(args)
    T_robot_base_camera = inverse_transform(T_camera_robot_base)
    T_board_robot_base = T_board_camera @ T_camera_robot_base

    debug = draw_debug_image(image_bgr, used_observations, K, float(args.tag_size_m))
    result = {
        "image": image_label,
        "timestamp_unix_s": time.time(),
        "input_shape_hw": [int(original_shape[0]), int(original_shape[1])],
        "processed_shape_hw": [int(image_bgr.shape[0]), int(image_bgr.shape[1])],
        "crop_eye": args.crop_eye,
        "apriltag_backend": backend.name,
        "tag_family": TAG_FAMILY,
        "camera_intrinsic": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "K": matrix_to_list(K),
            "distortion_model": "assumed zero distortion",
        },
        "tag_size_m": float(args.tag_size_m),
        "board": {
            "layout": f"{cols}x{rows}",
            "ids_row_major": [
                effective_tag_ids[row * cols : (row + 1) * cols]
                for row in range(rows)
            ],
            "configured_ids_row_major": [
                args.tag_ids[row * cols : (row + 1) * cols]
                for row in range((len(args.tag_ids) + cols - 1) // cols)
            ],
            "tag_id_source": tag_id_source,
            "gap_x_m": float(args.gap_x_m),
            "gap_y_m": float(args.gap_y_m),
            "origin": args.board_origin,
            "frame_convention": {
                "x": "right across the printed board",
                "y": "down across the printed board",
                "z": "into the printed board",
            },
            "T_board_tag_by_id": {
                str(tag_id): make_pose_json(T_board_tag)
                for tag_id, T_board_tag in sorted(T_board_tag_by_id.items())
            },
        },
        "detected_tag_ids": [obs.tag_id for obs in observations],
        "used_tag_ids": [obs.tag_id for obs in used_observations],
        "ignored_unknown_tag_ids": unknown_ids if args.ignore_unknown_tags or auto_tag_ids_active else [],
        "per_tag": per_tag_json,
        "fusion": {
            "method": "mean translation plus scipy.spatial.transform.Rotation.mean",
            "num_tags": len(T_board_camera_candidates),
            "T_board_camera": make_pose_json(T_board_camera),
            "T_camera_board": make_pose_json(inverse_transform(T_board_camera)),
            "board_camera_distance_m": float(np.linalg.norm(T_board_camera[:3, 3])),
        },
        "extrinsic": {
            "direction_used": "T_camera_robot_base maps robot_base -> camera",
            "T_camera_robot_base": make_pose_json(T_camera_robot_base),
            "T_robot_base_camera": make_pose_json(T_robot_base_camera),
        },
        "robot_base": {
            "formula": "T_board_robot_base = T_board_camera @ T_camera_robot_base",
            "T_board_robot_base": make_pose_json(T_board_robot_base),
            "T_robot_base_board": make_pose_json(inverse_transform(T_board_robot_base)),
            "board_robot_base_distance_m": float(np.linalg.norm(T_board_robot_base[:3, 3])),
        },
        "debug_image": str(args.debug_image),
    }
    return result, debug


def run_live(args: argparse.Namespace) -> int:
    require_runtime_dependencies()
    client = make_live_frame_source(args)
    backend = AprilTagBackend()

    print(
        "Live teleimager AprilTag pose estimation started: "
        f"host={args.teleimager_host} crop_eye={args.crop_eye}"
    )
    print("Press Ctrl-C to stop.")

    frame_count = 0
    success_count = 0
    last_print_s = 0.0
    last_write_s = 0.0
    try:
        while args.max_frames == 0 or frame_count < args.max_frames:
            tele_image = client.get_head_frame()
            bgr = tele_image.bgr
            if bgr is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            try:
                result, debug = estimate_from_bgr(
                    args,
                    bgr,
                    image_label=f"teleimager://{args.teleimager_host}/head_camera",
                    backend=backend,
                )
            except Exception as exc:
                now_s = time.time()
                debug = draw_no_pose_debug(bgr, args.crop_eye, str(exc))
                if now_s - last_write_s >= args.live_write_every_s:
                    write_debug_image(args.debug_image, debug)
                    last_write_s = now_s
                if args.debug_display:
                    cv2.imshow("AprilTag pose debug", debug)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                if now_s - last_print_s >= args.print_every_s:
                    print(f"[frame {frame_count}] no pose: {exc}")
                    last_print_s = now_s
                continue

            success_count += 1
            now_s = time.time()
            if now_s - last_write_s >= args.live_write_every_s:
                write_result_files(args, result)
                write_debug_image(args.debug_image, debug)
                last_write_s = now_s
            if now_s - last_print_s >= args.print_every_s:
                print_result_summary(result, prefix=f"[frame {frame_count} ok {success_count}] ")
                last_print_s = now_s

            if args.debug_display:
                cv2.imshow("AprilTag pose debug", debug)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        print("\nStopping live estimator.")
    finally:
        client.close()
        if args.debug_display:
            cv2.destroyAllWindows()

    return 0


def make_live_frame_source(args: argparse.Namespace):
    if args.teleimager_head_only:
        return HeadOnlyTeleimagerClient(
            host=args.teleimager_host,
            request_port=args.teleimager_request_port,
            head_port=args.teleimager_head_port,
        )

    ImageClient = import_teleimager_image_client()
    return ImageClient(
        host=args.teleimager_host,
        request_port=args.teleimager_request_port,
        request_bgr=True,
    )


class HeadOnlyTeleimagerClient:
    def __init__(self, host: str, request_port: int, head_port: int | None) -> None:
        ZMQ_Requester, ZMQ_SubscriberManager = import_teleimager_head_only_components()
        self._host = host
        self._requester = None
        self._subscriber_manager = ZMQ_SubscriberManager.get_instance()

        if head_port is None:
            self._requester = ZMQ_Requester(host, request_port)
            cam_config = self._requester.request()
            if cam_config is None:
                raise RuntimeError("failed to get teleimager camera configuration")
            head_config = cam_config.get("head_camera", {})
            if not head_config.get("enable_zmq", False):
                raise RuntimeError("teleimager head_camera enable_zmq is false")
            head_port = int(head_config["zmq_port"])

        self._head_port = int(head_port)
        self._subscriber_manager.subscribe(self._host, self._head_port, request_bgr=True)

    def get_head_frame(self):
        return self._subscriber_manager.subscribe(self._host, self._head_port, request_bgr=True)

    def close(self) -> None:
        self._subscriber_manager.close()
        if self._requester is not None:
            self._requester.close()


def ensure_teleimager_path() -> None:
    script_dir = Path(__file__).resolve().parent
    candidate = script_dir.parent / "teleimager" / "src"
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def import_teleimager_image_client():
    try:
        from teleimager.image_client import ImageClient

        return ImageClient
    except ImportError:
        ensure_teleimager_path()
        try:
            from teleimager.image_client import ImageClient

            return ImageClient
        except ImportError as exc:
            raise RuntimeError(
                "could not import teleimager.image_client. Run from the xr_teleoperate "
                "environment, or install teleimager in the active Python environment."
            ) from exc


def import_teleimager_head_only_components():
    try:
        from teleimager.image_client import ZMQ_Requester, ZMQ_SubscriberManager

        return ZMQ_Requester, ZMQ_SubscriberManager
    except ImportError:
        ensure_teleimager_path()
        try:
            from teleimager.image_client import ZMQ_Requester, ZMQ_SubscriberManager

            return ZMQ_Requester, ZMQ_SubscriberManager
        except ImportError as exc:
            raise RuntimeError(
                "could not import teleimager.image_client head-only components. Run from "
                "the xr_teleoperate environment, or install teleimager in the active Python environment."
            ) from exc


def write_result_files(args: argparse.Namespace, result: dict[str, Any]) -> None:
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")


def write_debug_image(debug_image_path: Path, debug: np.ndarray) -> None:
    debug_image_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(debug_image_path), debug):
        raise RuntimeError(f"failed to write debug image: {debug_image_path}")


def draw_no_pose_debug(image_bgr: np.ndarray, crop_eye: str, message: str) -> np.ndarray:
    debug = crop_eye_image(image_bgr, crop_eye)
    debug = debug.copy()
    put_text_with_outline(debug, f"NO APRILTAG DETECTED  crop={crop_eye}", (18, 34), color=(0, 255, 255), scale=0.75)
    put_text_with_outline(debug, message[:110], (18, 68), color=(255, 255, 255), scale=0.48)
    height, width = debug.shape[:2]
    cv2.rectangle(debug, (0, 0), (width - 1, height - 1), (0, 255, 255), 2)
    return debug


def print_result_summary(result: dict[str, Any], prefix: str = "") -> None:
    camera_translation = result["fusion"]["T_board_camera"]["translation_m"]
    base_translation = result["robot_base"]["T_board_robot_base"]["translation_m"]
    tag_id_source = result["board"].get("tag_id_source", "configured")
    source_text = "" if tag_id_source == "configured" else f"id_source={tag_id_source} "
    print(
        f"{prefix}"
        f"tags={result['used_tag_ids']} "
        f"{source_text}"
        f"cam_xyz=({camera_translation[0]:.3f},{camera_translation[1]:.3f},{camera_translation[2]:.3f})m "
        f"cam_dist={result['fusion']['board_camera_distance_m']:.3f}m "
        f"base_xyz=({base_translation[0]:.3f},{base_translation[1]:.3f},{base_translation[2]:.3f})m"
    )


def get_T_camera_robot_base(args: argparse.Namespace) -> np.ndarray:
    if args.T_camera_robot_base is not None:
        return np.asarray(args.T_camera_robot_base, dtype=np.float64)
    if args.T_robot_base_camera is not None:
        return inverse_transform(np.asarray(args.T_robot_base_camera, dtype=np.float64))
    return np.eye(4, dtype=np.float64)


def crop_eye_image(image_bgr: np.ndarray, crop_eye: str) -> np.ndarray:
    if crop_eye == "none":
        return image_bgr
    height, width = image_bgr.shape[:2]
    if width < 2:
        raise ValueError(f"cannot crop {crop_eye} eye from image width {width}")
    half_width = width // 2
    if crop_eye == "left":
        return image_bgr[:, :half_width].copy()
    if crop_eye == "right":
        return image_bgr[:, width - half_width :].copy()
    raise ValueError(f"unsupported crop_eye: {crop_eye}")


if __name__ == "__main__":
    raise SystemExit(main())


# Example:
# python xr_teleoperate/teleop/utils/estimate_robot_pose_from_apriltag.py \
#   --image path/to/head_cam.jpg \
#   --fx 615.0 --fy 615.0 --cx 320.0 --cy 240.0 \
#   --tag-size-m 0.100 \
#   --tag-layout 3x2 \
#   --tag-ids 0,1,2,3,4,5 \
#   --gap-x-m 0.020 \
#   --gap-y-m 0.020 \
#   --output-json result.json \
#   --debug-image debug.jpg
#
# Example with a non-identity extrinsic:
# python xr_teleoperate/teleop/utils/estimate_robot_pose_from_apriltag.py \
#   --image path/to/head_cam.jpg \
#   --fx 615.0 --fy 615.0 --cx 320.0 --cy 240.0 \
#   --tag-size-m 0.100 --tag-layout 3x2 --tag-ids 0,1,2,3,4,5 \
#   --gap-x-m 0.020 --gap-y-m 0.020 \
#   --T-camera-robot-base "1,0,0,0.05, 0,1,0,0.00, 0,0,1,0.12, 0,0,0,1"
