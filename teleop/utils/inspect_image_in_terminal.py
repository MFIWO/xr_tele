#!/usr/bin/env python3
"""Inspect an image from bash, with optional AprilTag detection.

This is useful inside Docker/SSH sessions where opening an image viewer is
annoying. It prints image stats and a small ANSI true-color preview directly in
the terminal.

Example:
  python teleop/utils/inspect_image_in_terminal.py /tmp/apriltag_debug.jpg --detect-tags
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview an image in the terminal.")
    parser.add_argument("image", type=Path, help="Image path to inspect")
    parser.add_argument("--max-width", type=int, default=96, help="Preview width in terminal cells")
    parser.add_argument("--detect-tags", action="store_true", help="Run tag36h11 detection and print detected ids")
    parser.add_argument("--family", default="tag36h11", help="AprilTag family for --detect-tags")
    parser.add_argument("--save-preview", type=Path, help="Optional resized preview image path")
    parser.add_argument(
        "--save-diagnostics",
        type=Path,
        help="Save a montage with gray/threshold/edges/quad-candidate views",
    )
    parser.add_argument(
        "--mode",
        choices=("ansi", "ascii", "edges"),
        default="ansi",
        help="Preview mode; ascii/edges survive copy-paste better than ansi",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if cv2 is None or np is None:
        missing = []
        if cv2 is None:
            missing.append("opencv-python")
        if np is None:
            missing.append("numpy")
        print(f"error: missing dependencies: {', '.join(missing)}", file=sys.stderr)
        return 1

    image_path: Path = args.image
    if not image_path.exists():
        print(f"error: image does not exist: {image_path}", file=sys.stderr)
        return 1

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"error: OpenCV could not read: {image_path}", file=sys.stderr)
        return 1

    print_image_stats(image_path, image_bgr)
    if args.detect_tags:
        detect_tags(image_bgr, args.family)
    if args.save_preview is not None:
        preview = resized_for_preview(image_bgr, args.max_width)
        args.save_preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_preview), preview)
        print(f"saved preview: {args.save_preview}")
    if args.save_diagnostics is not None:
        diagnostics, candidates = make_diagnostics_montage(image_bgr)
        args.save_diagnostics.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_diagnostics), diagnostics)
        print_quad_candidates(candidates)
        print(f"saved diagnostics: {args.save_diagnostics}")

    print(f"\nterminal preview ({args.mode}):")
    if args.mode == "ansi":
        print_ansi_preview(image_bgr, max_width=args.max_width)
    elif args.mode == "ascii":
        print_ascii_preview(image_bgr, max_width=args.max_width)
    elif args.mode == "edges":
        print_edge_preview(image_bgr, max_width=args.max_width)
    return 0


def print_image_stats(path: Path, image_bgr: np.ndarray) -> None:
    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    print(f"path: {path}")
    print(f"shape: height={height} width={width} channels={image_bgr.shape[2]}")
    print(f"gray: min={int(gray.min())} max={int(gray.max())} mean={float(gray.mean()):.1f} std={float(gray.std()):.1f}")


def detect_tags(image_bgr: np.ndarray, family: str) -> None:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    try:
        from pupil_apriltags import Detector

        detector = Detector(
            families=family,
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )
        detections = detector.detect(gray, estimate_tag_pose=False)
    except ImportError:
        try:
            import apriltag

            detector = apriltag.Detector(apriltag.DetectorOptions(families=family))
            detections = detector.detect(gray)
        except ImportError:
            print("apriltag detection: skipped; install pupil-apriltags or apriltag")
            return

    print(f"apriltag detection: family={family} count={len(detections)}")
    for det in detections:
        tag_id = int(getattr(det, "tag_id"))
        center = np.asarray(getattr(det, "center"), dtype=float)
        corners = np.asarray(getattr(det, "corners"), dtype=float)
        margin = getattr(det, "decision_margin", None)
        margin_text = "" if margin is None else f" margin={float(margin):.1f}"
        corner_text = " ".join(f"({x:.0f},{y:.0f})" for x, y in corners)
        print(f"  id={tag_id} center=({center[0]:.1f},{center[1]:.1f}){margin_text} corners={corner_text}")


def make_diagnostics_montage(image_bgr: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    edges = cv2.Canny(gray, 60, 160)

    candidate_view, candidates = draw_quad_candidates(image_bgr, edges)
    panels = [
        ("original", image_bgr),
        ("gray", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)),
        ("equalized", cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)),
        ("otsu", cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)),
        ("adaptive", cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)),
        ("edges", cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)),
        ("quad candidates", candidate_view),
    ]
    return tile_labeled_panels(panels, panel_width=480), candidates


def draw_quad_candidates(image_bgr: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]]]:
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(image_bgr.shape[0] * image_bgr.shape[1])
    candidates: list[dict[str, float]] = []
    view = image_bgr.copy()

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        area = abs(float(cv2.contourArea(approx)))
        if area < max(100.0, image_area * 0.00005):
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if w <= 0 or h <= 0:
            continue
        aspect = w / h
        if aspect < 0.35 or aspect > 2.8:
            continue
        candidates.append(
            {
                "area": area,
                "x": float(x),
                "y": float(y),
                "w": float(w),
                "h": float(h),
                "aspect": float(aspect),
            }
        )

    candidates.sort(key=lambda item: item["area"], reverse=True)
    for idx, item in enumerate(candidates[:30]):
        x, y, w, h = (int(item[key]) for key in ("x", "y", "w", "h"))
        color = (0, 255, 255) if idx < 10 else (0, 170, 255)
        cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)
        cv2.putText(view, str(idx), (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return view, candidates


def print_quad_candidates(candidates: list[dict[str, float]]) -> None:
    print(f"quad-like contour candidates: {len(candidates)}")
    for idx, item in enumerate(candidates[:15]):
        print(
            f"  #{idx:02d} bbox=({item['x']:.0f},{item['y']:.0f},{item['w']:.0f},{item['h']:.0f}) "
            f"area={item['area']:.0f} aspect={item['aspect']:.2f}"
        )


def tile_labeled_panels(panels: list[tuple[str, np.ndarray]], panel_width: int) -> np.ndarray:
    rendered = []
    for label, image in panels:
        height, width = image.shape[:2]
        scale = panel_width / max(width, 1)
        panel_height = max(1, int(round(height * scale)))
        panel = cv2.resize(image, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
        label_bar = np.full((32, panel_width, 3), 20, dtype=np.uint8)
        cv2.putText(label_bar, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA)
        rendered.append(np.vstack([label_bar, panel]))

    cols = 2
    rows = []
    for start in range(0, len(rendered), cols):
        row_panels = rendered[start : start + cols]
        row_height = max(panel.shape[0] for panel in row_panels)
        padded = []
        for panel in row_panels:
            if panel.shape[0] < row_height:
                pad = np.full((row_height - panel.shape[0], panel.shape[1], 3), 255, dtype=np.uint8)
                panel = np.vstack([panel, pad])
            padded.append(panel)
        if len(padded) < cols:
            padded.append(np.full((row_height, panel_width, 3), 255, dtype=np.uint8))
        rows.append(np.hstack(padded))
    return np.vstack(rows)


def resized_for_preview(image_bgr: np.ndarray, max_width: int) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    max_width = max(8, int(max_width))
    scale = min(1.0, max_width / max(width, 1))
    out_w = max(1, int(round(width * scale)))
    out_h = max(1, int(round(height * scale)))
    return cv2.resize(image_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)


def print_ansi_preview(image_bgr: np.ndarray, max_width: int) -> None:
    preview = resized_for_preview(image_bgr, max_width=max_width)
    # Two image rows per text row using upper-half block: foreground is top,
    # background is bottom. Convert BGR to RGB before emitting colors.
    preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    height, width = preview_rgb.shape[:2]
    if height % 2:
        pad = np.zeros((1, width, 3), dtype=preview_rgb.dtype)
        preview_rgb = np.vstack([preview_rgb, pad])
        height += 1

    reset = "\033[0m"
    rows: list[str] = []
    for y in range(0, height, 2):
        parts: list[str] = []
        top_row = preview_rgb[y]
        bottom_row = preview_rgb[y + 1]
        for top, bottom in zip(top_row, bottom_row):
            tr, tg, tb = (int(v) for v in top)
            br, bg, bb = (int(v) for v in bottom)
            parts.append(f"\033[38;2;{tr};{tg};{tb}m\033[48;2;{br};{bg};{bb}m▀")
        parts.append(reset)
        rows.append("".join(parts))
    print("\n".join(rows))


def print_ascii_preview(image_bgr: np.ndarray, max_width: int) -> None:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    preview = resize_gray_for_text(gray, max_width=max_width, cell_aspect=0.5)
    chars = np.asarray(list(" .:-=+*#%@"))
    idx = np.clip((preview.astype(np.float32) / 256.0 * len(chars)).astype(np.int32), 0, len(chars) - 1)
    print("\n".join("".join(chars[row]) for row in idx))


def print_edge_preview(image_bgr: np.ndarray, max_width: int) -> None:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    preview = resize_gray_for_text(edges, max_width=max_width, cell_aspect=0.5)
    chars = np.where(preview > 0, "#", " ")
    print("\n".join("".join(row) for row in chars))


def resize_gray_for_text(gray: np.ndarray, max_width: int, cell_aspect: float) -> np.ndarray:
    height, width = gray.shape[:2]
    max_width = max(8, int(max_width))
    scale = min(1.0, max_width / max(width, 1))
    out_w = max(1, int(round(width * scale)))
    out_h = max(1, int(round(height * scale * cell_aspect)))
    return cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    raise SystemExit(main())
