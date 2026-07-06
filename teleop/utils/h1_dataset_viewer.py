import argparse
import glob
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

try:
    import pyqtgraph as pg
except ImportError as exc:
    raise SystemExit(
        "pyqtgraph is required for h1_dataset_viewer.py. "
        "Install it in the active environment, for example: pip install pyqtgraph"
    ) from exc

try:
    import cv2
except ImportError:
    cv2 = None


DEFAULT_EPISODE_DIR = Path(__file__).resolve().parent / "data" / "pick cube" / "episode_0007"

ARM_JOINT_NAMES = [
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow_pitch",
    "elbow_roll",
    "wrist_pitch",
    "wrist_yaw",
]

DG2_JOINT_NAMES = [
    "thumb0",
    "thumb1",
    "thumb2",
    "index0",
    "index1",
    "middle0",
    "middle1",
    "ring0",
    "ring1",
    "little0",
    "little1",
    "spread0",
    "spread1",
]

TACTILE_FINGER_ORDER = ["thumb", "index", "middle", "ring", "little"]
TACTILE_PALM_NAMES = ["palm_0", "palm_1", "palm_2"]
TACTILE_SENSOR_ORDER = TACTILE_FINGER_ORDER + TACTILE_PALM_NAMES
TACTILE_NORMAL_MAX = 800.0
TACTILE_TANGENT_MAX = 800.0
TACTILE_PROXIMITY_MAX = 65535.0


def _frame_number(path):
    match = re.search(r"(\d{6})_color_(\d+)\.jpg$", str(path))
    return int(match.group(1)) if match else -1


def _camera_number(path):
    match = re.search(r"(\d{6})_color_(\d+)\.jpg$", str(path))
    return int(match.group(2)) if match else -1


def load_episode_json(path):
    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as f:
        text = f.read()

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        stripped = text.rstrip()
        repaired = None
        if exc.pos >= len(stripped) - 2:
            for suffix in ("\n]\n}\n", "\n}\n]\n}\n"):
                try:
                    repaired = json.loads(stripped + suffix)
                    print(f"Warning: repaired missing final JSON bracket(s) in memory: {json_path}")
                    break
                except json.JSONDecodeError:
                    pass
        if repaired is None:
            raise
        return repaired


def _safe_float(value, default=0.0):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _finger_values(raw_values):
    values = list(raw_values or [])
    values = (values + [0.0, 0.0, 65535.0, 0.0])[:4]
    return {
        "normal": _safe_float(values[0]),
        "tangent": _safe_float(values[1]),
        "direction": _safe_float(values[2], 65535.0),
        "proximity": _safe_float(values[3]),
    }


def _palm_values(raw_values, index):
    values = list(raw_values or [])
    base = index * 3
    chunk = (values[base : base + 3] + [0.0, 0.0, 65535.0])[:3]
    return {
        "normal": _safe_float(chunk[0]),
        "tangent": _safe_float(chunk[1]),
        "direction": _safe_float(chunk[2], 65535.0),
        "proximity": 0.0,
    }


def _direction_is_valid(direction):
    return 0.0 <= float(direction) <= 359.0


def _tactile_heat(values):
    normal = values.get("normal", 0.0)
    tangent = values.get("tangent", 0.0)
    force = math.sqrt(normal * normal + tangent * tangent)
    return float(
        np.clip(
            force / math.sqrt(TACTILE_NORMAL_MAX * TACTILE_NORMAL_MAX + TACTILE_TANGENT_MAX * TACTILE_TANGENT_MAX),
            0.0,
            1.0,
        )
    )


def parse_tactile_side(frame, side):
    tactile = ((frame.get("tactiles") or {}).get(side) or {})
    fingers_raw = tactile.get("fingers") or {}
    palm_raw = tactile.get("palm") or []
    return {
        "timestamp": tactile.get("timestamp"),
        "source": tactile.get("source", ""),
        "fingers": {
            name: _finger_values(fingers_raw.get(name))
            for name in TACTILE_FINGER_ORDER
        },
        "palm": {
            name: _palm_values(palm_raw, index)
            for index, name in enumerate(TACTILE_PALM_NAMES)
        },
    }


class H1EpisodeDataHandler:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.colors_dir = self.data_dir / "colors"
        self.json_file = self.data_dir / "data.json"
        self.image_sets = []
        self.json_data = []
        self.info = {}
        self.current_index = 0

        self.load_image_sets()
        self.load_json_data()

    def load_image_sets(self):
        if not self.colors_dir.exists():
            print(f"Colors directory does not exist: {self.colors_dir}")
            return

        frame_groups = {}
        for filename in glob.glob(str(self.colors_dir / "*_color_*.jpg")):
            frame_idx = _frame_number(filename)
            camera_idx = _camera_number(filename)
            if frame_idx < 0 or camera_idx < 0:
                continue
            frame_groups.setdefault(frame_idx, {})[camera_idx] = filename

        for frame_idx in sorted(frame_groups):
            cameras = frame_groups[frame_idx]
            self.image_sets.append(
                {
                    "index": frame_idx,
                    "head_left": cameras.get(0),
                    "head_right": cameras.get(1),
                    "wrist_left": cameras.get(2),
                    "wrist_right": cameras.get(3),
                }
            )

        print(f"Loaded {len(self.image_sets)} image frames from {self.colors_dir}")

    def load_json_data(self):
        if not self.json_file.exists():
            print(f"JSON file does not exist: {self.json_file}")
            return

        episode = load_episode_json(self.json_file)
        self.info = episode.get("info", {})
        self.json_data = episode.get("data", [])
        print(f"Loaded {len(self.json_data)} JSON frames from {self.json_file}")

    def get_total_frames(self):
        counts = [len(self.json_data)]
        if self.image_sets:
            counts.append(len(self.image_sets))
        return min(counts) if counts else 0

    def set_frame(self, frame_index):
        max_index = self.get_total_frames() - 1
        if 0 <= frame_index <= max_index:
            self.current_index = frame_index
            return True
        return False

    def next_frame(self):
        if self.current_index < self.get_total_frames() - 1:
            self.current_index += 1
            return True
        return False

    def prev_frame(self):
        if self.current_index > 0:
            self.current_index -= 1
            return True
        return False

    def read_current_images(self):
        if not self.image_sets or self.current_index >= len(self.image_sets):
            return None

        current_images = self.image_sets[self.current_index]
        images = {}
        for key, filepath in current_images.items():
            if key == "index" or filepath is None:
                continue
            if cv2 is None:
                images[key] = filepath
            else:
                img_bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    continue
                images[key] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        return {
            "images": images,
            "index": current_images["index"],
            "time": self.current_index / self.fps,
        }

    def read_current_json_data(self):
        if not self.json_data or self.current_index >= len(self.json_data):
            return None
        frame = self.json_data[self.current_index]
        return {
            "data": frame,
            "index": frame.get("idx", self.current_index),
            "time": self.current_index / self.fps,
        }

    @property
    def fps(self):
        image_info = self.info.get("image", {}) or {}
        return float(image_info.get("fps") or 30.0)


class HeadWristImagesTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.title_label = QLabel("HEAD / WRIST IMAGES")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #1b5e20;")
        self.layout.addWidget(self.title_label)

        self.grid_layout = QGridLayout()
        self.layout.addLayout(self.grid_layout)
        self.image_labels = {}
        self.create_image_widgets()

    def create_image_widgets(self):
        image_layout = [
            ("head_left", 0, 0, "Head Camera Left"),
            ("head_right", 0, 1, "Head Camera Right"),
            ("wrist_left", 1, 0, "Left Wrist Camera"),
            ("wrist_right", 1, 1, "Right Wrist Camera"),
        ]

        for key, row, col, title in image_layout:
            container = QWidget()
            container.setStyleSheet("border: 1px solid #c7c7c7; background-color: #fafafa;")
            container_layout = QVBoxLayout()
            container.setLayout(container_layout)

            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("font-weight: bold; font-size: 15px;")
            container_layout.addWidget(title_label)

            image_label = QLabel()
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setText("No Image")
            image_label.setStyleSheet("background-color: black; color: white;")
            image_label.setMinimumSize(320, 240)
            image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            image_label.setScaledContents(False)
            container_layout.addWidget(image_label)

            container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.image_labels[key] = image_label
            self.grid_layout.addWidget(container, row, col)

    def update_images(self, image_data):
        if image_data is None or "images" not in image_data:
            for label in self.image_labels.values():
                label.setText("No Image")
            return

        images = image_data["images"]
        for key, image_label in self.image_labels.items():
            if key not in images:
                image_label.clear()
                image_label.setText("No Image")
                continue

            image_value = images[key]
            if isinstance(image_value, str):
                pixmap = QPixmap(image_value)
            else:
                img_array = np.ascontiguousarray(image_value)
                h, w, ch = img_array.shape
                q_image = QImage(img_array.data, w, h, ch * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
            if pixmap.isNull():
                image_label.clear()
                image_label.setText("No Image")
                continue
            scaled_pixmap = pixmap.scaled(
                image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            image_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        parent = self.window()
        if hasattr(parent, "update_display"):
            parent.update_display()


class ArmHandJointDataTab(QWidget):
    def __init__(self, sample_rate=30.0, time_window=5.0):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.title_label = QLabel("ARM / INSPIRE DG2 HAND JOINT DATA (State: solid, Action: dashed)")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #4a148c;")
        self.layout.addWidget(self.title_label)

        self.time_window = float(time_window)
        self.sample_rate = float(sample_rate)
        self.total_samples = max(1, int(self.time_window * self.sample_rate))
        self.time_array = np.linspace(0, self.time_window, self.total_samples)
        self.write_index = 0
        self.valid_data_count = 0

        self.configs = {
            "left_arm": ("Left Arm (H1-2, 7 DOF)", ARM_JOINT_NAMES, "#1565c0"),
            "right_arm": ("Right Arm (H1-2, 7 DOF)", ARM_JOINT_NAMES, "#c62828"),
            "left_ee": ("Left Inspire DG2 Hand (13 DOF)", DG2_JOINT_NAMES, "#00838f"),
            "right_ee": ("Right Inspire DG2 Hand (13 DOF)", DG2_JOINT_NAMES, "#ef6c00"),
        }

        self.buffers = {}
        self.plot_widgets = {}
        self.state_curves = {}
        self.action_curves = {}
        for key, (_, names, _) in self.configs.items():
            n = len(names)
            self.buffers[key] = {
                "state": np.zeros((n, self.total_samples), dtype=np.float64),
                "action": np.zeros((n, self.total_samples), dtype=np.float64),
            }

        self.create_plots()

    def create_plots(self):
        self.grid_layout = QGridLayout()
        self.layout.addLayout(self.grid_layout)

        positions = {
            "left_arm": (0, 0),
            "right_arm": (0, 1),
            "left_ee": (1, 0),
            "right_ee": (1, 1),
        }

        for key, (row, col) in positions.items():
            title, names, color = self.configs[key]
            plot_widget = pg.PlotWidget()
            plot_widget.setTitle(title)
            plot_widget.setLabel("left", "Joint Value")
            plot_widget.setLabel("bottom", "Time (s)")
            plot_widget.addLegend(offset=(10, 10))
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            plot_widget.setXRange(0, self.time_window, padding=0)
            plot_widget.enableAutoRange(axis="x", enable=False)
            plot_widget.setMouseEnabled(x=False, y=True)
            plot_widget.getViewBox().setLimits(xMin=0, xMax=self.time_window)

            self.plot_widgets[key] = plot_widget
            self.state_curves[key] = []
            self.action_curves[key] = []

            for i, name in enumerate(names):
                hue = (i * 37) % 360
                state_pen = pg.mkPen(color=pg.hsvColor(hue / 360.0, 0.85, 0.95), width=2)
                action_pen = pg.mkPen(color=pg.hsvColor(hue / 360.0, 0.85, 0.95), width=2, style=QtCore.Qt.DashLine)
                self.state_curves[key].append(plot_widget.plot(pen=state_pen, name=f"S{i}:{name}"))
                self.action_curves[key].append(plot_widget.plot(pen=action_pen, name=f"A{i}:{name}"))

            self.grid_layout.addWidget(plot_widget, row, col)

    def reset_and_start_from_position(self):
        self.write_index = 0
        self.valid_data_count = 0
        for key in self.buffers:
            self.buffers[key]["state"].fill(0)
            self.buffers[key]["action"].fill(0)
        self.update_curves()

    def _copy_qpos(self, frame_root, key, target):
        qpos = ((frame_root.get(key, {}) or {}).get("qpos")) or []
        target[:, self.write_index] = 0.0
        for i, value in enumerate(qpos[: target.shape[0]]):
            target[i, self.write_index] = float(value)

    def update_plot(self, json_data_dict):
        if json_data_dict is None or "data" not in json_data_dict:
            return

        frame = json_data_dict["data"]
        states = frame.get("states", {}) or {}
        actions = frame.get("actions", {}) or {}

        for key in self.configs:
            self._copy_qpos(states, key, self.buffers[key]["state"])
            self._copy_qpos(actions, key, self.buffers[key]["action"])

        self.write_index = (self.write_index + 1) % self.total_samples
        self.valid_data_count = min(self.valid_data_count + 1, self.total_samples)
        self.update_curves()

    def get_display_data(self, key):
        if self.valid_data_count == 0:
            return None

        state = self.buffers[key]["state"]
        action = self.buffers[key]["action"]
        if self.valid_data_count < self.total_samples:
            indices = np.arange(self.valid_data_count)
            time_display = self.time_array[: self.valid_data_count]
        else:
            indices = np.concatenate((np.arange(self.write_index, self.total_samples), np.arange(0, self.write_index)))
            time_display = self.time_array
        return time_display, state[:, indices], action[:, indices]

    def update_curves(self):
        for key, (_, names, _) in self.configs.items():
            display_data = self.get_display_data(key)
            if display_data is None:
                for curve in self.state_curves.get(key, []) + self.action_curves.get(key, []):
                    curve.setData([], [])
                continue

            time_display, state, action = display_data
            for i in range(len(names)):
                self.state_curves[key][i].setData(time_display, state[i])
                self.action_curves[key][i].setData(time_display, action[i])


class TactileHandWidget(QWidget):
    def __init__(self, side_title):
        super().__init__()
        self.side_title = side_title
        self.tactile = None
        self.setMinimumSize(520, 430)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_tactile(self, tactile):
        self.tactile = tactile
        self.update()

    @staticmethod
    def _heat_color(heat):
        heat = float(np.clip(heat, 0.0, 1.0))
        r = int(245)
        g = int(245 - 145 * heat)
        b = int(245 - 190 * heat)
        return QColor(r, g, b)

    def _draw_arrow(self, painter, cx, cy, direction, tangent, radius):
        if not _direction_is_valid(direction) or tangent <= 0.0:
            return
        angle = math.radians(float(direction))
        length = radius * (0.65 + 0.95 * float(np.clip(tangent / TACTILE_TANGENT_MAX, 0.0, 1.0)))
        ex = cx + math.cos(angle) * length
        ey = cy - math.sin(angle) * length

        painter.setPen(QPen(QColor("#263238"), 3))
        painter.drawLine(int(cx), int(cy), int(ex), int(ey))

        head_len = max(10.0, radius * 0.22)
        for offset in (math.radians(150), math.radians(-150)):
            hx = ex + math.cos(angle + offset) * head_len
            hy = ey - math.sin(angle + offset) * head_len
            painter.drawLine(int(ex), int(ey), int(hx), int(hy))

    def _draw_sensor(self, painter, rect, label, values, circular=False):
        normal = values.get("normal", 0.0)
        tangent = values.get("tangent", 0.0)
        direction = values.get("direction", 65535.0)
        proximity = values.get("proximity", 0.0)
        heat = _tactile_heat(values)

        painter.setPen(QPen(QColor("#6d6d6d"), 1))
        painter.setBrush(self._heat_color(heat))
        if circular:
            painter.drawEllipse(rect)
        else:
            painter.drawRoundedRect(rect, 8, 8)

        center = rect.center()
        self._draw_arrow(painter, center.x(), center.y() + 6, direction, tangent, rect.height() * 0.34)

        painter.setPen(QColor("#111111"))
        painter.setFont(QFont("Sans Serif", 9, QFont.Bold))
        painter.drawText(rect.adjusted(4, 4, -4, -4), Qt.AlignTop | Qt.AlignHCenter, label)

        painter.setFont(QFont("Monospace", 8))
        direction_text = f"{int(direction):3d}" if _direction_is_valid(direction) else "---"
        lines = [
            f"N {normal:4.0f}",
            f"T {tangent:4.0f}",
            f"D {direction_text}",
        ]
        if not circular:
            lines.append(f"P {proximity:5.0f}")
        painter.drawText(rect.adjusted(5, 22, -5, -4), Qt.AlignLeft | Qt.AlignTop, "\n".join(lines))

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        painter.fillRect(self.rect(), QColor("#fbfbfb"))
        painter.setPen(QColor("#222222"))
        painter.setFont(QFont("Sans Serif", 13, QFont.Bold))
        painter.drawText(0, 10, w, 26, Qt.AlignHCenter, self.side_title)

        palm_rect = QtCore.QRectF(w * 0.26, h * 0.42, w * 0.48, h * 0.40)
        painter.setPen(QPen(QColor("#c4c4c4"), 2))
        painter.setBrush(QColor("#f1f3f4"))
        painter.drawRoundedRect(palm_rect, 24, 24)

        if not self.tactile:
            painter.setPen(QColor("#777777"))
            painter.setFont(QFont("Sans Serif", 12))
            painter.drawText(self.rect(), Qt.AlignCenter, "No tactile data")
            return

        finger_layout = {
            "thumb": (0.225, 0.35, 0.12, 0.24),
            "index": (0.345, 0.13, 0.10, 0.30),
            "middle": (0.450, 0.08, 0.10, 0.32),
            "ring": (0.555, 0.13, 0.10, 0.30),
            "little": (0.665, 0.23, 0.10, 0.27),
        }
        for name, (x, y, rw, rh) in finger_layout.items():
            rect = QtCore.QRectF(w * x, h * y, w * rw, h * rh)
            self._draw_sensor(painter, rect, name, self.tactile["fingers"].get(name, {}))

        palm_layout = {
            "palm_0": (0.27, 0.56),
            "palm_1": (0.43, 0.56),
            "palm_2": (0.59, 0.56),
        }
        palm_w = w * 0.14
        palm_h = h * 0.22
        for name, (x, y) in palm_layout.items():
            rect = QtCore.QRectF(w * x, h * y, palm_w, palm_h)
            self._draw_sensor(painter, rect, name, self.tactile["palm"].get(name, {}))

        timestamp = self.tactile.get("timestamp")
        source = self.tactile.get("source", "")
        footer = f"timestamp: {timestamp:.3f}" if isinstance(timestamp, (int, float)) else "timestamp: -"
        if source:
            footer += f" | {source}"
        painter.setPen(QColor("#555555"))
        painter.setFont(QFont("Sans Serif", 9))
        painter.drawText(8, h - 24, w - 16, 18, Qt.AlignLeft, footer)


class DG2TactileForceTab(QWidget):
    def __init__(self, sample_rate=30.0, time_window=5.0):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.title_label = QLabel("INSPIRE DG2 TACTILE FORCE DATA")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #004d40;")
        self.layout.addWidget(self.title_label)

        self.hand_layout = QGridLayout()
        self.layout.addLayout(self.hand_layout, stretch=3)
        self.hand_widgets = {
            "left_ee": TactileHandWidget("Left DG2 Tactile"),
            "right_ee": TactileHandWidget("Right DG2 Tactile"),
        }
        self.hand_layout.addWidget(self.hand_widgets["left_ee"], 0, 0)
        self.hand_layout.addWidget(self.hand_widgets["right_ee"], 0, 1)

        self.time_window = float(time_window)
        self.sample_rate = float(sample_rate)
        self.total_samples = max(1, int(self.time_window * self.sample_rate))
        self.time_array = np.linspace(0, self.time_window, self.total_samples)
        self.write_index = 0
        self.valid_data_count = 0
        self.buffers = {
            side: {
                sensor: np.zeros(self.total_samples, dtype=np.float64)
                for sensor in TACTILE_SENSOR_ORDER
            }
            for side in ("left_ee", "right_ee")
        }

        self.plot_layout = QGridLayout()
        self.layout.addLayout(self.plot_layout, stretch=2)
        self.force_plots = {}
        self.curves = {}
        plot_specs = {
            "left_ee": ("Left tactile force per contact", 0, 0),
            "right_ee": ("Right tactile force per contact", 0, 1),
        }
        for side, (title, row, col) in plot_specs.items():
            plot = pg.PlotWidget()
            plot.setTitle(title)
            plot.setLabel("left", "Force magnitude")
            plot.setLabel("bottom", "Time (s)")
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.setXRange(0, self.time_window, padding=0)
            plot.addLegend(offset=(10, 10))
            plot.enableAutoRange(axis="x", enable=False)
            plot.setMouseEnabled(x=False, y=True)
            plot.getViewBox().setLimits(xMin=0, xMax=self.time_window)
            self.force_plots[side] = plot
            self.plot_layout.addWidget(plot, row, col)

            for i, sensor in enumerate(TACTILE_SENSOR_ORDER):
                hue = (i * 41) % 360
                pen = pg.mkPen(color=pg.hsvColor(hue / 360.0, 0.85, 0.95), width=2)
                self.curves[(side, sensor)] = plot.plot(pen=pen, name=sensor)

    def reset_and_start_from_position(self):
        self.write_index = 0
        self.valid_data_count = 0
        for side in self.buffers:
            for channel in self.buffers[side]:
                self.buffers[side][channel].fill(0)
        for widget in self.hand_widgets.values():
            widget.set_tactile(None)
        self.update_curves()

    @staticmethod
    def _force_magnitude(values):
        normal = float(values.get("normal", 0.0))
        tangent = float(values.get("tangent", 0.0))
        return math.sqrt(normal * normal + tangent * tangent)

    def _sensor_force_values(self, tactile):
        values = {}
        fingers = tactile.get("fingers") or {}
        palm = tactile.get("palm") or {}
        for sensor in TACTILE_FINGER_ORDER:
            values[sensor] = self._force_magnitude(fingers.get(sensor, {}))
        for sensor in TACTILE_PALM_NAMES:
            values[sensor] = self._force_magnitude(palm.get(sensor, {}))
        return values

    def update_tactile(self, json_data_dict):
        if json_data_dict is None or "data" not in json_data_dict:
            return

        frame = json_data_dict["data"]
        for side in ("left_ee", "right_ee"):
            tactile = parse_tactile_side(frame, side)
            self.hand_widgets[side].set_tactile(tactile)
            force_values = self._sensor_force_values(tactile)
            for sensor in TACTILE_SENSOR_ORDER:
                self.buffers[side][sensor][self.write_index] = force_values.get(sensor, 0.0)

        self.write_index = (self.write_index + 1) % self.total_samples
        self.valid_data_count = min(self.valid_data_count + 1, self.total_samples)
        self.update_curves()

    def _display_indices(self):
        if self.valid_data_count == 0:
            return None, None
        if self.valid_data_count < self.total_samples:
            return self.time_array[: self.valid_data_count], np.arange(self.valid_data_count)
        indices = np.concatenate((np.arange(self.write_index, self.total_samples), np.arange(0, self.write_index)))
        return self.time_array, indices

    def update_curves(self):
        time_display, indices = self._display_indices()
        if indices is None:
            for curve in self.curves.values():
                curve.setData([], [])
            return
        for side in self.buffers:
            for sensor in self.buffers[side]:
                self.curves[(side, sensor)].setData(time_display, self.buffers[side][sensor][indices])


class H1DatasetViewerMainWindow(QMainWindow):
    def __init__(self, initial_dir=None):
        super().__init__()
        self.setWindowTitle("Unitree H1-2 Teleoperation Dataset Viewer")
        self.setGeometry(100, 100, 1500, 1000)
        self.data_handler = None
        self.initial_dir = Path(initial_dir).expanduser().resolve() if initial_dir else DEFAULT_EPISODE_DIR

        self.setup_ui()
        self.setup_timer()
        if self.initial_dir.exists():
            self.load_folder(str(self.initial_dir))

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        layout.addWidget(self.create_control_panel())

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            """
            QTabWidget::pane { border: 1px solid #ccc; }
            QTabBar::tab {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                padding: 8px 20px;
                margin-right: 2px;
                font-size: 16px;
                min-width: 220px;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
                border-bottom-color: #ffffff;
            }
            QTabBar::tab:hover { background-color: #e0e0e0; }
            """
        )

        self.camera_tab = HeadWristImagesTab()
        self.joint_tab = ArmHandJointDataTab()
        self.tactile_tab = DG2TactileForceTab()
        self.tabs.addTab(self.camera_tab, "Head & Wrist Images")
        self.tabs.addTab(self.joint_tab, "Arm & Hand Joint Data")
        self.tabs.addTab(self.tactile_tab, "DG2 Tactile Force")
        layout.addWidget(self.tabs)

    def create_control_panel(self):
        panel = QWidget()
        layout = QHBoxLayout()
        panel.setLayout(layout)

        self.folder_btn = QPushButton("Select Folder")
        self.folder_btn.clicked.connect(self.select_folder)
        layout.addWidget(self.folder_btn)

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_btn.setEnabled(False)
        layout.addWidget(self.play_btn)

        self.prev_btn = QPushButton("Prev")
        self.prev_btn.clicked.connect(self.prev_frame)
        self.prev_btn.setEnabled(False)
        layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_frame)
        self.next_btn.setEnabled(False)
        layout.addWidget(self.next_btn)

        speed_label = QLabel("Speed:")
        layout.addWidget(speed_label)

        self.speed_spinbox = QSpinBox()
        self.speed_spinbox.setRange(1, 120)
        self.speed_spinbox.setValue(30)
        self.speed_spinbox.setSuffix(" fps")
        self.speed_spinbox.valueChanged.connect(self.update_timer_interval)
        layout.addWidget(self.speed_spinbox)

        frame_label = QLabel("Frame:")
        layout.addWidget(frame_label)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.slider_changed)
        self.frame_slider.setEnabled(False)
        layout.addWidget(self.frame_slider)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.info_label = QLabel("No data loaded")
        layout.addWidget(self.info_label)

        return panel

    def setup_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_playing = False
        self.update_timer_interval()

    def update_timer_interval(self):
        self.timer.setInterval(max(1, 1000 // self.speed_spinbox.value()))

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select H1-2 Episode Directory (containing colors and data.json)",
            str(self.initial_dir.parent if self.initial_dir else Path.cwd()),
        )
        if folder:
            self.load_folder(folder)

    def load_folder(self, folder):
        self.data_handler = H1EpisodeDataHandler(folder)
        total_frames = self.data_handler.get_total_frames()
        if total_frames <= 0:
            self.info_label.setText("No valid H1-2 episode data found")
            return

        fps = int(round(self.data_handler.fps))
        self.speed_spinbox.setValue(max(1, min(120, fps)))
        self.frame_slider.setRange(0, total_frames - 1)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(True)
        self.progress_bar.setRange(0, total_frames - 1)

        self.play_btn.setEnabled(True)
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.joint_tab.reset_and_start_from_position()
        self.tactile_tab.reset_and_start_from_position()
        self.update_display()

    def toggle_playback(self):
        if self.is_playing:
            self.timer.stop()
            self.play_btn.setText("Play")
            self.is_playing = False
        else:
            self.timer.start()
            self.play_btn.setText("Pause")
            self.is_playing = True

    def update_frame(self):
        if not self.data_handler:
            return
        if not self.data_handler.next_frame():
            self.data_handler.set_frame(0)
            self.joint_tab.reset_and_start_from_position()
            self.tactile_tab.reset_and_start_from_position()
        self.update_display()

    def prev_frame(self):
        if not self.data_handler:
            return
        self.joint_tab.reset_and_start_from_position()
        self.tactile_tab.reset_and_start_from_position()
        self.data_handler.prev_frame()
        self.update_display()

    def next_frame(self):
        if not self.data_handler:
            return
        self.joint_tab.reset_and_start_from_position()
        self.tactile_tab.reset_and_start_from_position()
        self.data_handler.next_frame()
        self.update_display()

    def slider_changed(self):
        if not self.data_handler:
            return
        self.joint_tab.reset_and_start_from_position()
        self.tactile_tab.reset_and_start_from_position()
        self.data_handler.set_frame(self.frame_slider.value())
        self.update_display()

    def update_display(self):
        if not self.data_handler:
            return

        image_data = self.data_handler.read_current_images()
        json_data = self.data_handler.read_current_json_data()
        self.camera_tab.update_images(image_data)
        self.joint_tab.update_plot(json_data)
        self.tactile_tab.update_tactile(json_data)

        current_frame = self.data_handler.current_index
        total_frames = self.data_handler.get_total_frames()
        frame_idx = json_data["index"] if json_data else current_frame
        current_time = json_data["time"] if json_data else current_frame / self.data_handler.fps

        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(current_frame)
        self.frame_slider.blockSignals(False)
        self.progress_bar.setValue(current_frame)
        self.info_label.setText(
            f"Frame: {current_frame}/{total_frames - 1} | "
            f"JSON idx: {frame_idx} | Time: {current_time:.2f}s"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="View H1-2 episode images and arm/hand joint data.")
    parser.add_argument(
        "episode_dir",
        nargs="?",
        default=str(DEFAULT_EPISODE_DIR),
        help="episode directory containing colors/ and data.json",
    )
    return parser.parse_args()


def main():
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    args = parse_args()
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_X11InitThreads)
    window = H1DatasetViewerMainWindow(initial_dir=args.episode_dir)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
