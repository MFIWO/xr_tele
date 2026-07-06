import os
import json
import time
import argparse
import math
import inspect
import socket
from datetime import datetime
os.environ["RUST_LOG"] = "error"

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import logging_mp
except ImportError:
    import logging as logging_mp

try:
    import numpy as np
except ImportError:
    np = None

try:
    import rerun as rr
    import rerun.blueprint as rrb
except ImportError:
    rr = None
    rrb = None

logger_mp = logging_mp.getLogger(__name__)


def _local_ip_guess():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.2)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _supported_kwargs(func, kwargs):
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in params}


def _call_rerun_func(func, kwargs_options):
    last_error = None
    for kwargs in kwargs_options:
        try:
            return func(**_supported_kwargs(func, kwargs))
        except TypeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    return func()


def _flatten_numeric(value):
    out = []
    if value is None:
        return out
    if isinstance(value, bool):
        return [float(value)]
    if isinstance(value, (int, float)):
        value = float(value)
        return [value] if math.isfinite(value) else []
    if isinstance(value, list) or isinstance(value, tuple):
        for item in value:
            out.extend(_flatten_numeric(item))
    elif isinstance(value, dict):
        for item in value.values():
            out.extend(_flatten_numeric(item))
    elif np is not None and isinstance(value, np.ndarray):
        out.extend(_flatten_numeric(value.tolist()))
    return out


def _is_image_array(value):
    if np is None or not isinstance(value, np.ndarray):
        return False
    return value.ndim in (2, 3) and value.size > 0


def _to_rgb_image(image, color_format):
    if np is None or image is None:
        return None
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    if image.ndim != 3:
        return None
    if image.shape[2] == 4:
        if color_format == "bgra":
            if cv2 is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            return image[..., [2, 1, 0, 3]]
        return image
    if image.shape[2] != 3:
        return None
    if color_format == "bgr":
        if cv2 is not None:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image[..., ::-1]
    return image


class RerunEpisodeReader:
    def __init__(self, task_dir = ".", json_file="data.json"):
        self.task_dir = task_dir
        self.json_file = json_file

    def return_episode_data(self, episode_idx):
        # Load episode data on-demand
        episode_dir = os.path.join(self.task_dir, f"episode_{episode_idx:04d}")
        json_path = os.path.join(episode_dir, self.json_file)

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Episode {episode_idx} data.json not found.")

        with open(json_path, 'r', encoding='utf-8') as jsonf:
            json_file = json.load(jsonf)

        episode_data = []

        # Loop over the data entries and process each one
        for item_data in json_file['data']:
            # Process images and other data
            colors = self._process_images(item_data, 'colors', episode_dir)
            depths = self._process_images(item_data, 'depths', episode_dir)
            audios = self._process_audio(item_data, 'audios', episode_dir)

            # Append the data in the item_data list
            episode_data.append(
                {
                    'idx': item_data.get('idx', 0),
                    'colors': colors,
                    'depths': depths,
                    'states': item_data.get('states', {}),
                    'actions': item_data.get('actions', {}),
                    'tactiles': item_data.get('tactiles', {}),
                    'audios': audios,
                    '_image_color_format': 'rgb',
                }
            )

        return episode_data

    def _process_images(self, item_data, data_type, dir_path):
        images = {}
        if cv2 is None:
            logger_mp.warning("OpenCV is not installed; RerunEpisodeReader will skip saved images.")
            return images

        for key, file_name in item_data.get(data_type, {}).items():
            if file_name:
                file_path = os.path.join(dir_path, file_name)
                if os.path.exists(file_path):
                    image = cv2.imread(file_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images[key] = image
        return images

    def _process_audio(self, item_data, data_type, episode_dir):
        audio_data = {}

        for key, file_name in item_data.get(data_type, {}).items():
            if file_name:
                file_path = os.path.join(episode_dir, file_name)
                if os.path.exists(file_path) and np is not None:
                    try:
                        audio_data[key] = np.load(file_path, allow_pickle=False)
                    except Exception as exc:
                        logger_mp.warning("Failed to load audio file for Rerun: %s (%s)", file_path, exc)
        return audio_data

class RerunLogger:
    def __init__(
        self,
        prefix="",
        IdxRangeBoundary=30,
        memory_limit=None,
        image_jpeg_quality=75,
        viewer=None,
        web_port=None,
        ws_port=None,
    ):
        if rr is None or rrb is None:
            raise RuntimeError("rerun-sdk is not installed. Install xr_teleoperate/requirements.txt or pip install rerun-sdk.")
        self.prefix = prefix
        self.IdxRangeBoundary = IdxRangeBoundary
        self.image_jpeg_quality = image_jpeg_quality
        self.viewer = (viewer or os.getenv("TELEOP_RERUN_VIEWER", "web")).lower()
        self.web_port = int(web_port or os.getenv("TELEOP_RERUN_WEB_PORT", "9090"))
        self.ws_port = int(ws_port or os.getenv("TELEOP_RERUN_WS_PORT", "9876"))
        self.connect_addr = os.getenv("TELEOP_RERUN_ADDR", "127.0.0.1:9876")
        rr.init(datetime.now().strftime("Runtime_%Y%m%d_%H%M%S"))
        self._start_viewer(memory_limit)

        # Set up blueprint for live visualization
        if self.IdxRangeBoundary:
            try:
                self.setup_blueprint()
            except Exception as exc:
                logger_mp.warning("Rerun blueprint setup failed; continuing with default layout: %s", exc)
        self.log_status("viewer_ready", 1.0)

    def _start_viewer(self, memory_limit):
        if self.viewer in ("none", "off", "disabled"):
            logger_mp.info("Rerun viewer disabled by TELEOP_RERUN_VIEWER=%s", self.viewer)
            return

        if self.viewer in ("connect", "remote"):
            rr.connect(self.connect_addr)
            logger_mp.info("Rerun connected to native viewer at %s", self.connect_addr)
            return

        if self.viewer in ("web", "browser", "auto"):
            web_func = getattr(rr, "serve_web", None) or getattr(rr, "serve", None)
            if web_func is not None:
                bind = os.getenv("TELEOP_RERUN_BIND", "0.0.0.0")
                kwargs = {
                    "open_browser": False,
                    "web_port": self.web_port,
                    "ws_port": self.ws_port,
                    "bind": bind,
                    "server_memory_limit": memory_limit,
                    "memory_limit": memory_limit,
                }
                _call_rerun_func(
                    web_func,
                    [
                        kwargs,
                        {key: kwargs[key] for key in ("open_browser", "web_port", "ws_port", "server_memory_limit")},
                        {key: kwargs[key] for key in ("open_browser", "web_port", "ws_port")},
                        {"open_browser": False},
                    ],
                )
                public_host = os.getenv("TELEOP_RERUN_PUBLIC_HOST", _local_ip_guess())
                local_url = f"http://127.0.0.1:{self.web_port}?url=ws://127.0.0.1:{self.ws_port}"
                public_url = f"http://{public_host}:{self.web_port}?url=ws://{public_host}:{self.ws_port}"
                logger_mp.info(
                    "Rerun web viewer started. Open %s or %s",
                    local_url,
                    public_url,
                )
                return
            if self.viewer != "auto":
                raise RuntimeError("This rerun-sdk does not expose serve_web/serve. Set TELEOP_RERUN_VIEWER=native to use rr.spawn().")

        if self.viewer in ("native", "spawn", "auto"):
            if memory_limit:
                rr.spawn(memory_limit=memory_limit, hide_welcome_screen=True)
            else:
                rr.spawn(hide_welcome_screen=True)
            logger_mp.info("Rerun native viewer spawn requested.")
            return

        raise ValueError("TELEOP_RERUN_VIEWER must be one of: web, native, connect, auto, off")

    def setup_blueprint(self):
        views = []

        data_plot_paths = [
                           f"{self.prefix}left_arm", 
                           f"{self.prefix}right_arm", 
                           f"{self.prefix}left_ee", 
                           f"{self.prefix}right_ee",
                           f"{self.prefix}body",
                           f"{self.prefix}neck",
                           f"{self.prefix}tactiles",
                           f"{self.prefix}audios",
        ]
        for plot_path in data_plot_paths:
            view = rrb.TimeSeriesView(
                origin = plot_path,
                time_ranges=[
                    rrb.VisibleTimeRange(
                        "idx",
                        start = rrb.TimeRangeBoundary.cursor_relative(seq = -self.IdxRangeBoundary),
                        end = rrb.TimeRangeBoundary.cursor_relative(),
                    )
                ],
                plot_legend = rrb.PlotLegend(visible = True),
            )
            views.append(view)

        image_plot_paths = [
                            f"{self.prefix}colors/color_0",
                            f"{self.prefix}colors/color_1",
                            f"{self.prefix}colors/color_2",
                            f"{self.prefix}colors/color_3",
        ]
        for plot_path in image_plot_paths:
            view = rrb.Spatial2DView(
                origin = plot_path,
                time_ranges=[
                    rrb.VisibleTimeRange(
                        "idx",
                        start = rrb.TimeRangeBoundary.cursor_relative(seq = -self.IdxRangeBoundary),
                        end = rrb.TimeRangeBoundary.cursor_relative(),
                    )
                ],
            )
            views.append(view)

        grid = rrb.Grid(contents = views,
                        grid_columns=3,               
        )
        rr.send_blueprint(grid)

    def _log_numeric_values(self, path, values):
        for idx, val in enumerate(_flatten_numeric(values)):
            rr.log(f"{path}/{idx}", rr.Scalar(val))

    def log_status(self, name, value=1.0):
        rr.set_time_sequence("idx", 0)
        rr.log(f"{self.prefix}status/{name}", rr.Scalar(float(value)))

    def _log_numeric_summary(self, path, values):
        values = _flatten_numeric(values)
        if not values:
            return
        rr.log(f"{path}/count", rr.Scalar(len(values)))
        rr.log(f"{path}/mean", rr.Scalar(sum(values) / len(values)))
        rr.log(f"{path}/min", rr.Scalar(min(values)))
        rr.log(f"{path}/max", rr.Scalar(max(values)))

    def _log_state_or_action_group(self, group_name, group_data):
        for part, part_info in (group_data or {}).items():
            if not isinstance(part_info, dict):
                self._log_numeric_values(f"{self.prefix}{part}/{group_name}/value", part_info)
                continue
            logged_standard_field = False
            for field in ("qpos", "qvel", "torque"):
                values = part_info.get(field, [])
                flat_values = _flatten_numeric(values)
                if flat_values:
                    logged_standard_field = True
                    self._log_numeric_values(f"{self.prefix}{part}/{group_name}/{field}", flat_values)
            if not logged_standard_field:
                for key, values in part_info.items():
                    self._log_numeric_values(f"{self.prefix}{part}/{group_name}/{key}", values)

    def _log_images(self, item_data):
        color_format = item_data.get("_image_color_format", "bgr")
        colors = item_data.get('colors', {}) or {}
        for color_key, color_val in colors.items():
            if not _is_image_array(color_val):
                continue
            image = _to_rgb_image(color_val, color_format)
            if image is not None:
                rr.log(f"{self.prefix}colors/{color_key}", rr.Image(image).compress(jpeg_quality=self.image_jpeg_quality))

        depths = item_data.get('depths', {}) or {}
        for depth_key, depth_val in depths.items():
            if not _is_image_array(depth_val):
                continue
            image = _to_rgb_image(depth_val, color_format)
            if image is not None:
                rr.log(f"{self.prefix}depths/{depth_key}", rr.Image(image))

    def _log_tactiles(self, tactiles):
        for side, tactile_vals in (tactiles or {}).items():
            if side.startswith("_"):
                continue
            if not isinstance(tactile_vals, dict):
                self._log_numeric_values(f"{self.prefix}tactiles/{side}/value", tactile_vals)
                continue
            fingers = tactile_vals.get("fingers", {}) or {}
            for finger, values in fingers.items():
                self._log_numeric_values(f"{self.prefix}tactiles/{side}/fingers/{finger}", values)
            if "palm" in tactile_vals:
                self._log_numeric_values(f"{self.prefix}tactiles/{side}/palm", tactile_vals.get("palm"))
            self._log_numeric_summary(f"{self.prefix}tactiles/{side}/summary", tactile_vals)

    def _log_audios(self, audios):
        if np is None:
            return
        for audio_key, audio_val in (audios or {}).items():
            if not isinstance(audio_val, np.ndarray) or audio_val.size == 0:
                continue
            audio = np.asarray(audio_val, dtype=np.float64).reshape(-1)
            rr.log(f"{self.prefix}audios/{audio_key}/samples", rr.Scalar(audio.size))
            rr.log(f"{self.prefix}audios/{audio_key}/rms", rr.Scalar(float(np.sqrt(np.mean(audio * audio)))))
            rr.log(f"{self.prefix}audios/{audio_key}/peak_abs", rr.Scalar(float(np.max(np.abs(audio)))))

    def log_item_data(self, item_data: dict):
        rr.set_time_sequence("idx", item_data.get('idx', 0))

        self._log_state_or_action_group("states", item_data.get('states', {}) or {})
        self._log_state_or_action_group("actions", item_data.get('actions', {}) or {})
        self._log_images(item_data)
        self._log_tactiles(item_data.get('tactiles', {}) or {})
        self._log_audios(item_data.get('audios', {}) or {})

    def log_episode_data(self, episode_data: list):
        for item_data in episode_data:
            self.log_item_data(item_data)


def _episode_ids_from_task_dir(task_dir):
    episode_ids = []
    for name in sorted(os.listdir(task_dir)):
        if not name.startswith("episode_"):
            continue
        try:
            episode_ids.append(int(name.split("_", 1)[1]))
        except ValueError:
            continue
    return episode_ids


def main():
    parser = argparse.ArgumentParser(description="Replay saved xr_teleoperate episodes in Rerun.")
    parser.add_argument("--task-dir", default="./utils/data/pick cube", help="Directory containing episode_xxxx folders.")
    parser.add_argument("--episodes", type=int, nargs="+", default=None, help="Episode ids to replay. Defaults to all episodes in --task-dir.")
    parser.add_argument("--prefix", default="offline/")
    parser.add_argument("--memory-limit", default="300MB")
    parser.add_argument("--idx-window", type=int, default=300)
    parser.add_argument("--sleep", type=float, default=0.0, help="Delay between frames while replaying.")
    args = parser.parse_args()

    task_dir = os.path.abspath(args.task_dir)
    episode_ids = args.episodes if args.episodes is not None else _episode_ids_from_task_dir(task_dir)
    if not episode_ids:
        raise SystemExit(f"No episode_xxxx folders found in {task_dir}")

    reader = RerunEpisodeReader(task_dir=task_dir)
    rerun_logger = RerunLogger(prefix=args.prefix, IdxRangeBoundary=args.idx_window, memory_limit=args.memory_limit)
    for episode_id in episode_ids:
        logger_mp.info("Replaying episode_%04d in Rerun", episode_id)
        for item_data in reader.return_episode_data(episode_id):
            rerun_logger.log_item_data(item_data)
            if args.sleep > 0.0:
                time.sleep(args.sleep)
    logger_mp.info("Rerun replay complete.")


if __name__ == "__main__":
    main()
