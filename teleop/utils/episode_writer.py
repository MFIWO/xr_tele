import os
import cv2
import json
import datetime
import numpy as np
import time
import re
import copy
from .rerun_visualizer import RerunLogger
from queue import Queue, Empty
from threading import Thread
import logging_mp
logger_mp = logging_mp.getLogger(__name__)

def _deep_update(dst, src):
    for key, value in src.items():
        if key == "audio":
            dst[key] = value
        elif isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst

def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class EpisodeWriter():
    def __init__(
        self,
        task_dir,
        task_goal=None,
        task_desc=None,
        task_steps=None,
        frequency=30,
        image_size=[640, 480],
        rerun_log=True,
        metadata=None,
    ):
        """
        image_size: [width, height]
        """
        logger_mp.info("==> EpisodeWriter initializing...\n")
        self.task_dir = task_dir
        self.text = {
            "goal": "Pick up the red cup on the table.",
            "desc": "task description",
            "steps":"step1: do this; step2: do that; ...",
        }
        if task_goal is not None:
            self.text['goal'] = task_goal
        if task_desc is not None:
            self.text['desc'] = task_desc
        if task_steps is not None:
            self.text['steps'] = task_steps

        self.frequency = frequency
        self.image_size = image_size
        self.metadata = metadata or {}

        self.rerun_log = rerun_log
        if self.rerun_log:
            logger_mp.info("==> RerunLogger initializing...\n")
            try:
                self.rerun_logger = RerunLogger(
                    prefix="online/",
                    IdxRangeBoundary=60,
                    memory_limit="300MB",
                )
                logger_mp.info("==> RerunLogger initializing ok.\n")
            except Exception:
                self.rerun_log = False
                self.rerun_logger = None
                logger_mp.exception(
                    "RerunLogger initialization failed; continuing recording without visualization."
                )
        
        self.item_id = -1
        self.episode_id = -1
        if os.path.exists(self.task_dir):
            episode_ids = []
            for episode_dir in os.listdir(self.task_dir):
                match = re.fullmatch(r"episode_(\d+)", episode_dir)
                if match and os.path.isdir(os.path.join(self.task_dir, episode_dir)):
                    episode_ids.append(int(match.group(1)))
            self.episode_id = 0 if not episode_ids else max(episode_ids)
            logger_mp.info(f"==> task_dir directory already exist, now self.episode_id is:{self.episode_id}\n")
        else:
            os.makedirs(self.task_dir)
            logger_mp.info(f"==> episode directory does not exist, now create one.\n")
        self.data_info()
        self._base_info = copy.deepcopy(self.info)
        self._info_dirty = False

        self.is_available = True  # Indicates whether the class is available for new operations
        # Initialize the queue and worker thread
        self.item_data_queue = Queue(-1)
        self.stop_worker = False
        self.need_save = False  # Flag to indicate when save_episode is triggered
        self.items_written = 0
        self.items_failed = 0
        self.last_error = None
        self.worker_thread = Thread(target=self.process_queue)
        self.worker_thread.start()

        logger_mp.info("==> EpisodeWriter initialized successfully.\n")
    
    def is_ready(self):
        return self.is_available

    def data_info(self, version='1.0.0', date=None, author=None):
        self.info = {
                "version": "1.0.0" if version is None else version, 
                "date": datetime.date.today().strftime('%Y-%m-%d') if date is None else date,
                "author": "unitree" if author is None else author,
                "image": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "depth": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "audio": {"sample_rate": 16000, "channels": 1, "format":"PCM", "bits":16},    # PCM_S16
                "joint_names":{
                    "left_arm":   [],
                    "left_ee":  [],
                    "right_arm":  [],
                    "right_ee": [],
                    "body":       [],
                },

                "tactile_names": {
                    "left_ee": [],
                    "right_ee": [],
                }, 
                "sim_state": ""
            }
        if self.metadata:
            self.info["recording"] = self.metadata

    def update_info(self, updates):
        if not isinstance(updates, dict):
            raise TypeError("updates must be a dictionary")
        _deep_update(self.info, updates)
        self._info_dirty = True

 
    def create_episode(self):
        """
        Create a new episode.
        Returns:
            bool: True if the episode is successfully created, False otherwise.
        Note:
            Once successfully created, this function will only be available again after save_episode complete its save task.
        """
        if not self.is_available:
            logger_mp.info("==> The class is currently unavailable for new operations. Please wait until ongoing tasks are completed.")
            return False  # Return False if the class is unavailable

        # Reset episode-related data and create necessary directories
        self.info = copy.deepcopy(self._base_info)
        self._info_dirty = False
        self.item_id = -1
        self.episode_id = self.episode_id + 1
        
        self.episode_dir = os.path.join(self.task_dir, f"episode_{str(self.episode_id).zfill(4)}")
        self.color_dir = os.path.join(self.episode_dir, 'colors')
        self.depth_dir = os.path.join(self.episode_dir, 'depths')
        self.audio_dir = os.path.join(self.episode_dir, 'audios')
        self.json_path = os.path.join(self.episode_dir, 'data.json')
        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(self.color_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        with open(self.json_path, "w", encoding="utf-8") as f:
            f.write('{\n')
            f.write('"info": ' + json.dumps(self.info, ensure_ascii=False, indent=4, default=_json_default) + ',\n')
            f.write('"text": ' + json.dumps(self.text, ensure_ascii=False, indent=4, default=_json_default) + ',\n')
            f.write('"data": [\n')
        self.first_item = True   # Flag to handle commas in JSON array
        self.items_written = 0
        self.items_failed = 0
        self.last_error = None

        self.is_available = False  # After the episode is created, the class is marked as unavailable until the episode is successfully saved
        logger_mp.info(f"==> New episode created: {self.episode_dir}")
        logger_mp.info(f"==> Episode metadata/state path: {self.json_path}")
        if self.rerun_log:
            self.rerun_logger.log_status("episode_active", 1.0)
        return True  # Return True if the episode is successfully created
        
    def add_item(self, colors, depths=None, states=None, actions=None, tactiles=None, audios=None, sim_state=None, timestamps=None):
        # Increment the item ID
        self.item_id += 1
        # Create the item data dictionary
        item_data = {
            'idx': self.item_id,
            'colors': colors,
            'depths': depths,
            'states': states,
            'actions': actions,
            'tactiles': tactiles,
            'audios': audios,
            'sim_state': sim_state,
        }
        if timestamps is not None:
            item_data['timestamps'] = timestamps
        # Enqueue the item data
        self.item_data_queue.put(item_data)

    def process_queue(self):
        while not self.stop_worker or not self.item_data_queue.empty():
            # Process items in the queue
            try:
                item_data = self.item_data_queue.get(timeout=1)
                try:
                    self._process_item_data(item_data)
                except Exception as e:
                    self.items_failed += 1
                    self.last_error = repr(e)
                    logger_mp.exception(f"Error processing item_data (idx={item_data['idx']}): {e}")
                self.item_data_queue.task_done()
            except Empty:
                pass
        
            # Check if save_episode was triggered
            if self.need_save and self.item_data_queue.empty():
                self._save_episode()

    def _process_item_data(self, item_data):
        idx = item_data['idx']
        colors = item_data.get('colors', {})
        depths = item_data.get('depths', {})
        audios = item_data.get('audios', {})
        rerun_item_data = None
        if self.rerun_log:
            rerun_item_data = {
                **item_data,
                "colors": dict(colors or {}),
                "depths": dict(depths or {}),
                "audios": dict(audios or {}),
                "_image_color_format": "bgr",
            }

        # Save images
        if colors:
            for idx_color, (color_key, color) in enumerate(colors.items()):
                color_name = f'{str(idx).zfill(6)}_{color_key}.jpg'
                if not cv2.imwrite(os.path.join(self.color_dir, color_name), color):
                    logger_mp.warning("Failed to save color image idx=%s key=%s", idx, color_key)
                item_data['colors'][color_key] = os.path.join('colors', color_name)

        # Save depths
        if depths:
            for idx_depth, (depth_key, depth) in enumerate(depths.items()):
                depth_name = f'{str(idx).zfill(6)}_{depth_key}.jpg'
                if not cv2.imwrite(os.path.join(self.depth_dir, depth_name), depth):
                    logger_mp.warning("Failed to save depth image idx=%s key=%s", idx, depth_key)
                item_data['depths'][depth_key] = os.path.join('depths', depth_name)

        # Save audios
        if audios:
            for mic, audio in audios.items():
                audio_name = f'audio_{str(idx).zfill(6)}_{mic}.npy'
                np.save(os.path.join(self.audio_dir, audio_name), audio.astype(np.int16))
                item_data['audios'][mic] = os.path.join('audios', audio_name)

        # Update episode data
        serialized_item = json.dumps(item_data, ensure_ascii=False, indent=4, default=_json_default)
        with open(self.json_path, "a", encoding="utf-8") as f:
            if not self.first_item:
                f.write(",\n")
            f.write(serialized_item)
            self.first_item = False
            self.items_written += 1

        # Log data if necessary
        if self.rerun_log:
            self.rerun_logger.log_item_data(rerun_item_data if rerun_item_data is not None else item_data)

    def save_episode(self):
        """
        Trigger the save operation. This sets the save flag, and the process_queue thread will handle it.
        """
        self.need_save = True  # Set the save flag
        logger_mp.info(f"==> Episode saved start...")

    def _save_episode(self):
        """
        Save the episode data to a JSON file.
        """
        with open(self.json_path, "a", encoding="utf-8") as f:
            f.write("\n]\n}")      # Close the JSON array and object
        if self._info_dirty:
            self._rewrite_info()
        if self.rerun_log:
            self.rerun_logger.log_status("episode_active", 0.0)

        self.need_save = False     # Reset the save flag
        self.is_available = True   # Mark the class as available after saving
        logger_mp.info(
            f"==> Episode saved successfully to {self.json_path}. "
            f"frames_written={self.items_written} frames_failed={self.items_failed}"
        )
        if self.items_failed:
            logger_mp.error(
                f"==> Episode contains dropped frames. count={self.items_failed} "
                f"last_error={self.last_error}"
            )

    def _rewrite_info(self):
        tmp_json_path = self.json_path + ".tmp"
        with open(self.json_path, "r", encoding="utf-8") as f:
            episode = json.load(f)
        episode["info"] = self.info
        with open(tmp_json_path, "w", encoding="utf-8") as f:
            json.dump(episode, f, ensure_ascii=False, indent=4, default=_json_default)
            f.write("\n")
        os.replace(tmp_json_path, self.json_path)
        self._info_dirty = False

    def close(self):
        """
        Stop the worker thread and ensure all tasks are completed.
        """
        self.item_data_queue.join()
        if not self.is_available:  # If self.is_available is False, it means there is still data not saved.
            self.save_episode()
        while not self.is_available:
            time.sleep(0.01)
        self.stop_worker = True
        self.worker_thread.join()
