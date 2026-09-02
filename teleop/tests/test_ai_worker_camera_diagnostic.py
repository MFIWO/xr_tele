from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest

import numpy as np

from teleop.ai_worker_camera_diagnostic import (
    collect_camera_diagnostics,
    main,
    run_camera_diagnostic,
)
from teleop.robot_control.robotis_image_client import AI_WORKER_CAMERA_TOPICS


REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeImageClient:
    def __init__(self, domain_id=None, frames=None, timestamps=None):
        self.domain_id = domain_id
        self.closed = False
        self.frames = frames or {
            "head": SimpleNamespace(
                bgr=np.zeros((720, 1280, 3), dtype=np.uint8),
                fps=29.5,
            ),
            "left_wrist": SimpleNamespace(
                bgr=np.zeros((480, 640, 3), dtype=np.uint8),
                fps=30.0,
            ),
            "right_wrist": SimpleNamespace(
                bgr=np.zeros((480, 640, 3), dtype=np.uint8),
                fps=30.5,
            ),
        }
        now_ns = time.time_ns()
        self.timestamps = timestamps or {
            name: {"source_time_ns": None, "receive_time_ns": now_ns}
            for name in AI_WORKER_CAMERA_TOPICS
        }

    def get_cam_config(self):
        return {
            "head_camera": {"image_shape": [720, 1280]},
            "left_wrist_camera": {"image_shape": [480, 640]},
            "right_wrist_camera": {"image_shape": [480, 640]},
        }

    def get_frame_timestamps(self):
        return self.timestamps

    def get_head_frame(self):
        return self.frames.get("head")

    def get_left_wrist_frame(self):
        return self.frames.get("left_wrist")

    def get_right_wrist_frame(self):
        return self.frames.get("right_wrist")

    def close(self):
        self.closed = True


class AIWorkerCameraDiagnosticTest(unittest.TestCase):
    def test_fresh_three_camera_snapshot_reports_topics_shapes_fps_and_age(self):
        client = _FakeImageClient(domain_id=30)
        lines = []

        exit_code = run_camera_diagnostic(
            client,
            duration=0.0,
            freshness=0.5,
            output=lines.append,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(lines), 4)
        for name, topic in AI_WORKER_CAMERA_TOPICS.items():
            line = next(line for line in lines if line.startswith(f"{name}:"))
            self.assertIn("status=OK", line)
            self.assertIn(f"topic={topic}", line)
            self.assertIn("frame_shape=", line)
            self.assertIn("fps=", line)
            self.assertIn("age=", line)
        self.assertTrue(lines[-1].endswith("PASS (all three frames are fresh)"))

    def test_missing_and_stale_frames_fail(self):
        now_ns = 20_000_000_000
        frames = _FakeImageClient().frames
        frames["left_wrist"] = SimpleNamespace(bgr=None, fps=0.0)
        timestamps = {
            "head": {"receive_time_ns": now_ns - 10_000_000},
            "left_wrist": {"receive_time_ns": None},
            "right_wrist": {"receive_time_ns": now_ns - 2_000_000_000},
        }
        client = _FakeImageClient(frames=frames, timestamps=timestamps)

        results = collect_camera_diagnostics(
            client,
            freshness=0.5,
            now_ns=now_ns,
        )

        self.assertEqual(
            {result.name: result.status for result in results},
            {"head": "OK", "left_wrist": "MISSING", "right_wrist": "STALE"},
        )

    def test_main_passes_domain_to_only_factory_and_always_closes(self):
        clients = []
        lines = []

        def factory(domain_id):
            client = _FakeImageClient(domain_id=domain_id)
            clients.append(client)
            return client

        exit_code = main(
            ["--domain", "30", "--duration", "0.001", "--freshness", "1.0"],
            client_factory=factory,
            output=lines.append,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(clients), 1)
        self.assertEqual(clients[0].domain_id, 30)
        self.assertTrue(clients[0].closed)
        self.assertIn("camera readers only", lines[1])

    def test_import_does_not_load_any_motor_control_module(self):
        script = """
import builtins

real_import = builtins.__import__
blocked = {
    'teleop.robot_control.robotis_ai_worker',
    'teleop.robot_control.robot_hand_hx5_d20',
    'teleop.robot_control.robotis_ai_worker_lift',
    'teleop.robot_control.robotis_dds',
}
def guarded_import(name, *args, **kwargs):
    if name in blocked:
        raise AssertionError(f'diagnostic imported motor module: {name}')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

import teleop.ai_worker_camera_diagnostic
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
