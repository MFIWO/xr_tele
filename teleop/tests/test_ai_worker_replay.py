import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from teleop.ai_worker_replay import (
    _episode_domain,
    _qpos_group,
    _quintic_blend,
    main,
    parse_args,
)


def _episode():
    q_arm = [0.0] * 7
    q_hand = [0.0] * 20
    frame = {
        "idx": 0,
        "colors": {},
        "states": {
            "left_arm": {"qpos": q_arm},
            "right_arm": {"qpos": q_arm},
            "left_ee": {"qpos": q_hand},
            "right_ee": {"qpos": q_hand},
        },
        "actions": {
            "left_arm": {"qpos": q_arm},
            "right_arm": {"qpos": q_arm},
            "left_ee": {"qpos": q_hand},
            "right_ee": {"qpos": q_hand},
        },
    }
    return {
        "info": {
            "image": {"fps": 30.0},
            "recording": {
                "robot": {"arm": "AI_WORKER", "end_effector": "hx5_d20"},
                "ai_worker": {"dds_domain_id": 30},
            },
        },
        "data": [frame],
    }


class AIWorkerReplayTest(unittest.TestCase):
    def test_replay_requires_explicit_execute_for_dds(self):
        args = parse_args(["episode_0001"])
        self.assertFalse(args.execute)

        args = parse_args(["episode_0001", "--execute"])
        self.assertTrue(args.execute)

    def test_replay_rejects_disabling_every_command_group(self):
        with self.assertRaises(SystemExit):
            parse_args(["episode_0001", "--no-arm", "--no-hand"])

    def test_replay_rejects_unsafe_numeric_options(self):
        cases = [
            ["--hz", "nan"],
            ["--command-duration", "inf"],
            ["--startup-blend", "-1"],
            ["--arm-velocity-limit", "0"],
        ]
        for flags in cases:
            with self.subTest(flags=flags):
                with self.assertRaises(SystemExit):
                    parse_args(["episode_0001", *flags])

    def test_replay_group_preserves_left_then_right_slices(self):
        arm = np.arange(14, dtype=np.float64)
        hand = np.arange(100, 140, dtype=np.float64)

        group = _qpos_group(arm_q=arm, hand_q=hand)

        self.assertEqual(group["left_arm"]["qpos"], list(range(7)))
        self.assertEqual(group["right_arm"]["qpos"], list(range(7, 14)))
        self.assertEqual(group["left_ee"]["qpos"], list(range(100, 120)))
        self.assertEqual(group["right_ee"]["qpos"], list(range(120, 140)))

    def test_quintic_blend_has_stationary_endpoints(self):
        self.assertAlmostEqual(_quintic_blend(0.0), 0.0)
        self.assertAlmostEqual(_quintic_blend(1.0), 1.0)
        epsilon = 1e-5
        self.assertLess(_quintic_blend(epsilon) / epsilon, 1e-6)
        self.assertLess(
            (1.0 - _quintic_blend(1.0 - epsilon)) / epsilon,
            1e-6,
        )

    def test_episode_domain_reads_optional_metadata(self):
        self.assertEqual(_episode_domain(_episode()["info"]), 30)
        self.assertIsNone(_episode_domain({"recording": {}}))

    def test_visualization_only_main_never_constructs_dds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            episode_dir = Path(temp_dir) / "episode_0001"
            episode_dir.mkdir()
            (episode_dir / "data.json").write_text(
                json.dumps(_episode()), encoding="utf-8"
            )

            with mock.patch(
                "teleop.ai_worker_replay.socket.socket",
                side_effect=AssertionError(
                    "dry-run must not construct UDP/DDS support"
                ),
            ):
                result = main(
                    [
                        str(episode_dir),
                        "--dry-run",
                        "--no-visualize",
                        "--hz",
                        "100000",
                    ]
                )

            self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
