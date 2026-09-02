import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from teleop.ai_worker_replay import (
    _arm_home_q,
    _episode_domain,
    _qpos_group,
    _quintic_blend,
    _read_post_replay_choice,
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

    def test_post_replay_prompt_requires_executed_arm_replay(self):
        with self.assertRaises(SystemExit):
            parse_args(["episode_0001", "--post-replay-prompt"])
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "episode_0001",
                    "--execute",
                    "--no-arm",
                    "--post-replay-prompt",
                ]
            )

    def test_post_replay_prompt_accepts_r_and_q(self):
        for entered, expected in (("R\n", "r"), ("q\n", "q")):
            with self.subTest(entered=entered):
                output = io.StringIO()
                self.assertEqual(
                    _read_post_replay_choice(io.StringIO(entered), output),
                    expected,
                )

    def test_home_pose_uses_selected_ai_worker_model(self):
        np.testing.assert_array_equal(_arm_home_q("sg2"), np.zeros(14))
        self.assertAlmostEqual(_arm_home_q("sh5")[3], -1.57)
        self.assertAlmostEqual(_arm_home_q("sh5")[10], -1.57)

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

    def test_execute_checks_keyboard_estop_before_opening_visualizer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            episode_dir = Path(temp_dir) / "episode_0001"
            episode_dir.mkdir()
            (episode_dir / "data.json").write_text(
                json.dumps(_episode()), encoding="utf-8"
            )

            with (
                mock.patch("teleop.ai_worker_replay.PedalMotionInhibitor"),
                mock.patch("teleop.ai_worker_replay.PedalEstopReceiver") as receiver,
                mock.patch("teleop.ai_worker_replay.ReplayVisualizer") as visualizer,
            ):
                receiver.return_value.wait_for_state.return_value = False

                with self.assertRaisesRegex(
                    RuntimeError,
                    "pedal/keyboard ESTOP heartbeat",
                ):
                    main(
                        [
                            str(episode_dir),
                            "--execute",
                            "--no-hand",
                            "--viewer-hold-seconds",
                            "30",
                        ]
                    )

            visualizer.assert_not_called()

    def test_post_replay_r_repeats_then_q_moves_home(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            episode_dir = Path(temp_dir) / "episode_0001"
            episode_dir.mkdir()
            (episode_dir / "data.json").write_text(
                json.dumps(_episode()), encoding="utf-8"
            )

            arm_ctrl = mock.Mock()
            arm_ctrl.home_q = np.zeros(14)
            arm_ctrl.wait_for_joint_state.return_value = True
            arm_ctrl.sync_arm_command_to_measured.return_value = np.zeros(14)
            arm_ctrl.get_current_dual_arm_q.return_value = np.zeros(14)
            arm_ctrl.get_last_commanded_dual_arm_q.return_value = np.zeros(14)
            replay_visualizer = mock.Mock()
            replay_visualizer.dropped_frames = 0

            with (
                mock.patch("teleop.ai_worker_replay.PedalMotionInhibitor"),
                mock.patch(
                    "teleop.robot_control.robotis_ai_worker.AIWorkerArmController",
                    return_value=arm_ctrl,
                ),
                mock.patch(
                    "teleop.ai_worker_replay.ReplayVisualizer",
                    return_value=replay_visualizer,
                ),
                mock.patch(
                    "teleop.ai_worker_replay._read_post_replay_choice",
                    side_effect=("r", "q"),
                ) as read_choice,
                mock.patch("teleop.ai_worker_replay._blend_to_target") as blend,
                mock.patch(
                    "teleop.ai_worker_replay._wait_until",
                    return_value=False,
                ),
            ):
                result = main(
                    [
                        str(episode_dir),
                        "--execute",
                        "--no-hand",
                        "--no-pedal-estop",
                        "--post-replay-prompt",
                        "--hz",
                        "100000",
                        "--home-duration",
                        "3",
                    ]
                )

            self.assertEqual(result, 0)
            self.assertEqual(read_choice.call_count, 2)
            self.assertEqual(blend.call_count, 3)
            self.assertEqual([call.args[4] for call in blend.call_args_list], [2.0, 2.0, 3.0])
            np.testing.assert_array_equal(blend.call_args_list[-1].args[2], arm_ctrl.home_q)
            self.assertEqual(
                [
                    call.kwargs["timeline_idx"]
                    for call in replay_visualizer.submit.call_args_list
                ],
                [0, 1],
            )


if __name__ == "__main__":
    unittest.main()
