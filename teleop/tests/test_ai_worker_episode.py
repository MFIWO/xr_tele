import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from teleop.robot_control.robot_hand_hx5_d20 import (
    LEFT_LOWER as HX5_LEFT_LOWER,
    LEFT_UPPER as HX5_LEFT_UPPER,
    RIGHT_LOWER as HX5_RIGHT_LOWER,
    RIGHT_UPPER as HX5_RIGHT_UPPER,
)
from teleop.robot_control.robotis_ai_worker import (
    AI_WORKER_ARM_LOWER,
    AI_WORKER_ARM_UPPER,
)
from teleop.utils.ai_worker_episode import load_episode, preflight_ai_worker_episode


def _info(*, arm="AI_WORKER", end_effector="hx5_d20", fps=30.0):
    return {
        "image": {"fps": fps},
        "recording": {
            "robot": {
                "arm": arm,
                "end_effector": end_effector,
            }
        },
    }


def _inside(lower, upper):
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    return (lower + 0.37 * (upper - lower)).tolist()


def _frame(idx=0, *, states=True):
    frame = {
        "idx": idx,
        "colors": {
            "color_0": "colors/000000_color_0.jpg",
            "color_1": "colors/000000_color_1.jpg",
            "color_2": "colors/000000_color_2.jpg",
        },
        "actions": {
            "left_arm": {
                "qpos": _inside(AI_WORKER_ARM_LOWER[:7], AI_WORKER_ARM_UPPER[:7]),
            },
            "right_arm": {
                "qpos": _inside(AI_WORKER_ARM_LOWER[7:], AI_WORKER_ARM_UPPER[7:]),
            },
            "left_ee": {"qpos": _inside(HX5_LEFT_LOWER, HX5_LEFT_UPPER)},
            "right_ee": {"qpos": _inside(HX5_RIGHT_LOWER, HX5_RIGHT_UPPER)},
        },
    }
    if states:
        frame["states"] = {
            "left_arm": {"qpos": [10.0] * 7},
            "right_arm": {"qpos": [11.0] * 7},
            "left_ee": {"qpos": [12.0] * 20},
            "right_ee": {"qpos": [13.0] * 20},
        }
    return frame


class AIWorkerEpisodeTest(unittest.TestCase):
    def test_load_episode_accepts_directory_and_preserves_source_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            episode_dir = Path(temp_dir) / "episode_0001"
            episode_dir.mkdir()
            data_json = episode_dir / "data.json"
            data_json.write_text(
                json.dumps({"info": _info(fps=25), "data": [_frame()]}),
                encoding="utf-8",
            )

            fps, frames, resolved, info = load_episode(episode_dir)

            self.assertEqual(fps, 25.0)
            self.assertEqual(resolved, data_json)
            self.assertEqual(frames[0]["idx"], 0)
            self.assertEqual(info["recording"]["robot"]["arm"], "AI_WORKER")

    def test_load_episode_repairs_only_missing_final_container_closures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_json = Path(temp_dir) / "data.json"
            complete = json.dumps({"info": _info(), "data": [_frame()]})
            self.assertTrue(complete.endswith("]}"))
            data_json.write_text(complete[:-2], encoding="utf-8")

            fps, frames, _, _ = load_episode(data_json)

            self.assertEqual(fps, 30.0)
            self.assertEqual(len(frames), 1)

    def test_load_episode_does_not_repair_malformed_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_json = Path(temp_dir) / "data.json"
            data_json.write_text(
                '{"info":{"image":{"fps":30}},"data":[{"actions":BROKEN}]}',
                encoding="utf-8",
            )

            with self.assertRaises(json.JSONDecodeError):
                load_episode(data_json)

    def test_preflight_rejects_invalid_fps(self):
        for fps in [None, 0.0, -1.0, float("nan"), float("inf"), True]:
            with self.subTest(fps=fps):
                with self.assertRaisesRegex(ValueError, "info.image.fps"):
                    preflight_ai_worker_episode(_info(fps=fps), [_frame()])

    def test_preflight_requires_explicit_ai_worker_hx5_metadata(self):
        cases = [
            (None, "hx5_d20"),
            ("AI_WORKER", None),
            ("H1_2", "hx5_d20"),
            ("AI_WORKER", "inspire_dg2"),
        ]
        for arm, end_effector in cases:
            with self.subTest(arm=arm, end_effector=end_effector):
                with self.assertRaisesRegex(ValueError, "episode metadata"):
                    preflight_ai_worker_episode(
                        _info(arm=arm, end_effector=end_effector),
                        [_frame()],
                    )

    def test_preflight_metadata_override_does_not_relax_action_validation(self):
        frame = _frame()
        del frame["actions"]["left_arm"]

        with self.assertRaisesRegex(
            ValueError, r"missing actions\.left_arm\.qpos"
        ):
            preflight_ai_worker_episode(
                _info(arm="H1_2", end_effector="inspire_dg2"),
                [frame],
                allow_metadata_mismatch=True,
            )

    def test_preflight_never_falls_back_to_recorded_state_for_missing_action(self):
        frame = _frame(states=True)
        del frame["actions"]["left_ee"]
        self.assertEqual(len(frame["states"]["left_ee"]["qpos"]), 20)

        with self.assertRaisesRegex(ValueError, r"missing actions\.left_ee\.qpos"):
            preflight_ai_worker_episode(_info(), [frame])

    def test_preflight_rejects_bad_action_dimensions_and_values(self):
        cases = [
            ("left_arm", [0.0] * 6, "length 6 != 7"),
            ("right_arm", [0.0] * 8, "length 8 != 7"),
            ("left_ee", [0.0] * 19, "length 19 != 20"),
            ("right_ee", [0.0] * 21, "length 21 != 20"),
            ("left_arm", [float("nan")] + [0.0] * 6, "non-finite"),
            ("right_ee", [float("inf")] + [0.0] * 19, "non-finite"),
        ]
        for part, bad_qpos, message in cases:
            with self.subTest(part=part, message=message):
                frame = _frame()
                frame["actions"][part]["qpos"] = bad_qpos

                with self.assertRaisesRegex(ValueError, message):
                    preflight_ai_worker_episode(_info(), [frame])

    def test_preflight_rejects_out_of_bounds_actions(self):
        cases = [
            ("left_arm", 0, float(AI_WORKER_ARM_UPPER[0] + 0.01)),
            ("right_arm", 1, float(AI_WORKER_ARM_LOWER[8] - 0.01)),
            ("left_ee", 3, float(HX5_LEFT_LOWER[3] - 0.01)),
            ("right_ee", 7, float(HX5_RIGHT_UPPER[7] + 0.01)),
        ]
        for part, joint, value in cases:
            with self.subTest(part=part, joint=joint, value=value):
                frame = _frame()
                frame["actions"][part]["qpos"][joint] = value

                with self.assertRaisesRegex(ValueError, "outside joint bounds"):
                    preflight_ai_worker_episode(_info(), [frame])

    def test_preflight_preserves_hx5_side_and_joint_order_and_optional_states(self):
        frame = _frame(idx=17, states=True)
        validated = preflight_ai_worker_episode(_info(), [frame])

        self.assertEqual(len(validated), 1)
        replay_frame = validated[0]
        self.assertEqual(replay_frame.idx, 17)
        np.testing.assert_array_equal(
            replay_frame.left_ee_action,
            frame["actions"]["left_ee"]["qpos"],
        )
        np.testing.assert_array_equal(
            replay_frame.right_ee_action,
            frame["actions"]["right_ee"]["qpos"],
        )
        np.testing.assert_array_equal(
            replay_frame.hand_action,
            frame["actions"]["left_ee"]["qpos"]
            + frame["actions"]["right_ee"]["qpos"],
        )
        np.testing.assert_array_equal(
            replay_frame.recorded_states["left_arm"], [10.0] * 7
        )
        self.assertTrue(
            replay_frame.source_frame["colors"]["color_0"].endswith(
                "color_0.jpg"
            )
        )

    def test_preflight_allows_missing_recorded_states_and_disabled_component_actions(self):
        frame = _frame(states=False)
        del frame["actions"]["left_ee"]
        del frame["actions"]["right_ee"]

        validated = preflight_ai_worker_episode(
            _info(),
            [frame],
            replay_hand=False,
        )

        self.assertEqual(validated[0].recorded_states, {})
        self.assertIsNotNone(validated[0].arm_action)
        self.assertIsNone(validated[0].hand_action)

    def test_preflight_treats_empty_recorded_qpos_as_optional(self):
        frame = _frame(states=True)
        frame["states"]["left_ee"]["qpos"] = []

        validated = preflight_ai_worker_episode(_info(), [frame])

        self.assertNotIn("left_ee", validated[0].recorded_states)
        self.assertIsNotNone(validated[0].hand_action)

    def test_preflight_requires_at_least_one_replay_component(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            preflight_ai_worker_episode(
                _info(),
                [_frame()],
                replay_arm=False,
                replay_hand=False,
            )


if __name__ == "__main__":
    unittest.main()
