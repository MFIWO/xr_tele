import unittest

import numpy as np

from teleop.utils.replay_visualizer import (
    build_replay_series,
    camera_labels_from_info,
)


def _qpos(values):
    return {"qpos": values}


class ReplayVisualizerTest(unittest.TestCase):
    def test_build_replay_series_keeps_sources_and_signed_differences_separate(self):
        frame = {
            "states": {
                "left_arm": _qpos([1.0, 2.0]),
                "left_ee": _qpos([0.1, 0.2, 0.3]),
            },
            "actions": {
                "left_arm": _qpos([1.5, 1.0]),
                "left_ee": _qpos([0.2, 0.4, 0.6]),
            },
        }
        sent = {
            "left_arm": _qpos(np.array([1.4, 1.2])),
            "left_ee": _qpos([0.2, 0.3, 0.5]),
        }
        live = {
            "left_arm": _qpos([1.1, 1.8]),
            "left_ee": _qpos([0.15, 0.25, 0.55]),
        }

        series = build_replay_series(frame, sent, live)

        self.assertEqual(series["left_arm"]["recorded_state"], [1.0, 2.0])
        self.assertEqual(series["left_arm"]["target_action"], [1.5, 1.0])
        self.assertEqual(series["left_arm"]["sent_command"], [1.4, 1.2])
        self.assertEqual(series["left_arm"]["live_measured"], [1.1, 1.8])
        np.testing.assert_allclose(
            series["left_arm"]["target_minus_recorded"], [0.5, -1.0]
        )
        np.testing.assert_allclose(
            series["left_arm"]["target_minus_sent"], [0.1, -0.2]
        )
        np.testing.assert_allclose(
            series["left_arm"]["target_minus_live"], [0.4, -0.8]
        )
        np.testing.assert_allclose(
            series["left_arm"]["sent_minus_live"], [0.3, -0.6]
        )

    def test_build_replay_series_omits_missing_or_shape_mismatched_differences(self):
        frame = {
            "states": {"right_arm": _qpos([0.0, 1.0])},
            "actions": {"right_arm": _qpos([0.5, 1.5])},
        }
        sent = {"right_arm": _qpos([0.4])}

        series = build_replay_series(frame, sent_actions=sent, live_states={})

        self.assertEqual(series["right_arm"]["sent_command"], [0.4])
        self.assertNotIn("target_minus_sent", series["right_arm"])
        self.assertNotIn("live_measured", series["right_arm"])
        self.assertNotIn("target_minus_live", series["right_arm"])
        self.assertNotIn("sent_minus_live", series["right_arm"])
        self.assertNotIn("left_arm", series)

    def test_build_replay_series_rejects_nonfinite_vectors_without_fake_zeros(self):
        frame = {
            "states": {"left_arm": _qpos([0.0, float("nan")])},
            "actions": {"left_arm": _qpos([1.0, 2.0])},
        }

        series = build_replay_series(frame)

        self.assertEqual(series["left_arm"], {"target_action": [1.0, 2.0]})

    def test_camera_labels_prefers_explicit_metadata_in_either_mapping_direction(self):
        info = {
            "recording": {
                "robot": {"arm": "AI_WORKER"},
                "camera": {
                    "color_keys": {
                        "head_stereo_left": "color_4",
                        "color_7": "tool_closeup",
                    }
                },
            }
        }

        self.assertEqual(
            camera_labels_from_info(info),
            {
                "color_4": "head_stereo_left",
                "color_7": "tool_closeup",
            },
        )

    def test_camera_labels_uses_ai_worker_legacy_monocular_fallback(self):
        info = {
            "recording": {
                "robot": {"arm": "AI_WORKER"},
                "ai_worker": {"dds_domain_id": 30},
            }
        }

        self.assertEqual(
            camera_labels_from_info(info),
            {
                "color_0": "head",
                "color_1": "left_wrist",
                "color_2": "right_wrist",
            },
        )

    def test_camera_labels_does_not_guess_ai_worker_order_for_other_robots(self):
        self.assertEqual(
            camera_labels_from_info(
                {"recording": {"robot": {"arm": "H1_2"}}}
            ),
            {},
        )


if __name__ == "__main__":
    unittest.main()
