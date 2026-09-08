import unittest

from teleop.utils.loop_streamer import LoopRobotStreamer, ee_dim_per_hand


class LoopStreamerAIWorkerTest(unittest.TestCase):
    def test_ai_worker_constructor_and_channel_root(self):
        streamer = LoopRobotStreamer(
            "localhost:50051", "hx5_d20", 30, arm="AI_WORKER"
        )
        self.assertEqual(streamer._root, "ai_worker")
        self.assertEqual(streamer._ee_dim, 20)

    def test_ai_worker_hx5_schema_preserves_both_twenty_dof_hands(self):
        streamer = LoopRobotStreamer(
            "localhost:50051", "hx5_d20", 30, arm="AI_WORKER"
        )
        step = streamer._build_step(
            range(7),
            range(7, 14),
            range(7),
            range(7, 14),
            range(20),
            range(20, 40),
            range(40, 60),
            range(60, 80),
        )
        robot = step["ai_worker"]
        self.assertEqual(
            len(robot["observation"]["left_ee"]["joint_position"]), 20
        )
        self.assertEqual(
            len(robot["observation"]["right_ee"]["joint_position"]), 20
        )
        self.assertEqual(len(robot["action"]["left_ee"]["joint_position"]), 20)
        self.assertEqual(len(robot["action"]["right_ee"]["joint_position"]), 20)

    def test_ai_worker_2g_schema_has_one_gripper_joint_per_side(self):
        streamer = LoopRobotStreamer(
            "localhost:50051", "ai_worker_2g", 30, arm="AI_WORKER"
        )
        step = streamer._build_step(
            range(7),
            range(7, 14),
            range(7),
            range(7, 14),
            [0.2],
            [0.8],
            [0.3],
            [0.7],
        )

        robot = step["ai_worker"]
        self.assertEqual(
            robot["observation"]["left_ee"]["joint_position"], [0.2]
        )
        self.assertEqual(
            robot["observation"]["right_ee"]["joint_position"], [0.8]
        )
        self.assertEqual(robot["action"]["left_ee"]["joint_position"], [0.3])
        self.assertEqual(robot["action"]["right_ee"]["joint_position"], [0.7])

    def test_legacy_embodiments_keep_g1_root(self):
        streamer = LoopRobotStreamer(
            "localhost:50051", "dex3", 30, arm="H1_2"
        )
        self.assertIn("g1", streamer._build_step([], [], [], [], [], [], [], []))
        self.assertEqual(ee_dim_per_hand("hx5_d20"), 20)
        self.assertEqual(ee_dim_per_hand("ai_worker_2g"), 1)


if __name__ == "__main__":
    unittest.main()
