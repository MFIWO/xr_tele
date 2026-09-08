import tempfile
import unittest
from pathlib import Path

from teleop.robot_control.robot_hand_hx5_d20 import resolve_hx5_d20_urdf


class HX5D20UrdfResolutionTest(unittest.TestCase):
    def test_derives_hand_urdfs_from_selected_arm_checkout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            urdf_root = Path(temp_dir) / "ai_worker" / "ffw_description" / "urdf"
            arm_urdf = urdf_root / "ffw_sg2_rev1_follower" / "ffw_sg2_follower.urdf"

            self.assertEqual(
                resolve_hx5_d20_urdf("left", arm_urdf),
                (urdf_root / "common" / "hx5_d20" / "hx5_d20_left.urdf").resolve(),
            )
            self.assertEqual(
                resolve_hx5_d20_urdf("right", arm_urdf),
                (urdf_root / "common" / "hx5_d20" / "hx5_d20_right.urdf").resolve(),
            )

    def test_workspace_default_finds_external_repos_checkout(self):
        resolved = resolve_hx5_d20_urdf("left")

        self.assertTrue(resolved.is_file())
        self.assertIn("external_repos/ai_worker", str(resolved))

    def test_rejects_invalid_side(self):
        with self.assertRaisesRegex(ValueError, "side"):
            resolve_hx5_d20_urdf("middle")


if __name__ == "__main__":
    unittest.main()
