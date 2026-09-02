import unittest

import numpy as np

from teleop.utils.synthetic_image_client import SyntheticImageClient


class SyntheticImageClientTest(unittest.TestCase):
    def test_three_streams_match_declared_shapes(self):
        client = SyntheticImageClient(fps=20.0)
        config = client.get_cam_config()
        frames = {
            "head_camera": client.get_head_frame(),
            "left_wrist_camera": client.get_left_wrist_frame(),
            "right_wrist_camera": client.get_right_wrist_frame(),
        }
        for key, frame in frames.items():
            self.assertEqual(frame.bgr.shape[:2], tuple(config[key]["image_shape"]))
            self.assertEqual(frame.bgr.dtype, np.uint8)
            self.assertTrue(frame.bgr.flags.c_contiguous)
            self.assertIsNone(frame.jpg)
        client.close()


if __name__ == "__main__":
    unittest.main()
