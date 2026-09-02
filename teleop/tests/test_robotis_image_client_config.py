from types import SimpleNamespace
import unittest

from teleop.robot_control.robotis_image_client import RobotisDDSImageClient


class RobotisImageClientConfigTest(unittest.TestCase):
    def test_ai_worker_camera_config_is_explicit_dds_monocular(self):
        config = RobotisDDSImageClient._camera_config((480, 640))
        self.assertEqual(config["source"], "robotis_dds")
        self.assertTrue(config["enable_zmq"])
        self.assertFalse(config["enable_webrtc"])
        self.assertFalse(config["binocular"])
        self.assertEqual(config["image_shape"], [480, 640])

    def test_dds_camera_source_timestamp_conversion(self):
        message = SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(sec=12, nanosec=345))
        )
        self.assertEqual(
            RobotisDDSImageClient._source_time_ns(message),
            12_000_000_345,
        )
        self.assertIsNone(RobotisDDSImageClient._source_time_ns(SimpleNamespace()))


if __name__ == "__main__":
    unittest.main()
