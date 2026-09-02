import unittest

import numpy as np

from teleop.robot_control.robot_hand_hx5_d20 import (
    LEFT_JOINT_NAMES,
    LEFT_LOWER,
    LEFT_UPPER,
    RIGHT_JOINT_NAMES,
    RIGHT_LOWER,
    RIGHT_UPPER,
)
from teleop.robot_control.robot_hand_hx5_replay import AIWorkerHX5ReplayPublisher


class _JointState:
    def __init__(self, names, positions):
        self.name = list(names)
        self.position = list(positions)


class _FakeTransport:
    def __init__(self, topic_names, joint_state_callback=None):
        self.topic_names = dict(topic_names)
        self.callback = joint_state_callback
        self.calls = []
        self.closed = False

    def publish(self, key, joint_names, positions, duration):
        self.calls.append(
            {
                "key": key,
                "joint_names": tuple(joint_names),
                "positions": np.asarray(positions, dtype=np.float64).copy(),
                "duration": float(duration),
            }
        )

    def emit(self, names, positions):
        self.callback(_JointState(names, positions))

    def close(self):
        self.closed = True


class AIWorkerHX5ReplayPublisherTest(unittest.TestCase):
    def setUp(self):
        self.transport = None

        def factory(topic_names, joint_state_callback=None):
            self.transport = _FakeTransport(topic_names, joint_state_callback)
            return self.transport

        self.publisher = AIWorkerHX5ReplayPublisher(transport_factory=factory)

    def tearDown(self):
        self.publisher.close()

    def test_uses_existing_topics_and_preserves_left_right_order(self):
        left = (LEFT_LOWER + LEFT_UPPER) / 2.0
        right = (RIGHT_LOWER + RIGHT_UPPER) / 2.0

        sent = self.publisher.write(left, right)

        self.assertEqual(
            self.transport.topic_names,
            {
                "left": "/leader/joint_trajectory_command_broadcaster_left_hand/joint_trajectory",
                "right": "/leader/joint_trajectory_command_broadcaster_right_hand/joint_trajectory",
            },
        )
        self.assertEqual([call["key"] for call in self.transport.calls], ["left", "right"])
        self.assertEqual(self.transport.calls[0]["joint_names"], LEFT_JOINT_NAMES)
        self.assertEqual(self.transport.calls[1]["joint_names"], RIGHT_JOINT_NAMES)
        np.testing.assert_array_equal(self.transport.calls[0]["positions"], left)
        np.testing.assert_array_equal(self.transport.calls[1]["positions"], right)
        np.testing.assert_array_equal(sent, np.concatenate((left, right)))
        np.testing.assert_array_equal(
            self.publisher.get_last_commanded_dual_hand_q(), sent
        )

    def test_waits_until_all_finite_joint_positions_have_been_seen(self):
        left = (LEFT_LOWER + LEFT_UPPER) / 2.0
        right = (RIGHT_LOWER + RIGHT_UPPER) / 2.0
        self.transport.emit(LEFT_JOINT_NAMES, left)
        self.assertFalse(self.publisher.wait_for_joint_state(timeout=0.0))

        invalid_right = right.copy()
        invalid_right[-1] = np.nan
        self.transport.emit(RIGHT_JOINT_NAMES, invalid_right)
        self.assertFalse(self.publisher.wait_for_joint_state(timeout=0.0))

        self.transport.emit([RIGHT_JOINT_NAMES[-1]], [right[-1]])
        self.assertTrue(self.publisher.wait_for_joint_state(timeout=0.0))
        np.testing.assert_array_equal(
            self.publisher.get_current_dual_hand_q(), np.concatenate((left, right))
        )

    def test_invalid_right_target_is_rejected_before_left_publish(self):
        left = np.zeros(20)
        invalid_targets = [
            np.zeros(19),
            np.concatenate(([np.nan], np.zeros(19))),
            RIGHT_LOWER - 0.01,
            RIGHT_UPPER + 0.01,
        ]
        for right in invalid_targets:
            with self.subTest(right_shape=right.shape, right_first=right[0]):
                self.transport.calls.clear()
                with self.assertRaises(ValueError):
                    self.publisher.write(left, right)
                self.assertEqual(self.transport.calls, [])

    def test_hold_uses_last_command_until_state_is_complete_then_measured(self):
        first_left = (LEFT_LOWER + LEFT_UPPER) / 3.0
        first_right = (RIGHT_LOWER + RIGHT_UPPER) / 3.0
        self.publisher.write(first_left, first_right)
        self.transport.emit(LEFT_JOINT_NAMES, np.zeros(20))

        held_before_ready = self.publisher.hold_current()
        np.testing.assert_array_equal(
            held_before_ready, np.concatenate((first_left, first_right))
        )

        measured_left = (LEFT_LOWER + LEFT_UPPER) / 2.0
        measured_right = (RIGHT_LOWER + RIGHT_UPPER) / 2.0
        self.transport.emit(LEFT_JOINT_NAMES, measured_left)
        self.transport.emit(RIGHT_JOINT_NAMES, measured_right)
        held_after_ready = self.publisher.hold_current()
        np.testing.assert_array_equal(
            held_after_ready, np.concatenate((measured_left, measured_right))
        )

    def test_returned_arrays_are_copies_and_close_is_idempotent(self):
        returned = self.publisher.get_current_dual_hand_q()
        returned[0] = 123.0
        self.assertEqual(self.publisher.get_current_dual_hand_q()[0], 0.0)

        self.publisher.close()
        self.publisher.close()
        self.assertTrue(self.transport.closed)
        with self.assertRaises(RuntimeError):
            self.publisher.write(np.zeros(20), np.zeros(20))


if __name__ == "__main__":
    unittest.main()
