import socket
import time
import unittest

from teleop.ai_worker_pedal_teleop import (
    KEY_O, KEY_P, KEY_U, PedalAuxState, motion_is_allowed,
    TeleopSafetyReceiver,
)
from teleop.robot_control.robotis_ai_worker_lift import AIWorkerLiftController
from teleop.teleop_hand_and_arm import (
    PedalEstopReceiver,
    _safe_enter_hand_tracking_standby,
)


class FakeTransport:
    def __init__(self):
        self.commands = []

    def publish(self, key, joints, positions, duration):
        self.commands.append((key, joints, positions, duration))


def make_lift(position=-0.25):
    lift = AIWorkerLiftController.__new__(AIWorkerLiftController)
    lift._measured_position = position
    lift._command = None
    lift.command_duration = 0.08
    lift.transport = FakeTransport()
    return lift


class AIWorkerPedalControlsTest(unittest.TestCase):
    def test_aux_keys_drive_lift_and_toggle_estop_once_per_key_down(self):
        state = PedalAuxState()
        state.update("pedal", KEY_O, 1)
        self.assertEqual(state.lift_direction, 1)
        state.update("pedal", KEY_P, 1)
        self.assertEqual(state.lift_direction, 0)
        state.update("pedal", KEY_O, 0)
        self.assertEqual(state.lift_direction, -1)
        self.assertTrue(state.update("pedal", KEY_U, 1))
        self.assertTrue(state.estop)
        self.assertFalse(state.update("pedal", KEY_U, 2))
        self.assertTrue(state.estop)
        self.assertFalse(motion_is_allowed(state.estop, True))
        # A tracking reconnect/healthy heartbeat cannot implicitly clear U.
        self.assertTrue(state.estop)
        self.assertFalse(motion_is_allowed(state.estop, True))
        state.update("pedal", KEY_U, 0)
        state.update("pedal", KEY_U, 1)
        self.assertTrue(motion_is_allowed(state.estop, True))
        self.assertFalse(motion_is_allowed(state.estop, False))

    def test_lift_nudge_and_hold_are_bounded(self):
        lift = make_lift()
        self.assertAlmostEqual(lift.nudge(1, 0.05), -0.20)
        self.assertAlmostEqual(lift.nudge(-1, 1.0), lift.LOWER)
        self.assertAlmostEqual(lift.hold(), lift.LOWER)
        self.assertEqual(lift.transport.commands[-1][2], (lift.LOWER,))
        with self.assertRaises(ValueError):
            lift.nudge(1, float("nan"))
        with self.assertRaises(ValueError):
            lift.nudge(1, -0.01)

    def test_udp_estop_receiver_latches_latest_state(self):
        receiver = PedalEstopReceiver("127.0.0.1", 0)
        port = receiver._socket.getsockname()[1]
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sender.sendto(b"ESTOP 1", ("127.0.0.1", port))
            deadline = time.monotonic() + 1.0
            while not receiver.active and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(receiver.active)
            sender.sendto(b"ESTOP 0", ("127.0.0.1", port))
            deadline = time.monotonic() + 1.0
            while receiver.active and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(receiver.active)
        finally:
            sender.close()
            receiver.close()

    def test_tracking_heartbeat_fails_closed_and_expires(self):
        receiver = TeleopSafetyReceiver("127.0.0.1", 0, timeout=0.05)
        port = receiver.sock.getsockname()[1]
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.assertFalse(receiver.allowed)
            sender.sendto(b"MOTION 1", ("127.0.0.1", port))
            deadline = time.monotonic() + 1.0
            while not receiver.allowed and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertTrue(receiver.allowed)
            time.sleep(0.07)
            self.assertFalse(receiver.allowed)
        finally:
            sender.close()
            receiver.close()

    def test_hx5_tracking_loss_holds_last_command_without_opening_hand(self):
        class FakeHandController:
            def __init__(self):
                self._enabled = True
                self.open_calls = 0

            def enter_standby_open(self):
                self.open_calls += 1

        hand = FakeHandController()
        self.assertTrue(_safe_enter_hand_tracking_standby(hand, "hx5_d20"))
        self.assertFalse(hand._enabled)
        self.assertEqual(hand.open_calls, 0)

        self.assertTrue(_safe_enter_hand_tracking_standby(hand, "rh56f1"))
        self.assertEqual(hand.open_calls, 1)


if __name__ == "__main__":
    unittest.main()
