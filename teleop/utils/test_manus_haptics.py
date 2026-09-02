import json
import socket

import pytest

from teleop.utils.manus_haptics import (
    ManusHapticUDPSender,
    ManusNormalForceMapper,
)


def _packet(left=None, right=None):
    packet = {}
    if left is not None:
        packet["left_ee"] = {"fingers": left, "palm": []}
    if right is not None:
        packet["right_ee"] = {"fingers": right, "palm": []}
    return packet


def test_normal_force_maps_in_manus_finger_order():
    mapper = ManusNormalForceMapper(
        ema_alpha=1.0,
        deadband=0.0,
        normal_max=100.0,
        gamma=1.0,
    )
    packet = _packet(
        left={
            "little": [50, 0, 0, 0],
            "ring": [40, 0, 0, 0],
            "middle": [30, 0, 0, 0],
            "index": [20, 0, 0, 0],
            "thumb": [10, 0, 0, 0],
        }
    )

    powers = mapper.update(packet)
    debug = mapper.debug_snapshot()

    assert powers["left"] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
    assert powers["right"] == [0.0] * 5
    assert debug["left"]["raw_normal"] == pytest.approx([10, 20, 30, 40, 50])
    assert debug["left"]["filtered_normal"] == pytest.approx([10, 20, 30, 40, 50])


def test_normal_force_deadband_gamma_clamp_and_stale():
    mapper = ManusNormalForceMapper(
        ema_alpha=1.0,
        deadband=10.0,
        normal_max=100.0,
        gamma=2.0,
    )
    fingers = {
        "thumb": [5, 0, 0, 0],
        "index": [50, 0, 0, 0],
        "middle": [200, 0, 0, 0],
        "ring": [0, 0, 0, 0],
        "little": [0, 0, 0, 0],
    }

    powers = mapper.update(_packet(left=fingers, right=fingers), stale_sides=("right_ee",))

    assert powers["left"] == pytest.approx([0.0, 0.25, 1.0, 0.0, 0.0])
    assert powers["right"] == [0.0] * 5


def test_udp_sender_emits_both_hands_and_explicit_stop():
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(1.0)
    port = receiver.getsockname()[1]
    sender = ManusHapticUDPSender("127.0.0.1", port, send_hz=1000.0)
    try:
        assert sender.send({"left": [0.1] * 5, "right": [0.2] * 5}, force=True)
        first = json.loads(receiver.recvfrom(4096)[0].decode("utf-8"))
        assert first["type"] == "manus_haptics"
        assert first["left"] == pytest.approx([0.1] * 5)
        assert first["right"] == pytest.approx([0.2] * 5)

        assert sender.stop(force=True)
        stopped = json.loads(receiver.recvfrom(4096)[0].decode("utf-8"))
        assert stopped["left"] == [0.0] * 5
        assert stopped["right"] == [0.0] * 5
    finally:
        sender.close()
        receiver.close()
