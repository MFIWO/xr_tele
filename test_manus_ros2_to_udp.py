import pytest

from manus_ros2_to_udp import (
    _assign_haptic_route,
    _haptic_topic_for_glove_topic,
    _parse_haptic_packet,
)


def test_haptic_topic_is_derived_from_glove_topic():
    assert _haptic_topic_for_glove_topic("manus_glove_0") == "manus_glove_0/vibration_cmd"
    assert _haptic_topic_for_glove_topic("/manus_glove_1/") == "/manus_glove_1/vibration_cmd"


def test_haptic_routes_follow_reported_side_instead_of_glove_index():
    routes = {}

    assert _assign_haptic_route(routes, "Left", "manus_glove_0/vibration_cmd")
    assert _assign_haptic_route(routes, "RIGHT", "manus_glove_1/vibration_cmd")

    assert routes == {
        "left": "manus_glove_0/vibration_cmd",
        "right": "manus_glove_1/vibration_cmd",
    }


def test_haptic_route_remains_one_to_one_when_side_changes():
    routes = {"right": "manus_glove_0/vibration_cmd"}

    assert _assign_haptic_route(routes, "left", "manus_glove_0/vibration_cmd")

    assert routes == {"left": "manus_glove_0/vibration_cmd"}


def test_parse_haptic_packet_clamps_both_hands():
    parsed = _parse_haptic_packet({
        "type": "manus_haptics",
        "left": [-1.0, 0.1, 0.5, 1.0, 2.0],
        "right": [0.0, 0.2, 0.4, 0.6, 0.8],
    })

    assert parsed["left"] == pytest.approx([0.0, 0.1, 0.5, 1.0, 1.0])
    assert parsed["right"] == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8])


@pytest.mark.parametrize(
    "packet",
    [
        {},
        {"type": "other", "left": [0.0] * 5, "right": [0.0] * 5},
        {"type": "manus_haptics", "left": [0.0] * 4, "right": [0.0] * 5},
        {"type": "manus_haptics", "left": [0.0] * 5, "right": [float("nan")] * 5},
    ],
)
def test_parse_haptic_packet_rejects_invalid_payload(packet):
    with pytest.raises(ValueError):
        _parse_haptic_packet(packet)
