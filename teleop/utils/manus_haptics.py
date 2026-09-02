import json
import math
import socket
import time

from teleop.utils.rh5dg2_tactile import (
    BaselineEmaFilter,
    FINGER_ORDER,
    extract_side,
    normalize_packet,
)


MANUS_SIDES = ("left", "right")
ZERO_POWERS = [0.0] * len(FINGER_ORDER)


def _sanitize_powers(values):
    if not isinstance(values, (list, tuple)) or len(values) != len(FINGER_ORDER):
        raise ValueError(f"Manus haptic powers must contain {len(FINGER_ORDER)} values.")
    powers = []
    for value in values:
        power = float(value)
        if not math.isfinite(power):
            raise ValueError("Manus haptic powers must be finite.")
        powers.append(max(0.0, min(1.0, power)))
    return powers


class ManusNormalForceMapper:
    """Map RH5DG2 fingertip normal force to MANUS finger vibration power."""

    def __init__(
        self,
        baseline_seconds=0.0,
        ema_alpha=0.25,
        deadband=1.0,
        normal_max=2500.0,
        gamma=1.0,
    ):
        if not 0.0 < float(ema_alpha) <= 1.0:
            raise ValueError("ema_alpha must be greater than 0 and at most 1.")
        if float(baseline_seconds) < 0.0:
            raise ValueError("baseline_seconds must be zero or greater.")
        if float(deadband) < 0.0:
            raise ValueError("deadband must be zero or greater.")
        if float(normal_max) <= 0.0:
            raise ValueError("normal_max must be greater than zero.")
        if float(gamma) <= 0.0:
            raise ValueError("gamma must be greater than zero.")

        self.normal_max = float(normal_max)
        self.gamma = float(gamma)
        self.filters = {
            side: BaselineEmaFilter(
                alpha=ema_alpha,
                baseline_seconds=baseline_seconds,
                deadband=deadband,
            )
            for side in MANUS_SIDES
        }
        self.last_debug = {
            side: {
                "raw_normal": list(ZERO_POWERS),
                "filtered_normal": list(ZERO_POWERS),
            }
            for side in MANUS_SIDES
        }

    def reset(self, side=None):
        sides = MANUS_SIDES if side is None else (side,)
        for glove_side in sides:
            if glove_side not in self.filters:
                raise ValueError(f"Unsupported MANUS side: {glove_side}")
            self.filters[glove_side].reset()

    def update_side(self, packet, side):
        if side not in self.filters:
            raise ValueError(f"Unsupported MANUS side: {side}")
        tactile_side = f"{side}_ee"
        if not extract_side(packet, tactile_side):
            return None

        raw = normalize_packet(packet, tactile_side)
        filtered = self.filters[side].update(raw)
        raw_normal = [
            float(raw["fingers"][finger][0])
            for finger in FINGER_ORDER
        ]
        filtered_normal = [
            float(filtered.get(f"finger.{finger}.normal", 0.0))
            for finger in FINGER_ORDER
        ]
        self.last_debug[side] = {
            "raw_normal": raw_normal,
            "filtered_normal": filtered_normal,
        }
        return [
            max(
                0.0,
                min(
                    1.0,
                    filtered.get(f"finger.{finger}.normal", 0.0) / self.normal_max,
                ),
            )
            ** self.gamma
            for finger in FINGER_ORDER
        ]

    def debug_snapshot(self):
        return {
            side: {
                key: list(values)
                for key, values in self.last_debug[side].items()
            }
            for side in MANUS_SIDES
        }

    def update(self, packet, stale_sides=()):
        stale = set(stale_sides or ())
        powers = {}
        for side in MANUS_SIDES:
            if side in stale or f"{side}_ee" in stale:
                powers[side] = list(ZERO_POWERS)
                continue
            mapped = self.update_side(packet, side)
            powers[side] = list(ZERO_POWERS) if mapped is None else mapped
        return powers


class ManusHapticUDPSender:
    """Rate-limited heartbeat sender for the ROS-side MANUS haptic bridge."""

    def __init__(self, host, port, send_hz=20.0, source="xr_teleoperate"):
        if not str(host):
            raise ValueError("host must not be empty.")
        if not 1 <= int(port) <= 65535:
            raise ValueError("port must be between 1 and 65535.")
        if float(send_hz) <= 0.0:
            raise ValueError("send_hz must be greater than zero.")
        self.target = (str(host), int(port))
        self.send_period = 1.0 / float(send_hz)
        self.source = str(source)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sequence = 0
        self.last_send_time = 0.0
        self.last_powers = {side: list(ZERO_POWERS) for side in MANUS_SIDES}
        self.zero_sent = False
        self.closed = False

    def send(self, powers, force=False):
        if self.closed:
            return False
        now = time.monotonic()
        if not force and now - self.last_send_time < self.send_period:
            return False

        normalized = {
            side: _sanitize_powers(powers.get(side, ZERO_POWERS))
            for side in MANUS_SIDES
        }
        packet = {
            "source": self.source,
            "type": "manus_haptics",
            "timestamp": time.time(),
            "sequence": self.sequence,
            "left": normalized["left"],
            "right": normalized["right"],
        }
        raw = json.dumps(packet, separators=(",", ":")).encode("utf-8")
        self.socket.sendto(raw, self.target)
        self.sequence += 1
        self.last_send_time = now
        self.last_powers = normalized
        self.zero_sent = all(
            all(power == 0.0 for power in side_powers)
            for side_powers in normalized.values()
        )
        return True

    def stop(self, force=False):
        if self.closed or (self.zero_sent and not force):
            return False
        return self.send(
            {side: list(ZERO_POWERS) for side in MANUS_SIDES},
            force=True,
        )

    def close(self):
        if self.closed:
            return
        try:
            self.stop(force=True)
        finally:
            self.closed = True
            self.socket.close()
