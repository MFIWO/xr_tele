import math
import threading
import time


FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")
PALM_ORDER = ("right", "middle", "left")
FINGER_ALIASES = {
    "thumb": "thumb",
    "index": "index",
    "middle": "middle",
    "ring": "ring",
    "little": "little",
    "pinky": "little",
    "small": "little",
}


def _as_float(value, default=0.0):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _normalize_finger_values(value):
    if isinstance(value, dict):
        return [
            _as_float(value.get("normal", value.get("normal_force", 0.0))),
            _as_float(value.get("tangent", value.get("tangential", value.get("tangential_force", 0.0)))),
            _as_float(value.get("direction", value.get("tangential_direction", 65535.0)), 65535.0),
            _as_float(value.get("proximity", 0.0)),
        ]
    values = [_as_float(v) for v in _as_list(value)[:4]]
    while len(values) < 4:
        values.append(65535.0 if len(values) == 2 else 0.0)
    return values


def _normalize_palm_values(value):
    if isinstance(value, dict):
        zones = []
        for zone in PALM_ORDER:
            raw = value.get(zone, value.get(f"palm_{zone}", []))
            if isinstance(raw, dict):
                zones.extend([
                    _as_float(raw.get("normal", raw.get("normal_force", 0.0))),
                    _as_float(raw.get("tangent", raw.get("tangential", raw.get("tangential_force", 0.0)))),
                    _as_float(raw.get("direction", raw.get("tangential_direction", 65535.0)), 65535.0),
                ])
            else:
                part = [_as_float(v) for v in _as_list(raw)[:3]]
                while len(part) < 3:
                    part.append(65535.0 if len(part) == 2 else 0.0)
                zones.extend(part)
        return zones
    values = [_as_float(v) for v in _as_list(value)[:9]]
    while len(values) < 9:
        values.append(65535.0 if len(values) % 3 == 2 else 0.0)
    return values


def extract_side(packet, side):
    if not isinstance(packet, dict):
        return {}
    if isinstance(packet.get(side), dict):
        return packet[side]
    tactiles = packet.get("tactiles")
    if isinstance(tactiles, dict) and isinstance(tactiles.get(side), dict):
        return tactiles[side]
    if "fingers" in packet or "palm" in packet:
        return packet
    return {}


def normalize_packet(packet, side="right_ee"):
    data = extract_side(packet, side)
    fingers_in = data.get("fingers", {}) if isinstance(data, dict) else {}
    palm_in = data.get("palm", []) if isinstance(data, dict) else []

    fingers = {name: [0.0, 0.0, 65535.0, 0.0] for name in FINGER_ORDER}
    if isinstance(fingers_in, dict):
        for raw_name, raw_values in fingers_in.items():
            name = FINGER_ALIASES.get(str(raw_name).lower())
            if name:
                fingers[name] = _normalize_finger_values(raw_values)
    elif isinstance(fingers_in, (list, tuple)):
        for name, raw_values in zip(FINGER_ORDER, fingers_in):
            fingers[name] = _normalize_finger_values(raw_values)

    return {
        "timestamp": _as_float(packet.get("timestamp", time.time())) if isinstance(packet, dict) else time.time(),
        "side": side,
        "fingers": fingers,
        "palm": _normalize_palm_values(palm_in),
    }


class BaselineEmaFilter:
    def __init__(self, alpha=0.25, baseline_seconds=1.0, deadband=1.0):
        self.alpha = float(alpha)
        self.baseline_seconds = float(baseline_seconds)
        self.deadband = float(deadband)
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with getattr(self, "lock", threading.Lock()):
            self.baseline_start = time.monotonic()
            self.baseline_count = 0
            self.baseline_sum = {}
            self.baseline = {}
            self.ema = {}
            self.ready = self.baseline_seconds <= 0.0

    def update(self, raw):
        channels = flatten_force_channels(raw)
        now = time.monotonic()
        with self.lock:
            if not self.ready:
                for key, value in channels.items():
                    self.baseline_sum[key] = self.baseline_sum.get(key, 0.0) + value
                self.baseline_count += 1
                if now - self.baseline_start >= self.baseline_seconds:
                    count = max(1, self.baseline_count)
                    self.baseline = {key: value / count for key, value in self.baseline_sum.items()}
                    self.ema = {key: 0.0 for key in channels}
                    self.ready = True
                return {key: 0.0 for key in channels}

            filtered = {}
            for key, value in channels.items():
                centered = value - self.baseline.get(key, 0.0)
                if centered < self.deadband:
                    centered = 0.0
                previous = self.ema.get(key, 0.0)
                smoothed = (1.0 - self.alpha) * previous + self.alpha * centered
                self.ema[key] = smoothed
                filtered[key] = smoothed
            return filtered

    def status(self):
        with self.lock:
            elapsed = time.monotonic() - self.baseline_start
            return {
                "ready": self.ready,
                "samples": self.baseline_count,
                "elapsed": elapsed,
                "target_seconds": self.baseline_seconds,
            }


def flatten_force_channels(raw):
    out = {}
    for finger, values in raw["fingers"].items():
        out[f"finger.{finger}.normal"] = values[0]
        out[f"finger.{finger}.tangent"] = values[1]
        out[f"finger.{finger}.proximity"] = values[3]
    for idx, zone in enumerate(PALM_ORDER):
        base = idx * 3
        out[f"palm.{zone}.normal"] = raw["palm"][base]
        out[f"palm.{zone}.tangent"] = raw["palm"][base + 1]
    return out


class RH5DG2TactileHeatMapper:
    def __init__(
        self,
        side="right_ee",
        baseline_seconds=1.0,
        ema_alpha=0.25,
        deadband=1.0,
        normal_max=800.0,
        tangent_max=800.0,
        proximity_max=65535.0,
        proximity_weight=0.65,
    ):
        self.side = side
        self.normal_max = float(normal_max)
        self.tangent_max = float(tangent_max)
        self.proximity_max = float(proximity_max)
        self.proximity_weight = float(proximity_weight)
        self.filter = BaselineEmaFilter(ema_alpha, baseline_seconds, deadband)

    def reset_baseline(self):
        self.filter.reset()

    def status(self):
        return self.filter.status()

    def update(self, packet):
        raw = normalize_packet(packet, self.side)
        filtered = self.filter.update(raw)
        return build_tactile_view(
            raw,
            filtered,
            normal_max=self.normal_max,
            tangent_max=self.tangent_max,
            proximity_max=self.proximity_max,
            proximity_weight=self.proximity_weight,
        )


def build_tactile_view(raw, filtered, normal_max=800.0, tangent_max=800.0, proximity_max=65535.0, proximity_weight=0.65):
    fingers = []
    for name in FINGER_ORDER:
        values = raw["fingers"][name]
        normal = filtered.get(f"finger.{name}.normal", 0.0)
        tangent = filtered.get(f"finger.{name}.tangent", 0.0)
        proximity = filtered.get(f"finger.{name}.proximity", 0.0)
        heat = max(
            normal / max(float(normal_max), 1.0),
            tangent / max(float(tangent_max), 1.0),
            float(proximity_weight) * proximity / max(float(proximity_max), 1.0),
        )
        fingers.append({
            "name": name,
            "heat": max(0.0, min(1.0, heat)),
            "normal_raw": values[0],
            "tangent_raw": values[1],
            "direction_raw": values[2],
            "direction_valid": 0.0 <= values[2] <= 359.0,
            "proximity_raw": values[3],
            "normal": normal,
            "tangent": tangent,
            "proximity": proximity,
        })

    palm = []
    for idx, zone in enumerate(PALM_ORDER):
        base = idx * 3
        normal = filtered.get(f"palm.{zone}.normal", 0.0)
        tangent = filtered.get(f"palm.{zone}.tangent", 0.0)
        heat = max(
            normal / max(float(normal_max), 1.0),
            tangent / max(float(tangent_max), 1.0),
        )
        direction = raw["palm"][base + 2]
        palm.append({
            "name": zone,
            "heat": max(0.0, min(1.0, heat)),
            "normal_raw": raw["palm"][base],
            "tangent_raw": raw["palm"][base + 1],
            "direction_raw": direction,
            "direction_valid": 0.0 <= direction <= 359.0,
            "normal": normal,
            "tangent": tangent,
        })

    return {
        "timestamp": raw["timestamp"],
        "side": raw["side"],
        "fingers": fingers,
        "palm": palm,
    }
