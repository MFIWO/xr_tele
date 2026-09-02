"""Per-run communication / action / state logger for teleoperation.

Writes one JSONL file per teleop run under <log_dir>, capturing every control
cycle: robot arm state (q/dq), arm action commands (sol_q/tauff), hand
state/action arrays, DDS health flags, and motor temperatures at a slower
interval. Read-side only; never publishes anything to the robot.

Record types (field "type"):
  meta   - one line at open: run arguments, robot, host, start time
  cycle  - per control loop: state + action + dds health
  temps  - every temperature_interval seconds: all motor temperatures
  event  - explicit markers (start/stop/standby/save...) via log_event()
"""
import json
import os
import time
import datetime

import logging_mp

logger_mp = logging_mp.getLogger(__name__)


def _to_list(values):
    if values is None:
        return None
    try:
        return [round(float(v), 5) for v in values]
    except TypeError:
        return None


class CommStateLogger:
    def __init__(self, log_dir, meta=None, arm_ctrl=None,
                 temperature_interval=5.0, flush_interval=1.0):
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"comm_state_{stamp}.jsonl")
        self._fh = open(self.path, "a", buffering=1)
        self._arm_ctrl = arm_ctrl
        self._temperature_interval = max(0.0, float(temperature_interval))
        self._flush_interval = max(0.0, float(flush_interval))
        self._last_temp_ts = 0.0
        self._last_flush_ts = time.monotonic()
        self._closed = False
        header = {"type": "meta", "ts": time.time(), "start": stamp}
        if meta:
            header.update(meta)
        self._write(header)
        logger_mp.info("[comm log] writing communication/state log to %s", self.path)

    def _write(self, record):
        if self._closed:
            return
        try:
            self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            logger_mp.warning("[comm log] write failed: %s", exc)

    def _dds_health(self):
        arm = self._arm_ctrl
        if arm is None:
            return None
        return {
            "lowstate_stale": bool(getattr(arm, "_dds_lowstate_stale", False)),
            "lowcmd_failed": bool(getattr(arm, "_dds_lowcmd_failed", False)),
            "lowcmd_fail_count": int(getattr(arm, "_dds_lowcmd_failure_count", 0) or 0),
            "write_ok": arm.get_last_write_ok() if hasattr(arm, "get_last_write_ok") else None,
        }

    def _maybe_log_temperatures(self, now):
        if self._temperature_interval <= 0.0 or self._arm_ctrl is None:
            return
        if now - self._last_temp_ts < self._temperature_interval:
            return
        self._last_temp_ts = now
        try:
            temps = self._arm_ctrl.get_current_motor_temperatures()
        except Exception as exc:
            logger_mp.debug("[comm log] temperature read failed: %s", exc)
            return
        if temps:
            self._write({"type": "temps", "ts": now, "motor_temps_c": [
                list(t) if isinstance(t, (list, tuple)) else t for t in temps
            ]})

    def log_cycle(self, loop_count, arm_q=None, arm_dq=None,
                  arm_action_q=None, arm_action_tauff=None,
                  hand_state=None, hand_action=None, mode=None, extras=None):
        """Record one control cycle. Call once per main-loop iteration."""
        if self._closed:
            return
        now = time.time()
        record = {
            "type": "cycle",
            "ts": now,
            "loop": int(loop_count),
            "arm_q": _to_list(arm_q),
            "arm_dq": _to_list(arm_dq),
            "arm_action_q": _to_list(arm_action_q),
            "arm_action_tauff": _to_list(arm_action_tauff),
            "hand_state": _to_list(hand_state),
            "hand_action": _to_list(hand_action),
        }
        if mode is not None:
            record["mode"] = mode
        dds = self._dds_health()
        if dds is not None:
            record["dds"] = dds
        if extras:
            record.update(extras)
        self._write(record)
        self._maybe_log_temperatures(now)
        mono = time.monotonic()
        if mono - self._last_flush_ts >= self._flush_interval:
            self._last_flush_ts = mono
            try:
                self._fh.flush()
            except Exception:
                pass

    def log_event(self, event, **fields):
        record = {"type": "event", "ts": time.time(), "event": str(event)}
        record.update(fields)
        self._write(record)

    def close(self):
        if self._closed:
            return
        self.log_event("close")
        self._closed = True
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass
        logger_mp.info("[comm log] closed %s", self.path)
