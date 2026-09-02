"""DDS-free command sinks for previewing AI Worker arm and HX5 targets.

These classes deliberately implement the small controller surface consumed by
``teleop_hand_and_arm.py`` without constructing a transport.  They are useful
for exercising the real arm IK and hand retargeting pipeline on a development
host where publishing a robot command must be impossible.
"""

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
import logging
import threading
import time

import numpy as np

from teleop.robot_control.robot_hand_hx5_d20 import (
    HX5D20DexPilotRetargeter,
    HX5D20Retargeter,
)
from teleop.robot_control.robotis_ai_worker import (
    AI_WORKER_ARM_LOWER,
    AI_WORKER_ARM_UPPER,
    AI_WORKER_SH5_HOME_Q,
)


_LOGGER = logging.getLogger(__name__)


class _CommandMeter:
    """Keep a short monotonic-time window for a useful command-rate estimate."""

    def __init__(self, window_seconds=2.0):
        self.window_seconds = max(0.1, float(window_seconds))
        self.timestamps = deque()
        self.total = 0

    def tick(self, now):
        now = float(now)
        self.total += 1
        self.timestamps.append(now)
        cutoff = now - self.window_seconds
        while len(self.timestamps) > 2 and self.timestamps[0] < cutoff:
            self.timestamps.popleft()

    @property
    def rate_hz(self):
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0.0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed


class _CommandReporter:
    def __init__(self, label, report_interval=1.0, reporter=None):
        self.label = str(label)
        self.report_interval = max(0.0, float(report_interval))
        self.reporter = reporter if reporter is not None else _LOGGER.info
        self.last_report_time = float("-inf")
        self.last_report = None

    def maybe_report(self, target, meter, now, force=False):
        if not force and (
            self.report_interval <= 0.0
            or now - self.last_report_time < self.report_interval
        ):
            return None
        target_text = np.array2string(
            np.asarray(target, dtype=np.float64),
            precision=3,
            suppress_small=True,
            separator=",",
            max_line_width=240,
        )
        message = (
            f"[command sink] {self.label} count={meter.total} "
            f"rate_hz={meter.rate_hz:.2f} target={target_text}"
        )
        self.reporter(message)
        self.last_report_time = float(now)
        self.last_report = message
        return message


def _lock_for(shared_value):
    get_lock = getattr(shared_value, "get_lock", None)
    if callable(get_lock):
        return get_lock()
    return nullcontext()


def _read_landmarks(shared_value):
    with _lock_for(shared_value):
        return np.asarray(shared_value[:], dtype=np.float64).reshape(25, 3).copy()


def _write_shared(shared_value, values):
    shared_value[:] = np.asarray(values, dtype=np.float64).reshape(-1)


class AIWorkerArmCommandSink:
    """A perfect-follow, velocity-limited stand-in for the AI Worker arm DDS controller."""

    command_topic_description = "DDS-FREE AI Worker command sink (no robot publisher)"

    def __init__(
        self,
        command_duration=0.08,
        node_name="xr_tele_ai_worker_command_sink",
        home_q=None,
        ready_q=None,
        arm_velocity_limit=3.0,
        report_interval=1.0,
        reporter=None,
    ):
        del node_name
        self.command_duration = max(0.02, float(command_duration))
        self.home_q = (
            AI_WORKER_SH5_HOME_Q.copy()
            if home_q is None
            else np.asarray(home_q, dtype=np.float64).reshape(14).copy()
        )
        self.ready_q = (
            self.home_q.copy()
            if ready_q is None
            else np.asarray(ready_q, dtype=np.float64).reshape(14).copy()
        )
        self.arm_velocity_limit = max(0.01, float(arm_velocity_limit))
        self._q = self.home_q.copy()
        self._dq = np.zeros(14, dtype=np.float64)
        self._last_command = self.home_q.copy()
        self._last_command_time = time.monotonic()
        self._last_write_ok = False
        self._closed = False
        self._lock = threading.Lock()
        self._meter = _CommandMeter()
        self._reporter = _CommandReporter(
            "ai_worker_arms",
            report_interval=report_interval,
            reporter=reporter,
        )

    @property
    def command_count(self):
        with self._lock:
            return self._meter.total

    @property
    def command_rate_hz(self):
        with self._lock:
            return self._meter.rate_hz

    def wait_for_joint_state(self, timeout=0.0):
        del timeout
        return not self._closed

    def sync_arm_command_to_measured(self):
        with self._lock:
            self._last_command = self._q.copy()
            self._last_command_time = time.monotonic()
            return self._last_command.copy()

    def get_current_dual_arm_q(self):
        with self._lock:
            return self._q.copy()

    def get_current_dual_arm_dq(self):
        with self._lock:
            return self._dq.copy()

    def get_last_commanded_dual_arm_q(self):
        with self._lock:
            return self._last_command.copy()

    def get_current_motor_q(self):
        return self.get_current_dual_arm_q()

    def ctrl_dual_arm(self, q, tau=None):
        del tau
        target = np.asarray(q, dtype=np.float64).reshape(14)
        if not np.all(np.isfinite(target)):
            raise ValueError("AI Worker command-sink target must be finite.")
        target = np.clip(target, AI_WORKER_ARM_LOWER, AI_WORKER_ARM_UPPER)
        now = time.monotonic()
        with self._lock:
            if self._closed:
                return
            dt = max(now - self._last_command_time, 1.0 / 100.0)
            max_delta = self.arm_velocity_limit * dt
            limited = self._last_command + np.clip(
                target - self._last_command,
                -max_delta,
                max_delta,
            )
            self._dq = (limited - self._q) / dt
            self._q = limited.copy()
            self._last_command = limited.copy()
            self._last_command_time = now
            self._last_write_ok = True
            self._meter.tick(now)
            report_target = limited.copy()
            self._reporter.maybe_report(report_target, self._meter, now)

    def ctrl_dual_arm_smooth_to(self, q, duration, num_points=100):
        del num_points
        target = np.asarray(q, dtype=np.float64).reshape(14)
        if not np.all(np.isfinite(target)):
            raise ValueError("AI Worker command-sink trajectory target must be finite.")
        target = np.clip(target, AI_WORKER_ARM_LOWER, AI_WORKER_ARM_UPPER)
        with self._lock:
            start = self._q.copy()
        requested_duration = max(0.1, float(duration))
        minimum_duration = (
            1.875 * float(np.max(np.abs(target - start))) / self.arm_velocity_limit
        )
        trajectory_duration = max(requested_duration, minimum_duration)
        now = time.monotonic()
        with self._lock:
            if self._closed:
                return trajectory_duration
            self._q = target.copy()
            self._dq = np.zeros(14, dtype=np.float64)
            self._last_command = target.copy()
            self._last_command_time = now
            self._last_write_ok = True
            self._meter.tick(now)
            self._reporter.maybe_report(target, self._meter, now, force=True)
        return trajectory_duration

    def speed_gradual_max(self):
        return None

    def ctrl_dual_arm_go_home(self):
        self._capture_unlimited(self.home_q)

    def ctrl_dual_arm_go_ready(self):
        self._capture_unlimited(self.ready_q)

    def _capture_unlimited(self, target):
        now = time.monotonic()
        target = np.asarray(target, dtype=np.float64).reshape(14).copy()
        with self._lock:
            if self._closed:
                return
            self._q = target
            self._dq = np.zeros(14, dtype=np.float64)
            self._last_command = target.copy()
            self._last_command_time = now
            self._last_write_ok = True
            self._meter.tick(now)
            self._reporter.maybe_report(target, self._meter, now, force=True)

    def get_last_write_ok(self):
        with self._lock:
            return self._last_write_ok

    def print_report(self):
        now = time.monotonic()
        with self._lock:
            return self._reporter.maybe_report(
                self._last_command,
                self._meter,
                now,
                force=True,
            )

    def stop(self):
        self.close()

    def close(self):
        with self._lock:
            self._closed = True


class HX5D20CommandSink:
    """Run the existing HX5 retargeter and capture targets without DDS."""

    command_topic_description = "DDS-FREE HX5-D20 command sink (no robot publisher)"

    def __init__(
        self,
        left_hand_pos_array,
        right_hand_pos_array,
        dual_hand_data_lock,
        dual_hand_state_array,
        dual_hand_action_array,
        fps=50.0,
        smoothing_alpha=1.0,
        command_duration=0.08,
        thumb_yaw_gain=1.0,
        thumb_yaw_max=1.2,
        thumb_pitch_max=0.7,
        retarget_mode="geometric",
        left_hand_scale=1.0,
        right_hand_scale=1.0,
        report_interval=1.0,
        reporter=None,
        start_thread=True,
    ):
        self.command_duration = max(0.02, float(command_duration))
        self.left_input = left_hand_pos_array
        self.right_input = right_hand_pos_array
        self.data_lock = dual_hand_data_lock
        self.state_array = dual_hand_state_array
        self.action_array = dual_hand_action_array
        if retarget_mode == "dexpilot":
            self.left_retargeter = HX5D20DexPilotRetargeter(
                "left",
                smoothing_alpha=smoothing_alpha,
                hand_scale=left_hand_scale,
            )
            self.right_retargeter = HX5D20DexPilotRetargeter(
                "right",
                smoothing_alpha=smoothing_alpha,
                hand_scale=right_hand_scale,
            )
        elif retarget_mode == "geometric":
            self.left_retargeter = HX5D20Retargeter(
                "left",
                smoothing_alpha,
                thumb_yaw_gain,
                thumb_yaw_max,
                thumb_pitch_max,
            )
            self.right_retargeter = HX5D20Retargeter(
                "right",
                smoothing_alpha,
                thumb_yaw_gain,
                thumb_yaw_max,
                thumb_pitch_max,
            )
        else:
            raise ValueError("retarget_mode must be 'dexpilot' or 'geometric'")
        self.retarget_mode = retarget_mode
        self.period = 1.0 / max(1.0, float(fps))
        self._enabled = True
        self._running = True
        self._stopped = False
        self._update_lock = threading.Lock()
        self._meter = _CommandMeter()
        self._reporter = _CommandReporter(
            "hx5_d20",
            report_interval=report_interval,
            reporter=reporter,
        )
        self._last_left_target = np.zeros(20, dtype=np.float64)
        self._last_right_target = np.zeros(20, dtype=np.float64)
        self._thread = None
        if start_thread:
            self._thread = threading.Thread(
                target=self._run,
                name="hx5-d20-command-sink",
                daemon=True,
            )
            self._thread.start()

    @property
    def command_count(self):
        with self._update_lock:
            return self._meter.total

    @property
    def command_rate_hz(self):
        with self._update_lock:
            return self._meter.rate_hz

    def get_last_targets(self):
        with self._update_lock:
            return self._last_left_target.copy(), self._last_right_target.copy()

    def update_target(self, left_landmarks=None, right_landmarks=None):
        """Retarget one sample immediately and return copies of both 20-DoF targets."""
        if (left_landmarks is None) != (right_landmarks is None):
            raise ValueError("Both left and right landmarks must be supplied together.")
        if left_landmarks is None:
            left_points = _read_landmarks(self.left_input)
            right_points = _read_landmarks(self.right_input)
        else:
            left_points = np.asarray(left_landmarks, dtype=np.float64).reshape(25, 3)
            right_points = np.asarray(right_landmarks, dtype=np.float64).reshape(25, 3)

        left = self.left_retargeter.retarget(left_points)
        right = self.right_retargeter.retarget(right_points)
        self._capture_both(left, right)
        return left.copy(), right.copy()

    def _capture_both(self, left, right):
        left = np.asarray(left, dtype=np.float64).reshape(20)
        right = np.asarray(right, dtype=np.float64).reshape(20)
        combined = np.concatenate((left, right))
        now = time.monotonic()
        with self._update_lock:
            self._last_left_target = left.copy()
            self._last_right_target = right.copy()
            with self.data_lock:
                _write_shared(self.state_array, combined)
                _write_shared(self.action_array, combined)
            self._meter.tick(now)
            self._reporter.maybe_report(combined, self._meter, now)

    def _run(self):
        next_tick = time.monotonic()
        while self._running:
            if self._enabled:
                try:
                    self.update_target()
                except ValueError:
                    # Empty/stale landmarks are expected before tracking starts.
                    pass
                except Exception:
                    _LOGGER.exception("HX5-D20 command-sink retarget loop failed")
            next_tick += self.period
            time.sleep(max(0.0, next_tick - time.monotonic()))
            if next_tick < time.monotonic() - self.period:
                next_tick = time.monotonic()

    def enter_standby_open(self):
        self._enabled = False
        self._capture_both(np.zeros(20), np.zeros(20))

    def enter_auto(self):
        self._enabled = True

    def restore_initial_pose(self):
        if not self._stopped:
            self._capture_both(np.zeros(20), np.zeros(20))

    def print_report(self):
        now = time.monotonic()
        with self._update_lock:
            target = np.concatenate(
                (self._last_left_target, self._last_right_target)
            )
            return self._reporter.maybe_report(
                target,
                self._meter,
                now,
                force=True,
            )

    def stop(self):
        if self._stopped:
            return
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._capture_both(np.zeros(20), np.zeros(20))
        self._stopped = True

    def close(self):
        self.stop()


@dataclass(frozen=True)
class AIWorkerPreviewTargets:
    arm: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray


def preview_teleop_step(
    arm_ik,
    arm_sink,
    hand_sink,
    left_wrist_pose,
    right_wrist_pose,
    left_hand_landmarks,
    right_hand_landmarks,
):
    """Run one real IK/retarget step while routing every output to memory only."""
    current_q = arm_sink.get_current_dual_arm_q()
    current_dq = arm_sink.get_current_dual_arm_dq()
    arm_target, arm_feedforward = arm_ik.solve_ik(
        left_wrist_pose,
        right_wrist_pose,
        current_q,
        current_dq,
    )
    arm_sink.ctrl_dual_arm(arm_target, arm_feedforward)
    left_hand_target, right_hand_target = hand_sink.update_target(
        left_hand_landmarks,
        right_hand_landmarks,
    )
    return AIWorkerPreviewTargets(
        arm=arm_sink.get_last_commanded_dual_arm_q(),
        left_hand=left_hand_target,
        right_hand=right_hand_target,
    )
