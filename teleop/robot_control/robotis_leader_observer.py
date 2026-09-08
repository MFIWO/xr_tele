"""Read AI Worker SG2 actions and follower observations without commanding it."""

from __future__ import annotations

import threading
import time

import numpy as np

from teleop.robot_control.robotis_ai_worker import (
    AI_WORKER_2G_GRIPPER_JOINTS,
    AI_WORKER_LEFT_ARM_JOINTS,
    AI_WORKER_RIGHT_ARM_JOINTS,
)


AI_WORKER_LEADER_TOPICS = {
    "leader_left": "/leader/joint_trajectory_command_broadcaster_left/joint_trajectory",
    "leader_right": "/leader/joint_trajectory_command_broadcaster_right/joint_trajectory",
    "leader_head": "/leader/joystick_controller_left/joint_trajectory",
    "leader_lift": "/leader/joystick_controller_right/joint_trajectory",
    "leader_mobile": "/cmd_vel",
    "follower": "/joint_states",
    "follower_mobile": "/odom",
}

AI_WORKER_LEFT_SG2_JOINTS = AI_WORKER_LEFT_ARM_JOINTS + (
    AI_WORKER_2G_GRIPPER_JOINTS[0],
)
AI_WORKER_RIGHT_SG2_JOINTS = AI_WORKER_RIGHT_ARM_JOINTS + (
    AI_WORKER_2G_GRIPPER_JOINTS[1],
)
AI_WORKER_HEAD_JOINTS = ("head_joint1", "head_joint2")
AI_WORKER_LIFT_JOINTS = ("lift_joint",)


def _initialize_observer_state(observer, stale_after):
    observer.stale_after = max(0.05, float(stale_after))
    observer._lock = threading.Lock()
    observer._stop = threading.Event()
    observer._core_ready = threading.Event()
    observer._last_error = None
    observer._updated_at = {key: None for key in AI_WORKER_LEADER_TOPICS}
    observer._values = {
        "left_state_position": np.zeros(8, dtype=np.float64),
        "right_state_position": np.zeros(8, dtype=np.float64),
        "left_state_velocity": np.zeros(8, dtype=np.float64),
        "right_state_velocity": np.zeros(8, dtype=np.float64),
        "left_action_position": np.zeros(8, dtype=np.float64),
        "right_action_position": np.zeros(8, dtype=np.float64),
        "head_state_position": np.zeros(2, dtype=np.float64),
        "head_state_velocity": np.zeros(2, dtype=np.float64),
        "head_action_position": np.zeros(2, dtype=np.float64),
        "lift_state_position": np.zeros(1, dtype=np.float64),
        "lift_state_velocity": np.zeros(1, dtype=np.float64),
        "lift_action_position": np.zeros(1, dtype=np.float64),
        "mobile_state_velocity": np.zeros(3, dtype=np.float64),
        "mobile_action_velocity": np.zeros(3, dtype=np.float64),
    }


def _ordered_values(names, values, joint_order):
    mapping = dict(zip(names, values))
    if not all(name in mapping for name in joint_order):
        return None
    ordered = np.asarray([mapping[name] for name in joint_order], dtype=np.float64)
    return ordered if np.all(np.isfinite(ordered)) else None


def _trajectory_positions(message, joint_order):
    points = getattr(message, "points", ())
    if not points:
        return None
    return _ordered_values(
        getattr(message, "joint_names", ()),
        getattr(points[0], "positions", ()),
        joint_order,
    )


class AIWorkerLeaderObserver:
    """Maintain the latest complete SG2 leader/follower sample.

    This class creates readers only. It never constructs a DDS writer and can
    therefore run beside ``ffw_sg2_ai`` without competing for robot control.
    """

    _CORE_SOURCES = ("follower", "leader_left", "leader_right")

    def __init__(self, domain_id=None, stale_after=0.5, manager=None):
        _initialize_observer_state(self, stale_after)

        if manager is None:
            from robotis_dds_python.tools.topic_manager import TopicManager

            manager = TopicManager(domain_id=domain_id)
        self._manager = manager
        self._readers = self._create_readers(manager)
        self._threads = [
            threading.Thread(
                target=self._reader_loop,
                args=(name, reader),
                name=f"ai-worker-observer-{name}",
                daemon=True,
            )
            for name, reader in self._readers.items()
        ]
        for thread in self._threads:
            thread.start()

    @staticmethod
    def _create_readers(manager):
        from cyclonedds.core import Qos, Policy
        from robotis_dds_python.idl.geometry_msgs.msg import Twist_
        from robotis_dds_python.idl.nav_msgs.msg import Odometry_
        from robotis_dds_python.idl.sensor_msgs.msg import JointState_
        from robotis_dds_python.idl.trajectory_msgs.msg import JointTrajectory_

        qos = Qos(
            Policy.Reliability.BestEffort,
            Policy.Durability.Volatile,
            Policy.History.KeepLast(1),
        )
        trajectory_sources = (
            "leader_left",
            "leader_right",
            "leader_head",
            "leader_lift",
        )
        readers = {
            name: manager.topic_reader(
                topic_name=AI_WORKER_LEADER_TOPICS[name],
                topic_type=JointTrajectory_,
                qos=qos,
            )
            for name in trajectory_sources
        }
        readers["leader_mobile"] = manager.topic_reader(
            topic_name=AI_WORKER_LEADER_TOPICS["leader_mobile"],
            topic_type=Twist_,
            qos=qos,
        )
        readers["follower"] = manager.topic_reader(
            topic_name=AI_WORKER_LEADER_TOPICS["follower"],
            topic_type=JointState_,
            qos=qos,
        )
        readers["follower_mobile"] = manager.topic_reader(
            topic_name=AI_WORKER_LEADER_TOPICS["follower_mobile"],
            topic_type=Odometry_,
            qos=qos,
        )
        return readers

    def _reader_loop(self, source, reader):
        while not self._stop.is_set():
            try:
                messages = reader.take(N=8)
                if not messages:
                    time.sleep(0.002)
                    continue
                for message in messages:
                    self._process(source, message)
            except Exception as exc:
                if not self._stop.is_set():
                    self._last_error = f"{source}: {exc}"
                    time.sleep(0.02)

    def _mark_updated(self, source, updates):
        if not updates:
            return
        now = time.monotonic()
        with self._lock:
            for key, value in updates.items():
                self._values[key] = np.asarray(value, dtype=np.float64).copy()
            self._updated_at[source] = now
            if all(self._updated_at[key] is not None for key in self._CORE_SOURCES):
                self._core_ready.set()

    def _process(self, source, message):
        if source == "follower":
            self._process_joint_state(message)
            return
        if source == "leader_left":
            value = _trajectory_positions(message, AI_WORKER_LEFT_SG2_JOINTS)
            self._mark_updated(source, {"left_action_position": value} if value is not None else {})
            return
        if source == "leader_right":
            value = _trajectory_positions(message, AI_WORKER_RIGHT_SG2_JOINTS)
            self._mark_updated(source, {"right_action_position": value} if value is not None else {})
            return
        if source == "leader_head":
            value = _trajectory_positions(message, AI_WORKER_HEAD_JOINTS)
            self._mark_updated(source, {"head_action_position": value} if value is not None else {})
            return
        if source == "leader_lift":
            value = _trajectory_positions(message, AI_WORKER_LIFT_JOINTS)
            self._mark_updated(source, {"lift_action_position": value} if value is not None else {})
            return
        if source == "leader_mobile":
            linear = getattr(message, "linear", None)
            angular = getattr(message, "angular", None)
            value = np.asarray(
                [
                    getattr(linear, "x", 0.0),
                    getattr(linear, "y", 0.0),
                    getattr(angular, "z", 0.0),
                ],
                dtype=np.float64,
            )
            self._mark_updated(source, {"mobile_action_velocity": value})
            return
        if source == "follower_mobile":
            twist = getattr(getattr(message, "twist", None), "twist", None)
            linear = getattr(twist, "linear", None)
            angular = getattr(twist, "angular", None)
            value = np.asarray(
                [
                    getattr(linear, "x", 0.0),
                    getattr(linear, "y", 0.0),
                    getattr(angular, "z", 0.0),
                ],
                dtype=np.float64,
            )
            self._mark_updated(source, {"mobile_state_velocity": value})

    def _process_joint_state(self, message):
        names = getattr(message, "name", ())
        positions = getattr(message, "position", ())
        velocities = getattr(message, "velocity", ())
        left_position = _ordered_values(names, positions, AI_WORKER_LEFT_SG2_JOINTS)
        right_position = _ordered_values(names, positions, AI_WORKER_RIGHT_SG2_JOINTS)
        if left_position is None or right_position is None:
            return

        left_velocity = _ordered_values(names, velocities, AI_WORKER_LEFT_SG2_JOINTS)
        right_velocity = _ordered_values(names, velocities, AI_WORKER_RIGHT_SG2_JOINTS)
        head_position = _ordered_values(names, positions, AI_WORKER_HEAD_JOINTS)
        head_velocity = _ordered_values(names, velocities, AI_WORKER_HEAD_JOINTS)
        lift_position = _ordered_values(names, positions, AI_WORKER_LIFT_JOINTS)
        lift_velocity = _ordered_values(names, velocities, AI_WORKER_LIFT_JOINTS)
        updates = {
            "left_state_position": left_position,
            "right_state_position": right_position,
            "left_state_velocity": (
                left_velocity if left_velocity is not None else np.zeros(8)
            ),
            "right_state_velocity": (
                right_velocity if right_velocity is not None else np.zeros(8)
            ),
        }
        if head_position is not None:
            updates["head_state_position"] = head_position
        if head_velocity is not None:
            updates["head_state_velocity"] = head_velocity
        if lift_position is not None:
            updates["lift_state_position"] = lift_position
        if lift_velocity is not None:
            updates["lift_state_velocity"] = lift_velocity
        self._mark_updated("follower", updates)

    def wait_until_ready(self, timeout=45.0):
        return self._core_ready.wait(timeout=max(0.0, float(timeout)))

    def snapshot(self):
        now = time.monotonic()
        with self._lock:
            result = {key: value.tolist() for key, value in self._values.items()}
            result["valid"] = {
                key: updated is not None and now - updated <= self.stale_after
                for key, updated in self._updated_at.items()
            }
            result["age_seconds"] = {
                key: None if updated is None else max(0.0, now - updated)
                for key, updated in self._updated_at.items()
            }
        result["last_error"] = self._last_error
        return result

    def missing_core_sources(self):
        with self._lock:
            return [key for key in self._CORE_SOURCES if self._updated_at[key] is None]

    def close(self):
        self._stop.set()
        for reader in self._readers.values():
            close = getattr(reader, "close", None) or getattr(reader, "Close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        for thread in self._threads:
            thread.join(timeout=0.5)


class AIWorkerROSObserver(AIWorkerLeaderObserver):
    """Observe the SG2 ROS graph through the process-selected ROS 2 RMW.

    The ROBOTIS Meta Quest stack defaults to ``rmw_zenoh_cpp``.  Using rclpy
    here lets this observer join that graph instead of creating an independent
    CycloneDDS participant.  Only subscriptions are created.
    """

    def __init__(
        self,
        domain_id=None,
        stale_after=0.5,
        *,
        node=None,
        executor=None,
        message_types=None,
        qos_profile=None,
        rclpy_module=None,
    ):
        _initialize_observer_state(self, stale_after)
        self._manager = None
        self._readers = {}
        self._threads = []
        self._subscriptions = []
        self._context = None
        self._rclpy = rclpy_module
        self._owns_node = node is None
        self._owns_executor = executor is None

        if node is None:
            if self._rclpy is None:
                import rclpy

                self._rclpy = rclpy
            from rclpy.context import Context

            self._context = Context()
            self._rclpy.init(
                args=None,
                context=self._context,
                domain_id=domain_id,
            )
            node = self._rclpy.create_node(
                "xr_tele_meta_quest_loop_observer",
                context=self._context,
            )
        self._node = node

        if message_types is None:
            from geometry_msgs.msg import Twist
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import JointState
            from trajectory_msgs.msg import JointTrajectory

            message_types = {
                "trajectory": JointTrajectory,
                "twist": Twist,
                "joint_state": JointState,
                "odometry": Odometry,
            }
        if qos_profile is None:
            from rclpy.qos import (
                DurabilityPolicy,
                HistoryPolicy,
                QoSProfile,
                ReliabilityPolicy,
            )

            qos_profile = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )

        trajectory_sources = (
            "leader_left",
            "leader_right",
            "leader_head",
            "leader_lift",
        )
        for source in trajectory_sources:
            self._add_subscription(
                message_types["trajectory"],
                source,
                qos_profile,
            )
        self._add_subscription(
            message_types["twist"],
            "leader_mobile",
            qos_profile,
        )
        self._add_subscription(
            message_types["joint_state"],
            "follower",
            qos_profile,
        )
        self._add_subscription(
            message_types["odometry"],
            "follower_mobile",
            qos_profile,
        )

        if executor is None:
            from rclpy.executors import SingleThreadedExecutor

            executor = SingleThreadedExecutor(context=self._context)
        self._executor = executor
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._spin,
            name="ai-worker-meta-quest-ros-observer",
            daemon=True,
        )
        self._spin_thread.start()

    @property
    def node(self):
        """Return the subscription-only node so camera readers can share it."""
        return self._node

    def _add_subscription(self, message_type, source, qos_profile):
        callback = lambda message, source=source: self._process(source, message)
        subscription = self._node.create_subscription(
            message_type,
            AI_WORKER_LEADER_TOPICS[source],
            callback,
            qos_profile,
        )
        self._subscriptions.append(subscription)

    def _spin(self):
        while not self._stop.is_set():
            try:
                self._executor.spin_once(timeout_sec=0.1)
            except Exception as exc:
                if not self._stop.is_set():
                    self._last_error = f"rclpy executor: {exc}"
                    time.sleep(0.02)

    def close(self):
        self._stop.set()
        wake = getattr(self._executor, "wake", None)
        if callable(wake):
            wake()
        self._spin_thread.join(timeout=1.0)

        for subscription in self._subscriptions:
            try:
                self._node.destroy_subscription(subscription)
            except Exception:
                pass
        self._subscriptions.clear()

        try:
            self._executor.remove_node(self._node)
        except Exception:
            pass
        if self._owns_executor:
            try:
                self._executor.shutdown(timeout_sec=0.5)
            except TypeError:
                self._executor.shutdown()
            except Exception:
                pass
        if self._owns_node:
            try:
                self._node.destroy_node()
            except Exception:
                pass
        if self._context is not None and self._rclpy is not None:
            try:
                self._rclpy.shutdown(context=self._context)
            except Exception:
                pass
