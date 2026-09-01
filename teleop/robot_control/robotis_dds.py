"""Small ROBOTIS DDS transport shared by AI Worker arm and hand control."""

import os
import threading
import time


class RobotisJointTrajectoryTransport:
    """Publish JointTrajectory and fan out JointState using robotis_dds_python."""

    def __init__(self, topic_names, joint_state_callback=None):
        try:
            from robotis_dds_python.idl.builtin_interfaces.msg import Duration_, Time_
            from robotis_dds_python.idl.sensor_msgs.msg import JointState_
            from robotis_dds_python.idl.std_msgs.msg import Header_
            from robotis_dds_python.idl.trajectory_msgs.msg import (
                JointTrajectory_,
                JointTrajectoryPoint_,
            )
            from robotis_dds_python.tools.topic_manager import TopicManager
        except ImportError as exc:
            raise RuntimeError(
                "AI Worker DDS requires cyclonedds and robotis_dds_python in the xr_tele environment."
            ) from exc

        self.Duration = Duration_
        self.Time = Time_
        self.Header = Header_
        self.JointTrajectory = JointTrajectory_
        self.JointTrajectoryPoint = JointTrajectoryPoint_
        self.domain_id = int(os.environ.get("ROS_DOMAIN_ID", "0"))
        self.manager = TopicManager(domain_id=self.domain_id)
        self.writers = {
            key: self.manager.topic_writer(topic_name=name, topic_type=JointTrajectory_)
            for key, name in topic_names.items()
        }
        self.reader = self.manager.topic_reader(topic_name="/joint_states", topic_type=JointState_)
        self.callback = joint_state_callback
        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, name="robotis-joint-state", daemon=True)
        self.thread.start()

    def _reader_loop(self):
        while self.running:
            received = False
            try:
                for message in self.reader.take_iter():
                    received = True
                    if message is not None and self.callback is not None:
                        self.callback(message)
            except Exception:
                if self.running:
                    time.sleep(0.01)
            if not received:
                time.sleep(0.001)

    def publish(self, key, joint_names, positions, duration):
        seconds = int(duration)
        nanoseconds = int((duration - seconds) * 1e9)
        point = self.JointTrajectoryPoint(
            positions=[float(value) for value in positions],
            velocities=[],
            accelerations=[],
            effort=[],
            time_from_start=self.Duration(sec=seconds, nanosec=nanoseconds),
        )
        message = self.JointTrajectory(
            header=self.Header(
                # A zero ROS timestamp means "start immediately".  Do not put
                # the publisher's wall clock here: xr_tele commonly runs on a
                # different computer from ros2_control, and even modest clock
                # skew makes the controller reject every point as being in the
                # past.
                stamp=self.Time(sec=0, nanosec=0),
                frame_id="",
            ),
            joint_names=list(joint_names),
            points=[point],
        )
        self.writers[key].write(message)

    def publish_trajectory(
        self,
        key,
        joint_names,
        positions,
        times_from_start,
        velocities=None,
        accelerations=None,
    ):
        """Publish one complete, time-parameterized joint trajectory."""
        joint_names = list(joint_names)
        positions = list(positions)
        times_from_start = list(times_from_start)
        if not positions or len(positions) != len(times_from_start):
            raise ValueError("positions and times_from_start must have the same non-zero length.")

        velocities = [None] * len(positions) if velocities is None else list(velocities)
        accelerations = [None] * len(positions) if accelerations is None else list(accelerations)
        if len(velocities) != len(positions) or len(accelerations) != len(positions):
            raise ValueError("velocity and acceleration rows must match the trajectory length.")

        points = []
        previous_time = -1.0
        for position, velocity, acceleration, point_time in zip(
            positions,
            velocities,
            accelerations,
            times_from_start,
        ):
            if len(position) != len(joint_names):
                raise ValueError("each trajectory position row must match joint_names.")
            if velocity is not None and len(velocity) != len(joint_names):
                raise ValueError("each trajectory velocity row must match joint_names.")
            if acceleration is not None and len(acceleration) != len(joint_names):
                raise ValueError("each trajectory acceleration row must match joint_names.")

            point_time = float(point_time)
            if point_time < 0.0 or point_time <= previous_time:
                raise ValueError("trajectory times must be non-negative and strictly increasing.")
            previous_time = point_time
            total_nanoseconds = int(round(point_time * 1e9))
            seconds, nanoseconds = divmod(total_nanoseconds, 1_000_000_000)
            points.append(
                self.JointTrajectoryPoint(
                    positions=[float(value) for value in position],
                    velocities=[] if velocity is None else [float(value) for value in velocity],
                    accelerations=[] if acceleration is None else [float(value) for value in acceleration],
                    effort=[],
                    time_from_start=self.Duration(sec=seconds, nanosec=nanoseconds),
                )
            )

        message = self.JointTrajectory(
            header=self.Header(
                # A zero ROS timestamp means "start immediately".  Do not put
                # the publisher's wall clock here: xr_tele commonly runs on a
                # different computer from ros2_control, and even modest clock
                # skew makes the controller reject every point as being in the
                # past.
                stamp=self.Time(sec=0, nanosec=0),
                frame_id="",
            ),
            joint_names=joint_names,
            points=points,
        )
        self.writers[key].write(message)

    def close(self):
        if not self.running:
            return
        self.running = False
        try:
            self.reader.Close()
        except Exception:
            pass
        self.thread.join(timeout=1.0)
        for writer in self.writers.values():
            try:
                writer.Close()
            except Exception:
                pass
