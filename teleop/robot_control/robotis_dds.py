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
        now_ns = time.time_ns()
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
                stamp=self.Time(sec=now_ns // 1_000_000_000, nanosec=now_ns % 1_000_000_000),
                frame_id="",
            ),
            joint_names=list(joint_names),
            points=[point],
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
