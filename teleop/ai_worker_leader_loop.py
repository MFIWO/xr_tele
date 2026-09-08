"""Observe SG2 leader or ROBOTIS WebXR teleoperation and stream it to Loop."""

from __future__ import annotations

import logging
import os
import time

from teleop.robot_control.robotis_image_client import RobotisDDSImageClient
from teleop.robot_control.robotis_leader_observer import AIWorkerLeaderObserver
from teleop.utils.loop_streamer import (
    AIWorkerLeaderLoopStreamer,
    LoopCameraStreamer,
)


logger = logging.getLogger(__name__)


def _set_camera(loop_camera, setter_name, frame):
    if frame is None:
        return
    getattr(loop_camera, setter_name)(
        getattr(frame, "bgr", None),
        getattr(frame, "jpg", None),
    )


def _run_observer_loop(args, *, transport):
    """Run a command-free SG2 observation → Config Loop bridge until Ctrl+C."""
    if transport not in ("dds", "ros2"):
        raise ValueError(f"unsupported observer transport: {transport}")
    mode_name = "leader-master" if transport == "dds" else "meta-quest"
    owner_name = (
        "ffw_sg2_ai and the physical leader"
        if transport == "dds"
        else "ROBOTIS Vuer and Cyclo Control"
    )
    if not args.loop:
        raise ValueError(f"--{mode_name}-loop requires --loop.")
    if args.frequency <= 0.0:
        raise ValueError("--frequency must be greater than zero.")
    if args.ai_worker_ros_domain_id is not None:
        if args.ai_worker_ros_domain_id < 0:
            raise ValueError("--ai-worker-ros-domain-id must be zero or greater.")
        os.environ["ROS_DOMAIN_ID"] = str(args.ai_worker_ros_domain_id)

    observer = None
    image_client = None
    robot_streamer = None
    camera_streamer = None
    try:
        logger.warning(
            "[%s loop] OBSERVER ONLY: no arm, gripper, head, lift, or base "
            "command publisher will be created. %s retains control.",
            mode_name,
            owner_name,
        )
        if transport == "ros2":
            from teleop.robot_control.robotis_webxr_bridge_client import (
                AIWorkerWebXRBridgeClient,
            )

            observer = AIWorkerWebXRBridgeClient(
                domain_id=args.ai_worker_ros_domain_id,
                stale_after=args.leader_master_stale_timeout,
                ros_python=args.meta_quest_ros_python,
                ros_setup=args.meta_quest_ros_setup,
                rmw_implementation=args.meta_quest_rmw_implementation,
            )
            image_client = observer
        else:
            observer = AIWorkerLeaderObserver(
                domain_id=args.ai_worker_ros_domain_id,
                stale_after=args.leader_master_stale_timeout,
            )
            image_client = RobotisDDSImageClient(
                domain_id=args.ai_worker_ros_domain_id,
            )
        robot_streamer = AIWorkerLeaderLoopStreamer(
            args.loop_addr,
            args.frequency,
        )
        camera_streamer = LoopCameraStreamer(
            args.loop_addr,
            image_client.get_cam_config(),
        )
        robot_streamer.connect()
        camera_streamer.connect()

        logger.info(
            "[%s loop] connected transport=%s rmw=%s domain=%s loop=%s "
            "frequency=%.1fHz; "
            "set the task prompt and episode start/stop in Config Loop. Ctrl+C exits.",
            mode_name,
            transport,
            os.environ.get("RMW_IMPLEMENTATION", "process default"),
            os.environ.get("ROS_DOMAIN_ID", "0"),
            args.loop_addr,
            args.frequency,
        )
        logger.info(
            "[%s loop] startup/attention poses remain owned by %s; this "
            "observer never overrides them.",
            mode_name,
            owner_name,
        )

        period = 1.0 / float(args.frequency)
        next_tick = time.monotonic()
        last_status = float("-inf")
        startup_deadline = next_tick + args.leader_master_startup_timeout
        startup_warning_logged = False
        core_ready_logged = False
        while True:
            snapshot = observer.snapshot()
            robot_streamer.send_snapshot(time.time_ns() // 1000, snapshot)
            _set_camera(camera_streamer, "set_head", image_client.get_head_frame())
            _set_camera(
                camera_streamer,
                "set_left_wrist",
                image_client.get_left_wrist_frame(),
            )
            _set_camera(
                camera_streamer,
                "set_right_wrist",
                image_client.get_right_wrist_frame(),
            )

            now = time.monotonic()
            if not core_ready_logged and observer.wait_until_ready(0.0):
                core_ready_logged = True
                logger.info(
                    "[%s loop] follower and both 8-joint action streams are ready.",
                    mode_name,
                )
            elif (
                not core_ready_logged
                and not startup_warning_logged
                and now >= startup_deadline
            ):
                startup_warning_logged = True
                if transport == "ros2":
                    logger.warning(
                        "[%s loop] core ROS 2 sources are still missing after %.1fs: %s. "
                        "Streaming continues with source_valid=false. Confirm "
                        "ffw_sg2_follower_ai, motion_controller controller_type:=vr, "
                        "the Zenoh daemon/router, RMW_IMPLEMENTATION, and ROS_DOMAIN_ID.",
                        mode_name,
                        args.leader_master_startup_timeout,
                        observer.missing_core_sources(),
                    )
                else:
                    logger.warning(
                        "[%s loop] core DDS sources are still missing after %.1fs: %s. "
                        "Streaming continues with source_valid=false. ffw_sg2_ai publishes "
                        "arm actions only after leader auto mode is enabled; otherwise "
                        "check DDS/Zenoh bridging and ROS_DOMAIN_ID.",
                        mode_name,
                        args.leader_master_startup_timeout,
                        observer.missing_core_sources(),
                    )
            if (
                args.leader_master_status_rate > 0.0
                and now - last_status >= 1.0 / args.leader_master_status_rate
            ):
                last_status = now
                valid = snapshot["valid"]
                stale = [name for name, is_valid in valid.items() if not is_valid]
                logger.info(
                    "[%s loop] source_valid=%d/%d stale=%s last_error=%s camera_error=%s",
                    mode_name,
                    sum(valid.values()),
                    len(valid),
                    stale or "none",
                    snapshot.get("last_error") or "none",
                    getattr(image_client, "last_error", None) or "none",
                )

            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        logger.info("[%s loop] Ctrl+C received; stopping recorder bridge.", mode_name)
    finally:
        for resource in (
            camera_streamer,
            robot_streamer,
            image_client,
            observer,
        ):
            if resource is None:
                continue
            try:
                resource.close()
            except Exception as exc:
                logger.warning(
                    "[%s loop] cleanup failed for %s: %s",
                    mode_name,
                    resource.__class__.__name__,
                    exc,
                )
        logger.warning(
            "[%s loop] stopped without publishing a home/attention command; %s "
            "remains authoritative.",
            mode_name,
            owner_name,
        )


def run_leader_master_loop(args):
    """Run the command-free leader-master → Config Loop DDS bridge."""
    return _run_observer_loop(args, transport="dds")


def run_meta_quest_loop(args):
    """Run the command-free ROBOTIS WebXR → Config Loop ROS 2 bridge."""
    return _run_observer_loop(args, transport="ros2")
