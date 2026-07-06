#!/usr/bin/env python3
"""Probe RH5DG2/Inspire hand state DDS topics without publishing anything."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_


@dataclass
class TopicProbe:
    topic: str
    msg_type: Any
    type_name: str
    subscriber: Any = None
    count: int = 0
    last_seen: float = 0.0
    last_summary: str = ""


def _load_inspire_state_type():
    try:
        from inspire_sdkpy import inspire_dds

        return inspire_dds.inspire_hand_state
    except Exception as exc:
        print(f"[probe] inspire_sdkpy unavailable; skipping FTP-style topics: {type(exc).__name__}: {exc}")
        return None


def _fmt_motor_states(msg):
    states = list(getattr(msg, "states", []) or [])
    q = np.array([float(getattr(state, "q", 0.0)) for state in states], dtype=np.float64)
    dq = np.array([float(getattr(state, "dq", 0.0)) for state in states], dtype=np.float64)
    tau = np.array([float(getattr(state, "tau_est", 0.0)) for state in states], dtype=np.float64)
    if q.size == 0:
        return "states_len=0"
    return (
        f"states_len={len(states)} "
        f"q_first={np.round(q[: min(8, q.size)], 4).tolist()} "
        f"q_last={np.round(q[-min(8, q.size):], 4).tolist()} "
        f"dq_minmax=({dq.min():.4f},{dq.max():.4f}) "
        f"tau_minmax=({tau.min():.4f},{tau.max():.4f})"
    )


def _fmt_inspire_state(msg):
    angle = np.array(list(getattr(msg, "angle_act", []) or []), dtype=np.float64)
    force = np.array(list(getattr(msg, "force_act", []) or []), dtype=np.float64)
    temp = np.array(list(getattr(msg, "temp", []) or []), dtype=np.float64)
    parts = [f"angle_len={angle.size}"]
    if angle.size:
        parts.append(f"angle_first={np.round(angle[: min(13, angle.size)], 2).tolist()}")
        parts.append(f"angle_minmax=({angle.min():.2f},{angle.max():.2f})")
    if force.size:
        parts.append(f"force_minmax=({force.min():.2f},{force.max():.2f})")
    if temp.size:
        parts.append(f"temp_minmax=({temp.min():.2f},{temp.max():.2f})")
    return " ".join(parts)


def _summarize(msg, type_name):
    if type_name == "MotorStates_":
        return _fmt_motor_states(msg)
    return _fmt_inspire_state(msg)


def _make_probes():
    probes = [
        TopicProbe("rt/rh5dg2/state", MotorStates_, "MotorStates_"),
        TopicProbe("rt/inspire/state", MotorStates_, "MotorStates_"),
    ]
    inspire_state_type = _load_inspire_state_type()
    if inspire_state_type is not None:
        probes.extend(
            [
                TopicProbe("rt/rh5dg2_hand/state/l", inspire_state_type, "inspire_hand_state"),
                TopicProbe("rt/rh5dg2_hand/state/r", inspire_state_type, "inspire_hand_state"),
                TopicProbe("rt/inspire_hand/state/l", inspire_state_type, "inspire_hand_state"),
                TopicProbe("rt/inspire_hand/state/r", inspire_state_type, "inspire_hand_state"),
            ]
        )
    return probes


def main():
    parser = argparse.ArgumentParser(description="Subscribe-only DDS probe for RH5DG2/Inspire hand state topics.")
    parser.add_argument("--network-interface", type=str, default=None)
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--poll-sleep", type=float, default=0.002)
    args = parser.parse_args()

    print(
        "[probe] subscribe-only mode: no ChannelPublisher is created and no command topic is written. "
        f"domain={args.domain} network_interface={args.network_interface}"
    )
    ChannelFactoryInitialize(args.domain, networkInterface=args.network_interface)

    probes = _make_probes()
    for probe in probes:
        try:
            probe.subscriber = ChannelSubscriber(probe.topic, probe.msg_type)
            probe.subscriber.Init()
            print(f"[probe] subscribed topic={probe.topic} type={probe.type_name}")
        except Exception as exc:
            print(f"[probe] subscribe failed topic={probe.topic} type={probe.type_name}: {type(exc).__name__}: {exc}")
            probe.subscriber = None

    start = time.time()
    next_heartbeat = start + 1.0
    while time.time() - start < args.duration:
        now = time.time()
        for probe in probes:
            if probe.subscriber is None:
                continue
            msg = probe.subscriber.Read()
            if msg is None:
                continue
            probe.count += 1
            probe.last_seen = now
            probe.last_summary = _summarize(msg, probe.type_name)
            print(
                f"[probe hit] topic={probe.topic} type={probe.type_name} "
                f"count={probe.count} t={now - start:.3f}s {probe.last_summary}",
                flush=True,
            )
        if now >= next_heartbeat:
            active = [probe.topic for probe in probes if probe.count > 0]
            print(f"[probe heartbeat] elapsed={now - start:.1f}s active_topics={active}")
            next_heartbeat = now + 1.0
        time.sleep(max(args.poll_sleep, 0.0))

    print("[probe summary]")
    for probe in probes:
        status = "HIT" if probe.count else "NO_DATA"
        age = time.time() - probe.last_seen if probe.last_seen else None
        print(
            f"  {status} topic={probe.topic} type={probe.type_name} count={probe.count} "
            f"last_age_s={age:.3f}" if age is not None else
            f"  {status} topic={probe.topic} type={probe.type_name} count={probe.count} last_age_s=n/a"
        )
        if probe.last_summary:
            print(f"    last={probe.last_summary}")


if __name__ == "__main__":
    main()
