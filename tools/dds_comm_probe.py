#!/usr/bin/env python3
"""Read-only DDS probe for H1_2 teleop communication health.

Subscribes to rt/lowstate (robot state), rt/arm_sdk and rt/lowcmd (arm
commands) for a few seconds, then reports message rates, arm joint
positions, and motor temperatures. Publishes nothing.

Run inside the xr_teleop container:
    python3 /workspace/xr_teleoperate/tools/dds_comm_probe.py --iface enp44s0 --seconds 5
"""
import argparse
import time
import threading

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd, LowState_ as hg_LowState

H1_2_ARM_JOINTS = {
    13: "L.ShoulderPitch", 14: "L.ShoulderRoll", 15: "L.ShoulderYaw",
    16: "L.ElbowPitch", 17: "L.ElbowRoll", 18: "L.WristPitch", 19: "L.WristYaw",
    20: "R.ShoulderPitch", 21: "R.ShoulderRoll", 22: "R.ShoulderYaw",
    23: "R.ElbowPitch", 24: "R.ElbowRoll", 25: "R.WristPitch", 26: "R.WristYaw",
}


class TopicCounter:
    def __init__(self, topic, msg_type):
        self.topic = topic
        self.count = 0
        self.first_ts = None
        self.last_ts = None
        self.latest = None
        self.lock = threading.Lock()
        self.sub = ChannelSubscriber(topic, msg_type)
        self.sub.Init(self._handler, 10)

    def _handler(self, msg):
        now = time.monotonic()
        with self.lock:
            self.count += 1
            if self.first_ts is None:
                self.first_ts = now
            self.last_ts = now
            self.latest = msg

    def rate(self):
        with self.lock:
            if self.count < 2 or self.first_ts is None:
                return 0.0
            span = self.last_ts - self.first_ts
            return (self.count - 1) / span if span > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="enp44s0")
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()

    ChannelFactoryInitialize(0, args.iface)
    lowstate = TopicCounter("rt/lowstate", hg_LowState)
    arm_sdk = TopicCounter("rt/arm_sdk", hg_LowCmd)
    lowcmd = TopicCounter("rt/lowcmd", hg_LowCmd)

    print(f"[probe] listening {args.seconds:.0f}s on iface={args.iface} ...")
    time.sleep(args.seconds)

    print(f"\n[rt/lowstate] msgs={lowstate.count} rate={lowstate.rate():.1f} Hz")
    print(f"[rt/arm_sdk ] msgs={arm_sdk.count} rate={arm_sdk.rate():.1f} Hz (teleop arm commands)")
    print(f"[rt/lowcmd  ] msgs={lowcmd.count} rate={lowcmd.rate():.1f} Hz (debug-mode arm commands)")

    msg = lowstate.latest
    if msg is None:
        print("\n[state] NO lowstate received - robot state is NOT reaching this PC.")
        return 1

    print(f"\n[state] mode_machine={msg.mode_machine} tick={msg.tick}")
    try:
        soc = msg.bms_state.soc
        print(f"[state] battery soc={soc}%")
    except Exception:
        pass
    print(f"{'joint':<16}{'q(rad)':>9}{'dq':>8}{'tau':>8}{'temp(C)':>12}")
    for idx, name in H1_2_ARM_JOINTS.items():
        m = msg.motor_state[idx]
        temps = m.temperature
        if isinstance(temps, (list, tuple)):
            temp_str = "/".join(str(int(t)) for t in temps)
        else:
            temp_str = str(int(temps))
        print(f"{name:<16}{m.q:>9.3f}{m.dq:>8.3f}{m.tau_est:>8.2f}{temp_str:>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
