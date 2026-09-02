#!/usr/bin/env python3
"""Benchmark H1_2 CasADi IK solve time using arm_q trajectories from a comm_state log.

Replays recorded joint angles: FK(q) -> wrist targets -> solve_ik, and reports
the per-solve wall-time distribution. No robot connection needed.

Run inside the xr_teleop container (tv env), from the teleop directory:
    cd /workspace/xr_teleoperate/teleop
    /opt/conda/envs/tv/bin/python ../tools/ik_bench.py <comm_state.jsonl> [n_samples]
"""
import json
import sys
import time

import numpy as np
import pinocchio as pin

sys.path.append("..")
from teleop.robot_control.robot_arm_ik import H1_2_ArmIK


def main():
    log_path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 400

    qs = []
    with open(log_path) as f:
        for line in f:
            if '"cycle"' not in line:
                continue
            r = json.loads(line)
            if r.get("mode") == "ACTIVE" and r.get("arm_q") and len(r["arm_q"]) == 14:
                qs.append(np.array(r["arm_q"], dtype=np.float64))
            if len(qs) >= n + 200:
                break
    qs = qs[200:200 + n]  # skip startup
    print(f"loaded {len(qs)} arm_q samples from log")

    ik = H1_2_ArmIK(scale_input_poses=False)
    model = ik.reduced_robot.model
    data = ik.reduced_robot.data
    lid, rid = ik.L_hand_id, ik.R_hand_id

    def fk(q):
        pin.framesForwardKinematics(model, data, q)
        return data.oMf[lid].homogeneous.copy(), data.oMf[rid].homogeneous.copy()

    # warmup
    lw, rw = fk(qs[0])
    for _ in range(5):
        ik.solve_ik(lw, rw, qs[0], np.zeros(14))

    times = []
    cur = qs[0].copy()
    for q in qs:
        lw, rw = fk(q)
        t0 = time.perf_counter()
        sol_q, _ = ik.solve_ik(lw, rw, cur, np.zeros(14))
        times.append((time.perf_counter() - t0) * 1000)
        cur = np.asarray(sol_q, dtype=np.float64)

    times.sort()
    m = len(times)
    pct = lambda p: times[min(m - 1, int(m * p))]
    print(f"solve_ik ms over {m} solves:")
    print(f"  min={times[0]:.1f} p50={pct(0.5):.1f} p90={pct(0.9):.1f} "
          f"p99={pct(0.99):.1f} max={times[-1]:.1f} mean={sum(times)/m:.1f}")
    print(f"  solves >33.3ms (30Hz budget): {sum(1 for t in times if t > 33.3)} "
          f"({100*sum(1 for t in times if t > 33.3)/m:.1f}%)")
    print(f"  solves >50ms (20Hz budget): {sum(1 for t in times if t > 50)} "
          f"({100*sum(1 for t in times if t > 50)/m:.1f}%)")


if __name__ == "__main__":
    main()
