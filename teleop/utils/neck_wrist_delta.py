"""[ADDED] Wrist-delta anchors, adapted from MFIWO/ai_worker_teleop main.

Inputs are ALREADY in the robot IK frame: do not apply WebXR conversion again.
No hardware or ROS dependency. Caller must check tracking freshness.
"""
import numpy as np


def rigid_pose(pose):
    m = np.asarray(pose, dtype=np.float64)
    if m.shape != (4, 4) or not np.isfinite(m).all():
        raise ValueError("expected a finite 4x4 wrist pose")
    r = m[:3, :3]
    if (not np.allclose(m[3], [0, 0, 0, 1], atol=1e-3)
            or not np.allclose(r.T @ r, np.eye(3), atol=2e-2)
            or not np.isclose(np.linalg.det(r), 1, atol=2e-2)):
        raise ValueError("wrist pose is not a rigid transform")
    return m.copy()


class NeckWristDeltaMapper:
    def __init__(self, scale=1.0):
        if not np.isfinite(scale) or not 0.1 <= scale <= 1.5:
            raise ValueError("translation scale must be finite in [0.1, 1.5]")
        self.scale = float(scale)
        self.calibrated = False
        self.inputs = {}
        self.outputs = {}
        self.last_targets = {}

    def capture(self, inputs, output_anchors):
        """Atomically re-zero both wrists. Re-zero against last output for continuity."""
        if set(inputs) != {"left", "right"} or set(output_anchors) != {"left", "right"}:
            raise ValueError("both left/right wrists are mandatory")
        x = {k: rigid_pose(v) for k, v in inputs.items()}
        y = {k: rigid_pose(v) for k, v in output_anchors.items()}
        self.inputs, self.outputs = x, y
        self.last_targets = {k: v.copy() for k, v in y.items()}
        self.calibrated = True

    def targets(self, inputs):
        if not self.calibrated or set(inputs) != {"left", "right"}:
            raise ValueError("valid two-wrist calibration required")
        result = {}
        for side in ("left", "right"):
            live = rigid_pose(inputs[side])
            start, initial_target = self.inputs[side], self.outputs[side]
            target = np.eye(4)
            target[:3, 3] = initial_target[:3, 3] + self.scale * (
                live[:3, 3] - start[:3, 3])
            target[:3, :3] = (
                live[:3, :3] @ start[:3, :3].T @ initial_target[:3, :3])
            result[side] = rigid_pose(target)
        self.last_targets = {k: v.copy() for k, v in result.items()}
        return result
