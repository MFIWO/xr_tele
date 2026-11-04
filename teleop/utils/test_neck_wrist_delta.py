"""CPU-only checks: python -m unittest teleop.utils.test_neck_wrist_delta."""
import unittest
import numpy as np
from teleop.utils.neck_wrist_delta import NeckWristDeltaMapper


def pose(x=0.):
    m = np.eye(4)
    m[0, 3] = x
    return m


class TestNeckMapper(unittest.TestCase):
    def test_independent_delta(self):
        m = NeckWristDeltaMapper(.5)
        m.capture({"left": pose(1), "right": pose(-1)},
                  {"left": pose(.2), "right": pose(-.2)})
        goals = m.targets({"left": pose(1.4), "right": pose(-1.2)})
        self.assertAlmostEqual(goals["left"][0, 3], .4)
        self.assertAlmostEqual(goals["right"][0, 3], -.3)

    def test_rezero_preserves_targets(self):
        m = NeckWristDeltaMapper()
        zeros = {"left": pose(), "right": pose()}
        m.capture(zeros, zeros)
        inputs = {"left": pose(.13), "right": pose(-.21)}
        before = m.targets(inputs)
        m.capture(inputs, m.last_targets)
        after = m.targets(inputs)
        for side in zeros:
            np.testing.assert_allclose(before[side], after[side])

    def test_bad_frame_does_not_change_anchor(self):
        m = NeckWristDeltaMapper()
        zeros = {"left": pose(), "right": pose()}
        m.capture(zeros, zeros)
        bad = pose()
        bad[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            m.capture({"left": pose(), "right": bad}, zeros)
        np.testing.assert_allclose(m.targets(zeros)["left"], pose())

    def test_reflection_rejected(self):
        m = NeckWristDeltaMapper()
        flip = pose()
        flip[0, 0] = -1
        with self.assertRaises(ValueError):
            m.capture({"left": flip, "right": pose()},
                      {"left": pose(), "right": pose()})


if __name__ == "__main__":
    unittest.main()
