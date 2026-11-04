# AVP neck-mounted wrist delta (experimental)

This is an opt-in AI Worker-inspired wrist delta mapping for AVP worn at the
neck/chest. It **does not** bypass visionOS wear detection or restore cameras
and hand tracking that visionOS has stopped.

For simulation and offline evaluation:

```bash
python teleop/teleop_hand_and_arm.py --sim --arm H1_2 --input-mode hand \
  --arm-reference-mode fixed --avp-neck-mode --neck-translation-scale 1.0
```

The existing `r` starts teleoperation; after the AVP is secured in its chest
position press `c` to re-zero the two wrist anchors using fresh data.
The first valid active frame seeds the initial anchors. Re-zero preserves
the most recently mapped targets to avoid discontinuity.

The incoming poses are already converted into the existing robot IK frame.
Only relative translation and rotation deltas are applied; this is **not**
absolute human torso calibration, measured FK initialization, or neck motion
compensation. Use a stable chest fixture. Existing tracking timeout, arm
standby, and IK constraints remain active. If head tracking stops when AVP
is removed, arm tracking still stops: do not loosen freshness checks.
By default this feature is disabled.

Run `python -m unittest teleop.utils.test_neck_wrist_delta` for CPU tests.
