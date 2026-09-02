import ast
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = REPO_ROOT / "teleop" / "teleop_hand_and_arm.py"


def load_entrypoint_helpers(*names):
    tree = ast.parse(ENTRYPOINT.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "np": np,
        "logger_mp": SimpleNamespace(
            warning=lambda *args, **kwargs: None,
            error=lambda *args, **kwargs: None,
        ),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(ENTRYPOINT), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


class XRBackendSelectionTest(unittest.TestCase):
    def test_cli_keeps_vuer_as_the_default(self):
        tree = ast.parse(ENTRYPOINT.read_text(encoding="utf-8"))
        matches = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--xr-backend"
            ):
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            matches.append(keywords)

        self.assertEqual(len(matches), 1)
        keywords = matches[0]
        self.assertEqual(ast.literal_eval(keywords["default"]), "vuer")
        self.assertEqual(
            ast.literal_eval(keywords["choices"]),
            ["vuer", "visionproteleop"],
        )

    def test_help_does_not_require_optional_avp_stream(self):
        result = subprocess.run(
            [sys.executable, str(ENTRYPOINT), "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--xr-backend {vuer,visionproteleop}", result.stdout)

    def test_existing_vuer_teledata_imports_without_avp_stream(self):
        script = r'''
import builtins
import numpy as np

real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "avp_stream" or name.startswith("avp_stream."):
        raise AssertionError("Vuer path imported optional avp_stream")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

from televuer.tv_wrapper import TeleData, TeleVuerWrapper
sample = TeleData(np.eye(4), np.eye(4), np.eye(4))
assert sample.head_pose.shape == (4, 4)
assert TeleVuerWrapper.__name__ == "TeleVuerWrapper"
'''
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_native_enable_requires_fresh_settled_reanchor_state(self):
        entrypoint = load_entrypoint_helpers("_native_enable_preconditions_met")

        ready = {
            "state": "REANCHOR_REQUIRED",
            "session_alive": True,
            "settling_complete": True,
            "external_hold": False,
            "sample_age_s": 0.01,
        }
        self.assertTrue(entrypoint._native_enable_preconditions_met(ready, 0.25))
        for key, value in (
            ("state", "TRACKING_STALE"),
            ("session_alive", False),
            ("settling_complete", False),
            ("external_hold", True),
            ("sample_age_s", 0.3),
            ("sample_age_s", np.nan),
        ):
            rejected = dict(ready)
            rejected[key] = value
            with self.subTest(key=key, value=value):
                self.assertFalse(
                    entrypoint._native_enable_preconditions_met(rejected, 0.25)
                )

    def test_hx5_hold_pauses_worker_without_open_command(self):
        entrypoint = load_entrypoint_helpers(
            "_safe_enter_hand_standby_open",
            "_safe_enter_hand_tracking_standby",
        )

        class FakeHX5:
            _enabled = True

            def enter_standby_open(self):
                raise AssertionError("HX5 hold must not publish an open target")

        hand = FakeHX5()
        self.assertTrue(entrypoint._safe_enter_hand_tracking_standby(hand, "hx5_d20"))
        self.assertFalse(hand._enabled)

    def test_native_hx5_cleanup_closes_without_open_or_stop_command(self):
        entrypoint = load_entrypoint_helpers("_safe_stop_hx5_without_motion")

        class FakeThread:
            def __init__(self):
                self.alive = True
                self.join_calls = []

            def is_alive(self):
                return self.alive

            def join(self, timeout):
                self.join_calls.append(timeout)
                self.alive = False

        class FakeTransport:
            def __init__(self):
                self.close_calls = 0

            def close(self):
                self.close_calls += 1

        class FakeHX5:
            def __init__(self):
                self._enabled = True
                self._running = True
                self._stopped = False
                self._thread = FakeThread()
                self.transport = FakeTransport()

            def stop(self):
                raise AssertionError("native cleanup must not call zero/open stop()")

            def restore_initial_pose(self):
                raise AssertionError("native cleanup must not restore/open the hands")

        hand = FakeHX5()
        self.assertTrue(entrypoint._safe_stop_hx5_without_motion(hand, timeout=0.25))
        self.assertFalse(hand._enabled)
        self.assertFalse(hand._running)
        self.assertTrue(hand._stopped)
        self.assertEqual(hand._thread.join_calls, [0.25])
        self.assertEqual(hand.transport.close_calls, 1)

    def test_native_hx5_cleanup_keeps_transport_open_on_join_timeout(self):
        entrypoint = load_entrypoint_helpers("_safe_stop_hx5_without_motion")

        class StuckThread:
            def is_alive(self):
                return True

            def join(self, timeout):
                self.timeout = timeout

        class FakeTransport:
            def __init__(self):
                self.close_calls = 0

            def close(self):
                self.close_calls += 1

        hand = SimpleNamespace(
            _enabled=True,
            _running=True,
            _stopped=False,
            _thread=StuckThread(),
            transport=FakeTransport(),
        )
        self.assertFalse(entrypoint._safe_stop_hx5_without_motion(hand, timeout=0.0))
        self.assertFalse(hand._enabled)
        self.assertFalse(hand._running)
        self.assertFalse(hand._stopped)
        self.assertEqual(hand.transport.close_calls, 0)

    def test_native_hx5_is_paused_before_recorder_cleanup_and_skips_stop(self):
        source = ENTRYPOINT.read_text(encoding="utf-8")
        finalizer = source[source.index("    finally:", source.index("except KeyboardInterrupt")) :]
        pause = finalizer.index("_safe_enter_hand_tracking_standby(hand_ctrl, args.ee)")
        recorder_close = finalizer.index("recorder.close()")
        no_motion_cleanup = finalizer.index("_safe_stop_hx5_without_motion(hand_ctrl)")
        legacy_stop = finalizer.index("hand_ctrl.stop()")
        self.assertLess(pause, recorder_close)
        self.assertLess(no_motion_cleanup, legacy_stop)
        self.assertIn("if native_no_motion_hx5:", finalizer[:no_motion_cleanup])
        self.assertIn("else:", finalizer[no_motion_cleanup:legacy_stop])

    def test_native_arm_hold_reanchors_velocity_limiter(self):
        entrypoint = load_entrypoint_helpers("_safe_sync_arm_hold_to_measured")

        class FakeArm:
            def __init__(self):
                self.sync_calls = 0

            def sync_arm_command_to_measured(self):
                self.sync_calls += 1

        arm = FakeArm()
        self.assertTrue(entrypoint._safe_sync_arm_hold_to_measured(arm))
        self.assertEqual(arm.sync_calls, 1)

    def test_native_hardware_gate_precedes_production_controller_construction(self):
        source = ENTRYPOINT.read_text(encoding="utf-8")
        gate = source.index("if native_production_gate:", source.index("tv_wrapper = VisionProTeleopBackend"))
        arm_constructor = source.index("arm_ctrl = AIWorkerArmController(")
        hx5_constructor = source.index("hand_ctrl = hx5_controller_class(")
        self.assertLess(gate, arm_constructor)
        self.assertLess(gate, hx5_constructor)
        self.assertIn(
            "native_production_gate and not native_hardware_armed",
            source[gate:hx5_constructor],
        )


if __name__ == "__main__":
    unittest.main()
