"""Load and validate AI Worker episodes before replay starts DDS control.

The replay path intentionally accepts commands only from ``actions.*.qpos``.
Recorded states are useful for visualization, but are never substituted for a
missing action.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from teleop.robot_control.robot_hand_hx5_d20 import (
    LEFT_LOWER as HX5_LEFT_LOWER,
    LEFT_UPPER as HX5_LEFT_UPPER,
    RIGHT_LOWER as HX5_RIGHT_LOWER,
    RIGHT_UPPER as HX5_RIGHT_UPPER,
)
from teleop.robot_control.robotis_ai_worker import (
    AI_WORKER_ARM_LOWER,
    AI_WORKER_ARM_UPPER,
)


_PART_DIMS = {
    "left_arm": 7,
    "right_arm": 7,
    "left_ee": 20,
    "right_ee": 20,
}


@dataclass(frozen=True)
class AIWorkerReplayFrame:
    """Validated targets plus optional observations for one episode frame."""

    idx: int
    left_arm_action: np.ndarray | None
    right_arm_action: np.ndarray | None
    left_ee_action: np.ndarray | None
    right_ee_action: np.ndarray | None
    recorded_states: dict[str, np.ndarray]
    source_frame: dict[str, Any]

    @property
    def arm_action(self) -> np.ndarray | None:
        if self.left_arm_action is None or self.right_arm_action is None:
            return None
        return np.concatenate((self.left_arm_action, self.right_arm_action))

    @property
    def hand_action(self) -> np.ndarray | None:
        if self.left_ee_action is None or self.right_ee_action is None:
            return None
        return np.concatenate((self.left_ee_action, self.right_ee_action))


def _resolve_episode_json(path: str | Path) -> Path:
    episode_path = Path(path).expanduser()
    if not episode_path.is_absolute():
        episode_path = (Path.cwd() / episode_path).resolve()
    if episode_path.is_dir():
        episode_path = episode_path / "data.json"
    if not episode_path.is_file():
        raise FileNotFoundError(f"episode json not found: {episode_path}")
    return episode_path


def _load_json_with_final_closure_repair(episode_path: Path) -> dict[str, Any]:
    text = episode_path.read_text(encoding="utf-8")
    try:
        episode = json.loads(text)
    except json.JSONDecodeError as exc:
        stripped = text.rstrip()
        # Only repair an otherwise complete EpisodeWriter stream whose parser
        # stopped at EOF. Never try to recover malformed content in the middle.
        if exc.pos < len(stripped):
            raise
        episode = None
        for suffix in ("\n]\n}\n", "\n}\n]\n}\n"):
            try:
                candidate = json.loads(stripped + suffix)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                episode = candidate
                break
        if episode is None:
            raise exc

    if not isinstance(episode, dict):
        raise ValueError(f"episode root must be an object: {episode_path}")
    return episode


def _validated_fps(info: dict[str, Any]) -> float:
    image_info = info.get("image") if isinstance(info, dict) else None
    raw_fps = image_info.get("fps") if isinstance(image_info, dict) else None
    if isinstance(raw_fps, bool):
        raise ValueError("info.image.fps must be a finite number greater than zero")
    try:
        fps = float(raw_fps)
    except (TypeError, ValueError) as exc:
        raise ValueError("info.image.fps must be a finite number greater than zero") from exc
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("info.image.fps must be a finite number greater than zero")
    return fps


def load_episode(path: str | Path) -> tuple[float, list[dict[str, Any]], Path, dict[str, Any]]:
    """Load an EpisodeWriter JSON file and return ``fps, frames, path, info``."""

    episode_path = _resolve_episode_json(path)
    episode = _load_json_with_final_closure_repair(episode_path)
    info = episode.get("info")
    frames = episode.get("data")
    if not isinstance(info, dict):
        raise ValueError(f"episode info must be an object: {episode_path}")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"episode contains no frames: {episode_path}")
    if not all(isinstance(frame, dict) for frame in frames):
        raise ValueError(f"every episode data entry must be an object: {episode_path}")
    return _validated_fps(info), frames, episode_path, info


def _require_metadata(info: dict[str, Any], allow_metadata_mismatch: bool) -> None:
    if allow_metadata_mismatch:
        return
    recording = info.get("recording") or {}
    robot = (recording.get("robot") or {}) if isinstance(recording, dict) else {}
    arm = robot.get("arm") if isinstance(robot, dict) else None
    end_effector = robot.get("end_effector") if isinstance(robot, dict) else None
    if str(arm).strip().upper() != "AI_WORKER" or str(end_effector).strip().lower() != "hx5_d20":
        raise ValueError(
            "episode metadata must identify recording.robot.arm=AI_WORKER and "
            "recording.robot.end_effector=hx5_d20; pass allow_metadata_mismatch=True "
            "only for a manually verified legacy episode "
            f"(got arm={arm!r}, end_effector={end_effector!r})"
        )


def _numeric_qpos(
    frame: dict[str, Any],
    frame_number: int,
    root_name: str,
    part: str,
    expected_dim: int,
    *,
    required: bool,
) -> np.ndarray | None:
    root = frame.get(root_name)
    if root is None:
        if required:
            raise ValueError(f"frame {frame_number}: missing {root_name}.{part}.qpos")
        return None
    if not isinstance(root, dict):
        raise ValueError(f"frame {frame_number}: {root_name} must be an object")
    part_data = root.get(part)
    if part_data is None:
        if required:
            raise ValueError(f"frame {frame_number}: missing {root_name}.{part}.qpos")
        return None
    if not isinstance(part_data, dict) or "qpos" not in part_data or part_data.get("qpos") is None:
        if required:
            raise ValueError(f"frame {frame_number}: missing {root_name}.{part}.qpos")
        return None
    raw_qpos = part_data["qpos"]
    if not required and isinstance(raw_qpos, (list, tuple)) and len(raw_qpos) == 0:
        return None
    try:
        qpos = np.asarray(raw_qpos, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"frame {frame_number}: {root_name}.{part}.qpos is not numeric") from exc
    if qpos.size != expected_dim:
        raise ValueError(
            f"frame {frame_number}: {root_name}.{part}.qpos length "
            f"{qpos.size} != {expected_dim}"
        )
    if not np.isfinite(qpos).all():
        raise ValueError(f"frame {frame_number}: {root_name}.{part}.qpos contains non-finite values")
    return qpos.copy()


def _check_bounds(
    qpos: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    frame_number: int,
    part: str,
) -> None:
    below = np.flatnonzero(qpos < lower)
    above = np.flatnonzero(qpos > upper)
    if below.size or above.size:
        indices = np.concatenate((below, above)).tolist()
        raise ValueError(
            f"frame {frame_number}: actions.{part}.qpos is outside joint bounds "
            f"at indices {indices}"
        )


def _optional_recorded_states(frame: dict[str, Any], frame_number: int) -> dict[str, np.ndarray]:
    recorded: dict[str, np.ndarray] = {}
    for part, dim in _PART_DIMS.items():
        qpos = _numeric_qpos(
            frame,
            frame_number,
            "states",
            part,
            dim,
            required=False,
        )
        if qpos is not None:
            recorded[part] = qpos
    return recorded


def preflight_ai_worker_episode(
    info: dict[str, Any],
    frames: list[dict[str, Any]],
    replay_arm: bool = True,
    replay_hand: bool = True,
    allow_metadata_mismatch: bool = False,
) -> list[AIWorkerReplayFrame]:
    """Validate all requested replay commands before a DDS publisher is made."""

    if not replay_arm and not replay_hand:
        raise ValueError("at least one of replay_arm or replay_hand must be enabled")
    if not isinstance(info, dict):
        raise ValueError("episode info must be an object")
    if not isinstance(frames, list) or not frames:
        raise ValueError("episode contains no frames")

    _validated_fps(info)
    _require_metadata(info, allow_metadata_mismatch)

    validated: list[AIWorkerReplayFrame] = []
    arm_lower = np.asarray(AI_WORKER_ARM_LOWER, dtype=np.float64).reshape(14)
    arm_upper = np.asarray(AI_WORKER_ARM_UPPER, dtype=np.float64).reshape(14)
    bounds = {
        "left_arm": (arm_lower[:7], arm_upper[:7]),
        "right_arm": (arm_lower[7:], arm_upper[7:]),
        "left_ee": (
            np.asarray(HX5_LEFT_LOWER, dtype=np.float64).reshape(20),
            np.asarray(HX5_LEFT_UPPER, dtype=np.float64).reshape(20),
        ),
        "right_ee": (
            np.asarray(HX5_RIGHT_LOWER, dtype=np.float64).reshape(20),
            np.asarray(HX5_RIGHT_UPPER, dtype=np.float64).reshape(20),
        ),
    }

    for frame_number, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"frame {frame_number}: frame must be an object")

        actions: dict[str, np.ndarray | None] = {
            "left_arm": None,
            "right_arm": None,
            "left_ee": None,
            "right_ee": None,
        }
        requested_parts = []
        if replay_arm:
            requested_parts.extend(("left_arm", "right_arm"))
        if replay_hand:
            requested_parts.extend(("left_ee", "right_ee"))
        for part in requested_parts:
            qpos = _numeric_qpos(
                frame,
                frame_number,
                "actions",
                part,
                _PART_DIMS[part],
                required=True,
            )
            assert qpos is not None
            _check_bounds(qpos, *bounds[part], frame_number, part)
            actions[part] = qpos

        raw_idx = frame.get("idx", frame_number)
        if isinstance(raw_idx, bool):
            raise ValueError(f"frame {frame_number}: idx must be an integer")
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"frame {frame_number}: idx must be an integer") from exc

        validated.append(
            AIWorkerReplayFrame(
                idx=idx,
                left_arm_action=actions["left_arm"],
                right_arm_action=actions["right_arm"],
                left_ee_action=actions["left_ee"],
                right_ee_action=actions["right_ee"],
                recorded_states=_optional_recorded_states(frame, frame_number),
                source_frame=frame,
            )
        )

    return validated
