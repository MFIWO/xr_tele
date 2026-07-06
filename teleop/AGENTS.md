# Repository Guidelines

## Project Structure & Module Organization
This repository is a Python teleoperation workspace centered on top-level runtime scripts such as `teleop_hand_and_arm.py`, `h1_2_replay.py`, and robot-specific debug helpers. Core logic lives in `robot_control/` and `utils/`. Two installable packages are vendored in-tree: `teleimager/` for camera streaming and `televuer/` for XR display and input. Hand retargeting code lives under `robot_control/dex-retargeting/`. Treat `backUp/`, `__pycache__/`, recorded data, and copied scripts like `teleop_hand_and_arm copy.py` as non-source artifacts unless a task explicitly targets them.

## Build, Test, and Development Commands
Use Python 3.10 unless a package states otherwise.

- `pip install -e teleimager` installs the camera client/server package.
- `pip install -e "teleimager[server]"` adds server-side camera dependencies.
- `pip install -e televuer` installs the XR interface package.
- `pip install -e robot_control/dex-retargeting` installs the retargeting library.
- `python teleop_hand_and_arm.py` runs the main teleoperation entry point.
- `python -m teleimager.image_server --cf` lists detected cameras and formats.
- `python -m teleimager.image_client --host 127.0.0.1` connects to a running image server.
- `python televuer/example/test_televuer.py` or `python televuer/example/test_tv_wrapper.py` runs XR integration examples.

## Coding Style & Naming Conventions
Follow existing Python conventions: 4-space indentation, `snake_case` for functions and modules, `PascalCase` for classes, and uppercase names for state flags or constants. Prefer explicit imports, small helper functions, and descriptive script names tied to hardware or workflow, for example `debug_rh5dg2_retarget.py`. Keep comments short and operational. No formatter or linter is configured in this repo, so match surrounding style closely when editing.

## Testing Guidelines
This repo uses runnable integration scripts more than isolated unit tests. Add new checks near the affected package using `test_*.py` naming. Keep hardware assumptions explicit in the script header or argument help. Before opening a PR, run the relevant example or utility test locally, especially for camera, XR, audio, or robot-control changes.

## Commit & Pull Request Guidelines
Recent history uses short imperative subjects such as `Add RH5DG2 teleoperation support`, `fix: pass head_img.bgr...`, and `[support] H2.`. Keep commit titles concise, present-tense, and focused on one change. PRs should include the target robot or subsystem, a brief behavior summary, local test commands, linked issues, and screenshots or logs when changing XR, camera, or UI-facing behavior.

## Security & Configuration Tips
Do not commit certificates, private keys, recorded episodes, or machine-specific IPs. Prefer environment variables or `~/.config/xr_teleoperate/` for local cert configuration. Keep large generated files and device-specific calibration outputs out of commits unless they are intentional project assets.
