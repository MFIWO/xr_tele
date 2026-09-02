#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

apply_submodule_patch() {
    local module_path="$1"
    local patch_path="$2"

    if git -C "$repo_root/$module_path" apply --reverse --check "$repo_root/$patch_path" >/dev/null 2>&1; then
        printf 'Already applied: %s\n' "$module_path"
        return
    fi

    git -C "$repo_root/$module_path" apply --check "$repo_root/$patch_path"
    git -C "$repo_root/$module_path" apply "$repo_root/$patch_path"
    printf 'Applied: %s\n' "$module_path"
}

git -C "$repo_root" submodule update --init
apply_submodule_patch "teleop/robot_control/dex-retargeting" "patches/dex-retargeting.patch"
apply_submodule_patch "teleop/teleimager" "patches/teleimager.patch"
apply_submodule_patch "teleop/televuer" "patches/televuer.patch"
