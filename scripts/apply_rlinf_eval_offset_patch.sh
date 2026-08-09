#!/usr/bin/env bash
set -euo pipefail

: "${RLINF_DIR:?Set RLINF_DIR to the RLinf checkout}"
: "${PROJECT_DIR:?Set PROJECT_DIR to this project checkout}"

target="$RLINF_DIR/rlinf/envs/libero/libero_env.py"
patch_file="$PROJECT_DIR/patches/rlinf_eval_reset_offset.patch"

if grep -q 'eval_reset_start_idx' "$target"; then
  echo "RLinf eval offset patch already present"
  exit 0
fi

git -C "$RLINF_DIR" apply --check "$patch_file"
git -C "$RLINF_DIR" apply "$patch_file"
echo "Applied RLinf eval offset patch"
