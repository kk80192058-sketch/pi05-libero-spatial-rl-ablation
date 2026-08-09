#!/usr/bin/env bash
set -euo pipefail

# Usage:
# RLINF_DIR=/path/to/RLinf MODEL_PATH=/path/to/RLinf-Pi05-LIBERO-SFT \
#   bash scripts/run_single_rollout.sh

: "${RLINF_DIR:?Set RLINF_DIR to the RLinf checkout}"
: "${MODEL_PATH:?Set MODEL_PATH to the π0.5 checkpoint directory}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# OSMesa gives valid RGB frames in the current cloud container.
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"

cd "$RLINF_DIR"
bash evaluations/run_eval.sh libero_spatial_openpi_pi05_eval \
  rollout.model.model_path="$MODEL_PATH" \
  env.eval.total_num_envs=1 \
  env.eval.specific_reset_id=0 \
  env.eval.max_steps_per_rollout_epoch=240 \
  env.eval.max_episode_steps=240
