#!/usr/bin/env bash
set -euo pipefail

# Formal pre-/post-PPO comparison protocol for LIBERO-Spatial task 6.
# Usage:
# RLINF_DIR=/path/to/RLinf MODEL_PATH=/path/to/checkpoint \
#   bash scripts/run_task6_baseline.sh

: "${RLINF_DIR:?Set RLINF_DIR to the RLinf checkout}"
: "${MODEL_PATH:?Set MODEL_PATH to the checkpoint directory}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"

cd "$RLINF_DIR"
bash evaluations/run_eval.sh libero_spatial_openpi_pi05_eval \
  rollout.model.model_path="$MODEL_PATH" \
  env.eval.total_num_envs=10 \
  +env.eval.task_id_filter=[6] \
  env.eval.video_cfg.save_video=false \
  env.eval.max_steps_per_rollout_epoch=240 \
  env.eval.max_episode_steps=240
