#!/usr/bin/env bash
set -euo pipefail

# Evaluate one task on an ordered, disjoint reset-state window.
# Example: TASK_ID=6 RESET_START=10 NUM_ENVS=10 ... bash "$0"

: "${RLINF_DIR:?Set RLINF_DIR to the RLinf checkout}"
: "${PROJECT_DIR:?Set PROJECT_DIR to this project checkout}"
: "${MODEL_PATH:?Set MODEL_PATH to a π0.5 checkpoint directory}"
: "${TASK_ID:?Set TASK_ID (0-9 for LIBERO-Spatial)}"
: "${RESET_START:?Set RESET_START (e.g. 0 or 10)}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for this evaluation log}"

NUM_ENVS="${NUM_ENVS:-10}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"

bash "$PROJECT_DIR/scripts/apply_rlinf_eval_offset_patch.sh"

cd "$RLINF_DIR"
source "$RLINF_DIR/.venv/bin/activate"

bash evaluations/run_eval.sh libero_spatial_openpi_pi05_eval \
  rollout.model.model_path="$MODEL_PATH" \
  runner.logger.log_path="$OUTPUT_DIR" \
  env.eval.total_num_envs="$NUM_ENVS" \
  +env.eval.task_id_filter="[$TASK_ID]" \
  +env.eval.eval_reset_start_idx="$RESET_START" \
  env.eval.video_cfg.save_video=false \
  env.eval.max_steps_per_rollout_epoch=240 \
  env.eval.max_episode_steps=240
