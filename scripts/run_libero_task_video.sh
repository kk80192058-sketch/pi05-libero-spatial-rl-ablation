#!/usr/bin/env bash
set -euo pipefail

# Record one complete LIBERO task rollout with EGL GPU rendering.  EGL avoids
# the green frames produced by the container's OSMesa software renderer.
: "${RLINF_DIR:?Set RLINF_DIR to the RLinf checkout}"
: "${MODEL_PATH:?Set MODEL_PATH to a directly evaluable checkpoint}"
: "${TASK_ID:?Set TASK_ID (0-9 for LIBERO-Spatial)}"
: "${RESET_START:?Set RESET_START to the ordered reset state id}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for log and MP4}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export EMBODIED_PATH="$RLINF_DIR/examples/embodiment"

mkdir -p "$OUTPUT_DIR"
source "$RLINF_DIR/.venv/bin/activate"
python "$RLINF_DIR/evaluations/eval_embodied_agent.py" \
  --config-path "$RLINF_DIR/evaluations/libero" \
  --config-name libero_spatial_openpi_pi05_eval \
  runner.logger.log_path="$OUTPUT_DIR" \
  rollout.model.model_path="$MODEL_PATH" \
  env.eval.total_num_envs=1 \
  +env.eval.task_id_filter="[$TASK_ID]" \
  +env.eval.eval_reset_start_idx="$RESET_START" \
  env.eval.video_cfg.save_video=true \
  env.eval.video_cfg.video_base_dir="$OUTPUT_DIR/video" \
  env.eval.max_steps_per_rollout_epoch=240 \
  env.eval.max_episode_steps=240
