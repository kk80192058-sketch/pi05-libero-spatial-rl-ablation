#!/usr/bin/env bash
set -euo pipefail

# Minimal PPO run for the selected LIBERO-Spatial task 6.
# This is intended for an A800 80GB instance after the 4090 setup is copied.
# Usage:
# RLINF_DIR=/path/to/RLinf MODEL_PATH=/path/to/RLinf-Pi05-LIBERO-SFT \
#   OUTPUT_DIR=/path/to/output bash scripts/run_task6_ppo.sh

: "${RLINF_DIR:?Set RLINF_DIR to the RLinf checkout}"
: "${MODEL_PATH:?Set MODEL_PATH to the π0.5 SFT checkpoint directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for checkpoints and TensorBoard logs}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"

cd "$RLINF_DIR/examples/embodiment"
source "$RLINF_DIR/.venv/bin/activate"
export EMBODIED_PATH="$PWD"

python train_embodied_agent.py \
  --config-name libero_spatial_ppo_openpi_pi05 \
  actor.model.model_path="$MODEL_PATH" \
  rollout.model.model_path="$MODEL_PATH" \
  runner.logger.log_path="$OUTPUT_DIR" \
  runner.logger.experiment_name=task6_pi05_ppo \
  runner.max_epochs=20 \
  runner.val_check_interval=5 \
  runner.save_interval=5 \
  env.train.total_num_envs=8 \
  env.train.rollout_epoch=1 \
  +env.train.task_id_filter=[6] \
  env.train.video_cfg.save_video=false \
  env.eval.total_num_envs=10 \
  +env.eval.task_id_filter=[6] \
  env.eval.video_cfg.save_video=false \
  actor.micro_batch_size=8 \
  actor.global_batch_size=8
