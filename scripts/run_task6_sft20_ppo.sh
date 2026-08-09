#!/usr/bin/env bash
set -euo pipefail

# C20: online PPO starting from the B20 few-shot-SFT checkpoint.
# SFT_CHECKPOINT must point to the checkpoint's `actor` directory, e.g.
#   .../task6_sft20/checkpoints/global_step_500/actor

: "${SFT_CHECKPOINT:?Set SFT_CHECKPOINT to the B20 checkpoint actor directory}"
: "${PROJECT_DIR:?Set PROJECT_DIR to this project checkout}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for the C20 PPO result}"

if [[ ! -d "$SFT_CHECKPOINT" ]]; then
  echo "SFT_CHECKPOINT does not exist: $SFT_CHECKPOINT" >&2
  exit 2
fi

export RLINF_DIR="${RLINF_DIR:?Set RLINF_DIR to the RLinf checkout}"
export MODEL_PATH="$SFT_CHECKPOINT"
export MAX_EPOCHS="${MAX_EPOCHS:-10}"
export VAL_INTERVAL="${VAL_INTERVAL:-5}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
export TRAIN_ENVS="${TRAIN_ENVS:-8}"

exec bash "$PROJECT_DIR/scripts/run_task6_ppo.sh"
