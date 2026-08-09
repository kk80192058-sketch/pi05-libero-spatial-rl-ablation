#!/usr/bin/env bash
set -euo pipefail

# Systematically score all 10 LIBERO-Spatial tasks on reset states 0-9.
# Each task runs separately so its metric stays attributable to that task.

: "${RLINF_DIR:?Set RLINF_DIR to the RLinf checkout}"
: "${PROJECT_DIR:?Set PROJECT_DIR to this project checkout}"
: "${MODEL_PATH:?Set MODEL_PATH to the π0.5 checkpoint to diagnose}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for task-specific logs}"

mkdir -p "$OUTPUT_DIR"

for task_id in $(seq 0 9); do
  task_dir="$OUTPUT_DIR/task_${task_id}_resets_0_9"
  mkdir -p "$task_dir"
  echo "=== task ${task_id}: reset window 0-9 ==="
  TASK_ID="$task_id" \
  RESET_START=0 \
  NUM_ENVS=10 \
  RLINF_DIR="$RLINF_DIR" \
  PROJECT_DIR="$PROJECT_DIR" \
  MODEL_PATH="$MODEL_PATH" \
  OUTPUT_DIR="$task_dir" \
  bash "$PROJECT_DIR/scripts/run_libero_task_window_eval.sh" \
    2>&1 | tee "$task_dir/console.log"
done
