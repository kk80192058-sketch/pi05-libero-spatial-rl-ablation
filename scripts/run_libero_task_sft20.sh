#!/usr/bin/env bash
set -euo pipefail

# Reproducible 20-demo SFT for a selected LIBERO task.  The task identity is
# carried by DATA_PATH and EXPERIMENT_NAME; split.csv records the exact demos.
: "${RLINF_DIR:?Set RLINF_DIR to the RLinf checkout}"
: "${PROJECT_DIR:?Set PROJECT_DIR to this project checkout}"
: "${MODEL_PATH:?Set MODEL_PATH to the public pi0.5 SFT checkpoint}"
: "${DATA_PATH:?Set DATA_PATH to the fixed 20-demo LeRobot dataset}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for checkpoints and TensorBoard logs}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-task_sft20}"
MAX_STEPS="${MAX_STEPS:-500}"
SAVE_INTERVAL="${SAVE_INTERVAL:-250}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"

cd "$RLINF_DIR/examples/sft"
source "$RLINF_DIR/.venv/bin/activate"
export EMBODIED_PATH="$RLINF_DIR/examples/embodiment"
cp "$PROJECT_DIR/configs/libero_task_sft20_pi05.yaml" \
  "$RLINF_DIR/examples/sft/config/libero_task_sft20_pi05.yaml"

python train_vla_sft.py \
  --config-name libero_task_sft20_pi05 \
  actor.model.model_path="$MODEL_PATH" \
  data.train_data_paths="$DATA_PATH" \
  runner.logger.log_path="$OUTPUT_DIR" \
  runner.logger.experiment_name="$EXPERIMENT_NAME" \
  runner.max_steps="$MAX_STEPS" \
  runner.save_interval="$SAVE_INTERVAL"

# SFT checkpoints omit immutable OpenPI normalization assets needed at eval.
norm_source="$MODEL_PATH/physical-intelligence"
[[ -d "$norm_source/libero" ]] || { echo "Missing $norm_source/libero" >&2; exit 3; }
while IFS= read -r actor_dir; do
  cp -a "$norm_source" "$actor_dir/"
done < <(find "$OUTPUT_DIR" -type d -path '*/checkpoints/global_step_*/actor' | sort)
