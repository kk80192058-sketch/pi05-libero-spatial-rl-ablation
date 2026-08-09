#!/usr/bin/env bash
set -euo pipefail

# Few-shot SFT: adapt the public π0.5 LIBERO checkpoint to 20 fixed task-6
# demonstrations.  The result is group B20 in the experiment report.
#
# Example:
# RLINF_DIR=/path/to/RLinf PROJECT_DIR=/path/to/project \
# MODEL_PATH=/path/to/RLinf-Pi05-LIBERO-SFT \
# DATA_PATH=/path/to/libero_spatial_task6_train20 \
# OUTPUT_DIR=/path/to/output bash scripts/run_task6_sft20.sh

: "${RLINF_DIR:?Set RLINF_DIR to the RLinf checkout}"
: "${PROJECT_DIR:?Set PROJECT_DIR to this project checkout}"
: "${MODEL_PATH:?Set MODEL_PATH to the public π0.5 SFT checkpoint}"
: "${DATA_PATH:?Set DATA_PATH to the fixed 20-demo LeRobot dataset}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for checkpoints and TensorBoard logs}"

MAX_STEPS="${MAX_STEPS:-500}"
SAVE_INTERVAL="${SAVE_INTERVAL:-250}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"

cd "$RLINF_DIR/examples/sft"
source "$RLINF_DIR/.venv/bin/activate"
export EMBODIED_PATH="$RLINF_DIR/examples/embodiment"

# Keep the editable recipe in this repository, but install a runtime copy next
# to RLinf's model config groups so Hydra can resolve model/pi0_5.
cp "$PROJECT_DIR/configs/libero_task6_sft20_pi05.yaml" \
  "$RLINF_DIR/examples/sft/config/libero_task6_sft20_pi05.yaml"

python train_vla_sft.py \
  --config-name libero_task6_sft20_pi05 \
  actor.model.model_path="$MODEL_PATH" \
  data.train_data_paths="$DATA_PATH" \
  runner.logger.log_path="$OUTPUT_DIR" \
  runner.max_steps="$MAX_STEPS" \
  runner.save_interval="$SAVE_INTERVAL"
