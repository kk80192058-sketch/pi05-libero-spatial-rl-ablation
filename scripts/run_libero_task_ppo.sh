#!/usr/bin/env bash
set -euo pipefail

# PPO for one selected LIBERO-Spatial task; used for A1 (base+PPO) and C20
# (20-demo SFT+PPO).  Keeping task_id and seed explicit makes runs comparable.
: "${RLINF_DIR:?Set RLINF_DIR to the RLinf checkout}"
: "${MODEL_PATH:?Set MODEL_PATH to a policy checkpoint directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for checkpoints and TensorBoard logs}"
: "${TASK_ID:?Set TASK_ID to the LIBERO-Spatial task id}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-task${TASK_ID}_pi05_ppo}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
VAL_INTERVAL="${VAL_INTERVAL:-5}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
TRAIN_ENVS="${TRAIN_ENVS:-8}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
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
  runner.logger.experiment_name="$EXPERIMENT_NAME" \
  runner.max_epochs="$MAX_EPOCHS" \
  runner.val_check_interval="$VAL_INTERVAL" \
  runner.save_interval="$SAVE_INTERVAL" \
  env.train.total_num_envs="$TRAIN_ENVS" \
  env.train.rollout_epoch=1 \
  +env.train.task_id_filter="[$TASK_ID]" \
  env.train.video_cfg.save_video=false \
  env.eval.total_num_envs=10 \
  +env.eval.task_id_filter="[$TASK_ID]" \
  env.eval.video_cfg.save_video=false \
  actor.micro_batch_size="$TRAIN_ENVS" \
  actor.global_batch_size="$TRAIN_ENVS"

# PPO checkpoints contain learned weights but not OpenPI's immutable action
# normalization assets. Keep every saved actor checkpoint independently
# evaluable and usable as a later initialization (for example C20).
norm_source="$MODEL_PATH/physical-intelligence"
[[ -d "$norm_source/libero" ]] || { echo "Missing $norm_source/libero" >&2; exit 3; }
while IFS= read -r actor_dir; do
  [[ -d "$actor_dir/physical-intelligence/libero" ]] || cp -a "$norm_source" "$actor_dir/"
done < <(find "$OUTPUT_DIR" -type d -path '*/checkpoints/global_step_*/actor' | sort)
