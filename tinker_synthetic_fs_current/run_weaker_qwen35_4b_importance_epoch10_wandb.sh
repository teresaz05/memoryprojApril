#!/bin/bash
#SBATCH --job-name=synthfsW4IS
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --cpus-per-task=8
#SBATCH --mem=384G
#SBATCH --time=72:00:00
#SBATCH --output=/sailhome/teresaz/tinker_runs/slurm-%x-%j.out
#SBATCH --error=/sailhome/teresaz/tinker_runs/slurm-%x-%j.err

set -euo pipefail

cd /sailhome/teresaz/BrowseCompV2/exact_pipeline/tinker_synthetic_fs_current

missing=()
for key in TINKER_API_KEY GEMINI_API_KEY WANDB_API_KEY; do
  if [ -z "${!key:-}" ]; then
    missing+=("$key")
  fi
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "Missing required env vars: ${missing[*]}" >&2
  exit 2
fi

export TMPDIR=/tmp
export PIP_CACHE_DIR=/tmp/pip-cache-$USER
export PIP_PROGRESS_BAR=off
export WANDB_MODE=online
export WANDB_PROJECT=synthetic-fs-rl
export WANDB_CONSOLE=${WANDB_CONSOLE:-wrap}
export PYTHONUNBUFFERED=1
export MALLOC_ARENA_MAX=2

mkdir -p "$PIP_CACHE_DIR"
mkdir -p /sailhome/teresaz/tinker_runs/shell_logs

rm -rf /tmp/teresaz-tinker
/usr/bin/python3 -m venv /tmp/teresaz-tinker
source /tmp/teresaz-tinker/bin/activate

python -m ensurepip --upgrade
python -m pip install -U pip setuptools wheel
python -m pip install -v --prefer-binary -r requirements.txt

python -m py_compile synthetic_fs_env.py train_synthetic_fs_rl.py make_nonexcluded_eval50.py

python3 make_nonexcluded_eval50.py \
  --train-index ../tinker_fs_qa/train_q830_fs/index.jsonl \
  --old-eval-index ../tinker_fs_qa/train_q50_fs/index.jsonl \
  --excluded ../tinker_fs_qa/excluded100.jsonl \
  --out-dir ../tinker_fs_qa/train_q50_nonexcluded_fs

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3.5-4B}
MODEL_SLUG=$(echo "$MODEL_NAME" | tr '/[:upper:]' '-[:lower:]')
LORA_RANK=${LORA_RANK:-32}
LEARNING_RATE=${LEARNING_RATE:-4e-5}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-16}
GROUP_SIZE=${GROUP_SIZE:-4}
MAX_TURNS=${MAX_TURNS:-32}
ANSWERER_BACKEND=${ANSWERER_BACKEND:-gemini}
ANSWERER_MODEL=${ANSWERER_MODEL:-gemini-3.1-flash-lite-preview}
ANSWERER_BASE_URL=${ANSWERER_BASE_URL:-https://generativelanguage.googleapis.com/v1beta}
ANSWERER_API_KEY_ENV=${ANSWERER_API_KEY_ENV:-GEMINI_API_KEY}
ANSWERER_SLUG=$(echo "$ANSWERER_MODEL" | tr '/[:upper:]' '-[:lower:]')
BUILDER_EXECUTOR_BACKEND=${BUILDER_EXECUTOR_BACKEND:-openrouter}
BUILDER_EXECUTOR_MODEL=${BUILDER_EXECUTOR_MODEL:-llm}
BUILDER_EXECUTOR_BASE_URL=${BUILDER_EXECUTOR_BASE_URL:-https://iris-lab-ws--lateral-vllm-qwen35-4b.modal.run/v1}
BUILDER_EXECUTOR_API_KEY_ENV=${BUILDER_EXECUTOR_API_KEY_ENV:-}
BUILDER_EXECUTOR_MAX_OUTPUT_TOKENS=${BUILDER_EXECUTOR_MAX_OUTPUT_TOKENS:-512}
INDEX_JSONL=${INDEX_JSONL:-../tinker_fs_qa/train_q830_fs/index.jsonl}
EXCLUDED_QIDS_JSONL=${EXCLUDED_QIDS_JSONL:-../tinker_fs_qa/excluded100.jsonl}
EVAL_INDEX_JSONL=${EVAL_INDEX_JSONL:-../tinker_fs_qa/train_q50_nonexcluded_fs/index.jsonl}
TRAIN_ROWS=$(python - <<PY
import json
from pathlib import Path
index = Path("$INDEX_JSONL")
excluded_path = Path("$EXCLUDED_QIDS_JSONL") if "$EXCLUDED_QIDS_JSONL" else None
eval_path = Path("$EVAL_INDEX_JSONL") if "$EVAL_INDEX_JSONL" else None
rows = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
excluded = set()
if excluded_path and excluded_path.exists():
    for line in excluded_path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            excluded.add(str(row.get("question_id") or row.get("qid") or row.get("query_id")))
rows = [row for row in rows if str(row["question_id"]) not in excluded]
if eval_path and eval_path.exists():
    eval_qids = {str(json.loads(line)["question_id"]) for line in eval_path.read_text().splitlines() if line.strip()}
    rows = [row for row in rows if str(row["question_id"]) not in eval_qids]
print(len(rows))
PY
)
BATCHES_PER_EPOCH=$(( (TRAIN_ROWS + BATCH_SIZE - 1) / BATCH_SIZE ))
MAX_STEPS=${MAX_STEPS:-$((BATCHES_PER_EPOCH * TRAIN_EPOCHS))}
SAVE_EVERY=${SAVE_EVERY:-$BATCHES_PER_EPOCH}
ROLLING_SAVE_EVERY=${ROLLING_SAVE_EVERY:-10}
STREAM_MINIBATCH_GROUPS_PER_BATCH=${STREAM_MINIBATCH_GROUPS_PER_BATCH:-0}
STREAM_MINIBATCH_NUM_MINIBATCHES=${STREAM_MINIBATCH_NUM_MINIBATCHES:-0}
MAX_CONCURRENT_ROLLOUT_GROUPS=${MAX_CONCURRENT_ROLLOUT_GROUPS:-4}
FREE_ROLLOUT_AFTER_PREPARE=${FREE_ROLLOUT_AFTER_PREPARE:-False}
BASE_RUN_NAME=${BASE_RUN_NAME_OVERRIDE:-synthfs_${MODEL_SLUG}_is_answerability_epoch${TRAIN_EPOCHS}_rows${TRAIN_ROWS}_bs${BATCH_SIZE}_gs${GROUP_SIZE}_mt${MAX_TURNS}_ans32_ansrep4_trainans${ANSWERER_SLUG}_rank${LORA_RANK}_lr${LEARNING_RATE}_seed2}

if [ -n "${RESUME_RUN_DIR:-}" ]; then
  RUN_DIR="$RESUME_RUN_DIR"
  RUN_NAME="$(basename "$RUN_DIR")"
  LOGDIR_BEHAVIOR=resume
elif [ -n "${RESUME_RUN_NAME:-}" ]; then
  RUN_NAME="$RESUME_RUN_NAME"
  RUN_DIR=/sailhome/teresaz/tinker_runs/$RUN_NAME
  LOGDIR_BEHAVIOR=resume
else
  RUN_NAME=${BASE_RUN_NAME}_job${SLURM_JOB_ID}
  RUN_DIR=/sailhome/teresaz/tinker_runs/$RUN_NAME
  LOGDIR_BEHAVIOR=raise
fi
SHELL_LOG=/sailhome/teresaz/tinker_runs/shell_logs/${RUN_NAME}.train.log

if [ "$LOGDIR_BEHAVIOR" = "resume" ] && [ ! -f "$RUN_DIR/checkpoints.jsonl" ]; then
  echo "Requested resume, but no checkpoint file exists at $RUN_DIR/checkpoints.jsonl" >&2
  exit 3
fi

echo "RUN_NAME=$RUN_NAME"
echo "RUN_DIR=$RUN_DIR"
echo "SHELL_LOG=$SHELL_LOG"
echo "MODEL_NAME=$MODEL_NAME"
echo "LORA_RANK=$LORA_RANK"
echo "LEARNING_RATE=$LEARNING_RATE"
echo "TRAIN_EPOCHS=$TRAIN_EPOCHS"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "GROUP_SIZE=$GROUP_SIZE"
echo "MAX_TURNS=$MAX_TURNS"
echo "ANSWERER_BACKEND=$ANSWERER_BACKEND"
echo "ANSWERER_MODEL=$ANSWERER_MODEL"
echo "ANSWERER_BASE_URL=$ANSWERER_BASE_URL"
echo "ANSWERER_API_KEY_ENV=$ANSWERER_API_KEY_ENV"
echo "TRAIN_ROWS=$TRAIN_ROWS"
echo "BATCHES_PER_EPOCH=$BATCHES_PER_EPOCH"
echo "MAX_STEPS=$MAX_STEPS"
echo "SAVE_EVERY=$SAVE_EVERY"
echo "ROLLING_SAVE_EVERY=$ROLLING_SAVE_EVERY"
echo "STREAM_MINIBATCH_GROUPS_PER_BATCH=$STREAM_MINIBATCH_GROUPS_PER_BATCH"
echo "STREAM_MINIBATCH_NUM_MINIBATCHES=$STREAM_MINIBATCH_NUM_MINIBATCHES"
echo "MAX_CONCURRENT_ROLLOUT_GROUPS=$MAX_CONCURRENT_ROLLOUT_GROUPS"
echo "FREE_ROLLOUT_AFTER_PREPARE=$FREE_ROLLOUT_AFTER_PREPARE"
echo "BUILDER_EXECUTOR_BACKEND=$BUILDER_EXECUTOR_BACKEND"
echo "BUILDER_EXECUTOR_MODEL=$BUILDER_EXECUTOR_MODEL"
echo "BUILDER_EXECUTOR_BASE_URL=$BUILDER_EXECUTOR_BASE_URL"
echo "BUILDER_EXECUTOR_API_KEY_ENV=$BUILDER_EXECUTOR_API_KEY_ENV"
echo "BUILDER_EXECUTOR_MAX_OUTPUT_TOKENS=$BUILDER_EXECUTOR_MAX_OUTPUT_TOKENS"
echo "INDEX_JSONL=$INDEX_JSONL"
echo "EXCLUDED_QIDS_JSONL=$EXCLUDED_QIDS_JSONL"
echo "EVAL_INDEX_JSONL=$EVAL_INDEX_JSONL"
echo "LOGDIR_BEHAVIOR=$LOGDIR_BEHAVIOR"
if [ -f "$RUN_DIR/checkpoints.jsonl" ]; then
  echo "Last checkpoints:"
  tail -5 "$RUN_DIR/checkpoints.jsonl"
fi

EXTRA_ARGS=()
if [ -n "${TRAIN_EXTRA_ARGS:-}" ]; then
  read -r -a EXTRA_ARGS <<< "$TRAIN_EXTRA_ARGS"
fi
echo "TRAIN_EXTRA_ARGS=${TRAIN_EXTRA_ARGS:-}"

python train_synthetic_fs_rl.py \
  model_name="$MODEL_NAME" \
  lora_rank="$LORA_RANK" \
  learning_rate="$LEARNING_RATE" \
  batch_size="$BATCH_SIZE" \
  group_size="$GROUP_SIZE" \
  train_epochs="$TRAIN_EPOCHS" \
  reshuffle_each_epoch=True \
  max_steps="$MAX_STEPS" \
  stream_minibatch_groups_per_batch="$STREAM_MINIBATCH_GROUPS_PER_BATCH" \
  stream_minibatch_num_minibatches="$STREAM_MINIBATCH_NUM_MINIBATCHES" \
  max_concurrent_rollout_groups="$MAX_CONCURRENT_ROLLOUT_GROUPS" \
  free_rollout_after_prepare_minibatch="$FREE_ROLLOUT_AFTER_PREPARE" \
  index_jsonl="$INDEX_JSONL" \
  excluded_qids_jsonl="$EXCLUDED_QIDS_JSONL" \
  eval_index_jsonl="$EVAL_INDEX_JSONL" \
  max_turns="$MAX_TURNS" \
  loss_fn=importance_sampling \
  log_rl_diagnostics=True \
  builder_compaction_trigger_tokens=3000 \
  answerer_backend="$ANSWERER_BACKEND" \
  answerer_model="$ANSWERER_MODEL" \
  answerer_base_url="$ANSWERER_BASE_URL" \
  answerer_api_key_env="$ANSWERER_API_KEY_ENV" \
  judge_model=gemini-3.1-flash-lite-preview \
  builder_compaction_model=gemini-3.1-flash-lite-preview \
  builder_executor_backend="$BUILDER_EXECUTOR_BACKEND" \
  builder_executor_model="$BUILDER_EXECUTOR_MODEL" \
  builder_executor_base_url="$BUILDER_EXECUTOR_BASE_URL" \
  builder_executor_api_key_env="$BUILDER_EXECUTOR_API_KEY_ENV" \
  builder_executor_max_output_tokens="$BUILDER_EXECUTOR_MAX_OUTPUT_TOKENS" \
  reward_mode=hybrid \
  terminal_answerer_repeats=4 \
  answerability_delta_reward_scale=1.0 \
  answerability_probe_repeats=4 \
  answerability_probe_max_per_episode=4 \
  answerability_probe_interval_turns=8 \
  filesystem_maturity_scale=0.0 \
  step_filesystem_maturity_delta_scale=0.0 \
  step_construction_action_bonus=0.0 \
  step_non_construction_turn_penalty=0.0 \
  step_non_construction_streak_penalty=0.0 \
  step_tool_error_penalty=0.0 \
  termination_penalty=0.0 \
  empty_synthetic_penalty=0.0 \
  answerer_retrieval_cost_scale=0.0 \
  answerer_synthetic_read_cost_scale=0.0 \
  synthetic_success_bonus=0.0 \
  synthetic_usage_bonus=0.0 \
  raw_usage_ratio_penalty=0.0 \
  mature_stop_bonus=0.0 \
  save_every="$SAVE_EVERY" \
  rolling_save_every="$ROLLING_SAVE_EVERY" \
  rolling_ttl_seconds=604800 \
  log_path="$RUN_DIR" \
  wandb_project="$WANDB_PROJECT" \
  wandb_name="$RUN_NAME" \
  behavior_if_log_dir_exists="$LOGDIR_BEHAVIOR" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "$SHELL_LOG"
