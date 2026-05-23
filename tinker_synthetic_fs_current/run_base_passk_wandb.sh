#!/bin/bash
# Base/fine-tuned trajectory-level pass@k diagnostic.
# If SOURCE_RUN_DIR or SOURCE_RUN_NAME is set, this loads that run's final
# checkpoint. Otherwise, it evaluates the base model. It samples k builder
# trajectories per held-out question. Per-trajectory rollout JSON export is
# controlled by PASSK_ROLLOUT_JSON_EXPORT.
# Later pass@8 can be computed by combining this run with another independent
# k=4 run using a different seed/run.
#SBATCH --job-name=synthfsBaseK
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --cpus-per-task=8
#SBATCH --mem=384G
#SBATCH --time=24:00:00
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

python -m py_compile synthetic_fs_env.py train_synthetic_fs_rl.py make_nonexcluded_eval50.py analyze_passk_rollouts.py

python3 make_nonexcluded_eval50.py \
  --train-index ../tinker_fs_qa/train_q830_fs/index.jsonl \
  --old-eval-index ../tinker_fs_qa/train_q50_fs/index.jsonl \
  --excluded ../tinker_fs_qa/excluded100.jsonl \
  --out-dir ../tinker_fs_qa/train_q50_nonexcluded_fs

PASSK_K=${PASSK_K:-4}
PASSK_SEED=${PASSK_SEED:-2}
PASSK_BATCH_SIZE=${PASSK_BATCH_SIZE:-16}
PASSK_ANSWERER_REPEATS=${PASSK_ANSWERER_REPEATS:-1}
PASSK_EVAL_INDEX_JSONL=${PASSK_EVAL_INDEX_JSONL:-../tinker_fs_qa/train_q50_nonexcluded_fs/index.jsonl}
PASSK_NUM_GROUPS_TO_LOG=${PASSK_NUM_GROUPS_TO_LOG:-50}
PASSK_ROLLOUT_JSON_EXPORT=${PASSK_ROLLOUT_JSON_EXPORT:-True}
PASSK_FREE_ROLLOUT_AFTER_PREPARE=${PASSK_FREE_ROLLOUT_AFTER_PREPARE:-False}
PASSK_DISABLE_SAMPLE_TRAJECTORY_PRINTING=${PASSK_DISABLE_SAMPLE_TRAJECTORY_PRINTING:-True}
PASSK_STREAM_MINIBATCH_GROUPS_PER_BATCH=${PASSK_STREAM_MINIBATCH_GROUPS_PER_BATCH:-0}
PASSK_STREAM_MINIBATCH_NUM_MINIBATCHES=${PASSK_STREAM_MINIBATCH_NUM_MINIBATCHES:-0}
PASSK_MAX_CONCURRENT_ROLLOUT_GROUPS=${PASSK_MAX_CONCURRENT_ROLLOUT_GROUPS:-4}
PASSK_MODEL_NAME=${PASSK_MODEL_NAME:-Qwen/Qwen3.5-4B}
PASSK_MODEL_SLUG=$(echo "$PASSK_MODEL_NAME" | tr '/[:upper:]' '-[:lower:]')
PASSK_ANSWERER_BACKEND=${PASSK_ANSWERER_BACKEND:-openrouter}
PASSK_ANSWERER_MODEL=${PASSK_ANSWERER_MODEL:-llm}
PASSK_ANSWERER_BASE_URL=${PASSK_ANSWERER_BASE_URL:-https://iris-lab-ws--lateral-vllm-qwen35-4b.modal.run/v1}
PASSK_ANSWERER_API_KEY_ENV=${PASSK_ANSWERER_API_KEY_ENV:-}
PASSK_ANSWERER_SLUG=$(echo "$PASSK_ANSWERER_MODEL" | tr '/[:upper:]' '-[:lower:]')
BUILDER_EXECUTOR_BACKEND=${BUILDER_EXECUTOR_BACKEND:-openrouter}
BUILDER_EXECUTOR_MODEL=${BUILDER_EXECUTOR_MODEL:-llm}
BUILDER_EXECUTOR_BASE_URL=${BUILDER_EXECUTOR_BASE_URL:-https://iris-lab-ws--lateral-vllm-qwen35-4b.modal.run/v1}
BUILDER_EXECUTOR_API_KEY_ENV=${BUILDER_EXECUTOR_API_KEY_ENV:-}
BUILDER_EXECUTOR_MAX_OUTPUT_TOKENS=${BUILDER_EXECUTOR_MAX_OUTPUT_TOKENS:-512}

LOAD_ARGS=()
SOURCE_LABEL=base
if [ -n "${SOURCE_RUN_DIR:-}" ] || [ -n "${SOURCE_RUN_NAME:-}" ]; then
  if [ -n "${SOURCE_RUN_DIR:-}" ]; then
    SOURCE_DIR="$SOURCE_RUN_DIR"
  else
    SOURCE_DIR=/sailhome/teresaz/tinker_runs/$SOURCE_RUN_NAME
  fi
  if [ ! -f "$SOURCE_DIR/checkpoints.jsonl" ]; then
    echo "No checkpoints.jsonl found at $SOURCE_DIR/checkpoints.jsonl" >&2
    exit 3
  fi
  LATEST_CKPT=$(SOURCE_DIR="$SOURCE_DIR" python3 - <<'PY'
import json, os, pathlib, sys
path = pathlib.Path(os.environ["SOURCE_DIR"]) / "checkpoints.jsonl"
rows = [json.loads(line) for line in path.open() if line.strip()]
if not rows:
    raise SystemExit("checkpoints.jsonl is empty")
final_rows = [row for row in rows if row.get("name") == "final"]
last = final_rows[-1] if final_rows else rows[-1]
for key in (
    "state_path",
    "checkpoint_path",
    "load_checkpoint_path",
    "path",
    "checkpoint",
    "model_path",
    "sampling_model_path",
    "sampler_path",
):
    value = last.get(key)
    if isinstance(value, str) and value.strip():
        print(value.strip())
        break
else:
    print(json.dumps(last, indent=2), file=sys.stderr)
    raise SystemExit("Could not infer checkpoint path from latest checkpoints.jsonl row")
PY
)
  LOAD_ARGS=(load_checkpoint_path="$LATEST_CKPT")
  SOURCE_LABEL=ft_$(basename "$SOURCE_DIR")
else
  SOURCE_DIR=""
  LATEST_CKPT=""
fi

BASE_RUN_NAME=synthfs_passk_${SOURCE_LABEL}_${PASSK_MODEL_SLUG}_eval50_k${PASSK_K}_bs${PASSK_BATCH_SIZE}_mt32_ans32_ansrep${PASSK_ANSWERER_REPEATS}_ans${PASSK_ANSWERER_SLUG}_seed${PASSK_SEED}
RUN_NAME=${BASE_RUN_NAME}_job${SLURM_JOB_ID}
RUN_DIR=/sailhome/teresaz/tinker_runs/$RUN_NAME
SHELL_LOG=/sailhome/teresaz/tinker_runs/shell_logs/${RUN_NAME}.passk.log

cat <<INFO
SOURCE_DIR=$SOURCE_DIR
LATEST_CKPT=$LATEST_CKPT
RUN_NAME=$RUN_NAME
RUN_DIR=$RUN_DIR
SHELL_LOG=$SHELL_LOG
PASSK_K=$PASSK_K
PASSK_SEED=$PASSK_SEED
PASSK_BATCH_SIZE=$PASSK_BATCH_SIZE
PASSK_ANSWERER_REPEATS=$PASSK_ANSWERER_REPEATS
PASSK_EVAL_INDEX_JSONL=$PASSK_EVAL_INDEX_JSONL
PASSK_NUM_GROUPS_TO_LOG=$PASSK_NUM_GROUPS_TO_LOG
PASSK_ROLLOUT_JSON_EXPORT=$PASSK_ROLLOUT_JSON_EXPORT
PASSK_FREE_ROLLOUT_AFTER_PREPARE=$PASSK_FREE_ROLLOUT_AFTER_PREPARE
PASSK_DISABLE_SAMPLE_TRAJECTORY_PRINTING=$PASSK_DISABLE_SAMPLE_TRAJECTORY_PRINTING
PASSK_STREAM_MINIBATCH_GROUPS_PER_BATCH=$PASSK_STREAM_MINIBATCH_GROUPS_PER_BATCH
PASSK_STREAM_MINIBATCH_NUM_MINIBATCHES=$PASSK_STREAM_MINIBATCH_NUM_MINIBATCHES
PASSK_MAX_CONCURRENT_ROLLOUT_GROUPS=$PASSK_MAX_CONCURRENT_ROLLOUT_GROUPS
PASSK_MODEL_NAME=$PASSK_MODEL_NAME
PASSK_ANSWERER_BACKEND=$PASSK_ANSWERER_BACKEND
PASSK_ANSWERER_MODEL=$PASSK_ANSWERER_MODEL
PASSK_ANSWERER_BASE_URL=$PASSK_ANSWERER_BASE_URL
PASSK_ANSWERER_API_KEY_ENV=$PASSK_ANSWERER_API_KEY_ENV
BUILDER_EXECUTOR_BACKEND=$BUILDER_EXECUTOR_BACKEND
BUILDER_EXECUTOR_MODEL=$BUILDER_EXECUTOR_MODEL
BUILDER_EXECUTOR_BASE_URL=$BUILDER_EXECUTOR_BASE_URL
BUILDER_EXECUTOR_API_KEY_ENV=$BUILDER_EXECUTOR_API_KEY_ENV
BUILDER_EXECUTOR_MAX_OUTPUT_TOKENS=$BUILDER_EXECUTOR_MAX_OUTPUT_TOKENS
INFO

# max_steps=1 with learning_rate=0 performs one no-op update and triggers
# eval_every=1 evaluation. With SOURCE_RUN_* set, it first loads that checkpoint.
python train_synthetic_fs_rl.py \
  model_name="$PASSK_MODEL_NAME" \
  batch_size="$PASSK_BATCH_SIZE" \
  group_size="$PASSK_K" \
  seed="$PASSK_SEED" \
  max_turns=32 \
  learning_rate=0.0 \
  max_steps=1 \
  stream_minibatch_groups_per_batch="$PASSK_STREAM_MINIBATCH_GROUPS_PER_BATCH" \
  stream_minibatch_num_minibatches="$PASSK_STREAM_MINIBATCH_NUM_MINIBATCHES" \
  max_concurrent_rollout_groups="$PASSK_MAX_CONCURRENT_ROLLOUT_GROUPS" \
  eval_every=1 \
  builder_compaction_trigger_tokens=3000 \
  answerer_backend="$PASSK_ANSWERER_BACKEND" \
  answerer_model="$PASSK_ANSWERER_MODEL" \
  answerer_base_url="$PASSK_ANSWERER_BASE_URL" \
  answerer_api_key_env="$PASSK_ANSWERER_API_KEY_ENV" \
  judge_model=gemini-3.1-flash-lite-preview \
  builder_compaction_model=gemini-3.1-flash-lite-preview \
  builder_executor_backend="$BUILDER_EXECUTOR_BACKEND" \
  builder_executor_model="$BUILDER_EXECUTOR_MODEL" \
  builder_executor_base_url="$BUILDER_EXECUTOR_BASE_URL" \
  builder_executor_api_key_env="$BUILDER_EXECUTOR_API_KEY_ENV" \
  builder_executor_max_output_tokens="$BUILDER_EXECUTOR_MAX_OUTPUT_TOKENS" \
  reward_mode=hybrid \
  terminal_answerer_repeats="$PASSK_ANSWERER_REPEATS" \
  answerability_delta_reward_scale=0.0 \
  eval_index_jsonl="$PASSK_EVAL_INDEX_JSONL" \
  num_groups_to_log="$PASSK_NUM_GROUPS_TO_LOG" \
  rollout_json_export="$PASSK_ROLLOUT_JSON_EXPORT" \
  free_rollout_after_prepare_minibatch="$PASSK_FREE_ROLLOUT_AFTER_PREPARE" \
  disable_sample_trajectory_printing="$PASSK_DISABLE_SAMPLE_TRAJECTORY_PRINTING" \
  save_every=999999 \
  rolling_save_every=999999 \
  log_path="$RUN_DIR" \
  wandb_project="$WANDB_PROJECT" \
  wandb_name="$RUN_NAME" \
  behavior_if_log_dir_exists=raise \
  "${LOAD_ARGS[@]}" \
  2>&1 | tee -a "$SHELL_LOG"

case "${PASSK_ROLLOUT_JSON_EXPORT,,}" in
  1|true|yes|on)
    python analyze_passk_rollouts.py \
      "$RUN_DIR" \
      --k "$PASSK_K" \
      --eval-index-jsonl "$PASSK_EVAL_INDEX_JSONL" \
      --out-dir "$RUN_DIR" \
      2>&1 | tee -a "$SHELL_LOG"
    ;;
  *)
    echo "Skipping analyze_passk_rollouts.py because PASSK_ROLLOUT_JSON_EXPORT=$PASSK_ROLLOUT_JSON_EXPORT" \
      2>&1 | tee -a "$SHELL_LOG"
    ;;
esac
