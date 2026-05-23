#!/bin/bash
set -euo pipefail

# Launch the four remaining jobs in the 5-point Gemini-answerer sweep.
# Baseline already launched separately: rank32, lr4e-5.

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

mkdir -p /sailhome/teresaz/tinker_runs

export WANDB_CONSOLE=${WANDB_CONSOLE:-wrap}

export MODEL_NAME=Qwen/Qwen3.5-4B
export TRAIN_EPOCHS=10
export BATCH_SIZE=16
export GROUP_SIZE=4
export MAX_TURNS=32

export EXCLUDED_QIDS_JSONL=../tinker_fs_qa/excluded100.jsonl
export EVAL_INDEX_JSONL=../tinker_fs_qa/train_q50_nonexcluded_fs/index.jsonl

export ANSWERER_BACKEND=gemini
export ANSWERER_MODEL=gemini-3.1-flash-lite-preview
export ANSWERER_BASE_URL=https://generativelanguage.googleapis.com/v1beta
export ANSWERER_API_KEY_ENV=GEMINI_API_KEY

export BUILDER_EXECUTOR_BACKEND=openrouter
export BUILDER_EXECUTOR_MODEL=llm
export BUILDER_EXECUTOR_BASE_URL=https://iris-lab-ws--lateral-vllm-qwen3-5-35b-a3b.modal.run/v1
export BUILDER_EXECUTOR_API_KEY_ENV=
export BUILDER_EXECUTOR_MAX_OUTPUT_TOKENS=512

unset RESUME_RUN_DIR RESUME_RUN_NAME BASE_RUN_NAME_OVERRIDE

JOBS_FILE=${JOBS_FILE:-/sailhome/teresaz/tinker_runs/gemini_answerer_modal35b_minisweep_jobs.txt}
: > "$JOBS_FILE"

echo "# Baseline already launched: rank32 lr4e-5 job ${BASELINE_JOB_ID:-unknown}" | tee -a "$JOBS_FILE"
echo "# Remaining sweep jobs submitted at $(date -Is)" | tee -a "$JOBS_FILE"
echo "# job_id lora_rank learning_rate" | tee -a "$JOBS_FILE"

submit_one() {
  local rank="$1"
  local lr="$2"
  export LORA_RANK="$rank"
  export LEARNING_RATE="$lr"

  echo "Submitting Gemini-answerer sweep config: rank=${LORA_RANK} lr=${LEARNING_RATE}" >&2
  local jid
  jid=$(sbatch --parsable --export=ALL run_weaker_qwen35_4b_importance_epoch10_wandb.sh)
  echo "$jid $LORA_RANK $LEARNING_RATE" | tee -a "$JOBS_FILE"
}

# Rank sweep at current LR, excluding the already-running rank32/lr4e-5 baseline.
submit_one 64 4e-5
submit_one 128 4e-5

# LR sweep at current rank.
submit_one 32 1e-5
submit_one 32 8e-5

echo "Wrote jobs to $JOBS_FILE"
