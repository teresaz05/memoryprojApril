#!/bin/bash
#SBATCH --job-name=synthfsSumK
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/sailhome/teresaz/tinker_runs/slurm-%x-%j.out
#SBATCH --error=/sailhome/teresaz/tinker_runs/slurm-%x-%j.err

set -euo pipefail

cd /sailhome/teresaz/BrowseCompV2/exact_pipeline/tinker_synthetic_fs_current

missing=()
for key in OPENROUTER_API_KEY GEMINI_API_KEY; do
  if [ -z "${!key:-}" ]; then
    missing+=("$key")
  fi
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "Missing required env vars: ${missing[*]}" >&2
  exit 2
fi

export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
mkdir -p /sailhome/teresaz/tinker_runs/shell_logs

PASSK_K=${PASSK_K:-4}
PASSK_SEED=${PASSK_SEED:-2}
STREAM_MODEL=${STREAM_MODEL:-qwen/qwen3.5-35b-a3b}
STREAM_MODEL_SLUG=$(echo "$STREAM_MODEL" | tr '/[:upper:]' '-[:lower:]')
BASE_RUN_NAME=${BASE_RUN_NAME_OVERRIDE:-synthfs_streamsum_${STREAM_MODEL_SLUG}_eval50_k${PASSK_K}_seed${PASSK_SEED}}
RUN_NAME=${BASE_RUN_NAME}_job${SLURM_JOB_ID}
RUN_DIR=/sailhome/teresaz/tinker_runs/$RUN_NAME
SHELL_LOG=/sailhome/teresaz/tinker_runs/shell_logs/${RUN_NAME}.passk.log

echo "RUN_NAME=$RUN_NAME"
echo "RUN_DIR=$RUN_DIR"
echo "SHELL_LOG=$SHELL_LOG"
echo "STREAM_MODEL=$STREAM_MODEL"
echo "PASSK_K=$PASSK_K"
echo "PASSK_SEED=$PASSK_SEED"

mkdir -p "$RUN_DIR"

python -m py_compile synthetic_fs_env.py run_streaming_summary_passk.py

python run_streaming_summary_passk.py \
  --index-jsonl ../tinker_fs_qa/train_q50_nonexcluded_fs/index.jsonl \
  --out-dir "$RUN_DIR" \
  --k "$PASSK_K" \
  --seed "$PASSK_SEED" \
  --model-backend openrouter \
  --model "$STREAM_MODEL" \
  --model-base-url https://openrouter.ai/api/v1 \
  --model-api-key-env OPENROUTER_API_KEY \
  --judge-backend gemini \
  --judge-model gemini-3.1-flash-lite-preview \
  --judge-base-url https://generativelanguage.googleapis.com/v1beta \
  --judge-api-key-env GEMINI_API_KEY \
  --temperature "${STREAM_TEMPERATURE:-1.0}" \
  --doc-max-chars "${STREAM_DOC_MAX_CHARS:-24000}" \
  --summary-max-output-tokens "${STREAM_SUMMARY_MAX_OUTPUT_TOKENS:-900}" \
  --answer-max-output-tokens "${STREAM_ANSWER_MAX_OUTPUT_TOKENS:-128}" \
  2>&1 | tee -a "$SHELL_LOG"
