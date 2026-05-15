#!/bin/bash
#SBATCH --job-name=synthfsPI
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=/sailhome/teresaz/tinker_runs/slurm-%x-%j.out
#SBATCH --error=/sailhome/teresaz/tinker_runs/slurm-%x-%j.err

set -euo pipefail

cd /sailhome/teresaz/BrowseCompV2/exact_pipeline/tinker_synthetic_fs_current

missing=()
for key in TINKER_API_KEY OPENROUTER_API_KEY GEMINI_API_KEY WANDB_API_KEY; do
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

RUN_NAME=synthfs_qwen35_executor_pi_alltrain_bs32_gs4_5epoch_seed2_job${SLURM_JOB_ID}
RUN_DIR=/sailhome/teresaz/tinker_runs/$RUN_NAME
SHELL_LOG=/sailhome/teresaz/tinker_runs/shell_logs/${RUN_NAME}.train.log

echo "RUN_NAME=$RUN_NAME"
echo "RUN_DIR=$RUN_DIR"
echo "SHELL_LOG=$SHELL_LOG"

python train_synthetic_fs_rl.py \
  log_path="$RUN_DIR" \
  wandb_project="$WANDB_PROJECT" \
  wandb_name="$RUN_NAME" \
  behavior_if_log_dir_exists=raise \
  2>&1 | tee "$SHELL_LOG"
