#!/bin/bash
#SBATCH --job-name=synthfsAns
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
export WANDB_CONSOLE=wrap
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

BASE_RUN_NAME=synthfs_qwen35_executor_pi_alltrain_bs16_gs4_mt32_ans32_ansrep4_probe4int8_answerability_only_g31litepreview_seed2
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
echo "LOGDIR_BEHAVIOR=$LOGDIR_BEHAVIOR"
if [ -f "$RUN_DIR/checkpoints.jsonl" ]; then
  echo "Last checkpoints:"
  tail -5 "$RUN_DIR/checkpoints.jsonl"
fi

python train_synthetic_fs_rl.py \
  batch_size=16 \
  group_size=4 \
  max_turns=32 \
  builder_compaction_trigger_tokens=3000 \
  answerer_model=gemini-3.1-flash-lite-preview \
  judge_model=gemini-3.1-flash-lite-preview \
  builder_compaction_model=gemini-3.1-flash-lite-preview \
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
  save_every=5 \
  rolling_save_every=1 \
  rolling_ttl_seconds=604800 \
  log_path="$RUN_DIR" \
  wandb_project="$WANDB_PROJECT" \
  wandb_name="$RUN_NAME" \
  behavior_if_log_dir_exists="$LOGDIR_BEHAVIOR" \
  2>&1 | tee -a "$SHELL_LOG"
