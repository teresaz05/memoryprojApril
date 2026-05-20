#!/bin/bash
# Base/fine-tuned trajectory-level pass@k diagnostic.
# If SOURCE_RUN_DIR or SOURCE_RUN_NAME is set, this loads that run's final
# checkpoint. Otherwise, it evaluates the base model. It samples k builder
# trajectories per held-out question and exports per-trajectory rollout JSON.
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

BASE_RUN_NAME=synthfs_passk_${SOURCE_LABEL}_eval50_k${PASSK_K}_bs${PASSK_BATCH_SIZE}_mt32_ans32_ansrep${PASSK_ANSWERER_REPEATS}_g31litepreview_seed${PASSK_SEED}
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
INFO

# max_steps=1 with learning_rate=0 performs one no-op update and triggers
# eval_every=1 evaluation. With SOURCE_RUN_* set, it first loads that checkpoint.
python train_synthetic_fs_rl.py \
  batch_size="$PASSK_BATCH_SIZE" \
  group_size="$PASSK_K" \
  seed="$PASSK_SEED" \
  max_turns=32 \
  learning_rate=0.0 \
  max_steps=1 \
  eval_every=1 \
  builder_compaction_trigger_tokens=3000 \
  answerer_model=gemini-3.1-flash-lite-preview \
  judge_model=gemini-3.1-flash-lite-preview \
  builder_compaction_model=gemini-3.1-flash-lite-preview \
  reward_mode=hybrid \
  terminal_answerer_repeats="$PASSK_ANSWERER_REPEATS" \
  answerability_delta_reward_scale=0.0 \
  num_groups_to_log=50 \
  rollout_json_export=True \
  save_every=999999 \
  rolling_save_every=999999 \
  log_path="$RUN_DIR" \
  wandb_project="$WANDB_PROJECT" \
  wandb_name="$RUN_NAME" \
  behavior_if_log_dir_exists=raise \
  "${LOAD_ARGS[@]}" \
  2>&1 | tee -a "$SHELL_LOG"

python analyze_passk_rollouts.py \
  "$RUN_DIR" \
  --k "$PASSK_K" \
  --eval-index-jsonl ../tinker_fs_qa/train_q50_nonexcluded_fs/index.jsonl \
  --out-dir "$RUN_DIR" \
  2>&1 | tee -a "$SHELL_LOG"
