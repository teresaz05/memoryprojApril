# Tinker Synthetic Filesystem Current Setup

- Primary weaker-model RL target: `Qwen/Qwen3.5-4B`.
- Optimizer/loss: Tinker `loss_fn=ppo`.
- Reward for the next weaker-model runs: answerability-focused
- Epoch semantics: true reshuffled passes over the filtered train docsets.
- Held-out set: current 50 BrowseComp+ questions in `heldout_50_questions_browsecomp_plus_query_ids.json`.
- Note: Tinker PPO does not expose a critic/value function by default, so value-function metrics are not logged.

## Files

- `synthetic_fs_env.py`: synthetic filesystem environment, builder/answerer tools, reward logic, dataset splitting, and true epoch reshuffling.
- `train_synthetic_fs_rl.py`: Tinker RL entrypoint with PPO config, epoch controls, and RL diagnostics.
- `run_weaker_qwen35_4b_ppo_epoch10_wandb.sh`: main PI-requested weaker-model PPO epoch-10 launcher. Configure with `MODEL_NAME`, `LORA_RANK`, and `LEARNING_RATE`.
- `run_base_passk_wandb.sh`: pass@k evaluator for base or fine-tuned checkpoints.
- `analyze_passk_rollouts.py`: pass@k summary/per-question analysis helper.
- `run_eval_heldout_wandb.sh`: single-trajectory held-out eval helper for a trained checkpoint.
- `run_streaming_summary_passk.py`: streaming summarization baseline implementation.
- `run_streaming_summary_passk_wandb.sh`: SLURM wrapper for streaming-summary pass@k.
- `make_nonexcluded_eval50.py`: regenerates the current non-excluded held-out 50 split.
- `heldout_50_questions.json`: current 50 held-out questions.
- `heldout_50_questions_browsecomp_plus_query_ids.json`: same 50 held-out questions with both `query_id` and `question_id`.
- `requirements.txt`: minimal runtime dependencies.

## PPO Diagnostics Logged

- `optim/advantage_mean`
- `optim/advantage_std`
- `optim/advantage_min`
- `optim/advantage_max`
- `optim/advantage_abs_mean`
- `optim/zero_advantage_ratio`
- `optim/trajectory_turns_max`
- `optim/trajectory_turns_mean`
- `optim/action_tokens_max`
- `optim/action_tokens_mean`
- `optim/ppo_clip_low_ratio`
- `optim/ppo_clip_high_ratio`
- `optim/entropy` from Tinker/cookbook KL metrics

## Launch Weaker-Model PPO Run

```bash
cd /sailhome/teresaz/BrowseCompV2/exact_pipeline/tinker_synthetic_fs_current

for k in TINKER_API_KEY OPENROUTER_API_KEY GEMINI_API_KEY WANDB_API_KEY; do
  echo "$k: ${!k:+set}"
done

export WANDB_CONSOLE=wrap
export MODEL_NAME=Qwen/Qwen3.5-4B
export LORA_RANK=32
export LEARNING_RATE=1e-5
export TRAIN_EPOCHS=10

sbatch --export=ALL run_weaker_qwen35_4b_ppo_epoch10_wandb.sh
```

Sweep knobs:

```bash
export LORA_RANK=64      # or 128
export LEARNING_RATE=5e-6 # or 1e-6 / 8e-5
```

## Launch Streaming-Summarization Pass@4 Baseline

```bash
cd /sailhome/teresaz/BrowseCompV2/exact_pipeline/tinker_synthetic_fs_current

export PASSK_K=4
export PASSK_SEED=2
export STREAM_MODEL=qwen/qwen3.5-35b-a3b

sbatch --export=ALL run_streaming_summary_passk_wandb.sh
```

## Sync From Local To `sc`

```bash
cd /Users/teresaz/Downloads/cs191ResearchWinter/BrowseCompV2/exact_pipeline/tinker_synthetic_fs_current

rsync -av --delete ./ \
  teresaz@sc.stanford.edu:/sailhome/teresaz/BrowseCompV2/exact_pipeline/tinker_synthetic_fs_current/
```

