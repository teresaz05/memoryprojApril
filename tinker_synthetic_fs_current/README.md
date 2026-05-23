# Tinker Synthetic Filesystem Current Setup

- Primary weaker-model RL target: `Qwen/Qwen3.5-4B`.
- Optimizer/loss: Tinker `loss_fn=importance_sampling`.
- Current reward target: answerability-focused/simple reward, matching the stronger signs from the older stale-run setup.
- Epoch semantics: true reshuffled passes over the filtered train docsets.
- Builder executor / streaming baseline model: Modal vLLM endpoint for `Qwen/Qwen3.5-4B`
  (`https://iris-lab-ws--lateral-vllm-qwen35-4b.modal.run/v1`, model name `llm`).
- Held-out set: current 50 BrowseComp+ questions in `heldout_50_questions_browsecomp_plus_query_ids.json`.
- Note: value-function metrics are not logged because this setup does not use/expose a learned critic.

## Files

- `synthetic_fs_env.py`: synthetic filesystem environment, builder/answerer tools, reward logic, dataset splitting, and true epoch reshuffling.
- `train_synthetic_fs_rl.py`: Tinker RL entrypoint with importance-sampling config, epoch controls, and RL diagnostics.
- `run_weaker_qwen35_4b_importance_epoch10_wandb.sh`: main weaker-model importance-sampling epoch-10 launcher. Configure with `MODEL_NAME`, `LORA_RANK`, and `LEARNING_RATE`.
- `run_base_passk_wandb.sh`: pass@k evaluator for base or fine-tuned checkpoints.
- `analyze_passk_rollouts.py`: pass@k summary/per-question analysis helper.
- `run_eval_heldout_wandb.sh`: single-trajectory held-out eval helper for a trained checkpoint.
- `run_streaming_summary_passk.py`: streaming summarization baseline implementation.
- `run_streaming_summary_passk_wandb.sh`: SLURM wrapper for streaming-summary pass@k.
- `make_nonexcluded_eval50.py`: regenerates the current non-excluded held-out 50 split.
- `heldout_50_questions.json`: current 50 held-out questions.
- `heldout_50_questions_browsecomp_plus_query_ids.json`: same 50 held-out questions with both `query_id` and `question_id`.
- `requirements.txt`: minimal runtime dependencies.

## Diagnostics Logged

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
- `optim/importance_ratio_mean`
- `optim/importance_ratio_min`
- `optim/importance_ratio_max`
- `optim/entropy` from Tinker/cookbook KL metrics

## Launch Weaker-Model Importance-Sampling Run

```bash
cd /sailhome/teresaz/BrowseCompV2/exact_pipeline/tinker_synthetic_fs_current

for k in TINKER_API_KEY GEMINI_API_KEY WANDB_API_KEY; do
  echo "$k: ${!k:+set}"
done

export WANDB_CONSOLE=wrap
export MODEL_NAME=Qwen/Qwen3.5-4B
export LORA_RANK=32
export LEARNING_RATE=4e-5
export TRAIN_EPOCHS=10

sbatch --export=ALL run_weaker_qwen35_4b_importance_epoch10_wandb.sh
```

Sweep knobs:

```bash
export LORA_RANK=64       # or 128
export LEARNING_RATE=5e-6 # or 1e-6 / 8e-5
```

## Launch Streaming-Summarization Pass@4 Baseline

```bash
cd /sailhome/teresaz/BrowseCompV2/exact_pipeline/tinker_synthetic_fs_current

export PASSK_K=4
export PASSK_SEED=2
export STREAM_MODEL=llm
export STREAM_BASE_URL=https://iris-lab-ws--lateral-vllm-qwen35-4b.modal.run/v1
export STREAM_API_KEY_ENV=

sbatch --export=ALL run_streaming_summary_passk_wandb.sh
```

## Sync From Local To `sc`

```bash
cd /Users/teresaz/Downloads/cs191ResearchWinter/BrowseCompV2/exact_pipeline/tinker_synthetic_fs_current

rsync -av --delete ./ \
  teresaz@sc.stanford.edu:/sailhome/teresaz/BrowseCompV2/exact_pipeline/tinker_synthetic_fs_current/
```
