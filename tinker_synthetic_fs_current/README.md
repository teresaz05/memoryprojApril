# Current Tinker Synthetic Filesystem Setup

## Files

- `synthetic_fs_env.py`: substantive environment implementation: builder/answerer prompts, synthetic filesystem state, batched tools, frozen executor, compaction, reward, metrics, and dataset builder.
- `train_synthetic_fs_rl.py`: thin Tinker RL entrypoint. Its defaults match the current run.
- `make_nonexcluded_eval50.py`: creates the corrected 50-question eval split after deleting `excluded100`.
- `run_current_wandb.sh`: minimal Slurm/W&B launcher for the current run.
- `requirements.txt`: minimal Python package list.

## Current Run Defaults

- Planner model: `Qwen/Qwen3.5-35B-A3B` on Tinker.
- Frozen executor: `qwen/qwen3.5-35b-a3b` via OpenRouter.
- Answerer/judge: Gemini `gemini-2.5-flash-lite`.
- Answerer workspace: `synthetic_only`.
- All available training questions after deleting `excluded100` and holding out corrected eval50.
- `batch_size=32`, `group_size=4`, `max_steps=110`, `max_turns=96`, `answerer_max_turns=32`.
- If the answerer exhausts its browsing/tool budget, the environment forces one final answer-only call before judging.
- Intermediate reward includes one bounded answerability-progress probe per episode after the filesystem is partially mature:
  `answerability_delta_reward_scale=0.5`, `answerability_probe_max_per_episode=1`, `answerability_probe_min_maturity=0.45`.
- Answerer efficiency is now part of reward for successful answers:
  `answerer_retrieval_cost_scale=0.15` penalizes synthetic tokens read, and
  `answerer_synthetic_read_cost_scale=0.10` penalizes the number of synthetic files visited.
- Memory/logging safeguards are enabled by default:
  detailed per-step text/tool logs are off, reward history does not retain tool-result messages,
  terminal histories are trimmed after scoring, and `max_trajectory_tokens=140000`.

## Run On `sc`

```bash
cd /sailhome/teresaz/BrowseCompV2/exact_pipeline/tinker_synthetic_fs_current

read -rsp "TINKER_API_KEY: " TINKER_API_KEY; echo
export TINKER_API_KEY
read -rsp "OPENROUTER_API_KEY: " OPENROUTER_API_KEY; echo
export OPENROUTER_API_KEY
read -rsp "GEMINI_API_KEY: " GEMINI_API_KEY; echo
export GEMINI_API_KEY
read -rsp "WANDB_API_KEY: " WANDB_API_KEY; echo
export WANDB_API_KEY

sbatch --export=ALL run_current_wandb.sh
```

Expected split line:

```text
Synthetic FS dataset split: train_rows=680 eval_rows=50 excluded_qids=100
```
