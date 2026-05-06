from __future__ import annotations
"""Launch Tinker RL training for the filesystem QA task.

This file does the "training wiring":
1. define CLI/config fields
2. build the dataset builder
3. choose renderer / log directory defaults
4. create tinker_cookbook.rl.train.Config
5. call train.main(...)
"""

import asyncio
from datetime import datetime
from pathlib import Path

import chz

from filesystem_qa_env import FilesystemQADatasetBuilder
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl import train


@chz.chz
class CLIConfig:
    # These fields become the command-line/config surface for the training run.
    # chz uses key=value syntax, not argparse-style --flag value syntax.
    model_name: str = "Qwen/Qwen3.5-4B"
    lora_rank: int = 32
    renderer_name: str | None = None

    learning_rate: float = 4e-5
    batch_size: int = 32
    group_size: int = 4
    seed: int = 2
    max_tokens: int = 1024
    eval_every: int = 0
    max_steps: int | None = None

    index_jsonl: str = "./train_q50_fs/index.jsonl"
    reward_mode: str = "exact"
    judge_model: str = "qwen/qwen3.5-35b-a3b"
    judge_base_url: str = "https://openrouter.ai/api/v1"
    judge_api_key_env: str = "OPENROUTER_API_KEY"
    max_trajectory_tokens: int = 32 * 1024
    max_generation_tokens: int | None = None
    context_overflow_reward: float = -0.1
    eval_size: int = 5
    limit: int = 0

    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig) -> None:
    # Tinker renderers are model-family-specific prompt formatters. If the user
    # does not specify one, we ask the cookbook for the recommended renderer.
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    # The dataset builder is the object Tinker will call to get train/eval RL
    # datasets. It owns the per-example environment construction path.
    builder = FilesystemQADatasetBuilder(
        index_jsonl=cli_config.index_jsonl,
        model_name_for_tokenizer=cli_config.model_name,
        batch_size=cli_config.batch_size,
        group_size=cli_config.group_size,
        renderer_name=renderer_name,
        reward_mode=cli_config.reward_mode,
        judge_model=cli_config.judge_model,
        judge_base_url=cli_config.judge_base_url,
        judge_api_key_env=cli_config.judge_api_key_env,
        max_trajectory_tokens=cli_config.max_trajectory_tokens,
        max_generation_tokens=cli_config.max_generation_tokens,
        context_overflow_reward=cli_config.context_overflow_reward,
        seed=cli_config.seed,
        eval_size=cli_config.eval_size,
        limit=cli_config.limit,
    )

    # Build a readable run name for logs/checkpoints.
    model_name_short = cli_config.model_name.lower().replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"fs_qa_{model_name_short}_bs{cli_config.batch_size}_gs{cli_config.group_size}_"
        f"seed{cli_config.seed}_lr{cli_config.learning_rate}_rank{cli_config.lora_rank}_{date_and_time}"
    )

    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/rl_fs_qa/{run_name}"

    wandb_name = cli_config.wandb_name or run_name

    if not Path("/tmp").exists():
        raise ValueError("/tmp does not exist")

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # This is the main Tinker RL config object. Once built, train.main(config)
    # takes over and runs the rollout / optimization loop remotely.
    config = train.Config(
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        log_path=log_path,
        dataset_builder=builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        eval_every=cli_config.eval_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        lora_rank=cli_config.lora_rank,
        max_steps=cli_config.max_steps,
    )
    await train.main(config)


if __name__ == "__main__":
    # chz.entrypoint reads CLI key=value arguments into the CLIConfig dataclass.
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
