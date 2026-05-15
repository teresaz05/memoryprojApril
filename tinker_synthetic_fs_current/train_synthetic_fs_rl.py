from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import chz

from synthetic_fs_env import SyntheticFilesystemDatasetBuilder
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl import train


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3.5-35B-A3B"
    lora_rank: int = 32
    renderer_name: str | None = None

    learning_rate: float = 4e-5
    batch_size: int = 32
    group_size: int = 4
    seed: int = 2
    max_tokens: int = 4096
    eval_every: int = 0
    max_steps: int | None = 110

    index_jsonl: str = "../tinker_fs_qa/train_q830_fs/index.jsonl"
    reward_mode: str = "hybrid"

    answerer_backend: str = "gemini"
    answerer_model: str = "gemini-2.5-flash-lite"
    answerer_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    answerer_api_key_env: str = "GEMINI_API_KEY"

    judge_backend: str = "gemini"
    judge_model: str = "gemini-2.5-flash-lite"
    judge_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    judge_api_key_env: str = "GEMINI_API_KEY"

    max_turns: int = 96
    max_trajectory_tokens: int | None = 220000
    max_generation_tokens: int | None = None
    step_penalty: float = 0.0
    termination_penalty: float = 0.1
    raw_docs_penalty: float = 0.0
    empty_synthetic_penalty: float = 1.0
    synthetic_success_bonus: float = 0.0
    synthetic_usage_bonus: float = 0.0
    raw_usage_ratio_penalty: float = 0.0
    filesystem_maturity_scale: float = 1.0
    filesystem_coverage_weight: float = 0.35
    filesystem_expansion_weight: float = 0.3
    filesystem_organization_weight: float = 0.35
    filesystem_stop_weight: float = 0.0
    mature_stop_bonus: float = 0.0
    mature_stop_min_score: float = 0.8
    terminal_reward_clip_min: float = -1.0
    terminal_reward_clip_max: float = 3.0
    answerer_max_turns: int = 8
    answerer_workspace_mode: str = "synthetic_only"
    answerer_final_answer_max_tokens: int = 128
    answerer_retrieval_cost_scale: float = 0.0
    answerer_retrieval_cost_token_unit: float = 1000.0
    answerer_retrieval_cost_correct_only: bool = True
    judge_max_output_tokens: int = 64
    builder_compaction_enabled: bool = True
    builder_compaction_backend: str = "gemini"
    builder_compaction_model: str = "gemini-2.5-flash-lite"
    builder_compaction_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    builder_compaction_api_key_env: str = "GEMINI_API_KEY"
    builder_compaction_trigger_tokens: int = 4000
    builder_compaction_keep_recent_turns: int = 1
    builder_compaction_max_output_tokens: int = 800
    builder_compaction_input_max_chars: int = 60000
    builder_executor_enabled: bool = True
    builder_batch_tools_enabled: bool = True
    builder_executor_backend: str = "openrouter"
    builder_executor_model: str = "qwen/qwen3.5-35b-a3b"
    builder_executor_base_url: str = "https://openrouter.ai/api/v1"
    builder_executor_api_key_env: str = "OPENROUTER_API_KEY"
    builder_executor_max_source_chars: int = 16000
    builder_executor_max_output_tokens: int = 512
    step_construction_action_bonus: float = 0.05
    step_filesystem_maturity_delta_scale: float = 1.0
    step_non_construction_turn_penalty: float = 0.005
    step_non_construction_streak_penalty: float = 0.0
    step_non_construction_streak_free: int = 3
    step_tool_error_penalty: float = 0.05

    excluded_qids_jsonl: str = "../tinker_fs_qa/excluded100.jsonl"
    eval_index_jsonl: str = "../tinker_fs_qa/train_q50_nonexcluded_fs/index.jsonl"
    eval_size: int = 0
    limit: int = 0

    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig) -> None:
    if cli_config.max_generation_tokens is not None:
        raise ValueError(
            "max_generation_tokens is deprecated in this pipeline. "
            "The builder should not be method-limited by a per-turn generation cap. "
            "Remove max_generation_tokens and, if needed, use answerer_final_answer_max_tokens "
            "to control only the answerer's final answer length."
        )
    if cli_config.answerer_workspace_mode != "synthetic_only":
        raise ValueError("This clean setup only supports answerer_workspace_mode=synthetic_only.")
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    builder = SyntheticFilesystemDatasetBuilder(
        index_jsonl=cli_config.index_jsonl,
        model_name_for_tokenizer=cli_config.model_name,
        batch_size=cli_config.batch_size,
        group_size=cli_config.group_size,
        renderer_name=renderer_name,
        reward_mode=cli_config.reward_mode,
        answerer_backend=cli_config.answerer_backend,
        answerer_model=cli_config.answerer_model,
        answerer_base_url=cli_config.answerer_base_url,
        answerer_api_key_env=cli_config.answerer_api_key_env,
        judge_backend=cli_config.judge_backend,
        judge_model=cli_config.judge_model,
        judge_base_url=cli_config.judge_base_url,
        judge_api_key_env=cli_config.judge_api_key_env,
        max_turns=cli_config.max_turns,
        max_trajectory_tokens=cli_config.max_trajectory_tokens,
        max_generation_tokens=cli_config.max_generation_tokens,
        step_penalty=cli_config.step_penalty,
        termination_penalty=cli_config.termination_penalty,
        raw_docs_penalty=cli_config.raw_docs_penalty,
        empty_synthetic_penalty=cli_config.empty_synthetic_penalty,
        synthetic_success_bonus=cli_config.synthetic_success_bonus,
        synthetic_usage_bonus=cli_config.synthetic_usage_bonus,
        raw_usage_ratio_penalty=cli_config.raw_usage_ratio_penalty,
        filesystem_maturity_scale=cli_config.filesystem_maturity_scale,
        filesystem_coverage_weight=cli_config.filesystem_coverage_weight,
        filesystem_expansion_weight=cli_config.filesystem_expansion_weight,
        filesystem_organization_weight=cli_config.filesystem_organization_weight,
        filesystem_stop_weight=cli_config.filesystem_stop_weight,
        mature_stop_bonus=cli_config.mature_stop_bonus,
        mature_stop_min_score=cli_config.mature_stop_min_score,
        terminal_reward_clip_min=cli_config.terminal_reward_clip_min,
        terminal_reward_clip_max=cli_config.terminal_reward_clip_max,
        answerer_max_turns=cli_config.answerer_max_turns,
        answerer_workspace_mode=cli_config.answerer_workspace_mode,
        answerer_final_answer_max_tokens=cli_config.answerer_final_answer_max_tokens,
        answerer_retrieval_cost_scale=cli_config.answerer_retrieval_cost_scale,
        answerer_retrieval_cost_token_unit=cli_config.answerer_retrieval_cost_token_unit,
        answerer_retrieval_cost_correct_only=cli_config.answerer_retrieval_cost_correct_only,
        judge_max_output_tokens=cli_config.judge_max_output_tokens,
        builder_compaction_enabled=cli_config.builder_compaction_enabled,
        builder_compaction_backend=cli_config.builder_compaction_backend,
        builder_compaction_model=cli_config.builder_compaction_model,
        builder_compaction_base_url=cli_config.builder_compaction_base_url,
        builder_compaction_api_key_env=cli_config.builder_compaction_api_key_env,
        builder_compaction_trigger_tokens=cli_config.builder_compaction_trigger_tokens,
        builder_compaction_keep_recent_turns=cli_config.builder_compaction_keep_recent_turns,
        builder_compaction_max_output_tokens=cli_config.builder_compaction_max_output_tokens,
        builder_compaction_input_max_chars=cli_config.builder_compaction_input_max_chars,
        builder_executor_enabled=cli_config.builder_executor_enabled,
        builder_batch_tools_enabled=cli_config.builder_batch_tools_enabled,
        builder_executor_backend=cli_config.builder_executor_backend,
        builder_executor_model=cli_config.builder_executor_model,
        builder_executor_base_url=cli_config.builder_executor_base_url,
        builder_executor_api_key_env=cli_config.builder_executor_api_key_env,
        builder_executor_max_source_chars=cli_config.builder_executor_max_source_chars,
        builder_executor_max_output_tokens=cli_config.builder_executor_max_output_tokens,
        step_construction_action_bonus=cli_config.step_construction_action_bonus,
        step_filesystem_maturity_delta_scale=cli_config.step_filesystem_maturity_delta_scale,
        step_non_construction_turn_penalty=cli_config.step_non_construction_turn_penalty,
        step_non_construction_streak_penalty=cli_config.step_non_construction_streak_penalty,
        step_non_construction_streak_free=cli_config.step_non_construction_streak_free,
        step_tool_error_penalty=cli_config.step_tool_error_penalty,
        excluded_qids_jsonl=cli_config.excluded_qids_jsonl,
        eval_index_jsonl=cli_config.eval_index_jsonl,
        seed=cli_config.seed,
        eval_size=cli_config.eval_size,
        limit=cli_config.limit,
    )

    model_name_short = cli_config.model_name.lower().replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"synthetic_fs_{model_name_short}_bs{cli_config.batch_size}_"
        f"gs{cli_config.group_size}_seed{cli_config.seed}_lr{cli_config.learning_rate}_"
        f"rank{cli_config.lora_rank}_{date_and_time}"
    )

    log_path = cli_config.log_path or f"/tmp/tinker-examples/rl_synthetic_fs/{run_name}"
    wandb_name = cli_config.wandb_name or run_name

    if not Path("/tmp").exists():
        raise ValueError("/tmp does not exist")

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

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
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
