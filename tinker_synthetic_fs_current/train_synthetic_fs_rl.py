from __future__ import annotations

import asyncio
import gc
import json
import logging
import resource
from datetime import datetime
from pathlib import Path
from typing import Any

import chz
import torch

from synthetic_fs_env import SyntheticFilesystemDatasetBuilder
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl import train
from tinker_cookbook.utils.misc_utils import safezip


LOGGER = logging.getLogger(__name__)


def _maxrss_mb() -> float:
    # Linux reports ru_maxrss in KiB; macOS reports bytes. Remote SC runs Linux.
    maxrss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if maxrss > 10_000_000_000:
        return maxrss / (1024 * 1024)
    return maxrss / 1024


def _install_rollout_group_concurrency_limit(max_concurrent: int) -> None:
    """Throttle concurrent rollout groups while preserving logical batch size."""
    if max_concurrent <= 0:
        return
    if getattr(train, "_synthetic_fs_rollout_concurrency_patch", False):
        return

    from tinker_cookbook.rl import metric_util

    original_train_group_rollout = train.do_group_rollout_and_filter_constant_reward
    original_eval_group_rollout = metric_util.do_group_rollout
    original_eval_group_rollout_filtered = metric_util.do_group_rollout_and_filter_constant_reward
    semaphore_by_loop: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}

    def get_semaphore() -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        semaphore = semaphore_by_loop.get(loop)
        if semaphore is None:
            semaphore = asyncio.Semaphore(max_concurrent)
            semaphore_by_loop[loop] = semaphore
        return semaphore

    async def limited_train_group_rollout(*args: Any, **kwargs: Any):
        async with get_semaphore():
            return await original_train_group_rollout(*args, **kwargs)

    async def limited_eval_group_rollout(*args: Any, **kwargs: Any):
        async with get_semaphore():
            return await original_eval_group_rollout(*args, **kwargs)

    async def limited_eval_group_rollout_filtered(*args: Any, **kwargs: Any):
        async with get_semaphore():
            return await original_eval_group_rollout_filtered(*args, **kwargs)

    train.do_group_rollout_and_filter_constant_reward = limited_train_group_rollout
    metric_util.do_group_rollout = limited_eval_group_rollout
    metric_util.do_group_rollout_and_filter_constant_reward = limited_eval_group_rollout_filtered
    train._synthetic_fs_rollout_concurrency_patch = True
    LOGGER.info("Limited concurrent rollout groups to %d", max_concurrent)


def _clear_rollout_groups_for_memory(trajectory_groups_P: Any) -> dict[str, float]:
    groups = 0
    trajectories = 0
    transitions = 0
    for group in trajectory_groups_P or []:
        groups += 1
        group_trajectories = getattr(group, "trajectories_G", None)
        if group_trajectories is None:
            group_trajectories = getattr(group, "trajectories", None)
        for trajectory in list(group_trajectories or []):
            trajectories += 1
            trajectory_transitions = getattr(trajectory, "transitions", None)
            if trajectory_transitions is not None:
                try:
                    transitions += len(trajectory_transitions)
                    trajectory_transitions.clear()
                except Exception:
                    setattr(trajectory, "transitions", [])
            for attr in ("messages", "history", "logs", "metrics", "observations", "states"):
                value = getattr(trajectory, attr, None)
                if value is None:
                    continue
                try:
                    value.clear()
                except Exception:
                    try:
                        setattr(trajectory, attr, None)
                    except Exception:
                        pass
        if group_trajectories is not None:
            try:
                group_trajectories.clear()
            except Exception:
                try:
                    if hasattr(group, "trajectories_G"):
                        setattr(group, "trajectories_G", [])
                    elif hasattr(group, "trajectories"):
                        setattr(group, "trajectories", [])
                except Exception:
                    pass
    try:
        trajectory_groups_P.clear()
    except Exception:
        pass
    gc.collect()
    return {
        "rollout_memory_clear/groups": float(groups),
        "rollout_memory_clear/trajectories": float(trajectories),
        "rollout_memory_clear/transitions": float(transitions),
    }


def _install_rl_diagnostics_patch(
    *,
    loss_fn: str,
    ppo_clip_low_threshold: float,
    ppo_clip_high_threshold: float,
    free_rollout_after_prepare_minibatch: bool,
) -> None:
    """Add local RL diagnostics without changing Tinker's optimizer code."""
    if getattr(train, "_synthetic_fs_rl_diagnostics_patch", False):
        return

    original_prepare_minibatch = train.prepare_minibatch
    original_compute_full_batch = train.compute_full_batch_metrics_and_get_sampling_client

    async def prepare_minibatch_with_diagnostics(*args: Any, **kwargs: Any):
        data_D, metrics = await original_prepare_minibatch(*args, **kwargs)
        trajectory_groups_P = args[1] if len(args) > 1 else kwargs.get("trajectory_groups_P", [])
        advantage_chunks = []
        action_token_counts = []
        target_token_counts = []
        for datum in data_D:
            advantages = datum.loss_fn_inputs["advantages"].to_torch()
            mask = datum.loss_fn_inputs["mask"].to_torch() > 0
            if mask.any():
                action_advantages = advantages[mask].float()
                advantage_chunks.append(action_advantages)
                action_token_counts.append(float(mask.sum().item()))
            target_token_counts.append(float(len(datum.loss_fn_inputs["target_tokens"].data)))
        if advantage_chunks:
            flat_advantages = torch.cat(advantage_chunks)
            metrics.update(
                {
                    "optim/advantage_mean": float(flat_advantages.mean().item()),
                    "optim/advantage_std": float(flat_advantages.std(unbiased=False).item()),
                    "optim/advantage_min": float(flat_advantages.min().item()),
                    "optim/advantage_max": float(flat_advantages.max().item()),
                    "optim/advantage_abs_mean": float(flat_advantages.abs().mean().item()),
                    "optim/zero_advantage_ratio": float(
                        (flat_advantages.abs() <= 1e-12).float().mean().item()
                    ),
                }
            )
        if action_token_counts:
            metrics["optim/action_tokens_max"] = max(action_token_counts)
            metrics["optim/action_tokens_mean"] = sum(action_token_counts) / len(action_token_counts)
        if target_token_counts:
            metrics["optim/train_datum_tokens_max"] = max(target_token_counts)
            metrics["optim/train_datum_tokens_mean"] = sum(target_token_counts) / len(
                target_token_counts
            )
        trajectory_lengths = [
            float(len(traj.transitions))
            for group in trajectory_groups_P
            for traj in group.trajectories_G
        ]
        if trajectory_lengths:
            metrics["optim/trajectory_turns_max"] = max(trajectory_lengths)
            metrics["optim/trajectory_turns_mean"] = sum(trajectory_lengths) / len(
                trajectory_lengths
            )
        if free_rollout_after_prepare_minibatch:
            rss_before_mb = _maxrss_mb()
            clear_metrics = _clear_rollout_groups_for_memory(trajectory_groups_P)
            rss_after_mb = _maxrss_mb()
            metrics.update(clear_metrics)
            metrics["rollout_memory_clear/maxrss_before_mb"] = rss_before_mb
            metrics["rollout_memory_clear/maxrss_after_mb"] = rss_after_mb
            LOGGER.info(
                "Cleared rollout groups after minibatch preparation: "
                "groups=%s trajectories=%s transitions=%s maxrss_before_mb=%.1f maxrss_after_mb=%.1f",
                int(clear_metrics["rollout_memory_clear/groups"]),
                int(clear_metrics["rollout_memory_clear/trajectories"]),
                int(clear_metrics["rollout_memory_clear/transitions"]),
                rss_before_mb,
                rss_after_mb,
            )
        return data_D, metrics

    async def compute_full_batch_with_diagnostics(
        training_client,
        checkpoint_mgr,
        i_batch,
        data_D,
        training_logprobs_D,
        do_compute_post_kl,
    ):
        if training_logprobs_D:
            ratio_chunks = []
            for datum, training_logprobs in safezip(data_D, training_logprobs_D):
                sampled_logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
                action_mask = datum.loss_fn_inputs["mask"].to_torch() > 0
                if action_mask.any():
                    ratio_chunks.append(
                        torch.exp(training_logprobs[action_mask] - sampled_logprobs[action_mask])
                    )
            if ratio_chunks:
                flat_ratios = torch.cat(ratio_chunks).float()
                ratio_metrics = {
                    "optim/importance_ratio_mean": float(flat_ratios.mean().item()),
                    "optim/importance_ratio_min": float(flat_ratios.min().item()),
                    "optim/importance_ratio_max": float(flat_ratios.max().item()),
                }
                if loss_fn == "ppo":
                    ratio_metrics.update(
                        {
                            "optim/ppo_clip_low_ratio": float(
                                (flat_ratios < ppo_clip_low_threshold).float().mean().item()
                            ),
                            "optim/ppo_clip_high_ratio": float(
                                (flat_ratios > ppo_clip_high_threshold).float().mean().item()
                            ),
                            "optim/ppo_clip_low_threshold": float(ppo_clip_low_threshold),
                            "optim/ppo_clip_high_threshold": float(ppo_clip_high_threshold),
                        }
                    )
            else:
                ratio_metrics = {}
        else:
            ratio_metrics = {}
        sampling_client, metrics = await original_compute_full_batch(
            training_client,
            checkpoint_mgr,
            i_batch,
            data_D,
            training_logprobs_D,
            do_compute_post_kl,
        )
        metrics.update(ratio_metrics)
        return sampling_client, metrics

    train.prepare_minibatch = prepare_minibatch_with_diagnostics
    train.compute_full_batch_metrics_and_get_sampling_client = compute_full_batch_with_diagnostics
    train._synthetic_fs_rl_diagnostics_patch = True


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3.5-35B-A3B"
    lora_rank: int = 32
    renderer_name: str | None = None

    learning_rate: float = 4e-5
    batch_size: int = 16
    group_size: int = 4
    train_epochs: int = 1
    reshuffle_each_epoch: bool = True
    num_substeps: int = 1
    stream_minibatch_groups_per_batch: int = 0
    stream_minibatch_num_minibatches: int = 0
    max_concurrent_rollout_groups: int = 0
    loss_fn: str = "importance_sampling"
    loss_fn_config_json: str = ""
    ppo_clip_low_threshold: float = 0.8
    ppo_clip_high_threshold: float = 1.2
    log_rl_diagnostics: bool = True
    seed: int = 2
    max_tokens: int = 4096
    eval_every: int = 0
    max_steps: int | None = 110
    save_every: int = 5
    ttl_seconds: int | None = 604800
    rolling_save_every: int = 1
    rolling_ttl_seconds: int = 604800
    load_checkpoint_path: str | None = None

    index_jsonl: str = "../tinker_fs_qa/train_q830_fs/index.jsonl"
    reward_mode: str = "hybrid"

    answerer_backend: str = "gemini"
    answerer_model: str = "gemini-3.1-flash-lite-preview"
    answerer_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    answerer_api_key_env: str = "GEMINI_API_KEY"

    judge_backend: str = "gemini"
    judge_model: str = "gemini-3.1-flash-lite-preview"
    judge_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    judge_api_key_env: str = "GEMINI_API_KEY"

    max_turns: int = 32
    max_trajectory_tokens: int | None = 140000
    max_generation_tokens: int | None = None
    step_penalty: float = 0.0
    termination_penalty: float = 0.1
    raw_docs_penalty: float = 0.0
    empty_synthetic_penalty: float = 1.0
    synthetic_success_bonus: float = 0.0
    synthetic_usage_bonus: float = 0.0
    raw_usage_ratio_penalty: float = 0.0
    filesystem_maturity_scale: float = 0.5
    filesystem_coverage_weight: float = 0.35
    filesystem_expansion_weight: float = 0.3
    filesystem_organization_weight: float = 0.35
    filesystem_stop_weight: float = 0.0
    mature_stop_bonus: float = 0.0
    mature_stop_min_score: float = 0.8
    terminal_reward_clip_min: float = -1.0
    terminal_reward_clip_max: float = 3.0
    answerer_max_turns: int = 32
    answerer_workspace_mode: str = "synthetic_only"
    answerer_final_answer_max_tokens: int = 128
    answerer_retrieval_cost_scale: float = 0.15
    answerer_retrieval_cost_token_unit: float = 1000.0
    answerer_retrieval_cost_correct_only: bool = True
    answerer_synthetic_read_cost_scale: float = 0.10
    answerer_synthetic_read_cost_unit: float = 10.0
    terminal_answerer_repeats: int = 4
    answerability_delta_reward_scale: float = 0.5
    answerability_delta_min_abs: float = 0.25
    answerability_delta_allow_negative: bool = True
    answerability_probe_max_per_episode: int = 4
    answerability_probe_interval_turns: int = 8
    answerability_probe_min_maturity: float = 0.45
    answerability_probe_repeats: int = 4
    judge_max_output_tokens: int = 64
    log_step_details: bool = False
    log_compaction_summaries: bool = False
    retain_reward_tool_messages: bool = False
    trim_terminal_history_for_memory: bool = True
    return_empty_terminal_observation: bool = True
    clear_state_on_terminal_for_memory: bool = True
    disable_sample_trajectory_printing: bool = True
    free_rollout_after_prepare_minibatch: bool = False
    num_groups_to_log: int = 0
    rollout_json_export: bool = False
    builder_compaction_enabled: bool = True
    builder_compaction_backend: str = "gemini"
    builder_compaction_model: str = "gemini-3.1-flash-lite-preview"
    builder_compaction_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    builder_compaction_api_key_env: str = "GEMINI_API_KEY"
    builder_compaction_trigger_tokens: int = 3000
    builder_compaction_keep_recent_turns: int = 1
    builder_compaction_max_output_tokens: int = 800
    builder_compaction_input_max_chars: int = 60000
    builder_executor_enabled: bool = True
    builder_batch_tools_enabled: bool = True
    builder_executor_backend: str = "openrouter"
    builder_executor_model: str = "llm"
    builder_executor_base_url: str = "https://iris-lab-ws--lateral-vllm-qwen35-4b.modal.run/v1"
    builder_executor_api_key_env: str = ""
    builder_executor_max_source_chars: int = 16000
    builder_executor_max_output_tokens: int = 512
    step_construction_action_bonus: float = 0.05
    step_filesystem_maturity_delta_scale: float = 0.5
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
        train_epochs=cli_config.train_epochs,
        reshuffle_each_epoch=cli_config.reshuffle_each_epoch,
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
        answerer_synthetic_read_cost_scale=cli_config.answerer_synthetic_read_cost_scale,
        answerer_synthetic_read_cost_unit=cli_config.answerer_synthetic_read_cost_unit,
        terminal_answerer_repeats=cli_config.terminal_answerer_repeats,
        answerability_delta_reward_scale=cli_config.answerability_delta_reward_scale,
        answerability_delta_min_abs=cli_config.answerability_delta_min_abs,
        answerability_delta_allow_negative=cli_config.answerability_delta_allow_negative,
        answerability_probe_max_per_episode=cli_config.answerability_probe_max_per_episode,
        answerability_probe_interval_turns=cli_config.answerability_probe_interval_turns,
        answerability_probe_min_maturity=cli_config.answerability_probe_min_maturity,
        answerability_probe_repeats=cli_config.answerability_probe_repeats,
        judge_max_output_tokens=cli_config.judge_max_output_tokens,
        log_step_details=cli_config.log_step_details,
        log_compaction_summaries=cli_config.log_compaction_summaries,
        retain_reward_tool_messages=cli_config.retain_reward_tool_messages,
        trim_terminal_history_for_memory=cli_config.trim_terminal_history_for_memory,
        return_empty_terminal_observation=cli_config.return_empty_terminal_observation,
        clear_state_on_terminal_for_memory=cli_config.clear_state_on_terminal_for_memory,
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

    if cli_config.disable_sample_trajectory_printing:
        train.print_group = lambda *args, **kwargs: None

    _install_rollout_group_concurrency_limit(cli_config.max_concurrent_rollout_groups)

    streaming_requested = bool(
        cli_config.stream_minibatch_groups_per_batch or cli_config.stream_minibatch_num_minibatches
    )
    free_rollout_after_prepare_minibatch = cli_config.free_rollout_after_prepare_minibatch
    if free_rollout_after_prepare_minibatch and streaming_requested:
        # Tinker's streaming trainer keeps trajectory-group references for
        # end-of-step metrics after minibatches are prepared. Clearing them here
        # corrupts those metrics; streaming itself is the memory-control path.
        LOGGER.warning(
            "Disabling free_rollout_after_prepare_minibatch because stream_minibatch_config "
            "is enabled; streaming still trains one minibatch at a time."
        )
        free_rollout_after_prepare_minibatch = False

    if cli_config.log_rl_diagnostics or free_rollout_after_prepare_minibatch:
        _install_rl_diagnostics_patch(
            loss_fn=cli_config.loss_fn,
            ppo_clip_low_threshold=cli_config.ppo_clip_low_threshold,
            ppo_clip_high_threshold=cli_config.ppo_clip_high_threshold,
            free_rollout_after_prepare_minibatch=free_rollout_after_prepare_minibatch,
        )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    stream_minibatch_config = None
    if streaming_requested:
        if (
            cli_config.stream_minibatch_groups_per_batch <= 0
            or cli_config.stream_minibatch_num_minibatches <= 0
        ):
            raise ValueError(
                "stream_minibatch_groups_per_batch and stream_minibatch_num_minibatches "
                "must both be positive when either is set."
            )
        if cli_config.batch_size != cli_config.stream_minibatch_groups_per_batch:
            raise ValueError(
                "stream_minibatch_groups_per_batch must equal batch_size. "
                "Tinker's synchronous streaming trainer launches one worker per dataset batch group; "
                "a smaller value leaves unconsumed rollout workers, and a larger value waits forever."
            )
        if cli_config.stream_minibatch_groups_per_batch % cli_config.num_substeps != 0:
            raise ValueError(
                "stream_minibatch_groups_per_batch must be divisible by num_substeps."
            )
        groups_per_substep = cli_config.stream_minibatch_groups_per_batch // cli_config.num_substeps
        if groups_per_substep % cli_config.stream_minibatch_num_minibatches != 0:
            raise ValueError(
                "stream_minibatch_num_minibatches must divide "
                "stream_minibatch_groups_per_batch / num_substeps."
            )
        stream_minibatch_config = train.StreamMinibatchConfig(
            groups_per_batch=cli_config.stream_minibatch_groups_per_batch,
            num_minibatches=cli_config.stream_minibatch_num_minibatches,
        )

    loss_fn_config = json.loads(cli_config.loss_fn_config_json) if cli_config.loss_fn_config_json else None
    if cli_config.loss_fn == "ppo" and loss_fn_config is None:
        loss_fn_config = {
            "clip_low_threshold": cli_config.ppo_clip_low_threshold,
            "clip_high_threshold": cli_config.ppo_clip_high_threshold,
        }

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
        loss_fn=cli_config.loss_fn,
        loss_fn_config=loss_fn_config,
        num_substeps=cli_config.num_substeps,
        stream_minibatch_config=stream_minibatch_config,
        max_steps=cli_config.max_steps,
        save_every=cli_config.save_every,
        ttl_seconds=cli_config.ttl_seconds,
        rolling_save_every=cli_config.rolling_save_every,
        rolling_ttl_seconds=cli_config.rolling_ttl_seconds,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        num_groups_to_log=cli_config.num_groups_to_log,
        rollout_json_export=cli_config.rollout_json_export,
    )
    await train.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
