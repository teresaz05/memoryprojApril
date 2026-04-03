#!/usr/bin/env python3
"""Run the structured merge2 experiment without the old shell wrappers."""

from __future__ import annotations

import argparse
from pathlib import Path

from _script_utils import PACKAGE_ROOT, run_script_main, timestamp_now
from april_version_code.common import grading as grading_core
from april_version_code.common.llm_backends import load_local_env, model_slug, require_backend_credentials
from april_version_code.methods import cluster_bank_core
from april_version_code.methods import structured_merge2 as experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the structured merge2 experiment on the q50 support-only dataset.')
    parser.add_argument('--dataset-jsonl', default=str(PACKAGE_ROOT / 'data' / 'derived' / 'support_only' / 'support_only_q50_manydocs.jsonl'))
    parser.add_argument('--run-dir', default='')
    parser.add_argument('--llm-backend', choices=['gemini', 'openrouter'], default='openrouter')
    parser.add_argument('--model', default='qwen/qwen3.5-35b-a3b')
    parser.add_argument('--answer-model', default='')
    parser.add_argument('--judge-backend', choices=['gemini', 'openrouter'], default='openrouter')
    parser.add_argument('--judge-model', default='')
    parser.add_argument('--openrouter-base-url', default='https://openrouter.ai/api/v1')
    parser.add_argument('--openrouter-app-title', default='AprilVersionCode')
    parser.add_argument('--per-doc-summary-budget-tokens', type=int, default=100)
    parser.add_argument('--doc-cluster-style', choices=['list_only', 'titled'], default='titled')
    parser.add_argument('--doc-cluster-max-queries-per-bank', type=int, default=5)
    parser.add_argument('--max-doc-tokens', type=int, default=12000)
    parser.add_argument('--doc-truncate-strategy', choices=['head', 'middle', 'tail'], default='head')
    parser.add_argument('--max-docs-per-query', type=int, default=0)
    parser.add_argument('--progress-every', type=int, default=1)
    parser.add_argument('--grade-progress-every', type=int, default=1)
    parser.add_argument('--skip-grading', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_local_env(PACKAGE_ROOT)
    # Dry runs can skip credential checks because the core runner never calls the model backends.
    backends_to_check = []
    if not args.dry_run:
        backends_to_check.append(args.llm_backend)
    if not args.skip_grading:
        backends_to_check.append(args.judge_backend)
    require_backend_credentials(backends_to_check)

    answer_model = args.answer_model or args.model
    judge_model = args.judge_model or answer_model
    run_dir = Path(args.run_dir) if args.run_dir else experiment.build_default_run_dir(
        PACKAGE_ROOT,
        model_slug(answer_model),
        timestamp_now(),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'results').mkdir(exist_ok=True)
    (run_dir / 'graded').mkdir(exist_ok=True)
    (run_dir / 'traces').mkdir(exist_ok=True)

    layer1_stub = experiment.layer1_result_stub(args.doc_cluster_style)
    merged_stub = experiment.merged_result_stub(args.doc_cluster_style)
    layer1_result = run_dir / 'results' / f'{layer1_stub}.jsonl'
    merged_result = run_dir / 'results' / f'{merged_stub}.jsonl'
    trace_jsonl = run_dir / 'traces' / f'{merged_stub}_trace.jsonl'
    layer1_graded = run_dir / 'graded' / f'{layer1_stub}_scored.jsonl'
    merged_graded = run_dir / 'graded' / f'{merged_stub}_scored.jsonl'

    run_script_main(
        'cluster_bank_core',
        cluster_bank_core.main,
        [
            '--dataset_jsonl', args.dataset_jsonl,
            '--out_jsonl', str(merged_result),
            '--layer1_out_jsonl', str(layer1_result),
            '--trace_jsonl', str(trace_jsonl),
            '--llm_backend', args.llm_backend,
            '--model', args.model,
            '--answer_model', answer_model,
            '--openrouter_base_url', args.openrouter_base_url,
            '--openrouter_app_title', args.openrouter_app_title,
            '--summary_mode', experiment.SUMMARY_MODE,
            '--per_doc_summary_budget_tokens', str(args.per_doc_summary_budget_tokens),
            '--doc_cluster_memory_budget_tokens', '0',
            '--doc_cluster_max_queries_per_bank', str(args.doc_cluster_max_queries_per_bank),
            '--doc_cluster_style', args.doc_cluster_style,
            '--max_doc_tokens', str(args.max_doc_tokens),
            '--doc_truncate_strategy', args.doc_truncate_strategy,
            '--max_docs_per_query', str(args.max_docs_per_query),
            '--progress_every', str(args.progress_every),
            *( ['--dry_run'] if args.dry_run else [] ),
        ],
    )

    if not args.skip_grading:
        run_script_main(
            'grading',
            grading_core.main,
            [
                '--in_jsonl', str(layer1_result),
                '--out_jsonl', str(layer1_graded),
                '--dataset_jsonl', args.dataset_jsonl,
                '--judge_backend', args.judge_backend,
                '--judge_model', judge_model,
                '--openrouter_base_url', args.openrouter_base_url,
                '--openrouter_app_title', args.openrouter_app_title,
                '--progress_every', str(args.grade_progress_every),
            ],
        )
        run_script_main(
            'grading',
            grading_core.main,
            [
                '--in_jsonl', str(merged_result),
                '--out_jsonl', str(merged_graded),
                '--dataset_jsonl', args.dataset_jsonl,
                '--judge_backend', args.judge_backend,
                '--judge_model', judge_model,
                '--openrouter_base_url', args.openrouter_base_url,
                '--openrouter_app_title', args.openrouter_app_title,
                '--progress_every', str(args.grade_progress_every),
            ],
        )

    print(f'[done] run_dir={run_dir}')
    print(f'[done] layer1_result={layer1_result}')
    print(f'[done] merged_result={merged_result}')


if __name__ == '__main__':
    main()
