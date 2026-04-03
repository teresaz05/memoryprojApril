#!/usr/bin/env python3
"""Run the official RLM prompt-doc companion experiment.

This wrapper expects docsummaryaux layer-1 results as input, then feeds the raw support docs
plus one synthetic companion document per source document into the copied RLM runner.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _script_utils import PACKAGE_ROOT, run_script_main, timestamp_now
from april_version_code.common import grading as grading_core
from april_version_code.common.llm_backends import load_local_env, model_slug, require_backend_credentials
from april_version_code.methods import rlm_promptdocs_from_docsummaryaux as promptdoc_core


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the official RLM prompt-doc experiment on the q50 support-only dataset.')
    parser.add_argument('--dataset-jsonl', default=str(PACKAGE_ROOT / 'data' / 'derived' / 'support_only' / 'support_only_q50_manydocs.jsonl'))
    parser.add_argument('--docsummaryaux-results-jsonl', default='')
    parser.add_argument('--docsummaryaux-run-dir', default='')
    parser.add_argument('--run-dir', default='')
    parser.add_argument('--backend', choices=['gemini', 'openrouter'], default='openrouter')
    parser.add_argument('--model', default='qwen/qwen3.5-35b-a3b')
    parser.add_argument('--judge-backend', choices=['gemini', 'openrouter'], default='openrouter')
    parser.add_argument('--judge-model', default='')
    parser.add_argument('--openrouter-base-url', default='https://openrouter.ai/api/v1')
    parser.add_argument('--openrouter-app-title', default='AprilVersionCode')
    parser.add_argument('--doc-cluster-style', choices=['list_only', 'titled'], default='titled')
    parser.add_argument('--max-depth', type=int, default=1)
    parser.add_argument('--max-iterations', type=int, default=30)
    parser.add_argument('--completion-retries', type=int, default=2)
    parser.add_argument('--max-docs-per-query', type=int, default=0)
    parser.add_argument('--max-doc-tokens', type=int, default=12000)
    parser.add_argument('--doc-truncate-strategy', choices=['head', 'middle', 'tail'], default='head')
    parser.add_argument('--progress-every', type=int, default=1)
    parser.add_argument('--grade-progress-every', type=int, default=1)
    parser.add_argument('--skip-grading', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def resolve_docsummaryaux_jsonl(args: argparse.Namespace) -> Path:
    """Accept either a direct JSONL path or a docsummaryaux run directory."""
    if args.docsummaryaux_results_jsonl:
        return Path(args.docsummaryaux_results_jsonl)
    if args.docsummaryaux_run_dir:
        run_dir = Path(args.docsummaryaux_run_dir)
        return run_dir / 'results' / f'oracle_doc_cluster_bank_docsummaryaux_concat_{args.doc_cluster_style}_support_only_Nauto_Mfree.jsonl'
    raise ValueError('Provide --docsummaryaux-results-jsonl or --docsummaryaux-run-dir.')


def main() -> None:
    args = parse_args()
    load_local_env(PACKAGE_ROOT)
    # Dry runs can skip generation credentials, but grading still needs judge credentials if enabled.
    backends_to_check = []
    if not args.dry_run:
        backends_to_check.append(args.backend)
    if not args.skip_grading:
        backends_to_check.append(args.judge_backend)
    require_backend_credentials(backends_to_check)

    judge_model = args.judge_model or args.model
    docsummaryaux_results = resolve_docsummaryaux_jsonl(args)
    if not docsummaryaux_results.exists():
        raise FileNotFoundError(docsummaryaux_results)

    run_dir = Path(args.run_dir) if args.run_dir else (
        PACKAGE_ROOT
        / 'runs'
        / 'support_only_q50'
        / model_slug(args.model)
        / 'rlm_promptdocs_from_docsummaryaux'
        / timestamp_now()
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'results').mkdir(exist_ok=True)
    (run_dir / 'graded').mkdir(exist_ok=True)
    (run_dir / 'logs').mkdir(exist_ok=True)

    result_stub = f'rlm_official_promptdoc_pairs_from_docsummaryaux_{args.doc_cluster_style}_support_only_Nauto_Mfree'
    result_jsonl = run_dir / 'results' / f'{result_stub}.jsonl'
    run_log_jsonl = run_dir / 'logs' / f'{result_stub}_runlog.jsonl'
    graded_jsonl = run_dir / 'graded' / f'{result_stub}_scored.jsonl'

    run_script_main(
        'rlm_promptdocs_from_docsummaryaux',
        promptdoc_core.main,
        [
            '--dataset_jsonl', args.dataset_jsonl,
            '--docsummaryaux_results_jsonl', str(docsummaryaux_results),
            '--out_jsonl', str(result_jsonl),
            '--run_log_jsonl', str(run_log_jsonl),
            '--backend', args.backend,
            '--model', args.model,
            '--openrouter_base_url', args.openrouter_base_url,
            '--max_depth', str(args.max_depth),
            '--max_iterations', str(args.max_iterations),
            '--completion_retries', str(args.completion_retries),
            '--max_docs_per_query', str(args.max_docs_per_query),
            '--max_doc_tokens', str(args.max_doc_tokens),
            '--doc_truncate_strategy', args.doc_truncate_strategy,
            '--progress_every', str(args.progress_every),
            '--resume',
            *( ['--dry_run'] if args.dry_run else [] ),
        ],
    )

    if not args.skip_grading:
        run_script_main(
            'grading',
            grading_core.main,
            [
                '--in_jsonl', str(result_jsonl),
                '--out_jsonl', str(graded_jsonl),
                '--dataset_jsonl', args.dataset_jsonl,
                '--judge_backend', args.judge_backend,
                '--judge_model', judge_model,
                '--openrouter_base_url', args.openrouter_base_url,
                '--openrouter_app_title', args.openrouter_app_title,
                '--progress_every', str(args.grade_progress_every),
            ],
        )

    print(f'[done] run_dir={run_dir}')
    print(f'[done] result_jsonl={result_jsonl}')
    print(f'[done] graded_jsonl={graded_jsonl}')


if __name__ == '__main__':
    main()
