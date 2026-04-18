#!/usr/bin/env python3
"""Run the prose best-of-N selective-merge experiment from the April package."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

from _script_utils import PACKAGE_ROOT, run_script_main, timestamp_now
from april_version_code.common import grading as grading_core
from april_version_code.common.llm_backends import load_local_env, model_slug, require_backend_credentials
from april_version_code.methods import cluster_bank_core as cluster_core
from april_version_code.methods import prose_bestofn_core as experiment

EXPERIMENT_NAME = 'prose_bestofn_merge'
METHOD_NAME = 'oracle_doc_cluster_bank_prose_bestofn_concat_titled'
VARIANT_NAME = 'oracle_doc_cluster_bank_prose_bestofn_concat_titled_R5_top20pct_memory_vs_gold_answer'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the prose best-of-N selective-merge experiment on the packaged q50 support-only dataset.')
    parser.add_argument('--dataset-jsonl', default=str(PACKAGE_ROOT / 'data' / 'derived' / 'support_only' / 'support_only_q50_manydocs.jsonl'))
    parser.add_argument('--run-dir', default='')
    parser.add_argument('--checkpoint-dir', default='')
    parser.add_argument('--llm-backend', choices=['gemini', 'openrouter'], default='openrouter')
    parser.add_argument('--model', default='qwen/qwen3.5-35b-a3b')
    parser.add_argument('--answer-model', default='')
    parser.add_argument('--judge-backend', choices=['gemini', 'openrouter'], default='openrouter')
    parser.add_argument('--judge-model', default='')
    parser.add_argument('--openrouter-base-url', default='https://openrouter.ai/api/v1')
    parser.add_argument('--openrouter-app-title', default='AprilVersionCode')
    parser.add_argument('--embed-model', default='Qwen/Qwen3-Embedding-0.6B')
    parser.add_argument('--embed-device', default='cuda')
    parser.add_argument('--embed-batch-size', type=int, default=16)
    parser.add_argument('--doc-cluster-style', choices=['list_only', 'titled'], default='titled')
    parser.add_argument('--doc-cluster-max-queries-per-bank', type=int, default=5)
    parser.add_argument('--merge-rounds', type=int, default=5)
    parser.add_argument('--selection-fraction', type=float, default=0.20)
    parser.add_argument('--max-doc-tokens', type=int, default=12000)
    parser.add_argument('--doc-truncate-strategy', choices=['head', 'middle', 'tail'], default='head')
    parser.add_argument('--summary-temperature', type=float, default=0.0)
    parser.add_argument('--answer-temperature', type=float, default=0.0)
    parser.add_argument('--timeout-sec', type=int, default=300)
    parser.add_argument('--retries', type=int, default=5)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--progress-every', type=int, default=5)
    parser.add_argument('--grade-progress-every', type=int, default=5)
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--no-resume', action='store_false', dest='resume')
    parser.add_argument('--skip-answer', action='store_true')
    parser.add_argument('--skip-grading', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def build_default_run_dir(answer_model: str) -> Path:
    return PACKAGE_ROOT / 'runs' / 'support_only_q50' / model_slug(answer_model) / EXPERIMENT_NAME / timestamp_now()


def main() -> None:
    args = parse_args()
    load_local_env(PACKAGE_ROOT)
    backends_to_check = []
    if not args.dry_run:
        backends_to_check.append(args.llm_backend)
    if not args.skip_grading:
        backends_to_check.append(args.judge_backend)
    require_backend_credentials(backends_to_check)

    answer_model = args.answer_model or args.model
    judge_model = args.judge_model or answer_model
    run_dir = Path(args.run_dir) if args.run_dir else build_default_run_dir(answer_model)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'results').mkdir(exist_ok=True)
    (run_dir / 'graded').mkdir(exist_ok=True)
    (run_dir / 'traces').mkdir(exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (run_dir / 'checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.dataset_jsonl)
    out_jsonl = run_dir / 'results' / 'q50_prose_bestofn_merge.jsonl'
    trace_jsonl = run_dir / 'traces' / 'q50_prose_bestofn_merge.trace.jsonl'
    manifest_json = run_dir / 'q50_prose_bestofn_merge.manifest.json'
    graded_jsonl = run_dir / 'graded' / 'q50_prose_bestofn_merge.scored.jsonl'

    rows = experiment.common_slice(list(experiment.iter_jsonl(dataset_path)), args.start_index, args.limit)
    done_ids = experiment.load_done_ids(out_jsonl, skip_answer=args.skip_answer) if args.resume else set()
    mode = 'a' if args.resume else 'w'

    counter = cluster_core.base.TokenCounter('cl100k_base')
    llm_summary: Optional[Any] = None
    llm_answer: Optional[Any] = None
    embedder: Optional[Any] = None
    if not args.dry_run:
        llm_summary = experiment.build_llm(
            backend=args.llm_backend,
            model=args.model,
            timeout_sec=args.timeout_sec,
            retries=args.retries,
            counter=counter,
            base_url=args.openrouter_base_url,
        )
        if not args.skip_answer:
            llm_answer = experiment.build_llm(
                backend=args.llm_backend,
                model=answer_model,
                timeout_sec=args.timeout_sec,
                retries=args.retries,
                counter=counter,
                base_url=args.openrouter_base_url,
            )
        embedder = experiment.build_embedder(args.embed_model, args.embed_device, args.embed_batch_size)

    started = time.time()
    processed = 0
    skipped_completed = 0
    with out_jsonl.open(mode, encoding='utf-8') as fout:
        trace_handle = trace_jsonl.open(mode, encoding='utf-8')
        try:
            print(
                f"[start] dataset={dataset_path} rows={len(rows)} merge_rounds={args.merge_rounds} selection_fraction={args.selection_fraction}",
                flush=True,
            )
            for row in rows:
                qid = str(row.get('question_id', '') or '').strip()
                if not qid:
                    continue
                if qid in done_ids:
                    skipped_completed += 1
                    continue
                out_row = experiment.run_prose_multi_merge_experiment(
                    row=row,
                    llm_summary=llm_summary,
                    llm_answer=llm_answer,
                    embedder=embedder,
                    counter=counter,
                    doc_cluster_style=args.doc_cluster_style,
                    doc_cluster_max_queries_per_bank=args.doc_cluster_max_queries_per_bank,
                    max_doc_tokens=args.max_doc_tokens,
                    doc_truncate_strategy=args.doc_truncate_strategy,
                    summary_temperature=args.summary_temperature,
                    answer_temperature=args.answer_temperature,
                    merge_rounds=args.merge_rounds,
                    selection_fraction=args.selection_fraction,
                    dry_run=args.dry_run,
                    skip_answer=args.skip_answer,
                    trace_handle=trace_handle,
                    method=METHOD_NAME,
                    variant=VARIANT_NAME,
                    checkpoint_dir=checkpoint_dir,
                )
                out_row['backend'] = args.llm_backend
                out_row['openrouter_base_url'] = args.openrouter_base_url if args.llm_backend != 'gemini' else ''
                experiment.write_jsonl_row(fout, out_row)
                processed += 1
                done_ids.add(qid)
                if args.progress_every > 0 and processed % args.progress_every == 0:
                    print(
                        f"[progress] processed={processed} last_qid={qid} final_banks={len(out_row.get('final_bank_units') or [])} selected_banks={len(out_row.get('selected_final_banks') or [])}",
                        flush=True,
                    )
        finally:
            trace_handle.close()

    manifest = {
        'dataset_jsonl': str(dataset_path),
        'out_jsonl': str(out_jsonl),
        'trace_jsonl': str(trace_jsonl),
        'checkpoint_dir': str(checkpoint_dir),
        'method': METHOD_NAME,
        'variant': VARIANT_NAME,
        'llm_backend': args.llm_backend,
        'model': args.model,
        'answer_model': answer_model,
        'embed_model': args.embed_model,
        'doc_cluster_style': args.doc_cluster_style,
        'doc_cluster_max_queries_per_bank': args.doc_cluster_max_queries_per_bank,
        'merge_rounds': args.merge_rounds,
        'selection_fraction': args.selection_fraction,
        'rows_requested': len(rows),
        'rows_completed_this_run': processed,
        'rows_skipped_completed': skipped_completed,
        'runtime_sec': round(max(0.0, time.time() - started), 6),
    }
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    if not args.skip_grading:
        run_script_main(
            'grading',
            grading_core.main,
            [
                '--in_jsonl', str(out_jsonl),
                '--out_jsonl', str(graded_jsonl),
                '--dataset_jsonl', str(dataset_path),
                '--judge_backend', args.judge_backend,
                '--judge_model', judge_model,
                '--openrouter_base_url', args.openrouter_base_url,
                '--openrouter_app_title', args.openrouter_app_title,
                '--progress_every', str(args.grade_progress_every),
            ],
        )

    print(f'[done] run_dir={run_dir}')
    print(f'[done] result_jsonl={out_jsonl}')
    print(f'[done] manifest_json={manifest_json}')
    if not args.skip_grading:
        print(f'[done] graded_jsonl={graded_jsonl}')


if __name__ == '__main__':
    main()
