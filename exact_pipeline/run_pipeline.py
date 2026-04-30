#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

from best_of_many import (
    build_embedder,
    build_llm,
    common_slice,
    ensure_loaded_env,
    iter_jsonl,
    load_done_ids,
    run_prose_oracle_bestofmany_experiment,
    write_jsonl_row,
)
import layer1_merge as rodsc

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET = ROOT / "data" / "browsecomp_plus_support_only_q50_main.jsonl"
DEFAULT_OUT = ROOT / "runs" / "q50_prose_oracle_bestofmany" / "results" / "q50_prose_oracle_bestofmany.jsonl"
DEFAULT_TRACE = ROOT / "runs" / "q50_prose_oracle_bestofmany" / "traces" / "q50_prose_oracle_bestofmany.trace.jsonl"
DEFAULT_MANIFEST = ROOT / "runs" / "q50_prose_oracle_bestofmany" / "q50_prose_oracle_bestofmany.manifest.json"

METHOD_NAME = "oracle_doc_cluster_bank_bestofn_pipeline"


def determine_variant_name(args: argparse.Namespace) -> str:
    style_suffix = f"_{args.doc_cluster_style}"
    if args.skip_merge:
        return f"oracle_doc_cluster_bank_l1bon_nomerge{style_suffix}"
    if args.max_layer1_attempts <= 1 and args.max_merge_attempts <= 1:
        return f"oracle_doc_cluster_bank_nobon{style_suffix}"
    if args.max_layer1_attempts <= 1:
        return f"oracle_doc_cluster_bank_s1{style_suffix}"
    return f"oracle_doc_cluster_bank_full{style_suffix}"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run the q50 oracle best-of-many experiment: regenerate layer-1 multiple times, "
            "select the best attempt by gold-answer embedding score, then rerun merge trajectories "
            "multiple times from that layer-1 state and select the best merged attempt."
        )
    )
    ap.add_argument("--dataset_jsonl", default=str(DEFAULT_DATASET))
    ap.add_argument("--out_jsonl", default=str(DEFAULT_OUT))
    ap.add_argument("--trace_jsonl", default=str(DEFAULT_TRACE))
    ap.add_argument("--manifest_json", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--checkpoint_dir", default="")
    ap.add_argument("--backend", choices=["gemini", "openrouter", "openai_compat"], default="openai_compat")
    ap.add_argument("--model", default="qwen/qwen3.5-35b-a3b")
    ap.add_argument("--answer_model", default="")
    ap.add_argument("--openai_base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--embed_model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--embed_device", default="cuda")
    ap.add_argument("--embed_batch_size", type=int, default=16)
    ap.add_argument("--doc_cluster_style", choices=["list_only", "titled"], default="titled")
    ap.add_argument("--doc_cluster_max_queries_per_bank", type=int, default=5)
    ap.add_argument("--max_layer1_attempts", type=int, default=5)
    ap.add_argument("--min_layer1_attempts", type=int, default=1)
    ap.add_argument(
        "--single_layer1_attempt",
        action="store_true",
        help=(
            "Disable repeated layer-1 regeneration and run exactly one initial "
            "cluster-bank generation attempt, while still allowing repeated merge attempts."
        ),
    )
    ap.add_argument("--max_merge_attempts", type=int, default=5)
    ap.add_argument("--min_merge_attempts", type=int, default=1)
    ap.add_argument("--max_merge_rounds", type=int, default=5)
    ap.add_argument(
        "--skip_merge",
        action="store_true",
        help=(
            "Skip merge attempts entirely and answer directly from the selected layer-1 banks. "
            "Useful for layer1 best-of-n ablations without any merge stage."
        ),
    )
    ap.add_argument("--score_fraction", type=float, default=0.20)
    ap.add_argument("--max_doc_tokens", type=int, default=12000)
    ap.add_argument("--doc_truncate_strategy", choices=["head", "middle", "tail"], default="head")
    ap.add_argument("--summary_temperature", type=float, default=0.0)
    ap.add_argument("--answer_temperature", type=float, default=0.0)
    ap.add_argument("--timeout_sec", type=int, default=300)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress_every", type=int, default=5)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_false", dest="resume")
    ap.add_argument("--skip_answer", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ensure_loaded_env()

    if args.single_layer1_attempt:
        args.min_layer1_attempts = 1
        args.max_layer1_attempts = 1
    variant_name = determine_variant_name(args)

    dataset_path = Path(args.dataset_jsonl)
    out_path = Path(args.out_jsonl)
    trace_path = Path(args.trace_jsonl) if args.trace_jsonl else None
    manifest_path = Path(args.manifest_json) if args.manifest_json else None
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset jsonl not found: {dataset_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rows = common_slice(list(iter_jsonl(dataset_path)), args.start_index, args.limit)
    done_ids = load_done_ids(out_path, skip_answer=args.skip_answer) if args.resume else set()
    mode = "a" if args.resume else "w"

    counter = rodsc.base.TokenCounter("cl100k_base")
    llm_summary: Optional[Any] = None
    llm_answer: Optional[Any] = None
    embedder: Optional[Any] = None
    if not args.dry_run:
        llm_summary = build_llm(
            backend=args.backend,
            model=args.model,
            timeout_sec=args.timeout_sec,
            retries=args.retries,
            counter=counter,
            base_url=args.openai_base_url,
        )
        if not args.skip_answer:
            llm_answer = build_llm(
                backend=args.backend,
                model=args.answer_model or args.model,
                timeout_sec=args.timeout_sec,
                retries=args.retries,
                counter=counter,
                base_url=args.openai_base_url,
            )
        embedder = build_embedder(args.embed_model, args.embed_device, args.embed_batch_size)

    started = time.time()
    processed = 0
    skipped_completed = 0
    with out_path.open(mode, encoding="utf-8") as fout:
        trace_handle = trace_path.open(mode, encoding="utf-8") if trace_path is not None else None
        try:
            print(
                f"[start] dataset={dataset_path} rows={len(rows)} "
                f"max_layer1_attempts={args.max_layer1_attempts} "
                f"max_merge_attempts={args.max_merge_attempts} "
                f"max_merge_rounds={args.max_merge_rounds} "
                f"score_fraction={args.score_fraction}",
                flush=True,
            )
            for row in rows:
                qid = str(row.get("question_id", "") or "").strip()
                if not qid:
                    continue
                if qid in done_ids:
                    skipped_completed += 1
                    continue
                out_row = run_prose_oracle_bestofmany_experiment(
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
                    max_layer1_attempts=args.max_layer1_attempts,
                    min_layer1_attempts=args.min_layer1_attempts,
                    max_merge_attempts=args.max_merge_attempts,
                    min_merge_attempts=args.min_merge_attempts,
                    max_merge_rounds=args.max_merge_rounds,
                    score_fraction=args.score_fraction,
                    skip_merge=args.skip_merge,
                    dry_run=args.dry_run,
                    skip_answer=args.skip_answer,
                    trace_handle=trace_handle,
                    method=METHOD_NAME,
                    variant=variant_name,
                    checkpoint_dir=checkpoint_dir,
                )
                out_row["backend"] = args.backend
                out_row["openai_base_url"] = args.openai_base_url if args.backend != "gemini" else ""
                write_jsonl_row(fout, out_row)
                processed += 1
                done_ids.add(qid)
                if args.progress_every > 0 and processed % args.progress_every == 0:
                    print(
                        f"[progress] processed={processed} last_qid={qid} "
                        f"layer1_attempt={out_row.get('selected_layer1_attempt_index')} "
                        f"merge_attempt={out_row.get('selected_merge_attempt_index')} "
                        f"merge_rounds_completed={out_row.get('merge_rounds_completed')} "
                        f"final_banks={len(out_row.get('final_bank_units') or [])}",
                        flush=True,
                    )
        finally:
            if trace_handle is not None:
                trace_handle.close()

    manifest = {
        "dataset_jsonl": str(dataset_path),
        "out_jsonl": str(out_path),
        "trace_jsonl": str(trace_path) if trace_path is not None else "",
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else "",
        "method": METHOD_NAME,
        "variant": variant_name,
        "backend": args.backend,
        "model": args.model,
        "answer_model": args.answer_model or args.model,
        "openai_base_url": args.openai_base_url if args.backend != "gemini" else "",
        "embed_model": args.embed_model,
        "doc_cluster_style": args.doc_cluster_style,
        "doc_cluster_max_queries_per_bank": args.doc_cluster_max_queries_per_bank,
        "max_layer1_attempts": args.max_layer1_attempts,
        "min_layer1_attempts": args.min_layer1_attempts,
        "max_merge_attempts": args.max_merge_attempts,
        "min_merge_attempts": args.min_merge_attempts,
        "max_merge_rounds": args.max_merge_rounds,
        "skip_merge": args.skip_merge,
        "score_fraction": args.score_fraction,
        "rows_requested": len(rows),
        "rows_completed_this_run": processed,
        "rows_skipped_completed": skipped_completed,
        "runtime_sec": round(max(0.0, time.time() - started), 6),
    }
    if manifest_path is not None:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(out_path)
    if manifest_path is not None:
        print(manifest_path)


if __name__ == "__main__":
    main()
