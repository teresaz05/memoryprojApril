#!/usr/bin/env python3
"""Official RLM prompt-doc variant

    This is the current RLM experiment we care about: raw support documents remain directly in prompt
    context, and every source document gets one additional prompt document containing its query-aware
    summary plus cluster-bank text. The copied implementation keeps the experiment behavior stable."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from dotenv import load_dotenv
from rlm import RLM

from april_version_code.common import metadata as row_metadata
from april_version_code.methods import rlm_official_core as rbase
from april_version_code.methods import rlm_docsummaryaux_helpers as docpairs


METHOD_NAME = "rlm_official_promptdoc_pairs_from_docsummaryaux"
VARIANT_NAME = METHOD_NAME


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run official RLM with raw support documents kept directly in prompt context "
            "and per-document docsummaryaux companion documents appended as additional prompt docs."
        )
    )
    ap.add_argument("--dataset_jsonl", required=True)
    ap.add_argument("--docsummaryaux_results_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--run_log_jsonl", default="")
    ap.add_argument("--backend", choices=["gemini", "openrouter"], default="openrouter")
    ap.add_argument("--model", default="qwen/qwen3.5-397b-a17b")
    ap.add_argument("--openrouter_base_url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--max_depth", type=int, default=1)
    ap.add_argument("--max_iterations", type=int, default=30)
    ap.add_argument("--completion_retries", type=int, default=2)
    ap.add_argument("--max_docs_per_query", type=int, default=0)
    ap.add_argument("--max_doc_tokens", type=int, default=12000)
    ap.add_argument("--doc_truncate_strategy", choices=["head", "middle", "tail"], default="head")
    ap.add_argument("--max_doc_chars", type=int, default=0)
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress_every", type=int, default=10)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_false", dest="resume")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_depth < 0:
        raise ValueError("--max_depth must be >= 0.")
    if args.max_iterations <= 0:
        raise ValueError("--max_iterations must be > 0.")
    if args.completion_retries < 0:
        raise ValueError("--completion_retries must be >= 0.")
    if args.max_docs_per_query < 0:
        raise ValueError("--max_docs_per_query must be >= 0.")
    if args.max_doc_tokens < 0:
        raise ValueError("--max_doc_tokens must be >= 0.")
    if args.max_doc_chars < 0:
        raise ValueError("--max_doc_chars must be >= 0.")
    if args.start_index < 0:
        raise ValueError("--start_index must be >= 0.")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0.")
    if args.progress_every < 0:
        raise ValueError("--progress_every must be >= 0.")
    if args.backend == "openrouter" and not args.openrouter_base_url.strip():
        raise ValueError("--openrouter_base_url must be non-empty for openrouter backend.")



def is_completed_row(row: Dict[str, Any]) -> bool:
    if str(row.get("runtime_error", "")).strip():
        return False
    if not str(row.get("model_answer", "")).strip():
        return False
    return True



def load_done_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    done: Set[str] = set()
    for row in rbase.iter_jsonl(path):
        qid = str(row.get("question_id", "")).strip()
        if qid and is_completed_row(row):
            done.add(qid)
    return done



def build_context_payload(
    docs: Sequence[Dict[str, Any]],
    doc_cluster_banks: Sequence[Dict[str, Any]],
    counter: rbase.TokenCounter,
    max_doc_tokens: int,
    doc_truncate_strategy: str,
    max_doc_chars: int,
) -> Tuple[List[str], List[str], int, int, List[str], int, int]:
    by_doc_id: Dict[str, Dict[str, Any]] = {}
    for row in doc_cluster_banks:
        doc_id = str(row.get("doc_id", "") or "").strip()
        if doc_id and doc_id not in by_doc_id:
            by_doc_id[doc_id] = row

    payload: List[str] = []
    context_doc_ids: List[str] = []
    source_context_doc_ids: List[str] = []
    char_truncations = 0
    token_truncations = 0
    matched_doc_ids: Set[str] = set()
    nonempty_summaries = 0
    nonempty_cluster_banks = 0

    for doc in docs:
        doc_id = str(doc.get("doc_id", "")).strip()
        if not doc_id:
            continue
        raw_block, char_truncated, token_truncated = docpairs.build_truncated_doc_block(
            doc=doc,
            counter=counter,
            max_doc_tokens=max_doc_tokens,
            doc_truncate_strategy=doc_truncate_strategy,
            max_doc_chars=max_doc_chars,
        )
        payload.append(raw_block)
        context_doc_ids.append(f"{doc_id}__raw")
        source_context_doc_ids.append(doc_id)
        char_truncations += int(char_truncated)
        token_truncations += int(token_truncated)

        source_row = by_doc_id.get(doc_id)
        companion_block = docpairs.build_companion_block(doc_id, source_row)
        payload.append(companion_block)
        context_doc_ids.append(f"{doc_id}__clusters_summary")
        if source_row is not None:
            matched_doc_ids.add(doc_id)
            if str(source_row.get("source_doc_summary", "") or "").strip():
                nonempty_summaries += 1
            if str(source_row.get("cluster_bank_text", "") or "").strip():
                nonempty_cluster_banks += 1

    unmatched_doc_ids: List[str] = []
    for row in doc_cluster_banks:
        doc_id = str(row.get("doc_id", "") or "").strip()
        if not doc_id or doc_id in matched_doc_ids:
            continue
        unmatched_doc_ids.append(doc_id)
        companion_block = docpairs.build_companion_block(doc_id, row)
        payload.append(companion_block)
        context_doc_ids.append(f"{doc_id}__clusters_summary")
        if str(row.get("source_doc_summary", "") or "").strip():
            nonempty_summaries += 1
        if str(row.get("cluster_bank_text", "") or "").strip():
            nonempty_cluster_banks += 1

    return (
        payload or ["doc_id: \ntext:\nEmpty context."],
        context_doc_ids,
        char_truncations,
        token_truncations,
        unmatched_doc_ids,
        nonempty_summaries,
        nonempty_cluster_banks,
    )



def main() -> None:
    args = parse_args()
    validate_args(args)
    load_dotenv()

    dataset_path = Path(args.dataset_jsonl)
    source_path = Path(args.docsummaryaux_results_jsonl)
    out_path = Path(args.out_jsonl)
    run_log_path = Path(args.run_log_jsonl) if args.run_log_jsonl else None
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset jsonl not found: {dataset_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Docsummaryaux results jsonl not found: {source_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if run_log_path:
        run_log_path.parent.mkdir(parents=True, exist_ok=True)

    rows_all = list(rbase.iter_jsonl(dataset_path))
    rows = rows_all[args.start_index :]
    if args.limit > 0:
        rows = rows[: args.limit]

    docsummaryaux_rows: Dict[str, Dict[str, Any]] = {}
    for row in rbase.iter_jsonl(source_path):
        qid = str(row.get("question_id", "")).strip()
        if qid:
            docsummaryaux_rows[qid] = row

    done_ids = load_done_ids(out_path) if args.resume else set()
    mode = "a" if args.resume else "w"
    if not args.resume and run_log_path and run_log_path.exists():
        run_log_path.unlink()

    token_counter = rbase.TokenCounter("cl100k_base")
    if not args.dry_run:
        backend_api_key = rbase.resolve_backend_api_key(args.backend)
        rbase.install_rlm_backend_call_timer(args.backend)
    else:
        backend_api_key = ""
    rbase.TOTAL_RLM_GEMINI_CALL_WALL_TIME_SEC = 0.0

    processed = 0
    try:
        with out_path.open(mode, encoding="utf-8") as fout:
            for row in rows:
                qid = str(row.get("question_id", "")).strip()
                if not qid or qid in done_ids:
                    continue

                question = str(row.get("question", ""))
                gold_answer = str(row.get("gold_answer", ""))
                docs = list(row.get("docs") or row.get("stream_docs") or [])
                if args.max_docs_per_query > 0:
                    docs = docs[: args.max_docs_per_query]

                started = time.time()
                raw_model_answer = ""
                model_answer = ""
                answer_extraction_mode = ""
                answer_extraction_error = ""
                runtime_error = ""
                skip_reason = ""
                usage: Dict[str, Any] = {}
                context_payload = ["doc_id: \ntext:\nEmpty context."]
                context_doc_ids: List[str] = []
                source_context_doc_ids: List[str] = []
                doc_char_truncations = 0
                doc_token_truncations = 0
                unmatched_companion_doc_ids: List[str] = []
                num_source_doc_summaries_nonempty = 0
                num_doc_cluster_banks_nonempty = 0
                source_row = docsummaryaux_rows.get(qid)
                doc_cluster_banks = list(source_row.get("doc_cluster_banks") or []) if isinstance(source_row, dict) else []

                rbase.write_run_log(
                    run_log_path,
                    "example_start",
                    {
                        "question_id": qid,
                        "variant": VARIANT_NAME,
                        "num_dataset_docs": len(docs),
                        "has_docsummaryaux_source": bool(source_row),
                    },
                )

                if source_row is None:
                    runtime_error = "docsummaryaux_source_missing"
                    skip_reason = "docsummaryaux_source_missing"
                else:
                    (
                        context_payload,
                        context_doc_ids,
                        doc_char_truncations,
                        doc_token_truncations,
                        unmatched_companion_doc_ids,
                        num_source_doc_summaries_nonempty,
                        num_doc_cluster_banks_nonempty,
                    ) = build_context_payload(
                        docs=docs,
                        doc_cluster_banks=doc_cluster_banks,
                        counter=token_counter,
                        max_doc_tokens=args.max_doc_tokens,
                        doc_truncate_strategy=args.doc_truncate_strategy,
                        max_doc_chars=args.max_doc_chars,
                    )
                    source_context_doc_ids = [str(d.get("doc_id", "")).strip() for d in docs if str(d.get("doc_id", "")).strip()]

                lm_call_wall_before = rbase.TOTAL_RLM_GEMINI_CALL_WALL_TIME_SEC
                lm_call_wall_time_sec = 0.0
                if args.dry_run:
                    raw_model_answer = "DRY_RUN"
                    model_answer = "DRY_RUN"
                    answer_extraction_mode = "dry_run"
                elif not runtime_error:
                    try:
                        rlm_kwargs: Dict[str, Any] = {
                            "backend": args.backend,
                            "backend_kwargs": {"model_name": args.model, "api_key": backend_api_key},
                            "environment": "local",
                            "max_depth": args.max_depth,
                            "max_iterations": args.max_iterations,
                            "verbose": bool(args.verbose),
                        }
                        if args.backend == "openrouter":
                            rlm_kwargs["backend_kwargs"]["base_url"] = args.openrouter_base_url
                        rlm = RLM(**rlm_kwargs)
                        completion_obj = rbase.completion_with_retries(
                            rlm_obj=rlm,
                            prompt=context_payload,
                            root_prompt=question,
                            retries=args.completion_retries,
                        )
                        raw_model_answer = str(getattr(completion_obj, "response", "") or "").strip()
                        (
                            model_answer,
                            answer_extraction_mode,
                            answer_extraction_error,
                        ) = rbase.extract_final_answer(raw_model_answer)
                        if answer_extraction_error:
                            runtime_error = f"answer_extraction_error: {answer_extraction_error}"
                            skip_reason = "answer_extraction_error"
                        usage_obj = getattr(completion_obj, "usage_summary", None)
                        usage = usage_obj.to_dict() if usage_obj else {}
                    except Exception as exc:  # noqa: BLE001
                        runtime_error = f"{type(exc).__name__}: {exc}"
                        skip_reason = "runtime_error"

                lm_call_wall_time_sec = max(
                    0.0,
                    float(rbase.TOTAL_RLM_GEMINI_CALL_WALL_TIME_SEC - lm_call_wall_before),
                )
                latency_sec = round(time.time() - started, 3)
                is_exact = rbase.exact_match(model_answer, gold_answer) if model_answer else False
                usage_totals = rbase.extract_rlm_usage_totals(usage)
                stream_doc_tokens = [token_counter.count(str(d.get("text", ""))) for d in docs]

                out_row = {
                    "question_id": qid,
                    "question": question,
                    "gold_answer": gold_answer,
                    "variant": VARIANT_NAME,
                    "method": METHOD_NAME,
                    "docsummaryaux_source_method": str(source_row.get("method", "") if source_row else ""),
                    "docsummaryaux_source_variant": str(source_row.get("variant", "") if source_row else ""),
                    "docsummaryaux_results_jsonl": str(source_path),
                    "model": args.model,
                    "backend": args.backend,
                    "max_depth": args.max_depth,
                    "max_iterations": args.max_iterations,
                    "model_answer": model_answer,
                    "raw_model_answer": raw_model_answer,
                    "answer_extraction_mode": answer_extraction_mode,
                    "answer_extraction_error": answer_extraction_error,
                    "is_exact_match": bool(is_exact),
                    "row_failed": bool(runtime_error),
                    "latency_sec": latency_sec,
                    "runtime_error": runtime_error,
                    "skip_reason": skip_reason,
                    "augmentation_mode": "prompt_doc_pairs",
                    "num_stream_docs": len(docs),
                    "num_context_docs_loaded": len(context_payload),
                    "context_doc_ids": context_doc_ids,
                    "source_context_doc_ids": source_context_doc_ids,
                    "num_raw_prompt_docs": len(source_context_doc_ids),
                    "num_companion_prompt_docs": max(0, len(context_payload) - len(source_context_doc_ids)),
                    "docpair_unmatched_source_doc_ids": unmatched_companion_doc_ids,
                    "doc_truncate_strategy": args.doc_truncate_strategy,
                    "max_doc_tokens": args.max_doc_tokens,
                    "doc_char_truncations": doc_char_truncations,
                    "doc_token_truncations": doc_token_truncations,
                    "stream_doc_tokens": stream_doc_tokens,
                    "stream_total_tokens": sum(stream_doc_tokens),
                    "num_source_doc_summaries_nonempty": num_source_doc_summaries_nonempty,
                    "num_doc_cluster_banks_nonempty": num_doc_cluster_banks_nonempty,
                    "rlm_usage": usage,
                    "rlm_usage_totals": usage_totals,
                    "total_lm_calls": usage_totals["calls"],
                    "total_lm_input_tokens": usage_totals["input_tokens"],
                    "total_lm_output_tokens": usage_totals["output_tokens"],
                    "total_lm_tokens": usage_totals["total_tokens"],
                    "lm_call_wall_time_sec": round(float(lm_call_wall_time_sec), 6),
                    "total_lm_wall_time_sec": round(float(lm_call_wall_time_sec), 6),
                    "answer_tokens": token_counter.count(model_answer),
                    "memory_text": "",
                    "memory_tokens": 0,
                    "execute_calls": 0,
                    "memory_state_present": False,
                    "dry_run": bool(args.dry_run),
                }
                row_metadata.attach_sample_metadata(out_row, row)
                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                fout.flush()
                done_ids.add(qid)

                rbase.write_run_log(
                    run_log_path,
                    "example_done",
                    {
                        "question_id": qid,
                        "variant": VARIANT_NAME,
                        "latency_sec": latency_sec,
                        "lm_call_wall_time_sec": round(float(lm_call_wall_time_sec), 6),
                        "is_exact_match": bool(is_exact),
                        "runtime_error": runtime_error,
                        "total_lm_tokens": usage_totals["total_tokens"],
                    },
                )

                processed += 1
                if args.progress_every > 0 and processed % args.progress_every == 0:
                    print(
                        f"[progress] processed={processed}/{len(rows)} qid={qid} total_lm_tokens={usage_totals['total_tokens']}",
                        flush=True,
                    )
    finally:
        if not args.dry_run:
            rbase.uninstall_rlm_backend_call_timer(args.backend)


if __name__ == "__main__":
    main()
