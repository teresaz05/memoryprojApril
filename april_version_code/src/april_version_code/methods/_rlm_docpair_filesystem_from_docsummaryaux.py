#!/usr/bin/env python3
"""Internal legacy docpair RLM runner copied for dependency completeness.

    The April package does not expose this filesystem-docpair variant as a primary experiment, but
    the copied cluster-bank core can optionally import it. Keeping the file here avoids a hidden
    dependency on the old BrowseCompV2 script tree."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from dotenv import load_dotenv
from rlm import RLM

from april_version_code.common import metadata as row_metadata
from april_version_code.methods import rlm_official_core as rbase


METHOD_NAME = "rlm_official_docpair_filesystem_from_docsummaryaux"
VARIANT_NAME = METHOD_NAME
MAX_FILENAME_COMPONENT = 80


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run official RLM with support documents exposed as local raw-doc files plus "
            "per-document docsummaryaux companion files."
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


def sanitize_filename_component(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip())
    value = value.strip("._")
    if not value:
        value = "doc"
    return value[:MAX_FILENAME_COMPONENT]


def build_truncated_doc_block(
    doc: Dict[str, Any],
    counter: rbase.TokenCounter,
    max_doc_tokens: int,
    doc_truncate_strategy: str,
    max_doc_chars: int,
) -> Tuple[str, bool, bool]:
    doc_id = str(doc.get("doc_id", "")).strip()
    txt = str(doc.get("text", "") or "")
    char_truncated = False
    token_truncated = False
    if max_doc_chars > 0 and len(txt) > max_doc_chars:
        txt = txt[:max_doc_chars]
        char_truncated = True
    if max_doc_tokens > 0 and counter.count(txt) > max_doc_tokens:
        txt = counter.truncate(txt, max_doc_tokens, strategy=doc_truncate_strategy)
        token_truncated = True
    block = "\n".join([f"doc_id: {doc_id}", "text:", txt])
    return block, char_truncated, token_truncated


def build_companion_block(doc_id: str, source_row: Optional[Dict[str, Any]]) -> str:
    lines = [
        f"doc_id: {doc_id}__clusters_summary",
        f"source_doc_id: {doc_id}",
        "",
    ]
    if source_row is None:
        lines.append("No derived summary or cluster-bank text was available for this source document.")
        return "\n".join(lines).strip()

    summary_text = str(source_row.get("source_doc_summary", "") or "").strip()
    cluster_bank_text = str(source_row.get("cluster_bank_text", "") or "").strip()
    if summary_text:
        lines.extend(["DOCUMENT_SUMMARY:", summary_text, ""])
    if cluster_bank_text:
        lines.extend(["DOCUMENT_CLUSTER_BANKS:", cluster_bank_text, ""])
    if not summary_text and not cluster_bank_text:
        lines.append("No derived summary or cluster-bank text was available for this source document.")
    return "\n".join(lines).strip()


def build_docpair_files(
    docs: Sequence[Dict[str, Any]],
    doc_cluster_banks: Sequence[Dict[str, Any]],
    counter: rbase.TokenCounter,
    max_doc_tokens: int,
    doc_truncate_strategy: str,
    max_doc_chars: int,
) -> Tuple[List[Dict[str, Any]], List[str], int, int, List[str]]:
    by_doc_id: Dict[str, Dict[str, Any]] = {}
    for row in doc_cluster_banks:
        doc_id = str(row.get("doc_id", "") or "").strip()
        if doc_id and doc_id not in by_doc_id:
            by_doc_id[doc_id] = row

    file_specs: List[Dict[str, Any]] = []
    source_context_doc_ids: List[str] = []
    char_truncations = 0
    token_truncations = 0
    used_companions: Set[str] = set()

    for idx, doc in enumerate(docs, start=1):
        doc_id = str(doc.get("doc_id", "")).strip()
        if not doc_id:
            continue
        source_context_doc_ids.append(doc_id)
        raw_block, char_truncated, token_truncated = build_truncated_doc_block(
            doc=doc,
            counter=counter,
            max_doc_tokens=max_doc_tokens,
            doc_truncate_strategy=doc_truncate_strategy,
            max_doc_chars=max_doc_chars,
        )
        char_truncations += int(char_truncated)
        token_truncations += int(token_truncated)

        source_row = by_doc_id.get(doc_id)
        if source_row is not None:
            used_companions.add(doc_id)

        safe_doc_id = sanitize_filename_component(doc_id)
        base_stem = f"doc_{idx:02d}_{safe_doc_id}"
        raw_filename = f"{base_stem}__raw.txt"
        companion_filename = f"{base_stem}__clusters_summary.txt"
        companion_block = build_companion_block(doc_id, source_row)

        file_specs.append(
            {
                "filename": raw_filename,
                "kind": "raw",
                "logical_doc_id": f"{doc_id}__raw",
                "source_doc_id": doc_id,
                "doc_idx": idx,
                "content": raw_block,
                "content_tokens": counter.count(raw_block),
            }
        )
        file_specs.append(
            {
                "filename": companion_filename,
                "kind": "clusters_summary",
                "logical_doc_id": f"{doc_id}__clusters_summary",
                "source_doc_id": doc_id,
                "doc_idx": idx,
                "content": companion_block,
                "content_tokens": counter.count(companion_block),
            }
        )

    unmatched_doc_ids: List[str] = []
    for row in doc_cluster_banks:
        doc_id = str(row.get("doc_id", "") or "").strip()
        if not doc_id or doc_id in used_companions:
            continue
        unmatched_doc_ids.append(doc_id)
        safe_doc_id = sanitize_filename_component(doc_id)
        doc_idx = int(row.get("doc_idx", 0) or 0)
        companion_filename = f"doc_{doc_idx:02d}_{safe_doc_id}__clusters_summary_orphan.txt"
        companion_block = build_companion_block(doc_id, row)
        file_specs.append(
            {
                "filename": companion_filename,
                "kind": "clusters_summary_orphan",
                "logical_doc_id": f"{doc_id}__clusters_summary",
                "source_doc_id": doc_id,
                "doc_idx": doc_idx,
                "content": companion_block,
                "content_tokens": counter.count(companion_block),
            }
        )

    return file_specs, source_context_doc_ids, char_truncations, token_truncations, unmatched_doc_ids


def build_setup_code(file_specs: Sequence[Dict[str, Any]]) -> str:
    file_map = {
        str(spec["filename"]): str(spec["content"])
        for spec in file_specs
    }
    return (
        "from pathlib import Path\n"
        f"docpair_files = {file_map!r}\n"
        "available_external_files = []\n"
        "for _filename, _contents in docpair_files.items():\n"
        "    Path(_filename).write_text(_contents, encoding='utf-8')\n"
        "    available_external_files.append(_filename)\n"
        "available_external_files.sort()\n"
    )


def build_context_payload(file_specs: Sequence[Dict[str, Any]]) -> List[str]:
    if not file_specs:
        return ["doc_id: local_file_index\ntext:\nNo local files were created for this question."]

    lines = [
        "doc_id: local_file_index",
        "text:",
        "Local files are available in the working directory.",
        "For each source document there is a raw support-document file and a derived companion file.",
        "Raw files contain the truncated support document text.",
        "Companion files contain a query-aware document summary and cluster-bank memories derived from the corresponding source document.",
        "Use raw files as primary evidence. Use companion files to locate and compress relevant evidence, then verify against raw files when needed.",
        "",
        "FILE_INDEX:",
    ]
    for spec in file_specs:
        lines.append(
            f"- {spec['filename']} (kind={spec['kind']}, source_doc_id={spec['source_doc_id']})"
        )
    return ["\n".join(lines)]


def inventory_for_output(file_specs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "filename": str(spec["filename"]),
            "kind": str(spec["kind"]),
            "logical_doc_id": str(spec["logical_doc_id"]),
            "source_doc_id": str(spec["source_doc_id"]),
            "doc_idx": int(spec["doc_idx"]),
            "content_tokens": int(spec["content_tokens"]),
        }
        for spec in file_specs
    ]


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
                setup_code = ""
                context_payload = ["doc_id: local_file_index\ntext:\nNo local files were created for this question."]
                source_context_doc_ids: List[str] = []
                file_specs: List[Dict[str, Any]] = []
                file_inventory: List[Dict[str, Any]] = []
                doc_char_truncations = 0
                doc_token_truncations = 0
                unmatched_companion_doc_ids: List[str] = []
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
                    file_specs, source_context_doc_ids, doc_char_truncations, doc_token_truncations, unmatched_companion_doc_ids = build_docpair_files(
                        docs=docs,
                        doc_cluster_banks=doc_cluster_banks,
                        counter=token_counter,
                        max_doc_tokens=args.max_doc_tokens,
                        doc_truncate_strategy=args.doc_truncate_strategy,
                        max_doc_chars=args.max_doc_chars,
                    )
                    context_payload = build_context_payload(file_specs)
                    setup_code = build_setup_code(file_specs)
                    file_inventory = inventory_for_output(file_specs)

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
                            "environment_kwargs": {"setup_code": setup_code},
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
                    "augmentation_mode": "filesystem_doc_pairs",
                    "num_stream_docs": len(docs),
                    "num_context_docs_loaded": len(context_payload),
                    "context_doc_ids": ["local_file_index"],
                    "source_context_doc_ids": source_context_doc_ids,
                    "num_raw_doc_files": sum(1 for spec in file_specs if spec["kind"] == "raw"),
                    "num_companion_doc_files": sum(
                        1 for spec in file_specs if str(spec["kind"]).startswith("clusters_summary")
                    ),
                    "num_total_docpair_files": len(file_specs),
                    "docpair_files": file_inventory,
                    "docpair_unmatched_source_doc_ids": unmatched_companion_doc_ids,
                    "doc_truncate_strategy": args.doc_truncate_strategy,
                    "max_doc_tokens": args.max_doc_tokens,
                    "doc_char_truncations": doc_char_truncations,
                    "doc_token_truncations": doc_token_truncations,
                    "stream_doc_tokens": stream_doc_tokens,
                    "stream_total_tokens": sum(stream_doc_tokens),
                    "num_source_doc_summaries_nonempty": sum(
                        1 for item in doc_cluster_banks if str(item.get("source_doc_summary", "") or "").strip()
                    ),
                    "num_doc_cluster_banks_nonempty": sum(
                        1 for item in doc_cluster_banks if str(item.get("cluster_bank_text", "") or "").strip()
                    ),
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
                        f"[progress] processed={processed}/{len(rows)} qid={qid} "
                        f"total_lm_tokens={usage_totals['total_tokens']} "
                        f"raw_files={out_row['num_raw_doc_files']} "
                        f"companion_files={out_row['num_companion_doc_files']}",
                        flush=True,
                    )
    finally:
        if not args.dry_run:
            rbase.uninstall_rlm_backend_call_timer(args.backend)

    totals = rbase.aggregate_output_totals(out_path)
    manifest = {
        "dataset_jsonl": str(dataset_path),
        "docsummaryaux_results_jsonl": str(source_path),
        "out_jsonl": str(out_path),
        "run_log_jsonl": str(run_log_path) if run_log_path else "",
        "variant": VARIANT_NAME,
        "method": METHOD_NAME,
        "backend": args.backend,
        "model": args.model,
        "openrouter_base_url": args.openrouter_base_url,
        "max_depth": args.max_depth,
        "max_iterations": args.max_iterations,
        "completion_retries": args.completion_retries,
        "max_docs_per_query": args.max_docs_per_query,
        "max_doc_tokens": args.max_doc_tokens,
        "doc_truncate_strategy": args.doc_truncate_strategy,
        "max_doc_chars": args.max_doc_chars,
        "start_index": args.start_index,
        "limit": args.limit,
        "resume": bool(args.resume),
        "dry_run": bool(args.dry_run),
        "fidelity_notes": {
            "root_prompt": "question",
            "prompt_context_format": "List[str] local file index note only",
            "raw_doc_channel": "filesystem files",
            "companion_channel": "filesystem files",
            "companion_contents": "per-document query-aware summary plus per-document cluster-bank memories",
            "file_policy": "one raw file plus one companion file per source document",
            "doc_cap_policy": {
                "max_doc_tokens": args.max_doc_tokens,
                "doc_truncate_strategy": args.doc_truncate_strategy,
                "max_doc_chars": args.max_doc_chars,
            },
        },
        "rlms_metadata": rbase.get_rlms_metadata(),
        "processed_targets": totals["rows_written_total"],
        "processed_targets_completed": totals["rows_completed_total"],
        "processed_targets_this_run": processed,
        "rows_runtime_error_total": totals["rows_runtime_error_total"],
        "usage_totals": {
            "calls": totals["total_lm_calls"],
            "input_tokens": totals["total_lm_input_tokens"],
            "output_tokens": totals["total_lm_output_tokens"],
            "total_tokens": totals["total_lm_tokens"],
            "total_wall_time_sec": round(float(totals["total_lm_wall_time_sec"]), 6),
        },
        "variant_totals": totals["variants"],
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] wrote results to {out_path}")
    print(f"[done] wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
