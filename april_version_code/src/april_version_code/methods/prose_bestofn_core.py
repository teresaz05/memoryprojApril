
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, TextIO, Tuple

from dotenv import load_dotenv

from april_version_code.methods import cluster_bank_core as rodsc
from april_version_code.methods import rlm_official_core as rlm_base
from april_version_code.methods import stream_oracle_assisted_dynamic_cluster_bank as cbase


CHECKPOINT_VERSION = 1

OVERLAPPING_MERGE_CLUSTER_BANKS_PLAN_PROMPT = """You are planning repeated evidence-bank merges for downstream question answering.

Goal:
- Propose evidence-grounded merge groups over INPUT_CLUSTER_BANKS for TARGET_QUERY.
- Overlap is allowed: the same bank_id may appear in multiple groups when it contains evidence relevant to multiple distinct themes.

Rules:
1. Use only TARGET_QUERY and INPUT_CLUSTER_BANKS.
2. Merge banks only if they describe the same entity, event, fact cluster, or strongly complementary answer-bearing theme.
3. Do not merge banks just because they are from the same broad topic.
4. Preserve granularity by default. If merging would create a vague or overly broad bank, keep those banks separate.
5. If one bank supports multiple distinct themes, you may include it in multiple groups.
6. If two groups share a bank because that bank supports both themes, do not collapse those groups together unless they are truly one coherent theme.
7. Banks omitted from every group will be kept as singleton groups automatically, so you do not need to mention every bank_id.
8. If banks contain conflicting evidence about the same target theme, you may still group them so the merge step can preserve attributed alternatives.
9. HEURISTIC_HIGH_PRIORITY_GROUPS lists pairs or groups that appear strongly complementary by shared anchors. Keep those members together unless there is a clear evidence-grounded reason not to.
10. SOURCE_DOCUMENT_SUMMARIES contains auxiliary source summaries for the source documents behind some banks. Use them only as auxiliary merge context.
11. Output STRICT JSON only in this schema:
   {{
     "groups": [
       {{"bank_ids": ["BANK_ID_1", "BANK_ID_2"]}}
     ]
   }}
12. No prose, no markdown, no extra keys.

TARGET_QUERY:
{target_query}

INPUT_CLUSTER_BANKS:
{bank_units}

SOURCE_DOCUMENT_SUMMARIES:
{source_doc_summaries}

HEURISTIC_HIGH_PRIORITY_GROUPS:
{heuristic_groups}
"""


BEST_OF_MANY_LAYER1_CONTROLLER_PROMPT = """You are controlling an oracle experiment that repeatedly regenerates layer-1 memory banks.

Goal:
- Decide whether another full layer-1 generation attempt is worth running.
- You may use the reported oracle embedding scores against the gold answer to decide if more exploration is worthwhile.

Rules:
1. Prefer stopping when the best score looks strong and recent attempts are no longer improving materially.
2. Prefer continuing when improvements are still arriving or current scores remain weak.
3. Respect the hard cap: if CURRENT_ATTEMPT equals MAX_ATTEMPTS, you must stop.
4. Output STRICT JSON only:
   {{
     "continue_search": 0 or 1,
     "reason": "short string <= 30 words"
   }}

QUESTION:
{question}

CURRENT_ATTEMPT:
{current_attempt}

MIN_ATTEMPTS:
{min_attempts}

MAX_ATTEMPTS:
{max_attempts}

ATTEMPT_SUMMARIES:
{attempt_summaries}
"""


BEST_OF_MANY_MERGE_CONTROLLER_PROMPT = """You are controlling an oracle experiment that repeatedly reruns merge trajectories starting from the same layer-1 banks.

Goal:
- Decide whether another merge attempt is worth running.
- You may use the reported oracle embedding scores against the gold answer to decide if more exploration is worthwhile.

Rules:
1. Prefer stopping when the best score looks strong and recent attempts are no longer improving materially.
2. Prefer continuing when improvements are still arriving or current scores remain weak.
3. Respect the hard cap: if CURRENT_ATTEMPT equals MAX_ATTEMPTS, you must stop.
4. Output STRICT JSON only:
   {{
     "continue_search": 0 or 1,
     "reason": "short string <= 30 words"
   }}

QUESTION:
{question}

CURRENT_ATTEMPT:
{current_attempt}

MIN_ATTEMPTS:
{min_attempts}

MAX_ATTEMPTS:
{max_attempts}

ATTEMPT_SUMMARIES:
{attempt_summaries}
"""


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_done_ids(path: Path, skip_answer: bool) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for row in iter_jsonl(path):
        qid = str(row.get("question_id", "") or "").strip()
        if not qid:
            continue
        if str(row.get("runtime_error", "") or "").strip():
            continue
        if not skip_answer and not str(row.get("model_answer", "") or "").strip():
            continue
        done.add(qid)
    return done


def write_jsonl_row(handle: TextIO, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    cbase.base.flush_jsonl_handle(handle)


def zero_usage_dict() -> Dict[str, Any]:
    return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "wall_time_sec": 0.0}


def normalize_usage_dict(usage: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(usage, dict):
        return zero_usage_dict()
    return {
        "calls": int(usage.get("calls", 0) or 0),
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "wall_time_sec": round(float(usage.get("wall_time_sec", 0.0) or 0.0), 6),
    }


def add_usage_dicts(lhs: Optional[Dict[str, Any]], rhs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    left = normalize_usage_dict(lhs)
    right = normalize_usage_dict(rhs)
    return {
        "calls": left["calls"] + right["calls"],
        "input_tokens": left["input_tokens"] + right["input_tokens"],
        "output_tokens": left["output_tokens"] + right["output_tokens"],
        "wall_time_sec": round(left["wall_time_sec"] + right["wall_time_sec"], 6),
    }


def completed_row_for_resume(row: Optional[Dict[str, Any]], skip_answer: bool) -> bool:
    if not isinstance(row, dict):
        return False
    if str(row.get("runtime_error", "") or "").strip():
        return False
    if skip_answer:
        return True
    return bool(str(row.get("model_answer", "") or "").strip())


def sanitize_checkpoint_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return cleaned or "unknown"


def question_checkpoint_path(checkpoint_dir: Optional[Path], qid: str) -> Optional[Path]:
    if checkpoint_dir is None or not str(qid or "").strip():
        return None
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"{sanitize_checkpoint_component(qid)}.checkpoint.json"


def load_question_checkpoint(
    checkpoint_dir: Optional[Path],
    qid: str,
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    path = question_checkpoint_path(checkpoint_dir, qid)
    if path is None or not path.exists():
        return None
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(checkpoint, dict):
        return None
    if int(checkpoint.get("version", 0) or 0) != CHECKPOINT_VERSION:
        return None
    if str(checkpoint.get("question_id", "") or "").strip() != str(qid or "").strip():
        return None
    stored_config = checkpoint.get("config")
    if not isinstance(stored_config, dict):
        return None
    for key, value in config.items():
        if stored_config.get(key) != value:
            return None
    return checkpoint


def save_question_checkpoint(
    checkpoint_dir: Optional[Path],
    qid: str,
    state: Dict[str, Any],
) -> None:
    path = question_checkpoint_path(checkpoint_dir, qid)
    if path is None:
        return
    payload = {
        "version": CHECKPOINT_VERSION,
        "question_id": str(qid or "").strip(),
        "updated_at": time.time(),
        **state,
    }
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def build_cluster_memory_text(bank_units: Sequence[Dict[str, Any]], style: str) -> str:
    # The answering agent expects the same cluster-bank text layout as the existing prose pipeline, so we rebuild the memory blob through the same helper.
    cluster_bank_map: Dict[str, Dict[str, Any]] = {}
    memory_bank_map: Dict[str, str] = {}
    selected_keys: List[str] = []
    for idx, bank_unit in enumerate(bank_units, start=1):
        cluster = dict(bank_unit.get("cluster") or {})
        memory = str(bank_unit.get("memory", "") or "").strip()
        if not cluster or not memory:
            continue
        cluster_key = str(bank_unit.get("cluster_key", "") or "").strip() or cbase.cluster_key(cluster, style)
        if cluster_key in cluster_bank_map:
            cluster_key = f"{cluster_key}__dup_{idx}"
        selected_keys.append(cluster_key)
        cluster_bank_map[cluster_key] = cluster
        memory_bank_map[cluster_key] = memory
    if not selected_keys:
        return ""
    return cbase.build_cluster_memory_blob(selected_keys, cluster_bank_map, memory_bank_map, style)


def serialize_bank_units(bank_units: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for bank_unit in bank_units:
        row = {
            "bank_id": str(bank_unit.get("bank_id", "") or ""),
            "cluster": dict(bank_unit.get("cluster") or {}),
            "memory": str(bank_unit.get("memory", "") or ""),
            "memory_tokens": int(bank_unit.get("memory_tokens", 0) or 0),
            "source_bank_ids": list(bank_unit.get("source_bank_ids") or []),
            "lineage_bank_ids": list(bank_unit.get("lineage_bank_ids") or []),
        }
        if bank_unit.get("memory_structured") is not None:
            row["memory_structured"] = bank_unit.get("memory_structured")
        rows.append(row)
    return rows


def parse_merge_groups_allow_overlap(raw: str, valid_bank_ids: Sequence[str]) -> List[List[str]]:
    valid_set = {str(bank_id) for bank_id in valid_bank_ids if str(bank_id).strip()}
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    parsed: Any = None
    try:
        parsed = json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
            except Exception:
                parsed = None

    groups_raw: List[Any] = []
    if isinstance(parsed, dict):
        groups_raw = list(parsed.get("groups") or [])
    elif isinstance(parsed, list):
        groups_raw = list(parsed)

    seen_group_keys: set[Tuple[str, ...]] = set()
    out: List[List[str]] = []
    assigned_anywhere: set[str] = set()
    for item in groups_raw:
        bank_ids_raw = item.get("bank_ids") if isinstance(item, dict) else item
        if not isinstance(bank_ids_raw, list):
            continue
        cleaned: List[str] = []
        seen_within_group: set[str] = set()
        for bank_id in bank_ids_raw:
            cleaned_id = str(bank_id or "").strip()
            if not cleaned_id or cleaned_id not in valid_set or cleaned_id in seen_within_group:
                continue
            cleaned.append(cleaned_id)
            seen_within_group.add(cleaned_id)
        if not cleaned:
            continue
        group_key = tuple(cleaned)
        if group_key in seen_group_keys:
            continue
        seen_group_keys.add(group_key)
        out.append(cleaned)
        assigned_anywhere.update(cleaned)

    # banks omitted by the planner are carried forward as singleton groups.
    for bank_id in valid_bank_ids:
        cleaned_id = str(bank_id or "").strip()
        if cleaned_id and cleaned_id not in assigned_anywhere:
            out.append([cleaned_id])
    return out


def _extract_json_obj(raw: str) -> str:
    txt = str(raw or "").strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*", "", txt.strip(), flags=re.IGNORECASE)
        txt = re.sub(r"\s*```$", "", txt.strip())
    start = txt.find("{")
    end = txt.rfind("}")
    if start >= 0 and end > start:
        return txt[start : end + 1]
    return txt


def _parse_continue_decision(raw: str, key_name: str) -> tuple[bool, str]:
    try:
        obj = json.loads(_extract_json_obj(raw))
    except Exception:
        obj = {}
    value = obj.get(key_name, 0)
    continue_search = bool(int(value)) if isinstance(value, (int, float, str)) and str(value).strip() else False
    reason = str(obj.get("reason", "") or "").strip()
    return continue_search, reason


def _build_scored_bank_rows(
    bank_units: Sequence[Dict[str, Any]],
    score_map: Dict[str, float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for bank_unit in bank_units:
        bank_id = str(bank_unit.get("bank_id", "") or "").strip()
        rows.append(
            {
                "bank_id": bank_id,
                "score": float(score_map.get(bank_id, 0.0)),
                "cluster": dict(bank_unit.get("cluster") or {}),
                "memory_tokens": int(bank_unit.get("memory_tokens", 0) or 0),
                "source_bank_ids": list(bank_unit.get("source_bank_ids") or []),
                "lineage_bank_ids": list(bank_unit.get("lineage_bank_ids") or []),
            }
        )
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows


def score_bank_unit_attempt(
    embedder: Optional[Any],
    gold_answer: str,
    bank_units: Sequence[Dict[str, Any]],
    score_fraction: float,
) -> Dict[str, Any]:
    score_map = score_bank_units_by_gold_answer(embedder, gold_answer, bank_units)
    scored_rows = _build_scored_bank_rows(bank_units, score_map)
    if not scored_rows:
        return {
            "score_map": score_map,
            "scored_rows": [],
            "keep_count": 0,
            "top_score": 0.0,
            "mean_top_score": 0.0,
            "sum_top_score": 0.0,
        }
    if score_fraction <= 0.0 or score_fraction > 1.0:
        raise ValueError("score_fraction must be in (0, 1].")
    keep_count = max(1, int(math.ceil(len(scored_rows) * score_fraction)))
    kept_rows = scored_rows[:keep_count]
    sum_top_score = float(sum(row["score"] for row in kept_rows))
    mean_top_score = float(sum_top_score / keep_count) if keep_count else 0.0
    top_score = float(kept_rows[0]["score"]) if kept_rows else 0.0
    return {
        "score_map": score_map,
        "scored_rows": scored_rows,
        "keep_count": keep_count,
        "top_score": top_score,
        "mean_top_score": mean_top_score,
        "sum_top_score": sum_top_score,
    }


def _summarize_attempts_for_controller(attempt_summaries: Sequence[Dict[str, Any]]) -> str:
    compact_rows: List[Dict[str, Any]] = []
    for row in attempt_summaries:
        compact_rows.append(
            {
                "attempt_index": int(row.get("attempt_index", 0) or 0),
                "bank_count": int(row.get("bank_count", 0) or 0),
                "keep_count": int(row.get("keep_count", 0) or 0),
                "top_score": round(float(row.get("top_score", 0.0) or 0.0), 6),
                "mean_top_score": round(float(row.get("mean_top_score", 0.0) or 0.0), 6),
                "rounds_completed": int(row.get("merge_rounds_completed", 0) or 0),
                "stop_reason": str(row.get("stop_reason", "") or ""),
            }
        )
    return json.dumps(compact_rows, ensure_ascii=False, indent=2)


def controller_should_continue_attempts(
    llm_summary: Optional[Any],
    *,
    question: str,
    attempt_summaries: Sequence[Dict[str, Any]],
    current_attempt: int,
    min_attempts: int,
    max_attempts: int,
    prompt_template: str,
    dry_run: bool,
    temperature: float,
) -> tuple[bool, str, str]:
    if current_attempt >= max_attempts:
        return False, "max_attempts_reached", ""
    if current_attempt < min_attempts:
        return True, "below_min_attempts", ""
    if dry_run or llm_summary is None:
        return False, "dry_run_stop", ""
    prompt = prompt_template.format(
        question=question,
        current_attempt=current_attempt,
        min_attempts=min_attempts,
        max_attempts=max_attempts,
        attempt_summaries=_summarize_attempts_for_controller(attempt_summaries),
    )
    raw = llm_summary.generate(prompt, temperature=temperature).strip()
    should_continue, reason = _parse_continue_decision(raw, "continue_search")
    return should_continue, reason or "controller_no_reason", raw


def answer_from_cluster_memory(
    llm_answer: Optional[Any],
    question: str,
    memory_text: str,
    temperature: float,
    dry_run: bool,
    skip_answer: bool,
) -> tuple[str, str]:
    if skip_answer:
        return "", ""
    if dry_run:
        return "DRY_RUN", ""
    try:
        if llm_answer is None:
            raise RuntimeError("Answer model is not initialized.")
        prompt = rodsc.ANSWER_FROM_DOC_CLUSTER_BANKS_PROMPT.format(
            target_query=question,
            memory_text=memory_text if memory_text else "(empty)",
        )
        return llm_answer.generate(prompt, temperature=temperature).strip(), ""
    except Exception as exc:  # noqa: BLE001
        return "", str(exc)


def build_layer1_prose_bank_state(
    question: str,
    docs: Sequence[Dict[str, Any]],
    llm_summary: Optional[Any],
    counter: rodsc.base.TokenCounter,
    doc_cluster_style: str,
    doc_cluster_max_queries_per_bank: int,
    max_doc_tokens: int,
    doc_truncate_strategy: str,
    summary_temperature: float,
    dry_run: bool,
    resume_state: Optional[Dict[str, Any]] = None,
    checkpoint_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, Dict[str, Any]]:
    resume_state = resume_state if isinstance(resume_state, dict) else {}
    doc_cluster_banks: List[Dict[str, Any]] = list(resume_state.get("doc_cluster_banks") or [])
    bank_units_all: List[Dict[str, Any]] = list(resume_state.get("bank_units_all") or [])
    kept_blocks: List[str] = list(resume_state.get("kept_blocks") or [])
    doc_truncations = int(resume_state.get("doc_truncations", 0) or 0)
    processed_doc_count = min(len(doc_cluster_banks), len(docs))

    for doc_idx, doc in enumerate(docs[processed_doc_count:], start=processed_doc_count + 1):
        doc_id = str(doc.get("doc_id", "") or "").strip()
        raw_doc_text = rodsc.base.format_doc_for_prompt(doc)
        raw_doc_tokens = counter.count(raw_doc_text)
        doc_text = raw_doc_text
        if max_doc_tokens > 0 and raw_doc_tokens > max_doc_tokens:
            doc_text = counter.truncate(raw_doc_text, max_tokens=max_doc_tokens, strategy=doc_truncate_strategy)
            doc_truncations += 1

        doc_for_cluster = rodsc.doc_with_text(doc, doc_text)
        cluster_bank_text = ""
        cluster_memory_fallbacks = 0
        cluster_error = ""
        selected_clusters: List[Dict[str, Any]] = []
        selected_cluster_keys: List[str] = []
        memory_bank: Dict[str, str] = {}
        bank_units_doc: List[Dict[str, Any]] = []

        try:
            if dry_run:
                selected_clusters = cbase.build_fallback_clusters(
                    [doc_for_cluster],
                    1,
                    doc_cluster_style,
                    doc_cluster_max_queries_per_bank,
                )
            else:
                if llm_summary is None:
                    raise RuntimeError("Summary model is not initialized.")
                selected_clusters = rodsc.generate_candidate_clusters_warm_auto(
                    llm=llm_summary,
                    warm_docs=[doc_for_cluster],
                    max_queries_per_cluster=doc_cluster_max_queries_per_bank,
                    style=doc_cluster_style,
                    temperature=summary_temperature,
                )
            if not selected_clusters:
                selected_clusters = cbase.build_fallback_clusters(
                    [doc_for_cluster],
                    1,
                    doc_cluster_style,
                    doc_cluster_max_queries_per_bank,
                )

            cluster_bank = {cbase.cluster_key(cluster, doc_cluster_style): cluster for cluster in selected_clusters}
            for cluster in selected_clusters:
                cluster_key = cbase.cluster_key(cluster, doc_cluster_style)
                selected_cluster_keys.append(cluster_key)
                if dry_run:
                    cluster_memory = doc_text
                else:
                    if llm_summary is None:
                        raise RuntimeError("Summary model is not initialized.")
                    cluster_memory = rodsc.initialize_cluster_memory_unbounded(
                        llm=llm_summary,
                        cluster=cluster,
                        document=doc_for_cluster,
                        temperature=summary_temperature,
                        style=doc_cluster_style,
                    ).strip()
                if not cluster_memory:
                    cluster_memory = doc_text
                    cluster_memory_fallbacks += 1
                memory_bank[cluster_key] = cluster_memory

            for bank_idx, cluster in enumerate(selected_clusters, start=1):
                cluster_key = selected_cluster_keys[bank_idx - 1]
                bank_text = cbase.build_cluster_memory_blob([cluster_key], cluster_bank, memory_bank, doc_cluster_style)
                bank_unit = {
                    "bank_id": f"doc{doc_idx}_bank{bank_idx}",
                    "doc_idx": doc_idx,
                    "doc_id": doc_id,
                    "bank_idx": bank_idx,
                    "cluster_key": cluster_key,
                    "cluster": cluster,
                    "memory": memory_bank.get(cluster_key, ""),
                    "memory_tokens": counter.count(memory_bank.get(cluster_key, "")),
                    "bank_text": bank_text,
                    "source_bank_ids": [f"doc{doc_idx}_bank{bank_idx}"],
                    "lineage_bank_ids": [f"doc{doc_idx}_bank{bank_idx}"],
                }
                bank_units_doc.append(bank_unit)
                bank_units_all.append(bank_unit)

            cluster_bank_text = cbase.build_cluster_memory_blob(
                selected_cluster_keys,
                cluster_bank,
                memory_bank,
                doc_cluster_style,
            )
            if cluster_bank_text:
                kept_blocks.append(rodsc.format_cluster_bank_block(doc_idx, doc_id, cluster_bank_text))
        except Exception as exc:  # noqa: BLE001
            # A per-document fallback keeps the whole experiment running even when
            # one document causes a generation/parse failure.
            cluster_error = str(exc)
            selected_clusters = cbase.build_fallback_clusters(
                [doc_for_cluster],
                1,
                doc_cluster_style,
                doc_cluster_max_queries_per_bank,
            )
            cluster_key = cbase.cluster_key(selected_clusters[0], doc_cluster_style)
            cluster_memory = doc_text
            memory_bank = {cluster_key: cluster_memory}
            selected_cluster_keys = [cluster_key]
            cluster_memory_fallbacks += 1
            bank_text = cbase.build_cluster_memory_blob(selected_cluster_keys, {cluster_key: selected_clusters[0]}, memory_bank, doc_cluster_style)
            bank_unit = {
                "bank_id": f"doc{doc_idx}_bank1",
                "doc_idx": doc_idx,
                "doc_id": doc_id,
                "bank_idx": 1,
                "cluster_key": cluster_key,
                "cluster": selected_clusters[0],
                "memory": cluster_memory,
                "memory_tokens": counter.count(cluster_memory),
                "bank_text": bank_text,
                "source_bank_ids": [f"doc{doc_idx}_bank1"],
                "lineage_bank_ids": [f"doc{doc_idx}_bank1"],
            }
            bank_units_doc = [bank_unit]
            bank_units_all.append(bank_unit)
            cluster_bank_text = bank_text
            if cluster_bank_text:
                kept_blocks.append(rodsc.format_cluster_bank_block(doc_idx, doc_id, cluster_bank_text))

        doc_cluster_banks.append(
            {
                "doc_idx": doc_idx,
                "doc_id": doc_id,
                "is_gold": bool(doc.get("is_gold", False)),
                "raw_doc_tokens": raw_doc_tokens,
                "doc_tokens_after_cap": counter.count(doc_text),
                "num_clusters": len(selected_clusters),
                "selected_clusters": selected_clusters,
                "selected_cluster_keys": selected_cluster_keys,
                "memory_bank": memory_bank,
                "cluster_bank_text": cluster_bank_text,
                "cluster_bank_tokens": counter.count(cluster_bank_text),
                "cluster_bank_empty": not bool(cluster_bank_text),
                "cluster_bank_error": cluster_error,
                "cluster_memory_fallbacks": cluster_memory_fallbacks,
                "bank_units": bank_units_doc,
            }
        )

        if checkpoint_callback is not None:
            layer1_memory_text_partial = "\n\n".join(kept_blocks).strip()
            checkpoint_callback(
                {
                    "completed_doc_count": len(doc_cluster_banks),
                    "doc_cluster_banks": doc_cluster_banks,
                    "bank_units_all": bank_units_all,
                    "kept_blocks": kept_blocks,
                    "doc_truncations": doc_truncations,
                    "layer1_memory_text": layer1_memory_text_partial,
                    "layer1_stats": {
                        "num_docs": len(docs),
                        "num_doc_truncations": doc_truncations,
                        "num_layer1_bank_units": len(bank_units_all),
                        "num_doc_cluster_banks_nonempty": sum(
                            1 for item in doc_cluster_banks if item.get("cluster_bank_text")
                        ),
                    },
                }
            )

    layer1_memory_text = "\n\n".join(kept_blocks).strip()
    stats = {
        "num_docs": len(docs),
        "num_doc_truncations": doc_truncations,
        "num_layer1_bank_units": len(bank_units_all),
        "num_doc_cluster_banks_nonempty": sum(1 for item in doc_cluster_banks if item.get("cluster_bank_text")),
    }
    return doc_cluster_banks, bank_units_all, layer1_memory_text, stats


def bank_units_from_merged_cluster_banks(
    merged_cluster_banks: Sequence[Dict[str, Any]],
    round_index: int,
    counter: rodsc.base.TokenCounter,
    style: str,
) -> List[Dict[str, Any]]:
    next_bank_units: List[Dict[str, Any]] = []
    for idx, merged_entry in enumerate(merged_cluster_banks, start=1):
        bank_id = f"round{round_index}_bank{idx}"
        cluster = dict(merged_entry.get("cluster") or {})
        memory = str(merged_entry.get("memory", "") or "").strip()
        bank_unit = {
            "bank_id": bank_id,
            "doc_idx": 0,
            "doc_id": "",
            "bank_idx": idx,
            "cluster_key": cbase.cluster_key(cluster, style) if cluster else "",
            "cluster": cluster,
            "memory": memory,
            "memory_tokens": counter.count(memory),
            "bank_text": str(merged_entry.get("cluster_bank_text", "") or ""),
            "source_bank_ids": list(merged_entry.get("source_bank_ids") or []),
            "lineage_bank_ids": list(merged_entry.get("lineage_bank_ids") or []),
        }
        if merged_entry.get("memory_structured") is not None:
            bank_unit["memory_structured"] = merged_entry.get("memory_structured")
        next_bank_units.append(bank_unit)
    return next_bank_units


def run_overlap_merge_round(
    question: str,
    bank_units_in: Sequence[Dict[str, Any]],
    llm_summary: Optional[Any],
    counter: rodsc.base.TokenCounter,
    doc_cluster_banks: Sequence[Dict[str, Any]],
    doc_cluster_style: str,
    doc_cluster_max_queries_per_bank: int,
    summary_temperature: float,
    dry_run: bool,
    round_index: int,
    trace_handle: Optional[TextIO] = None,
    resume_state: Optional[Dict[str, Any]] = None,
    checkpoint_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    valid_bank_ids = [str(bank_unit.get("bank_id", "") or "").strip() for bank_unit in bank_units_in if str(bank_unit.get("bank_id", "") or "").strip()]
    if not valid_bank_ids:
        return [], {"round_index": round_index, "planner_raw": "", "merge_groups": [], "num_output_banks": 0}

    resume_state = resume_state if isinstance(resume_state, dict) else {}
    planner_raw = ""
    emit_planner_trace = True
    forced_merge_groups = list(resume_state.get("forced_merge_groups") or [])
    merge_groups = list(resume_state.get("merge_groups") or [])
    if resume_state.get("round_index") == round_index and merge_groups:
        planner_raw = str(resume_state.get("planner_raw", "") or "")
        emit_planner_trace = False
    else:
        forced_merge_groups = rodsc.build_forced_merge_groups(bank_units_in, doc_cluster_style)
        if dry_run:
            merge_groups = forced_merge_groups or [[bank_id] for bank_id in valid_bank_ids]
        else:
            if llm_summary is None:
                raise RuntimeError("Summary model is not initialized.")
            planner_prompt = OVERLAPPING_MERGE_CLUSTER_BANKS_PLAN_PROMPT.format(
                target_query=question,
                bank_units=rodsc.format_bank_units_for_merge_planner(bank_units_in, doc_cluster_style) or "(empty)",
                source_doc_summaries=rodsc.format_source_doc_summaries_for_merge(bank_units_in, doc_cluster_banks) or "(none)",
                heuristic_groups=rodsc.format_heuristic_groups(forced_merge_groups),
            )
            planner_raw = llm_summary.generate(planner_prompt, temperature=summary_temperature).strip()
            merge_groups = parse_merge_groups_allow_overlap(planner_raw, valid_bank_ids)
            if not merge_groups:
                merge_groups = forced_merge_groups or [[bank_id] for bank_id in valid_bank_ids]

    if trace_handle is not None and emit_planner_trace:
        write_jsonl_row(
            trace_handle,
            {
                "phase": "merge_planner",
                "round_index": round_index,
                "question": question,
                "planner_raw": planner_raw,
                "heuristic_high_priority_groups": forced_merge_groups,
                "merge_groups": merge_groups,
            },
        )

    bank_unit_by_id = {str(bank_unit.get("bank_id", "") or ""): bank_unit for bank_unit in bank_units_in}
    merged_cluster_bank_map: Dict[str, Dict[str, Any]] = {}
    merged_memory_bank_map: Dict[str, str] = {}
    merged_cluster_banks: List[Dict[str, Any]] = [dict(row) for row in list(resume_state.get("merged_cluster_banks") or [])]

    for merged_entry in merged_cluster_banks:
        merged_cluster = dict(merged_entry.get("cluster") or {})
        merged_memory = str(merged_entry.get("memory", "") or "").strip()
        merged_key = str(merged_entry.get("cluster_key", "") or "").strip() or cbase.cluster_key(merged_cluster, doc_cluster_style)
        merged_entry["cluster_key"] = merged_key
        merged_cluster_bank_map[merged_key] = merged_cluster
        merged_memory_bank_map[merged_key] = merged_memory

    start_group_index = len(merged_cluster_banks) + 1

    for group_idx, group_bank_ids in enumerate(merge_groups, start=1):
        if group_idx < start_group_index:
            continue
        group_bank_units = [bank_unit_by_id[bank_id] for bank_id in group_bank_ids if bank_id in bank_unit_by_id]
        if not group_bank_units:
            continue

        merged_bank_raw = ""
        merged_bank_validation_error = ""
        if len(group_bank_units) == 1:
            only = group_bank_units[0]
            merged_bank = {
                "cluster": dict(only.get("cluster") or {}),
                "memory": str(only.get("memory", "") or "").strip(),
            }
            if only.get("memory_structured") is not None:
                merged_bank["memory_structured"] = only.get("memory_structured")
            merged_bank_raw = "passthrough_singleton"
        elif dry_run:
            merged_bank = rodsc.fallback_merged_bank(
                group_bank_units,
                style=doc_cluster_style,
                max_queries_per_cluster=doc_cluster_max_queries_per_bank,
                structured_memory=False,
            )
        else:
            if llm_summary is None:
                raise RuntimeError("Summary model is not initialized.")
            style_rule, _ = cbase.style_rule_and_schema(doc_cluster_style)
            merged_bank_schema = '"title": "...", "queries": ["...", "..."], "memory": "..."' if doc_cluster_style == "titled" else '"queries": ["...", "..."], "memory": "..."'
            merge_exec_prompt = rodsc.MERGE_CLUSTER_BANKS_EXECUTION_PROMPT.format(
                target_query=question,
                max_queries_per_cluster=doc_cluster_max_queries_per_bank,
                group_banks=rodsc.format_bank_units_for_merge_planner(group_bank_units, doc_cluster_style) or "(empty)",
                source_doc_summaries=rodsc.format_source_doc_summaries_for_merge(group_bank_units, doc_cluster_banks) or "(none)",
                style_rule=style_rule,
                merged_bank_schema=merged_bank_schema,
            )
            merged_bank_raw = llm_summary.generate(merge_exec_prompt, temperature=summary_temperature).strip()
            parsed_merged_bank = rodsc.parse_merged_bank_json(
                merged_bank_raw,
                style=doc_cluster_style,
                max_queries_per_cluster=doc_cluster_max_queries_per_bank,
                structured_memory=False,
            )
            if parsed_merged_bank is not None:
                is_valid_merge, merged_bank_validation_error = rodsc.validate_merged_bank_against_sources(
                    parsed_merged_bank,
                    group_bank_units,
                    doc_cluster_style,
                )
                if not is_valid_merge:
                    parsed_merged_bank = None
            merged_bank = parsed_merged_bank or rodsc.fallback_merged_bank(
                group_bank_units,
                style=doc_cluster_style,
                max_queries_per_cluster=doc_cluster_max_queries_per_bank,
                structured_memory=False,
            )

        merged_cluster = dict(merged_bank.get("cluster") or {})
        merged_memory = str(merged_bank.get("memory", "") or "").strip()
        merged_key = cbase.cluster_key(merged_cluster, doc_cluster_style)
        if merged_key in merged_cluster_bank_map:
            merged_key = f"{merged_key}__round_{round_index}_group_{group_idx}"
        merged_cluster_bank_map[merged_key] = merged_cluster
        merged_memory_bank_map[merged_key] = merged_memory

        lineage_bank_ids: List[str] = []
        seen_lineage: set[str] = set()
        for bank_unit in group_bank_units:
            for lineage_id in list(bank_unit.get("lineage_bank_ids") or [bank_unit.get("bank_id")]):
                cleaned_id = str(lineage_id or "").strip()
                if cleaned_id and cleaned_id not in seen_lineage:
                    lineage_bank_ids.append(cleaned_id)
                    seen_lineage.add(cleaned_id)

        merged_cluster_bank = {
            "merged_bank_id": f"round{round_index}_group_{group_idx}",
            "source_bank_ids": list(group_bank_ids),
            "lineage_bank_ids": lineage_bank_ids,
            "group_size": len(group_bank_ids),
            "cluster_key": merged_key,
            "cluster": merged_cluster,
            "memory": merged_memory,
            "memory_tokens": counter.count(merged_memory),
            "cluster_bank_text": cbase.build_cluster_memory_blob(
                [merged_key],
                merged_cluster_bank_map,
                merged_memory_bank_map,
                doc_cluster_style,
            ),
            "merge_bank_raw": merged_bank_raw,
            "merge_bank_validation_error": merged_bank_validation_error,
        }
        if merged_bank.get("memory_structured") is not None:
            merged_cluster_bank["memory_structured"] = merged_bank.get("memory_structured")
        merged_cluster_banks.append(merged_cluster_bank)

        if trace_handle is not None:
            write_jsonl_row(
                trace_handle,
                {
                    "phase": "merge_execution",
                    "round_index": round_index,
                    "question": question,
                    "merged_bank_id": merged_cluster_bank["merged_bank_id"],
                    "source_bank_ids": list(group_bank_ids),
                    "lineage_bank_ids": lineage_bank_ids,
                    "merge_bank_raw": merged_bank_raw,
                    "merge_bank_validation_error": merged_bank_validation_error,
                    "merged_cluster_bank": merged_cluster_bank,
                },
            )

        if checkpoint_callback is not None:
            checkpoint_callback(
                {
                    "round_index": round_index,
                    "planner_raw": planner_raw,
                    "merge_groups": merge_groups,
                    "forced_merge_groups": forced_merge_groups,
                    "merged_cluster_banks": merged_cluster_banks,
                }
            )

    round_summary = {
        "round_index": round_index,
        "planner_raw": planner_raw,
        "merge_groups": merge_groups,
        "num_input_banks": len(bank_units_in),
        "num_output_banks": len(merged_cluster_banks),
        "forced_merge_groups": forced_merge_groups,
    }
    return merged_cluster_banks, round_summary


def score_bank_units_by_gold_answer(
    embedder: Optional[Any],
    gold_answer: str,
    bank_units: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    selected_keys = [str(bank_unit.get("bank_id", "") or "").strip() for bank_unit in bank_units if str(bank_unit.get("bank_id", "") or "").strip()]
    memory_bank = {
        str(bank_unit.get("bank_id", "") or "").strip(): str(bank_unit.get("memory", "") or "").strip()
        for bank_unit in bank_units
        if str(bank_unit.get("bank_id", "") or "").strip()
    }
    return cbase.score_clusters_by_memory_text(
        embedder=embedder,
        target_text=gold_answer,
        selected_keys=selected_keys,
        memory_bank=memory_bank,
    )


def select_top_fraction_bank_units(
    bank_units: Sequence[Dict[str, Any]],
    score_map: Dict[str, float],
    fraction: float,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    if not bank_units:
        return [], [], 0
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError("selection fraction must be in (0, 1].")

    scored_rows: List[Dict[str, Any]] = []
    for bank_unit in bank_units:
        bank_id = str(bank_unit.get("bank_id", "") or "").strip()
        scored_rows.append(
            {
                "bank_id": bank_id,
                "score": float(score_map.get(bank_id, 0.0)),
                "cluster": dict(bank_unit.get("cluster") or {}),
                "memory_tokens": int(bank_unit.get("memory_tokens", 0) or 0),
                "source_bank_ids": list(bank_unit.get("source_bank_ids") or []),
                "lineage_bank_ids": list(bank_unit.get("lineage_bank_ids") or []),
            }
        )
    scored_rows.sort(key=lambda row: row["score"], reverse=True)
    keep_count = max(1, int(math.ceil(len(bank_units) * fraction)))
    keep_ids = {row["bank_id"] for row in scored_rows[:keep_count]}
    selected_bank_units = [bank_unit for bank_unit in bank_units if str(bank_unit.get("bank_id", "") or "").strip() in keep_ids]
    return selected_bank_units, scored_rows, keep_count


def run_adaptive_merge_trajectory(
    *,
    question: str,
    initial_bank_units: Sequence[Dict[str, Any]],
    llm_summary: Optional[Any],
    counter: rodsc.base.TokenCounter,
    doc_cluster_banks: Sequence[Dict[str, Any]],
    doc_cluster_style: str,
    doc_cluster_max_queries_per_bank: int,
    summary_temperature: float,
    dry_run: bool,
    max_merge_rounds: int,
    trace_handle: Optional[TextIO] = None,
    trace_prefix: str = "",
    resume_state: Optional[Dict[str, Any]] = None,
    checkpoint_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    resume_state = resume_state if isinstance(resume_state, dict) else {}
    current_bank_units = list(resume_state.get("current_bank_units") or initial_bank_units)
    merge_round_summaries: List[Dict[str, Any]] = list(resume_state.get("merge_round_summaries") or [])
    final_merged_cluster_banks: List[Dict[str, Any]] = list(resume_state.get("final_merged_cluster_banks") or [])
    stop_reason = "max_merge_rounds_reached"
    active_round_state = resume_state.get("active_round") if isinstance(resume_state, dict) else None

    for round_index in range(len(merge_round_summaries) + 1, max_merge_rounds + 1):
        if not current_bank_units:
            stop_reason = "no_banks_remaining"
            break
        if len(current_bank_units) <= 1:
            stop_reason = "singleton_bank_remaining"
            break

        round_resume_state = None
        if isinstance(active_round_state, dict) and int(active_round_state.get("round_index", 0) or 0) == round_index:
            round_resume_state = active_round_state

        def merge_round_checkpoint_callback(active_round_payload: Dict[str, Any]) -> None:
            if checkpoint_callback is None:
                return
            checkpoint_callback(
                {
                    "current_bank_units": current_bank_units,
                    "merge_round_summaries": merge_round_summaries,
                    "final_merged_cluster_banks": final_merged_cluster_banks,
                    "active_round": dict(active_round_payload),
                }
            )

        merged_cluster_banks, round_summary = run_overlap_merge_round(
            question=question,
            bank_units_in=current_bank_units,
            llm_summary=llm_summary,
            counter=counter,
            doc_cluster_banks=doc_cluster_banks,
            doc_cluster_style=doc_cluster_style,
            doc_cluster_max_queries_per_bank=doc_cluster_max_queries_per_bank,
            summary_temperature=summary_temperature,
            dry_run=dry_run,
            round_index=round_index,
            trace_handle=trace_handle,
            resume_state=round_resume_state,
            checkpoint_callback=merge_round_checkpoint_callback,
        )
        merge_round_summaries.append(round_summary)
        final_merged_cluster_banks = merged_cluster_banks
        next_bank_units = bank_units_from_merged_cluster_banks(
            merged_cluster_banks,
            round_index=round_index,
            counter=counter,
            style=doc_cluster_style,
        )
        if trace_handle is not None:
            write_jsonl_row(
                trace_handle,
                {
                    "phase": "adaptive_merge_round_summary",
                    "trace_prefix": trace_prefix,
                    "round_index": round_index,
                    "num_input_banks": len(current_bank_units),
                    "num_output_banks": len(next_bank_units),
                    "round_summary": round_summary,
                },
            )

        if not next_bank_units:
            current_bank_units = []
            stop_reason = "empty_post_merge"
            if checkpoint_callback is not None:
                checkpoint_callback(
                    {
                        "current_bank_units": current_bank_units,
                        "merge_round_summaries": merge_round_summaries,
                        "final_merged_cluster_banks": final_merged_cluster_banks,
                        "active_round": None,
                    }
                )
            break

        effective_merge_happened = len(next_bank_units) < len(current_bank_units)
        current_bank_units = next_bank_units
        if checkpoint_callback is not None:
            checkpoint_callback(
                {
                    "current_bank_units": current_bank_units,
                    "merge_round_summaries": merge_round_summaries,
                    "final_merged_cluster_banks": final_merged_cluster_banks,
                    "active_round": None,
                }
            )
        if not effective_merge_happened:
            stop_reason = "planner_chose_no_further_reduction"
            break

    return {
        "final_bank_units": current_bank_units,
        "merge_round_summaries": merge_round_summaries,
        "final_merged_cluster_banks": final_merged_cluster_banks,
        "merge_rounds_completed": len(merge_round_summaries),
        "stop_reason": stop_reason,
    }


def run_prose_oracle_bestofmany_experiment(
    *,
    row: Dict[str, Any],
    llm_summary: Optional[Any],
    llm_answer: Optional[Any],
    embedder: Optional[Any],
    counter: rodsc.base.TokenCounter,
    doc_cluster_style: str,
    doc_cluster_max_queries_per_bank: int,
    max_doc_tokens: int,
    doc_truncate_strategy: str,
    summary_temperature: float,
    answer_temperature: float,
    max_layer1_attempts: int,
    min_layer1_attempts: int,
    max_merge_attempts: int,
    min_merge_attempts: int,
    max_merge_rounds: int,
    score_fraction: float,
    dry_run: bool,
    skip_answer: bool,
    trace_handle: Optional[TextIO],
    method: str,
    variant: str,
    checkpoint_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    question = str(row.get("question", "") or "")
    qid = str(row.get("question_id", "") or "").strip()
    gold_answer = str(row.get("gold_answer", "") or "")
    docs = list(row.get("docs") or row.get("stream_docs") or [])
    checkpoint_config = {
        "question": question,
        "method": method,
        "variant": variant,
        "doc_cluster_style": doc_cluster_style,
        "doc_cluster_max_queries_per_bank": int(doc_cluster_max_queries_per_bank),
        "max_doc_tokens": int(max_doc_tokens),
        "doc_truncate_strategy": doc_truncate_strategy,
        "max_layer1_attempts": int(max_layer1_attempts),
        "min_layer1_attempts": int(min_layer1_attempts),
        "max_merge_attempts": int(max_merge_attempts),
        "min_merge_attempts": int(min_merge_attempts),
        "max_merge_rounds": int(max_merge_rounds),
        "score_fraction": float(score_fraction),
        "skip_answer": bool(skip_answer),
    }
    checkpoint = load_question_checkpoint(checkpoint_dir, qid, checkpoint_config)
    if checkpoint and trace_handle is not None:
        write_jsonl_row(
            trace_handle,
            {
                "phase": "checkpoint_resume",
                "question_id": qid,
                "question": question,
                "status": str(checkpoint.get("status", "") or ""),
                "checkpoint_path": str(question_checkpoint_path(checkpoint_dir, qid)) if checkpoint_dir is not None else "",
            },
        )

    final_result_checkpoint = checkpoint.get("final_result") if isinstance(checkpoint, dict) else None
    if completed_row_for_resume(final_result_checkpoint, skip_answer):
        return dict(final_result_checkpoint)

    summary_before = rodsc.usage_snapshot(llm_summary)
    answer_before = rodsc.usage_snapshot(llm_answer)
    started = time.time()

    layer1_search_state = checkpoint.get("layer1_search_state") if isinstance(checkpoint, dict) else None
    merge_search_state = checkpoint.get("merge_search_state") if isinstance(checkpoint, dict) else None
    selection_state = checkpoint.get("selection_state") if isinstance(checkpoint, dict) else None
    summary_usage_completed = normalize_usage_dict(checkpoint.get("summary_usage_completed") if isinstance(checkpoint, dict) else None)
    answer_usage_completed = normalize_usage_dict(checkpoint.get("answer_usage_completed") if isinstance(checkpoint, dict) else None)

    def current_summary_usage() -> Dict[str, Any]:
        return add_usage_dicts(summary_usage_completed, rodsc.usage_delta(summary_before, rodsc.usage_snapshot(llm_summary)))

    def current_answer_usage() -> Dict[str, Any]:
        return add_usage_dicts(answer_usage_completed, rodsc.usage_delta(answer_before, rodsc.usage_snapshot(llm_answer)))

    def persist_checkpoint(status: str) -> None:
        save_question_checkpoint(
            checkpoint_dir,
            qid,
            {
                "status": status,
                "config": checkpoint_config,
                "summary_usage_completed": current_summary_usage(),
                "answer_usage_completed": current_answer_usage(),
                "layer1_search_state": layer1_search_state,
                "merge_search_state": merge_search_state,
                "selection_state": selection_state,
                "final_result": final_result_checkpoint,
            },
        )

    layer1_attempt_records: List[Dict[str, Any]] = list((layer1_search_state or {}).get("attempt_records") or [])
    best_layer1_record: Optional[Dict[str, Any]] = None
    for attempt_record in layer1_attempt_records:
        if best_layer1_record is None or (
            attempt_record["mean_top_score"],
            attempt_record["top_score"],
            -attempt_record["attempt_index"],
        ) > (
            best_layer1_record["mean_top_score"],
            best_layer1_record["top_score"],
            -best_layer1_record["attempt_index"],
        ):
            best_layer1_record = attempt_record

    for attempt_index in range(len(layer1_attempt_records) + 1, max_layer1_attempts + 1):
        active_attempt_state = None
        if isinstance(layer1_search_state, dict):
            candidate_active = layer1_search_state.get("active_attempt")
            if isinstance(candidate_active, dict) and int(candidate_active.get("attempt_index", 0) or 0) == attempt_index:
                active_attempt_state = candidate_active

        def layer1_attempt_checkpoint_callback(layer1_state: Dict[str, Any]) -> None:
            nonlocal layer1_search_state
            layer1_search_state = {
                "attempt_records": layer1_attempt_records,
                "active_attempt": {
                    "attempt_index": attempt_index,
                    "layer1_state": dict(layer1_state),
                },
            }
            persist_checkpoint("layer1_attempt_in_progress")

        doc_cluster_banks, bank_units_all, layer1_memory_text, layer1_stats = build_layer1_prose_bank_state(
            question=question,
            docs=docs,
            llm_summary=llm_summary,
            counter=counter,
            doc_cluster_style=doc_cluster_style,
            doc_cluster_max_queries_per_bank=doc_cluster_max_queries_per_bank,
            max_doc_tokens=max_doc_tokens,
            doc_truncate_strategy=doc_truncate_strategy,
            summary_temperature=summary_temperature,
            dry_run=dry_run,
            resume_state=(active_attempt_state or {}).get("layer1_state") if isinstance(active_attempt_state, dict) else None,
            checkpoint_callback=layer1_attempt_checkpoint_callback,
        )
        score_info = score_bank_unit_attempt(embedder, gold_answer, bank_units_all, score_fraction=score_fraction)
        attempt_record = {
            "attempt_index": attempt_index,
            "bank_count": len(bank_units_all),
            "keep_count": score_info["keep_count"],
            "top_score": score_info["top_score"],
            "mean_top_score": score_info["mean_top_score"],
            "sum_top_score": score_info["sum_top_score"],
            "score_rows": score_info["scored_rows"],
            "score_map": score_info["score_map"],
            "doc_cluster_banks": doc_cluster_banks,
            "bank_units_all": bank_units_all,
            "layer1_memory_text": layer1_memory_text,
            "layer1_stats": layer1_stats,
        }
        layer1_attempt_records.append(attempt_record)
        if trace_handle is not None:
            write_jsonl_row(
                trace_handle,
                {
                    "phase": "layer1_attempt_complete",
                    "question_id": qid,
                    "attempt_index": attempt_index,
                    "bank_count": len(bank_units_all),
                    "keep_count": score_info["keep_count"],
                    "top_score": score_info["top_score"],
                    "mean_top_score": score_info["mean_top_score"],
                    "layer1_stats": layer1_stats,
                },
            )
        if best_layer1_record is None or (
            attempt_record["mean_top_score"],
            attempt_record["top_score"],
            -attempt_record["attempt_index"],
        ) > (
            best_layer1_record["mean_top_score"],
            best_layer1_record["top_score"],
            -best_layer1_record["attempt_index"],
        ):
            best_layer1_record = attempt_record

        should_continue, continue_reason, controller_raw = controller_should_continue_attempts(
            llm_summary,
            question=question,
            attempt_summaries=layer1_attempt_records,
            current_attempt=attempt_index,
            min_attempts=min_layer1_attempts,
            max_attempts=max_layer1_attempts,
            prompt_template=BEST_OF_MANY_LAYER1_CONTROLLER_PROMPT,
            dry_run=dry_run,
            temperature=summary_temperature,
        )
        attempt_record["controller_continue"] = bool(should_continue)
        attempt_record["controller_reason"] = continue_reason
        attempt_record["controller_raw"] = controller_raw
        layer1_search_state = {
            "attempt_records": layer1_attempt_records,
            "active_attempt": None,
        }
        persist_checkpoint("layer1_attempt_complete")
        if trace_handle is not None:
            write_jsonl_row(
                trace_handle,
                {
                    "phase": "layer1_attempt_controller",
                    "question_id": qid,
                    "attempt_index": attempt_index,
                    "continue_search": bool(should_continue),
                    "reason": continue_reason,
                    "controller_raw": controller_raw,
                },
            )
        if not should_continue:
            break

    if best_layer1_record is None:
        raise RuntimeError("No layer1 attempts were produced.")

    merge_attempt_records: List[Dict[str, Any]] = list((merge_search_state or {}).get("attempt_records") or [])
    best_merge_record: Optional[Dict[str, Any]] = None
    for attempt_record in merge_attempt_records:
        if best_merge_record is None or (
            attempt_record["mean_top_score"],
            attempt_record["top_score"],
            -attempt_record["attempt_index"],
        ) > (
            best_merge_record["mean_top_score"],
            best_merge_record["top_score"],
            -best_merge_record["attempt_index"],
        ):
            best_merge_record = attempt_record
    initial_bank_units = list(best_layer1_record["bank_units_all"])
    selected_doc_cluster_banks = list(best_layer1_record["doc_cluster_banks"])
    selected_layer1_memory_text = str(best_layer1_record["layer1_memory_text"] or "")
    selected_layer1_stats = dict(best_layer1_record["layer1_stats"] or {})

    for attempt_index in range(len(merge_attempt_records) + 1, max_merge_attempts + 1):
        active_attempt_state = None
        if isinstance(merge_search_state, dict):
            candidate_active = merge_search_state.get("active_attempt")
            if isinstance(candidate_active, dict) and int(candidate_active.get("attempt_index", 0) or 0) == attempt_index:
                active_attempt_state = candidate_active

        def merge_attempt_checkpoint_callback(merge_state: Dict[str, Any]) -> None:
            nonlocal merge_search_state
            merge_search_state = {
                "selected_layer1_attempt_index": int(best_layer1_record["attempt_index"]),
                "attempt_records": merge_attempt_records,
                "active_attempt": {
                    "attempt_index": attempt_index,
                    "merge_state": dict(merge_state),
                },
            }
            persist_checkpoint("merge_attempt_in_progress")

        merge_result = run_adaptive_merge_trajectory(
            question=question,
            initial_bank_units=initial_bank_units,
            llm_summary=llm_summary,
            counter=counter,
            doc_cluster_banks=selected_doc_cluster_banks,
            doc_cluster_style=doc_cluster_style,
            doc_cluster_max_queries_per_bank=doc_cluster_max_queries_per_bank,
            summary_temperature=summary_temperature,
            dry_run=dry_run,
            max_merge_rounds=max_merge_rounds,
            trace_handle=trace_handle,
            trace_prefix=f"merge_attempt_{attempt_index}",
            resume_state=(active_attempt_state or {}).get("merge_state") if isinstance(active_attempt_state, dict) else None,
            checkpoint_callback=merge_attempt_checkpoint_callback,
        )
        final_bank_units = list(merge_result["final_bank_units"])
        score_info = score_bank_unit_attempt(embedder, gold_answer, final_bank_units, score_fraction=score_fraction)
        attempt_record = {
            "attempt_index": attempt_index,
            "bank_count": len(final_bank_units),
            "keep_count": score_info["keep_count"],
            "top_score": score_info["top_score"],
            "mean_top_score": score_info["mean_top_score"],
            "sum_top_score": score_info["sum_top_score"],
            "score_rows": score_info["scored_rows"],
            "score_map": score_info["score_map"],
            "final_bank_units": final_bank_units,
            "merge_round_summaries": list(merge_result["merge_round_summaries"]),
            "final_merged_cluster_banks": list(merge_result["final_merged_cluster_banks"]),
            "merge_rounds_completed": int(merge_result["merge_rounds_completed"]),
            "stop_reason": str(merge_result["stop_reason"] or ""),
        }
        merge_attempt_records.append(attempt_record)
        if trace_handle is not None:
            write_jsonl_row(
                trace_handle,
                {
                    "phase": "merge_attempt_complete",
                    "question_id": qid,
                    "attempt_index": attempt_index,
                    "bank_count": len(final_bank_units),
                    "keep_count": score_info["keep_count"],
                    "top_score": score_info["top_score"],
                    "mean_top_score": score_info["mean_top_score"],
                    "merge_rounds_completed": attempt_record["merge_rounds_completed"],
                    "stop_reason": attempt_record["stop_reason"],
                },
            )
        if best_merge_record is None or (
            attempt_record["mean_top_score"],
            attempt_record["top_score"],
            -attempt_record["attempt_index"],
        ) > (
            best_merge_record["mean_top_score"],
            best_merge_record["top_score"],
            -best_merge_record["attempt_index"],
        ):
            best_merge_record = attempt_record

        should_continue, continue_reason, controller_raw = controller_should_continue_attempts(
            llm_summary,
            question=question,
            attempt_summaries=merge_attempt_records,
            current_attempt=attempt_index,
            min_attempts=min_merge_attempts,
            max_attempts=max_merge_attempts,
            prompt_template=BEST_OF_MANY_MERGE_CONTROLLER_PROMPT,
            dry_run=dry_run,
            temperature=summary_temperature,
        )
        attempt_record["controller_continue"] = bool(should_continue)
        attempt_record["controller_reason"] = continue_reason
        attempt_record["controller_raw"] = controller_raw
        merge_search_state = {
            "selected_layer1_attempt_index": int(best_layer1_record["attempt_index"]),
            "attempt_records": merge_attempt_records,
            "active_attempt": None,
        }
        persist_checkpoint("merge_attempt_complete")
        if trace_handle is not None:
            write_jsonl_row(
                trace_handle,
                {
                    "phase": "merge_attempt_controller",
                    "question_id": qid,
                    "attempt_index": attempt_index,
                    "continue_search": bool(should_continue),
                    "reason": continue_reason,
                    "controller_raw": controller_raw,
                },
            )
        if not should_continue:
            break

    if best_merge_record is None:
        raise RuntimeError("No merge attempts were produced.")

    final_bank_units = list(best_merge_record["final_bank_units"])
    final_merged_cluster_banks = list(best_merge_record["final_merged_cluster_banks"])
    merge_round_summaries = list(best_merge_record["merge_round_summaries"])
    memory_similarity_scores = dict(best_merge_record["score_map"])
    selected_bank_scores = list(best_merge_record["score_rows"])
    selection_keep_count = int(best_merge_record["keep_count"])
    if isinstance(selection_state, dict):
        memory_text = str(selection_state.get("memory_text", "") or "")
        memory_tokens = int(selection_state.get("memory_tokens", counter.count(memory_text)) or counter.count(memory_text))
    else:
        memory_text = build_cluster_memory_text(final_bank_units, doc_cluster_style)
        memory_tokens = counter.count(memory_text)
        selection_state = {
            "memory_text": memory_text,
            "memory_tokens": memory_tokens,
        }
        persist_checkpoint("selection_complete")

    model_answer, answer_error = answer_from_cluster_memory(
        llm_answer=llm_answer,
        question=question,
        memory_text=memory_text,
        temperature=answer_temperature,
        dry_run=dry_run,
        skip_answer=skip_answer,
    )

    summary_usage = rodsc.usage_delta(summary_before, rodsc.usage_snapshot(llm_summary))
    answer_usage = rodsc.usage_delta(answer_before, rodsc.usage_snapshot(llm_answer))
    total_calls = int(summary_usage["calls"] + answer_usage["calls"])
    total_in = int(summary_usage["input_tokens"] + answer_usage["input_tokens"])
    total_out = int(summary_usage["output_tokens"] + answer_usage["output_tokens"])
    total_wall = float(summary_usage["wall_time_sec"] + answer_usage["wall_time_sec"])

    out_row = {
        "question_id": qid,
        "question": question,
        "gold_answer": gold_answer,
        "dataset_type": str(row.get("dataset_type", "") or ""),
        "num_support_docs": int(row.get("num_support_docs", 0) or 0),
        "num_noise_docs": int(row.get("num_noise_docs", 0) or 0),
        "num_docs": len(docs),
        "method": method,
        "variant": variant,
        "model": getattr(llm_summary, "model", "") if llm_summary is not None else "",
        "answer_model": getattr(llm_answer, "model", "") if llm_answer is not None else "",
        "model_answer": model_answer,
        "runtime_error": "",
        "answer_error": answer_error,
        "is_exact_match": bool(rlm_base.exact_match(model_answer, gold_answer) if model_answer else False),
        "merge_rounds_requested": int(max_merge_rounds),
        "merge_rounds_completed": int(best_merge_record["merge_rounds_completed"]),
        "selection_metric": "oracle_best_of_many_mean_top_bank_similarity",
        "selection_fraction": score_fraction,
        "selection_keep_count": selection_keep_count,
        "memory_text": memory_text,
        "memory_tokens": memory_tokens,
        "layer1_memory_text": selected_layer1_memory_text,
        "layer1_memory_tokens": counter.count(selected_layer1_memory_text),
        "doc_cluster_style": doc_cluster_style,
        "doc_cluster_max_queries_per_bank": doc_cluster_max_queries_per_bank,
        "max_doc_tokens": max_doc_tokens,
        "doc_truncate_strategy": doc_truncate_strategy,
        "summary_lm_usage": summary_usage,
        "answer_lm_usage": answer_usage,
        "total_lm_calls": total_calls,
        "total_lm_input_tokens": total_in,
        "total_lm_output_tokens": total_out,
        "total_lm_tokens": total_in + total_out,
        "total_lm_wall_time_sec": round(total_wall, 6),
        "runtime_sec": round(max(0.0, time.time() - started), 6),
        "layer1_stats": selected_layer1_stats,
        "doc_cluster_banks": selected_doc_cluster_banks,
        "layer1_bank_units": serialize_bank_units(initial_bank_units),
        "merge_round_summaries": merge_round_summaries,
        "final_bank_units": serialize_bank_units(final_bank_units),
        "selected_final_banks": serialize_bank_units(final_bank_units),
        "selected_final_bank_scores": selected_bank_scores,
        "final_memory_similarity_scores": memory_similarity_scores,
        "final_merged_cluster_banks": final_merged_cluster_banks,
        "layer1_attempt_summaries": [
            {
                "attempt_index": rec["attempt_index"],
                "bank_count": rec["bank_count"],
                "keep_count": rec["keep_count"],
                "top_score": rec["top_score"],
                "mean_top_score": rec["mean_top_score"],
                "controller_continue": rec.get("controller_continue"),
                "controller_reason": rec.get("controller_reason", ""),
            }
            for rec in layer1_attempt_records
        ],
        "selected_layer1_attempt_index": int(best_layer1_record["attempt_index"]),
        "selected_layer1_attempt_top_score": float(best_layer1_record["top_score"]),
        "selected_layer1_attempt_mean_top_score": float(best_layer1_record["mean_top_score"]),
        "merge_attempt_summaries": [
            {
                "attempt_index": rec["attempt_index"],
                "bank_count": rec["bank_count"],
                "keep_count": rec["keep_count"],
                "top_score": rec["top_score"],
                "mean_top_score": rec["mean_top_score"],
                "merge_rounds_completed": rec["merge_rounds_completed"],
                "stop_reason": rec["stop_reason"],
                "controller_continue": rec.get("controller_continue"),
                "controller_reason": rec.get("controller_reason", ""),
            }
            for rec in merge_attempt_records
        ],
        "selected_merge_attempt_index": int(best_merge_record["attempt_index"]),
        "selected_merge_attempt_top_score": float(best_merge_record["top_score"]),
        "selected_merge_attempt_mean_top_score": float(best_merge_record["mean_top_score"]),
        "selected_merge_attempt_stop_reason": str(best_merge_record["stop_reason"]),
        "max_layer1_attempts": int(max_layer1_attempts),
        "max_merge_attempts": int(max_merge_attempts),
        "max_merge_rounds": int(max_merge_rounds),
    }
    final_result_checkpoint = out_row
    if completed_row_for_resume(out_row, skip_answer):
        persist_checkpoint("completed")
    else:
        persist_checkpoint("answer_incomplete")
    return out_row


def run_prose_multi_merge_experiment(
    row: Dict[str, Any],
    llm_summary: Optional[Any],
    llm_answer: Optional[Any],
    embedder: Optional[Any],
    counter: rodsc.base.TokenCounter,
    doc_cluster_style: str,
    doc_cluster_max_queries_per_bank: int,
    max_doc_tokens: int,
    doc_truncate_strategy: str,
    summary_temperature: float,
    answer_temperature: float,
    merge_rounds: int,
    selection_fraction: Optional[float],
    dry_run: bool,
    skip_answer: bool,
    trace_handle: Optional[TextIO],
    method: str,
    variant: str,
    checkpoint_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    question = str(row.get("question", "") or "")
    qid = str(row.get("question_id", "") or "").strip()
    gold_answer = str(row.get("gold_answer", "") or "")
    docs = list(row.get("docs") or row.get("stream_docs") or [])
    checkpoint_config = {
        "question": question,
        "method": method,
        "variant": variant,
        "merge_rounds": int(merge_rounds),
        "selection_fraction": selection_fraction,
        "doc_cluster_style": doc_cluster_style,
        "doc_cluster_max_queries_per_bank": int(doc_cluster_max_queries_per_bank),
        "max_doc_tokens": int(max_doc_tokens),
        "doc_truncate_strategy": doc_truncate_strategy,
        "skip_answer": bool(skip_answer),
    }
    checkpoint = load_question_checkpoint(checkpoint_dir, qid, checkpoint_config)
    if checkpoint and trace_handle is not None:
        write_jsonl_row(
            trace_handle,
            {
                "phase": "checkpoint_resume",
                "question_id": qid,
                "question": question,
                "status": str(checkpoint.get("status", "") or ""),
                "checkpoint_path": str(question_checkpoint_path(checkpoint_dir, qid)) if checkpoint_dir is not None else "",
            },
        )

    final_result_checkpoint = checkpoint.get("final_result") if isinstance(checkpoint, dict) else None
    if completed_row_for_resume(final_result_checkpoint, skip_answer):
        return dict(final_result_checkpoint)

    summary_before = rodsc.usage_snapshot(llm_summary)
    answer_before = rodsc.usage_snapshot(llm_answer)
    started = time.time()

    checkpoint_layer1_state = checkpoint.get("layer1_state") if isinstance(checkpoint, dict) else None
    checkpoint_merge_state = checkpoint.get("merge_state") if isinstance(checkpoint, dict) else None
    checkpoint_selection_state = checkpoint.get("selection_state") if isinstance(checkpoint, dict) else None
    summary_usage_completed = normalize_usage_dict(checkpoint.get("summary_usage_completed") if isinstance(checkpoint, dict) else None)
    answer_usage_completed = normalize_usage_dict(checkpoint.get("answer_usage_completed") if isinstance(checkpoint, dict) else None)

    def current_summary_usage() -> Dict[str, Any]:
        return add_usage_dicts(summary_usage_completed, rodsc.usage_delta(summary_before, rodsc.usage_snapshot(llm_summary)))

    def current_answer_usage() -> Dict[str, Any]:
        return add_usage_dicts(answer_usage_completed, rodsc.usage_delta(answer_before, rodsc.usage_snapshot(llm_answer)))

    def persist_checkpoint(status: str) -> None:
        save_question_checkpoint(
            checkpoint_dir,
            qid,
            {
                "status": status,
                "config": checkpoint_config,
                "summary_usage_completed": current_summary_usage(),
                "answer_usage_completed": current_answer_usage(),
                "layer1_state": checkpoint_layer1_state,
                "merge_state": checkpoint_merge_state,
                "selection_state": checkpoint_selection_state,
                "final_result": final_result_checkpoint,
            },
        )

    if isinstance(checkpoint_layer1_state, dict) and checkpoint_layer1_state.get("completed_doc_count") == len(docs):
        doc_cluster_banks = list(checkpoint_layer1_state.get("doc_cluster_banks") or [])
        bank_units_all = list(checkpoint_layer1_state.get("bank_units_all") or [])
        layer1_memory_text = str(checkpoint_layer1_state.get("layer1_memory_text", "") or "")
        layer1_stats = dict(checkpoint_layer1_state.get("layer1_stats") or {})
    else:
        def layer1_checkpoint_callback(layer1_state: Dict[str, Any]) -> None:
            nonlocal checkpoint_layer1_state
            checkpoint_layer1_state = dict(layer1_state)
            persist_checkpoint("layer1_in_progress")

        doc_cluster_banks, bank_units_all, layer1_memory_text, layer1_stats = build_layer1_prose_bank_state(
            question=question,
            docs=docs,
            llm_summary=llm_summary,
            counter=counter,
            doc_cluster_style=doc_cluster_style,
            doc_cluster_max_queries_per_bank=doc_cluster_max_queries_per_bank,
            max_doc_tokens=max_doc_tokens,
            doc_truncate_strategy=doc_truncate_strategy,
            summary_temperature=summary_temperature,
            dry_run=dry_run,
            resume_state=checkpoint_layer1_state if isinstance(checkpoint_layer1_state, dict) else None,
            checkpoint_callback=layer1_checkpoint_callback,
        )
        checkpoint_layer1_state = {
            "completed_doc_count": len(docs),
            "doc_cluster_banks": doc_cluster_banks,
            "bank_units_all": bank_units_all,
            "kept_blocks": [rodsc.format_cluster_bank_block(item.get("doc_idx", 0), item.get("doc_id", ""), item.get("cluster_bank_text", "")) for item in doc_cluster_banks if item.get("cluster_bank_text")],
            "doc_truncations": int(layer1_stats.get("num_doc_truncations", 0) or 0),
            "layer1_memory_text": layer1_memory_text,
            "layer1_stats": layer1_stats,
        }
        persist_checkpoint("layer1_complete")

    merge_round_summaries: List[Dict[str, Any]] = list((checkpoint_merge_state or {}).get("merge_round_summaries") or [])
    final_merged_cluster_banks: List[Dict[str, Any]] = list((checkpoint_merge_state or {}).get("final_merged_cluster_banks") or [])
    current_bank_units = list((checkpoint_merge_state or {}).get("current_bank_units") or bank_units_all)
    active_round_state = (checkpoint_merge_state or {}).get("active_round") if isinstance(checkpoint_merge_state, dict) else None
    if len(merge_round_summaries) < merge_rounds or (isinstance(active_round_state, dict) and active_round_state):
        checkpoint_selection_state = None
    runtime_error = ""
    try:
        for round_index in range(len(merge_round_summaries) + 1, merge_rounds + 1):
            round_resume_state = None
            if isinstance(checkpoint_merge_state, dict):
                active_round = checkpoint_merge_state.get("active_round")
                if isinstance(active_round, dict) and int(active_round.get("round_index", 0) or 0) == round_index:
                    round_resume_state = active_round

            def merge_round_checkpoint_callback(active_round_state: Dict[str, Any]) -> None:
                nonlocal checkpoint_merge_state
                checkpoint_merge_state = {
                    "current_bank_units": current_bank_units,
                    "merge_round_summaries": merge_round_summaries,
                    "final_merged_cluster_banks": final_merged_cluster_banks,
                    "active_round": dict(active_round_state),
                }
                persist_checkpoint("merge_round_in_progress")

            merged_cluster_banks, round_summary = run_overlap_merge_round(
                question=question,
                bank_units_in=current_bank_units,
                llm_summary=llm_summary,
                counter=counter,
                doc_cluster_banks=doc_cluster_banks,
                doc_cluster_style=doc_cluster_style,
                doc_cluster_max_queries_per_bank=doc_cluster_max_queries_per_bank,
                summary_temperature=summary_temperature,
                dry_run=dry_run,
                round_index=round_index,
                trace_handle=trace_handle,
                resume_state=round_resume_state,
                checkpoint_callback=merge_round_checkpoint_callback,
            )
            merge_round_summaries.append(round_summary)
            final_merged_cluster_banks = merged_cluster_banks
            current_bank_units = bank_units_from_merged_cluster_banks(
                merged_cluster_banks,
                round_index=round_index,
                counter=counter,
                style=doc_cluster_style,
            )
            checkpoint_merge_state = {
                "current_bank_units": current_bank_units,
                "merge_round_summaries": merge_round_summaries,
                "final_merged_cluster_banks": final_merged_cluster_banks,
                "active_round": None,
            }
            persist_checkpoint("merge_round_complete")
            if not current_bank_units:
                break
    except Exception as exc:  # noqa: BLE001
        runtime_error = f"merge_failed: {type(exc).__name__}: {exc}"
        final_merged_cluster_banks = list((checkpoint_merge_state or {}).get("final_merged_cluster_banks") or final_merged_cluster_banks)
        current_bank_units = list((checkpoint_merge_state or {}).get("current_bank_units") or bank_units_all)
        persist_checkpoint("merge_failed")

    if isinstance(checkpoint_selection_state, dict):
        final_bank_units = list(checkpoint_selection_state.get("final_bank_units") or current_bank_units)
        selected_bank_units = list(checkpoint_selection_state.get("selected_bank_units") or final_bank_units)
        selected_bank_scores = list(checkpoint_selection_state.get("selected_bank_scores") or [])
        selection_keep_count = int(checkpoint_selection_state.get("selection_keep_count", len(selected_bank_units)) or len(selected_bank_units))
        memory_similarity_scores = dict(checkpoint_selection_state.get("memory_similarity_scores") or {})
        memory_text = str(checkpoint_selection_state.get("memory_text", "") or "")
        memory_tokens = int(checkpoint_selection_state.get("memory_tokens", counter.count(memory_text)) or counter.count(memory_text))
    else:
        final_bank_units = current_bank_units
        selected_bank_units = list(final_bank_units)
        selected_bank_scores: List[Dict[str, Any]] = []
        selection_keep_count = len(selected_bank_units)
        memory_similarity_scores: Dict[str, float] = {}
        if final_bank_units:
            memory_similarity_scores = score_bank_units_by_gold_answer(embedder, gold_answer, final_bank_units)
            if selection_fraction is not None:
                selected_bank_units, selected_bank_scores, selection_keep_count = select_top_fraction_bank_units(
                    final_bank_units,
                    memory_similarity_scores,
                    selection_fraction,
                )
            else:
                selected_bank_scores = [
                    {
                        "bank_id": str(bank_unit.get("bank_id", "") or ""),
                        "score": float(memory_similarity_scores.get(str(bank_unit.get("bank_id", "") or ""), 0.0)),
                        "cluster": dict(bank_unit.get("cluster") or {}),
                        "memory_tokens": int(bank_unit.get("memory_tokens", 0) or 0),
                        "source_bank_ids": list(bank_unit.get("source_bank_ids") or []),
                        "lineage_bank_ids": list(bank_unit.get("lineage_bank_ids") or []),
                    }
                    for bank_unit in final_bank_units
                ]
                selected_bank_scores.sort(key=lambda row: row["score"], reverse=True)

        memory_text = build_cluster_memory_text(selected_bank_units, doc_cluster_style)
        memory_tokens = counter.count(memory_text)
        checkpoint_selection_state = {
            "final_bank_units": final_bank_units,
            "selected_bank_units": selected_bank_units,
            "selected_bank_scores": selected_bank_scores,
            "selection_keep_count": selection_keep_count,
            "memory_similarity_scores": memory_similarity_scores,
            "memory_text": memory_text,
            "memory_tokens": memory_tokens,
        }
        persist_checkpoint("selection_complete")

    model_answer, answer_error = answer_from_cluster_memory(
        llm_answer=llm_answer,
        question=question,
        memory_text=memory_text,
        temperature=answer_temperature,
        dry_run=dry_run,
        skip_answer=skip_answer,
    )

    summary_usage = current_summary_usage()
    answer_usage = current_answer_usage()
    total_calls = int(summary_usage["calls"] + answer_usage["calls"])
    total_in = int(summary_usage["input_tokens"] + answer_usage["input_tokens"])
    total_out = int(summary_usage["output_tokens"] + answer_usage["output_tokens"])
    total_wall = float(summary_usage["wall_time_sec"] + answer_usage["wall_time_sec"])

    out_row = {
        "question_id": qid,
        "question": question,
        "gold_answer": gold_answer,
        "dataset_type": str(row.get("dataset_type", "") or ""),
        "num_support_docs": int(row.get("num_support_docs", 0) or 0),
        "num_noise_docs": int(row.get("num_noise_docs", 0) or 0),
        "num_docs": len(docs),
        "method": method,
        "variant": variant,
        "model": getattr(llm_summary, "model", "") if llm_summary is not None else "",
        "answer_model": getattr(llm_answer, "model", "") if llm_answer is not None else "",
        "model_answer": model_answer,
        "runtime_error": runtime_error,
        "answer_error": answer_error,
        "is_exact_match": bool(rlm_base.exact_match(model_answer, gold_answer) if model_answer else False),
        "merge_rounds_requested": merge_rounds,
        "merge_rounds_completed": len(merge_round_summaries),
        "selection_metric": "memory_vs_gold_answer",
        "selection_fraction": selection_fraction,
        "selection_keep_count": selection_keep_count,
        "memory_text": memory_text,
        "memory_tokens": memory_tokens,
        "layer1_memory_text": layer1_memory_text,
        "layer1_memory_tokens": counter.count(layer1_memory_text),
        "doc_cluster_style": doc_cluster_style,
        "doc_cluster_max_queries_per_bank": doc_cluster_max_queries_per_bank,
        "max_doc_tokens": max_doc_tokens,
        "doc_truncate_strategy": doc_truncate_strategy,
        "summary_lm_usage": summary_usage,
        "answer_lm_usage": answer_usage,
        "total_lm_calls": total_calls,
        "total_lm_input_tokens": total_in,
        "total_lm_output_tokens": total_out,
        "total_lm_tokens": total_in + total_out,
        "total_lm_wall_time_sec": round(total_wall, 6),
        "runtime_sec": round(max(0.0, time.time() - started), 6),
        "layer1_stats": layer1_stats,
        "doc_cluster_banks": doc_cluster_banks,
        "layer1_bank_units": serialize_bank_units(bank_units_all),
        "merge_round_summaries": merge_round_summaries,
        "final_bank_units": serialize_bank_units(final_bank_units),
        "selected_final_banks": serialize_bank_units(selected_bank_units),
        "selected_final_bank_scores": selected_bank_scores,
        "final_memory_similarity_scores": memory_similarity_scores,
        "final_merged_cluster_banks": final_merged_cluster_banks,
    }

    if completed_row_for_resume(out_row, skip_answer):
        final_result_checkpoint = out_row
        persist_checkpoint("completed")
    else:
        persist_checkpoint("answer_incomplete")
    return out_row


def build_llm(
    backend: str,
    model: str,
    timeout_sec: int,
    retries: int,
    counter: rodsc.base.TokenCounter,
    base_url: str,
) -> Any:
    if backend == "gemini":
        return rodsc.make_llm(
            backend=backend,
            model=model,
            retry_policy=rodsc.base.RetryPolicy(retries=retries),
            timeout_sec=timeout_sec,
            counter=counter,
            base_url=base_url,
            http_referer="",
            app_title="",
        )
    if backend not in {"openrouter", "openai_compat"}:
        raise RuntimeError(f"Unsupported backend: {backend}")
    return rodsc.make_llm(
        backend=backend,
        model=model,
        retry_policy=rodsc.base.RetryPolicy(retries=retries),
        timeout_sec=timeout_sec,
        counter=counter,
        base_url=base_url,
        http_referer="",
        app_title="",
    )


def build_embedder(model_name: str, device: str, batch_size: int) -> Any:
    return cbase.base.QwenEmbedder(model_name=model_name, device=device, batch_size=batch_size)


def common_slice(rows: Sequence[Dict[str, Any]], start_index: int, limit: int) -> List[Dict[str, Any]]:
    sliced = list(rows[start_index:])
    if limit > 0:
        sliced = sliced[:limit]
    return sliced


def ensure_loaded_env() -> None:
    load_dotenv()
