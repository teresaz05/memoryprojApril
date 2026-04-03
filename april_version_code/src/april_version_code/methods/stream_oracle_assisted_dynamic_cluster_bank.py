#!/usr/bin/env python3
"""Dynamic cluster-bank utilities copied into the April package.

    The cluster-bank experiments are thin wrappers around this shared implementation. Keeping the
    file nearly identical to the original runner reduces the risk of accidental behavior drift."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from dotenv import load_dotenv

from april_version_code.methods import stream_oracle_assisted_dynamic_bank_with_summary as base
from april_version_code.common import metadata as row_metadata


CLUSTER_GENERATION_WARM_PROMPT = """You are generating candidate clusters of possible future user questions from a warm-start document prefix.

You are given WARM_START_DOCUMENTS (the first z streamed documents).
Generate exactly NUM_CLUSTERS candidate CLUSTERS.

A cluster is one coherent information-seeking direction containing one or more related factual questions.
The number of questions per cluster is up to you, but keep each cluster compact and coherent.

Rules:
1. Use only information grounded in WARM_START_DOCUMENTS.
2. Clusters must be meaningfully distinct from one another.
3. Do not repeat or paraphrase the same question across clusters.
4. Questions must be specific, factual, and answer-oriented.
5. Prefer concrete entities, dates, numbers, titles, places, or explicit relations.
6. Each cluster should contain only related questions; do not mix unrelated themes.
7. Use between 1 and MAX_QUERIES_PER_CLUSTER questions per cluster.
8. {style_rule}
9. Generate information-seeking questions only, not instructions, summaries, or meta-prompts.
10. Do not try to infer any hidden target question; propose plausible future user questions only from the observed evidence.

Output format:
- Return STRICT JSON only.
- Use exactly this schema:
  {{
    "clusters": [
      {cluster_schema}
    ]
  }}
- Return exactly NUM_CLUSTERS clusters.
- No markdown, no prose, no extra keys.

NUM_CLUSTERS:
{num_clusters}

MAX_QUERIES_PER_CLUSTER:
{max_queries_per_cluster}

WARM_START_DOCUMENTS:
{warm_documents}
"""


CLUSTER_GENERATION_DYNAMIC_PROMPT = """You are improving a dynamic cluster bank for downstream answering.

You are given:
- CURRENT_CLUSTERS,
- CURRENT_CLUSTER_MEMORY_BANKS,
- NEW_DOCUMENT_CHUNK.

Generate exactly NUM_CLUSTERS candidate replacement clusters.

Rules:
1. New clusters must be grounded in CURRENT_CLUSTER_MEMORY_BANKS and/or NEW_DOCUMENT_CHUNK.
2. Clusters should be more useful and evidence-targeted than weak CURRENT_CLUSTERS.
3. Do not output near-duplicate clusters or paraphrases of CURRENT_CLUSTERS.
4. Clusters must be meaningfully distinct from one another.
5. Do not repeat or paraphrase the same question across clusters.
6. Questions must be specific, factual, and answer-oriented.
7. Use between 1 and MAX_QUERIES_PER_CLUSTER questions per cluster.
8. {style_rule}
9. Generate information-seeking questions only, not instructions, summaries, or meta-prompts.
10. If current evidence contains authorship/source ambiguity, preserve that distinction instead of collapsing it into generic cluster questions.
11. Do not try to infer any hidden target question; propose plausible future user questions only from the observed evidence.

Output format:
- Return STRICT JSON only.
- Use exactly this schema:
  {{
    "clusters": [
      {cluster_schema}
    ]
  }}
- Return exactly NUM_CLUSTERS clusters.
- No markdown, no prose, no extra keys.

CURRENT_CLUSTERS:
{current_clusters}

CURRENT_CLUSTER_MEMORY_BANKS:
{current_memory_banks}

NEW_DOCUMENT_CHUNK:
{new_document_chunk}

NUM_CLUSTERS:
{num_clusters}

MAX_QUERIES_PER_CLUSTER:
{max_queries_per_cluster}
"""


CLUSTER_REPAIR_PROMPT = """The previous output was invalid or had too few valid clusters.

Return STRICT JSON only in this schema:
{{
  "clusters": [
    {cluster_schema}
  ]
}}

Rules:
1. Return exactly NUM_ADDITIONAL_CLUSTERS additional clusters.
2. Do NOT repeat any question already present in EXISTING_CLUSTERS or CURRENT_CLUSTERS.
3. New clusters must be grounded in CONTEXT_BLOCK.
4. Clusters must be distinct, factual, and useful for downstream answering.
5. {style_rule}
6. Return JSON only. No markdown, no prose, no extra keys.
7. Generate information-seeking questions only, not instructions, summaries, or meta-prompts.

NUM_ADDITIONAL_CLUSTERS:
{num_clusters}

MAX_QUERIES_PER_CLUSTER:
{max_queries_per_cluster}

EXISTING_CLUSTERS:
{existing_clusters}

CURRENT_CLUSTERS:
{current_clusters}

CONTEXT_BLOCK:
{context_block}
"""


INIT_CLUSTER_MEMORY_PROMPT = """You are building a bounded memory bank for one target question cluster from warm-start documents.

Objective:
- Produce a concise memory that will help answer the related questions in TARGET_CLUSTER later.

Rules:
1. Use only WARM_START_DOCUMENTS; do not use outside knowledge.
2. Keep only information relevant or plausibly relevant to TARGET_CLUSTER.
3. Prefer concrete evidence: entities, dates, numbers, titles, locations, explicit relations.
4. Remove low-value details and redundancy.
5. If evidence conflicts, keep conflicting claims as separate attributed entries.
6. Sort retained facts in strict descending importance for TARGET_CLUSTER.
7. Place the most answer-critical facts first.
8. Do not output absence-style statements unless they are themselves target evidence.

Output:
- Plain text memory only (no JSON, no markdown, no preamble).
- Concise, evidence-rich statements.

TARGET_CLUSTER:
{target_cluster}

SOFT_MEMORY_TARGET_TOKENS:
{memory_budget_tokens}

WARM_START_DOCUMENTS:
{warm_documents}
"""


REFRESH_CLUSTER_MEMORY_PROMPT = """You are updating one bounded cluster-specific memory bank as new streamed documents arrive.

Objective:
- Update CURRENT_MEMORY for TARGET_CLUSTER using NEW_DOCUMENT_CHUNK.

Rules:
1. Use only CURRENT_MEMORY and NEW_DOCUMENT_CHUNK; no outside knowledge.
2. Keep information relevant or plausibly relevant to TARGET_CLUSTER.
3. Preserve previously retained critical facts unless NEW_DOCUMENT_CHUNK provides stronger corrective evidence.
4. Prefer concrete evidence: entities, dates, numbers, titles, places, explicit relations.
5. Remove redundancy and low-value details.
6. If conflicts exist, keep separate attributed alternatives instead of collapsing.
7. Sort retained facts in strict descending importance for TARGET_CLUSTER.
8. Put the most answer-critical facts first.
9. Do not output absence-style statements unless they are target evidence.
10. If NEW_DOCUMENT_CHUNK adds no useful stronger evidence, keep CURRENT_MEMORY mostly unchanged.

Output:
- Plain text memory only (no JSON, no markdown, no preamble).
- Concise, evidence-rich statements.

TARGET_CLUSTER:
{target_cluster}

SOFT_MEMORY_TARGET_TOKENS:
{memory_budget_tokens}

CURRENT_MEMORY:
{current_memory}

NEW_DOCUMENT_CHUNK:
{new_document_chunk}
"""


NEW_CLUSTER_FULL_REFRESH_PROMPT = """You are initializing memory for a newly introduced question cluster after a dynamic cluster-bank update.

Task:
- Build memory for TARGET_CLUSTER using:
  1) OLD_CLUSTER_MEMORY_BANKS_SNAPSHOT (all prior banks before update),
  2) NEW_DOCUMENT_CHUNK.

Rules:
1. Use only provided content; no outside knowledge.
2. Keep only information relevant or plausibly relevant to TARGET_CLUSTER.
3. Prefer concrete evidence (entities, dates, numbers, titles, locations, explicit relations).
4. Remove redundancy and low-value details.
5. If conflicts exist, keep attributed alternatives separate.
6. Sort facts by importance for TARGET_CLUSTER.
7. Put the most answer-critical facts first.
8. Do not output absence-style statements unless they are themselves target evidence.
9. If the answer may hinge on who said, wrote, authored, published, discovered, or attributed something, preserve that attribution explicitly.

Output:
- Plain text memory only.
- No JSON, no markdown, no preamble.

TARGET_CLUSTER:
{target_cluster}

SOFT_MEMORY_TARGET_TOKENS:
{memory_budget_tokens}

OLD_CLUSTER_MEMORY_BANKS_SNAPSHOT:
{old_memory_banks}

NEW_DOCUMENT_CHUNK:
{new_document_chunk}
"""


ANSWER_FROM_CLUSTER_BANK_PROMPT = """You are answering TARGET_QUERY using a structured specialist-memory artifact.

Rules:
1. Use only the provided structured memories.
2. Do not use outside knowledge.
3. Provide your best-supported final answer from the provided memories.
4. If multiple candidate facts conflict, prefer the one with strongest direct evidence and specific attribution.
5. Return only one short final answer string.
6. No explanation, no markdown, no bullets, no prefixes.

TARGET_QUERY:
{target_query}

STRUCTURED_MEMORIES:
{memory_banks}

FINAL_ANSWER:
"""


ANSWER_FROM_CLUSTER_BANK_WITH_SUMMARY_PROMPT = """You are answering TARGET_QUERY using:
1. a shared query-agnostic summary memory bank, and
2. structured specialist cluster-memory banks.

Rules:
1. Use only the provided memories.
2. Do not use outside knowledge.
3. Prefer specific cluster-bank evidence over generic summary evidence when they conflict.
4. If multiple candidate facts conflict, prefer the one with strongest direct evidence and specific attribution.
5. Return only one short final answer string.
6. No explanation, no markdown, no bullets, no prefixes.

TARGET_QUERY:
{target_query}

SUMMARY_MEMORY_BANK:
{summary_memory_bank}

STRUCTURED_CLUSTER_MEMORIES:
{memory_banks}

FINAL_ANSWER:
"""


def style_rule_and_schema(style: str) -> Tuple[str, str]:
    if style == "titled":
        return (
            "Each cluster must include a short descriptive title and a list of related questions. "
            "The title must be evidence-preserving, grounded in the provided evidence, and written as a short noun phrase rather than a question. "
            "Do not make the title answer-seeking, benchmark-like, or overly inferential. "
            "Do not write titles such as final-answer claims, hidden-target guesses, or global conclusions that are not directly stated in the evidence.",
            '{"title": "...", "queries": ["...", "..."]}',
        )
    return (
        "Each cluster should be represented only by its list of related questions; do not include titles.",
        '{"queries": ["...", "..."]}',
    )


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


INVALID_CLUSTER_TITLE_PATTERNS = (
    re.compile(r"\?$"),
    re.compile(r"^(what|which|who|when|where|why|how)\b", re.IGNORECASE),
    re.compile(r"\b(final answer|correct answer|the answer)\b", re.IGNORECASE),
    re.compile(r"\bthis is the\b", re.IGNORECASE),
    re.compile(r"\bthe (?:race|event|paper|team|song|book) where\b", re.IGNORECASE),
    re.compile(r"\bis the (?:race|event|paper|team|song|book) where\b", re.IGNORECASE),
    re.compile(r"\bby december 20\d{2}\b", re.IGNORECASE),
    re.compile(r"\ball (?:the )?(?:top\s+\d+\s+)?finishers\b", re.IGNORECASE),
)


def title_is_evidence_preserving(title: str) -> bool:
    normalized = normalize_text(title)
    if not normalized:
        return False
    for pattern in INVALID_CLUSTER_TITLE_PATTERNS:
        if pattern.search(normalized):
            return False
    return True


def fallback_title_from_queries(queries: Sequence[str]) -> str:
    if not queries:
        return "Evidence cluster"
    text = normalize_text(str(queries[0]).rstrip("?. "))
    cleanup_patterns = (
        r"^(?:in\s+which\s+year|in\s+what\s+year)\s+did\s+",
        r"^(?:what|which|who|when|where|why|how)\s+(?:is|are|was|were|did|does|do|can|could|would|will|has|have|had|may|might|should)\s+",
        r"^(?:what|which|who|when|where|why|how)\s+",
    )
    for pattern in cleanup_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        text = normalize_text(text)
    if not text:
        return "Evidence cluster"
    if len(text) > 80:
        text = text[:80].rsplit(" ", 1)[0] or text[:80]
    return text[:1].upper() + text[1:]


def text_similarity(a: str, b: str) -> float:
    an = normalize_text(a).lower()
    bn = normalize_text(b).lower()
    if not an or not bn:
        return 0.0
    if an == bn:
        return 1.0
    ratio = difflib.SequenceMatcher(None, an, bn).ratio()
    at = {tok for tok in re.findall(r"[a-z0-9]+", an) if len(tok) > 2}
    bt = {tok for tok in re.findall(r"[a-z0-9]+", bn) if len(tok) > 2}
    if not at or not bt:
        return float(ratio)
    inter = at & bt
    union = at | bt
    jaccard = len(inter) / len(union) if union else 0.0
    containment = len(inter) / min(len(at), len(bt)) if min(len(at), len(bt)) > 0 else 0.0
    return float(max(ratio, jaccard, containment))


def normalize_cluster(raw: Any, style: str, max_queries_per_cluster: int) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    queries_raw = raw.get("queries") or raw.get("questions") or []
    if not isinstance(queries_raw, list):
        return None
    queries: List[str] = []
    seen: Set[str] = set()
    for item in queries_raw:
        q = base.normalize_candidate_query(str(item))
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(q)
        if len(queries) >= max_queries_per_cluster:
            break
    if not queries:
        return None
    title = normalize_text(str(raw.get("title", "")))
    if style == "titled":
        if not title_is_evidence_preserving(title):
            title = fallback_title_from_queries(queries)
        if not title_is_evidence_preserving(title):
            title = "Evidence cluster"
    if style == "list_only":
        title = ""
    return {"title": title, "queries": queries}


def cluster_key(cluster: Dict[str, Any], style: str) -> str:
    payload = {
        "title": normalize_text(str(cluster.get("title", ""))) if style == "titled" else "",
        "queries": [base.normalize_candidate_query(q) for q in list(cluster.get("queries") or [])],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def cluster_similarity_stats(a: Dict[str, Any], b: Dict[str, Any], style: str) -> Dict[str, float]:
    a_queries = [base.normalize_candidate_query(q) for q in list(a.get("queries") or []) if base.normalize_candidate_query(q)]
    b_queries = [base.normalize_candidate_query(q) for q in list(b.get("queries") or []) if base.normalize_candidate_query(q)]
    if not a_queries or not b_queries:
        return {"max_query_similarity": 0.0, "avg_query_coverage": 0.0, "title_similarity": 0.0}
    a_best = [max(text_similarity(qa, qb) for qb in b_queries) for qa in a_queries]
    b_best = [max(text_similarity(qb, qa) for qa in a_queries) for qb in b_queries]
    title_similarity = 0.0
    if style == "titled":
        title_similarity = text_similarity(str(a.get("title", "")), str(b.get("title", "")))
    return {
        "max_query_similarity": float(max(max(a_best), max(b_best))),
        "avg_query_coverage": float(min(sum(a_best) / len(a_best), sum(b_best) / len(b_best))),
        "title_similarity": float(title_similarity),
    }


def clusters_too_similar(a: Dict[str, Any], b: Dict[str, Any], style: str) -> bool:
    a_query_keys = {base.normalize_candidate_query(q).lower() for q in list(a.get("queries") or []) if base.normalize_candidate_query(q)}
    b_query_keys = {base.normalize_candidate_query(q).lower() for q in list(b.get("queries") or []) if base.normalize_candidate_query(q)}
    if a_query_keys & b_query_keys:
        return True
    stats = cluster_similarity_stats(a, b, style)
    if stats["max_query_similarity"] >= 0.94:
        return True
    if stats["avg_query_coverage"] >= 0.80:
        return True
    if style == "titled" and stats["title_similarity"] >= 0.92 and stats["max_query_similarity"] >= 0.70:
        return True
    return False


def select_diverse_clusters(
    scored_clusters: Sequence[Dict[str, Any]],
    num_clusters: int,
    style: str,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    deferred: List[Dict[str, Any]] = []
    seen_keys: Set[str] = set()
    for item in scored_clusters:
        cluster = item.get("cluster") or {}
        key = cluster_key(cluster, style)
        if key in seen_keys:
            continue
        if any(clusters_too_similar(cluster, kept.get("cluster") or {}, style) for kept in selected):
            deferred.append(item)
            continue
        seen_keys.add(key)
        selected.append(item)
        if len(selected) >= num_clusters:
            return selected
    for item in deferred:
        cluster = item.get("cluster") or {}
        key = cluster_key(cluster, style)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append(item)
        if len(selected) >= num_clusters:
            break
    return selected


def cluster_label(cluster: Dict[str, Any], style: str) -> str:
    lines: List[str] = []
    if style == "titled":
        lines.append(f"title: {normalize_text(str(cluster.get('title', '')))}")
    lines.append("questions:")
    for i, q in enumerate(list(cluster.get("queries") or []), start=1):
        lines.append(f"{i}. {q}")
    return "\n".join(lines).strip()


def cluster_blob(selected_keys: Sequence[str], cluster_bank: Dict[str, Dict[str, Any]], memory_bank: Dict[str, str], style: str) -> str:
    blocks: List[str] = []
    for i, key in enumerate(selected_keys, start=1):
        cluster = cluster_bank.get(key, {"queries": []})
        blocks.append(
            "\n".join(
                [
                    f"[MEMORY_BANK_{i}]",
                    "cluster:",
                    cluster_label(cluster, style),
                    "memory:",
                    memory_bank.get(key, ""),
                ]
            )
        )
    return "\n\n".join(blocks).strip()


def parse_cluster_candidates(raw: str, style: str, max_queries_per_cluster: int) -> List[Dict[str, Any]]:
    txt = (raw or "").strip()
    if not txt:
        return []
    if txt.startswith("```"):
        txt = txt.strip()
        txt = txt.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    obj: Any = None
    parsed = False
    try:
        obj = json.loads(txt)
        parsed = True
    except Exception:
        parsed = False
    if not parsed:
        start = txt.find("{")
        end = txt.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(txt[start : end + 1])
                parsed = True
            except Exception:
                parsed = False
    clusters_raw: List[Any] = []
    if parsed:
        if isinstance(obj, dict):
            clusters_raw = obj.get("clusters") or []
        elif isinstance(obj, list):
            clusters_raw = obj
    out: List[Dict[str, Any]] = []
    seen_keys: Set[str] = set()
    used_query_keys: Set[str] = set()
    for item in clusters_raw:
        cluster = normalize_cluster(item, style=style, max_queries_per_cluster=max_queries_per_cluster)
        if not cluster:
            continue
        cluster_queries: List[str] = []
        for q in cluster["queries"]:
            qkey = q.lower()
            if qkey in used_query_keys:
                continue
            if any(text_similarity(q, kept) >= 0.94 for kept in cluster_queries):
                continue
            cluster_queries.append(q)
        cluster["queries"] = cluster_queries
        if not cluster["queries"]:
            continue
        key = cluster_key(cluster, style)
        if key in seen_keys:
            continue
        if any(clusters_too_similar(cluster, kept, style) for kept in out):
            continue
        seen_keys.add(key)
        for q in cluster["queries"]:
            used_query_keys.add(q.lower())
        out.append(cluster)
    return out


def build_fallback_clusters(docs: Sequence[Dict[str, Any]], needed: int, style: str, max_queries_per_cluster: int) -> List[Dict[str, Any]]:
    fallback_queries = base.build_fallback_queries(docs, max(needed * max(2, max_queries_per_cluster), needed))
    out: List[Dict[str, Any]] = []
    i = 0
    while len(out) < needed and i < len(fallback_queries):
        chunk = fallback_queries[i : i + min(2, max_queries_per_cluster)]
        i += min(2, max_queries_per_cluster)
        if not chunk:
            break
        cluster = {"title": chunk[0][:80] if style == "titled" else "", "queries": chunk}
        out.append(cluster)
    while len(out) < needed:
        idx = len(out) + 1
        out.append(
            {
                "title": f"Fallback cluster {idx}" if style == "titled" else "",
                "queries": [f"What key factual relationship is stated in the streamed documents ({idx})?"],
            }
        )
    return out[:needed]


def generate_candidate_clusters_warm(
    llm: Any,
    warm_docs: Sequence[Dict[str, Any]],
    num_clusters: int,
    max_queries_per_cluster: int,
    style: str,
    temperature: float,
) -> List[Dict[str, Any]]:
    style_rule, schema = style_rule_and_schema(style)
    warm_docs_block = base.format_doc_chunk_for_prompt(warm_docs)
    prompt = CLUSTER_GENERATION_WARM_PROMPT.format(
        num_clusters=num_clusters,
        max_queries_per_cluster=max_queries_per_cluster,
        warm_documents=warm_docs_block if warm_docs_block else "(empty)",
        style_rule=style_rule,
        cluster_schema=schema,
    )
    raw = llm.generate(prompt, temperature=temperature)
    out = parse_cluster_candidates(raw, style=style, max_queries_per_cluster=max_queries_per_cluster)
    if len(out) >= num_clusters:
        return out[:num_clusters]
    needed = num_clusters - len(out)
    repair_prompt = CLUSTER_REPAIR_PROMPT.format(
        num_clusters=needed,
        max_queries_per_cluster=max_queries_per_cluster,
        existing_clusters=json.dumps(out, ensure_ascii=False),
        current_clusters="[]",
        context_block=warm_docs_block if warm_docs_block else "(empty)",
        style_rule=style_rule,
        cluster_schema=schema,
    )
    repair_raw = llm.generate(repair_prompt, temperature=temperature)
    repaired = parse_cluster_candidates(repair_raw, style=style, max_queries_per_cluster=max_queries_per_cluster)
    combined = out + repaired
    uniq: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    used_query_keys: Set[str] = set()
    for cluster in combined:
        c = {"title": cluster.get("title", ""), "queries": []}
        for q in cluster.get("queries", []):
            qkey = q.lower()
            if qkey in used_query_keys:
                continue
            used_query_keys.add(qkey)
            c["queries"].append(q)
        if not c["queries"]:
            continue
        key = cluster_key(c, style)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
        if len(uniq) >= num_clusters:
            break
    if len(uniq) < num_clusters:
        for cluster in build_fallback_clusters(warm_docs, num_clusters, style, max_queries_per_cluster):
            key = cluster_key(cluster, style)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(cluster)
            if len(uniq) >= num_clusters:
                break
    return uniq[:num_clusters]


def generate_candidate_clusters_dynamic(
    llm: Any,
    current_clusters: Sequence[Dict[str, Any]],
    current_memory_bank: Dict[str, str],
    docs_chunk: Sequence[Dict[str, Any]],
    num_clusters: int,
    max_queries_per_cluster: int,
    style: str,
    temperature: float,
) -> List[Dict[str, Any]]:
    style_rule, schema = style_rule_and_schema(style)
    current_clusters_json = json.dumps(current_clusters, ensure_ascii=False)
    current_keys = [cluster_key(c, style) for c in current_clusters]
    current_cluster_bank = {cluster_key(c, style): c for c in current_clusters}
    current_memory_block = cluster_blob(current_keys, current_cluster_bank, current_memory_bank, style)
    chunk_block = base.format_doc_chunk_for_prompt(docs_chunk)
    prompt = CLUSTER_GENERATION_DYNAMIC_PROMPT.format(
        current_clusters=current_clusters_json,
        current_memory_banks=current_memory_block if current_memory_block else "(empty)",
        new_document_chunk=chunk_block if chunk_block else "(empty)",
        num_clusters=num_clusters,
        max_queries_per_cluster=max_queries_per_cluster,
        style_rule=style_rule,
        cluster_schema=schema,
    )
    raw = llm.generate(prompt, temperature=temperature)
    out = parse_cluster_candidates(raw, style=style, max_queries_per_cluster=max_queries_per_cluster)
    if len(out) >= num_clusters:
        return out[:num_clusters]
    needed = num_clusters - len(out)
    repair_prompt = CLUSTER_REPAIR_PROMPT.format(
        num_clusters=needed,
        max_queries_per_cluster=max_queries_per_cluster,
        existing_clusters=json.dumps(out, ensure_ascii=False),
        current_clusters=current_clusters_json,
        context_block="\n\n".join(
            [
                f"CURRENT_CLUSTER_MEMORY_BANKS:\n{current_memory_block if current_memory_block else '(empty)'}",
                f"NEW_DOCUMENT_CHUNK:\n{chunk_block if chunk_block else '(empty)'}",
            ]
        ),
        style_rule=style_rule,
        cluster_schema=schema,
    )
    repair_raw = llm.generate(repair_prompt, temperature=temperature)
    repaired = parse_cluster_candidates(repair_raw, style=style, max_queries_per_cluster=max_queries_per_cluster)
    current_query_keys = {q.lower() for c in current_clusters for q in c.get("queries", [])}
    uniq: List[Dict[str, Any]] = []
    seen_clusters: Set[str] = set()
    used_query_keys: Set[str] = set(current_query_keys)
    for cluster in out + repaired:
        c = {"title": cluster.get("title", ""), "queries": []}
        for q in cluster.get("queries", []):
            qkey = q.lower()
            if qkey in used_query_keys:
                continue
            c["queries"].append(q)
        if not c["queries"]:
            continue
        key = cluster_key(c, style)
        if key in seen_clusters:
            continue
        seen_clusters.add(key)
        for q in c["queries"]:
            used_query_keys.add(q.lower())
        uniq.append(c)
        if len(uniq) >= num_clusters:
            break
    if len(uniq) < num_clusters:
        for cluster in build_fallback_clusters(docs_chunk, num_clusters, style, max_queries_per_cluster):
            key = cluster_key(cluster, style)
            if key in seen_clusters:
                continue
            seen_clusters.add(key)
            uniq.append(cluster)
            if len(uniq) >= num_clusters:
                break
    return uniq[:num_clusters]


def score_clusters_by_best_query(
    embedder: base.QwenEmbedder,
    target_query: str,
    clusters: Sequence[Dict[str, Any]],
    style: str,
) -> List[Dict[str, Any]]:
    query_to_cluster_keys: Dict[str, List[str]] = {}
    cluster_by_key: Dict[str, Dict[str, Any]] = {}
    unique_queries: List[str] = []
    seen_queries: Set[str] = set()
    for cluster in clusters:
        key = cluster_key(cluster, style)
        cluster_by_key[key] = cluster
        for q in cluster.get("queries", []):
            qn = base.normalize_candidate_query(q)
            if not qn:
                continue
            qkey = qn.lower()
            query_to_cluster_keys.setdefault(qkey, []).append(key)
            if qkey in seen_queries:
                continue
            seen_queries.add(qkey)
            unique_queries.append(qn)
    if not unique_queries:
        return []
    vecs = embedder.embed([target_query] + unique_queries)
    qv = vecs[0]
    query_scores: Dict[str, float] = {}
    for idx, q in enumerate(unique_queries):
        query_scores[q.lower()] = base.cosine_sim(qv, vecs[idx + 1])
    scored: List[Dict[str, Any]] = []
    seen_cluster_keys: Set[str] = set()
    for cluster in clusters:
        key = cluster_key(cluster, style)
        if key in seen_cluster_keys:
            continue
        seen_cluster_keys.add(key)
        best_query = ""
        best_score = -1.0
        for q in cluster.get("queries", []):
            score = float(query_scores.get(q.lower(), 0.0))
            if score > best_score:
                best_score = score
                best_query = q
        scored.append(
            {
                "cluster": cluster_by_key[key],
                "score": float(max(best_score, 0.0)),
                "best_query": best_query,
            }
        )
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def scored_clusters_from_score_map(
    clusters: Sequence[Dict[str, Any]],
    style: str,
    score_map: Dict[str, float],
    objective_field_name: str = "",
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    seen_cluster_keys: Set[str] = set()
    for cluster in clusters:
        key = cluster_key(cluster, style)
        if key in seen_cluster_keys:
            continue
        seen_cluster_keys.add(key)
        item = {
            "cluster": cluster,
            "score": float(score_map.get(key, 0.0)),
            "best_query": str((cluster.get("queries") or [""])[0] or ""),
        }
        if objective_field_name:
            item[objective_field_name] = float(score_map.get(key, 0.0))
        scored.append(item)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def score_clusters_by_memory(
    embedder: base.QwenEmbedder,
    target_query: str,
    selected_keys: Sequence[str],
    cluster_bank: Dict[str, Dict[str, Any]],
    memory_bank: Dict[str, str],
    style: str,
) -> Dict[str, float]:
    return score_clusters_by_memory_text(
        embedder=embedder,
        target_text=target_query,
        selected_keys=selected_keys,
        memory_bank=memory_bank,
    )


def score_clusters_by_memory_text(
    embedder: Optional[base.QwenEmbedder],
    target_text: str,
    selected_keys: Sequence[str],
    memory_bank: Dict[str, str],
) -> Dict[str, float]:
    target_text = str(target_text or "").strip()
    if embedder is None or not target_text:
        return {}
    keys_to_embed: List[str] = []
    memory_texts: List[str] = []
    for key in selected_keys:
        memory = str(memory_bank.get(key, "") or "").strip()
        if not memory:
            continue
        keys_to_embed.append(key)
        memory_texts.append(memory)
    if not memory_texts:
        return {}
    vecs = embedder.embed([target_text] + memory_texts)
    qv = vecs[0]
    scores: Dict[str, float] = {}
    for idx, key in enumerate(keys_to_embed):
        scores[key] = float(base.cosine_sim(qv, vecs[idx + 1]))
    for key in selected_keys:
        scores.setdefault(key, 0.0)
    return scores


def attach_memory_scores(
    scored_clusters: Sequence[Dict[str, Any]],
    memory_scores: Dict[str, float],
    style: str,
) -> List[Dict[str, Any]]:
    return attach_named_scores(
        scored_clusters=scored_clusters,
        style=style,
        memory_score=memory_scores,
    )


def attach_named_scores(
    scored_clusters: Sequence[Dict[str, Any]],
    style: str,
    **named_scores: Dict[str, float],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in scored_clusters:
        cluster = item.get("cluster") or {}
        key = cluster_key(cluster, style)
        enriched = dict(item)
        for field_name, score_map in named_scores.items():
            if key in score_map:
                enriched[field_name] = float(score_map[key])
        out.append(enriched)
    return out


def serialize_cluster_scores(scored: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in scored:
        row = {
            "cluster": item.get("cluster", {}),
            "score": float(item.get("score", 0.0) or 0.0),
            "best_query": str(item.get("best_query", "") or ""),
        }
        for key, value in item.items():
            if key == "memory_score" or key.startswith("memory_score_"):
                row[key] = None if value is None else float(value)
        out.append(row)
    return out


def score_field_top1_and_mean(
    scored_clusters: Sequence[Dict[str, Any]],
    field_name: str,
) -> Tuple[Optional[float], Optional[float]]:
    top1: Optional[float] = None
    values: List[float] = []
    for idx, item in enumerate(scored_clusters):
        value = item.get(field_name)
        if value is None:
            continue
        value_f = float(value)
        values.append(value_f)
        if idx == 0:
            top1 = value_f
    mean = float(sum(values) / len(values)) if values else None
    return top1, mean


def mean_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def normalized_cluster_queries(cluster: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for q in list(cluster.get("queries") or []):
        qn = base.normalize_candidate_query(q)
        if not qn:
            continue
        qkey = qn.lower()
        if qkey in seen:
            continue
        seen.add(qkey)
        out.append(qn)
    return out


def normalized_centroid(vectors: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    if not vectors:
        return None
    mat = np.asarray(vectors, dtype=np.float32)
    centroid = np.mean(mat, axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm == 0.0:
        return None
    return (centroid / norm).astype(np.float32)


def build_cluster_state_snapshot(
    selected_keys: Sequence[str],
    cluster_bank: Dict[str, Dict[str, Any]],
    memory_bank: Dict[str, str],
    scored_clusters: Sequence[Dict[str, Any]],
    style: str,
    counter: base.TokenCounter,
    embedder: Optional[base.QwenEmbedder],
) -> Dict[str, Any]:
    score_map: Dict[str, Dict[str, Any]] = {}
    for item in scored_clusters:
        cluster = item.get("cluster") or {}
        if not isinstance(cluster, dict):
            continue
        score_map[cluster_key(cluster, style)] = item

    cluster_rows: List[Dict[str, Any]] = []
    all_queries: List[str] = []
    seen_queries: Set[str] = set()
    for key in selected_keys:
        cluster = cluster_bank.get(key, {"queries": []})
        queries = normalized_cluster_queries(cluster)
        for q in queries:
            qkey = q.lower()
            if qkey in seen_queries:
                continue
            seen_queries.add(qkey)
            all_queries.append(q)
        score_item = score_map.get(key, {})
        representative_query = str(score_item.get("best_query", "") or "")
        if not representative_query and queries:
            representative_query = queries[0]
        memory_text = str(memory_bank.get(key, "") or "")
        cluster_rows.append(
            {
                "cluster_key": key,
                "cluster": cluster,
                "cluster_label": cluster_label(cluster, style),
                "cluster_size": len(queries),
                "representative_query": representative_query,
                "member_queries": queries,
                "is_empty": len(queries) == 0,
                "is_degenerate": len(queries) <= 1,
                "memory_tokens": counter.count(memory_text),
                "mean_intra_cluster_similarity": None,
            }
        )

    mean_inter_cluster_similarity: Optional[float] = None
    if embedder is not None and all_queries:
        vecs = embedder.embed(all_queries)
        query_vec_by_text = {q.lower(): vecs[idx] for idx, q in enumerate(all_queries)}
        centroids: Dict[str, np.ndarray] = {}
        for row in cluster_rows:
            member_vecs = [query_vec_by_text[q.lower()] for q in row["member_queries"] if q.lower() in query_vec_by_text]
            centroid = normalized_centroid(member_vecs)
            if centroid is None:
                row["mean_intra_cluster_similarity"] = None
                continue
            centroids[row["cluster_key"]] = centroid
            row["mean_intra_cluster_similarity"] = float(
                sum(base.cosine_sim(v, centroid) for v in member_vecs) / len(member_vecs)
            ) if member_vecs else None
        inter_vals: List[float] = []
        centroid_keys = [row["cluster_key"] for row in cluster_rows if row["cluster_key"] in centroids]
        for i in range(len(centroid_keys)):
            for j in range(i + 1, len(centroid_keys)):
                inter_vals.append(float(base.cosine_sim(centroids[centroid_keys[i]], centroids[centroid_keys[j]])))
        mean_inter_cluster_similarity = mean_or_none(inter_vals)

    total_memory_tokens = int(sum(int(row["memory_tokens"]) for row in cluster_rows))
    cluster_sizes = [int(row["cluster_size"]) for row in cluster_rows]
    representative_queries = [str(row["representative_query"]) for row in cluster_rows]
    member_queries = [list(row["member_queries"]) for row in cluster_rows]
    memory_tokens_by_cluster = {str(row["cluster_key"]): int(row["memory_tokens"]) for row in cluster_rows}
    mean_memory_tokens = (float(total_memory_tokens) / len(cluster_rows)) if cluster_rows else None

    return {
        "num_selected_clusters": len(cluster_rows),
        "cluster_sizes": cluster_sizes,
        "mean_cluster_size": mean_or_none([float(x) for x in cluster_sizes]),
        "min_cluster_size": min(cluster_sizes) if cluster_sizes else None,
        "max_cluster_size": max(cluster_sizes) if cluster_sizes else None,
        "mean_intra_cluster_similarity": mean_or_none(
            [row.get("mean_intra_cluster_similarity") for row in cluster_rows]
        ),
        "mean_inter_cluster_similarity": mean_inter_cluster_similarity,
        "empty_cluster_count": sum(1 for row in cluster_rows if bool(row["is_empty"])),
        "degenerate_cluster_count": sum(1 for row in cluster_rows if bool(row["is_degenerate"])),
        "representative_queries": representative_queries,
        "cluster_member_queries": member_queries,
        "memory_tokens_total": total_memory_tokens,
        "mean_memory_tokens_per_bank": mean_memory_tokens,
        "memory_tokens_by_cluster": memory_tokens_by_cluster,
        "clusters": cluster_rows,
    }


def build_oracle_gold_memory_reference(
    row: Dict[str, Any],
    counter: base.TokenCounter,
    budget_tokens: int,
    truncate_strategy: str,
) -> Dict[str, Any]:
    candidate_fields = (
        "oracle_gold_memory_text",
        "oracle_gold_memory",
        "oracle_memory_text",
        "oracle_memory",
        "oracle_rewrite",
    )
    for field_name in candidate_fields:
        field_value = str(row.get(field_name, "") or "").strip()
        if not field_value:
            continue
        raw_tokens = counter.count(field_value)
        used_text = field_value
        was_truncated = False
        if budget_tokens > 0:
            used_text, was_truncated = base.truncate_to_budget(
                field_value,
                counter=counter,
                budget_tokens=budget_tokens,
                truncate_strategy=truncate_strategy,
            )
        return {
            "text": used_text,
            "source": field_name,
            "raw_tokens": raw_tokens,
            "used_tokens": counter.count(used_text),
            "was_truncated": bool(was_truncated),
            "gold_doc_count": 0,
        }

    docs = list(row.get("docs") or [])
    gold_docs = [doc for doc in docs if isinstance(doc, dict) and bool(doc.get("is_gold", False))]
    raw_text = "\n\n".join(
        str(doc.get("text", "") or "").strip()
        for doc in gold_docs
        if str(doc.get("text", "") or "").strip()
    ).strip()
    raw_tokens = counter.count(raw_text)
    used_text = raw_text
    was_truncated = False
    if raw_text and budget_tokens > 0:
        used_text, was_truncated = base.truncate_to_budget(
            raw_text,
            counter=counter,
            budget_tokens=budget_tokens,
            truncate_strategy=truncate_strategy,
        )
    return {
        "text": used_text,
        "source": "gold_doc_concat" if raw_text else "missing",
        "raw_tokens": raw_tokens,
        "used_tokens": counter.count(used_text),
        "was_truncated": bool(was_truncated),
        "gold_doc_count": len(gold_docs),
    }


def build_initial_cluster_memory(llm: Any, cluster: Dict[str, Any], warm_docs: Sequence[Dict[str, Any]], budget_tokens: int, temperature: float, style: str) -> str:
    prompt = INIT_CLUSTER_MEMORY_PROMPT.format(
        target_cluster=cluster_label(cluster, style),
        memory_budget_tokens=budget_tokens,
        warm_documents=base.format_doc_chunk_for_prompt(warm_docs) if warm_docs else "(empty)",
    )
    return llm.generate(prompt, temperature=temperature).strip()


def refresh_cluster_memory(llm: Any, cluster: Dict[str, Any], current_memory: str, docs_chunk: Sequence[Dict[str, Any]], budget_tokens: int, temperature: float, style: str) -> str:
    prompt = REFRESH_CLUSTER_MEMORY_PROMPT.format(
        target_cluster=cluster_label(cluster, style),
        memory_budget_tokens=budget_tokens,
        current_memory=current_memory if current_memory else "(empty)",
        new_document_chunk=base.format_doc_chunk_for_prompt(docs_chunk) if docs_chunk else "(empty)",
    )
    out = llm.generate(prompt, temperature=temperature).strip()
    return out if out else current_memory


def full_refresh_new_cluster_memory(llm: Any, cluster: Dict[str, Any], old_keys: Sequence[str], old_cluster_bank: Dict[str, Dict[str, Any]], old_memory_bank: Dict[str, str], docs_chunk: Sequence[Dict[str, Any]], budget_tokens: int, temperature: float, style: str) -> str:
    prompt = NEW_CLUSTER_FULL_REFRESH_PROMPT.format(
        target_cluster=cluster_label(cluster, style),
        memory_budget_tokens=budget_tokens,
        old_memory_banks=cluster_blob(old_keys, old_cluster_bank, old_memory_bank, style) or "(empty)",
        new_document_chunk=base.format_doc_chunk_for_prompt(docs_chunk) if docs_chunk else "(empty)",
    )
    return llm.generate(prompt, temperature=temperature).strip()


def build_cluster_memory_blob(selected_keys: Sequence[str], cluster_bank: Dict[str, Dict[str, Any]], memory_bank: Dict[str, str], style: str) -> str:
    return cluster_blob(selected_keys, cluster_bank, memory_bank, style)


def build_flat_cluster_memory_blob(selected_keys: Sequence[str], memory_bank: Dict[str, str]) -> str:
    blocks: List[str] = []
    for key in selected_keys:
        memory = str(memory_bank.get(key, "") or "").strip()
        if memory:
            blocks.append(memory)
    return "\n\n".join(blocks).strip()


def selection_metric_target_field_name(selection_metric: str) -> str:
    if selection_metric == "memory_vs_gold_answer":
        return "memory_score_gold_answer"
    if selection_metric == "memory_vs_oracle_gold_memory":
        return "memory_score_oracle_gold_memory"
    return ""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Oracle-assisted warm-start + dynamic cluster bank streaming memory pipeline.")
    ap.add_argument("--dataset_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--trace_jsonl", default="")
    ap.add_argument("--llm_backend", choices=["gemini", "openrouter"], default="openrouter")
    ap.add_argument("--model", default="qwen/qwen3.5-397b-a17b")
    ap.add_argument("--query_gen_backend", choices=["gemini", "hf", "openrouter"], default="openrouter")
    ap.add_argument("--query_gen_model", default="")
    ap.add_argument("--query_gen_hf_max_new_tokens", type=int, default=512)
    ap.add_argument("--query_gen_hf_device_map", default="auto")
    ap.add_argument("--openrouter_base_url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--openrouter_http_referer", default="")
    ap.add_argument("--openrouter_app_title", default="")
    ap.add_argument("--answer_model", default="")
    ap.add_argument("--embed_model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--embed_device", default="cpu")
    ap.add_argument("--embed_batch_size", type=int, default=16)
    ap.add_argument("--memory_budget_tokens", type=int, default=500)
    ap.add_argument("--z_warm_docs", type=int, default=1)
    ap.add_argument("--num_bank_queries", type=int, default=2)
    ap.add_argument("--answer_top_j", type=int, default=1)
    ap.add_argument("--candidate_multiplier", type=int, default=4)
    ap.add_argument("--refresh_stride_docs", type=int, default=2)
    ap.add_argument("--max_queries_per_cluster", type=int, default=5)
    ap.add_argument("--selection_metric", choices=["query", "memory_vs_gold_answer", "memory_vs_oracle_gold_memory"], default="query")
    ap.add_argument("--cluster_style", choices=["list_only", "titled"], default="list_only")
    ap.add_argument("--answer_render_style", choices=["banked", "flat"], default="banked")
    ap.add_argument("--answer_use_all_banks", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--summary_budget_tokens", type=int, default=0)
    ap.add_argument("--proxy_oracle_memory_budget_tokens", type=int, default=0)
    ap.add_argument("--overflow_policy", choices=["truncate_only"], default="truncate_only")
    ap.add_argument("--truncate_strategy", choices=["head", "middle", "tail"], default="head")
    ap.add_argument("--doc_truncate_strategy", choices=["head", "middle", "tail"], default="head")
    ap.add_argument("--max_doc_tokens", type=int, default=12000)
    ap.add_argument("--max_docs_per_query", type=int, default=0)
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress_every", type=int, default=10)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--timeout_sec", type=int, default=300)
    ap.add_argument("--query_gen_temperature", type=float, default=0.2)
    ap.add_argument("--update_temperature", type=float, default=0.0)
    ap.add_argument("--answer_temperature", type=float, default=0.0)
    ap.add_argument("--log_selection_details", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_false", dest="resume")
    ap.add_argument("--skip_answer", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.memory_budget_tokens <= 0:
        raise ValueError("--memory_budget_tokens must be > 0.")
    if args.z_warm_docs < 0:
        raise ValueError("--z_warm_docs must be >= 0.")
    if args.num_bank_queries <= 0:
        raise ValueError("--num_bank_queries must be > 0.")
    if args.answer_top_j <= 0 or args.answer_top_j > args.num_bank_queries:
        raise ValueError("--answer_top_j must be > 0 and <= --num_bank_queries.")
    if args.candidate_multiplier < 1:
        raise ValueError("--candidate_multiplier must be >= 1.")
    if args.refresh_stride_docs <= 0:
        raise ValueError("--refresh_stride_docs must be > 0.")
    if args.max_doc_tokens < 0 or args.max_docs_per_query < 0:
        raise ValueError("--max_doc_tokens and --max_docs_per_query must be >= 0.")
    if args.retries < 0 or args.timeout_sec <= 0:
        raise ValueError("--retries must be >= 0 and --timeout_sec must be > 0.")
    if args.progress_every < 0 or args.start_index < 0 or args.limit < 0:
        raise ValueError("progress/start/limit values must be >= 0.")
    if args.embed_batch_size <= 0 or args.query_gen_hf_max_new_tokens <= 0:
        raise ValueError("embed/query_gen HF sizes must be > 0.")
    if args.max_queries_per_cluster <= 0:
        raise ValueError("--max_queries_per_cluster must be > 0.")
    if args.summary_budget_tokens < 0:
        raise ValueError("--summary_budget_tokens must be >= 0.")
    if args.proxy_oracle_memory_budget_tokens < 0:
        raise ValueError("--proxy_oracle_memory_budget_tokens must be >= 0.")
    if args.selection_metric not in {"query", "memory_vs_gold_answer", "memory_vs_oracle_gold_memory"}:
        raise ValueError("--selection_metric is invalid.")


def is_completed_row(row: Dict[str, Any], skip_answer: bool) -> bool:
    if str(row.get("runtime_error", "")).strip():
        return False
    if bool(row.get("row_failed", False)):
        return False
    if row.get("update_errors"):
        return False
    if str(row.get("answer_error", "")).strip():
        return False
    if not skip_answer and not str(row.get("model_answer", "")).strip():
        return False
    return True


def load_done_ids(path: Path, skip_answer: bool) -> Set[str]:
    if not path.exists():
        return set()
    done: Set[str] = set()
    for row in base.iter_jsonl(path):
        qid = str(row.get("question_id", "")).strip()
        if qid and is_completed_row(row, skip_answer=skip_answer):
            done.add(qid)
    return done


def aggregate_output_totals(path: Path, skip_answer: bool) -> Dict[str, Any]:
    stats = {
        "rows_written_total": 0,
        "rows_completed_total": 0,
        "rows_runtime_error_total": 0,
        "query_gen_calls": 0,
        "query_gen_input_tokens": 0,
        "query_gen_output_tokens": 0,
        "update_calls": 0,
        "update_input_tokens": 0,
        "update_output_tokens": 0,
        "answer_calls": 0,
        "answer_input_tokens": 0,
        "answer_output_tokens": 0,
        "query_gen_wall_time_sec": 0.0,
        "update_wall_time_sec": 0.0,
        "answer_wall_time_sec": 0.0,
        "total_lm_wall_time_sec": 0.0,
    }
    if not path.exists():
        return stats
    for row in base.iter_jsonl(path):
        stats["rows_written_total"] += 1
        if is_completed_row(row, skip_answer=skip_answer):
            stats["rows_completed_total"] += 1
        if str(row.get("runtime_error", "")).strip():
            stats["rows_runtime_error_total"] += 1
        qg = row.get("query_gen_lm_usage") if isinstance(row.get("query_gen_lm_usage"), dict) else {}
        u = row.get("update_lm_usage") if isinstance(row.get("update_lm_usage"), dict) else {}
        a = row.get("answer_lm_usage") if isinstance(row.get("answer_lm_usage"), dict) else {}
        stats["query_gen_calls"] += base._as_int(qg.get("calls"))
        stats["query_gen_input_tokens"] += base._as_int(qg.get("input_tokens"))
        stats["query_gen_output_tokens"] += base._as_int(qg.get("output_tokens"))
        stats["update_calls"] += base._as_int(u.get("calls"))
        stats["update_input_tokens"] += base._as_int(u.get("input_tokens"))
        stats["update_output_tokens"] += base._as_int(u.get("output_tokens"))
        stats["answer_calls"] += base._as_int(a.get("calls"))
        stats["answer_input_tokens"] += base._as_int(a.get("input_tokens"))
        stats["answer_output_tokens"] += base._as_int(a.get("output_tokens"))
        stats["query_gen_wall_time_sec"] += base._as_float(qg.get("wall_time_sec"))
        stats["update_wall_time_sec"] += base._as_float(u.get("wall_time_sec"))
        stats["answer_wall_time_sec"] += base._as_float(a.get("wall_time_sec"))
        total_row_wall = row.get("total_lm_wall_time_sec")
        if total_row_wall is None:
            total_row_wall = base._as_float(qg.get("wall_time_sec")) + base._as_float(u.get("wall_time_sec")) + base._as_float(a.get("wall_time_sec"))
        stats["total_lm_wall_time_sec"] += base._as_float(total_row_wall)
    return stats


def main() -> None:
    args = parse_args()
    validate_args(args)
    load_dotenv()

    dataset_path = Path(args.dataset_jsonl)
    out_path = Path(args.out_jsonl)
    trace_path = Path(args.trace_jsonl) if args.trace_jsonl else None
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset jsonl not found: {dataset_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_path:
        trace_path.parent.mkdir(parents=True, exist_ok=True)

    rows_all = list(base.iter_jsonl(dataset_path))
    rows = rows_all[args.start_index :]
    if args.limit > 0:
        rows = rows[: args.limit]
    run_started = time.time()

    done_ids = load_done_ids(out_path, skip_answer=args.skip_answer) if args.resume else set()
    mode = "a" if args.resume else "w"

    counter = base.TokenCounter("cl100k_base")
    retry_policy = base.RetryPolicy(retries=args.retries)
    query_gen_model_name = (
        args.query_gen_model
        if args.query_gen_model
        else (
            "Qwen/Qwen2.5-4B-Instruct"
            if args.query_gen_backend == "hf"
            else ("qwen/qwen3-next-80b-a3b-instruct" if args.query_gen_backend == "openrouter" else args.model)
        )
    )
    llm_update: Optional[Any] = None
    llm_answer: Optional[Any] = None
    llm_query_gen: Optional[Any] = None
    embedder: Optional[base.QwenEmbedder] = None

    if not args.dry_run:
        if args.query_gen_backend == "gemini":
            llm_query_gen = base.GeminiClient(model=query_gen_model_name, retry_policy=retry_policy, timeout_sec=args.timeout_sec)
        elif args.query_gen_backend == "hf":
            llm_query_gen = base.HFInstructClient(
                model_name=query_gen_model_name,
                retry_policy=retry_policy,
                max_new_tokens=args.query_gen_hf_max_new_tokens,
                device_map=args.query_gen_hf_device_map,
                token_counter=counter,
            )
        else:
            llm_query_gen = base.OpenRouterClient(
                model=query_gen_model_name,
                retry_policy=retry_policy,
                timeout_sec=args.timeout_sec,
                token_counter=counter,
                base_url=args.openrouter_base_url,
                http_referer=args.openrouter_http_referer,
                app_title=args.openrouter_app_title,
            )
        if args.llm_backend == "gemini":
            llm_update = base.GeminiClient(model=args.model, retry_policy=retry_policy, timeout_sec=args.timeout_sec)
        else:
            llm_update = base.OpenRouterClient(
                model=args.model,
                retry_policy=retry_policy,
                timeout_sec=args.timeout_sec,
                token_counter=counter,
                base_url=args.openrouter_base_url,
                http_referer=args.openrouter_http_referer,
                app_title=args.openrouter_app_title,
            )
        embedder = base.QwenEmbedder(model_name=args.embed_model, device=args.embed_device, batch_size=args.embed_batch_size)
        if not args.skip_answer:
            answer_model = args.answer_model or args.model
            if args.llm_backend == "gemini":
                llm_answer = base.GeminiClient(model=answer_model, retry_policy=retry_policy, timeout_sec=args.timeout_sec)
            else:
                llm_answer = base.OpenRouterClient(
                    model=answer_model,
                    retry_policy=retry_policy,
                    timeout_sec=args.timeout_sec,
                    token_counter=counter,
                    base_url=args.openrouter_base_url,
                    http_referer=args.openrouter_http_referer,
                    app_title=args.openrouter_app_title,
                )

    with out_path.open(mode, encoding="utf-8") as fout:
        trace_file = trace_path.open(mode, encoding="utf-8") if trace_path else None
        try:
            print(
                f"[start] dataset={args.dataset_jsonl} rows={len(rows)} num_bank_queries={args.num_bank_queries} answer_top_j={args.answer_top_j} answer_use_all_banks={args.answer_use_all_banks}",
                flush=True,
            )
            processed = 0
            skipped_completed = 0
            for row in rows:
                qid = str(row.get("question_id", "")).strip()
                if not qid:
                    continue
                if qid in done_ids:
                    skipped_completed += 1
                    continue

                question = str(row.get("question", ""))
                gold_answer = str(row.get("gold_answer", ""))
                docs = list(row.get("docs") or [])
                if args.max_docs_per_query > 0:
                    docs = docs[: args.max_docs_per_query]

                runtime_error = ""
                update_errors: List[str] = []
                answer_error = ""
                started = time.time()

                query_gen_before = (
                    llm_query_gen.total_calls,
                    llm_query_gen.total_input_tokens,
                    llm_query_gen.total_output_tokens,
                    llm_query_gen.total_wall_time_sec,
                ) if llm_query_gen else (0, 0, 0, 0.0)
                update_before = (
                    llm_update.total_calls,
                    llm_update.total_input_tokens,
                    llm_update.total_output_tokens,
                    llm_update.total_wall_time_sec,
                ) if llm_update else (0, 0, 0, 0.0)
                answer_before = (
                    llm_answer.total_calls,
                    llm_answer.total_input_tokens,
                    llm_answer.total_output_tokens,
                    llm_answer.total_wall_time_sec,
                ) if llm_answer else (0, 0, 0, 0.0)

                capped_docs: List[Dict[str, Any]] = []
                doc_truncations = 0
                for doc in docs:
                    raw_doc_text = base.format_doc_for_prompt(doc)
                    raw_doc_tokens = counter.count(raw_doc_text)
                    doc_text = raw_doc_text
                    was_truncated = False
                    if args.max_doc_tokens > 0 and raw_doc_tokens > args.max_doc_tokens:
                        doc_text = counter.truncate(raw_doc_text, max_tokens=args.max_doc_tokens, strategy=args.doc_truncate_strategy)
                        doc_truncations += 1
                        was_truncated = True
                    capped_docs.append(
                        {
                            "doc_id": str(doc.get("doc_id", "")),
                            "text": doc_text,
                            "raw_doc_tokens": raw_doc_tokens,
                            "doc_tokens_after_cap": counter.count(doc_text),
                            "was_truncated": was_truncated,
                        }
                    )

                z_eff = min(args.z_warm_docs, len(capped_docs))
                warm_docs = capped_docs[:z_eff]
                remaining_docs = capped_docs[z_eff:]
                chunks = base.chunk_docs(remaining_docs, args.refresh_stride_docs)
                print(
                    f"[row_start] qid={qid} docs={len(capped_docs)} warm_docs={len(warm_docs)} remaining_docs={len(remaining_docs)} chunks={len(chunks)}",
                    flush=True,
                )

                proxy_oracle_budget_tokens = (
                    int(args.proxy_oracle_memory_budget_tokens)
                    if int(args.proxy_oracle_memory_budget_tokens) > 0
                    else int(args.memory_budget_tokens)
                )
                oracle_gold_memory_ref = build_oracle_gold_memory_reference(
                    row=row,
                    counter=counter,
                    budget_tokens=proxy_oracle_budget_tokens,
                    truncate_strategy=args.truncate_strategy,
                )
                if args.selection_metric == "query":
                    selection_metric_target_text = question
                elif args.selection_metric == "memory_vs_gold_answer":
                    selection_metric_target_text = gold_answer
                else:
                    selection_metric_target_text = str(oracle_gold_memory_ref.get("text", "") or "")
                selection_objective_field = selection_metric_target_field_name(args.selection_metric)

                num_candidates = max(args.num_bank_queries, args.candidate_multiplier * args.num_bank_queries)
                candidate_clusters_initial: List[Dict[str, Any]] = []
                candidate_cluster_scores_initial: List[Dict[str, Any]] = []
                selected_cluster_keys: List[str] = []
                selected_clusters: List[Dict[str, Any]] = []
                selected_cluster_scores: List[Dict[str, Any]] = []
                initial_selected_clusters: List[Dict[str, Any]] = []
                initial_selected_cluster_scores: List[Dict[str, Any]] = []
                cluster_bank: Dict[str, Dict[str, Any]] = {}
                memory_bank: Dict[str, str] = {}
                cluster_update_events: List[Dict[str, Any]] = []
                cluster_update_count = 0
                cluster_keep_count = 0
                cluster_replace_count = 0
                reintroduced_cluster_count = 0
                new_cluster_full_refresh_count = 0
                cluster_bank_state_over_time: List[Dict[str, Any]] = []
                bank_token_usage_over_time: List[Dict[str, Any]] = []
                dropped_cluster_cache: Dict[str, Dict[str, Any]] = {}
                cluster_gen_fallback_used = False
                overflow_truncate_events = 0
                overflow_compress_calls = 0
                warm_selected_memory_bank: Dict[str, str] = {}
                summary_memory = ""
                summary_update_errors: List[str] = []
                summary_update_failures = 0
                summary_memory_tokens_over_time: List[Dict[str, Any]] = []

                try:
                    if args.dry_run:
                        candidate_clusters_initial = build_fallback_clusters(warm_docs, num_candidates, args.cluster_style, args.max_queries_per_cluster)
                        selected_clusters = candidate_clusters_initial[: args.num_bank_queries]
                        selected_cluster_scores = [
                            {"cluster": c, "score": float(1.0 - (i * 0.01)), "best_query": c.get("queries", [""])[0] if c.get("queries") else ""}
                            for i, c in enumerate(selected_clusters)
                        ]
                        candidate_cluster_scores_initial = [
                            {"cluster": c, "score": float(1.0 - (i * 0.01)), "best_query": c.get("queries", [""])[0] if c.get("queries") else ""}
                            for i, c in enumerate(candidate_clusters_initial)
                        ]
                    else:
                        if llm_query_gen is None or embedder is None:
                            raise RuntimeError("Query-gen model/embedder not initialized.")
                        candidate_clusters_initial = generate_candidate_clusters_warm(
                            llm=llm_query_gen,
                            warm_docs=warm_docs,
                            num_clusters=num_candidates,
                            max_queries_per_cluster=args.max_queries_per_cluster,
                            style=args.cluster_style,
                            temperature=args.query_gen_temperature,
                        )
                        if args.selection_metric == "query":
                            candidate_cluster_scores_initial = score_clusters_by_best_query(
                                embedder=embedder,
                                target_query=question,
                                clusters=candidate_clusters_initial,
                                style=args.cluster_style,
                            )
                        else:
                            candidate_memory_bank_initial: Dict[str, str] = {}
                            for cluster in candidate_clusters_initial:
                                key = cluster_key(cluster, args.cluster_style)
                                init_mem = build_initial_cluster_memory(
                                    llm_update,
                                    cluster,
                                    warm_docs,
                                    args.memory_budget_tokens,
                                    args.update_temperature,
                                    args.cluster_style,
                                )
                                init_mem, was_truncated = base.truncate_to_budget(
                                    init_mem,
                                    counter=counter,
                                    budget_tokens=args.memory_budget_tokens,
                                    truncate_strategy=args.truncate_strategy,
                                )
                                if was_truncated:
                                    overflow_truncate_events += 1
                                candidate_memory_bank_initial[key] = init_mem
                            candidate_keys_initial = [cluster_key(c, args.cluster_style) for c in candidate_clusters_initial]
                            candidate_memory_scores_initial = score_clusters_by_memory_text(
                                embedder=embedder,
                                target_text=selection_metric_target_text,
                                selected_keys=candidate_keys_initial,
                                memory_bank=candidate_memory_bank_initial,
                            )
                            candidate_cluster_scores_initial = scored_clusters_from_score_map(
                                candidate_clusters_initial,
                                args.cluster_style,
                                candidate_memory_scores_initial,
                                objective_field_name=selection_objective_field,
                            )
                        selected_cluster_scores = select_diverse_clusters(
                            candidate_cluster_scores_initial,
                            args.num_bank_queries,
                            args.cluster_style,
                        )
                        selected_clusters = [x["cluster"] for x in selected_cluster_scores]
                        if args.selection_metric != "query":
                            warm_selected_memory_bank = {
                                cluster_key(c, args.cluster_style): candidate_memory_bank_initial.get(cluster_key(c, args.cluster_style), "")
                                for c in selected_clusters
                            }
                        if len(selected_clusters) < args.num_bank_queries:
                            seen = {cluster_key(c, args.cluster_style) for c in selected_clusters}
                            for c in candidate_clusters_initial:
                                k = cluster_key(c, args.cluster_style)
                                if k in seen:
                                    continue
                                if any(clusters_too_similar(c, kept, args.cluster_style) for kept in selected_clusters):
                                    continue
                                seen.add(k)
                                selected_clusters.append(c)
                                selected_cluster_scores.append({"cluster": c, "score": 0.0, "best_query": c.get("queries", [""])[0] if c.get("queries") else ""})
                                if len(selected_clusters) >= args.num_bank_queries:
                                    break
                        if len(selected_clusters) < args.num_bank_queries:
                            for c in build_fallback_clusters(warm_docs, args.num_bank_queries, args.cluster_style, args.max_queries_per_cluster):
                                k = cluster_key(c, args.cluster_style)
                                if k in {cluster_key(x, args.cluster_style) for x in selected_clusters}:
                                    continue
                                selected_clusters.append(c)
                                selected_cluster_scores.append({"cluster": c, "score": 0.0, "best_query": c.get("queries", [""])[0] if c.get("queries") else ""})
                                if len(selected_clusters) >= args.num_bank_queries:
                                    break
                    initial_selected_clusters = list(selected_clusters)
                    initial_selected_cluster_scores = serialize_cluster_scores(selected_cluster_scores)
                except Exception as exc:  # noqa: BLE001
                    update_errors.append(f"cluster_bank_build: {exc}")
                    if not selected_clusters:
                        cluster_gen_fallback_used = True
                        fallback_clusters = build_fallback_clusters(warm_docs, args.num_bank_queries, args.cluster_style, args.max_queries_per_cluster)
                        selected_clusters = fallback_clusters[: args.num_bank_queries]
                        selected_cluster_scores = [{"cluster": c, "score": 0.0, "best_query": c.get("queries", [""])[0] if c.get("queries") else ""} for c in selected_clusters]
                        candidate_clusters_initial = list(candidate_clusters_initial) or fallback_clusters
                        candidate_cluster_scores_initial = serialize_cluster_scores(selected_cluster_scores)
                        initial_selected_clusters = list(selected_clusters)
                        initial_selected_cluster_scores = serialize_cluster_scores(selected_cluster_scores)
                    if not selected_clusters:
                        runtime_error = "cluster_generation_or_selection_failed"

                cluster_bank = {cluster_key(c, args.cluster_style): c for c in selected_clusters}
                selected_cluster_keys = [cluster_key(c, args.cluster_style) for c in selected_clusters]
                if args.selection_metric == "query":
                    memory_bank = {key: "" for key in selected_cluster_keys}
                else:
                    memory_bank = {key: warm_selected_memory_bank.get(key, "") for key in selected_cluster_keys}

                if trace_file and args.log_selection_details:
                    base.write_jsonl_row(
                        trace_file,
                        {
                            "question_id": qid,
                            "phase": "warm_cluster_selection",
                            "num_candidates_generated": len(candidate_clusters_initial),
                            "candidate_clusters": candidate_clusters_initial,
                            "candidate_cluster_scores": serialize_cluster_scores(candidate_cluster_scores_initial),
                            "selected_clusters": initial_selected_clusters,
                            "selected_cluster_scores": initial_selected_cluster_scores,
                        },
                    )
                    base.flush_jsonl_handle(trace_file)

                if not runtime_error and args.selection_metric == "query":
                    for key in selected_cluster_keys:
                        cluster = cluster_bank[key]
                        try:
                            if args.dry_run:
                                init_mem = counter.truncate(base.format_doc_chunk_for_prompt(warm_docs), 256, strategy="head")
                            else:
                                if llm_update is None:
                                    raise RuntimeError("Update model is not initialized.")
                                init_mem = build_initial_cluster_memory(llm_update, cluster, warm_docs, args.memory_budget_tokens, args.update_temperature, args.cluster_style)
                            init_mem, was_truncated = base.truncate_to_budget(init_mem, counter=counter, budget_tokens=args.memory_budget_tokens, truncate_strategy=args.truncate_strategy)
                            if was_truncated:
                                overflow_truncate_events += 1
                            memory_bank[key] = init_mem
                            if trace_file:
                                base.write_jsonl_row(
                                    trace_file,
                                    {
                                        "question_id": qid,
                                        "phase": "init_cluster_memory",
                                        "cluster": cluster,
                                        "memory_tokens": counter.count(init_mem),
                                        "memory_budget_tokens": args.memory_budget_tokens,
                                        "was_truncated": was_truncated,
                                    },
                                )
                                base.flush_jsonl_handle(trace_file)
                        except Exception as exc:  # noqa: BLE001
                            runtime_error = "init_cluster_memory_failed"
                            update_errors.append(f"init_cluster={cluster}: {exc}")
                            break

                if not runtime_error:
                    initial_cluster_state = build_cluster_state_snapshot(
                        selected_keys=selected_cluster_keys,
                        cluster_bank=cluster_bank,
                        memory_bank=memory_bank,
                        scored_clusters=selected_cluster_scores,
                        style=args.cluster_style,
                        counter=counter,
                        embedder=embedder,
                    )
                    initial_cluster_state["phase"] = "warm_after_init"
                    initial_cluster_state["chunk_idx"] = 0
                    cluster_bank_state_over_time.append(initial_cluster_state)
                    bank_token_usage_over_time.append(
                        {
                            "phase": "warm_after_init",
                            "chunk_idx": 0,
                            "memory_tokens_total": initial_cluster_state.get("memory_tokens_total"),
                            "memory_tokens_by_cluster": initial_cluster_state.get("memory_tokens_by_cluster"),
                        }
                    )
                    if trace_file:
                        base.write_jsonl_row(
                            trace_file,
                            {
                                "question_id": qid,
                                "phase": "cluster_bank_state",
                                "chunk_idx": 0,
                                "stage": "warm_after_init",
                                "cluster_state": initial_cluster_state,
                            },
                        )
                        base.flush_jsonl_handle(trace_file)

                if not runtime_error and args.summary_budget_tokens > 0:
                    try:
                        if args.dry_run:
                            summary_candidate = counter.truncate(
                                base.format_doc_chunk_for_prompt(warm_docs),
                                max(1, min(args.summary_budget_tokens, 512)),
                                strategy="head",
                            )
                        else:
                            if llm_update is None:
                                raise RuntimeError("Update model is not initialized.")
                            summary_candidate = base.update_summary_memory(
                                llm=llm_update,
                                current_memory="",
                                docs_chunk=warm_docs,
                                budget_tokens=args.summary_budget_tokens,
                                temperature=args.update_temperature,
                            )
                        summary_memory, _ = base.truncate_to_budget(
                            summary_candidate,
                            counter=counter,
                            budget_tokens=args.summary_budget_tokens,
                            truncate_strategy=args.truncate_strategy,
                        )
                    except Exception as exc:  # noqa: BLE001
                        summary_update_failures += 1
                        summary_update_errors.append("warm_summary: {}".format(exc))
                        summary_memory = ""
                    summary_memory_tokens_over_time.append(
                        {
                            "phase": "warm_after_init",
                            "chunk_idx": 0,
                            "summary_memory_tokens": counter.count(summary_memory),
                        }
                    )
                    if trace_file:
                        base.write_jsonl_row(
                            trace_file,
                            {
                                "question_id": qid,
                                "phase": "summary_memory",
                                "chunk_idx": 0,
                                "stage": "warm_after_init",
                                "summary_memory_tokens": counter.count(summary_memory),
                                "summary_budget_tokens": args.summary_budget_tokens,
                                "summary_update_failures": summary_update_failures,
                            },
                        )
                        base.flush_jsonl_handle(trace_file)

                if not runtime_error:
                    for chunk_idx, docs_chunk in enumerate(chunks, start=1):
                        old_selected_keys = list(selected_cluster_keys)
                        old_cluster_bank = {key: cluster_bank[key] for key in old_selected_keys}
                        old_memory_snapshot = {key: memory_bank.get(key, "") for key in old_selected_keys}
                        old_selected_clusters = [old_cluster_bank[key] for key in old_selected_keys]
                        new_candidates: List[Dict[str, Any]] = []
                        new_selected_clusters: List[Dict[str, Any]] = []
                        new_selected_scores: List[Dict[str, Any]] = []
                        selection_pool_scores: List[Dict[str, Any]] = []
                        old_key_set = set(old_selected_keys)
                        candidate_memory_bank_chunk: Dict[str, str] = {}
                        candidate_memory_source: Dict[str, str] = {}
                        try:
                            if args.dry_run:
                                new_candidates = build_fallback_clusters(docs_chunk, num_candidates, args.cluster_style, args.max_queries_per_cluster)
                                dry_pool = old_selected_clusters + new_candidates
                                new_selected_clusters = dry_pool[: args.num_bank_queries]
                                selection_pool_scores = [
                                    {"cluster": c, "score": float(1.0 - (i * 0.01)), "best_query": c.get("queries", [""])[0] if c.get("queries") else ""}
                                    for i, c in enumerate(dry_pool)
                                ]
                                new_selected_scores = selection_pool_scores[: args.num_bank_queries]
                            else:
                                if llm_query_gen is None or embedder is None:
                                    raise RuntimeError("Query-gen model/embedder not initialized.")
                                new_candidates = generate_candidate_clusters_dynamic(
                                    llm=llm_query_gen,
                                    current_clusters=old_selected_clusters,
                                    current_memory_bank=old_memory_snapshot,
                                    docs_chunk=docs_chunk,
                                    num_clusters=num_candidates,
                                    max_queries_per_cluster=args.max_queries_per_cluster,
                                    style=args.cluster_style,
                                    temperature=args.query_gen_temperature,
                                )
                                pool_by_key: Dict[str, Dict[str, Any]] = {}
                                for cluster in old_selected_clusters + new_candidates:
                                    pool_by_key[cluster_key(cluster, args.cluster_style)] = cluster
                                if args.selection_metric == "query":
                                    selection_pool_scores = score_clusters_by_best_query(
                                        embedder=embedder,
                                        target_query=question,
                                        clusters=list(pool_by_key.values()),
                                        style=args.cluster_style,
                                    )
                                else:
                                    for key, cluster in pool_by_key.items():
                                        if key in old_key_set:
                                            candidate = refresh_cluster_memory(
                                                llm_update,
                                                cluster,
                                                old_memory_snapshot.get(key, ""),
                                                docs_chunk,
                                                args.memory_budget_tokens,
                                                args.update_temperature,
                                                args.cluster_style,
                                            )
                                            candidate_memory_source[key] = "existing"
                                        elif key in dropped_cluster_cache:
                                            candidate = refresh_cluster_memory(
                                                llm_update,
                                                cluster,
                                                str(dropped_cluster_cache[key].get("memory", "") or ""),
                                                docs_chunk,
                                                args.memory_budget_tokens,
                                                args.update_temperature,
                                                args.cluster_style,
                                            )
                                            candidate_memory_source[key] = "reintroduced"
                                        else:
                                            candidate = full_refresh_new_cluster_memory(
                                                llm_update,
                                                cluster,
                                                old_selected_keys,
                                                old_cluster_bank,
                                                old_memory_snapshot,
                                                docs_chunk,
                                                args.memory_budget_tokens,
                                                args.update_temperature,
                                                args.cluster_style,
                                            )
                                            candidate_memory_source[key] = "new"
                                        candidate, was_truncated = base.truncate_to_budget(
                                            candidate,
                                            counter=counter,
                                            budget_tokens=args.memory_budget_tokens,
                                            truncate_strategy=args.truncate_strategy,
                                        )
                                        if was_truncated:
                                            overflow_truncate_events += 1
                                        candidate_memory_bank_chunk[key] = candidate
                                    selection_memory_scores = score_clusters_by_memory_text(
                                        embedder=embedder,
                                        target_text=selection_metric_target_text,
                                        selected_keys=list(pool_by_key.keys()),
                                        memory_bank=candidate_memory_bank_chunk,
                                    )
                                    selection_pool_scores = scored_clusters_from_score_map(
                                        list(pool_by_key.values()),
                                        args.cluster_style,
                                        selection_memory_scores,
                                        objective_field_name=selection_objective_field,
                                    )
                                new_selected_scores = select_diverse_clusters(
                                    selection_pool_scores,
                                    args.num_bank_queries,
                                    args.cluster_style,
                                )
                                new_selected_clusters = [x["cluster"] for x in new_selected_scores]
                                if len(new_selected_clusters) < args.num_bank_queries:
                                    seen = {cluster_key(c, args.cluster_style) for c in new_selected_clusters}
                                    for cluster in pool_by_key.values():
                                        k = cluster_key(cluster, args.cluster_style)
                                        if k in seen:
                                            continue
                                        if any(clusters_too_similar(cluster, kept, args.cluster_style) for kept in new_selected_clusters):
                                            continue
                                        seen.add(k)
                                        new_selected_clusters.append(cluster)
                                        new_selected_scores.append({"cluster": cluster, "score": 0.0, "best_query": cluster.get("queries", [""])[0] if cluster.get("queries") else ""})
                                        if len(new_selected_clusters) >= args.num_bank_queries:
                                            break
                        except Exception as exc:  # noqa: BLE001
                            update_errors.append(f"cluster_bank_rebuild_chunk={chunk_idx}: {exc}")
                            cluster_gen_fallback_used = True
                            new_selected_clusters = list(old_selected_clusters)
                            prev_score_map = {cluster_key(x["cluster"], args.cluster_style): x for x in selected_cluster_scores if isinstance(x, dict) and isinstance(x.get("cluster"), dict)}
                            new_selected_scores = [prev_score_map.get(cluster_key(c, args.cluster_style), {"cluster": c, "score": 0.0, "best_query": c.get("queries", [""])[0] if c.get("queries") else ""}) for c in new_selected_clusters]
                            selection_pool_scores = list(new_selected_scores)

                        new_selected_keys = [cluster_key(c, args.cluster_style) for c in new_selected_clusters]
                        new_key_set = set(new_selected_keys)
                        cluster_set_changed = old_key_set != new_key_set
                        if cluster_set_changed:
                            cluster_update_count += 1
                        else:
                            cluster_keep_count += 1
                        cluster_replace_count += sum(1 for key in old_selected_keys if key not in new_key_set)
                        for key in old_selected_keys:
                            if key not in new_key_set:
                                dropped_cluster_cache[key] = {"cluster": old_cluster_bank[key], "memory": old_memory_snapshot.get(key, "")}

                        if trace_file:
                            base.write_jsonl_row(
                                trace_file,
                                {
                                    "question_id": qid,
                                    "phase": "cluster_bank_reselection",
                                    "chunk_idx": chunk_idx,
                                    "chunk_size_docs": len(docs_chunk),
                                    "cluster_set_changed": bool(cluster_set_changed),
                                    "generated_candidate_clusters": new_candidates if args.log_selection_details else [],
                                    "selection_pool_scores": serialize_cluster_scores(selection_pool_scores) if args.log_selection_details else [],
                                    "new_selected_clusters": new_selected_clusters,
                                    "new_selected_cluster_scores": serialize_cluster_scores(new_selected_scores),
                                },
                            )
                            base.flush_jsonl_handle(trace_file)

                        updated_cluster_bank = {cluster_key(c, args.cluster_style): c for c in new_selected_clusters}
                        updated_memory_bank: Dict[str, str] = {}
                        if args.selection_metric == "query":
                            for key, cluster in updated_cluster_bank.items():
                                try:
                                    if key in old_key_set:
                                        prev = old_memory_snapshot.get(key, "")
                                        if args.dry_run:
                                            candidate = (prev + "\n\n" + counter.truncate(base.format_doc_chunk_for_prompt(docs_chunk), 256, strategy="head")).strip()
                                        else:
                                            if llm_update is None:
                                                raise RuntimeError("Update model is not initialized.")
                                            candidate = refresh_cluster_memory(llm_update, cluster, prev, docs_chunk, args.memory_budget_tokens, args.update_temperature, args.cluster_style)
                                    elif key in dropped_cluster_cache:
                                        reintroduced_cluster_count += 1
                                        prev = str(dropped_cluster_cache[key].get("memory", "") or "")
                                        if args.dry_run:
                                            candidate = (prev + "\n\n" + counter.truncate(base.format_doc_chunk_for_prompt(docs_chunk), 256, strategy="head")).strip()
                                        else:
                                            if llm_update is None:
                                                raise RuntimeError("Update model is not initialized.")
                                            candidate = refresh_cluster_memory(llm_update, cluster, prev, docs_chunk, args.memory_budget_tokens, args.update_temperature, args.cluster_style)
                                    else:
                                        new_cluster_full_refresh_count += 1
                                        if args.dry_run:
                                            candidate = counter.truncate(cluster_blob(old_selected_keys, old_cluster_bank, old_memory_snapshot, args.cluster_style) + "\n\n" + base.format_doc_chunk_for_prompt(docs_chunk), 256, strategy="head")
                                        else:
                                            if llm_update is None:
                                                raise RuntimeError("Update model is not initialized.")
                                            candidate = full_refresh_new_cluster_memory(llm_update, cluster, old_selected_keys, old_cluster_bank, old_memory_snapshot, docs_chunk, args.memory_budget_tokens, args.update_temperature, args.cluster_style)
                                    candidate, was_truncated = base.truncate_to_budget(candidate, counter=counter, budget_tokens=args.memory_budget_tokens, truncate_strategy=args.truncate_strategy)
                                    if was_truncated:
                                        overflow_truncate_events += 1
                                    updated_memory_bank[key] = candidate
                                    if trace_file:
                                        base.write_jsonl_row(
                                            trace_file,
                                            {
                                                "question_id": qid,
                                                "phase": "refresh_cluster_memory",
                                                "chunk_idx": chunk_idx,
                                                "cluster": cluster,
                                                "memory_tokens": counter.count(candidate),
                                                "memory_budget_tokens": args.memory_budget_tokens,
                                                "was_truncated": was_truncated,
                                            },
                                        )
                                        base.flush_jsonl_handle(trace_file)
                                except Exception as exc:  # noqa: BLE001
                                    runtime_error = f"cluster_refresh_failed_chunk_{chunk_idx}"
                                    update_errors.append(f"cluster_refresh_chunk={chunk_idx} cluster={cluster}: {exc}")
                                    break
                        else:
                            try:
                                for key, cluster in updated_cluster_bank.items():
                                    source = candidate_memory_source.get(key, "")
                                    if source == "reintroduced":
                                        reintroduced_cluster_count += 1
                                    elif source == "new":
                                        new_cluster_full_refresh_count += 1
                                    updated_memory_bank[key] = candidate_memory_bank_chunk.get(
                                        key,
                                        old_memory_snapshot.get(key, ""),
                                    )
                                    if trace_file:
                                        base.write_jsonl_row(
                                            trace_file,
                                            {
                                                "question_id": qid,
                                                "phase": "refresh_cluster_memory",
                                                "chunk_idx": chunk_idx,
                                                "cluster": cluster,
                                                "memory_tokens": counter.count(updated_memory_bank[key]),
                                                "memory_budget_tokens": args.memory_budget_tokens,
                                                "was_truncated": False,
                                                "selection_driven_refresh": True,
                                            },
                                        )
                                        base.flush_jsonl_handle(trace_file)
                            except Exception as exc:  # noqa: BLE001
                                runtime_error = f"cluster_refresh_failed_chunk_{chunk_idx}"
                                update_errors.append(f"cluster_refresh_chunk={chunk_idx} cluster={cluster}: {exc}")
                                break
                        if runtime_error:
                            break
                        cluster_bank = updated_cluster_bank
                        memory_bank = updated_memory_bank
                        selected_cluster_keys = [cluster_key(c, args.cluster_style) for c in new_selected_clusters]
                        selected_clusters = list(new_selected_clusters)
                        selected_cluster_scores = list(new_selected_scores)
                        if args.summary_budget_tokens > 0:
                            try:
                                if args.dry_run:
                                    summary_candidate = (
                                        summary_memory + "\n\n" + counter.truncate(
                                            base.format_doc_chunk_for_prompt(docs_chunk),
                                            max(1, min(args.summary_budget_tokens, 256)),
                                            strategy="head",
                                        )
                                    ).strip()
                                else:
                                    if llm_update is None:
                                        raise RuntimeError("Update model is not initialized.")
                                    summary_candidate = base.update_summary_memory(
                                        llm=llm_update,
                                        current_memory=summary_memory,
                                        docs_chunk=docs_chunk,
                                        budget_tokens=args.summary_budget_tokens,
                                        temperature=args.update_temperature,
                                    )
                                summary_memory, _ = base.truncate_to_budget(
                                    summary_candidate,
                                    counter=counter,
                                    budget_tokens=args.summary_budget_tokens,
                                    truncate_strategy=args.truncate_strategy,
                                )
                            except Exception as exc:  # noqa: BLE001
                                summary_update_failures += 1
                                summary_update_errors.append("chunk_{}: {}".format(chunk_idx, exc))
                            summary_memory_tokens_over_time.append(
                                {
                                    "phase": "after_chunk_refresh",
                                    "chunk_idx": chunk_idx,
                                    "summary_memory_tokens": counter.count(summary_memory),
                                }
                            )
                            if trace_file:
                                base.write_jsonl_row(
                                    trace_file,
                                    {
                                        "question_id": qid,
                                        "phase": "summary_memory",
                                        "chunk_idx": chunk_idx,
                                        "stage": "after_chunk_refresh",
                                        "summary_memory_tokens": counter.count(summary_memory),
                                        "summary_budget_tokens": args.summary_budget_tokens,
                                        "summary_update_failures": summary_update_failures,
                                    },
                                )
                                base.flush_jsonl_handle(trace_file)
                        cluster_update_events.append(
                            {
                                "chunk_idx": chunk_idx,
                                "cluster_set_changed": bool(cluster_set_changed),
                                "selected_clusters": selected_clusters,
                                "num_new_candidates": len(new_candidates),
                            }
                        )
                        current_cluster_state = build_cluster_state_snapshot(
                            selected_keys=selected_cluster_keys,
                            cluster_bank=cluster_bank,
                            memory_bank=memory_bank,
                            scored_clusters=selected_cluster_scores,
                            style=args.cluster_style,
                            counter=counter,
                            embedder=embedder,
                        )
                        current_cluster_state["phase"] = "after_chunk_refresh"
                        current_cluster_state["chunk_idx"] = chunk_idx
                        cluster_bank_state_over_time.append(current_cluster_state)
                        bank_token_usage_over_time.append(
                            {
                                "phase": "after_chunk_refresh",
                                "chunk_idx": chunk_idx,
                                "memory_tokens_total": current_cluster_state.get("memory_tokens_total"),
                                "memory_tokens_by_cluster": current_cluster_state.get("memory_tokens_by_cluster"),
                            }
                        )
                        if trace_file:
                            base.write_jsonl_row(
                                trace_file,
                                {
                                    "question_id": qid,
                                    "phase": "cluster_bank_state",
                                    "chunk_idx": chunk_idx,
                                    "stage": "after_chunk_refresh",
                                    "cluster_state": current_cluster_state,
                                },
                            )
                            base.flush_jsonl_handle(trace_file)

                answer_top_j_effective = min(args.answer_top_j, len(selected_cluster_keys))
                answer_selected_clusters: List[Dict[str, Any]] = []
                answer_selected_cluster_keys: List[str] = []
                answer_selected_scores: List[Dict[str, Any]] = []
                if selected_clusters:
                    if args.answer_use_all_banks:
                        answer_selected_clusters = list(selected_clusters)
                        answer_selected_scores = list(selected_cluster_scores)
                    elif args.dry_run:
                        answer_selected_clusters = selected_clusters[:answer_top_j_effective]
                        answer_selected_scores = selected_cluster_scores[:answer_top_j_effective]
                    else:
                        if embedder is None:
                            raise RuntimeError("Embedder not initialized for answer selection.")
                        scored_j = score_clusters_by_best_query(embedder=embedder, target_query=question, clusters=selected_clusters, style=args.cluster_style)
                        answer_selected_clusters = [x["cluster"] for x in scored_j[:answer_top_j_effective]]
                        answer_selected_scores = scored_j[:answer_top_j_effective]
                    answer_selected_cluster_keys = [cluster_key(c, args.cluster_style) for c in answer_selected_clusters]
                    if args.answer_use_all_banks:
                        answer_top_j_effective = len(answer_selected_cluster_keys)

                final_answer = ""
                if runtime_error:
                    answer_error = "skipped_due_to_update_error"
                elif args.skip_answer:
                    final_answer = ""
                elif args.dry_run:
                    final_answer = "DRY_RUN"
                else:
                    try:
                        if llm_answer is None:
                            raise RuntimeError("Answer model is not initialized.")
                        if args.answer_render_style == "flat":
                            memory_banks_blob = build_flat_cluster_memory_blob(answer_selected_cluster_keys, memory_bank)
                        else:
                            memory_banks_blob = build_cluster_memory_blob(answer_selected_cluster_keys, cluster_bank, memory_bank, args.cluster_style)
                        if args.summary_budget_tokens > 0:
                            answer_prompt = ANSWER_FROM_CLUSTER_BANK_WITH_SUMMARY_PROMPT.format(
                                target_query=question,
                                summary_memory_bank=summary_memory if summary_memory else "(empty)",
                                memory_banks=memory_banks_blob if memory_banks_blob else "(empty)",
                            )
                        else:
                            answer_prompt = ANSWER_FROM_CLUSTER_BANK_PROMPT.format(
                                target_query=question,
                                memory_banks=memory_banks_blob if memory_banks_blob else "(empty)",
                            )
                        final_answer = llm_answer.generate(answer_prompt, temperature=args.answer_temperature).strip()
                    except Exception as exc:  # noqa: BLE001
                        answer_error = f"answer_failed: {exc}"

                query_gen_after = (
                    llm_query_gen.total_calls,
                    llm_query_gen.total_input_tokens,
                    llm_query_gen.total_output_tokens,
                    llm_query_gen.total_wall_time_sec,
                ) if llm_query_gen else (0, 0, 0, 0.0)
                update_after = (
                    llm_update.total_calls,
                    llm_update.total_input_tokens,
                    llm_update.total_output_tokens,
                    llm_update.total_wall_time_sec,
                ) if llm_update else (0, 0, 0, 0.0)
                answer_after = (
                    llm_answer.total_calls,
                    llm_answer.total_input_tokens,
                    llm_answer.total_output_tokens,
                    llm_answer.total_wall_time_sec,
                ) if llm_answer else (0, 0, 0, 0.0)
                query_gen_usage = {
                    "calls": query_gen_after[0] - query_gen_before[0],
                    "input_tokens": query_gen_after[1] - query_gen_before[1],
                    "output_tokens": query_gen_after[2] - query_gen_before[2],
                    "wall_time_sec": round(query_gen_after[3] - query_gen_before[3], 6),
                }
                update_usage = {
                    "calls": update_after[0] - update_before[0],
                    "input_tokens": update_after[1] - update_before[1],
                    "output_tokens": update_after[2] - update_before[2],
                    "wall_time_sec": round(update_after[3] - update_before[3], 6),
                }
                answer_usage = {
                    "calls": answer_after[0] - answer_before[0],
                    "input_tokens": answer_after[1] - answer_before[1],
                    "output_tokens": answer_after[2] - answer_before[2],
                    "wall_time_sec": round(answer_after[3] - answer_before[3], 6),
                }
                total_lm_calls = query_gen_usage["calls"] + update_usage["calls"] + answer_usage["calls"]
                total_lm_input_tokens = query_gen_usage["input_tokens"] + update_usage["input_tokens"] + answer_usage["input_tokens"]
                total_lm_output_tokens = query_gen_usage["output_tokens"] + update_usage["output_tokens"] + answer_usage["output_tokens"]
                total_lm_wall_time_sec = query_gen_usage["wall_time_sec"] + update_usage["wall_time_sec"] + answer_usage["wall_time_sec"]

                memory_tokens_by_cluster = {key: counter.count(memory_bank.get(key, "")) for key in selected_cluster_keys}
                memory_tokens_total = int(sum(memory_tokens_by_cluster.values()))
                if args.answer_render_style == "flat":
                    memory_text = build_flat_cluster_memory_blob(selected_cluster_keys, memory_bank)
                else:
                    memory_text = build_cluster_memory_blob(selected_cluster_keys, cluster_bank, memory_bank, args.cluster_style)
                memory_scores_by_cluster: Dict[str, float] = {}
                memory_scores_by_cluster_gold_answer: Dict[str, float] = {}
                memory_scores_by_cluster_oracle_gold_memory: Dict[str, float] = {}
                answer_memory_scores: Dict[str, float] = {}
                answer_memory_scores_gold_answer: Dict[str, float] = {}
                answer_memory_scores_oracle_gold_memory: Dict[str, float] = {}
                answer_query_scores: List[Dict[str, Any]] = []
                if not args.dry_run and embedder is not None and selected_cluster_keys:
                    memory_scores_by_cluster = score_clusters_by_memory(
                        embedder=embedder,
                        target_query=question,
                        selected_keys=selected_cluster_keys,
                        cluster_bank=cluster_bank,
                        memory_bank=memory_bank,
                        style=args.cluster_style,
                    )
                    memory_scores_by_cluster_gold_answer = score_clusters_by_memory_text(
                        embedder=embedder,
                        target_text=gold_answer,
                        selected_keys=selected_cluster_keys,
                        memory_bank=memory_bank,
                    )
                    memory_scores_by_cluster_oracle_gold_memory = score_clusters_by_memory_text(
                        embedder=embedder,
                        target_text=str(oracle_gold_memory_ref.get("text", "") or ""),
                        selected_keys=selected_cluster_keys,
                        memory_bank=memory_bank,
                    )
                    if answer_selected_cluster_keys:
                        answer_query_scores = score_clusters_by_best_query(
                            embedder=embedder,
                            target_query=question,
                            clusters=answer_selected_clusters,
                            style=args.cluster_style,
                        )
                        answer_memory_scores = {
                            key: memory_scores_by_cluster.get(key, 0.0)
                            for key in answer_selected_cluster_keys
                        }
                        answer_memory_scores_gold_answer = {
                            key: memory_scores_by_cluster_gold_answer.get(key, 0.0)
                            for key in answer_selected_cluster_keys
                        }
                        answer_memory_scores_oracle_gold_memory = {
                            key: memory_scores_by_cluster_oracle_gold_memory.get(key, 0.0)
                            for key in answer_selected_cluster_keys
                        }
                selected_cluster_scores_with_memory = attach_named_scores(
                    selected_cluster_scores,
                    args.cluster_style,
                    memory_score=memory_scores_by_cluster,
                    memory_score_gold_answer=memory_scores_by_cluster_gold_answer,
                    memory_score_oracle_gold_memory=memory_scores_by_cluster_oracle_gold_memory,
                )
                answer_selected_scores_with_memory = attach_named_scores(
                    answer_selected_scores,
                    args.cluster_style,
                    memory_score=answer_memory_scores,
                    memory_score_gold_answer=answer_memory_scores_gold_answer,
                    memory_score_oracle_gold_memory=answer_memory_scores_oracle_gold_memory,
                )
                sim_top1 = answer_selected_scores[0]["score"] if answer_selected_scores else None
                sim_topj_mean = (float(sum(x["score"] for x in answer_selected_scores) / len(answer_selected_scores)) if answer_selected_scores else None)
                if answer_query_scores:
                    query_top1 = answer_query_scores[0]["score"]
                    query_topj_mean = float(sum(x["score"] for x in answer_query_scores) / len(answer_query_scores))
                elif args.dry_run and answer_selected_scores:
                    query_top1 = answer_selected_scores[0]["score"]
                    query_topj_mean = float(sum(x["score"] for x in answer_selected_scores) / len(answer_selected_scores))
                else:
                    query_top1 = None
                    query_topj_mean = None
                mem_top1, mem_topj_mean = score_field_top1_and_mean(
                    answer_selected_scores_with_memory,
                    "memory_score",
                )
                gold_answer_mem_top1, gold_answer_mem_topj_mean = score_field_top1_and_mean(
                    answer_selected_scores_with_memory,
                    "memory_score_gold_answer",
                )
                oracle_gold_mem_top1, oracle_gold_mem_topj_mean = score_field_top1_and_mean(
                    answer_selected_scores_with_memory,
                    "memory_score_oracle_gold_memory",
                )
                final_cluster_state = (
                    dict(cluster_bank_state_over_time[-1])
                    if cluster_bank_state_over_time else
                    build_cluster_state_snapshot(
                        selected_keys=selected_cluster_keys,
                        cluster_bank=cluster_bank,
                        memory_bank=memory_bank,
                        scored_clusters=selected_cluster_scores,
                        style=args.cluster_style,
                        counter=counter,
                        embedder=embedder,
                    )
                )
                method_name = (
                    "oracle_assisted_warm_dynamic_flat_cluster_bank"
                    if args.answer_render_style == "flat"
                    else f"oracle_assisted_warm_dynamic_cluster_bank_{args.cluster_style}"
                )
                if args.summary_budget_tokens > 0:
                    method_name = f"{method_name}_with_summary"
                method_name = f"{method_name}_sel_{args.selection_metric}"
                variant_name = (
                    f"{method_name}_N{args.num_bank_queries}_J{args.answer_top_j}"
                )
                out_row = {
                    "variant": variant_name,
                    "method": method_name,
                    "selection_metric": args.selection_metric,
                    "question_id": qid,
                    "question": question,
                    "gold_answer": gold_answer,
                    "llm_backend": args.llm_backend,
                    "update_model": args.model,
                    "query_gen_backend": args.query_gen_backend,
                    "query_gen_model": query_gen_model_name if not args.dry_run else "",
                    "query_gen_fallback_used": bool(cluster_gen_fallback_used),
                    "answer_model": (args.answer_model or args.model) if not args.skip_answer else "",
                    "embed_model": args.embed_model if not args.dry_run else "",
                    "cluster_style": args.cluster_style,
                    "answer_render_style": args.answer_render_style,
                    "max_queries_per_cluster": args.max_queries_per_cluster,
                    "num_stream_docs": len(capped_docs),
                    "z_warm_docs": args.z_warm_docs,
                    "z_warm_docs_effective": z_eff,
                    "refresh_stride_docs": args.refresh_stride_docs,
                    "num_refresh_chunks": len(chunks),
                    "num_bank_queries": args.num_bank_queries,
                    "num_bank_clusters": args.num_bank_queries,
                    "answer_top_j": args.answer_top_j,
                    "answer_top_j_effective": answer_top_j_effective,
                    "answer_use_all_banks": bool(args.answer_use_all_banks),
                    "candidate_multiplier": args.candidate_multiplier,
                    "summary_budget_tokens": args.summary_budget_tokens,
                    "summary_memory": summary_memory,
                    "summary_memory_tokens": counter.count(summary_memory),
                    "summary_update_failures": summary_update_failures,
                    "summary_update_errors": summary_update_errors,
                    "summary_memory_tokens_over_time": summary_memory_tokens_over_time,
                    "log_selection_details": bool(args.log_selection_details),
                    "candidate_clusters_initial": candidate_clusters_initial,
                    "candidate_cluster_scores_initial": serialize_cluster_scores(candidate_cluster_scores_initial),
                    "initial_selected_clusters": initial_selected_clusters,
                    "initial_selected_cluster_scores": initial_selected_cluster_scores,
                    "initial_cluster_state": cluster_bank_state_over_time[0] if cluster_bank_state_over_time else {},
                    "selected_clusters": selected_clusters,
                    "selected_cluster_scores": serialize_cluster_scores(selected_cluster_scores_with_memory),
                    "final_cluster_state": final_cluster_state,
                    "cluster_bank_state_over_time": cluster_bank_state_over_time,
                    "bank_token_usage_over_time": bank_token_usage_over_time,
                    "cluster_update_attempts": len(chunks),
                    "cluster_update_count": cluster_update_count,
                    "cluster_keep_count": cluster_keep_count,
                    "cluster_replace_count": cluster_replace_count,
                    "reintroduced_cluster_count": reintroduced_cluster_count,
                    "new_cluster_full_refresh_count": new_cluster_full_refresh_count,
                    "cluster_update_events": cluster_update_events,
                    "answer_selected_clusters": answer_selected_clusters,
                    "answer_selected_cluster_scores": serialize_cluster_scores(answer_selected_scores_with_memory),
                    "cluster_similarity": {
                        "answer_selected_top1": sim_top1,
                        "answer_selected_topj_mean": sim_topj_mean,
                        "answer_selected_query_top1": query_top1,
                        "answer_selected_query_topj_mean": query_topj_mean,
                        "answer_selected_memory_top1": mem_top1,
                        "answer_selected_memory_topj_mean": mem_topj_mean,
                        "answer_selected_memory_vs_question_top1": mem_top1,
                        "answer_selected_memory_vs_question_topj_mean": mem_topj_mean,
                        "answer_selected_memory_vs_gold_answer_top1": gold_answer_mem_top1,
                        "answer_selected_memory_vs_gold_answer_topj_mean": gold_answer_mem_topj_mean,
                        "answer_selected_memory_vs_oracle_gold_memory_top1": oracle_gold_mem_top1,
                        "answer_selected_memory_vs_oracle_gold_memory_topj_mean": oracle_gold_mem_topj_mean,
                    },
                    "oracle_gold_memory_text": str(oracle_gold_memory_ref.get("text", "") or ""),
                    "oracle_gold_memory_source": str(oracle_gold_memory_ref.get("source", "") or ""),
                    "oracle_gold_memory_tokens_raw": int(oracle_gold_memory_ref.get("raw_tokens", 0) or 0),
                    "oracle_gold_memory_tokens": int(oracle_gold_memory_ref.get("used_tokens", 0) or 0),
                    "oracle_gold_memory_was_truncated": bool(oracle_gold_memory_ref.get("was_truncated", False)),
                    "oracle_gold_memory_budget_tokens": int(proxy_oracle_budget_tokens),
                    "oracle_gold_memory_gold_doc_count": int(oracle_gold_memory_ref.get("gold_doc_count", 0) or 0),
                    "gold_answer_tokens": counter.count(gold_answer),
                    "memory_bank_by_cluster": [
                        {
                            "cluster": cluster_bank[key],
                            "memory": memory_bank.get(key, ""),
                            "memory_tokens": memory_tokens_by_cluster.get(key, 0),
                            "memory_similarity": (
                                float(memory_scores_by_cluster[key])
                                if key in memory_scores_by_cluster else None
                            ),
                            "memory_similarity_gold_answer": (
                                float(memory_scores_by_cluster_gold_answer[key])
                                if key in memory_scores_by_cluster_gold_answer else None
                            ),
                            "memory_similarity_oracle_gold_memory": (
                                float(memory_scores_by_cluster_oracle_gold_memory[key])
                                if key in memory_scores_by_cluster_oracle_gold_memory else None
                            ),
                        }
                        for key in selected_cluster_keys
                    ],
                    "memory_tokens_by_cluster": memory_tokens_by_cluster,
                    "memory_text": memory_text,
                    "memory_tokens": memory_tokens_total,
                    "memory_budget_tokens": args.memory_budget_tokens,
                    "overflow_policy": args.overflow_policy,
                    "truncate_strategy": args.truncate_strategy,
                    "doc_truncate_strategy": args.doc_truncate_strategy,
                    "doc_truncations": doc_truncations,
                    "overflow_compress_calls": overflow_compress_calls,
                    "overflow_truncate_events": overflow_truncate_events,
                    "row_failed": bool(runtime_error),
                    "model_answer": final_answer,
                    "update_errors": update_errors,
                    "answer_error": answer_error,
                    "runtime_error": runtime_error,
                    "query_gen_lm_usage": query_gen_usage,
                    "update_lm_usage": update_usage,
                    "answer_lm_usage": answer_usage,
                    "total_lm_calls": total_lm_calls,
                    "total_lm_input_tokens": total_lm_input_tokens,
                    "total_lm_output_tokens": total_lm_output_tokens,
                    "total_lm_tokens": total_lm_input_tokens + total_lm_output_tokens,
                    "query_gen_lm_wall_time_sec": query_gen_usage["wall_time_sec"],
                    "update_lm_wall_time_sec": update_usage["wall_time_sec"],
                    "answer_lm_wall_time_sec": answer_usage["wall_time_sec"],
                    "total_lm_wall_time_sec": round(total_lm_wall_time_sec, 6),
                    "runtime_sec": round(time.time() - started, 3),
                    "dry_run": bool(args.dry_run),
                }
                row_metadata.attach_sample_metadata(out_row, row)
                base.write_jsonl_row(fout, out_row)
                base.flush_jsonl_handle(fout)
                if trace_file:
                    base.write_jsonl_row(
                        trace_file,
                        {
                            "question_id": qid,
                            "phase": "summary",
                            "num_stream_docs": len(capped_docs),
                            "num_selected_clusters": len(selected_clusters),
                            "num_answer_selected_clusters": len(answer_selected_clusters),
                            "cluster_update_attempts": len(chunks),
                            "cluster_update_count": cluster_update_count,
                            "selection_metric": args.selection_metric,
                            "answer_selected_query_top1": query_top1,
                            "answer_selected_query_topj_mean": query_topj_mean,
                            "answer_selected_memory_top1": mem_top1,
                            "answer_selected_memory_topj_mean": mem_topj_mean,
                            "answer_selected_memory_vs_gold_answer_top1": gold_answer_mem_top1,
                            "answer_selected_memory_vs_gold_answer_topj_mean": gold_answer_mem_topj_mean,
                            "answer_selected_memory_vs_oracle_gold_memory_top1": oracle_gold_mem_top1,
                            "answer_selected_memory_vs_oracle_gold_memory_topj_mean": oracle_gold_mem_topj_mean,
                            "memory_tokens": memory_tokens_total,
                            "final_cluster_state": final_cluster_state,
                            "overflow_truncate_events": overflow_truncate_events,
                            "runtime_error": runtime_error,
                        },
                    )
                    base.flush_jsonl_handle(trace_file)
                if runtime_error or answer_error:
                    first_update_error = update_errors[0] if update_errors else ""
                    print(
                        f"[row_error] qid={qid} runtime_error={runtime_error or 'none'} answer_error={answer_error or 'none'} update_error={first_update_error or 'none'}",
                        flush=True,
                    )
                processed += 1
                done_ids.add(qid)
                if args.progress_every > 0 and processed % args.progress_every == 0:
                    sim_str = ""
                    if answer_selected_scores:
                        scores = [f"{float(x['score']):.4f}" for x in answer_selected_scores]
                        sim_str = f" query_sim=[{','.join(scores)}]"
                    mem_sim_str = ""
                    if answer_selected_scores_with_memory:
                        mem_scores = [
                            f"{float(x['memory_score']):.4f}"
                            for x in answer_selected_scores_with_memory
                            if x.get("memory_score") is not None
                        ]
                        if mem_scores:
                            mem_sim_str = f" memory_sim=[{','.join(mem_scores)}]"
                    gold_mem_sim_str = ""
                    if answer_selected_scores_with_memory:
                        gold_mem_scores = [
                            f"{float(x['memory_score_gold_answer']):.4f}"
                            for x in answer_selected_scores_with_memory
                            if x.get("memory_score_gold_answer") is not None
                        ]
                        if gold_mem_scores:
                            gold_mem_sim_str = f" gold_answer_mem_sim=[{','.join(gold_mem_scores)}]"
                    oracle_mem_sim_str = ""
                    if answer_selected_scores_with_memory:
                        oracle_mem_scores = [
                            f"{float(x['memory_score_oracle_gold_memory']):.4f}"
                            for x in answer_selected_scores_with_memory
                            if x.get("memory_score_oracle_gold_memory") is not None
                        ]
                        if oracle_mem_scores:
                            oracle_mem_sim_str = f" oracle_mem_sim=[{','.join(oracle_mem_scores)}]"
                    final_c_str = ""
                    if answer_selected_clusters:
                        c0 = cluster_label(answer_selected_clusters[0], args.cluster_style)
                        c0_short = (c0[:50] + "…") if len(c0) > 50 else c0
                        final_c_str = f" final_cluster=\"{c0_short}\""
                    print(
                        f"[progress] processed={processed} last_qid={qid} memory_tokens={memory_tokens_total} cluster_updates={cluster_update_count}/{len(chunks)}{sim_str}{mem_sim_str}{gold_mem_sim_str}{oracle_mem_sim_str}{final_c_str}",
                        flush=True,
                    )
        finally:
            if trace_file:
                trace_file.close()

    totals = aggregate_output_totals(out_path, skip_answer=args.skip_answer)
    answer_model_name = "" if args.skip_answer else (args.answer_model or args.model)
    manifest = {
        "method": f"oracle_assisted_warm_dynamic_cluster_bank_{args.cluster_style}",
        "dataset_jsonl": str(dataset_path),
        "out_jsonl": str(out_path),
        "trace_jsonl": str(trace_path) if trace_path else "",
        "llm_backend": args.llm_backend,
        "query_gen_backend": args.query_gen_backend,
        "query_gen_model": query_gen_model_name,
        "query_gen_hf_max_new_tokens": args.query_gen_hf_max_new_tokens,
        "query_gen_hf_device_map": args.query_gen_hf_device_map,
        "openrouter_base_url": args.openrouter_base_url,
        "openrouter_http_referer": args.openrouter_http_referer,
        "openrouter_app_title": args.openrouter_app_title,
        "update_model": args.model,
        "answer_model": answer_model_name,
        "embed_model": args.embed_model,
        "embed_device": args.embed_device,
        "embed_batch_size": args.embed_batch_size,
        "memory_budget_tokens": args.memory_budget_tokens,
        "proxy_oracle_memory_budget_tokens": (
            args.proxy_oracle_memory_budget_tokens
            if args.proxy_oracle_memory_budget_tokens > 0
            else args.memory_budget_tokens
        ),
        "z_warm_docs": args.z_warm_docs,
        "num_bank_queries": args.num_bank_queries,
        "answer_top_j": args.answer_top_j,
        "candidate_multiplier": args.candidate_multiplier,
        "refresh_stride_docs": args.refresh_stride_docs,
        "max_queries_per_cluster": args.max_queries_per_cluster,
        "cluster_style": args.cluster_style,
        "overflow_policy": args.overflow_policy,
        "memory_truncate_strategy": args.truncate_strategy,
        "doc_truncate_strategy": args.doc_truncate_strategy,
        "max_doc_tokens": args.max_doc_tokens,
        "max_docs_per_query": args.max_docs_per_query,
        "start_index": args.start_index,
        "limit": args.limit,
        "resume": bool(args.resume),
        "skip_answer": bool(args.skip_answer),
        "dry_run": bool(args.dry_run),
        "retries": args.retries,
        "timeout_sec": args.timeout_sec,
        "query_gen_temperature": args.query_gen_temperature,
        "update_temperature": args.update_temperature,
        "answer_temperature": args.answer_temperature,
        "log_selection_details": bool(args.log_selection_details),
        "rows_targeted": len(rows),
        "rows_skipped_completed": skipped_completed,
        "rows_processed_this_run": totals["rows_written_total"] - skipped_completed,
        "rows_written_total": totals["rows_written_total"],
        "rows_completed_total": totals["rows_completed_total"],
        "rows_runtime_error_total": totals["rows_runtime_error_total"],
        "query_gen_calls_total": totals["query_gen_calls"],
        "query_gen_input_tokens_total": totals["query_gen_input_tokens"],
        "query_gen_output_tokens_total": totals["query_gen_output_tokens"],
        "update_calls_total": totals["update_calls"],
        "update_input_tokens_total": totals["update_input_tokens"],
        "update_output_tokens_total": totals["update_output_tokens"],
        "answer_calls_total": totals["answer_calls"],
        "answer_input_tokens_total": totals["answer_input_tokens"],
        "answer_output_tokens_total": totals["answer_output_tokens"],
        "query_gen_wall_time_sec_total": round(totals["query_gen_wall_time_sec"], 6),
        "update_wall_time_sec_total": round(totals["update_wall_time_sec"], 6),
        "answer_wall_time_sec_total": round(totals["answer_wall_time_sec"], 6),
        "total_lm_wall_time_sec_total": round(totals["total_lm_wall_time_sec"], 6),
        "runtime_sec_total": round(time.time() - run_started, 3),
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[done] query_gen_calls={totals['query_gen_calls']} update_calls={totals['update_calls']} answer_calls={totals['answer_calls']} total_lm_wall_time_sec={manifest['total_lm_wall_time_sec_total']}",
        flush=True,
    )
    print(f"[done] wrote manifest to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
