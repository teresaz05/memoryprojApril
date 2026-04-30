#!/usr/bin/env python3
from __future__ import annotations

import difflib
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import support as base
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

def build_cluster_memory_blob(selected_keys: Sequence[str], cluster_bank: Dict[str, Dict[str, Any]], memory_bank: Dict[str, str], style: str) -> str:
    return cluster_blob(selected_keys, cluster_bank, memory_bank, style)
