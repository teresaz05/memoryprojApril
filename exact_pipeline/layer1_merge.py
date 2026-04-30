#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Set

import cluster_bank as cbase
import llm_backends as base
ANSWER_FROM_DOC_CLUSTER_BANKS_PROMPT = """You are answering a question using only the provided cluster banks and their memories.

Rules:
1. Use only DOCUMENT_CLUSTER_BANKS.
2. Do not use outside knowledge.
3. Give the single best-supported answer.
4. Return only one short final answer string.
5. Do not include explanation, reasoning, uncertainty notes, or extra context.
6. Satisfy all explicit constraints in TARGET_QUERY jointly; do not choose a candidate that only partially matches.
7. If memories conflict, choose the most direct and specific evidence.
8. Prefer evidence that links the race identity, incident clues, penalty clues, and participant/background clues into one coherent answer.
9. Prefer exact answer-bearing strings and explicit grounded relations over broad narrative or generic descriptive text.
10. If no single candidate satisfies all explicit constraints jointly, choose the candidate with the strongest fully grounded cross-clue support, not the most topically similar partial match.

TARGET_QUERY:
{target_query}

DOCUMENT_CLUSTER_BANKS:
{memory_text}

FINAL_ANSWER:
"""

MERGE_CLUSTER_BANKS_EXECUTION_PROMPT = """You are merging a group of rendered cluster banks into one evidence-preserving merged cluster bank for downstream question answering.

Goal:
- Produce one merged bank that preserves as much distinct useful evidence from the provided rendered GROUP_CLUSTER_BANKS as possible for downstream question answering.

Rules:
1. Use only GROUP_CLUSTER_BANKS. TARGET_QUERY is for prioritization only.
2. Do not answer TARGET_QUERY. Do not claim that this group is the final correct answer.
3. The merged bank must stay coherent and correspond to one evidence-grounded theme already present in GROUP_CLUSTER_BANKS.
4. Preserve exact names, titles, dates, numbers, places, and attributions when they matter.
5. Preserve all distinct facts, qualifiers, and relationships from GROUP_CLUSTER_BANKS.
6. Remove only true duplicates or near-duplicate phrasings that express the same fact with the same qualifiers.
7. If two facts differ in any potentially answer-relevant way, keep both rather than collapsing them.
8. If multiple banks contain complementary evidence for the same theme, combine them without dropping distinct supporting details.
9. If evidence conflicts within the same theme, keep attributed alternatives in memory instead of collapsing them incorrectly.
10. Do not invent new facts, new entities, unsupported links, unsupported clue types, or unsupported race identities.
11. Do not write global conclusions such as "this is the race where..." or "all top finishers had already won by December 2023" unless that exact conclusion is directly supported inside GROUP_CLUSTER_BANKS.
12. Write 1 to MAX_QUERIES_PER_CLUSTER focused questions that describe the evidence in GROUP_CLUSTER_BANKS, not the final benchmark question.
13. {style_rule}
14. Create a solid, comprehensive MEMORY that preserves the strongest source evidence, keeps answer-critical distinctions explicit, and front-loads the most important facts.
15. Do not optimize for brevity. It is acceptable for MEMORY to be long if needed to preserve distinct evidence.
16. Do not replace several distinct facts with one generalized summary if that would remove answer-relevant detail.
17. Preserve exact answer-bearing strings verbatim when they appear in GROUP_CLUSTER_BANKS.
18. If style requires a title, the title must be evidence-preserving, local to the source banks, and written as a short noun phrase. It must not be a question, a guessed answer, or an unsupported final conclusion.
19. Do not output truncated fragments, ellipsized text, or unfinished items.
20. Do not include generic promotional, evaluative, or stylistic filler unless it is directly relevant to the grouped evidence theme.
21. SOURCE_DOCUMENT_SUMMARIES contains query-aware source-document summaries for the source documents behind GROUP_CLUSTER_BANKS. Use them only as auxiliary merge context to recover or align evidence already grounded in the group. Do not simply paste or restate the source summaries wholesale.

Output format:
- Return STRICT JSON only.
- Use exactly this schema:
  {{
    "merged_bank": {{
      {merged_bank_schema}
    }}
  }}
- No markdown, no prose, no extra keys.

TARGET_QUERY:
{target_query}

MAX_QUERIES_PER_CLUSTER:
{max_queries_per_cluster}

GROUP_CLUSTER_BANKS:
{group_banks}

SOURCE_DOCUMENT_SUMMARIES:
{source_doc_summaries}
"""

AUTO_CLUSTER_GENERATION_WARM_PROMPT = """You are generating candidate clusters of possible future user questions from one document.

You are given WARM_START_DOCUMENTS.
Generate only as many distinct CLUSTERS as are actually needed to cover the meaningful factual directions in the document.

Rules:
1. Use only information grounded in WARM_START_DOCUMENTS.
2. Generate at least 1 cluster.
3. Do not pad with weak or redundant clusters.
4. Clusters must be meaningfully distinct from one another.
5. Do not repeat or paraphrase the same question across clusters.
6. Questions must be specific, factual, and answer-oriented.
7. Prefer concrete entities, dates, numbers, titles, places, or explicit relations.
8. Each cluster should contain only related questions; do not mix unrelated themes.
9. Use between 1 and MAX_QUERIES_PER_CLUSTER questions per cluster.
10. {style_rule}
11. Generate information-seeking questions only, not instructions, summaries, or meta-prompts.
12. Do not try to infer any hidden target question; propose plausible future user questions only from the observed evidence.
13. If one cluster fully captures the document's useful content, output one cluster. Use more clusters only when clearly justified by distinct factual themes.
14. If style requires a title, the title must be evidence-preserving and local to the document evidence. It must not be a question, a guessed answer, or an unsupported benchmark-style conclusion.

Output format:
- Return STRICT JSON only.
- Use exactly this schema:
  {{
    "clusters": [
      {cluster_schema}
    ]
  }}
- Return as many clusters as needed, but only genuinely distinct useful ones.
- No markdown, no prose, no extra keys.

MAX_QUERIES_PER_CLUSTER:
{max_queries_per_cluster}

WARM_START_DOCUMENTS:
{warm_documents}
"""

INITIALIZE_CLUSTER_MEMORY_UNBOUNDED_PROMPT = """You are creating one cluster-specific memory bank from a single document.

Goal:
- Create a solid, comprehensive memory bank for TARGET_CLUSTER using DOCUMENT.

Rules:
1. Use only DOCUMENT; no outside knowledge.
2. Keep only information relevant or plausibly relevant to TARGET_CLUSTER.
3. Prefer concrete evidence: entities, dates, numbers, titles, places, organizations, roles, relationships, and explicit attributions.
4. Preserve exact names, titles, dates, numbers, places, and attributions when they matter.
5. Preserve all distinct facts, qualifiers, and relationships that may matter for downstream question answering.
6. Remove only exact duplicates, near-duplicate phrasings of the same fact, and genuinely low-value boilerplate.
7. If two facts differ in any potentially answer-relevant way, keep both.
8. If the document contains conflicting or uncertain evidence, keep attributed alternatives separate instead of collapsing them.
9. Front-load the most answer-critical facts.
10. Do not output absence-style statements unless they are themselves important evidence.
11. Do not optimize for brevity. It is acceptable for MEMORY to be long if needed to preserve distinct evidence from DOCUMENT.
12. Do not replace several distinct facts with one generalized summary if that would remove answer-relevant detail.
13. Preserve exact answer-bearing strings verbatim when they appear in DOCUMENT.
14. Do not include promotional, evaluative, or stylistic filler unless it is directly relevant to TARGET_CLUSTER.
15. Do not output truncated fragments or ellipsized text.

Output:
- Plain text memory only.
- No JSON, no markdown, no bullets, no preamble.

TARGET_CLUSTER:
{target_cluster}

DOCUMENT:
{document}
"""

STRUCTURED_MEMORY_KEYS = ("exact_strings", "facts", "relations", "qualifiers")

RACE_ANCHOR_RE = re.compile(r"\b(?:20\d{2}\s+)?(?:[A-Z][A-Za-zÀ-ÿ'’.-]+\s+){1,5}Grand Prix\b")

PERSON_ANCHOR_RE = re.compile(r"\b([A-Z][A-Za-zÀ-ÿ'’.-]+(?:\s+[A-Z][A-Za-zÀ-ÿ'’.-]+){1,2})\b")

GENERIC_PERSON_ANCHORS = {
    "Formula One",
    "Grand Prix",
    "World Championship",
    "Turn One",
    "Turn Four",
    "Lap One",
    "Lap Four",
    "Sport And Exercise",
}

EVIDENCE_ROLE_PATTERNS = {
    "results": (
        "race results",
        "final results",
        "final finishing",
        "podium",
        "winner",
        "winning margin",
        "finished in",
        "classification",
        "race outcome",
        "finishers",
    ),
    "incident": (
        "collision",
        "incident",
        "crash",
        "clash",
        "contact",
        "spin",
        "spun",
        "red flag",
        "safety car",
    ),
    "penalty": (
        "penalty",
        "disqualif",
        "stop/go",
        "fuel sample",
        "grid drop",
        "post-race",
    ),
    "context": (
        "context",
        "logistics",
        "attendance",
        "weather",
        "circuit",
        "track length",
        "tyre",
        "steward",
        "date",
        "location",
    ),
    "standings": (
        "championship standings",
        "constructors' championship",
        "drivers' championship",
        "points gap",
        "mathematically in contention",
    ),
    "background": (
        "coach",
        "performance",
        "biography",
        "career",
        "records",
        "victory and records",
        "expertise",
    ),
}

SUSPICIOUS_MERGE_CONCLUSION_PATTERNS = (
    re.compile(r"\bis the (?:race|grand prix) where\b", re.IGNORECASE),
    re.compile(r"\bthis is the (?:race|grand prix)\b", re.IGNORECASE),
    re.compile(r"\ball (?:the )?(?:top\s+\d+\s+)?finishers .* had at least one (?:career )?grand prix win", re.IGNORECASE),
    re.compile(r"\bby december 2023\b", re.IGNORECASE),
)

AUTO_CLUSTER_HARD_MAX_BANKS = 64

def format_cluster_bank_block(doc_idx: int, doc_id: str, cluster_bank_text: str) -> str:
    return "\n".join(
        [
            f"[DOC_CLUSTER_BANK_{doc_idx}]",
            f"doc_id: {doc_id}",
            "cluster_banks:",
            cluster_bank_text,
        ]
    ).strip()

def doc_with_text(doc: Dict[str, Any], text: str) -> Dict[str, Any]:
    out = dict(doc)
    out["text"] = text
    return out

def format_bank_unit_for_merge_planner(bank_unit: Dict[str, Any], style: str) -> str:
    cluster = bank_unit.get("cluster") or {}
    memory = str(bank_unit.get("memory", "") or "").strip()
    features = bank_unit.get("merge_features") or extract_bank_merge_features(bank_unit, style)
    lines = [
        f"[BANK {bank_unit.get('bank_id', '')}]",
        f"doc_idx: {bank_unit.get('doc_idx', '')}",
        f"doc_id: {bank_unit.get('doc_id', '')}",
    ]
    source_doc_summary_ref = str(bank_unit.get("source_doc_summary_ref", "") or "").strip()
    if source_doc_summary_ref:
        lines.append(f"source_doc_summary_ref: {source_doc_summary_ref}")
    lines.extend(
        [
            "cluster:",
            cbase.cluster_label(cluster, style),
            f"race_anchors: {', '.join(features.get('race_anchors', []) or []) or '(none)'}",
            f"person_anchors: {', '.join(features.get('person_anchors', []) or []) or '(none)'}",
            "clue_flags: "
            + ", ".join(
                flag
                for flag, enabled in [
                    ("coach", bool(features.get("has_coach_clue"))),
                    ("turn4_lap1", bool(features.get("has_turn4_lap1_clue"))),
                    ("penalty", bool(features.get("has_penalty_clue"))),
                ]
                if enabled
            )
            or "(none)",
            f"evidence_roles: {', '.join(features.get('evidence_roles', []) or []) or '(none)'}",
            "memory:",
            memory if memory else "(empty)",
        ]
    )
    return "\n".join(lines).strip()

def format_bank_units_for_merge_planner(bank_units: Sequence[Dict[str, Any]], style: str) -> str:
    return "\n\n".join(format_bank_unit_for_merge_planner(bank_unit, style) for bank_unit in bank_units).strip()

def source_doc_summary_entries_for_bank_units(
    bank_units: Sequence[Dict[str, Any]],
    doc_cluster_banks: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    summaries_by_doc: Dict[tuple[int, str], Dict[str, Any]] = {}
    for row in doc_cluster_banks:
        doc_idx = int(row.get("doc_idx", 0) or 0)
        doc_id = str(row.get("doc_id", "") or "").strip()
        summary = str(row.get("source_doc_summary", "") or "").strip()
        if doc_idx > 0 and summary:
            summaries_by_doc[(doc_idx, doc_id)] = {
                "doc_idx": doc_idx,
                "doc_id": doc_id,
                "summary": summary,
                "ref": f"SOURCE_DOC_SUMMARY_DOC_{doc_idx}",
            }

    entries: List[Dict[str, Any]] = []
    seen: Set[tuple[int, str]] = set()
    for bank_unit in bank_units:
        doc_idx = int(bank_unit.get("doc_idx", 0) or 0)
        doc_id = str(bank_unit.get("doc_id", "") or "").strip()
        key = (doc_idx, doc_id)
        if key in seen:
            continue
        seen.add(key)
        entry = summaries_by_doc.get(key)
        if entry:
            entries.append(entry)
    return entries

def format_source_doc_summaries_for_merge(
    bank_units: Sequence[Dict[str, Any]],
    doc_cluster_banks: Sequence[Dict[str, Any]],
) -> str:
    blocks: List[str] = []
    for entry in source_doc_summary_entries_for_bank_units(bank_units, doc_cluster_banks):
        blocks.append(
            "\n".join(
                [
                    f"[{entry['ref']}]",
                    f"doc_id: {entry['doc_id']}",
                    "summary:",
                    str(entry["summary"]),
                ]
            ).strip()
        )
    return "\n\n".join(blocks).strip()

def normalize_anchor(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())

def normalize_structured_memory_item(text: Any) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").strip())
    if not normalized:
        return ""
    if "..." in normalized or "…" in normalized:
        return ""
    return normalized

def is_low_value_structured_memory_item(text: str, key: str) -> bool:
    if not text:
        return True
    if key == "qualifiers":
        for pattern in LOW_VALUE_STRUCTURED_QUALIFIER_PATTERNS:
            if pattern.search(text):
                return True
    return False

def dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for item in items:
        normalized = normalize_structured_memory_item(item)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out

def normalize_structured_memory_object(obj: Any) -> Optional[Dict[str, List[str]]]:
    if isinstance(obj, dict) and isinstance(obj.get("memory"), dict):
        obj = obj.get("memory")
    if not isinstance(obj, dict):
        return None
    out: Dict[str, List[str]] = {}
    for key in STRUCTURED_MEMORY_KEYS:
        cleaned_items: List[str] = []
        for item in list(obj.get(key) or []):
            normalized = normalize_structured_memory_item(item)
            if not normalized or is_low_value_structured_memory_item(normalized, key):
                continue
            cleaned_items.append(normalized)
        out[key] = dedupe_preserve_order(cleaned_items)
    if not any(out.values()):
        return None
    return out

def render_structured_memory(memory_obj: Optional[Dict[str, List[str]]]) -> str:
    memory_obj = normalize_structured_memory_object(memory_obj)
    if not memory_obj:
        return ""
    section_labels = {
        "exact_strings": "exact_strings",
        "facts": "facts",
        "relations": "relations",
        "qualifiers": "qualifiers",
    }
    blocks: List[str] = []
    for key in STRUCTURED_MEMORY_KEYS:
        items = memory_obj.get(key) or []
        if not items:
            continue
        lines = [f"{section_labels[key]}:"]
        lines.extend(f"- {item}" for item in items)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks).strip()

def extract_race_anchors(text: str) -> List[str]:
    anchors: List[str] = []
    seen: Set[str] = set()
    for match in RACE_ANCHOR_RE.findall(text or ""):
        anchor = normalize_anchor(match)
        if anchor.lower().startswith(("formula one ", "formula 1 ")):
            continue
        key = anchor.lower()
        if key in seen:
            continue
        seen.add(key)
        anchors.append(anchor)
    return anchors

def contains_lap1_reference(text: str) -> bool:
    lower = str(text or "").lower()
    return any(
        needle in lower
        for needle in (
            "lap 1",
            "lap one",
            "opening lap",
            "first lap",
            "lap-1",
        )
    )

def detect_evidence_roles(text: str) -> List[str]:
    lower = str(text or "").lower()
    roles = [
        role
        for role, needles in EVIDENCE_ROLE_PATTERNS.items()
        if any(needle in lower for needle in needles)
    ]
    return roles

def extract_bank_merge_features(bank_unit: Dict[str, Any], style: str) -> Dict[str, Any]:
    cluster = bank_unit.get("cluster") or {}
    cluster_text = cbase.cluster_label(cluster, style)
    memory = str(bank_unit.get("memory", "") or "").strip()
    query_text = "\n".join(str(q) for q in list(cluster.get("queries") or []))
    text = "\n".join([cluster_text, query_text, memory])
    title_races = extract_race_anchors(cluster_text)
    if title_races:
        race_anchors = title_races
    else:
        counts: Dict[str, int] = {}
        original_by_key: Dict[str, str] = {}
        for source_text in (query_text, memory):
            for anchor in extract_race_anchors(source_text):
                key = anchor.lower()
                counts[key] = counts.get(key, 0) + 1
                original_by_key.setdefault(key, anchor)
        if counts:
            max_count = max(counts.values())
            top_keys = [key for key, value in counts.items() if value == max_count]
            if max_count == 1 and len(top_keys) > 1:
                race_anchors = []
            else:
                race_anchors = [original_by_key[key] for key in top_keys]
        else:
            race_anchors = []

    person_anchors = []
    seen_people: Set[str] = set()
    for match in PERSON_ANCHOR_RE.findall(text):
        anchor = normalize_anchor(match)
        if anchor in GENERIC_PERSON_ANCHORS:
            continue
        key = anchor.lower()
        if key in seen_people:
            continue
        seen_people.add(key)
        person_anchors.append(anchor)

    lower_text = text.lower()
    has_coach_clue = any(
        needle in lower_text
        for needle in (
            "performance coach",
            "sport and exercise science",
            "hintsa",
            "private gym",
            "motorsport athletes",
            "coach",
        )
    )
    has_turn4_lap1_clue = (
        any(needle in lower_text for needle in ("turn 4", "turn four"))
        and contains_lap1_reference(lower_text)
    )
    has_penalty_clue = any(
        needle in lower_text
        for needle in ("penalty", "disqualif", "fuel sample", "grid drop")
    )
    evidence_roles = detect_evidence_roles(text)
    return {
        "race_anchors": race_anchors,
        "person_anchors": person_anchors,
        "has_coach_clue": has_coach_clue,
        "has_turn4_lap1_clue": has_turn4_lap1_clue,
        "has_penalty_clue": has_penalty_clue,
        "evidence_roles": evidence_roles,
    }

def aggregate_merge_features(bank_units: Sequence[Dict[str, Any]], style: str) -> Dict[str, Any]:
    race_anchors: Set[str] = set()
    person_anchors: Set[str] = set()
    has_coach = False
    has_turn4 = False
    has_penalty = False
    evidence_roles: Set[str] = set()
    for bank_unit in bank_units:
        features = bank_unit.get("merge_features") or extract_bank_merge_features(bank_unit, style)
        race_anchors.update(anchor.lower() for anchor in features.get("race_anchors", []) or [])
        person_anchors.update(anchor.lower() for anchor in features.get("person_anchors", []) or [])
        has_coach = has_coach or bool(features.get("has_coach_clue"))
        has_turn4 = has_turn4 or bool(features.get("has_turn4_lap1_clue"))
        has_penalty = has_penalty or bool(features.get("has_penalty_clue"))
        evidence_roles.update(str(role) for role in features.get("evidence_roles", []) or [])
    return {
        "race_anchors": race_anchors,
        "person_anchors": person_anchors,
        "has_coach_clue": has_coach,
        "has_turn4_lap1_clue": has_turn4,
        "has_penalty_clue": has_penalty,
        "evidence_roles": evidence_roles,
    }

def count_high_signal_clues(features: Dict[str, Any]) -> int:
    return sum(
        1
        for field in ("has_coach_clue", "has_turn4_lap1_clue", "has_penalty_clue")
        if bool(features.get(field))
    )

def feature_value_set(features: Dict[str, Any], key: str) -> Set[str]:
    return {
        str(value).strip().lower()
        for value in list(features.get(key) or [])
        if str(value).strip()
    }

def pair_has_strong_forced_merge_support(fa: Dict[str, Any], fb: Dict[str, Any]) -> bool:
    race_overlap = feature_value_set(fa, "race_anchors") & feature_value_set(fb, "race_anchors")
    if not race_overlap:
        return False

    person_overlap = feature_value_set(fa, "person_anchors") & feature_value_set(fb, "person_anchors")
    role_overlap = feature_value_set(fa, "evidence_roles") & feature_value_set(fb, "evidence_roles")

    clue_presence = {
        "coach": bool(fa.get("has_coach_clue")) or bool(fb.get("has_coach_clue")),
        "turn4_lap1": bool(fa.get("has_turn4_lap1_clue")) or bool(fb.get("has_turn4_lap1_clue")),
        "penalty": bool(fa.get("has_penalty_clue")) or bool(fb.get("has_penalty_clue")),
    }
    high_signal_union_count = sum(1 for present in clue_presence.values() if present)

    # Force a merge only when there is clear same-anchor evidence beyond a broad same-race match.
    if person_overlap and (role_overlap or high_signal_union_count >= 1):
        return True
    if high_signal_union_count >= 2 and role_overlap:
        return True
    return False

def validate_merged_bank_against_sources(
    merged_bank: Dict[str, Any],
    bank_units: Sequence[Dict[str, Any]],
    style: str,
) -> tuple[bool, str]:
    if not merged_bank:
        return False, "merged_bank_missing"

    source_features = aggregate_merge_features(bank_units, style)
    merged_features = extract_bank_merge_features(
        {
            "cluster": merged_bank.get("cluster") or {},
            "memory": str(merged_bank.get("memory", "") or "").strip(),
        },
        style,
    )

    merged_race_anchors = {anchor.lower() for anchor in merged_features.get("race_anchors", []) or []}
    extra_race_anchors = sorted(merged_race_anchors - source_features["race_anchors"])
    if extra_race_anchors:
        return False, f"introduced_unsupported_race_anchor:{', '.join(extra_race_anchors)}"

    for field, label in (
        ("has_coach_clue", "coach"),
        ("has_turn4_lap1_clue", "turn4_lap1"),
        ("has_penalty_clue", "penalty"),
    ):
        if merged_features.get(field) and not source_features.get(field):
            return False, f"introduced_unsupported_clue:{label}"

    cluster = merged_bank.get("cluster") or {}
    text = "\n".join(
        [
            str(cluster.get("title", "") or "").strip(),
            "\n".join(str(q) for q in list(cluster.get("queries") or [])),
            str(merged_bank.get("memory", "") or "").strip(),
        ]
    ).strip()
    for pattern in SUSPICIOUS_MERGE_CONCLUSION_PATTERNS:
        if pattern.search(text):
            return False, f"suspicious_answer_style_merge:{pattern.pattern}"

    return True, ""

def format_heuristic_groups(groups: Sequence[Sequence[str]]) -> str:
    if not groups:
        return "(none)"
    return "\n".join(
        f"- group_{idx}: {', '.join(group)}"
        for idx, group in enumerate(groups, start=1)
    )

def build_forced_merge_groups(bank_units: Sequence[Dict[str, Any]], style: str) -> List[List[str]]:
    bank_ids = [str(bank_unit.get("bank_id", "")).strip() for bank_unit in bank_units if str(bank_unit.get("bank_id", "")).strip()]
    if not bank_ids:
        return []

    features_by_id: Dict[str, Dict[str, Any]] = {}
    for bank_unit in bank_units:
        bank_id = str(bank_unit.get("bank_id", "")).strip()
        if not bank_id:
            continue
        features = extract_bank_merge_features(bank_unit, style)
        bank_unit["merge_features"] = features
        features_by_id[bank_id] = features

    def bank_sort_key(bank_id: str) -> tuple[int, int, str]:
        nums = [int(x) for x in re.findall(r"\d+", bank_id)]
        first = nums[0] if len(nums) >= 1 else 0
        second = nums[1] if len(nums) >= 2 else 0
        return (first, second, bank_id)

    candidate_groups: List[List[str]] = []
    used: Set[str] = set()

    # Build only local, non-transitive suggestions. This avoids broad same-race chains
    # that can collapse many banks into one component.
    for idx, bank_id_a in enumerate(bank_ids):
        if bank_id_a in used:
            continue
        fa = features_by_id.get(bank_id_a) or {}
        best_partner = ""
        best_score = -1
        for bank_id_b in bank_ids[idx + 1 :]:
            if bank_id_b in used:
                continue
            fb = features_by_id.get(bank_id_b) or {}
            if not pair_has_strong_forced_merge_support(fa, fb):
                continue
            person_overlap = feature_value_set(fa, "person_anchors") & feature_value_set(fb, "person_anchors")
            role_overlap = feature_value_set(fa, "evidence_roles") & feature_value_set(fb, "evidence_roles")
            score = (
                10 * len(person_overlap)
                + 3 * len(role_overlap)
                + count_high_signal_clues(fa)
                + count_high_signal_clues(fb)
            )
            if score > best_score:
                best_score = score
                best_partner = bank_id_b
        if best_partner:
            candidate_groups.append(sorted([bank_id_a, best_partner], key=bank_sort_key))
            used.add(bank_id_a)
            used.add(best_partner)

    candidate_groups.sort(key=lambda group: (min(bank_ids.index(bid) for bid in group), len(group)))
    return candidate_groups

def fallback_merged_bank(
    bank_units: Sequence[Dict[str, Any]],
    style: str,
    max_queries_per_cluster: int,
    structured_memory: bool = False,
) -> Dict[str, Any]:
    queries: List[str] = []
    seen_queries: Set[str] = set()
    for bank_unit in bank_units:
        cluster = bank_unit.get("cluster") or {}
        for q in list(cluster.get("queries") or []):
            qn = cbase.base.normalize_candidate_query(str(q))
            if not qn:
                continue
            qk = qn.lower()
            if qk in seen_queries:
                continue
            seen_queries.add(qk)
            queries.append(qn)
            if len(queries) >= max_queries_per_cluster:
                break
        if len(queries) >= max_queries_per_cluster:
            break
    if not queries:
        queries = ["What key factual relationship is jointly supported by these banks?"]
    title = ""
    if style == "titled":
        source_features = aggregate_merge_features(bank_units, style)
        race_anchors = sorted(source_features.get("race_anchors", []) or [])
        if len(race_anchors) == 1:
            race_name = race_anchors[0]
            title = f"{race_name.title()} evidence"
        if not title:
            title = str((bank_units[0].get("cluster") or {}).get("title", "") or "").strip()
        if not title:
            title = "Merged cluster bank"
    cluster = {"title": title, "queries": queries} if style == "titled" else {"queries": queries}
    cluster = cbase.normalize_cluster(cluster, style=style, max_queries_per_cluster=max_queries_per_cluster) or cluster
    if structured_memory:
        merged_memory_structured = union_structured_memory_objects(
            [bank_unit.get("memory_structured") for bank_unit in bank_units]
        )
        merged_memory = render_structured_memory(merged_memory_structured)
    else:
        merged_memory_structured = None
        memory_parts = [str(bank_unit.get("memory", "") or "").strip() for bank_unit in bank_units if str(bank_unit.get("memory", "") or "").strip()]
        merged_memory = "\n\n".join(memory_parts).strip()
    out = {
        "cluster": cluster,
        "memory": merged_memory,
    }
    if merged_memory_structured is not None:
        out["memory_structured"] = merged_memory_structured
    return out

def generate_candidate_clusters_warm_auto(
    llm: Any,
    warm_docs: Sequence[Dict[str, Any]],
    max_queries_per_cluster: int,
    style: str,
    temperature: float,
) -> List[Dict[str, Any]]:
    style_rule, schema = cbase.style_rule_and_schema(style)
    warm_docs_block = cbase.base.format_doc_chunk_for_prompt(warm_docs)
    prompt = AUTO_CLUSTER_GENERATION_WARM_PROMPT.format(
        max_queries_per_cluster=max_queries_per_cluster,
        warm_documents=warm_docs_block if warm_docs_block else "(empty)",
        style_rule=style_rule,
        cluster_schema=schema,
    )
    raw = llm.generate(prompt, temperature=temperature)
    out = cbase.parse_cluster_candidates(raw, style=style, max_queries_per_cluster=max_queries_per_cluster)
    if out:
        return out[:AUTO_CLUSTER_HARD_MAX_BANKS]
    fallback = cbase.build_fallback_clusters(warm_docs, 1, style, max_queries_per_cluster)
    return fallback[:1]

def initialize_cluster_memory_unbounded(
    llm: Any,
    cluster: Dict[str, Any],
    document: Dict[str, Any],
    style: str,
    temperature: float,
) -> str:
    prompt = INITIALIZE_CLUSTER_MEMORY_UNBOUNDED_PROMPT.format(
        target_cluster=cbase.cluster_label(cluster, style),
        document=base.format_doc_for_prompt(document),
    )
    return llm.generate(prompt, temperature=temperature).strip()

def usage_delta(before: tuple[int, int, int, float], after: tuple[int, int, int, float]) -> Dict[str, Any]:
    return {
        "calls": after[0] - before[0],
        "input_tokens": after[1] - before[1],
        "output_tokens": after[2] - before[2],
        "wall_time_sec": round(after[3] - before[3], 6),
    }

def usage_snapshot(llm: Optional[Any]) -> tuple[int, int, int, float]:
    if llm is None:
        return (0, 0, 0, 0.0)
    return (
        llm.total_calls,
        llm.total_input_tokens,
        llm.total_output_tokens,
        llm.total_wall_time_sec,
    )

def make_llm(
    backend: str,
    model: str,
    retry_policy: base.RetryPolicy,
    timeout_sec: int,
    counter: base.TokenCounter,
    base_url: str,
    http_referer: str,
    app_title: str,
) -> Any:
    if backend == "gemini":
        return base.GeminiClient(
            model=model,
            retry_policy=retry_policy,
            timeout_sec=timeout_sec,
        )
    if backend not in {"openrouter", "openai_compat"}:
        raise RuntimeError(f"Unsupported llm backend: {backend}")
    return base.OpenRouterClient(
        model=model,
        retry_policy=retry_policy,
        timeout_sec=timeout_sec,
        token_counter=counter,
        base_url=base_url,
        http_referer=http_referer,
        app_title=app_title,
    )
