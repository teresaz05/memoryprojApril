#!/usr/bin/env python3
"""Shared cluster-bank experiment core copied into the April package.

    The prose, structured, and docsummaryaux experiments all call into this file with different
    summary-mode settings. The implementation is intentionally preserved rather than aggressively
    refactored so we keep exact behavior while still presenting a cleaner top-level package layout."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from dotenv import load_dotenv

from april_version_code.common import metadata as row_metadata
from april_version_code.methods import stream_oracle_assisted_dynamic_cluster_bank as cbase
from april_version_code.methods import stream_oracle_memory as base


QUERY_AWARE_FILTERING_DOC_SUMMARY_PROMPT = """You are writing a query-focused summary of one document.

Goal:
- Maximize the chance of answering TARGET_QUERY correctly after all document summaries are concatenated.

Rules:
1. Use only DOCUMENT.
2. Extract only facts relevant or plausibly relevant to TARGET_QUERY.
3. Prioritize direct answer evidence, then concise supporting facts that help disambiguate entities, titles, dates, numbers, places, and relationships.
4. Preserve exact names, titles, numbers, dates, and attributions when they matter.
5. Omit boilerplate, broad background, and repetition.
6. If DOCUMENT is not useful for TARGET_QUERY, output exactly: none
7. Output plain text only. No markdown fences, no bullets, no preamble.
8. Keep the highest-value facts first.
9. Aim to stay within SOFT_SUMMARY_TARGET_TOKENS, but optimize for answer quality first.

TARGET_QUERY:
{target_query}

SOFT_SUMMARY_TARGET_TOKENS:
{summary_budget_tokens}

DOCUMENT:
{document}
"""


QUERY_AWARE_AUX_DOC_SUMMARY_PROMPT = """You are writing a query-focused source-document summary for downstream merging and question answering.

Goal:
- Produce an evidence-dense source summary that preserves the facts from DOCUMENT most useful for TARGET_QUERY and for later evidence merging.

Rules:
1. Use only DOCUMENT.
2. Front-load direct answer-bearing evidence, then concise supporting facts that help disambiguate entities, titles, dates, numbers, places, roles, and relationships.
3. Preserve exact answer-bearing strings, names, titles, numbers, dates, rankings, penalties, places, and attributions when they matter.
4. If TARGET_QUERY contains multiple constraints, preserve the best evidence for each constraint separately instead of smoothing them into one vague statement.
5. If DOCUMENT contains uncertainty, alternatives, or conflicting claims, preserve those distinctions explicitly rather than collapsing them.
6. Do not invent links, identities, causal claims, or final conclusions that are not directly grounded in DOCUMENT.
7. Do not rewrite the document as a generic topic overview. Keep only evidence that would help answer or merge for TARGET_QUERY.
8. Remove boilerplate, broad background, and repetition, but do not delete distinct facts just to make the summary shorter.
9. If little is relevant to TARGET_QUERY, still produce a short factual summary of the document instead of outputting "none".
10. Always output a non-empty summary.
11. Never output "none".
12. Output plain text only. No markdown fences, no bullets, no preamble.
13. Use dense factual sentences, not meta-commentary.
14. Keep the highest-value facts first.
15. Aim to stay within SOFT_SUMMARY_TARGET_TOKENS when possible, but optimize for answer quality first.

TARGET_QUERY:
{target_query}

SOFT_SUMMARY_TARGET_TOKENS:
{summary_budget_tokens}

DOCUMENT:
{document}
"""


QUERY_AWARE_AUX_DOC_SUMMARY_UNBOUNDED_PROMPT = """You are writing a query-focused source-document summary for downstream merging and question answering.

Goal:
- Produce an evidence-dense source summary that preserves the facts from DOCUMENT most useful for TARGET_QUERY and for later evidence merging.

Rules:
1. Use only DOCUMENT.
2. Front-load direct answer-bearing evidence, then concise supporting facts that help disambiguate entities, titles, dates, numbers, places, roles, and relationships.
3. Preserve exact answer-bearing strings, names, titles, numbers, dates, rankings, penalties, places, and attributions when they matter.
4. If TARGET_QUERY contains multiple constraints, preserve the best evidence for each constraint separately instead of smoothing them into one vague statement.
5. If DOCUMENT contains uncertainty, alternatives, or conflicting claims, preserve those distinctions explicitly rather than collapsing them.
6. Do not invent links, identities, causal claims, or final conclusions that are not directly grounded in DOCUMENT.
7. Do not rewrite the document as a generic topic overview. Keep only evidence that would help answer or merge for TARGET_QUERY.
8. Remove boilerplate, broad background, and repetition, but do not delete distinct facts just to make the summary shorter.
9. If little is relevant to TARGET_QUERY, still produce a short factual summary of the document instead of outputting "none".
10. Always output a non-empty summary.
11. Never output "none".
12. Output plain text only. No markdown fences, no bullets, no preamble.
13. Use dense factual sentences, not meta-commentary.
14. Keep the highest-value facts first.
15. Let the evidence determine length. Be concise where possible, but do not omit distinct answer-relevant facts.

TARGET_QUERY:
{target_query}

DOCUMENT:
{document}
"""


GENERIC_ALL_DOCS_SUMMARY_PROMPT = """You are writing a concise factual summary of one document for downstream question answering.

Goal:
- Produce a compact, evidence-rich summary that preserves the document's most important factual content.

Rules:
1. Use only DOCUMENT.
2. Do not assume a target query.
3. Prioritize concrete, answerable facts: names, titles, dates, numbers, places, organizations, roles, relationships, and explicit attributions.
4. Preserve exact names, titles, numbers, dates, and attributions when they matter.
5. Omit boilerplate, stylistic filler, and repetition.
6. Always output a non-empty summary.
7. Never output "none".
8. Output plain text only. No markdown fences, no bullets, no preamble.
9. Keep the highest-value facts first.
10. Aim to stay within SOFT_SUMMARY_TARGET_TOKENS, but optimize for factual utility first.

SOFT_SUMMARY_TARGET_TOKENS:
{summary_budget_tokens}

DOCUMENT:
{document}
"""


ANSWER_FROM_DOC_SUMMARIES_PROMPT = """You are answering a question using only concatenated query-focused document summaries.

Rules:
1. Use only DOCUMENT_SUMMARIES.
2. Do not use outside knowledge.
3. Give the single best-supported answer.
4. Return only one short final answer string.
5. Do not include explanation, reasoning, uncertainty notes, or extra context.
6. If summaries conflict, choose the most direct and specific evidence.

TARGET_QUERY:
{target_query}

DOCUMENT_SUMMARIES:
{memory_text}

FINAL_ANSWER:
"""


ANSWER_FROM_SUMMARIES_AND_CLUSTER_BANKS_PROMPT = """You are answering a question using both evidence summaries and cluster-bank memories.

Rules:
1. Use only EVIDENCE_SUMMARIES and CLUSTER_BANKS.
2. Do not use outside knowledge.
3. Give the single best-supported answer.
4. Return only one short final answer string.
5. Do not include explanation, reasoning, uncertainty notes, or extra context.
6. Prefer answers supported consistently by both representations.
7. If the representations differ, trust the most direct and specific grounded evidence.

TARGET_QUERY:
{target_query}

EVIDENCE_SUMMARIES:
{summary_text}

CLUSTER_BANKS:
{bank_text}

FINAL_ANSWER:
"""


ANSWER_FROM_DOC_CLUSTER_BANKS_PROMPT = """You are answering a question using only concatenated per-document cluster banks and their memories.

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


MERGE_CLUSTER_BANKS_PLAN_PROMPT = """You are planning merges across document-level cluster banks for downstream question answering.

Goal:
- Partition INPUT_CLUSTER_BANKS into groups that should be merged into stronger banks for TARGET_QUERY.

Rules:
1. Use only TARGET_QUERY and INPUT_CLUSTER_BANKS.
2. Every bank_id must appear in exactly one group.
3. Singleton groups are allowed and preferred when a bank should stay separate.
4. Merge banks only if they describe the same entity, event, fact cluster, or strongly complementary answer-bearing theme.
5. Do not merge banks just because they are from the same broad topic.
6. If merging would create a vague or overly broad bank, keep those banks separate.
7. If banks contain conflicting evidence about the same target theme, you may still group them so the next step can preserve attributed alternatives.
8. Prefer merges that improve answer utility for TARGET_QUERY through consolidation, disambiguation, or stronger evidence aggregation.
9. Output STRICT JSON only in this schema:
   {{
     "groups": [
       {{"bank_ids": ["BANK_ID_1", "BANK_ID_2"]}}
     ]
   }}
10. No prose, no markdown, no extra keys.
11. HEURISTIC_HIGH_PRIORITY_GROUPS lists bank groups that appear strongly complementary by shared race/entity/clue anchors. Keep those members together unless there is a clear evidence-grounded reason not to.
12. Preserve granularity by default. Do not collapse many banks into a few groups unless each resulting group is clearly one coherent anchor with directly complementary evidence.
13. If several banks share only a broad event/topic anchor but capture different sub-events, participants, or evidence roles, keep them separate unless the overlap is strong and specific.
14. SOURCE_DOCUMENT_SUMMARIES contains query-aware source-document summaries for the documents behind some banks. Use them only as auxiliary merge context. Do not merge banks just because their source summaries are broadly topically similar.

TARGET_QUERY:
{target_query}

INPUT_CLUSTER_BANKS:
{bank_units}

SOURCE_DOCUMENT_SUMMARIES:
{source_doc_summaries}

HEURISTIC_HIGH_PRIORITY_GROUPS:
{heuristic_groups}
"""


MERGE_CLUSTER_BANKS_EXECUTION_PROMPT = """You are merging a group of structured cluster banks into one evidence-preserving merged cluster bank for downstream question answering.

Goal:
- Produce one merged bank that preserves as much distinct useful evidence from GROUP_CLUSTER_BANKS as possible for downstream question answering.

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


MERGED_CLUSTER_SUMMARY_FOR_ANSWER_PROMPT = """You are writing one merged evidence summary for downstream question answering.

Goal:
- Produce one evidence-dense merged summary block for a single merged evidence cluster.

Rules:
1. Use only GROUP_CLUSTER_BANKS and SOURCE_DOCUMENT_SUMMARIES.
2. Do not answer TARGET_QUERY.
3. Preserve all distinct answer-relevant facts, qualifiers, names, titles, dates, numbers, places, organizations, roles, relationships, rankings, penalties, and explicit attributions.
4. Front-load the strongest answer-bearing evidence and the details most likely to disambiguate this cluster from nearby clusters.
5. Remove only true duplicates or near-duplicate phrasings of the same fact.
6. If evidence differs in any potentially answer-relevant way, keep the alternatives explicit rather than collapsing them.
7. Do not merge distinct people, events, placements, penalties, or timelines into one generalized statement unless GROUP_CLUSTER_BANKS clearly grounds that merge.
8. Use SOURCE_DOCUMENT_SUMMARIES as the primary material, and GROUP_CLUSTER_BANKS to recover exact strings, distinctions, and grounded structure that must not be lost.
9. Do not introduce unsupported cross-cluster conclusions, benchmark-style guesses, or broad topical summaries.
10. Do not replace several concrete facts with one abstract paraphrase if that would weaken answerability.
11. Output plain text only. No markdown fences, no bullets, no preamble.
12. Use dense factual sentences, not meta-commentary.
13. Let the evidence determine length. Be concise where possible, but do not omit distinct answer-relevant facts.

TARGET_QUERY:
{target_query}

GROUP_CLUSTER_BANKS:
{group_banks}

SOURCE_DOCUMENT_SUMMARIES:
{source_doc_summaries}

MERGED_CLUSTER_SUMMARY:
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


INITIALIZE_STRUCTURED_CLUSTER_MEMORY_PROMPT = """You are creating one structured cluster-specific memory bank from a single document.

Goal:
- Create a structured evidence bank for TARGET_CLUSTER using DOCUMENT.

Rules:
1. Use only DOCUMENT; no outside knowledge.
2. Keep only information relevant or plausibly relevant to TARGET_CLUSTER.
3. Preserve all distinct facts, qualifiers, and relationships that may matter for downstream question answering.
4. Remove only exact duplicates, near-duplicate phrasings of the same fact, and genuinely low-value boilerplate.
5. If two facts differ in any potentially answer-relevant way, keep both.
6. Preserve exact answer-bearing strings verbatim when they appear in DOCUMENT.
7. Do not optimize for brevity. It is acceptable for the structured memory to be long if needed to preserve distinct evidence.
8. Do not replace several distinct facts with one generalized summary if that would remove answer-relevant detail.
9. If the document contains conflicting or uncertain evidence, keep attributed alternatives separate instead of collapsing them.
10. Every list item should be a short atomic entry, not a paragraph.
11. Do not output a prose summary paragraph.
12. Do not output truncated fragments, ellipsized text, or unfinished list items.
13. Do not include generic promotional, evaluative, or biographical filler unless it is directly relevant to TARGET_CLUSTER.
14. Keep the sections semantically distinct:
   - exact_strings: short verbatim strings worth preserving exactly, such as names, titles, dates, numbers, official labels, quoted phrases, or answer-bearing spans.
   - facts: atomic factual statements directly grounded in DOCUMENT.
   - relations: explicit subject-object relationships directly grounded in DOCUMENT.
   - qualifiers: short modifiers attached to facts or relations, such as year, role, rank, category, lap, turn, location, numeric value, or attribution. Do not use qualifiers for generic commentary.
15. Avoid redundant paraphrases across sections. Preserve all distinct evidence, but do not restate the same claim in multiple sections unless the section semantics are genuinely different.

Output format:
- Return STRICT JSON only.
- Use exactly this schema:
  {{
    "memory": {{
      "exact_strings": ["..."],
      "facts": ["..."],
      "relations": ["..."],
      "qualifiers": ["..."]
    }}
  }}
- Every field must be present, even if empty.
- No markdown, no prose, no extra keys.

TARGET_CLUSTER:
{target_cluster}

DOCUMENT:
{document}
"""


MERGE_STRUCTURED_CLUSTER_BANKS_EXECUTION_PROMPT = """You are merging a group of structured cluster banks into one evidence-preserving merged cluster bank for downstream question answering.

Goal:
- Produce one merged bank that preserves as much distinct useful evidence from GROUP_CLUSTER_BANKS as possible.

Rules:
1. Use only GROUP_CLUSTER_BANKS. TARGET_QUERY is for prioritization only.
2. Do not answer TARGET_QUERY. Do not claim that this group is the final correct answer.
3. The merged bank must stay coherent and correspond to one evidence-grounded theme already present in GROUP_CLUSTER_BANKS.
4. Preserve exact answer-bearing strings verbatim when they appear in GROUP_CLUSTER_BANKS.
5. Preserve all distinct facts, qualifiers, and relationships from GROUP_CLUSTER_BANKS.
6. Remove only true duplicates or near-duplicate phrasings that express the same fact with the same qualifiers.
7. If two facts differ in any potentially answer-relevant way, keep both rather than collapsing them.
8. If multiple banks contain complementary evidence for the same theme, combine them without dropping distinct supporting details.
9. If evidence conflicts within the same theme, keep attributed alternatives separate instead of collapsing them incorrectly.
10. Do not invent new facts, new entities, unsupported links, unsupported clue types, or unsupported race identities.
11. Do not write global conclusions such as "this is the race where..." unless that exact conclusion is directly supported inside GROUP_CLUSTER_BANKS.
12. Write 1 to MAX_QUERIES_PER_CLUSTER focused questions that describe the evidence in GROUP_CLUSTER_BANKS, not the final benchmark question.
13. {style_rule}
14. Do not output a prose summary paragraph in MEMORY. MEMORY must remain structured.
15. If style requires a title, the title must be evidence-preserving, local to the source banks, and written as a short noun phrase. It must not be a question, a guessed answer, or an unsupported final conclusion.
16. Do not output truncated fragments, ellipsized text, or unfinished items anywhere in MEMORY.
17. Keep structured sections semantically strict: exact_strings for verbatim spans, facts for atomic statements, relations for explicit grounded links, qualifiers for short modifiers only.
18. Avoid generic promotional or evaluative filler unless it is directly relevant to the grouped evidence theme.

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
"""


AUTO_CLUSTER_HARD_MAX_BANKS = 64
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

CLUSTER_BANK_SUMMARY_MODES = {
    "per_doc_full_cluster_banks",
    "per_doc_full_cluster_banks_merge2",
    "per_doc_full_cluster_banks_merge2_docsummaryaux",
    "per_doc_structured_cluster_banks",
    "per_doc_structured_cluster_banks_merge2",
}

MERGE2_SUMMARY_MODES = {
    "per_doc_full_cluster_banks_merge2",
    "per_doc_full_cluster_banks_merge2_docsummaryaux",
    "per_doc_structured_cluster_banks_merge2",
}

STRUCTURED_CLUSTER_BANK_SUMMARY_MODES = {
    "per_doc_structured_cluster_banks",
    "per_doc_structured_cluster_banks_merge2",
}

DOCSUMMARYAUX_SUMMARY_MODE = "per_doc_full_cluster_banks_merge2_docsummaryaux"


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def is_completed_row(row: Dict[str, Any], skip_answer: bool) -> bool:
    if str(row.get("runtime_error", "")).strip():
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
    for row in iter_jsonl(path):
        qid = str(row.get("question_id", "")).strip()
        if qid and is_completed_row(row, skip_answer=skip_answer):
            done.add(qid)
    return done


def format_summary_block(doc_idx: int, doc_id: str, summary_text: str) -> str:
    return "\n".join(
        [
            f"[DOC_SUMMARY_{doc_idx}]",
            f"doc_id: {doc_id}",
            "summary:",
            summary_text,
        ]
    ).strip()


def format_cluster_bank_block(doc_idx: int, doc_id: str, cluster_bank_text: str) -> str:
    return "\n".join(
        [
            f"[DOC_CLUSTER_BANK_{doc_idx}]",
            f"doc_id: {doc_id}",
            "cluster_banks:",
            cluster_bank_text,
        ]
    ).strip()


def format_merged_summary_block(
    merged_bank_id: str,
    cluster: Dict[str, Any],
    style: str,
    summary_text: str,
) -> str:
    return "\n".join(
        [
            f"[MERGED_CLUSTER_SUMMARY_{merged_bank_id}]",
            f"merged_bank_id: {merged_bank_id}",
            "cluster:",
            cbase.cluster_label(cluster, style),
            "summary:",
            summary_text,
        ]
    ).strip()


def format_merged_cluster_bank_block(
    merged_bank_id: str,
    cluster_bank_text: str,
) -> str:
    return "\n".join(
        [
            f"[MERGED_CLUSTER_BANK_{merged_bank_id}]",
            f"merged_bank_id: {merged_bank_id}",
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


def build_docsummaryaux_layer1_memory(doc_cluster_banks: Sequence[Dict[str, Any]]) -> str:
    summary_blocks: List[str] = []
    bank_blocks: List[str] = []
    for row in doc_cluster_banks:
        summary_text = str(row.get("source_doc_summary", "") or "").strip()
        doc_idx = int(row.get("doc_idx", 0) or 0)
        doc_id = str(row.get("doc_id", "") or "").strip()
        cluster_bank_text = str(row.get("cluster_bank_text", "") or "").strip()
        if summary_text:
            summary_blocks.append(format_summary_block(doc_idx, doc_id, summary_text))
        if cluster_bank_text:
            bank_blocks.append(format_cluster_bank_block(doc_idx, doc_id, cluster_bank_text))
    sections: List[str] = []
    if summary_blocks:
        sections.append("DOCUMENT_SUMMARIES:\n" + "\n\n".join(summary_blocks))
    if bank_blocks:
        sections.append("DOCUMENT_CLUSTER_BANKS:\n" + "\n\n".join(bank_blocks))
    return "\n\n".join(sections).strip()


def parse_merge_groups(raw: str, valid_bank_ids: Sequence[str]) -> List[List[str]]:
    valid_set = set(valid_bank_ids)
    txt = (raw or "").strip()
    if txt.startswith("```"):
        txt = txt.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    obj: Any = None
    try:
        obj = json.loads(txt)
    except Exception:
        start = txt.find("{")
        end = txt.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(txt[start : end + 1])
            except Exception:
                obj = None
    groups_raw: List[Any] = []
    if isinstance(obj, dict):
        groups_raw = obj.get("groups") or []
    elif isinstance(obj, list):
        groups_raw = obj

    groups: List[List[str]] = []
    seen: Set[str] = set()
    for item in groups_raw:
        bank_ids_raw: List[Any] = []
        if isinstance(item, dict):
            bank_ids_raw = item.get("bank_ids") or []
        elif isinstance(item, list):
            bank_ids_raw = item
        cleaned: List[str] = []
        for bank_id in bank_ids_raw:
            bid = str(bank_id or "").strip()
            if not bid or bid not in valid_set or bid in seen:
                continue
            cleaned.append(bid)
            seen.add(bid)
        if cleaned:
            groups.append(cleaned)

    for bank_id in valid_bank_ids:
        if bank_id not in seen:
            groups.append([bank_id])
            seen.add(bank_id)
    return groups


def parse_merged_bank_json(
    raw: str,
    style: str,
    max_queries_per_cluster: int,
    structured_memory: bool = False,
) -> Optional[Dict[str, Any]]:
    txt = (raw or "").strip()
    if txt.startswith("```"):
        txt = txt.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    obj: Any = None
    try:
        obj = json.loads(txt)
    except Exception:
        start = txt.find("{")
        end = txt.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(txt[start : end + 1])
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        return None
    bank_obj = obj.get("merged_bank") if isinstance(obj.get("merged_bank"), dict) else obj
    if not isinstance(bank_obj, dict):
        return None
    if structured_memory:
        memory_structured = normalize_structured_memory_object(bank_obj.get("memory"))
        memory = render_structured_memory(memory_structured)
    else:
        memory_structured = None
        memory = str(bank_obj.get("memory", "") or "").strip()
    cluster = cbase.normalize_cluster(bank_obj, style=style, max_queries_per_cluster=max_queries_per_cluster)
    if not cluster or not memory:
        return None
    out = {
        "cluster": cluster,
        "memory": memory,
    }
    if memory_structured is not None:
        out["memory_structured"] = memory_structured
    return out


def normalize_anchor(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


LOW_VALUE_STRUCTURED_QUALIFIER_PATTERNS = (
    re.compile(r"^the race was a day of\b", re.IGNORECASE),
    re.compile(r"\bincredible day\b", re.IGNORECASE),
)


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


def parse_structured_memory_json(raw: str) -> Optional[Dict[str, List[str]]]:
    txt = (raw or "").strip()
    if txt.startswith("```"):
        txt = txt.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    obj: Any = None
    try:
        obj = json.loads(txt)
    except Exception:
        start = txt.find("{")
        end = txt.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(txt[start : end + 1])
            except Exception:
                obj = None
    return normalize_structured_memory_object(obj)


def fallback_structured_memory_for_document(doc_text: str) -> Optional[Dict[str, List[str]]]:
    return normalize_structured_memory_object(
        {
            "exact_strings": [],
            "facts": [doc_text],
            "relations": [],
            "qualifiers": [],
        }
    )


def generate_query_aware_aux_doc_summary(
    llm: Optional[Any],
    counter: base.TokenCounter,
    question: str,
    doc_text: str,
    summary_budget_tokens: int,
    temperature: float,
    dry_run: bool,
) -> tuple[str, str, bool]:
    if summary_budget_tokens > 0:
        fallback_summary = counter.truncate(
            doc_text,
            max_tokens=summary_budget_tokens,
            strategy="head",
        ).strip()
    else:
        fallback_summary = doc_text.strip()
    if not fallback_summary:
        return "", "", False
    if dry_run:
        return fallback_summary, "", False
    try:
        if llm is None:
            raise RuntimeError("Summary model is not initialized.")
        if summary_budget_tokens > 0:
            prompt = QUERY_AWARE_AUX_DOC_SUMMARY_PROMPT.format(
                target_query=question,
                summary_budget_tokens=summary_budget_tokens,
                document=doc_text,
            )
        else:
            prompt = QUERY_AWARE_AUX_DOC_SUMMARY_UNBOUNDED_PROMPT.format(
                target_query=question,
                document=doc_text,
            )
        raw_summary = llm.generate(
            prompt,
            temperature=temperature,
        ).strip()
        if base.is_none_snippet(raw_summary):
            raw_summary = ""
        if summary_budget_tokens > 0:
            summary_text = counter.truncate(
                raw_summary,
                max_tokens=summary_budget_tokens,
                strategy="head",
            ).strip()
        else:
            summary_text = raw_summary.strip()
        if summary_text:
            return summary_text, "", False
        return fallback_summary, "", True
    except Exception as exc:  # noqa: BLE001
        return fallback_summary, str(exc), True


def generate_merged_cluster_summary_for_answer(
    llm: Optional[Any],
    question: str,
    group_bank_units: Sequence[Dict[str, Any]],
    doc_cluster_banks: Sequence[Dict[str, Any]],
    style: str,
    temperature: float,
    dry_run: bool,
) -> tuple[str, str, bool]:
    source_entries = source_doc_summary_entries_for_bank_units(group_bank_units, doc_cluster_banks)
    fallback_summary = "\n\n".join(str(entry.get("summary", "") or "").strip() for entry in source_entries if str(entry.get("summary", "") or "").strip()).strip()
    if not fallback_summary:
        fallback_summary = "\n\n".join(
            str(bank_unit.get("memory", "") or "").strip()
            for bank_unit in group_bank_units
            if str(bank_unit.get("memory", "") or "").strip()
        ).strip()
    if not fallback_summary:
        return "", "", False
    if len(source_entries) == 1:
        return str(source_entries[0]["summary"]).strip(), "", False
    if dry_run:
        return fallback_summary, "", False
    try:
        if llm is None:
            raise RuntimeError("Summary model is not initialized.")
        prompt = MERGED_CLUSTER_SUMMARY_FOR_ANSWER_PROMPT.format(
            target_query=question,
            group_banks=format_bank_units_for_merge_planner(group_bank_units, style) or "(empty)",
            source_doc_summaries=format_source_doc_summaries_for_merge(group_bank_units, doc_cluster_banks) or "(none)",
        )
        merged_summary = llm.generate(
            prompt,
            temperature=temperature,
        ).strip()
        if merged_summary:
            return merged_summary, "", False
        return fallback_summary, "", True
    except Exception as exc:  # noqa: BLE001
        return fallback_summary, str(exc), True


def attach_docsummaryaux_merged_summaries(
    llm: Optional[Any],
    counter: base.TokenCounter,
    question: str,
    merged_cluster_banks: Sequence[Dict[str, Any]],
    bank_unit_by_id: Dict[str, Dict[str, Any]],
    doc_cluster_banks: Sequence[Dict[str, Any]],
    style: str,
    temperature: float,
    dry_run: bool,
) -> str:
    summary_blocks: List[str] = []
    bank_blocks: List[str] = []
    for merged_entry in merged_cluster_banks:
        group_bank_units = [
            bank_unit_by_id[bank_id]
            for bank_id in list(merged_entry.get("source_bank_ids") or [])
            if bank_id in bank_unit_by_id
        ]
        summary_text, summary_error, fallback_used = generate_merged_cluster_summary_for_answer(
            llm=llm,
            question=question,
            group_bank_units=group_bank_units,
            doc_cluster_banks=doc_cluster_banks,
            style=style,
            temperature=temperature,
            dry_run=dry_run,
        )
        merged_entry["merged_summary"] = summary_text
        merged_entry["merged_summary_tokens"] = counter.count(summary_text) if summary_text else 0
        merged_entry["merged_summary_error"] = summary_error
        merged_entry["merged_summary_fallback_used"] = bool(fallback_used)
        if summary_text:
            summary_blocks.append(
                format_merged_summary_block(
                    merged_bank_id=str(merged_entry.get("merged_bank_id", "") or ""),
                    cluster=dict(merged_entry.get("cluster") or {}),
                    style=style,
                    summary_text=summary_text,
                )
            )
        cluster_bank_text = str(merged_entry.get("cluster_bank_text", "") or "").strip()
        if cluster_bank_text:
            bank_blocks.append(
                format_merged_cluster_bank_block(
                    merged_bank_id=str(merged_entry.get("merged_bank_id", "") or ""),
                    cluster_bank_text=cluster_bank_text,
                )
            )
    sections: List[str] = []
    if summary_blocks:
        sections.append("MERGED_CLUSTER_SUMMARIES:\n" + "\n\n".join(summary_blocks))
    if bank_blocks:
        sections.append("MERGED_CLUSTER_BANKS:\n" + "\n\n".join(bank_blocks))
    return "\n\n".join(sections).strip()


def build_singleton_merged_outputs(
    bank_units_all: Sequence[Dict[str, Any]],
    style: str,
    counter: Any,
) -> tuple[List[List[str]], List[Dict[str, Any]], str]:
    merge_groups: List[List[str]] = []
    merged_cluster_banks: List[Dict[str, Any]] = []
    merged_cluster_bank_map: Dict[str, Dict[str, Any]] = {}
    merged_memory_bank_map: Dict[str, str] = {}
    merged_keys: List[str] = []

    for group_idx, bank_unit in enumerate(bank_units_all, start=1):
        bank_id = str(bank_unit.get("bank_id", "")).strip()
        if not bank_id:
            continue
        merge_groups.append([bank_id])
        merged_cluster = dict(bank_unit.get("cluster") or {})
        merged_memory = str(bank_unit.get("memory", "") or "").strip()
        merged_key = cbase.cluster_key(merged_cluster, style)
        if merged_key in merged_cluster_bank_map:
            merged_key = f"{merged_key}__group_{group_idx}"
        merged_keys.append(merged_key)
        merged_cluster_bank_map[merged_key] = merged_cluster
        merged_memory_bank_map[merged_key] = merged_memory
        merged_entry = {
            "merged_bank_id": f"merged_group_{group_idx}",
            "source_bank_ids": [bank_id],
            "group_size": 1,
            "cluster": merged_cluster,
            "memory": merged_memory,
            "memory_tokens": counter.count(merged_memory),
            "cluster_bank_text": cbase.build_cluster_memory_blob(
                [merged_key],
                merged_cluster_bank_map,
                merged_memory_bank_map,
                style,
            ),
            "merge_bank_raw": "passthrough_merge_fallback",
            "merge_bank_validation_error": "",
        }
        if bank_unit.get("memory_structured") is not None:
            merged_entry["memory_structured"] = bank_unit.get("memory_structured")
        merged_cluster_banks.append(merged_entry)

    merged_memory_text = ""
    if merged_keys:
        merged_memory_text = cbase.build_cluster_memory_blob(
            merged_keys,
            merged_cluster_bank_map,
            merged_memory_bank_map,
            style,
        )
    return merge_groups, merged_cluster_banks, merged_memory_text


def union_structured_memory_objects(memory_objects: Sequence[Optional[Dict[str, List[str]]]]) -> Optional[Dict[str, List[str]]]:
    merged: Dict[str, List[str]] = {key: [] for key in STRUCTURED_MEMORY_KEYS}
    for memory_obj in memory_objects:
        normalized = normalize_structured_memory_object(memory_obj)
        if not normalized:
            continue
        for key in STRUCTURED_MEMORY_KEYS:
            merged[key].extend(normalized.get(key) or [])
    merged = {key: dedupe_preserve_order(values) for key, values in merged.items()}
    if not any(merged.values()):
        return None
    return merged


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


def reconcile_merge_groups(
    planner_groups: Sequence[Sequence[str]],
    forced_groups: Sequence[Sequence[str]],
    valid_bank_ids: Sequence[str],
) -> List[List[str]]:
    if planner_groups:
        return [list(group) for group in planner_groups if group]

    bank_ids = [bank_id for bank_id in valid_bank_ids if bank_id]
    parent = {bank_id: bank_id for bank_id in bank_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for group in forced_groups:
        cleaned = [bank_id for bank_id in group if bank_id in parent]
        if len(cleaned) < 2:
            continue
        first = cleaned[0]
        for bank_id in cleaned[1:]:
            union(first, bank_id)

    merged: Dict[str, List[str]] = {}
    for bank_id in bank_ids:
        merged.setdefault(find(bank_id), []).append(bank_id)
    out = [group for group in merged.values()]
    out.sort(key=lambda group: (min(bank_ids.index(bid) for bid in group), len(group)))
    return out


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


def initialize_structured_cluster_memory(
    llm: Any,
    cluster: Dict[str, Any],
    document: Dict[str, Any],
    style: str,
    temperature: float,
) -> Optional[Dict[str, List[str]]]:
    prompt = INITIALIZE_STRUCTURED_CLUSTER_MEMORY_PROMPT.format(
        target_cluster=cbase.cluster_label(cluster, style),
        document=base.format_doc_for_prompt(document),
    )
    raw = llm.generate(prompt, temperature=temperature).strip()
    return parse_structured_memory_json(raw)


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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Non-streaming oracle doc summarization baseline: summarize each doc independently, concatenate summaries, then answer."
    )
    ap.add_argument("--dataset_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--layer1_out_jsonl", default="")
    ap.add_argument("--layer1_rlm_out_jsonl", default="")
    ap.add_argument("--trace_jsonl", default="")
    ap.add_argument("--llm_backend", choices=["gemini", "openrouter"], default="openrouter")
    ap.add_argument("--model", default="qwen/qwen3.5-397b-a17b")
    ap.add_argument("--answer_model", default="")
    ap.add_argument("--rlm_backend", choices=["", "gemini", "openrouter"], default="")
    ap.add_argument("--rlm_model", default="")
    ap.add_argument("--openrouter_base_url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--openrouter_http_referer", default="")
    ap.add_argument("--openrouter_app_title", default="")
    ap.add_argument(
        "--summary_mode",
        choices=[
            "query_aware_filtering",
            "generic_all_docs",
            "per_doc_full_cluster_banks",
            "per_doc_full_cluster_banks_merge2",
            "per_doc_full_cluster_banks_merge2_docsummaryaux",
            "per_doc_structured_cluster_banks",
            "per_doc_structured_cluster_banks_merge2",
        ],
        default="query_aware_filtering",
    )
    ap.add_argument("--per_doc_summary_budget_tokens", type=int, default=100)
    ap.add_argument("--aux_doc_summary_budget_tokens", type=int, default=0)
    ap.add_argument("--doc_cluster_memory_budget_tokens", type=int, default=0)
    ap.add_argument("--doc_cluster_max_queries_per_bank", type=int, default=5)
    ap.add_argument("--doc_cluster_style", choices=["list_only", "titled"], default="titled")
    ap.add_argument("--max_doc_tokens", type=int, default=12000)
    ap.add_argument("--doc_truncate_strategy", choices=["head", "middle", "tail"], default="head")
    ap.add_argument("--max_docs_per_query", type=int, default=0)
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress_every", type=int, default=10)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--rlm_completion_retries", type=int, default=2)
    ap.add_argument("--rlm_max_depth", type=int, default=1)
    ap.add_argument("--rlm_max_iterations", type=int, default=30)
    ap.add_argument("--timeout_sec", type=int, default=300)
    ap.add_argument("--summary_temperature", type=float, default=0.0)
    ap.add_argument("--answer_temperature", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_false", dest="resume")
    ap.add_argument("--skip_answer", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.per_doc_summary_budget_tokens <= 0:
        raise ValueError("--per_doc_summary_budget_tokens must be > 0.")
    if args.aux_doc_summary_budget_tokens < 0:
        raise ValueError("--aux_doc_summary_budget_tokens must be >= 0.")
    if args.max_doc_tokens < 0:
        raise ValueError("--max_doc_tokens must be >= 0.")
    if args.doc_cluster_memory_budget_tokens < 0:
        raise ValueError("--doc_cluster_memory_budget_tokens must be >= 0.")
    if args.doc_cluster_max_queries_per_bank <= 0:
        raise ValueError("--doc_cluster_max_queries_per_bank must be > 0.")
    if args.max_docs_per_query < 0:
        raise ValueError("--max_docs_per_query must be >= 0.")
    if args.start_index < 0 or args.limit < 0:
        raise ValueError("--start_index and --limit must be >= 0.")
    if args.progress_every < 0:
        raise ValueError("--progress_every must be >= 0.")
    if args.retries < 0:
        raise ValueError("--retries must be >= 0.")
    if args.rlm_completion_retries < 0:
        raise ValueError("--rlm_completion_retries must be >= 0.")
    if args.rlm_max_depth < 0:
        raise ValueError("--rlm_max_depth must be >= 0.")
    if args.rlm_max_iterations <= 0:
        raise ValueError("--rlm_max_iterations must be > 0.")
    if args.timeout_sec <= 0:
        raise ValueError("--timeout_sec must be > 0.")
    if args.layer1_rlm_out_jsonl and args.summary_mode != DOCSUMMARYAUX_SUMMARY_MODE:
        raise ValueError("--layer1_rlm_out_jsonl is only supported with per_doc_full_cluster_banks_merge2_docsummaryaux.")


def method_variant_names(
    summary_mode: str,
    per_doc_summary_budget_tokens: int,
    doc_cluster_memory_budget_tokens: int,
    doc_cluster_style: str,
    aux_doc_summary_budget_tokens: int,
) -> tuple[str, str]:
    del doc_cluster_memory_budget_tokens
    if summary_mode == "per_doc_full_cluster_banks":
        method = f"oracle_doc_cluster_bank_concat_{doc_cluster_style}"
        variant = f"{method}_Nauto_Mfree"
        return method, variant
    if summary_mode == "per_doc_full_cluster_banks_merge2":
        method = f"oracle_doc_cluster_bank_merge2_concat_{doc_cluster_style}"
        variant = f"{method}_Nauto_Mfree"
        return method, variant
    if summary_mode == "per_doc_full_cluster_banks_merge2_docsummaryaux":
        method = f"oracle_doc_cluster_bank_merge2_docsummaryaux_concat_{doc_cluster_style}"
        variant = f"{method}_Nauto_Mfree"
        return method, variant
    if summary_mode == "per_doc_structured_cluster_banks":
        method = f"oracle_doc_structured_cluster_bank_concat_{doc_cluster_style}"
        variant = f"{method}_Nauto_Mfree"
        return method, variant
    if summary_mode == "per_doc_structured_cluster_banks_merge2":
        method = f"oracle_doc_structured_cluster_bank_merge2_concat_{doc_cluster_style}"
        variant = f"{method}_Nauto_Mfree"
        return method, variant
    if summary_mode == "generic_all_docs":
        method = "oracle_doc_summary_concat_generic_alldocs"
    else:
        method = "oracle_doc_summary_concat_queryaware_filter"
    variant = f"{method}_D{per_doc_summary_budget_tokens}"
    return method, variant


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
    return base.OpenRouterClient(
        model=model,
        retry_policy=retry_policy,
        timeout_sec=timeout_sec,
        token_counter=counter,
        base_url=base_url,
        http_referer=http_referer,
        app_title=app_title,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    load_dotenv()

    dataset_path = Path(args.dataset_jsonl)
    out_path = Path(args.out_jsonl)
    layer1_out_path = Path(args.layer1_out_jsonl) if args.layer1_out_jsonl else None
    layer1_rlm_out_path = Path(args.layer1_rlm_out_jsonl) if args.layer1_rlm_out_jsonl else None
    trace_path = Path(args.trace_jsonl) if args.trace_jsonl else None
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset jsonl not found: {dataset_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if layer1_out_path:
        layer1_out_path.parent.mkdir(parents=True, exist_ok=True)
    if layer1_rlm_out_path:
        layer1_rlm_out_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_path:
        trace_path.parent.mkdir(parents=True, exist_ok=True)

    rows_all = list(iter_jsonl(dataset_path))
    rows = rows_all[args.start_index :]
    if args.limit > 0:
        rows = rows[: args.limit]

    merged_done_ids = load_done_ids(out_path, skip_answer=args.skip_answer) if args.resume else set()
    layer1_done_ids = load_done_ids(layer1_out_path, skip_answer=args.skip_answer) if (args.resume and layer1_out_path) else set()
    layer1_rlm_done_ids = load_done_ids(layer1_rlm_out_path, skip_answer=False) if (args.resume and layer1_rlm_out_path) else set()
    mode = "a" if args.resume else "w"

    counter = base.TokenCounter("cl100k_base")
    retry_policy = base.RetryPolicy(retries=args.retries)

    llm_summary: Optional[Any] = None
    llm_answer: Optional[Any] = None
    if not args.dry_run:
        llm_summary = make_llm(
            backend=args.llm_backend,
            model=args.model,
            retry_policy=retry_policy,
            timeout_sec=args.timeout_sec,
            counter=counter,
            base_url=args.openrouter_base_url,
            http_referer=args.openrouter_http_referer,
            app_title=args.openrouter_app_title,
        )
        if not args.skip_answer:
            llm_answer = make_llm(
                backend=args.llm_backend,
                model=args.answer_model or args.model,
                retry_policy=retry_policy,
                timeout_sec=args.timeout_sec,
                counter=counter,
                base_url=args.openrouter_base_url,
                http_referer=args.openrouter_http_referer,
                app_title=args.openrouter_app_title,
            )

    rlm_backend = (args.rlm_backend or args.llm_backend).strip()
    rlm_model = (args.rlm_model or args.answer_model or args.model).strip()
    rlm_base_mod: Optional[Any] = None
    rlm_docpairs_mod: Optional[Any] = None
    RLMClass: Optional[Any] = None
    rlm_backend_api_key = ""
    if layer1_rlm_out_path:
        from april_version_code.methods import rlm_official_core as rlm_base_runtime
        from april_version_code.methods import _rlm_docpair_filesystem_from_docsummaryaux as rlm_docpairs_runtime
        from rlm import RLM as RuntimeRLM

        rlm_base_mod = rlm_base_runtime
        rlm_docpairs_mod = rlm_docpairs_runtime
        RLMClass = RuntimeRLM
        if not args.dry_run:
            rlm_backend_api_key = rlm_base_mod.resolve_backend_api_key(rlm_backend)
            rlm_base_mod.install_rlm_backend_call_timer(rlm_backend)
        rlm_base_mod.TOTAL_RLM_GEMINI_CALL_WALL_TIME_SEC = 0.0

    def answer_from_memory(
        question_text: str,
        memory_text_value: str,
        use_cluster_prompt: bool,
        hybrid_summary_text: str = "",
        hybrid_bank_text: str = "",
    ) -> tuple[str, str]:
        if construction_error:
            return "", "skipped_due_to_summary_error"
        if args.skip_answer:
            return "", ""
        if args.dry_run:
            return "DRY_RUN", ""
        try:
            if llm_answer is None:
                raise RuntimeError("Answer model is not initialized.")
            if hybrid_summary_text or hybrid_bank_text:
                prompt = ANSWER_FROM_SUMMARIES_AND_CLUSTER_BANKS_PROMPT.format(
                    target_query=question_text,
                    summary_text=hybrid_summary_text if hybrid_summary_text else "(empty)",
                    bank_text=hybrid_bank_text if hybrid_bank_text else "(empty)",
                )
            else:
                prompt = (
                    ANSWER_FROM_DOC_CLUSTER_BANKS_PROMPT.format(
                        target_query=question_text,
                        memory_text=memory_text_value if memory_text_value else "(empty)",
                    )
                    if use_cluster_prompt
                    else ANSWER_FROM_DOC_SUMMARIES_PROMPT.format(
                        target_query=question_text,
                        memory_text=memory_text_value if memory_text_value else "(empty)",
                    )
                )
            return llm_answer.generate(prompt, temperature=args.answer_temperature).strip(), ""
        except Exception as exc:  # noqa: BLE001
            return "", str(exc)

    def split_hybrid_memory(memory_text_value: str) -> tuple[str, str]:
        summary_text = ""
        bank_text = ""
        summary_marker = "DOCUMENT_SUMMARIES:\n"
        bank_marker = "\n\nDOCUMENT_CLUSTER_BANKS:\n"
        merged_summary_marker = "MERGED_CLUSTER_SUMMARIES:\n"
        merged_bank_marker = "\n\nMERGED_CLUSTER_BANKS:\n"
        if memory_text_value.startswith(summary_marker) and bank_marker in memory_text_value:
            summary_text, bank_text = memory_text_value.split(bank_marker, 1)
            summary_text = summary_text.removeprefix(summary_marker).strip()
            bank_text = bank_text.strip()
            return summary_text, bank_text
        if memory_text_value.startswith(merged_summary_marker) and merged_bank_marker in memory_text_value:
            summary_text, bank_text = memory_text_value.split(merged_bank_marker, 1)
            summary_text = summary_text.removeprefix(merged_summary_marker).strip()
            bank_text = bank_text.strip()
            return summary_text, bank_text
        return "", ""

    with out_path.open(mode, encoding="utf-8") as fout:
        layer1_fout = layer1_out_path.open(mode, encoding="utf-8") if layer1_out_path else None
        layer1_rlm_fout = layer1_rlm_out_path.open(mode, encoding="utf-8") if layer1_rlm_out_path else None
        trace_file = trace_path.open(mode, encoding="utf-8") if trace_path else None
        try:
            def answer_with_rlm_docpairs(
                qid_value: str,
                question_text: str,
                gold_answer_text: str,
                docs_value: Sequence[Dict[str, Any]],
                doc_cluster_banks_value: Sequence[Dict[str, Any]],
                source_method_value: str,
                source_variant_value: str,
                dataset_row: Dict[str, Any],
            ) -> Dict[str, Any]:
                if rlm_base_mod is None or rlm_docpairs_mod is None:
                    raise RuntimeError("RLM doc-pair integration requested but helper modules are unavailable.")

                file_specs, source_context_doc_ids, doc_char_truncations_rlm, doc_token_truncations_rlm, unmatched_doc_ids = (
                    rlm_docpairs_mod.build_docpair_files(
                        docs=docs_value,
                        doc_cluster_banks=doc_cluster_banks_value,
                        counter=counter,
                        max_doc_tokens=args.max_doc_tokens,
                        doc_truncate_strategy=args.doc_truncate_strategy,
                        max_doc_chars=0,
                    )
                )
                context_payload = rlm_docpairs_mod.build_context_payload(file_specs)
                setup_code = rlm_docpairs_mod.build_setup_code(file_specs)
                file_inventory = rlm_docpairs_mod.inventory_for_output(file_specs)

                raw_model_answer = ""
                model_answer = ""
                answer_extraction_mode = ""
                answer_extraction_error = ""
                runtime_error = ""
                skip_reason = ""
                usage: Dict[str, Any] = {}
                rlm_started = time.time()
                rlm_wall_before = rlm_base_mod.TOTAL_RLM_GEMINI_CALL_WALL_TIME_SEC

                if args.dry_run:
                    raw_model_answer = "DRY_RUN"
                    model_answer = "DRY_RUN"
                    answer_extraction_mode = "dry_run"
                else:
                    try:
                        if RLMClass is None:
                            raise RuntimeError("RLM class is not initialized.")
                        rlm_kwargs: Dict[str, Any] = {
                            "backend": rlm_backend,
                            "backend_kwargs": {"model_name": rlm_model, "api_key": rlm_backend_api_key},
                            "environment": "local",
                            "max_depth": args.rlm_max_depth,
                            "max_iterations": args.rlm_max_iterations,
                            "verbose": False,
                            "environment_kwargs": {"setup_code": setup_code},
                        }
                        if rlm_backend == "openrouter":
                            rlm_kwargs["backend_kwargs"]["base_url"] = args.openrouter_base_url
                        rlm_obj = RLMClass(**rlm_kwargs)
                        completion_obj = rlm_base_mod.completion_with_retries(
                            rlm_obj=rlm_obj,
                            prompt=context_payload,
                            root_prompt=question_text,
                            retries=args.rlm_completion_retries,
                        )
                        raw_model_answer = str(getattr(completion_obj, "response", "") or "").strip()
                        (
                            model_answer,
                            answer_extraction_mode,
                            answer_extraction_error,
                        ) = rlm_base_mod.extract_final_answer(raw_model_answer)
                        if answer_extraction_error:
                            runtime_error = f"answer_extraction_error: {answer_extraction_error}"
                            skip_reason = "answer_extraction_error"
                        usage_obj = getattr(completion_obj, "usage_summary", None)
                        usage = usage_obj.to_dict() if usage_obj else {}
                    except Exception as exc:  # noqa: BLE001
                        runtime_error = f"{type(exc).__name__}: {exc}"
                        skip_reason = "runtime_error"

                rlm_wall_time_sec = max(
                    0.0,
                    float(rlm_base_mod.TOTAL_RLM_GEMINI_CALL_WALL_TIME_SEC - rlm_wall_before),
                )
                usage_totals = rlm_base_mod.extract_rlm_usage_totals(usage)
                stream_doc_tokens = [counter.count(str(d.get("text", ""))) for d in docs_value]
                out_row = {
                    "question_id": qid_value,
                    "question": question_text,
                    "gold_answer": gold_answer_text,
                    "variant": rlm_docpairs_mod.VARIANT_NAME,
                    "method": rlm_docpairs_mod.METHOD_NAME,
                    "docsummaryaux_source_method": source_method_value,
                    "docsummaryaux_source_variant": source_variant_value,
                    "docsummaryaux_results_jsonl": str(layer1_out_path) if layer1_out_path else "",
                    "model": rlm_model,
                    "backend": rlm_backend,
                    "max_depth": args.rlm_max_depth,
                    "max_iterations": args.rlm_max_iterations,
                    "model_answer": model_answer,
                    "raw_model_answer": raw_model_answer,
                    "answer_extraction_mode": answer_extraction_mode,
                    "answer_extraction_error": answer_extraction_error,
                    "is_exact_match": bool(rlm_base_mod.exact_match(model_answer, gold_answer_text) if model_answer else False),
                    "row_failed": bool(runtime_error),
                    "latency_sec": round(time.time() - rlm_started, 3),
                    "runtime_error": runtime_error,
                    "skip_reason": skip_reason,
                    "augmentation_mode": "filesystem_doc_pairs",
                    "num_stream_docs": len(docs_value),
                    "num_context_docs_loaded": len(context_payload),
                    "context_doc_ids": ["local_file_index"],
                    "source_context_doc_ids": source_context_doc_ids,
                    "num_raw_doc_files": sum(1 for spec in file_specs if spec["kind"] == "raw"),
                    "num_companion_doc_files": sum(
                        1 for spec in file_specs if str(spec["kind"]).startswith("clusters_summary")
                    ),
                    "num_total_docpair_files": len(file_specs),
                    "docpair_files": file_inventory,
                    "docpair_unmatched_source_doc_ids": unmatched_doc_ids,
                    "doc_truncate_strategy": args.doc_truncate_strategy,
                    "max_doc_tokens": args.max_doc_tokens,
                    "doc_char_truncations": doc_char_truncations_rlm,
                    "doc_token_truncations": doc_token_truncations_rlm,
                    "stream_doc_tokens": stream_doc_tokens,
                    "stream_total_tokens": sum(stream_doc_tokens),
                    "num_source_doc_summaries_nonempty": sum(
                        1 for item in doc_cluster_banks_value if str(item.get("source_doc_summary", "") or "").strip()
                    ),
                    "num_doc_cluster_banks_nonempty": sum(
                        1 for item in doc_cluster_banks_value if str(item.get("cluster_bank_text", "") or "").strip()
                    ),
                    "rlm_usage": usage,
                    "rlm_usage_totals": usage_totals,
                    "total_lm_calls": usage_totals["calls"],
                    "total_lm_input_tokens": usage_totals["input_tokens"],
                    "total_lm_output_tokens": usage_totals["output_tokens"],
                    "total_lm_tokens": usage_totals["total_tokens"],
                    "lm_call_wall_time_sec": round(float(rlm_wall_time_sec), 6),
                    "total_lm_wall_time_sec": round(float(rlm_wall_time_sec), 6),
                    "answer_tokens": counter.count(model_answer),
                    "memory_text": "",
                    "memory_tokens": 0,
                    "execute_calls": 0,
                    "memory_state_present": False,
                    "dry_run": bool(args.dry_run),
                }
                row_metadata.attach_sample_metadata(out_row, dataset_row)
                return out_row

            processed = 0
            for row in rows:
                qid = str(row.get("question_id", "")).strip()
                if not qid:
                    continue
                need_merged_output = qid not in merged_done_ids
                need_layer1_output = layer1_fout is not None and qid not in layer1_done_ids
                need_layer1_rlm_output = layer1_rlm_fout is not None and qid not in layer1_rlm_done_ids
                if not (need_merged_output or need_layer1_output or need_layer1_rlm_output):
                    continue

                question = str(row.get("question", ""))
                gold_answer = str(row.get("gold_answer", ""))
                method_name, variant_name = method_variant_names(
                    args.summary_mode,
                    args.per_doc_summary_budget_tokens,
                    args.doc_cluster_memory_budget_tokens,
                    args.doc_cluster_style,
                    args.aux_doc_summary_budget_tokens,
                )
                docs = list(row.get("docs") or row.get("stream_docs") or [])
                if args.max_docs_per_query > 0:
                    docs = docs[: args.max_docs_per_query]

                summary_before_row = usage_snapshot(llm_summary)

                started = time.time()
                construction_error = ""
                merge_error = ""
                doc_truncations = 0
                summary_truncations = 0
                doc_summaries: List[Dict[str, Any]] = []
                doc_cluster_banks: List[Dict[str, Any]] = []
                bank_units_all: List[Dict[str, Any]] = []
                merge_plan_raw = ""
                merge_groups: List[List[str]] = []
                merged_cluster_banks: List[Dict[str, Any]] = []
                kept_blocks: List[str] = []
                layer1_memory_text = ""
                merged_memory_text = ""

                for doc_idx, doc in enumerate(docs, start=1):
                    doc_id = str(doc.get("doc_id", "")).strip()
                    raw_doc_text = base.format_doc_for_prompt(doc)
                    raw_doc_tokens = counter.count(raw_doc_text)
                    doc_text = raw_doc_text
                    if args.max_doc_tokens > 0 and raw_doc_tokens > args.max_doc_tokens:
                        doc_text = counter.truncate(
                            raw_doc_text,
                            max_tokens=args.max_doc_tokens,
                            strategy=args.doc_truncate_strategy,
                        )
                        doc_truncations += 1

                    source_doc_summary = ""
                    source_doc_summary_error = ""
                    source_doc_summary_fallback_used = False
                    source_doc_summary_ref = ""
                    if args.summary_mode == "per_doc_full_cluster_banks_merge2_docsummaryaux":
                        source_doc_summary, source_doc_summary_error, source_doc_summary_fallback_used = generate_query_aware_aux_doc_summary(
                            llm=llm_summary,
                            counter=counter,
                            question=question,
                            doc_text=doc_text,
                            summary_budget_tokens=args.aux_doc_summary_budget_tokens,
                            temperature=args.summary_temperature,
                            dry_run=args.dry_run,
                        )
                        if source_doc_summary:
                            source_doc_summary_ref = f"SOURCE_DOC_SUMMARY_DOC_{doc_idx}"

                    if args.summary_mode in CLUSTER_BANK_SUMMARY_MODES:
                        summary_error = ""
                        cluster_bank_text = ""
                        cluster_memory_fallbacks = 0
                        doc_for_cluster = doc_with_text(doc, doc_text)
                        selected_clusters: List[Dict[str, Any]] = []
                        selected_cluster_keys: List[str] = []
                        memory_bank: Dict[str, str] = {}
                        memory_bank_structured: Dict[str, Dict[str, List[str]]] = {}
                        bank_units_doc: List[Dict[str, Any]] = []

                        try:
                            if args.dry_run:
                                selected_clusters = cbase.build_fallback_clusters(
                                    [doc_for_cluster],
                                    1,
                                    args.doc_cluster_style,
                                    args.doc_cluster_max_queries_per_bank,
                                )
                            else:
                                if llm_summary is None:
                                    raise RuntimeError("Summary model is not initialized.")
                                selected_clusters = generate_candidate_clusters_warm_auto(
                                    llm=llm_summary,
                                    warm_docs=[doc_for_cluster],
                                    max_queries_per_cluster=args.doc_cluster_max_queries_per_bank,
                                    style=args.doc_cluster_style,
                                    temperature=args.summary_temperature,
                                )
                            if not selected_clusters:
                                selected_clusters = cbase.build_fallback_clusters(
                                    [doc_for_cluster],
                                    1,
                                    args.doc_cluster_style,
                                    args.doc_cluster_max_queries_per_bank,
                                )

                            cluster_bank = {
                                cbase.cluster_key(cluster, args.doc_cluster_style): cluster
                                for cluster in selected_clusters
                            }
                            for cluster in selected_clusters:
                                key = cbase.cluster_key(cluster, args.doc_cluster_style)
                                selected_cluster_keys.append(key)
                                cluster_memory_structured: Optional[Dict[str, List[str]]] = None
                                if args.dry_run:
                                    if args.summary_mode in STRUCTURED_CLUSTER_BANK_SUMMARY_MODES:
                                        cluster_memory_structured = normalize_structured_memory_object(
                                            {
                                                "exact_strings": [],
                                                "facts": [doc_text],
                                                "relations": [],
                                                "qualifiers": [],
                                            }
                                        )
                                        cluster_memory = render_structured_memory(cluster_memory_structured)
                                    else:
                                        cluster_memory = doc_text
                                else:
                                    if llm_summary is None:
                                        raise RuntimeError("Summary model is not initialized.")
                                    if args.summary_mode in STRUCTURED_CLUSTER_BANK_SUMMARY_MODES:
                                        cluster_memory_structured = initialize_structured_cluster_memory(
                                            llm=llm_summary,
                                            cluster=cluster,
                                            document=doc_for_cluster,
                                            temperature=args.summary_temperature,
                                            style=args.doc_cluster_style,
                                        )
                                        cluster_memory = render_structured_memory(cluster_memory_structured)
                                    else:
                                        cluster_memory = initialize_cluster_memory_unbounded(
                                            llm=llm_summary,
                                            cluster=cluster,
                                            document=doc_for_cluster,
                                            temperature=args.summary_temperature,
                                            style=args.doc_cluster_style,
                                        ).strip()
                                if not cluster_memory:
                                    if args.summary_mode in STRUCTURED_CLUSTER_BANK_SUMMARY_MODES:
                                        cluster_memory_structured = normalize_structured_memory_object(
                                            {
                                                "exact_strings": [],
                                                "facts": [doc_text],
                                                "relations": [],
                                                "qualifiers": [],
                                            }
                                        )
                                        cluster_memory = render_structured_memory(cluster_memory_structured)
                                    else:
                                        cluster_memory = doc_text
                                    cluster_memory_fallbacks += 1
                                memory_bank[key] = cluster_memory
                                if cluster_memory_structured is not None:
                                    memory_bank_structured[key] = cluster_memory_structured
                            for bank_idx, cluster in enumerate(selected_clusters, start=1):
                                key = selected_cluster_keys[bank_idx - 1]
                                bank_text = cbase.build_cluster_memory_blob(
                                    [key],
                                    cluster_bank,
                                    memory_bank,
                                    args.doc_cluster_style,
                                )
                                bank_unit = {
                                    "bank_id": f"doc{doc_idx}_bank{bank_idx}",
                                    "doc_idx": doc_idx,
                                    "doc_id": doc_id,
                                    "bank_idx": bank_idx,
                                    "cluster_key": key,
                                    "cluster": cluster,
                                    "memory": memory_bank.get(key, ""),
                                    "bank_text": bank_text,
                                }
                                if source_doc_summary_ref:
                                    bank_unit["source_doc_summary_ref"] = source_doc_summary_ref
                                if key in memory_bank_structured:
                                    bank_unit["memory_structured"] = memory_bank_structured[key]
                                bank_units_doc.append(bank_unit)
                                bank_units_all.append(bank_unit)
                            cluster_bank_text = cbase.build_cluster_memory_blob(
                                selected_cluster_keys,
                                cluster_bank,
                                memory_bank,
                                args.doc_cluster_style,
                            )
                            if cluster_bank_text:
                                if args.summary_mode == DOCSUMMARYAUX_SUMMARY_MODE:
                                    if source_doc_summary:
                                        kept_blocks.append(format_summary_block(doc_idx, doc_id, source_doc_summary))
                                elif cluster_bank_text:
                                    kept_blocks.append(format_cluster_bank_block(doc_idx, doc_id, cluster_bank_text))
                        except Exception as exc:  # noqa: BLE001
                            summary_error = str(exc)
                            selected_clusters = cbase.build_fallback_clusters(
                                [doc_for_cluster],
                                1,
                                args.doc_cluster_style,
                                args.doc_cluster_max_queries_per_bank,
                            )
                            cluster_bank = {
                                cbase.cluster_key(cluster, args.doc_cluster_style): cluster
                                for cluster in selected_clusters
                            }
                            selected_cluster_keys = []
                            memory_bank = {}
                            memory_bank_structured = {}
                            bank_units_doc = []
                            cluster_memory_fallbacks += 1
                            for bank_idx, cluster in enumerate(selected_clusters, start=1):
                                key = cbase.cluster_key(cluster, args.doc_cluster_style)
                                selected_cluster_keys.append(key)
                                if args.summary_mode in STRUCTURED_CLUSTER_BANK_SUMMARY_MODES:
                                    cluster_memory_structured = fallback_structured_memory_for_document(doc_text)
                                    cluster_memory = render_structured_memory(cluster_memory_structured)
                                    if cluster_memory_structured is not None:
                                        memory_bank_structured[key] = cluster_memory_structured
                                else:
                                    cluster_memory = doc_text
                                memory_bank[key] = cluster_memory
                                bank_text = cbase.build_cluster_memory_blob(
                                    [key],
                                    cluster_bank,
                                    memory_bank,
                                    args.doc_cluster_style,
                                )
                                bank_unit = {
                                    "bank_id": f"doc{doc_idx}_bank{bank_idx}",
                                    "doc_idx": doc_idx,
                                    "doc_id": doc_id,
                                    "bank_idx": bank_idx,
                                    "cluster_key": key,
                                    "cluster": cluster,
                                    "memory": cluster_memory,
                                    "bank_text": bank_text,
                                }
                                if source_doc_summary_ref:
                                    bank_unit["source_doc_summary_ref"] = source_doc_summary_ref
                                if key in memory_bank_structured:
                                    bank_unit["memory_structured"] = memory_bank_structured[key]
                                bank_units_doc.append(bank_unit)
                                bank_units_all.append(bank_unit)
                            cluster_bank_text = cbase.build_cluster_memory_blob(
                                selected_cluster_keys,
                                cluster_bank,
                                memory_bank,
                                args.doc_cluster_style,
                            )
                            if args.summary_mode == DOCSUMMARYAUX_SUMMARY_MODE:
                                if source_doc_summary:
                                    kept_blocks.append(format_summary_block(doc_idx, doc_id, source_doc_summary))
                            elif cluster_bank_text:
                                kept_blocks.append(format_cluster_bank_block(doc_idx, doc_id, cluster_bank_text))

                        doc_cluster_row = {
                            "doc_idx": doc_idx,
                            "doc_id": doc_id,
                            "is_gold": bool(doc.get("is_gold", False)),
                            "raw_doc_tokens": raw_doc_tokens,
                            "doc_tokens_after_cap": counter.count(doc_text),
                            "num_clusters": len(selected_clusters),
                            "selected_clusters": selected_clusters,
                            "selected_cluster_keys": selected_cluster_keys,
                            "memory_bank": memory_bank,
                            "memory_bank_structured": memory_bank_structured,
                            "cluster_bank_text": cluster_bank_text,
                            "cluster_bank_tokens": counter.count(cluster_bank_text),
                            "cluster_bank_empty": not bool(cluster_bank_text),
                            "cluster_bank_error": summary_error,
                            "cluster_memory_fallbacks": cluster_memory_fallbacks,
                            "source_doc_summary": source_doc_summary,
                            "source_doc_summary_tokens": counter.count(source_doc_summary),
                            "source_doc_summary_error": source_doc_summary_error,
                            "source_doc_summary_fallback_used": bool(source_doc_summary_fallback_used),
                            "bank_units": bank_units_doc,
                        }
                        doc_cluster_banks.append(doc_cluster_row)

                        if trace_file:
                            trace_file.write(
                                json.dumps(
                                    {
                                        "question_id": qid,
                                        "question": question,
                                        **doc_cluster_row,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                        if summary_error and not cluster_bank_text:
                            construction_error = construction_error or f"cluster_bank_failed_doc_{doc_idx}"
                            break
                    else:
                        raw_summary_text = ""
                        summary_text = ""
                        summary_error = ""
                        summary_fallback_used = False
                        if args.dry_run:
                            raw_summary_text = counter.truncate(doc_text, args.per_doc_summary_budget_tokens, strategy="head")
                            summary_text = raw_summary_text
                        else:
                            try:
                                if llm_summary is None:
                                    raise RuntimeError("Summary model is not initialized.")
                                if args.summary_mode == "generic_all_docs":
                                    prompt = GENERIC_ALL_DOCS_SUMMARY_PROMPT.format(
                                        summary_budget_tokens=args.per_doc_summary_budget_tokens,
                                        document=doc_text,
                                    )
                                else:
                                    prompt = QUERY_AWARE_FILTERING_DOC_SUMMARY_PROMPT.format(
                                        target_query=question,
                                        summary_budget_tokens=args.per_doc_summary_budget_tokens,
                                        document=doc_text,
                                    )
                                raw_summary_text = llm_summary.generate(
                                    prompt,
                                    temperature=args.summary_temperature,
                                ).strip()
                                if args.summary_mode == "query_aware_filtering" and base.is_none_snippet(raw_summary_text):
                                    summary_text = ""
                                else:
                                    summary_text = raw_summary_text
                                if args.summary_mode == "generic_all_docs" and (not summary_text or base.is_none_snippet(summary_text)):
                                    summary_text = counter.truncate(
                                        doc_text,
                                        max_tokens=args.per_doc_summary_budget_tokens,
                                        strategy="head",
                                    )
                                    summary_fallback_used = True
                            except Exception as exc:  # noqa: BLE001
                                summary_error = str(exc)
                                construction_error = f"summary_failed_doc_{doc_idx}"

                        if summary_text:
                            truncated_summary = counter.truncate(
                                summary_text,
                                max_tokens=args.per_doc_summary_budget_tokens,
                                strategy="head",
                            )
                            if truncated_summary != summary_text:
                                summary_truncations += 1
                            summary_text = truncated_summary
                            kept_blocks.append(format_summary_block(doc_idx, doc_id, summary_text))

                        doc_summary_row = {
                            "doc_idx": doc_idx,
                            "doc_id": doc_id,
                            "is_gold": bool(doc.get("is_gold", False)),
                            "raw_doc_tokens": raw_doc_tokens,
                            "doc_tokens_after_cap": counter.count(doc_text),
                            "raw_summary_text": raw_summary_text,
                            "summary_text": summary_text,
                            "summary_tokens": counter.count(summary_text),
                            "summary_truncated": bool(summary_text and raw_summary_text and summary_text != raw_summary_text),
                            "summary_empty": not bool(summary_text),
                            "summary_error": summary_error,
                            "summary_fallback_used": bool(summary_fallback_used),
                        }
                        doc_summaries.append(doc_summary_row)

                        if trace_file:
                            trace_file.write(
                                json.dumps(
                                    {
                                        "question_id": qid,
                                        "question": question,
                                        **doc_summary_row,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                        if summary_error:
                            break

                layer1_memory_text = "\n\n".join(block for block in kept_blocks if block.strip()).strip()
                if args.summary_mode == DOCSUMMARYAUX_SUMMARY_MODE:
                    layer1_memory_text = build_docsummaryaux_layer1_memory(doc_cluster_banks)
                summary_after_layer1 = usage_snapshot(llm_summary)

                layer1_answer_before = usage_snapshot(llm_answer)
                layer1_final_answer = ""
                layer1_answer_error = ""
                if args.summary_mode in CLUSTER_BANK_SUMMARY_MODES:
                    layer1_summary_text_for_answer = ""
                    layer1_bank_text_for_answer = ""
                    if args.summary_mode == DOCSUMMARYAUX_SUMMARY_MODE:
                        layer1_summary_text_for_answer, layer1_bank_text_for_answer = split_hybrid_memory(layer1_memory_text)
                    layer1_final_answer, layer1_answer_error = answer_from_memory(
                        question,
                        layer1_memory_text,
                        use_cluster_prompt=args.summary_mode != DOCSUMMARYAUX_SUMMARY_MODE,
                        hybrid_summary_text=layer1_summary_text_for_answer,
                        hybrid_bank_text=layer1_bank_text_for_answer,
                    )
                else:
                    layer1_final_answer, layer1_answer_error = answer_from_memory(
                        question,
                        layer1_memory_text,
                        use_cluster_prompt=False,
                    )
                layer1_answer_after = usage_snapshot(llm_answer)

                merged_memory_text = layer1_memory_text
                if not construction_error and args.summary_mode in MERGE2_SUMMARY_MODES:
                    valid_bank_ids = [str(bank_unit.get("bank_id", "")).strip() for bank_unit in bank_units_all if str(bank_unit.get("bank_id", "")).strip()]
                    if bank_units_all:
                        try:
                            forced_merge_groups = build_forced_merge_groups(bank_units_all, args.doc_cluster_style)
                            if args.dry_run:
                                merge_groups = reconcile_merge_groups(
                                    [[bank_id] for bank_id in valid_bank_ids],
                                    forced_merge_groups,
                                    valid_bank_ids,
                                )
                            else:
                                if llm_summary is None:
                                    raise RuntimeError("Summary model is not initialized.")
                                merge_plan_prompt = MERGE_CLUSTER_BANKS_PLAN_PROMPT.format(
                                    target_query=question,
                                    bank_units=format_bank_units_for_merge_planner(bank_units_all, args.doc_cluster_style) or "(empty)",
                                    source_doc_summaries=format_source_doc_summaries_for_merge(bank_units_all, doc_cluster_banks) or "(none)",
                                    heuristic_groups=format_heuristic_groups(forced_merge_groups),
                                )
                                merge_plan_raw = llm_summary.generate(
                                    merge_plan_prompt,
                                    temperature=args.summary_temperature,
                                ).strip()
                                planner_groups = parse_merge_groups(merge_plan_raw, valid_bank_ids)
                                merge_groups = reconcile_merge_groups(
                                    planner_groups,
                                    forced_merge_groups,
                                    valid_bank_ids,
                                )
                                if trace_file:
                                    trace_file.write(
                                        json.dumps(
                                            {
                                                "question_id": qid,
                                                "question": question,
                                                "phase": "merge_planner",
                                                "merge_plan_raw": merge_plan_raw,
                                                "heuristic_high_priority_groups": forced_merge_groups,
                                                "merge_groups": merge_groups,
                                            },
                                            ensure_ascii=False,
                                        )
                                        + "\n"
                                    )
                            bank_unit_by_id = {str(bank_unit["bank_id"]): bank_unit for bank_unit in bank_units_all}
                            merged_cluster_bank_map: Dict[str, Dict[str, Any]] = {}
                            merged_memory_bank_map: Dict[str, str] = {}
                            merged_keys: List[str] = []
                            for group_idx, group_bank_ids in enumerate(merge_groups, start=1):
                                group_bank_units = [
                                    bank_unit_by_id[bank_id]
                                    for bank_id in group_bank_ids
                                    if bank_id in bank_unit_by_id
                                ]
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
                                elif args.dry_run:
                                    merged_bank = fallback_merged_bank(
                                        group_bank_units,
                                        style=args.doc_cluster_style,
                                        max_queries_per_cluster=args.doc_cluster_max_queries_per_bank,
                                        structured_memory=args.summary_mode in STRUCTURED_CLUSTER_BANK_SUMMARY_MODES,
                                    )
                                else:
                                    style_rule, _ = cbase.style_rule_and_schema(args.doc_cluster_style)
                                    if args.summary_mode in STRUCTURED_CLUSTER_BANK_SUMMARY_MODES:
                                        if args.doc_cluster_style == "titled":
                                            merged_bank_schema = '"title": "...", "queries": ["...", "..."], "memory": {"exact_strings": ["..."], "facts": ["..."], "relations": ["..."], "qualifiers": ["..."]}'
                                        else:
                                            merged_bank_schema = '"queries": ["...", "..."], "memory": {"exact_strings": ["..."], "facts": ["..."], "relations": ["..."], "qualifiers": ["..."]}'
                                        merge_exec_prompt = MERGE_STRUCTURED_CLUSTER_BANKS_EXECUTION_PROMPT.format(
                                            target_query=question,
                                            max_queries_per_cluster=args.doc_cluster_max_queries_per_bank,
                                            group_banks=format_bank_units_for_merge_planner(group_bank_units, args.doc_cluster_style) or "(empty)",
                                            source_doc_summaries=format_source_doc_summaries_for_merge(group_bank_units, doc_cluster_banks) or "(none)",
                                            style_rule=style_rule,
                                            merged_bank_schema=merged_bank_schema,
                                        )
                                    else:
                                        if args.doc_cluster_style == "titled":
                                            merged_bank_schema = '"title": "...", "queries": ["...", "..."], "memory": "..."'
                                        else:
                                            merged_bank_schema = '"queries": ["...", "..."], "memory": "..."'
                                        merge_exec_prompt = MERGE_CLUSTER_BANKS_EXECUTION_PROMPT.format(
                                            target_query=question,
                                            max_queries_per_cluster=args.doc_cluster_max_queries_per_bank,
                                            group_banks=format_bank_units_for_merge_planner(group_bank_units, args.doc_cluster_style) or "(empty)",
                                            source_doc_summaries=format_source_doc_summaries_for_merge(group_bank_units, doc_cluster_banks) or "(none)",
                                            style_rule=style_rule,
                                            merged_bank_schema=merged_bank_schema,
                                        )
                                    merged_bank_raw = llm_summary.generate(
                                        merge_exec_prompt,
                                        temperature=args.summary_temperature,
                                    ).strip()
                                    parsed_merged_bank = parse_merged_bank_json(
                                        merged_bank_raw,
                                        style=args.doc_cluster_style,
                                        max_queries_per_cluster=args.doc_cluster_max_queries_per_bank,
                                        structured_memory=args.summary_mode in STRUCTURED_CLUSTER_BANK_SUMMARY_MODES,
                                    )
                                    if parsed_merged_bank is not None:
                                        is_valid_merge, merged_bank_validation_error = validate_merged_bank_against_sources(
                                            parsed_merged_bank,
                                            group_bank_units,
                                            args.doc_cluster_style,
                                        )
                                        if not is_valid_merge:
                                            parsed_merged_bank = None
                                    merged_bank = parsed_merged_bank or fallback_merged_bank(
                                        group_bank_units,
                                        style=args.doc_cluster_style,
                                        max_queries_per_cluster=args.doc_cluster_max_queries_per_bank,
                                        structured_memory=args.summary_mode in STRUCTURED_CLUSTER_BANK_SUMMARY_MODES,
                                    )
                                merged_cluster = merged_bank["cluster"]
                                merged_memory = str(merged_bank["memory"] or "").strip()
                                merged_key = cbase.cluster_key(merged_cluster, args.doc_cluster_style)
                                if merged_key in merged_cluster_bank_map:
                                    merged_key = f"{merged_key}__group_{group_idx}"
                                merged_keys.append(merged_key)
                                merged_cluster_bank_map[merged_key] = merged_cluster
                                merged_memory_bank_map[merged_key] = merged_memory
                                merged_cluster_banks.append(
                                    {
                                        "merged_bank_id": f"merged_group_{group_idx}",
                                        "source_bank_ids": group_bank_ids,
                                        "group_size": len(group_bank_ids),
                                        "cluster": merged_cluster,
                                        "memory": merged_memory,
                                        "memory_tokens": counter.count(merged_memory),
                                        "cluster_bank_text": cbase.build_cluster_memory_blob(
                                            [merged_key],
                                            merged_cluster_bank_map,
                                            merged_memory_bank_map,
                                            args.doc_cluster_style,
                                        ),
                                        "merge_bank_raw": merged_bank_raw,
                                        "merge_bank_validation_error": merged_bank_validation_error,
                                    }
                                )
                                if merged_bank.get("memory_structured") is not None:
                                    merged_cluster_banks[-1]["memory_structured"] = merged_bank.get("memory_structured")
                                if trace_file:
                                    trace_file.write(
                                        json.dumps(
                                            {
                                                "question_id": qid,
                                                "question": question,
                                                "phase": "merge_execution",
                                                "merged_bank_id": f"merged_group_{group_idx}",
                                                "source_bank_ids": group_bank_ids,
                                                "merge_bank_raw": merged_bank_raw,
                                                "merge_bank_validation_error": merged_bank_validation_error,
                                                "merged_cluster_bank": merged_cluster_banks[-1],
                                            },
                                            ensure_ascii=False,
                                        )
                                        + "\n"
                                    )
                            if merged_keys:
                                merged_memory_text = cbase.build_cluster_memory_blob(
                                    merged_keys,
                                    merged_cluster_bank_map,
                                    merged_memory_bank_map,
                                    args.doc_cluster_style,
                                )
                            else:
                                merged_memory_text = ""
                        except Exception as exc:  # noqa: BLE001
                            merge_error = f"merge2_failed: {exc}"
                            merge_groups, merged_cluster_banks, merged_memory_text = build_singleton_merged_outputs(
                                bank_units_all,
                                args.doc_cluster_style,
                                counter,
                            )
                            if trace_file:
                                trace_file.write(
                                    json.dumps(
                                        {
                                            "question_id": qid,
                                            "question": question,
                                            "phase": "merge_fallback",
                                            "merge_error": merge_error,
                                            "fallback_merge_groups": merge_groups,
                                            "fallback_num_merged_cluster_banks": len(merged_cluster_banks),
                                            "fallback_memory_tokens": counter.count(merged_memory_text),
                                        },
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                            if merged_memory_text.strip():
                                merge_error = ""
                    else:
                        merged_memory_text = ""

                if args.summary_mode == DOCSUMMARYAUX_SUMMARY_MODE:
                    bank_unit_by_id = {str(bank_unit.get("bank_id", "") or ""): bank_unit for bank_unit in bank_units_all}
                    merged_memory_text = attach_docsummaryaux_merged_summaries(
                        llm=llm_summary,
                        counter=counter,
                        question=question,
                        merged_cluster_banks=merged_cluster_banks,
                        bank_unit_by_id=bank_unit_by_id,
                        doc_cluster_banks=doc_cluster_banks,
                        style=args.doc_cluster_style,
                        temperature=args.summary_temperature,
                        dry_run=args.dry_run,
                    )
                    if not merged_memory_text:
                        merged_memory_text = layer1_memory_text

                summary_after_merged = usage_snapshot(llm_summary)

                merged_answer_before = usage_snapshot(llm_answer)
                merged_final_answer = layer1_final_answer
                merged_answer_error = layer1_answer_error
                if args.summary_mode in MERGE2_SUMMARY_MODES:
                    if merge_error and not merged_memory_text:
                        merged_final_answer = ""
                        merged_answer_error = "skipped_due_to_merge_error"
                    else:
                        merged_summary_text_for_answer = ""
                        merged_bank_text_for_answer = ""
                        if args.summary_mode == DOCSUMMARYAUX_SUMMARY_MODE:
                            merged_summary_text_for_answer, merged_bank_text_for_answer = split_hybrid_memory(merged_memory_text)
                        merged_final_answer, merged_answer_error = answer_from_memory(
                            question,
                            merged_memory_text,
                            use_cluster_prompt=args.summary_mode != DOCSUMMARYAUX_SUMMARY_MODE,
                            hybrid_summary_text=merged_summary_text_for_answer,
                            hybrid_bank_text=merged_bank_text_for_answer,
                        )
                merged_answer_after = usage_snapshot(llm_answer)

                layer1_summary_usage = usage_delta(summary_before_row, summary_after_layer1)
                layer1_answer_usage = usage_delta(layer1_answer_before, layer1_answer_after)
                merged_summary_usage = usage_delta(summary_before_row, summary_after_merged)
                merged_answer_usage = usage_delta(merged_answer_before, merged_answer_after)

                def build_out_row(
                    method_value: str,
                    variant_value: str,
                    summary_mode_value: str,
                    memory_text_value: str,
                    runtime_error_value: str,
                    answer_error_value: str,
                    final_answer_value: str,
                    summary_usage_value: Dict[str, Any],
                    answer_usage_value: Dict[str, Any],
                    merge_plan_raw_value: str,
                    merge_groups_value: List[List[str]],
                    merged_cluster_banks_value: List[Dict[str, Any]],
                    stage_label: str,
                    num_merge_groups_value: int,
                    num_merged_cluster_banks_value: int,
                ) -> Dict[str, Any]:
                    total_lm_calls = summary_usage_value["calls"] + answer_usage_value["calls"]
                    total_lm_input_tokens = summary_usage_value["input_tokens"] + answer_usage_value["input_tokens"]
                    total_lm_output_tokens = summary_usage_value["output_tokens"] + answer_usage_value["output_tokens"]
                    total_lm_wall_time_sec = summary_usage_value["wall_time_sec"] + answer_usage_value["wall_time_sec"]
                    out_row = {
                        "variant": variant_value,
                        "method": method_value,
                        "question_id": qid,
                        "question": question,
                        "gold_answer": gold_answer,
                        "llm_backend": args.llm_backend,
                        "summary_model": args.model,
                        "answer_model": (args.answer_model or args.model) if not args.skip_answer else "",
                        "num_stream_docs": len(docs),
                        "num_doc_summaries_nonempty": sum(1 for item in doc_summaries if item["summary_text"]),
                        "num_source_doc_summaries_nonempty": sum(
                            1
                            for item in doc_cluster_banks
                            if str(item.get("source_doc_summary", "") or "").strip()
                        ),
                        "num_doc_cluster_banks_nonempty": sum(1 for item in doc_cluster_banks if item["cluster_bank_text"]),
                        "num_layer1_bank_units": len(bank_units_all),
                        "num_merge_groups": num_merge_groups_value,
                        "num_merged_cluster_banks": num_merged_cluster_banks_value,
                        "summary_mode": summary_mode_value,
                        "stage_label": stage_label,
                        "per_doc_summary_budget_tokens": args.per_doc_summary_budget_tokens,
                        "aux_doc_summary_budget_tokens": args.aux_doc_summary_budget_tokens,
                        "doc_cluster_hard_max_banks": AUTO_CLUSTER_HARD_MAX_BANKS,
                        "doc_cluster_memory_budget_tokens": 0,
                        "doc_cluster_max_queries_per_bank": args.doc_cluster_max_queries_per_bank,
                        "doc_cluster_style": args.doc_cluster_style,
                        "max_doc_tokens": args.max_doc_tokens,
                        "doc_truncate_strategy": args.doc_truncate_strategy,
                        "doc_truncations": doc_truncations,
                        "summary_truncations": summary_truncations,
                        "doc_summaries": doc_summaries,
                        "doc_cluster_banks": doc_cluster_banks,
                        "bank_units_all": bank_units_all,
                        "merge_plan_raw": merge_plan_raw_value,
                        "merge_groups": merge_groups_value,
                        "merged_cluster_banks": merged_cluster_banks_value,
                        "memory_text": memory_text_value,
                        "memory_tokens": counter.count(memory_text_value),
                        "row_failed": bool(runtime_error_value or answer_error_value),
                        "model_answer": final_answer_value,
                        "runtime_error": runtime_error_value,
                        "answer_error": answer_error_value,
                        "update_lm_usage": summary_usage_value,
                        "summary_lm_usage": summary_usage_value,
                        "answer_lm_usage": answer_usage_value,
                        "total_lm_calls": total_lm_calls,
                        "total_lm_input_tokens": total_lm_input_tokens,
                        "total_lm_output_tokens": total_lm_output_tokens,
                        "total_lm_tokens": total_lm_input_tokens + total_lm_output_tokens,
                        "summary_lm_wall_time_sec": summary_usage_value["wall_time_sec"],
                        "answer_lm_wall_time_sec": answer_usage_value["wall_time_sec"],
                        "total_lm_wall_time_sec": round(total_lm_wall_time_sec, 6),
                        "runtime_sec": round(time.time() - started, 3),
                        "dry_run": bool(args.dry_run),
                    }
                    row_metadata.attach_sample_metadata(out_row, row)
                    return out_row

                if args.summary_mode in MERGE2_SUMMARY_MODES:
                    if args.summary_mode == "per_doc_structured_cluster_banks_merge2":
                        layer1_method_name, layer1_variant_name = method_variant_names(
                            "per_doc_structured_cluster_banks",
                            args.per_doc_summary_budget_tokens,
                            args.doc_cluster_memory_budget_tokens,
                            args.doc_cluster_style,
                            args.aux_doc_summary_budget_tokens,
                        )
                        layer1_summary_mode_value = "per_doc_structured_cluster_banks"
                    elif args.summary_mode == DOCSUMMARYAUX_SUMMARY_MODE:
                        layer1_method_name = f"oracle_doc_cluster_bank_docsummaryaux_concat_{args.doc_cluster_style}"
                        layer1_variant_name = f"{layer1_method_name}_Nauto_Mfree"
                        layer1_summary_mode_value = DOCSUMMARYAUX_SUMMARY_MODE
                    else:
                        layer1_method_name, layer1_variant_name = method_variant_names(
                            "per_doc_full_cluster_banks",
                            args.per_doc_summary_budget_tokens,
                            args.doc_cluster_memory_budget_tokens,
                            args.doc_cluster_style,
                            args.aux_doc_summary_budget_tokens,
                        )
                        layer1_summary_mode_value = "per_doc_full_cluster_banks"
                    layer1_out_row = build_out_row(
                        method_value=layer1_method_name,
                        variant_value=layer1_variant_name,
                        summary_mode_value=layer1_summary_mode_value,
                        memory_text_value=layer1_memory_text,
                        runtime_error_value=construction_error,
                        answer_error_value=layer1_answer_error,
                        final_answer_value=layer1_final_answer,
                        summary_usage_value=layer1_summary_usage,
                        answer_usage_value=layer1_answer_usage,
                        merge_plan_raw_value="",
                        merge_groups_value=[],
                        merged_cluster_banks_value=[],
                        stage_label="layer1_concat",
                        num_merge_groups_value=0,
                        num_merged_cluster_banks_value=0,
                    )
                    if need_layer1_output and layer1_fout:
                        layer1_fout.write(json.dumps(layer1_out_row, ensure_ascii=False) + "\n")
                        layer1_fout.flush()
                        layer1_done_ids.add(qid)
                    if need_layer1_rlm_output and layer1_rlm_fout:
                        if construction_error:
                            layer1_rlm_out_row = {
                                "question_id": qid,
                                "question": question,
                                "gold_answer": gold_answer,
                                "variant": rlm_docpairs_mod.VARIANT_NAME if rlm_docpairs_mod else "rlm_official_docpair_filesystem_from_docsummaryaux",
                                "method": rlm_docpairs_mod.METHOD_NAME if rlm_docpairs_mod else "rlm_official_docpair_filesystem_from_docsummaryaux",
                                "docsummaryaux_source_method": layer1_method_name,
                                "docsummaryaux_source_variant": layer1_variant_name,
                                "docsummaryaux_results_jsonl": str(layer1_out_path) if layer1_out_path else "",
                                "model": rlm_model,
                                "backend": rlm_backend,
                                "max_depth": args.rlm_max_depth,
                                "max_iterations": args.rlm_max_iterations,
                                "model_answer": "",
                                "raw_model_answer": "",
                                "answer_extraction_mode": "",
                                "answer_extraction_error": "",
                                "is_exact_match": False,
                                "row_failed": True,
                                "latency_sec": 0.0,
                                "runtime_error": f"docsummaryaux_source_generation_failed: {construction_error}",
                                "skip_reason": "docsummaryaux_source_generation_failed",
                                "augmentation_mode": "filesystem_doc_pairs",
                                "num_stream_docs": len(docs),
                                "num_context_docs_loaded": 0,
                                "context_doc_ids": [],
                                "source_context_doc_ids": [],
                                "num_raw_doc_files": 0,
                                "num_companion_doc_files": 0,
                                "num_total_docpair_files": 0,
                                "docpair_files": [],
                                "docpair_unmatched_source_doc_ids": [],
                                "doc_truncate_strategy": args.doc_truncate_strategy,
                                "max_doc_tokens": args.max_doc_tokens,
                                "doc_char_truncations": 0,
                                "doc_token_truncations": 0,
                                "stream_doc_tokens": [counter.count(str(d.get("text", ""))) for d in docs],
                                "stream_total_tokens": sum(counter.count(str(d.get("text", ""))) for d in docs),
                                "num_source_doc_summaries_nonempty": 0,
                                "num_doc_cluster_banks_nonempty": 0,
                                "rlm_usage": {},
                                "rlm_usage_totals": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                                "total_lm_calls": 0,
                                "total_lm_input_tokens": 0,
                                "total_lm_output_tokens": 0,
                                "total_lm_tokens": 0,
                                "lm_call_wall_time_sec": 0.0,
                                "total_lm_wall_time_sec": 0.0,
                                "answer_tokens": 0,
                                "memory_text": "",
                                "memory_tokens": 0,
                                "execute_calls": 0,
                                "memory_state_present": False,
                                "dry_run": bool(args.dry_run),
                            }
                            row_metadata.attach_sample_metadata(layer1_rlm_out_row, row)
                        else:
                            layer1_rlm_out_row = answer_with_rlm_docpairs(
                                qid_value=qid,
                                question_text=question,
                                gold_answer_text=gold_answer,
                                docs_value=docs,
                                doc_cluster_banks_value=doc_cluster_banks,
                                source_method_value=layer1_method_name,
                                source_variant_value=layer1_variant_name,
                                dataset_row=row,
                            )
                        layer1_rlm_fout.write(json.dumps(layer1_rlm_out_row, ensure_ascii=False) + "\n")
                        layer1_rlm_fout.flush()
                        layer1_rlm_done_ids.add(qid)

                out_row = build_out_row(
                    method_value=method_name,
                    variant_value=variant_name,
                    summary_mode_value=args.summary_mode,
                    memory_text_value=merged_memory_text if args.summary_mode in MERGE2_SUMMARY_MODES else layer1_memory_text,
                    runtime_error_value=(construction_error or merge_error) if args.summary_mode in MERGE2_SUMMARY_MODES else construction_error,
                    answer_error_value=merged_answer_error if args.summary_mode in MERGE2_SUMMARY_MODES else layer1_answer_error,
                    final_answer_value=merged_final_answer if args.summary_mode in MERGE2_SUMMARY_MODES else layer1_final_answer,
                    summary_usage_value=merged_summary_usage if args.summary_mode in MERGE2_SUMMARY_MODES else layer1_summary_usage,
                    answer_usage_value=merged_answer_usage if args.summary_mode in MERGE2_SUMMARY_MODES else layer1_answer_usage,
                    merge_plan_raw_value=merge_plan_raw if args.summary_mode in MERGE2_SUMMARY_MODES else "",
                    merge_groups_value=merge_groups if args.summary_mode in MERGE2_SUMMARY_MODES else [],
                    merged_cluster_banks_value=merged_cluster_banks if args.summary_mode in MERGE2_SUMMARY_MODES else [],
                    stage_label="layer2_merged" if args.summary_mode in MERGE2_SUMMARY_MODES else "single_stage",
                    num_merge_groups_value=len(merge_groups) if args.summary_mode in MERGE2_SUMMARY_MODES else 0,
                    num_merged_cluster_banks_value=len(merged_cluster_banks) if args.summary_mode in MERGE2_SUMMARY_MODES else 0,
                )

                if need_merged_output:
                    fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    fout.flush()
                    merged_done_ids.add(qid)

                processed += 1
                if args.progress_every > 0 and processed % args.progress_every == 0:
                    if args.summary_mode in {"per_doc_full_cluster_banks", "per_doc_structured_cluster_banks"}:
                        print(
                            f"[progress] processed={processed} last_qid={qid} "
                            f"nonempty_doc_cluster_banks={out_row['num_doc_cluster_banks_nonempty']} "
                            f"memory_tokens={out_row['memory_tokens']}",
                            flush=True,
                        )
                    elif args.summary_mode in MERGE2_SUMMARY_MODES:
                        print(
                            f"[progress] processed={processed} last_qid={qid} "
                            f"layer1_banks={out_row['num_layer1_bank_units']} "
                            f"merged_banks={out_row['num_merged_cluster_banks']} "
                            f"memory_tokens={out_row['memory_tokens']}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[progress] processed={processed} last_qid={qid} "
                            f"nonempty_doc_summaries={out_row['num_doc_summaries_nonempty']} "
                            f"memory_tokens={out_row['memory_tokens']}",
                            flush=True,
                        )
        finally:
            if layer1_fout:
                layer1_fout.close()
            if layer1_rlm_fout:
                layer1_rlm_fout.close()
            if trace_file:
                trace_file.close()
            if layer1_rlm_out_path and not args.dry_run and rlm_base_mod is not None:
                rlm_base_mod.uninstall_rlm_backend_call_timer(rlm_backend)

    if args.dry_run:
        print("[done] dry-run complete.", flush=True)
    else:
        totals = base.aggregate_output_totals(out_path, skip_answer=args.skip_answer)
        print(
            "[done] "
            f"summary_calls={totals['update_calls']} summary_tokens_in={totals['update_input_tokens']} "
            f"summary_tokens_out={totals['update_output_tokens']} "
            f"answer_calls={totals['answer_calls']} answer_tokens_in={totals['answer_input_tokens']} "
            f"answer_tokens_out={totals['answer_output_tokens']} "
            f"summary_lm_wall_time_sec={totals['update_wall_time_sec']:.3f} "
            f"answer_lm_wall_time_sec={totals['answer_wall_time_sec']:.3f} "
            f"total_lm_wall_time_sec={totals['total_lm_wall_time_sec']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
