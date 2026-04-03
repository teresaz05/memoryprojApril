#!/usr/bin/env python3
"""Dynamic bank-with-summary utilities"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import tiktoken
from dotenv import load_dotenv
from april_version_code.common import metadata as row_metadata
try:
    from google import genai
except ImportError:
    genai = None


QUERY_GENERATION_WARM_PROMPT = """You are generating candidate user questions from a warm-start document prefix.

You are given WARM_START_DOCUMENTS (the first z streamed documents).
Generate exactly NUM_CANDIDATES diverse, concrete, factual questions that a user might ask.

Rules:
1. Use only information grounded in WARM_START_DOCUMENTS.
2. Questions must be specific and answer-oriented (avoid vague or stylistic prompts).
3. Prefer questions requiring concrete entities, dates, numbers, titles, places, or explicit relations.
4. Maximize diversity across candidates (different entities/events/angles).
5. Avoid duplicates and near-duplicates.
6. Keep each question concise and well-formed.
7. Generate information-seeking questions only, not instructions, summaries, or meta-prompts.
8. Do not try to infer any hidden target question; propose plausible future user questions only from the observed evidence.

Output format:
- Return STRICT JSON only.
- Use exactly this schema:
  {{
    "queries": ["q1", "q2", "..."]
  }}
- "queries" must have exactly NUM_CANDIDATES strings.
- No markdown, no prose, no extra keys.

NUM_CANDIDATES:
{num_candidates}

WARM_START_DOCUMENTS:
{warm_documents}
"""


QUERY_GENERATION_DYNAMIC_PROMPT = """You are improving a dynamic query bank for downstream answering.

You are given:
- CURRENT_QUERIES,
- CURRENT_MEMORY_BANKS,
- SUMMARY_MEMORY_BANK (query-agnostic running summary),
- NEW_DOCUMENT_CHUNK.

Generate exactly NUM_CANDIDATES candidate replacement queries.

Requirements:
1. Queries must be grounded in CURRENT_MEMORY_BANKS and/or SUMMARY_MEMORY_BANK and/or NEW_DOCUMENT_CHUNK.
2. Queries should be more useful and evidence-targeted than weak CURRENT_QUERIES.
3. Do not output near-duplicates/paraphrases of CURRENT_QUERIES.
4. Each query must differ in a significant way from CURRENT_QUERIES (different entity, relation, scope, disambiguation, or evidence angle).
5. Keep questions factual, specific, and concise.
6. Maximize diversity across candidates.
7. Generate information-seeking questions only, not instructions, summaries, or meta-prompts.
8. If current evidence contains authorship/source ambiguity, preserve that distinction instead of collapsing it into generic questions.
9. Do not try to infer any hidden target question; propose plausible future user questions only from the observed evidence.

Output format:
- Return STRICT JSON only.
- Use exactly:
  {{
    "queries": ["q1", "q2", "..."]
  }}
- "queries" must contain exactly NUM_CANDIDATES strings.
- No markdown, no prose, no extra keys.

CURRENT_QUERIES:
{current_queries}

CURRENT_MEMORY_BANKS:
{current_memory_banks}

SUMMARY_MEMORY_BANK:
{summary_memory_bank}

NEW_DOCUMENT_CHUNK:
{new_document_chunk}

NUM_CANDIDATES:
{num_candidates}
"""


QUERY_GENERATION_REPAIR_PROMPT = """The previous output was invalid or had too few valid questions.

You must return STRICT JSON in this schema only:
{{
  "queries": ["..."]
}}

Requirements:
1. Return exactly NUM_ADDITIONAL_CANDIDATES additional questions.
2. Do NOT repeat any question in EXISTING_QUERIES or CURRENT_QUERIES.
3. Questions must be grounded in CONTEXT_BLOCK.
4. Questions must be specific, factual, diverse, and materially different from CURRENT_QUERIES.
5. Return JSON only. No markdown, no prose, no extra keys.
6. Generate information-seeking questions only, not instructions, summaries, or meta-prompts.

NUM_ADDITIONAL_CANDIDATES:
{num_candidates}

EXISTING_QUERIES:
{existing_queries}

CURRENT_QUERIES:
{current_queries}

CONTEXT_BLOCK:
{context_block}
"""


NEW_QUERY_FULL_REFRESH_PROMPT = """You are initializing memory for a newly introduced query after a dynamic query-bank update.

Task:
- Build memory for TARGET_QUERY using:
  1) OLD_MEMORY_BANKS_SNAPSHOT (all prior banks before update),
  2) SUMMARY_MEMORY_BANK,
  3) NEW_DOCUMENT_CHUNK (newly streamed docs).

Rules:
1. Use only provided content; no outside knowledge.
2. Keep only information relevant or plausibly relevant to TARGET_QUERY.
3. Prefer concrete evidence (entities, dates, numbers, titles, locations, explicit relations).
4. Remove redundancy and low-value details.
5. If conflicts exist, keep attributed alternatives separate.
6. Sort facts by importance for TARGET_QUERY.
7. Put the most answer-critical facts first.
8. Do not output absence-style statements unless they are themselves target evidence.
9. If the answer may hinge on who said, wrote, authored, published, discovered, or attributed something, preserve that attribution explicitly.

Output:
- Plain text memory only.
- No JSON, no markdown, no preamble.

TARGET_QUERY:
{target_query}

SOFT_MEMORY_TARGET_TOKENS:
{memory_budget_tokens}

OLD_MEMORY_BANKS_SNAPSHOT:
{old_memory_banks}

SUMMARY_MEMORY_BANK:
{summary_memory_bank}

NEW_DOCUMENT_CHUNK:
{new_document_chunk}
"""


INIT_MEMORY_PROMPT = """You are building a bounded memory bank for one target query from warm-start documents.

Objective:
- Produce a concise memory that will help answer TARGET_QUERY later.

Rules:
1. Use only WARM_START_DOCUMENTS; do not use outside knowledge.
2. Keep only information relevant or plausibly relevant to TARGET_QUERY.
3. Prefer concrete evidence: entities, dates, numbers, titles, locations, explicit relations.
4. Remove low-value details and redundancy.
5. If evidence conflicts, keep conflicting claims as separate attributed entries (do not collapse).
6. Sort retained facts in strict descending importance for TARGET_QUERY.
7. Place the most answer-critical facts first.
8. Do not output absence-style statements unless they are themselves target evidence.
9. If the answer may hinge on who said, wrote, authored, published, discovered, or attributed something, preserve that attribution explicitly.

Output:
- Plain text memory only (no JSON, no markdown, no preamble).
- Concise, evidence-rich statements.

TARGET_QUERY:
{target_query}

SOFT_MEMORY_TARGET_TOKENS:
{memory_budget_tokens}

WARM_START_DOCUMENTS:
{warm_documents}
"""


REFRESH_MEMORY_PROMPT = """You are updating one bounded query-specific memory bank as new streamed documents arrive.

Objective:
- Update CURRENT_MEMORY for TARGET_QUERY using NEW_DOCUMENT_CHUNK.

Rules:
1. Use only CURRENT_MEMORY, SUMMARY_MEMORY_BANK, and NEW_DOCUMENT_CHUNK; no outside knowledge.
2. Keep information relevant or plausibly relevant to TARGET_QUERY.
3. Preserve previously retained critical facts unless NEW_DOCUMENT_CHUNK provides stronger corrective evidence.
4. Prefer concrete evidence: entities, dates, numbers, titles, places, explicit relations.
5. Remove redundancy and low-value details.
6. If conflicts exist, keep separate attributed alternatives instead of collapsing.
7. Sort retained facts in strict descending importance for TARGET_QUERY.
8. Put the most answer-critical facts first.
9. Do not output absence-style statements unless they are target evidence.
10. If NEW_DOCUMENT_CHUNK adds no useful stronger evidence, keep CURRENT_MEMORY mostly unchanged.
11. If the answer may hinge on who said, wrote, authored, published, discovered, or attributed something, preserve that attribution explicitly.

Output:
- Plain text memory only (no JSON, no markdown, no preamble).
- Concise, evidence-rich statements.

TARGET_QUERY:
{target_query}

SOFT_MEMORY_TARGET_TOKENS:
{memory_budget_tokens}

CURRENT_MEMORY:
{current_memory}

SUMMARY_MEMORY_BANK:
{summary_memory_bank}

NEW_DOCUMENT_CHUNK:
{new_document_chunk}
"""


SUMMARY_REWRITE_PROMPT = """You are an evidence-grounded memory optimizer for query-agnostic streaming QA.

You maintain one bounded memory bank while documents arrive over time.

Primary objective:
- Maximize future answerability across many possible factual questions that may be asked later.

Hard constraints:
1. Use only CURRENT_MEMORY and NEW_DOCUMENT_CHUNK. Do not use outside knowledge.
2. Keep only facts that are likely useful for future QA (direct facts, definitions, entity relations, time facts, numeric facts).
3. Prefer concrete, checkable details: names, entities, dates, numbers, titles, locations, and explicit relationships.
4. Remove redundancy and low-value details aggressively.
5. Handle conflicts with explicit tie-breaks:
   - Prefer direct explicit evidence over inference.
   - Prefer more specific evidence over generic evidence.
   - Prefer internally consistent high-support facts over weak single-mention clues.
   - If a conflict cannot be resolved, keep both but mark one as primary and the other as alternate.
6. If NEW_DOCUMENT_CHUNK adds no useful information, keep memory concise and stable.
7. Order output by general QA utility: most reusable high-value facts first.
8. Keep output concise and dense, but optimize retained utility first.
9. SOFT_MEMORY_TARGET_TOKENS is guidance, not a hard target.
10. Do NOT drop previously retained critical facts unless memory is well over budget and those facts are clearly lower-priority than stronger evidence.
11. If memory has room, retain useful high-value information rather than over-compressing.
12. Never store absence-style statements as memory facts (for example: "not mentioned", "unknown", "not provided in this document", "cannot determine from this document"), unless they are themselves explicit factual claims.
13. Stability preference: if NEW_DOCUMENT_CHUNK does not add clearly stronger or corrective evidence for a retained fact, keep the corresponding CURRENT_MEMORY content unchanged. Minor cleanup (deduplication, concise wording) is allowed, but avoid unnecessary rewrites.
14. When multiple documents attribute facts differently or identify different authors/speakers/origins, preserve those source-specific attributions explicitly instead of collapsing them into one merged statement.

Output requirements:
- Output plain text memory only (no JSON, no markdown fences, no preamble).
- Put the most reusable high-value facts at the beginning.
- Sort entries in strict descending importance for future QA utility.
- Keep concise, evidence-rich statements.
- Do not include explicit gap-tracking sections.

SOFT_MEMORY_TARGET_TOKENS:
{memory_budget_tokens}

CURRENT_MEMORY:
{current_memory}

NEW_DOCUMENT_CHUNK:
{new_document_chunk}
"""


ANSWER_FROM_BANK_PROMPT = """You are answering TARGET_QUERY using selected query-specific memory banks.

Rules:
1. Use only SELECTED_MEMORY_BANKS content.
2. Do not use outside knowledge.
3. Provide your best-supported final answer from the provided memories.
4. If multiple candidate facts conflict, prefer the one with strongest direct evidence and specific attribution.
5. Return only one short final answer string.
6. No explanation, no markdown, no bullets, no prefixes.

TARGET_QUERY:
{target_query}

SELECTED_MEMORY_BANKS:
{memory_banks}

FINAL_ANSWER:
"""


@dataclass
class RetryPolicy:
    retries: int = 5
    initial_backoff_sec: float = 3.0
    max_backoff_sec: float = 45.0


class TokenCounter:
    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.enc = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self.enc.encode(text or "", disallowed_special=()))

    def truncate(self, text: str, max_tokens: int, strategy: str = "head") -> str:
        if max_tokens <= 0:
            return ""
        toks = self.enc.encode(text or "", disallowed_special=())
        if len(toks) <= max_tokens:
            return text
        if strategy == "tail":
            keep = toks[-max_tokens:]
        elif strategy == "middle":
            front = max_tokens // 2
            back = max_tokens - front
            keep = toks[:front] + toks[-back:]
        else:
            keep = toks[:max_tokens]
        return self.enc.decode(keep)


class GeminiClient:
    def __init__(
        self,
        model: str,
        retry_policy: RetryPolicy,
        timeout_sec: int,
    ) -> None:
        if genai is None:
            raise RuntimeError("google-genai is required for the Gemini backend but is not installed.")
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in environment.")
        self.client = genai.Client(api_key=api_key, http_options={"timeout": int(timeout_sec * 1000)})
        self.model = model
        self.retry_policy = retry_policy

        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_wall_time_sec = 0.0

    def _update_usage(self, resp: Any) -> None:
        usage = getattr(resp, "usage_metadata", None)
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        candidate_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        self.total_calls += 1
        self.total_input_tokens += prompt_tokens
        self.total_output_tokens += candidate_tokens

    @staticmethod
    def _extract_text(resp: Any) -> str:
        txt = (getattr(resp, "text", None) or "").strip()
        if txt:
            return txt
        candidates = getattr(resp, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) or []
            chunks: List[str] = []
            for part in parts:
                chunk = getattr(part, "text", None)
                if chunk:
                    chunks.append(str(chunk))
            joined = "\n".join(c for c in chunks if c.strip()).strip()
            if joined:
                return joined
        return ""

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        backoff = self.retry_policy.initial_backoff_sec
        last_error: Optional[Exception] = None
        for attempt in range(self.retry_policy.retries + 1):
            try:
                call_started = time.perf_counter()
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={"temperature": temperature},
                )
                self.total_wall_time_sec += max(0.0, time.perf_counter() - call_started)
                self._update_usage(resp)
                text = self._extract_text(resp)
                if text:
                    return text
                if attempt < self.retry_policy.retries:
                    time.sleep(min(backoff, self.retry_policy.max_backoff_sec))
                    backoff *= 2
                    continue
                raise RuntimeError("Model returned empty response after all retries.")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.retry_policy.retries:
                    time.sleep(min(backoff, self.retry_policy.max_backoff_sec))
                    backoff *= 2
                    continue
                break
        raise RuntimeError(f"Generation failed after retries: {last_error}")


class OpenRouterClient:
    """OpenRouter chat-completions client for query generation."""

    def __init__(
        self,
        model: str,
        retry_policy: RetryPolicy,
        timeout_sec: int,
        token_counter: TokenCounter,
        base_url: str = "https://openrouter.ai/api/v1",
        http_referer: str = "",
        app_title: str = "",
    ) -> None:
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Set OPENROUTER_API_KEY in environment for openrouter backend.")
        self.api_key = api_key
        self.model = model
        self.retry_policy = retry_policy
        self.timeout_sec = int(timeout_sec)
        self.token_counter = token_counter
        self.base_url = base_url.rstrip("/")
        self.http_referer = http_referer.strip()
        self.app_title = app_title.strip()

        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_wall_time_sec = 0.0

    def _request_once(self, prompt: str, temperature: float) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
            "reasoning": {"effort": "none", "exclude": True},
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_title:
            headers["X-Title"] = self.app_title

        req = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            data = resp.read().decode("utf-8")
        return json.loads(data)

    def _content_to_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            parts: List[str] = []
            for key in ("text", "content", "value", "output_text"):
                part = self._content_to_text(value.get(key))
                if part:
                    parts.append(part)
            return "\n".join(parts).strip()
        if isinstance(value, list):
            parts = [self._content_to_text(item) for item in value]
            return "\n".join([p for p in parts if p]).strip()
        return ""

    def _extract_text(self, resp: Dict[str, Any]) -> str:
        choices = resp.get("choices") or []
        for choice in choices:
            msg = choice.get("message") or {}
            for candidate in (
                msg.get("content"),
                choice.get("text"),
                choice.get("output_text"),
                choice.get("delta"),
            ):
                text = self._content_to_text(candidate)
                if text:
                    return text
        top_level = self._content_to_text(resp.get("output_text"))
        if top_level:
            return top_level
        return ""

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        backoff = self.retry_policy.initial_backoff_sec
        last_error: Optional[Exception] = None
        for attempt in range(self.retry_policy.retries + 1):
            try:
                call_started = time.perf_counter()
                resp = self._request_once(prompt, temperature=temperature)
                self.total_wall_time_sec += max(0.0, time.perf_counter() - call_started)
                text = self._extract_text(resp)
                if not text:
                    raise RuntimeError("OpenRouter returned empty assistant content.")

                usage = resp.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                completion_tokens = int(usage.get("completion_tokens") or 0)
                # Fallback when usage is absent.
                if prompt_tokens <= 0:
                    prompt_tokens = self.token_counter.count(prompt)
                if completion_tokens <= 0:
                    completion_tokens = self.token_counter.count(text)

                self.total_calls += 1
                self.total_input_tokens += prompt_tokens
                self.total_output_tokens += completion_tokens
                return text
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="ignore").strip()
                except Exception:
                    detail = ""
                detail_suffix = f"; body={detail[:280]}" if detail else ""
                last_error = RuntimeError(f"HTTP Error {exc.code}: {exc.reason}{detail_suffix}")
                # Free OpenRouter models often return persistent 429s; fail fast and let caller fallback.
                if int(getattr(exc, "code", 0) or 0) == 429:
                    break
                if attempt < self.retry_policy.retries:
                    time.sleep(min(backoff, self.retry_policy.max_backoff_sec))
                    backoff *= 2
                    continue
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.retry_policy.retries:
                    time.sleep(min(backoff, self.retry_policy.max_backoff_sec))
                    backoff *= 2
                    continue
                break
        raise RuntimeError(f"OpenRouter generation failed after retries: {last_error}")


class HFInstructClient:
    """Local Hugging Face instruct model client (token usage via proxy tokenizer counts)."""

    def __init__(
        self,
        model_name: str,
        retry_policy: RetryPolicy,
        max_new_tokens: int,
        device_map: str,
        token_counter: TokenCounter,
    ) -> None:
        self.model_name = model_name
        self.retry_policy = retry_policy
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.token_counter = token_counter
        try:
            from transformers import pipeline
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to import transformers for HF query-generation backend. "
                "Install with: pip install transformers"
            ) from exc

        try:
            self.pipe = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                device_map=device_map,
            )
            self.tokenizer = self.pipe.tokenizer
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load HF instruct model '{model_name}'. "
                "Check model name/access and local resources."
            ) from exc

        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_wall_time_sec = 0.0

    def _to_model_input(self, prompt: str) -> str:
        prompt = str(prompt or "")
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        backoff = self.retry_policy.initial_backoff_sec
        last_error: Optional[Exception] = None
        model_input = self._to_model_input(prompt)
        do_sample = bool(temperature and temperature > 0)

        for attempt in range(self.retry_policy.retries + 1):
            try:
                kwargs: Dict[str, Any] = {
                    "max_new_tokens": int(self.max_new_tokens),
                    "do_sample": do_sample,
                    "return_full_text": False,
                }
                if do_sample:
                    kwargs["temperature"] = float(max(0.1, temperature))
                call_started = time.perf_counter()
                outputs = self.pipe(model_input, **kwargs)
                self.total_wall_time_sec += max(0.0, time.perf_counter() - call_started)
                if not outputs:
                    raise RuntimeError("HF model returned empty generation list.")
                out = str(outputs[0].get("generated_text", "")).strip()

                self.total_calls += 1
                self.total_input_tokens += self.token_counter.count(model_input)
                self.total_output_tokens += self.token_counter.count(out)

                if out:
                    return out
                if attempt < self.retry_policy.retries:
                    time.sleep(min(backoff, self.retry_policy.max_backoff_sec))
                    backoff *= 2
                    continue
                raise RuntimeError("HF model returned empty response after retries.")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.retry_policy.retries:
                    time.sleep(min(backoff, self.retry_policy.max_backoff_sec))
                    backoff *= 2
                    continue
                break
        raise RuntimeError(f"HF generation failed after retries: {last_error}")


class QwenEmbedder:
    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to import sentence-transformers for Qwen embedding. "
                "Install with: pip install sentence-transformers"
            ) from exc
        try:
            self.model = SentenceTransformer(
                model_name,
                trust_remote_code=True,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load embedding model '{model_name}'. "
                "Check model availability and network/cache."
            ) from exc

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(vecs, dtype=np.float32)


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def is_completed_row(row: Dict[str, Any], skip_answer: bool) -> bool:
    if str(row.get("runtime_error", "")).strip():
        return False
    if bool(row.get("summary_update_failed", False)):
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
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(row.get("question_id", "")).strip()
            if qid and is_completed_row(row, skip_answer=skip_answer):
                done.add(qid)
    return done


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def aggregate_output_totals(path: Path, skip_answer: bool) -> Dict[str, Any]:
    stats = {
        "rows_written_total": 0,
        "rows_completed_total": 0,
        "rows_runtime_error_total": 0,
        "rows_summary_update_failed_total": 0,
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
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            stats["rows_written_total"] += 1
            if is_completed_row(row, skip_answer=skip_answer):
                stats["rows_completed_total"] += 1
            if str(row.get("runtime_error", "")).strip():
                stats["rows_runtime_error_total"] += 1
            if bool(row.get("summary_update_failed", False)):
                stats["rows_summary_update_failed_total"] += 1
            qg = row.get("query_gen_lm_usage") if isinstance(row.get("query_gen_lm_usage"), dict) else {}
            u = row.get("update_lm_usage") if isinstance(row.get("update_lm_usage"), dict) else {}
            a = row.get("answer_lm_usage") if isinstance(row.get("answer_lm_usage"), dict) else {}
            stats["query_gen_calls"] += _as_int(qg.get("calls"))
            stats["query_gen_input_tokens"] += _as_int(qg.get("input_tokens"))
            stats["query_gen_output_tokens"] += _as_int(qg.get("output_tokens"))
            stats["update_calls"] += _as_int(u.get("calls"))
            stats["update_input_tokens"] += _as_int(u.get("input_tokens"))
            stats["update_output_tokens"] += _as_int(u.get("output_tokens"))
            stats["answer_calls"] += _as_int(a.get("calls"))
            stats["answer_input_tokens"] += _as_int(a.get("input_tokens"))
            stats["answer_output_tokens"] += _as_int(a.get("output_tokens"))
            stats["query_gen_wall_time_sec"] += _as_float(qg.get("wall_time_sec"))
            stats["update_wall_time_sec"] += _as_float(u.get("wall_time_sec"))
            stats["answer_wall_time_sec"] += _as_float(a.get("wall_time_sec"))
            total_row_wall = row.get("total_lm_wall_time_sec")
            if total_row_wall is None:
                total_row_wall = (
                    _as_float(qg.get("wall_time_sec"))
                    + _as_float(u.get("wall_time_sec"))
                    + _as_float(a.get("wall_time_sec"))
                )
            stats["total_lm_wall_time_sec"] += _as_float(total_row_wall)
    return stats


def format_doc_for_prompt(doc: Dict[str, Any]) -> str:
    parts = [
        f"doc_id: {doc.get('doc_id', '')}",
        "text:",
        str(doc.get("text", "")),
    ]
    return "\n".join(parts)


def format_doc_chunk_for_prompt(docs_chunk: Sequence[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for i, doc in enumerate(docs_chunk, start=1):
        blocks.append(f"[DOC_{i}]\n{format_doc_for_prompt(doc)}")
    return "\n\n".join(blocks)


def chunk_docs(docs: Sequence[Dict[str, Any]], stride: int) -> List[List[Dict[str, Any]]]:
    if stride <= 0:
        raise ValueError("stride must be > 0")
    out: List[List[Dict[str, Any]]] = []
    i = 0
    while i < len(docs):
        out.append(list(docs[i : i + stride]))
        i += stride
    return out


def normalize_candidate_query(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"^\d+[\)\.\-\:\s]+", "", s)
    s = s.strip("`\"' ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_query_candidates(raw: str) -> List[str]:
    txt = (raw or "").strip()
    if not txt:
        return []
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*", "", txt.strip(), flags=re.IGNORECASE)
        txt = re.sub(r"\s*```$", "", txt.strip())

    candidates: List[str] = []
    obj: Any = None
    parse_ok = False
    try:
        obj = json.loads(txt)
        parse_ok = True
    except Exception:
        parse_ok = False

    if not parse_ok:
        for opener, closer in (("{", "}"), ("[", "]")):
            start = txt.find(opener)
            end = txt.rfind(closer)
            if start < 0 or end <= start:
                continue
            snippet = txt[start : end + 1]
            try:
                obj = json.loads(snippet)
                parse_ok = True
                break
            except Exception:
                continue

    if parse_ok:
        arr: List[Any]
        if isinstance(obj, dict):
            arr = obj.get("queries") or obj.get("candidates") or obj.get("questions") or []
        elif isinstance(obj, list):
            arr = obj
        else:
            arr = []
        for item in arr:
            q = normalize_candidate_query(str(item))
            if q:
                candidates.append(q)
    else:
        # fallback: line parsing
        lines = []
        for x in txt.splitlines():
            y = re.sub(r"^\s*[-*]\s+", "", x)
            y = re.sub(r"^\s*\d+[\)\.\-:]\s+", "", y)
            y = normalize_candidate_query(y)
            if y:
                lines.append(y)
        candidates = lines

    uniq: List[str] = []
    seen: Set[str] = set()
    for q in candidates:
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(q)
    return uniq


def build_fallback_queries(warm_docs: Sequence[Dict[str, Any]], needed: int) -> List[str]:
    out: List[str] = []
    for idx, doc in enumerate(warm_docs, start=1):
        did = str(doc.get("doc_id", "")).strip() or f"doc_{idx}"
        out.extend(
            [
                f"What is the main claim in document {did}?",
                f"Which people, places, or organizations are central in document {did}?",
                f"What dated events or numeric facts are stated in document {did}?",
            ]
        )
        if len(out) >= needed:
            break
    while len(out) < needed:
        n = len(out) + 1
        out.append(f"What key factual relationship is stated in the warm-start documents ({n})?")
    return out[:needed]


def generate_candidate_queries_warm(
    llm: Any,
    warm_docs: Sequence[Dict[str, Any]],
    num_candidates: int,
    temperature: float,
) -> List[str]:
    warm_docs_block = format_doc_chunk_for_prompt(warm_docs)
    prompt = QUERY_GENERATION_WARM_PROMPT.format(
        num_candidates=num_candidates,
        warm_documents=warm_docs_block if warm_docs_block else "(empty)",
    )
    raw = llm.generate(prompt, temperature=temperature)
    out = parse_query_candidates(raw)
    out = out[:num_candidates]
    if len(out) >= num_candidates:
        return out

    needed = num_candidates - len(out)
    repair_prompt = QUERY_GENERATION_REPAIR_PROMPT.format(
        num_candidates=needed,
        existing_queries=json.dumps(out, ensure_ascii=False),
        current_queries="[]",
        context_block=warm_docs_block if warm_docs_block else "(empty)",
    )
    repair_raw = llm.generate(repair_prompt, temperature=temperature)
    repair = parse_query_candidates(repair_raw)
    all_q = out + repair

    uniq: List[str] = []
    seen: Set[str] = set()
    for q in all_q:
        k = q.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(q)
        if len(uniq) >= num_candidates:
            break
    if len(uniq) < num_candidates:
        for q in build_fallback_queries(warm_docs, num_candidates):
            k = q.lower()
            if k in seen:
                continue
            seen.add(k)
            uniq.append(q)
            if len(uniq) >= num_candidates:
                break
    return uniq[:num_candidates]


def generate_candidate_queries_dynamic(
    llm: Any,
    current_queries: Sequence[str],
    current_memory_bank: Dict[str, str],
    summary_memory: str,
    docs_chunk: Sequence[Dict[str, Any]],
    num_candidates: int,
    temperature: float,
) -> List[str]:
    current_queries_block = json.dumps(list(current_queries), ensure_ascii=False)
    current_memory_block = build_answer_memory_blob(current_queries, current_memory_bank)
    chunk_block = format_doc_chunk_for_prompt(docs_chunk)
    prompt = QUERY_GENERATION_DYNAMIC_PROMPT.format(
        current_queries=current_queries_block,
        current_memory_banks=current_memory_block if current_memory_block else "(empty)",
        summary_memory_bank=summary_memory if summary_memory else "(empty)",
        new_document_chunk=chunk_block if chunk_block else "(empty)",
        num_candidates=num_candidates,
    )
    raw = llm.generate(prompt, temperature=temperature)
    out = parse_query_candidates(raw)[:num_candidates]
    if len(out) >= num_candidates:
        return out

    needed = num_candidates - len(out)
    repair_prompt = QUERY_GENERATION_REPAIR_PROMPT.format(
        num_candidates=needed,
        existing_queries=json.dumps(out, ensure_ascii=False),
        current_queries=current_queries_block,
        context_block="\n\n".join(
            [
                f"CURRENT_MEMORY_BANKS:\n{current_memory_block if current_memory_block else '(empty)'}",
                f"SUMMARY_MEMORY_BANK:\n{summary_memory if summary_memory else '(empty)'}",
                f"NEW_DOCUMENT_CHUNK:\n{chunk_block if chunk_block else '(empty)'}",
            ]
        ),
    )
    repair_raw = llm.generate(repair_prompt, temperature=temperature)
    repair = parse_query_candidates(repair_raw)
    all_q = out + repair

    uniq: List[str] = []
    seen: Set[str] = {q.lower() for q in current_queries}
    for q in all_q:
        k = q.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(q)
        if len(uniq) >= num_candidates:
            break
    if len(uniq) < num_candidates:
        for q in build_fallback_queries(docs_chunk, num_candidates):
            k = q.lower()
            if k in seen:
                continue
            seen.add(k)
            uniq.append(q)
            if len(uniq) >= num_candidates:
                break
    return uniq[:num_candidates]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # embeddings are normalized in QwenEmbedder, but keep safe fallback
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def unique_normalized_queries(candidates: Sequence[str]) -> List[str]:
    uniq: List[str] = []
    seen: Set[str] = set()
    for q in candidates:
        qn = normalize_candidate_query(q)
        if not qn:
            continue
        key = qn.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(qn)
    return uniq


def score_candidates_by_similarity(
    embedder: QwenEmbedder,
    target_query: str,
    candidates: Sequence[str],
) -> List[Tuple[str, float]]:
    uniq = unique_normalized_queries(candidates)
    if not uniq:
        return []
    vecs = embedder.embed([target_query] + uniq)
    qv = vecs[0]
    scored = [(uniq[i], cosine_sim(qv, vecs[i + 1])) for i in range(len(uniq))]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def top_n_by_similarity(
    embedder: QwenEmbedder,
    target_query: str,
    candidates: Sequence[str],
    n: int,
) -> List[Tuple[str, float]]:
    if n <= 0:
        return []
    scored = score_candidates_by_similarity(embedder, target_query, candidates)
    return scored[:n]


def truncate_to_budget(
    text: str,
    counter: TokenCounter,
    budget_tokens: int,
    truncate_strategy: str,
) -> Tuple[str, bool]:
    text = (text or "").strip()
    tokens = counter.count(text)
    if tokens <= budget_tokens:
        return text, False
    return counter.truncate(text, budget_tokens, strategy=truncate_strategy), True


def build_initial_memory(
    llm: GeminiClient,
    target_query: str,
    warm_docs: Sequence[Dict[str, Any]],
    budget_tokens: int,
    temperature: float,
) -> str:
    prompt = INIT_MEMORY_PROMPT.format(
        target_query=target_query,
        memory_budget_tokens=budget_tokens,
        warm_documents=format_doc_chunk_for_prompt(warm_docs) if warm_docs else "(empty)",
    )
    return llm.generate(prompt, temperature=temperature).strip()


def refresh_memory(
    llm: GeminiClient,
    target_query: str,
    current_memory: str,
    summary_memory: str,
    docs_chunk: Sequence[Dict[str, Any]],
    budget_tokens: int,
    temperature: float,
) -> str:
    prompt = REFRESH_MEMORY_PROMPT.format(
        target_query=target_query,
        memory_budget_tokens=budget_tokens,
        current_memory=current_memory if current_memory else "(empty)",
        summary_memory_bank=summary_memory if summary_memory else "(empty)",
        new_document_chunk=format_doc_chunk_for_prompt(docs_chunk) if docs_chunk else "(empty)",
    )
    out = llm.generate(prompt, temperature=temperature).strip()
    return out if out else current_memory


def full_refresh_new_query_memory(
    llm: Any,
    target_query: str,
    old_queries: Sequence[str],
    old_memory_banks: Dict[str, str],
    summary_memory: str,
    docs_chunk: Sequence[Dict[str, Any]],
    budget_tokens: int,
    temperature: float,
) -> str:
    prompt = NEW_QUERY_FULL_REFRESH_PROMPT.format(
        target_query=target_query,
        memory_budget_tokens=budget_tokens,
        old_memory_banks=build_answer_memory_blob(old_queries, old_memory_banks) or "(empty)",
        summary_memory_bank=summary_memory if summary_memory else "(empty)",
        new_document_chunk=format_doc_chunk_for_prompt(docs_chunk) if docs_chunk else "(empty)",
    )
    return llm.generate(prompt, temperature=temperature).strip()


def update_summary_memory(
    llm: GeminiClient,
    current_memory: str,
    docs_chunk: Sequence[Dict[str, Any]],
    budget_tokens: int,
    temperature: float,
) -> str:
    prompt = SUMMARY_REWRITE_PROMPT.format(
        memory_budget_tokens=budget_tokens,
        current_memory=current_memory if current_memory else "(empty)",
        new_document_chunk=format_doc_chunk_for_prompt(docs_chunk) if docs_chunk else "(empty)",
    )
    out = llm.generate(prompt, temperature=temperature).strip()
    return out if out else current_memory


def build_answer_memory_blob(
    selected_queries: Sequence[str],
    memory_bank: Dict[str, str],
) -> str:
    blocks: List[str] = []
    for i, q in enumerate(selected_queries, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[MEMORY_BANK_{i}]",
                    f"query: {q}",
                    "memory:",
                    memory_bank.get(q, ""),
                ]
            )
        )
    return "\n\n".join(blocks).strip()


CANONICAL_TRACE_SCHEMA_VERSION = "browsecompv2_canonical_trace_v1"


def infer_sample_id_from_path(path: Optional[Path]) -> str:
    if path is None:
        return ""
    for parent in [path.parent, *path.parents]:
        if re.fullmatch(r"sample_\d+", parent.name):
            return parent.name
    return ""


def normalize_scored_queries(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        query = normalize_candidate_query(str(item.get("query", "")))
        if not query:
            continue
        out.append(
            {
                "query": query,
                "score": float(item.get("score", 0.0) or 0.0),
            }
        )
    return out


def filter_scored_queries(
    scored_queries: Sequence[Dict[str, Any]],
    allowed_queries: Sequence[str],
) -> List[Dict[str, Any]]:
    allowed = {normalize_candidate_query(q).lower() for q in allowed_queries if normalize_candidate_query(q)}
    return [x for x in normalize_scored_queries(scored_queries) if x["query"].lower() in allowed]


def doc_id_list(docs: Sequence[Dict[str, Any]]) -> List[str]:
    return [str(d.get("doc_id", "")) for d in docs]


def serialize_memory_bank(
    memory_bank: Dict[str, str],
    selected_queries: Sequence[str],
    counter: TokenCounter,
) -> Tuple[Dict[str, str], List[Dict[str, Any]], int]:
    ordered_bank: Dict[str, str] = {}
    per_query: List[Dict[str, Any]] = []
    total_tokens = 0
    for q in selected_queries:
        mem = str(memory_bank.get(q, "") or "")
        tok = counter.count(mem)
        ordered_bank[q] = mem
        per_query.append({"query": q, "memory_tokens": tok})
        total_tokens += tok
    return ordered_bank, per_query, int(total_tokens)


def build_canonical_doc_store_row(
    sample_id: str,
    question_id: str,
    question: str,
    gold_answer: str,
    docs: Sequence[Dict[str, Any]],
    z_warm_docs_config: int,
    z_warm_docs_effective: int,
    refresh_stride_docs: int,
    max_doc_tokens: int,
    doc_truncate_strategy: str,
    emitted_at_unix: float,
) -> Dict[str, Any]:
    stream_docs: List[Dict[str, Any]] = []
    for idx, doc in enumerate(docs):
        stream_docs.append(
            {
                "stream_index": idx,
                "doc_id": str(doc.get("doc_id", "")),
                "source_text": str(doc.get("source_text", "")),
                "pipeline_doc_text": str(doc.get("text", "")),
                "pipeline_prompt_text": format_doc_for_prompt(doc),
                "raw_doc_tokens_pipeline_basis": int(doc.get("raw_doc_tokens", 0) or 0),
                "doc_tokens_after_cap_pipeline_basis": int(doc.get("doc_tokens_after_cap", 0) or 0),
                "was_truncated": bool(doc.get("was_truncated", False)),
            }
        )
    return {
        "record_type": "canonical_doc_store",
        "schema_version": CANONICAL_TRACE_SCHEMA_VERSION,
        "sample_id": sample_id,
        "question_id": question_id,
        "question": question,
        "hidden_target_query": question,
        "gold_answer": gold_answer,
        "num_stream_docs": len(stream_docs),
        "z_warm_docs_config": z_warm_docs_config,
        "z_warm_docs_effective": z_warm_docs_effective,
        "refresh_stride_docs": refresh_stride_docs,
        "max_doc_tokens": max_doc_tokens,
        "doc_truncate_strategy": doc_truncate_strategy,
        "stream_docs": stream_docs,
        "emitted_at_unix": emitted_at_unix,
    }


def build_canonical_snapshot_base(
    sample_id: str,
    qid: str,
    question: str,
    gold_answer: str,
    timestep: int,
    phase: str,
    num_stream_docs: int,
    docs_seen: int,
    prefix_docs: Sequence[Dict[str, Any]],
    new_chunk_docs: Sequence[Dict[str, Any]],
    remaining_docs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "record_type": "canonical_timestep_snapshot",
        "schema_version": CANONICAL_TRACE_SCHEMA_VERSION,
        "sample_id": sample_id,
        "question_id": qid,
        "question": question,
        "hidden_target_query": question,
        "gold_answer": gold_answer,
        "timepoint": timestep,
        "phase": phase,
        "num_stream_docs": num_stream_docs,
        "docs_seen": docs_seen,
        "document_prefix_doc_ids": doc_id_list(prefix_docs),
        "document_prefix_doc_count": len(prefix_docs),
        "new_chunk_doc_ids": doc_id_list(new_chunk_docs),
        "new_chunk_doc_count": len(new_chunk_docs),
        "remaining_doc_ids": doc_id_list(remaining_docs),
        "remaining_doc_count": len(remaining_docs),
    }


def selection_metric_target_field_name(selection_metric: str) -> str:
    if selection_metric == "memory_vs_gold_answer":
        return "memory_vs_gold_answer_similarity"
    return "query_similarity"


def load_existing_keys(path: Path, key_fields: Sequence[str]) -> Set[Tuple[str, ...]]:
    keys: Set[Tuple[str, ...]] = set()
    if not path.exists():
        return keys
    for row in iter_jsonl(path):
        key = tuple(str(row.get(field, "")).strip() for field in key_fields)
        if any(not part for part in key):
            continue
        keys.add(key)
    return keys


def write_jsonl_row(handle: Any, payload: Dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def flush_jsonl_handle(handle: Any) -> None:
    handle.flush()
    try:
        os.fsync(handle.fileno())
    except (AttributeError, OSError, ValueError):
        pass


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Oracle-assisted warm-start + dynamic query bank + summary bank streaming memory pipeline."
    )
    ap.add_argument("--dataset_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--trace_jsonl", default="")
    ap.add_argument(
        "--canonical_trace_jsonl",
        default="",
        help="Canonical per-timestep raw trace JSONL. Defaults to <trace_jsonl>_canonical.jsonl when --trace_jsonl is set.",
    )
    ap.add_argument(
        "--canonical_doc_store_jsonl",
        default="",
        help="Canonical per-question capped-doc store JSONL. Defaults to <trace_jsonl>_canonical_docs.jsonl when --trace_jsonl is set.",
    )
    ap.add_argument(
        "--sample_id",
        default="",
        help="Optional sample identifier recorded in canonical raw traces.",
    )
    ap.add_argument(
        "--disable_canonical_trace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable canonical raw trace/doc-store emission even when --trace_jsonl is set.",
    )

    ap.add_argument("--llm_backend", choices=["gemini", "openrouter"], default="openrouter")
    ap.add_argument("--model", default="qwen/qwen3.5-397b-a17b")
    ap.add_argument(
        "--query_gen_backend",
        choices=["gemini", "hf", "openrouter"],
        default="openrouter",
        help="Backend for candidate query generation model.",
    )
    ap.add_argument(
        "--query_gen_model",
        default="",
        help="Model name for query generation; defaults to --model for gemini backend.",
    )
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
    ap.add_argument("--summary_budget_tokens", type=int, default=1000)
    ap.add_argument("--z_warm_docs", type=int, default=1)
    ap.add_argument("--num_bank_queries", type=int, default=2)
    ap.add_argument("--answer_top_j", type=int, default=1)
    ap.add_argument("--candidate_multiplier", type=int, default=2)
    ap.add_argument("--selection_metric", choices=["query", "memory_vs_gold_answer"], default="query")
    ap.add_argument("--refresh_stride_docs", type=int, default=2)

    ap.add_argument(
        "--overflow_policy",
        choices=["truncate_only"],
        default="truncate_only",
    )
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
    ap.add_argument(
        "--log_query_selection_details",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When enabled, traces and result rows include full candidate-pool similarity details per timestep.",
    )
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_false", dest="resume")
    ap.add_argument("--skip_answer", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.memory_budget_tokens <= 0:
        raise ValueError("--memory_budget_tokens must be > 0.")
    if args.summary_budget_tokens <= 0:
        raise ValueError("--summary_budget_tokens must be > 0.")
    if args.z_warm_docs < 0:
        raise ValueError("--z_warm_docs must be >= 0.")
    if args.num_bank_queries <= 0:
        raise ValueError("--num_bank_queries must be > 0.")
    if args.answer_top_j <= 0:
        raise ValueError("--answer_top_j must be > 0.")
    if args.answer_top_j > args.num_bank_queries:
        raise ValueError("--answer_top_j must be <= --num_bank_queries.")
    if args.candidate_multiplier < 1:
        raise ValueError("--candidate_multiplier must be >= 1.")
    if args.selection_metric not in {"query", "memory_vs_gold_answer"}:
        raise ValueError("--selection_metric is invalid.")
    if args.refresh_stride_docs <= 0:
        raise ValueError("--refresh_stride_docs must be > 0.")
    if args.max_doc_tokens < 0:
        raise ValueError("--max_doc_tokens must be >= 0.")
    if args.max_docs_per_query < 0:
        raise ValueError("--max_docs_per_query must be >= 0.")
    if args.retries < 0:
        raise ValueError("--retries must be >= 0.")
    if args.timeout_sec <= 0:
        raise ValueError("--timeout_sec must be > 0.")
    if args.progress_every < 0:
        raise ValueError("--progress_every must be >= 0.")
    if args.start_index < 0:
        raise ValueError("--start_index must be >= 0.")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0.")
    if args.embed_batch_size <= 0:
        raise ValueError("--embed_batch_size must be > 0.")
    if args.query_gen_hf_max_new_tokens <= 0:
        raise ValueError("--query_gen_hf_max_new_tokens must be > 0.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    load_dotenv()

    dataset_path = Path(args.dataset_jsonl)
    out_path = Path(args.out_jsonl)
    trace_path = Path(args.trace_jsonl) if args.trace_jsonl else None
    canonical_trace_path = Path(args.canonical_trace_jsonl) if args.canonical_trace_jsonl else None
    canonical_doc_store_path = Path(args.canonical_doc_store_jsonl) if args.canonical_doc_store_jsonl else None
    if args.disable_canonical_trace:
        canonical_trace_path = None
        canonical_doc_store_path = None
    elif trace_path and canonical_trace_path is None:
        canonical_trace_path = trace_path.with_name(trace_path.stem + "_canonical.jsonl")
    if not args.disable_canonical_trace and trace_path and canonical_doc_store_path is None:
        canonical_doc_store_path = trace_path.with_name(trace_path.stem + "_canonical_docs.jsonl")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset jsonl not found: {dataset_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_path:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
    if canonical_trace_path:
        canonical_trace_path.parent.mkdir(parents=True, exist_ok=True)
    if canonical_doc_store_path:
        canonical_doc_store_path.parent.mkdir(parents=True, exist_ok=True)

    rows_all = list(iter_jsonl(dataset_path))
    rows = rows_all[args.start_index :]
    if args.limit > 0:
        rows = rows[: args.limit]
    run_started = time.time()

    done_ids = load_done_ids(out_path, skip_answer=args.skip_answer) if args.resume else set()
    mode = "a" if args.resume else "w"
    sample_id = args.sample_id or infer_sample_id_from_path(out_path) or infer_sample_id_from_path(trace_path)
    existing_canonical_doc_keys = (
        load_existing_keys(canonical_doc_store_path, ["sample_id", "question_id"])
        if args.resume and canonical_doc_store_path
        else set()
    )

    counter = TokenCounter("cl100k_base")
    retry_policy = RetryPolicy(retries=args.retries)
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
    embedder: Optional[QwenEmbedder] = None

    if not args.dry_run:
        if args.query_gen_backend == "gemini":
            llm_query_gen = GeminiClient(
                model=query_gen_model_name,
                retry_policy=retry_policy,
                timeout_sec=args.timeout_sec,
            )
        elif args.query_gen_backend == "hf":
            llm_query_gen = HFInstructClient(
                model_name=query_gen_model_name,
                retry_policy=retry_policy,
                max_new_tokens=args.query_gen_hf_max_new_tokens,
                device_map=args.query_gen_hf_device_map,
                token_counter=counter,
            )
        else:
            llm_query_gen = OpenRouterClient(
                model=query_gen_model_name,
                retry_policy=retry_policy,
                timeout_sec=args.timeout_sec,
                token_counter=counter,
                base_url=args.openrouter_base_url,
                http_referer=args.openrouter_http_referer,
                app_title=args.openrouter_app_title,
            )
        if args.llm_backend == "gemini":
            llm_update = GeminiClient(
                model=args.model,
                retry_policy=retry_policy,
                timeout_sec=args.timeout_sec,
            )
        else:
            llm_update = OpenRouterClient(
                model=args.model,
                retry_policy=retry_policy,
                timeout_sec=args.timeout_sec,
                token_counter=counter,
                base_url=args.openrouter_base_url,
                http_referer=args.openrouter_http_referer,
                app_title=args.openrouter_app_title,
            )
        embedder = QwenEmbedder(
            model_name=args.embed_model,
            device=args.embed_device,
            batch_size=args.embed_batch_size,
        )
        if not args.skip_answer:
            answer_model = args.answer_model or args.model
            if args.llm_backend == "gemini":
                llm_answer = GeminiClient(
                    model=answer_model,
                    retry_policy=retry_policy,
                    timeout_sec=args.timeout_sec,
                )
            else:
                llm_answer = OpenRouterClient(
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
        canonical_trace_file = canonical_trace_path.open(mode, encoding="utf-8") if canonical_trace_path else None
        canonical_doc_store_file = canonical_doc_store_path.open(mode, encoding="utf-8") if canonical_doc_store_path else None
        try:
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
                row_attempt_started_unix = started
                canonical_rows: List[Dict[str, Any]] = []
                canonical_doc_store_row: Optional[Dict[str, Any]] = None

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

                # 1) Prepare capped streamed docs
                capped_docs: List[Dict[str, Any]] = []
                doc_truncations = 0
                for doc in docs:
                    raw_doc_text = format_doc_for_prompt(doc)
                    raw_doc_tokens = counter.count(raw_doc_text)
                    doc_text = raw_doc_text
                    was_truncated = False
                    if args.max_doc_tokens > 0 and raw_doc_tokens > args.max_doc_tokens:
                        doc_text = counter.truncate(
                            raw_doc_text,
                            max_tokens=args.max_doc_tokens,
                            strategy=args.doc_truncate_strategy,
                        )
                        doc_truncations += 1
                        was_truncated = True
                    capped_docs.append(
                        {
                            "doc_id": str(doc.get("doc_id", "")),
                            "source_text": str(doc.get("text", "")),
                            "text": doc_text,
                            "raw_doc_tokens": raw_doc_tokens,
                            "doc_tokens_after_cap": counter.count(doc_text),
                            "was_truncated": was_truncated,
                        }
                    )

                z_eff = min(args.z_warm_docs, len(capped_docs))
                warm_docs = capped_docs[:z_eff]
                remaining_docs = capped_docs[z_eff:]
                chunks = chunk_docs(remaining_docs, args.refresh_stride_docs)
                if args.selection_metric == "query":
                    selection_metric_target_text = question
                else:
                    selection_metric_target_text = gold_answer
                selection_objective_field = selection_metric_target_field_name(args.selection_metric)
                canonical_doc_store_row = build_canonical_doc_store_row(
                    sample_id=sample_id,
                    question_id=qid,
                    question=question,
                    gold_answer=gold_answer,
                    docs=capped_docs,
                    z_warm_docs_config=args.z_warm_docs,
                    z_warm_docs_effective=z_eff,
                    refresh_stride_docs=args.refresh_stride_docs,
                    max_doc_tokens=args.max_doc_tokens,
                    doc_truncate_strategy=args.doc_truncate_strategy,
                    emitted_at_unix=started,
                )

                # 2) Maintain a separate null-style summary memory bank across seen docs.
                summary_memory = ""
                summary_update_errors: List[str] = []
                summary_update_failed = False
                summary_overflow_truncate_events = 0
                if warm_docs:
                    summary_init_error = ""
                    summary_init_was_truncated = False
                    try:
                        if args.dry_run:
                            warm_blob = format_doc_chunk_for_prompt(warm_docs)
                            summary_candidate = counter.truncate(
                                warm_blob,
                                max(1, min(args.summary_budget_tokens, 512)),
                                strategy="head",
                            )
                        else:
                            if llm_update is None:
                                raise RuntimeError("Update model is not initialized.")
                            summary_candidate = update_summary_memory(
                                llm=llm_update,
                                current_memory="",
                                docs_chunk=warm_docs,
                                budget_tokens=args.summary_budget_tokens,
                                temperature=args.update_temperature,
                            )
                        summary_candidate, summary_init_was_truncated = truncate_to_budget(
                            summary_candidate,
                            counter=counter,
                            budget_tokens=args.summary_budget_tokens,
                            truncate_strategy=args.truncate_strategy,
                        )
                        if summary_init_was_truncated:
                            summary_overflow_truncate_events += 1
                        summary_memory = summary_candidate
                    except Exception as exc:  # noqa: BLE001
                        summary_init_error = str(exc)
                        summary_update_failed = True
                        summary_update_errors.append(f"summary_init: {exc}")
                    if trace_file:
                        trace_file.write(
                            json.dumps(
                                {
                                    "question_id": qid,
                                    "phase": "summary_init",
                                    "chunk_size_docs": len(warm_docs),
                                    "summary_tokens": counter.count(summary_memory),
                                    "summary_budget_tokens": args.summary_budget_tokens,
                                    "was_truncated": bool(summary_init_was_truncated),
                                    "step_error": summary_init_error,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        flush_jsonl_handle(trace_file)

                # 3) Generate initial 2*N candidate queries from warm docs
                num_candidates = max(args.num_bank_queries, args.candidate_multiplier * args.num_bank_queries)
                candidate_queries: List[str] = []
                candidate_pool_scores_initial: List[Dict[str, Any]] = []
                selected_queries: List[str] = []
                selected_query_scores: List[Dict[str, Any]] = []
                initial_selected_queries: List[str] = []
                initial_selected_query_scores: List[Dict[str, Any]] = []
                query_update_events: List[Dict[str, Any]] = []
                query_update_count = 0
                query_keep_count = 0
                query_replace_count = 0
                reintroduced_query_count = 0
                new_query_full_refresh_count = 0
                dropped_query_cache: Dict[str, str] = {}
                query_gen_fallback_used = False

                try:
                    if args.dry_run:
                        candidate_queries = build_fallback_queries(warm_docs, num_candidates)
                        dry_pool = unique_normalized_queries(candidate_queries)
                        candidate_pool_scores_initial = [
                            {"query": q, "score": float(1.0 - (i * 0.01))}
                            for i, q in enumerate(dry_pool)
                        ]
                        selected_queries = dry_pool[: args.num_bank_queries]
                        selected_query_scores = [
                            {"query": q, "score": float(1.0 - (i * 0.01))}
                            for i, q in enumerate(selected_queries)
                        ]
                    else:
                        if llm_query_gen is None or embedder is None:
                            raise RuntimeError("Query-gen model/embedder not initialized.")
                        candidate_queries = generate_candidate_queries_warm(
                            llm=llm_query_gen,
                            warm_docs=warm_docs,
                            num_candidates=num_candidates,
                            temperature=args.query_gen_temperature,
                        )
                        scored_all = score_candidates_by_similarity(
                            embedder=embedder,
                            target_query=selection_metric_target_text,
                            candidates=candidate_queries,
                        )
                        candidate_pool_scores_initial = [
                            {"query": q, "score": float(s)} for q, s in scored_all
                        ]
                        scored = scored_all[: args.num_bank_queries]
                        selected_queries = [q for q, _ in scored]
                        selected_query_scores = [
                            {"query": q, "score": float(s)} for q, s in scored
                        ]
                        if len(selected_queries) < args.num_bank_queries:
                            seen = {q.lower() for q in selected_queries}
                            for q in candidate_queries:
                                k = q.lower()
                                if k in seen:
                                    continue
                                selected_queries.append(q)
                                selected_query_scores.append({"query": q, "score": 0.0})
                                seen.add(k)
                                if len(selected_queries) >= args.num_bank_queries:
                                    break
                        if len(selected_queries) < args.num_bank_queries:
                            for q in build_fallback_queries(warm_docs, args.num_bank_queries):
                                k = q.lower()
                                if k in {x.lower() for x in selected_queries}:
                                    continue
                                selected_queries.append(q)
                                selected_query_scores.append({"query": q, "score": 0.0})
                                if len(selected_queries) >= args.num_bank_queries:
                                    break
                    initial_selected_queries = list(selected_queries)
                    initial_selected_query_scores = list(selected_query_scores)
                except Exception as exc:  # noqa: BLE001
                    update_errors.append(f"query_bank_build: {exc}")
                    if not selected_queries:
                        query_gen_fallback_used = True
                        fallback_qs = build_fallback_queries(warm_docs, args.num_bank_queries)
                        selected_queries = fallback_qs[: args.num_bank_queries]
                        selected_query_scores = [{"query": q, "score": 0.0} for q in selected_queries]
                        candidate_queries = list(candidate_queries) or fallback_qs
                        initial_selected_queries = list(selected_queries)
                        initial_selected_query_scores = list(selected_query_scores)
                    if not selected_queries:
                        runtime_error = "query_generation_or_selection_failed"

                if trace_file and args.log_query_selection_details:
                    trace_file.write(
                        json.dumps(
                            {
                                "question_id": qid,
                                "phase": "warm_query_selection",
                                "timestep": 0,
                                "docs_seen": z_eff,
                                "warm_doc_ids": [str(d.get("doc_id", "")) for d in warm_docs],
                                "num_candidates_requested": num_candidates,
                                "num_candidates_generated": len(candidate_queries),
                                "candidate_queries": candidate_queries,
                                "candidate_pool_scores": candidate_pool_scores_initial,
                                "selected_queries": list(selected_queries),
                                "selected_query_scores": list(selected_query_scores),
                                "query_gen_fallback_used": bool(query_gen_fallback_used),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    flush_jsonl_handle(trace_file)

                # 3) Build per-query memory banks (init from warm docs)
                memory_bank: Dict[str, str] = {q: "" for q in selected_queries}
                overflow_truncate_events = 0
                overflow_compress_calls = 0  # fixed truncate-only method

                if not runtime_error:
                    # init from warm docs
                    for q in selected_queries:
                        try:
                            if args.dry_run:
                                warm_blob = format_doc_chunk_for_prompt(warm_docs)
                                init_mem = counter.truncate(warm_blob, 256, strategy="head")
                            else:
                                if llm_update is None:
                                    raise RuntimeError("Update model is not initialized.")
                                init_mem = build_initial_memory(
                                    llm=llm_update,
                                    target_query=q,
                                    warm_docs=warm_docs,
                                    budget_tokens=args.memory_budget_tokens,
                                    temperature=args.update_temperature,
                                )
                            init_mem, was_truncated = truncate_to_budget(
                                init_mem,
                                counter=counter,
                                budget_tokens=args.memory_budget_tokens,
                                truncate_strategy=args.truncate_strategy,
                            )
                            if was_truncated:
                                overflow_truncate_events += 1
                            memory_bank[q] = init_mem
                            if trace_file:
                                trace_file.write(
                                    json.dumps(
                                        {
                                            "question_id": qid,
                                            "phase": "init_memory",
                                            "query": q,
                                            "memory_tokens": counter.count(init_mem),
                                            "memory_budget_tokens": args.memory_budget_tokens,
                                            "was_truncated": was_truncated,
                                        },
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                                flush_jsonl_handle(trace_file)
                        except Exception as exc:  # noqa: BLE001
                            runtime_error = "init_memory_failed"
                            update_errors.append(f"init_query={q}: {exc}")
                            break

                warm_memory_bank_after, warm_memory_tokens_after, warm_query_memory_tokens_after = serialize_memory_bank(
                    memory_bank,
                    selected_queries,
                    counter,
                )
                canonical_rows.append(
                    {
                        **build_canonical_snapshot_base(
                            sample_id=sample_id,
                            qid=qid,
                            question=question,
                            gold_answer=gold_answer,
                            timestep=0,
                            phase="warm",
                            num_stream_docs=len(capped_docs),
                            docs_seen=z_eff,
                            prefix_docs=warm_docs,
                            new_chunk_docs=warm_docs,
                            remaining_docs=remaining_docs,
                        ),
                        "oracle_selection_source": "hidden_target_query_embedding",
                        "num_candidates_requested": num_candidates,
                        "num_candidates_generated": len(candidate_queries),
                        "query_gen_fallback_used": bool(query_gen_fallback_used),
                        "current_queries_before": [],
                        "current_query_scores_before": [],
                        "current_memory_banks_before": {},
                        "current_memory_tokens_by_query_before": [],
                        "query_memory_tokens_total_before": 0,
                        "summary_memory_for_query_gen": summary_memory,
                        "summary_memory_tokens_for_query_gen": counter.count(summary_memory),
                        "summary_update_error": summary_init_error if warm_docs else "",
                        "generated_candidate_queries": list(candidate_queries),
                        "generated_candidate_scores": normalize_scored_queries(candidate_pool_scores_initial),
                        "selection_pool_ranked_queries": normalize_scored_queries(candidate_pool_scores_initial),
                        "selected_queries_after": list(selected_queries),
                        "selected_query_scores_after": normalize_scored_queries(selected_query_scores),
                        "query_set_changed": True,
                        "memory_banks_after": warm_memory_bank_after,
                        "memory_tokens_by_query_after": warm_memory_tokens_after,
                        "query_memory_tokens_total_after": warm_query_memory_tokens_after,
                        "combined_memory_tokens_after": int(warm_query_memory_tokens_after + counter.count(summary_memory)),
                        "chunk_size_docs": len(warm_docs),
                    }
                )

                if not runtime_error:
                    # dynamic update/refresh after every chunk of 2 docs (or last 1)
                    docs_seen_so_far = z_eff
                    for chunk_idx, docs_chunk in enumerate(chunks, start=1):
                        docs_seen_so_far += len(docs_chunk)
                        chunk_doc_ids = [str(d.get("doc_id", "")) for d in docs_chunk]
                        old_queries = list(selected_queries)
                        old_query_scores = normalize_scored_queries(selected_query_scores)
                        old_memory_snapshot = {q: memory_bank.get(q, "") for q in old_queries}
                        old_memory_bank_serialized, old_memory_tokens_serialized, old_query_memory_tokens_total = serialize_memory_bank(
                            old_memory_snapshot,
                            old_queries,
                            counter,
                        )
                        summary_step_error = ""
                        summary_was_truncated = False
                        summary_before = summary_memory
                        try:
                            if args.dry_run:
                                chunk_blob = format_doc_chunk_for_prompt(docs_chunk)
                                summary_candidate = (
                                    (summary_memory + "\n\n" + counter.truncate(chunk_blob, 256, strategy="head")).strip()
                                    if summary_memory
                                    else counter.truncate(chunk_blob, 256, strategy="head")
                                )
                            else:
                                if llm_update is None:
                                    raise RuntimeError("Update model is not initialized.")
                                summary_candidate = update_summary_memory(
                                    llm=llm_update,
                                    current_memory=summary_memory,
                                    docs_chunk=docs_chunk,
                                    budget_tokens=args.summary_budget_tokens,
                                    temperature=args.update_temperature,
                                )
                            summary_candidate, summary_was_truncated = truncate_to_budget(
                                summary_candidate,
                                counter=counter,
                                budget_tokens=args.summary_budget_tokens,
                                truncate_strategy=args.truncate_strategy,
                            )
                            if summary_was_truncated:
                                summary_overflow_truncate_events += 1
                            summary_memory = summary_candidate
                        except Exception as exc:  # noqa: BLE001
                            summary_step_error = str(exc)
                            summary_update_failed = True
                            summary_update_errors.append(f"summary_refresh_chunk={chunk_idx}: {exc}")
                            summary_memory = summary_before
                        if trace_file:
                            trace_file.write(
                                json.dumps(
                                    {
                                        "question_id": qid,
                                        "phase": "summary_refresh",
                                        "chunk_idx": chunk_idx,
                                        "chunk_size_docs": len(docs_chunk),
                                        "summary_tokens": counter.count(summary_memory),
                                        "summary_budget_tokens": args.summary_budget_tokens,
                                        "was_truncated": bool(summary_was_truncated),
                                        "step_error": summary_step_error,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            flush_jsonl_handle(trace_file)
                        new_candidates: List[str] = []
                        new_selected_queries: List[str] = []
                        new_selected_scores: List[Dict[str, Any]] = []
                        selection_pool_scores: List[Dict[str, Any]] = []
                        query_gen_fallback_used_step = False
                        try:
                            if args.dry_run:
                                new_candidates = build_fallback_queries(docs_chunk, num_candidates)
                                scored_pool = old_queries + new_candidates
                                dry_pool = unique_normalized_queries(scored_pool)
                                selection_pool_scores = [
                                    {"query": q, "score": float(1.0 - (i * 0.01))}
                                    for i, q in enumerate(dry_pool)
                                ]
                                new_selected_queries = dry_pool[: args.num_bank_queries]
                                new_selected_scores = [
                                    {"query": q, "score": float(1.0 - (i * 0.01))}
                                    for i, q in enumerate(new_selected_queries)
                                ]
                            else:
                                if llm_query_gen is None or embedder is None:
                                    raise RuntimeError("Query-gen model/embedder not initialized.")
                                new_candidates = generate_candidate_queries_dynamic(
                                    llm=llm_query_gen,
                                    current_queries=old_queries,
                                    current_memory_bank=old_memory_snapshot,
                                    summary_memory=summary_memory,
                                    docs_chunk=docs_chunk,
                                    num_candidates=num_candidates,
                                    temperature=args.query_gen_temperature,
                                )
                                scored_pool = old_queries + new_candidates
                                scored_all = score_candidates_by_similarity(
                                    embedder=embedder,
                                    target_query=selection_metric_target_text,
                                    candidates=scored_pool,
                                )
                                selection_pool_scores = [
                                    {"query": q, "score": float(s)} for q, s in scored_all
                                ]
                                scored = scored_all[: args.num_bank_queries]
                                new_selected_queries = [q for q, _ in scored]
                                new_selected_scores = [{"query": q, "score": float(s)} for q, s in scored]
                                if len(new_selected_queries) < args.num_bank_queries:
                                    seen = {q.lower() for q in new_selected_queries}
                                    for q in scored_pool:
                                        k = q.lower()
                                        if k in seen:
                                            continue
                                        seen.add(k)
                                        new_selected_queries.append(q)
                                        new_selected_scores.append({"query": q, "score": 0.0})
                                        if len(new_selected_queries) >= args.num_bank_queries:
                                            break
                                if len(new_selected_queries) < args.num_bank_queries:
                                    for q in build_fallback_queries(docs_chunk, args.num_bank_queries):
                                        k = q.lower()
                                        if k in {x.lower() for x in new_selected_queries}:
                                            continue
                                        new_selected_queries.append(q)
                                        new_selected_scores.append({"query": q, "score": 0.0})
                                        if len(new_selected_queries) >= args.num_bank_queries:
                                            break
                        except Exception as exc:  # noqa: BLE001
                            update_errors.append(f"query_bank_rebuild_chunk={chunk_idx}: {exc}")
                            query_gen_fallback_used = True
                            query_gen_fallback_used_step = True
                            new_candidates = []
                            new_selected_queries = list(old_queries)
                            prev_score_map: Dict[str, float] = {}
                            for x in selected_query_scores:
                                q_key = str(x.get("query", ""))
                                score_raw = x.get("score", 0.0)
                                try:
                                    prev_score_map[q_key] = float(score_raw)
                                except Exception:
                                    prev_score_map[q_key] = 0.0
                            new_selected_scores = [
                                {"query": q, "score": float(prev_score_map.get(q, 0.0))}
                                for q in new_selected_queries
                            ]
                            selection_pool_scores = [
                                {"query": q, "score": float(prev_score_map.get(q, 0.0))}
                                for q in unique_normalized_queries(old_queries)
                            ]

                        old_query_set = set(old_queries)
                        new_query_set = set(new_selected_queries)
                        query_set_changed = old_query_set != new_query_set
                        if query_set_changed:
                            query_update_count += 1
                        else:
                            query_keep_count += 1
                        query_replace_count += sum(1 for q in old_queries if q not in new_query_set)

                        for q in old_queries:
                            if q not in new_query_set:
                                dropped_query_cache[q] = old_memory_snapshot.get(q, "")

                        if trace_file:
                            trace_payload: Dict[str, Any] = {
                                "question_id": qid,
                                "phase": "query_bank_reselection",
                                "chunk_idx": chunk_idx,
                                "timestep": chunk_idx,
                                "chunk_size_docs": len(docs_chunk),
                                "docs_seen": docs_seen_so_far,
                                "chunk_doc_ids": chunk_doc_ids,
                                "query_set_changed": bool(query_set_changed),
                                "num_new_candidates": len(new_candidates),
                                "new_selected_queries": new_selected_queries,
                                "new_selected_query_scores": new_selected_scores,
                                "summary_tokens": counter.count(summary_memory),
                                "summary_update_error": summary_step_error,
                            }
                            if args.log_query_selection_details:
                                trace_payload.update(
                                    {
                                        "old_selected_queries": old_queries,
                                        "generated_candidates": new_candidates,
                                        "selection_pool_scores": selection_pool_scores,
                                    }
                                )
                            trace_file.write(json.dumps(trace_payload, ensure_ascii=False) + "\n")
                            flush_jsonl_handle(trace_file)

                        if query_set_changed:
                            updated_memory_bank: Dict[str, str] = {}
                            for q in new_selected_queries:
                                try:
                                    if q in old_query_set:
                                        prev = old_memory_snapshot.get(q, "")
                                        if args.dry_run:
                                            chunk_blob = format_doc_chunk_for_prompt(docs_chunk)
                                            candidate = (prev + "\n\n" + counter.truncate(chunk_blob, 256, strategy="head")).strip()
                                        else:
                                            if llm_update is None:
                                                raise RuntimeError("Update model is not initialized.")
                                            candidate = refresh_memory(
                                                llm=llm_update,
                                                target_query=q,
                                                current_memory=prev,
                                                summary_memory=summary_memory,
                                                docs_chunk=docs_chunk,
                                                budget_tokens=args.memory_budget_tokens,
                                                temperature=args.update_temperature,
                                            )
                                    elif q in dropped_query_cache:
                                        reintroduced_query_count += 1
                                        prev = dropped_query_cache.get(q, "")
                                        if args.dry_run:
                                            chunk_blob = format_doc_chunk_for_prompt(docs_chunk)
                                            candidate = (prev + "\n\n" + counter.truncate(chunk_blob, 256, strategy="head")).strip()
                                        else:
                                            if llm_update is None:
                                                raise RuntimeError("Update model is not initialized.")
                                            candidate = refresh_memory(
                                                llm=llm_update,
                                                target_query=q,
                                                current_memory=prev,
                                                summary_memory=summary_memory,
                                                docs_chunk=docs_chunk,
                                                budget_tokens=args.memory_budget_tokens,
                                                temperature=args.update_temperature,
                                            )
                                    else:
                                        new_query_full_refresh_count += 1
                                        if args.dry_run:
                                            old_blob = build_answer_memory_blob(old_queries, old_memory_snapshot)
                                            chunk_blob = format_doc_chunk_for_prompt(docs_chunk)
                                            candidate = counter.truncate((old_blob + "\n\n" + chunk_blob).strip(), 256, strategy="head")
                                        else:
                                            if llm_update is None:
                                                raise RuntimeError("Update model is not initialized.")
                                            candidate = full_refresh_new_query_memory(
                                                llm=llm_update,
                                                target_query=q,
                                                old_queries=old_queries,
                                                old_memory_banks=old_memory_snapshot,
                                                summary_memory=summary_memory,
                                                docs_chunk=docs_chunk,
                                                budget_tokens=args.memory_budget_tokens,
                                                temperature=args.update_temperature,
                                            )

                                    candidate, was_truncated = truncate_to_budget(
                                        candidate,
                                        counter=counter,
                                        budget_tokens=args.memory_budget_tokens,
                                        truncate_strategy=args.truncate_strategy,
                                    )
                                    if was_truncated:
                                        overflow_truncate_events += 1
                                    updated_memory_bank[q] = candidate
                                    if trace_file:
                                        trace_file.write(
                                            json.dumps(
                                                {
                                                    "question_id": qid,
                                                    "phase": "dynamic_refresh_memory",
                                                    "chunk_idx": chunk_idx,
                                                    "chunk_size_docs": len(docs_chunk),
                                                    "query": q,
                                                    "memory_tokens": counter.count(candidate),
                                                    "memory_budget_tokens": args.memory_budget_tokens,
                                                    "was_truncated": was_truncated,
                                                },
                                                ensure_ascii=False,
                                            )
                                            + "\n"
                                        )
                                        flush_jsonl_handle(trace_file)
                                except Exception as exc:  # noqa: BLE001
                                    runtime_error = f"dynamic_refresh_failed_chunk_{chunk_idx}"
                                    update_errors.append(f"dynamic_refresh_chunk={chunk_idx} query={q}: {exc}")
                                    break
                            if runtime_error:
                                break
                            memory_bank = updated_memory_bank
                        else:
                            for q in old_queries:
                                try:
                                    prev = old_memory_snapshot.get(q, "")
                                    if args.dry_run:
                                        chunk_blob = format_doc_chunk_for_prompt(docs_chunk)
                                        candidate = (prev + "\n\n" + counter.truncate(chunk_blob, 256, strategy="head")).strip()
                                    else:
                                        if llm_update is None:
                                            raise RuntimeError("Update model is not initialized.")
                                        candidate = refresh_memory(
                                            llm=llm_update,
                                            target_query=q,
                                            current_memory=prev,
                                            summary_memory=summary_memory,
                                            docs_chunk=docs_chunk,
                                            budget_tokens=args.memory_budget_tokens,
                                            temperature=args.update_temperature,
                                        )
                                    candidate, was_truncated = truncate_to_budget(
                                        candidate,
                                        counter=counter,
                                        budget_tokens=args.memory_budget_tokens,
                                        truncate_strategy=args.truncate_strategy,
                                    )
                                    if was_truncated:
                                        overflow_truncate_events += 1
                                    memory_bank[q] = candidate
                                    if trace_file:
                                        trace_file.write(
                                            json.dumps(
                                                {
                                                    "question_id": qid,
                                                    "phase": "refresh_memory_no_query_set_change",
                                                    "chunk_idx": chunk_idx,
                                                    "chunk_size_docs": len(docs_chunk),
                                                    "query": q,
                                                    "memory_tokens": counter.count(candidate),
                                                    "memory_budget_tokens": args.memory_budget_tokens,
                                                    "was_truncated": was_truncated,
                                                },
                                                ensure_ascii=False,
                                            )
                                            + "\n"
                                        )
                                        flush_jsonl_handle(trace_file)
                                except Exception as exc:  # noqa: BLE001
                                    runtime_error = f"refresh_failed_chunk_{chunk_idx}"
                                    update_errors.append(f"refresh_chunk={chunk_idx} query={q}: {exc}")
                                    break
                            if runtime_error:
                                break

                        selected_queries = list(new_selected_queries)
                        selected_query_scores = list(new_selected_scores)
                        query_update_events.append(
                            {
                                "chunk_idx": chunk_idx,
                                "timestep": chunk_idx,
                                "docs_seen": docs_seen_so_far,
                                "chunk_doc_ids": chunk_doc_ids,
                                "query_set_changed": bool(query_set_changed),
                                "selected_queries": list(selected_queries),
                                "selected_query_scores": list(selected_query_scores),
                                "num_new_candidates": len(new_candidates),
                                "summary_tokens": counter.count(summary_memory),
                                "summary_update_error": summary_step_error,
                                "old_selected_queries": old_queries if args.log_query_selection_details else [],
                                "generated_candidates": new_candidates if args.log_query_selection_details else [],
                                "selection_pool_scores": selection_pool_scores if args.log_query_selection_details else [],
                            }
                        )
                        current_prefix_docs = capped_docs[:docs_seen_so_far]
                        future_docs = capped_docs[docs_seen_so_far:]
                        memory_bank_after_serialized, memory_tokens_after_serialized, query_memory_tokens_after_serialized = serialize_memory_bank(
                            memory_bank,
                            selected_queries,
                            counter,
                        )
                        canonical_rows.append(
                            {
                                **build_canonical_snapshot_base(
                                    sample_id=sample_id,
                                    qid=qid,
                                    question=question,
                                    gold_answer=gold_answer,
                                    timestep=chunk_idx,
                                    phase="update",
                                    num_stream_docs=len(capped_docs),
                                    docs_seen=docs_seen_so_far,
                                    prefix_docs=current_prefix_docs,
                                    new_chunk_docs=docs_chunk,
                                    remaining_docs=future_docs,
                                ),
                                "oracle_selection_source": "hidden_target_query_embedding",
                                "num_candidates_requested": num_candidates,
                                "num_candidates_generated": len(new_candidates),
                                "query_gen_fallback_used": bool(query_gen_fallback_used_step),
                                "current_queries_before": old_queries,
                                "current_query_scores_before": old_query_scores,
                                "current_memory_banks_before": old_memory_bank_serialized,
                                "current_memory_tokens_by_query_before": old_memory_tokens_serialized,
                                "query_memory_tokens_total_before": old_query_memory_tokens_total,
                                "summary_memory_before_chunk": summary_before,
                                "summary_memory_before_chunk_tokens": counter.count(summary_before),
                                "summary_memory_for_query_gen": summary_memory,
                                "summary_memory_tokens_for_query_gen": counter.count(summary_memory),
                                "summary_update_error": summary_step_error,
                                "summary_was_truncated": bool(summary_was_truncated),
                                "generated_candidate_queries": list(new_candidates),
                                "generated_candidate_scores": filter_scored_queries(selection_pool_scores, new_candidates),
                                "selection_pool_ranked_queries": normalize_scored_queries(selection_pool_scores),
                                "selected_queries_after": list(selected_queries),
                                "selected_query_scores_after": normalize_scored_queries(selected_query_scores),
                                "query_set_changed": bool(query_set_changed),
                                "memory_banks_after": memory_bank_after_serialized,
                                "memory_tokens_by_query_after": memory_tokens_after_serialized,
                                "query_memory_tokens_total_after": query_memory_tokens_after_serialized,
                                "combined_memory_tokens_after": int(
                                    query_memory_tokens_after_serialized + counter.count(summary_memory)
                                ),
                                "chunk_idx": chunk_idx,
                                "chunk_size_docs": len(docs_chunk),
                                "reintroduced_query_count_so_far": reintroduced_query_count,
                                "new_query_full_refresh_count_so_far": new_query_full_refresh_count,
                            }
                        )

                # 4) Select top-j bank queries for answer
                answer_top_j_effective = min(args.answer_top_j, len(selected_queries))
                answer_selected_queries: List[str] = []
                answer_selected_scores: List[Dict[str, Any]] = []

                if selected_queries:
                    if args.dry_run:
                        answer_selected_queries = selected_queries[:answer_top_j_effective]
                        answer_selected_scores = selected_query_scores[:answer_top_j_effective]
                    else:
                        if embedder is None:
                            raise RuntimeError("Embedder not initialized for answer selection.")
                        scored_j = top_n_by_similarity(
                            embedder=embedder,
                            target_query=question,
                            candidates=selected_queries,
                            n=answer_top_j_effective,
                        )
                        answer_selected_queries = [q for q, _ in scored_j]
                        answer_selected_scores = [{"query": q, "score": float(s)} for q, s in scored_j]

                # 5) Answer from selected memory banks
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
                        memory_banks_blob = build_answer_memory_blob(answer_selected_queries, memory_bank)
                        answer_prompt = ANSWER_FROM_BANK_PROMPT.format(
                            target_query=question,
                            memory_banks=memory_banks_blob if memory_banks_blob else "(empty)",
                        )
                        final_answer = llm_answer.generate(
                            answer_prompt,
                            temperature=args.answer_temperature,
                        ).strip()
                    except Exception as exc:  # noqa: BLE001
                        answer_error = str(exc)

                # 6) Usage + output row
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
                total_lm_input_tokens = (
                    query_gen_usage["input_tokens"] + update_usage["input_tokens"] + answer_usage["input_tokens"]
                )
                total_lm_output_tokens = (
                    query_gen_usage["output_tokens"] + update_usage["output_tokens"] + answer_usage["output_tokens"]
                )
                total_lm_wall_time_sec = (
                    query_gen_usage["wall_time_sec"]
                    + update_usage["wall_time_sec"]
                    + answer_usage["wall_time_sec"]
                )

                memory_tokens_by_query = {q: counter.count(memory_bank.get(q, "")) for q in selected_queries}
                query_memory_tokens_total = int(sum(memory_tokens_by_query.values()))
                summary_memory_tokens = counter.count(summary_memory)
                combined_memory_tokens = int(query_memory_tokens_total + summary_memory_tokens)
                memory_text = build_answer_memory_blob(selected_queries, memory_bank)

                sim_top1 = answer_selected_scores[0]["score"] if answer_selected_scores else None
                sim_topj_mean = (
                    float(sum(x["score"] for x in answer_selected_scores) / len(answer_selected_scores))
                    if answer_selected_scores
                    else None
                )

                out_row = {
                    "variant": f"oracle_assisted_warm_dynamic_bank_with_summary_N{args.num_bank_queries}_J{args.answer_top_j}",
                    "method": "oracle_assisted_warm_dynamic_bank_with_summary",
                    "question_id": qid,
                    "question": question,
                    "gold_answer": gold_answer,
                    "llm_backend": args.llm_backend,
                    "update_model": args.model,
                    "query_gen_backend": args.query_gen_backend,
                    "query_gen_model": query_gen_model_name if not args.dry_run else "",
                    "query_gen_fallback_used": bool(query_gen_fallback_used),
                    "answer_model": (args.answer_model or args.model) if not args.skip_answer else "",
                    "embed_model": args.embed_model if not args.dry_run else "",
                    "num_stream_docs": len(capped_docs),
                    "z_warm_docs": args.z_warm_docs,
                    "z_warm_docs_effective": z_eff,
                    "refresh_stride_docs": args.refresh_stride_docs,
                    "num_refresh_chunks": len(chunks),
                    "num_bank_queries": args.num_bank_queries,
                    "answer_top_j": args.answer_top_j,
                    "answer_top_j_effective": answer_top_j_effective,
                    "candidate_multiplier": args.candidate_multiplier,
                    "selection_metric": args.selection_metric,
                    "log_query_selection_details": bool(args.log_query_selection_details),
                    "num_generated_queries_initial": len(candidate_queries),
                    "candidate_queries_initial": candidate_queries,
                    "candidate_pool_scores_initial": candidate_pool_scores_initial,
                    "initial_selected_queries": initial_selected_queries,
                    "initial_selected_query_scores": initial_selected_query_scores,
                    "selected_queries": selected_queries,
                    "selected_query_scores": selected_query_scores,
                    "query_update_attempts": len(chunks),
                    "query_update_count": query_update_count,
                    "query_keep_count": query_keep_count,
                    "query_replace_count": query_replace_count,
                    "reintroduced_query_count": reintroduced_query_count,
                    "new_query_full_refresh_count": new_query_full_refresh_count,
                    "query_update_events": query_update_events,
                    "answer_selected_queries": answer_selected_queries,
                    "answer_selected_query_scores": answer_selected_scores,
                    "query_similarity": {
                        "answer_selected_top1": sim_top1,
                        "answer_selected_topj_mean": sim_topj_mean,
                    },
                    "memory_bank_by_query": memory_bank,
                    "memory_tokens_by_query": memory_tokens_by_query,
                    "memory_text": memory_text,
                    "query_memory_tokens_total": query_memory_tokens_total,
                    "summary_memory_text": summary_memory,
                    "summary_memory_tokens": summary_memory_tokens,
                    "combined_memory_tokens": combined_memory_tokens,
                    "memory_tokens": combined_memory_tokens,
                    "memory_budget_tokens": args.memory_budget_tokens,
                    "memory_budget_tokens_per_query": args.memory_budget_tokens,
                    "summary_budget_tokens": args.summary_budget_tokens,
                    "overflow_policy": args.overflow_policy,
                    "truncate_strategy": args.truncate_strategy,
                    "doc_truncate_strategy": args.doc_truncate_strategy,
                    "doc_truncations": doc_truncations,
                    "overflow_compress_calls": overflow_compress_calls,
                    "overflow_truncate_events": overflow_truncate_events,
                    "summary_overflow_truncate_events": summary_overflow_truncate_events,
                    "summary_update_failed": bool(summary_update_failed),
                    "summary_update_error_count": len(summary_update_errors),
                    "summary_update_errors": summary_update_errors,
                    "row_failed": bool(runtime_error or summary_update_failed),
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

                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                flush_jsonl_handle(fout)
                row_attempt_finished_unix = time.time()
                final_row_failed = bool(runtime_error or summary_update_failed)
                final_query_similarity = {
                    "answer_selected_top1": sim_top1,
                    "answer_selected_topj_mean": sim_topj_mean,
                }
                if canonical_doc_store_file and canonical_doc_store_row:
                    doc_key = (sample_id, qid)
                    if doc_key not in existing_canonical_doc_keys:
                        write_jsonl_row(canonical_doc_store_file, canonical_doc_store_row)
                        flush_jsonl_handle(canonical_doc_store_file)
                        existing_canonical_doc_keys.add(doc_key)
                if canonical_trace_file:
                    for canonical_row in canonical_rows:
                        canonical_row.update(
                            {
                                "memory_budget_tokens_per_query": args.memory_budget_tokens,
                                "summary_budget_tokens": args.summary_budget_tokens,
                                "num_bank_queries": args.num_bank_queries,
                                "answer_top_j": args.answer_top_j,
                                "candidate_multiplier": args.candidate_multiplier,
                                "selection_metric": args.selection_metric,
                                "refresh_stride_docs": args.refresh_stride_docs,
                                "overflow_policy": args.overflow_policy,
                                "truncate_strategy": args.truncate_strategy,
                                "doc_truncate_strategy": args.doc_truncate_strategy,
                                "max_doc_tokens": args.max_doc_tokens,
                                "doc_truncations_for_row": doc_truncations,
                                "llm_backend": args.llm_backend,
                                "update_model": args.model,
                                "query_gen_backend": args.query_gen_backend,
                                "query_gen_model": query_gen_model_name if not args.dry_run else "",
                                "answer_model": (args.answer_model or args.model) if not args.skip_answer else "",
                                "embed_model": args.embed_model if not args.dry_run else "",
                                "actual_final_selected_queries": list(answer_selected_queries),
                                "actual_final_selected_query_scores": normalize_scored_queries(answer_selected_scores),
                                "actual_final_query_similarity": final_query_similarity,
                                "actual_final_answer": final_answer,
                                "runtime_error": runtime_error,
                                "answer_error": answer_error,
                                "summary_update_failed": bool(summary_update_failed),
                                "summary_update_errors": list(summary_update_errors),
                                "row_failed": final_row_failed,
                                "query_gen_lm_usage": dict(query_gen_usage),
                                "update_lm_usage": dict(update_usage),
                                "answer_lm_usage": dict(answer_usage),
                                "total_lm_calls": total_lm_calls,
                                "total_lm_input_tokens": total_lm_input_tokens,
                                "total_lm_output_tokens": total_lm_output_tokens,
                                "total_lm_tokens": total_lm_input_tokens + total_lm_output_tokens,
                                "total_lm_wall_time_sec": round(total_lm_wall_time_sec, 6),
                                "row_attempt_started_unix": row_attempt_started_unix,
                                "row_attempt_finished_unix": row_attempt_finished_unix,
                                "row_runtime_sec": round(row_attempt_finished_unix - row_attempt_started_unix, 6),
                            }
                        )
                        write_jsonl_row(canonical_trace_file, canonical_row)
                    flush_jsonl_handle(canonical_trace_file)
                if runtime_error or answer_error or summary_update_failed:
                    first_update_error = update_errors[0] if update_errors else ""
                    first_summary_error = summary_update_errors[0] if summary_update_errors else ""
                    print(
                        f"[row_error] qid={qid} runtime_error={runtime_error or 'none'} "
                        f"answer_error={answer_error or 'none'} "
                        f"summary_update_failed={bool(summary_update_failed)} "
                        f"update_error={first_update_error or 'none'} "
                        f"summary_error={first_summary_error or 'none'}",
                        flush=True,
                    )
                if trace_file:
                    trace_file.write(
                        json.dumps(
                            {
                                "question_id": qid,
                                "phase": "summary",
                                "num_stream_docs": len(capped_docs),
                                "num_generated_queries_initial": len(candidate_queries),
                                "num_selected_queries": len(selected_queries),
                                "num_answer_selected_queries": len(answer_selected_queries),
                                "query_update_attempts": len(chunks),
                                "query_update_count": query_update_count,
                                "query_memory_tokens_total": query_memory_tokens_total,
                                "summary_memory_tokens": summary_memory_tokens,
                                "combined_memory_tokens": combined_memory_tokens,
                                "overflow_truncate_events": overflow_truncate_events,
                                "summary_overflow_truncate_events": summary_overflow_truncate_events,
                                "summary_update_failed": bool(summary_update_failed),
                                "runtime_error": runtime_error,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    flush_jsonl_handle(trace_file)
                processed += 1
                if qid:
                    done_ids.add(qid)
                if args.progress_every > 0 and processed % args.progress_every == 0:
                    sim_str = ""
                    if answer_selected_scores:
                        scores = [f"{x['score']:.3f}" for x in answer_selected_scores]
                        sim_str = f" sim_scores=[{','.join(scores)}]"
                    final_q_str = ""
                    if answer_selected_queries:
                        q0 = answer_selected_queries[0]
                        q0_short = (q0[:50] + "…") if len(q0) > 50 else q0
                        final_q_str = f" final_query=\"{q0_short}\""
                    print(
                        f"[progress] processed={processed} last_qid={qid} "
                        f"memory_tokens={combined_memory_tokens} "
                        f"(query={query_memory_tokens_total},summary={summary_memory_tokens}) "
                        f"query_updates={query_update_count}/{len(chunks)}{sim_str}{final_q_str}",
                        flush=True,
                    )
        finally:
            if trace_file:
                trace_file.close()
            if canonical_trace_file:
                canonical_trace_file.close()
            if canonical_doc_store_file:
                canonical_doc_store_file.close()

    query_gen_calls = llm_query_gen.total_calls if llm_query_gen else 0
    query_gen_in = llm_query_gen.total_input_tokens if llm_query_gen else 0
    query_gen_out = llm_query_gen.total_output_tokens if llm_query_gen else 0
    update_calls = llm_update.total_calls if llm_update else 0
    update_in = llm_update.total_input_tokens if llm_update else 0
    update_out = llm_update.total_output_tokens if llm_update else 0
    answer_calls = llm_answer.total_calls if llm_answer else 0
    answer_in = llm_answer.total_input_tokens if llm_answer else 0
    answer_out = llm_answer.total_output_tokens if llm_answer else 0
    query_gen_wall = llm_query_gen.total_wall_time_sec if llm_query_gen else 0.0
    update_wall = llm_update.total_wall_time_sec if llm_update else 0.0
    answer_wall = llm_answer.total_wall_time_sec if llm_answer else 0.0

    if args.dry_run:
        print("[done] dry-run complete.")
    else:
        print(
            "[done] "
            f"query_gen_calls={query_gen_calls} query_gen_tokens_in={query_gen_in} query_gen_tokens_out={query_gen_out} "
            f"update_calls={update_calls} update_tokens_in={update_in} update_tokens_out={update_out} "
            f"answer_calls={answer_calls} answer_tokens_in={answer_in} answer_tokens_out={answer_out} "
            f"query_gen_lm_wall_time_sec={query_gen_wall:.3f} "
            f"update_lm_wall_time_sec={update_wall:.3f} "
            f"answer_lm_wall_time_sec={answer_wall:.3f} "
            f"total_lm_wall_time_sec={(query_gen_wall + update_wall + answer_wall):.3f}",
            flush=True,
        )

    try:
        git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            if not args.dry_run
            else ""
        )
    except Exception:
        git_commit = ""

    finished = time.time()
    answer_model_name = "" if args.skip_answer else (args.answer_model or args.model)
    totals = aggregate_output_totals(out_path, skip_answer=args.skip_answer)
    manifest = {
        "method": "oracle_assisted_warm_dynamic_bank_with_summary",
        "dataset_jsonl": str(dataset_path),
        "out_jsonl": str(out_path),
        "trace_jsonl": str(trace_path) if trace_path else "",
        "canonical_trace_jsonl": str(canonical_trace_path) if canonical_trace_path else "",
        "canonical_doc_store_jsonl": str(canonical_doc_store_path) if canonical_doc_store_path else "",
        "canonical_trace_schema_version": CANONICAL_TRACE_SCHEMA_VERSION,
        "sample_id": sample_id,
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
        "memory_budget_tokens_per_query": args.memory_budget_tokens,
        "summary_budget_tokens": args.summary_budget_tokens,
        "z_warm_docs": args.z_warm_docs,
        "num_bank_queries": args.num_bank_queries,
        "answer_top_j": args.answer_top_j,
        "candidate_multiplier": args.candidate_multiplier,
        "selection_metric": args.selection_metric,
        "refresh_stride_docs": args.refresh_stride_docs,
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
        "log_query_selection_details": bool(args.log_query_selection_details),
        "tokenizer_budget": "tiktoken/cl100k_base",
        "tokenizer_note": (
            "Token budgeting/truncation uses tiktoken proxy tokenizer; "
            "Gemini-internal token counts may differ."
        ),
        "rows_selected": len(rows),
        "rows_processed": totals["rows_completed_total"],
        "rows_processed_this_run": processed,
        "rows_written_total": totals["rows_written_total"],
        "rows_runtime_error_total": totals["rows_runtime_error_total"],
        "rows_summary_update_failed_total": totals["rows_summary_update_failed_total"],
        "rows_skipped_completed": skipped_completed,
        "lm_usage_totals": {
            "query_gen_calls": totals["query_gen_calls"],
            "query_gen_input_tokens": totals["query_gen_input_tokens"],
            "query_gen_output_tokens": totals["query_gen_output_tokens"],
            "update_calls": totals["update_calls"],
            "update_input_tokens": totals["update_input_tokens"],
            "update_output_tokens": totals["update_output_tokens"],
            "answer_calls": totals["answer_calls"],
            "answer_input_tokens": totals["answer_input_tokens"],
            "answer_output_tokens": totals["answer_output_tokens"],
            "query_gen_wall_time_sec": round(float(totals["query_gen_wall_time_sec"]), 6),
            "update_wall_time_sec": round(float(totals["update_wall_time_sec"]), 6),
            "answer_wall_time_sec": round(float(totals["answer_wall_time_sec"]), 6),
            "total_wall_time_sec": round(float(totals["total_lm_wall_time_sec"]), 6),
            "total_calls": totals["query_gen_calls"] + totals["update_calls"] + totals["answer_calls"],
            "total_input_tokens": (
                totals["query_gen_input_tokens"] + totals["update_input_tokens"] + totals["answer_input_tokens"]
            ),
            "total_output_tokens": (
                totals["query_gen_output_tokens"] + totals["update_output_tokens"] + totals["answer_output_tokens"]
            ),
            "total_tokens": (
                totals["query_gen_input_tokens"]
                + totals["query_gen_output_tokens"]
                + totals["update_input_tokens"]
                + totals["update_output_tokens"]
                + totals["answer_input_tokens"]
                + totals["answer_output_tokens"]
            ),
        },
        "lm_usage_this_run": {
            "query_gen_calls": query_gen_calls,
            "query_gen_input_tokens": query_gen_in,
            "query_gen_output_tokens": query_gen_out,
            "update_calls": update_calls,
            "update_input_tokens": update_in,
            "update_output_tokens": update_out,
            "answer_calls": answer_calls,
            "answer_input_tokens": answer_in,
            "answer_output_tokens": answer_out,
            "query_gen_wall_time_sec": round(float(query_gen_wall), 6),
            "update_wall_time_sec": round(float(update_wall), 6),
            "answer_wall_time_sec": round(float(answer_wall), 6),
            "total_wall_time_sec": round(float(query_gen_wall + update_wall + answer_wall), 6),
            "total_calls": query_gen_calls + update_calls + answer_calls,
            "total_input_tokens": query_gen_in + update_in + answer_in,
            "total_output_tokens": query_gen_out + update_out + answer_out,
            "total_tokens": query_gen_in + query_gen_out + update_in + update_out + answer_in + answer_out,
        },
        "run_started_unix": run_started,
        "run_finished_unix": finished,
        "runtime_sec": round(finished - run_started, 3),
        "git_commit": git_commit,
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] wrote manifest to {manifest_path}", flush=True)
    if canonical_trace_path:
        print(f"[done] canonical trace jsonl: {canonical_trace_path}", flush=True)
    if canonical_doc_store_path:
        print(f"[done] canonical doc store jsonl: {canonical_doc_store_path}", flush=True)


if __name__ == "__main__":
    main()
