#!/usr/bin/env python3
"""Oracle-memory baseline utilities."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import requests
import tiktoken
from dotenv import load_dotenv
try:
    from google import genai
except ImportError:
    genai = None
from april_version_code.common import metadata as row_metadata


class HardTimeoutError(TimeoutError):
    pass


class _HardTimeout:
    def __init__(self, seconds: int, label: str) -> None:
        self.seconds = max(1, int(seconds))
        self.label = label
        self.enabled = (
            os.name != "nt"
            and threading.current_thread() is threading.main_thread()
            and hasattr(signal, "SIGALRM")
            and hasattr(signal, "setitimer")
            and hasattr(signal, "ITIMER_REAL")
        )
        self._prev_handler: Any = None
        self._prev_timer: Optional[tuple[float, float]] = None

    def _handle_timeout(self, signum: int, frame: Any) -> None:
        raise HardTimeoutError(f"{self.label} hard-timed out after {self.seconds}s")

    def __enter__(self) -> "_HardTimeout":
        if not self.enabled:
            return self
        self._prev_handler = signal.getsignal(signal.SIGALRM)
        self._prev_timer = signal.setitimer(signal.ITIMER_REAL, self.seconds, 0.0)
        signal.signal(signal.SIGALRM, self._handle_timeout)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if not self.enabled:
            return None
        signal.setitimer(signal.ITIMER_REAL, 0.0, 0.0)
        if self._prev_handler is not None:
            signal.signal(signal.SIGALRM, self._prev_handler)
        if self._prev_timer is not None:
            delay, interval = self._prev_timer
            if delay > 0 or interval > 0:
                signal.setitimer(signal.ITIMER_REAL, delay, interval)
        return None

REWRITE_UPDATE_PROMPT = """You are an evidence-grounded memory optimizer for streaming QA.

You maintain a bounded memory bank for one known target question while documents arrive one at a time.

Primary objective:
- Maximize the probability of answering TARGET_QUERY correctly after all future documents are processed.

Hard constraints:
1. Use only CURRENT_MEMORY and NEW_DOCUMENT. Do not use outside knowledge.
2. Keep only information that is relevant or plausibly relevant to TARGET_QUERY.
3. Prefer concrete, checkable facts: names, entities, dates, numbers, titles, places, and explicit relationships.
4. Remove redundancy and low-value details aggressively.
5. Handle conflicts with explicit tie-breaks:
   - Prefer direct explicit evidence over inference.
   - Prefer more specific evidence over generic evidence.
   - Prefer internally consistent high-support facts over weak single-mention clues.
   - If a conflict cannot be resolved, keep both but mark one as primary and the other as alternate.
6. If NEW_DOCUMENT adds no useful information, keep memory concise and stable.
7. Order output by answer utility: most answer-critical information first.
8. Keep output concise and dense, but optimize for answer quality first.
9. SOFT_MEMORY_TARGET_TOKENS is guidance, not a hard target.
10. Do NOT drop previously retained critical facts unless memory is well over budget and those facts are clearly lower-priority than stronger evidence.
11. If memory has room, retain useful high-value information rather than over-compressing.
12. Never store absence-style statements as memory facts (for example: "not mentioned", "unknown", "not provided in this document", "cannot determine from this document"), unless they are themselves the target evidence.
13. Stability preference: if NEW_DOCUMENT does not add clearly stronger or corrective evidence for a retained fact, keep the corresponding CURRENT_MEMORY content unchanged. Minor cleanup (deduplication, concise wording) is allowed, but avoid unnecessary rewrites.
14. Provenance-preserving conflict rule: when different documents provide conflicting or differently attributed claims, keep them as separate attributed entries instead of collapsing into one merged statement. Use compact source tags (for example, "Source: doc_id=...") on those entries.
15. If the answer may depend on who wrote, said, discovered, published, created, or attributed something, preserve that source-specific attribution explicitly rather than collapsing it away.

Output requirements:
- Output plain text memory only (no JSON, no markdown fences, no preamble).
- Put the most answer-critical facts at the beginning.
- Sort entries in strict descending importance for TARGET_QUERY.
- Keep concise, evidence-rich statements.
- Do not include explicit "missing info"/gap tracking.

TARGET_QUERY:
{target_query}

SOFT_MEMORY_TARGET_TOKENS:
{memory_budget_tokens}

CURRENT_MEMORY:
{current_memory}

NEW_DOCUMENT:
{new_document}
"""


APPEND_EXTRACT_PROMPT = """You are extracting query-relevant memory from one incoming document.

Task:
- Read TARGET_QUERY and NEW_DOCUMENT.
- Extract only information useful for answering TARGET_QUERY.
- Prioritize direct answer evidence, then high-value supporting facts.
- Omit irrelevant details, boilerplate, and repetition.
- Preserve source-specific attribution when authorship, publication, discovery, or quoted source identity could matter.
- Do not output absence-style statements (for example: "not mentioned", "unknown", "not provided in this document"), unless that absence is itself explicit target evidence.
- If document is not useful, output exactly: none

Output requirements:
- Output plain text snippet only (no JSON, no markdown fences, no preamble).
- Put highest-value facts first.
- Sort facts in strict descending importance for TARGET_QUERY.

TARGET_QUERY:
{target_query}

NEW_DOCUMENT:
{new_document}
"""


COMPRESS_PROMPT = """You are compressing a memory bank to a strict token budget.

Goal:
- Keep only the information most likely to help answer TARGET_QUERY correctly.
- Preserve direct answer evidence first.
- Keep key supporting facts and critical disambiguators.
- Remove repetition, weak clues, and low-value context.
- Keep highest-value facts first in the output.

Hard requirements:
1. Rank all retained information by answer utility for TARGET_QUERY.
2. Place the most answer-critical items at the very beginning of the output.
3. Keep concise evidence-rich facts (entities, dates, numbers, titles, relations).
4. Resolve conflicts with explicit tie-breaks:
   - Prefer direct explicit evidence over inference.
   - Prefer more specific evidence over generic evidence.
   - Prefer internally consistent high-support facts over weak single-mention clues.
   - If unresolved, keep primary+alternate and label clearly.
5. Drop lower-priority details before dropping high-priority facts.
6. Do not add external knowledge; use only MEMORY_TO_COMPRESS.
7. Output should be front-loaded: if truncation later happens, the beginning should still contain the best evidence.
8. Remove absence-style statements ("not mentioned", "unknown", "not provided") unless they are essential target evidence.
9. Preserve source-specific attribution when the target may hinge on who said, wrote, authored, discovered, or published something.

Output requirements:
- Output plain text memory only (no JSON, no markdown fences, no preamble).
- Front-load answer-critical facts at the beginning.
- Sort retained facts in strict descending importance for TARGET_QUERY.
- Keep concise evidence-rich statements.

TARGET_QUERY:
{target_query}

MAX_MEMORY_TOKENS:
{memory_budget_tokens}

MEMORY_TO_COMPRESS:
{memory_text}
"""


ANSWER_FROM_MEMORY_PROMPT = """You are answering a question using only the provided memory bank.

Rules:
1. Use only MEMORY_BANK content.
2. Do not use outside knowledge.
3. Provide your best-supported answer from MEMORY_BANK, even if evidence is limited.
4. Return only one short final answer string.
5. Do not include explanation, reasoning, uncertainty notes, or extra context.
6. Do not use markdown, bullets, quotes, or prefixes.
7. If MEMORY_BANK contains conflicting attributed candidates, choose the candidate with the strongest direct support and most specific attribution in MEMORY_BANK.

TARGET_QUERY:
{target_query}

MEMORY_BANK:
{memory_text}

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
            raise RuntimeError("google-genai is required for llm_backend=gemini but is not installed.")
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
        self.connect_timeout_sec = max(1, min(30, self.timeout_sec))
        self.read_timeout_sec = max(1, self.timeout_sec)
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
            "Connection": "close",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_title:
            headers["X-Title"] = self.app_title
        helper = r"""
import json, sys, requests
payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
resp = requests.post(
    payload["url"],
    data=payload["body"].encode("utf-8"),
    headers=payload["headers"],
    timeout=(payload["connect_timeout_sec"], payload["read_timeout_sec"]),
)
out = {
    "status_code": int(resp.status_code),
    "text": resp.text,
}
sys.stdout.write(json.dumps(out))
"""
        child_payload = {
            "url": f"{self.base_url}/chat/completions",
            "body": body.decode("utf-8"),
            "headers": headers,
            "connect_timeout_sec": self.connect_timeout_sec,
            "read_timeout_sec": self.read_timeout_sec,
        }
        try:
            completed = subprocess.run(
                [sys.executable, "-c", helper],
                input=json.dumps(child_payload).encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_sec + 10,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise HardTimeoutError(f"OpenRouter subprocess hard-timed out after {self.timeout_sec + 10}s") from exc

        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"OpenRouter subprocess failed with code {completed.returncode}: {stderr[:280]}")

        try:
            envelope = json.loads(completed.stdout.decode("utf-8"))
        except json.JSONDecodeError as exc:
            stderr = completed.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"OpenRouter subprocess returned invalid JSON: {stderr[:280]}") from exc

        status_code = int(envelope.get("status_code") or 0)
        text = str(envelope.get("text") or "")
        if status_code >= 400:
            detail = text[:280].strip()
            raise RuntimeError(f"HTTP Error {status_code}: body={detail}" if detail else f"HTTP Error {status_code}")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            detail = text[:280].strip()
            detail_suffix = f"; body={detail}" if detail else ""
            raise RuntimeError(f"OpenRouter returned invalid JSON{detail_suffix}") from exc

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
                if prompt_tokens <= 0:
                    prompt_tokens = self.token_counter.count(prompt)
                if completion_tokens <= 0:
                    completion_tokens = self.token_counter.count(text)
                self.total_calls += 1
                self.total_input_tokens += prompt_tokens
                self.total_output_tokens += completion_tokens
                return text
            except HardTimeoutError as exc:
                last_error = RuntimeError(str(exc))
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


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def is_completed_row(row: Dict[str, Any], skip_answer: bool) -> bool:
    if str(row.get("runtime_error", "")).strip():
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
        "update_calls": 0,
        "update_input_tokens": 0,
        "update_output_tokens": 0,
        "answer_calls": 0,
        "answer_input_tokens": 0,
        "answer_output_tokens": 0,
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
            u = row.get("update_lm_usage") if isinstance(row.get("update_lm_usage"), dict) else {}
            a = row.get("answer_lm_usage") if isinstance(row.get("answer_lm_usage"), dict) else {}
            stats["update_calls"] += _as_int(u.get("calls"))
            stats["update_input_tokens"] += _as_int(u.get("input_tokens"))
            stats["update_output_tokens"] += _as_int(u.get("output_tokens"))
            stats["answer_calls"] += _as_int(a.get("calls"))
            stats["answer_input_tokens"] += _as_int(a.get("input_tokens"))
            stats["answer_output_tokens"] += _as_int(a.get("output_tokens"))
            stats["update_wall_time_sec"] += _as_float(u.get("wall_time_sec"))
            stats["answer_wall_time_sec"] += _as_float(a.get("wall_time_sec"))
            total_row_wall = row.get("total_lm_wall_time_sec")
            if total_row_wall is None:
                total_row_wall = _as_float(u.get("wall_time_sec")) + _as_float(a.get("wall_time_sec"))
            stats["total_lm_wall_time_sec"] += _as_float(total_row_wall)
    return stats


def format_doc_for_prompt(doc: Dict[str, Any]) -> str:
    # Do not expose gold/evidence/negative labels to the memory updater.
    # Oracle here means known target query, not access to evaluation labels.
    parts = [
        f"doc_id: {doc.get('doc_id', '')}",
    ]
    parts.extend(
        [
            "text:",
            doc.get("text", ""),
        ]
    )
    return "\n".join(parts)


def is_none_snippet(text: str) -> bool:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = t.strip(" .!?\t\n\r")
    none_patterns = {
        "none",
        "n/a",
        "na",
        "not useful",
        "no relevant information",
        "no relevant facts",
        "irrelevant",
    }
    return t in none_patterns


def apply_overflow_policy(
    memory_text: str,
    target_query: str,
    budget_tokens: int,
    counter: TokenCounter,
    llm: Optional[GeminiClient],
    overflow_policy: str,
    truncate_strategy: str,
    temperature: float,
) -> tuple[str, Dict[str, int]]:
    stats = {"compress_calls": 0, "truncate_events": 0}
    cur = (memory_text or "").strip()
    cur_tokens = counter.count(cur)
    if cur_tokens <= budget_tokens:
        return cur, stats

    if overflow_policy in {"compress_then_truncate", "compress_only"}:
        if llm is not None:
            compress_prompt = COMPRESS_PROMPT.format(
                target_query=target_query,
                memory_budget_tokens=budget_tokens,
                memory_text=cur,
            )
            cur = llm.generate(compress_prompt, temperature=temperature).strip()
            stats["compress_calls"] += 1
            cur_tokens = counter.count(cur)

    if cur_tokens <= budget_tokens:
        return cur, stats

    if overflow_policy in {"compress_then_truncate", "truncate_only"}:
        cur = counter.truncate(cur, budget_tokens, strategy=truncate_strategy)
        stats["truncate_events"] += 1
        return cur, stats

    raise RuntimeError(
        f"Memory exceeds budget after overflow handling: tokens={cur_tokens}, budget={budget_tokens}"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Streaming oracle memory builder (known target query, one doc at a time)."
    )
    ap.add_argument("--dataset_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--trace_jsonl", default="")
    ap.add_argument("--llm_backend", choices=["gemini", "openrouter"], default="openrouter")
    ap.add_argument("--model", default="qwen/qwen3.5-397b-a17b")
    ap.add_argument("--answer_model", default="")
    ap.add_argument("--openrouter_base_url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--openrouter_http_referer", default="")
    ap.add_argument("--openrouter_app_title", default="")
    ap.add_argument("--memory_budget_tokens", type=int, default=2000)
    ap.add_argument("--target_field", choices=["question", "gold_answer"], default="question")
    ap.add_argument("--update_mode", choices=["rewrite", "append"], default="rewrite")
    ap.add_argument(
        "--overflow_policy",
        choices=["compress_then_truncate", "truncate_only", "compress_only"],
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
    ap.add_argument("--update_temperature", type=float, default=0.0)
    ap.add_argument("--answer_temperature", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_false", dest="resume")
    ap.add_argument("--skip_answer", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.memory_budget_tokens <= 0:
        raise ValueError("--memory_budget_tokens must be > 0.")
    if args.max_doc_tokens < 0:
        raise ValueError("--max_doc_tokens must be >= 0.")
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
    if args.max_docs_per_query < 0:
        raise ValueError("--max_docs_per_query must be >= 0.")
    if args.dry_run and args.overflow_policy == "compress_only":
        raise ValueError(
            "--dry_run cannot be used with --overflow_policy=compress_only. "
            "compress_only requires an LLM compression step."
        )


def update_target_text(question: str, gold_answer: str, target_field: str) -> str:
    if target_field == "gold_answer":
        return str(gold_answer or "")
    return str(question or "")


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

    rows_all = list(iter_jsonl(dataset_path))
    rows = rows_all[args.start_index :]
    if args.limit > 0:
        rows = rows[: args.limit]
    run_started = time.time()

    done_ids = load_done_ids(out_path, skip_answer=args.skip_answer) if args.resume else set()
    mode = "a" if args.resume else "w"

    counter = TokenCounter("cl100k_base")
    retry_policy = RetryPolicy(retries=args.retries)

    llm_update: Optional[Any] = None
    llm_answer: Optional[Any] = None
    if not args.dry_run:
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
        try:
            processed = 0
            skipped_completed = 0
            for idx, row in enumerate(rows, start=1):
                qid = str(row.get("question_id", "")).strip()
                if not qid:
                    continue
                if qid in done_ids:
                    skipped_completed += 1
                    continue

                question = row.get("question", "")
                gold_answer = row.get("gold_answer", "")
                update_target = update_target_text(question, gold_answer, args.target_field)
                docs = list(row.get("docs") or [])
                if args.max_docs_per_query > 0:
                    docs = docs[: args.max_docs_per_query]

                memory_text = ""
                memory_overflow_compress_calls = 0
                memory_overflow_truncate_events = 0
                doc_truncations = 0
                update_errors: List[str] = []
                runtime_error = ""
                started = time.time()
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

                update_step_count = 0
                for step_idx, doc in enumerate(docs, start=1):
                    raw_doc_text = format_doc_for_prompt(doc)
                    raw_doc_tokens = counter.count(raw_doc_text)
                    doc_text = raw_doc_text
                    if args.max_doc_tokens > 0 and raw_doc_tokens > args.max_doc_tokens:
                        doc_text = counter.truncate(
                            raw_doc_text,
                            max_tokens=args.max_doc_tokens,
                            strategy=args.doc_truncate_strategy,
                        )
                        doc_truncations += 1

                    step_error = ""
                    try:
                        if args.dry_run:
                            dry_snippet = counter.truncate(doc_text, 256, strategy="head")
                            candidate = (
                                (memory_text + "\n\n" + dry_snippet).strip()
                                if (memory_text and dry_snippet)
                                else (memory_text or dry_snippet)
                            )
                        else:
                            if llm_update is None:
                                raise RuntimeError("Update model is not initialized.")

                            if args.update_mode == "rewrite":
                                prompt = REWRITE_UPDATE_PROMPT.format(
                                    target_query=update_target,
                                    memory_budget_tokens=args.memory_budget_tokens,
                                    current_memory=memory_text if memory_text else "(empty)",
                                    new_document=doc_text,
                                )
                                candidate = llm_update.generate(
                                    prompt, temperature=args.update_temperature
                                ).strip()
                            else:
                                prompt = APPEND_EXTRACT_PROMPT.format(
                                    target_query=update_target,
                                    new_document=doc_text,
                                )
                                doc_snippet = llm_update.generate(
                                    prompt, temperature=args.update_temperature
                                ).strip()
                                if is_none_snippet(doc_snippet):
                                    candidate = memory_text
                                else:
                                    candidate = (
                                        (memory_text + "\n\n" + doc_snippet).strip()
                                        if memory_text
                                        else doc_snippet
                                    )

                        candidate, ov_stats = apply_overflow_policy(
                            memory_text=candidate,
                            target_query=update_target,
                            budget_tokens=args.memory_budget_tokens,
                            counter=counter,
                            llm=llm_update if not args.dry_run else None,
                            overflow_policy=args.overflow_policy,
                            truncate_strategy=args.truncate_strategy,
                            temperature=args.update_temperature,
                        )
                        memory_overflow_compress_calls += ov_stats["compress_calls"]
                        memory_overflow_truncate_events += ov_stats["truncate_events"]
                        memory_text = candidate
                        update_step_count += 1
                    except Exception as exc:  # noqa: BLE001
                        update_errors.append(f"step={step_idx}: {exc}")
                        step_error = str(exc)
                        runtime_error = f"update_failed_step_{step_idx}"

                    if trace_file:
                        trace_row = {
                            "question_id": qid,
                            "step_idx": step_idx,
                            "doc_id": str(doc.get("doc_id", "")),
                            "is_gold": bool(doc.get("is_gold", False)),
                            "is_evidence": bool(doc.get("is_evidence", False)),
                            "is_negative": bool(doc.get("is_negative", False)),
                            "raw_doc_tokens": raw_doc_tokens,
                            "doc_tokens_after_cap": counter.count(doc_text),
                            "memory_tokens": counter.count(memory_text),
                            "memory_budget_tokens": args.memory_budget_tokens,
                            "overflow_compress_calls_total": memory_overflow_compress_calls,
                            "overflow_truncate_events_total": memory_overflow_truncate_events,
                            "step_error": step_error,
                        }
                        trace_file.write(json.dumps(trace_row, ensure_ascii=False) + "\n")
                    if step_error:
                        break

                final_answer = ""
                answer_error = ""
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
                        answer_prompt = ANSWER_FROM_MEMORY_PROMPT.format(
                            target_query=question,
                            memory_text=memory_text if memory_text else "(empty)",
                        )
                        final_answer = llm_answer.generate(
                            answer_prompt, temperature=args.answer_temperature
                        ).strip()
                    except Exception as exc:  # noqa: BLE001
                        answer_error = str(exc)
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
                total_lm_calls = update_usage["calls"] + answer_usage["calls"]
                total_lm_input_tokens = update_usage["input_tokens"] + answer_usage["input_tokens"]
                total_lm_output_tokens = update_usage["output_tokens"] + answer_usage["output_tokens"]
                total_lm_wall_time_sec = update_usage["wall_time_sec"] + answer_usage["wall_time_sec"]

                out_row = {
                    "variant": (
                        f"oracle_rewrite_target_{args.target_field}_M{args.memory_budget_tokens}"
                        if args.update_mode == "rewrite"
                        else f"oracle_append_target_{args.target_field}_M{args.memory_budget_tokens}"
                    ),
                    "method": (
                        f"oracle_{args.update_mode}_target_{args.target_field}"
                    ),
                    "question_id": qid,
                    "question": question,
                    "gold_answer": gold_answer,
                    "llm_backend": args.llm_backend,
                    "update_model": args.model,
                    "answer_model": (args.answer_model or args.model) if not args.skip_answer else "",
                    "num_stream_docs": len(docs),
                    "memory_text": memory_text,
                    "memory_tokens": counter.count(memory_text),
                    "memory_budget_tokens": args.memory_budget_tokens,
                    "target_field": args.target_field,
                    "update_target_text": update_target,
                    "update_mode": args.update_mode,
                    "update_attempt_count": len(docs),
                    "update_step_count": update_step_count,
                    "overflow_policy": args.overflow_policy,
                    "truncate_strategy": args.truncate_strategy,
                    "doc_truncate_strategy": args.doc_truncate_strategy,
                    "doc_truncations": doc_truncations,
                    "overflow_compress_calls": memory_overflow_compress_calls,
                    "overflow_truncate_events": memory_overflow_truncate_events,
                    "row_failed": bool(runtime_error or answer_error),
                    "model_answer": final_answer,
                    "update_errors": update_errors,
                    "answer_error": answer_error,
                    "runtime_error": runtime_error,
                    "update_lm_usage": update_usage,
                    "answer_lm_usage": answer_usage,
                    "total_lm_calls": total_lm_calls,
                    "total_lm_input_tokens": total_lm_input_tokens,
                    "total_lm_output_tokens": total_lm_output_tokens,
                    "total_lm_tokens": total_lm_input_tokens + total_lm_output_tokens,
                    "update_lm_wall_time_sec": update_usage["wall_time_sec"],
                    "answer_lm_wall_time_sec": answer_usage["wall_time_sec"],
                    "total_lm_wall_time_sec": round(total_lm_wall_time_sec, 6),
                    "runtime_sec": round(time.time() - started, 3),
                    "dry_run": bool(args.dry_run),
                }
                row_metadata.attach_sample_metadata(out_row, row)

                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                fout.flush()
                if runtime_error or answer_error:
                    first_update_error = update_errors[0] if update_errors else ""
                    print(
                        f"[row_error] qid={qid} runtime_error={runtime_error or 'none'} "
                        f"answer_error={answer_error or 'none'} "
                        f"update_error={first_update_error or 'none'}",
                        flush=True,
                    )
                processed += 1
                if args.progress_every > 0 and processed % args.progress_every == 0:
                    print(
                        f"[progress] processed={processed} last_qid={qid} "
                        f"memory_tokens={out_row['memory_tokens']}",
                        flush=True,
                    )
        finally:
            if trace_file:
                trace_file.close()

    update_calls = llm_update.total_calls if llm_update else 0
    update_in = llm_update.total_input_tokens if llm_update else 0
    update_out = llm_update.total_output_tokens if llm_update else 0
    answer_calls = llm_answer.total_calls if llm_answer else 0
    answer_in = llm_answer.total_input_tokens if llm_answer else 0
    answer_out = llm_answer.total_output_tokens if llm_answer else 0
    update_wall = llm_update.total_wall_time_sec if llm_update else 0.0
    answer_wall = llm_answer.total_wall_time_sec if llm_answer else 0.0

    if args.dry_run:
        print("[done] dry-run complete.")
    else:
        print(
            "[done] "
            f"update_calls={update_calls} update_tokens_in={update_in} update_tokens_out={update_out} "
            f"answer_calls={answer_calls} answer_tokens_in={answer_in} answer_tokens_out={answer_out} "
            f"update_lm_wall_time_sec={update_wall:.3f} answer_lm_wall_time_sec={answer_wall:.3f} "
            f"total_lm_wall_time_sec={(update_wall + answer_wall):.3f}",
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
        "dataset_jsonl": str(dataset_path),
        "out_jsonl": str(out_path),
        "trace_jsonl": str(trace_path) if trace_path else "",
        "llm_backend": args.llm_backend,
        "update_model": args.model,
        "answer_model": answer_model_name,
        "openrouter_base_url": args.openrouter_base_url,
        "openrouter_http_referer": args.openrouter_http_referer,
        "openrouter_app_title": args.openrouter_app_title,
        "memory_budget_tokens": args.memory_budget_tokens,
        "target_field": args.target_field,
        "update_mode": args.update_mode,
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
        "update_temperature": args.update_temperature,
        "answer_temperature": args.answer_temperature,
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
        "rows_skipped_completed": skipped_completed,
        "lm_usage_totals": {
            "update_calls": totals["update_calls"],
            "update_input_tokens": totals["update_input_tokens"],
            "update_output_tokens": totals["update_output_tokens"],
            "answer_calls": totals["answer_calls"],
            "answer_input_tokens": totals["answer_input_tokens"],
            "answer_output_tokens": totals["answer_output_tokens"],
            "update_wall_time_sec": round(float(totals["update_wall_time_sec"]), 6),
            "answer_wall_time_sec": round(float(totals["answer_wall_time_sec"]), 6),
            "total_wall_time_sec": round(float(totals["total_lm_wall_time_sec"]), 6),
            "total_calls": totals["update_calls"] + totals["answer_calls"],
            "total_input_tokens": totals["update_input_tokens"] + totals["answer_input_tokens"],
            "total_output_tokens": totals["update_output_tokens"] + totals["answer_output_tokens"],
            "total_tokens": (
                totals["update_input_tokens"]
                + totals["update_output_tokens"]
                + totals["answer_input_tokens"]
                + totals["answer_output_tokens"]
            ),
        },
        "lm_usage_this_run": {
            "update_calls": update_calls,
            "update_input_tokens": update_in,
            "update_output_tokens": update_out,
            "answer_calls": answer_calls,
            "answer_input_tokens": answer_in,
            "answer_output_tokens": answer_out,
            "update_wall_time_sec": round(float(update_wall), 6),
            "answer_wall_time_sec": round(float(answer_wall), 6),
            "total_wall_time_sec": round(float(update_wall + answer_wall), 6),
            "total_calls": update_calls + answer_calls,
            "total_input_tokens": update_in + answer_in,
            "total_output_tokens": update_out + answer_out,
            "total_tokens": update_in + update_out + answer_in + answer_out,
        },
        "run_started_unix": run_started,
        "run_finished_unix": finished,
        "runtime_sec": round(finished - run_started, 3),
        "git_commit": git_commit,
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] wrote manifest to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
