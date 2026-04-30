#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import tiktoken
try:
    from google import genai
except ImportError:
    genai = None

class HardTimeoutError(TimeoutError):
    pass

def resolve_openai_compatible_api_key(base_url: str = "") -> str:
    """Resolve an API key for any OpenAI-compatible endpoint.

    OpenRouter and local vLLM both speak the same chat-completions protocol.
    We keep the existing client class but allow either environment layout:
    - OPENROUTER_API_KEY for hosted OpenRouter usage
    - VLLM_API_KEY / OPENAI_COMPAT_API_KEY / OPENAI_API_KEY for self-hosted usage
    """
    normalized_base_url = str(base_url or "").strip().lower()
    prefer_openrouter = "openrouter.ai" in normalized_base_url
    env_order = (
        ["OPENROUTER_API_KEY", "OPENAI_COMPAT_API_KEY", "VLLM_API_KEY", "OPENAI_API_KEY"]
        if prefer_openrouter
        else ["OPENAI_COMPAT_API_KEY", "VLLM_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"]
    )
    for env_name in env_order:
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    if prefer_openrouter:
        raise RuntimeError(
            "Set OPENROUTER_API_KEY (or another OpenAI-compatible key env var) for hosted OpenRouter usage."
        )
    raise RuntimeError(
        "Set VLLM_API_KEY, OPENAI_COMPAT_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY "
        "for the configured OpenAI-compatible endpoint."
    )

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
        self.api_key = resolve_openai_compatible_api_key(base_url)
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
