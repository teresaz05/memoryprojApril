#!/usr/bin/env python3
"""Final-answer grading implementation used by the April experiment package.

    This is a lightly adapted copy of ``BrowseCompV2/scripts/grade_answers.py``. It stays close to
    the original file so grading behavior matches the runs we have already been using."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import tiktoken
from dotenv import load_dotenv
try:
    from google import genai
except ImportError:
    genai = None


ANSWER_GRADING_PROMPT = """You are grading answer correctness against a gold answer.

Return STRICT JSON with exactly these keys:
- correct_binary: 0 or 1
- score_0_100: integer 0..100
- reasoning: short string (<= 35 words)

Rules:
1. Compare PREDICTED_ANSWER to GOLD_ANSWER for QUESTION.
2. correct_binary=1 only if semantically equivalent to GOLD_ANSWER.
3. Use score_0_100 to reflect quality:
   - 100: exact/equivalent answer
   - 90-99: very close, minor formatting or tiny wording variation
   - 70-89: mostly correct, slight omission/imprecision
   - 40-69: partially correct
   - 1-39: mostly incorrect but small overlap
   - 0: incorrect, irrelevant, empty, or refusal
4. Do not use outside knowledge. Judge only from QUESTION + GOLD_ANSWER + PREDICTED_ANSWER.
5. Semantically equivalent paraphrases should receive 100 even if wording, punctuation, article use, or order differs.
6. For list/set answers, ignore order but do not ignore missing core items or extra core items.
7. Do not give partial credit for mere topicality. Give nonzero credit only when PREDICTED_ANSWER overlaps materially with GOLD_ANSWER.
8. If PREDICTED_ANSWER adds an extra entity/detail that changes the meaning of GOLD_ANSWER, it is not fully correct.
9. Output valid JSON only. No markdown, no extra text.
"""


@dataclass
class RetryPolicy:
    retries: int = 5
    initial_backoff_sec: float = 2.0
    max_backoff_sec: float = 30.0


class TokenCounter:
    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.enc = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self.enc.encode(text or "", disallowed_special=()))


class GeminiClient:
    def __init__(self, model: str, retry_policy: RetryPolicy, timeout_sec: int) -> None:
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
                raise RuntimeError("Judge model returned empty output after retries.")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.retry_policy.retries:
                    time.sleep(min(backoff, self.retry_policy.max_backoff_sec))
                    backoff *= 2
                    continue
                break
        raise RuntimeError(f"Judge generation failed after retries: {last_error}")


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
                    raise RuntimeError("OpenRouter judge returned empty assistant content.")
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
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="ignore").strip()
                except Exception:
                    detail = ""
                detail_suffix = f"; body={detail[:280]}" if detail else ""
                last_error = RuntimeError(f"HTTP Error {exc.code}: {exc.reason}{detail_suffix}")
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
        raise RuntimeError(f"OpenRouter judge generation failed after retries: {last_error}")


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Allow incremental grading against a file that may still be
                    # receiving its last JSONL row from a live writer.
                    continue


def normalize_answer(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s:/\-.]", "", s)
    return s


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def flush_jsonl_handle(handle: Any) -> None:
    handle.flush()
    try:
        os.fsync(handle.fileno())
    except (AttributeError, OSError, ValueError):
        pass


def _extract_json_obj(raw: str) -> str:
    txt = (raw or "").strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*", "", txt.strip(), flags=re.IGNORECASE)
        txt = re.sub(r"\s*```$", "", txt.strip())
    m = re.search(r"\{[\s\S]*\}", txt)
    return m.group(0) if m else txt


def judge_answer(
    llm: Any,
    question: str,
    gold: str,
    pred: str,
) -> Dict[str, Any]:
    if exact_match(pred, gold) and str(pred).strip():
        return {
            "correct_binary": 1,
            "score_0_100": 100,
            "reasoning": "exact_normalized_match",
            "judge_raw": "deterministic_exact_match",
        }

    prompt = "\n\n".join(
        [
            ANSWER_GRADING_PROMPT,
            f"QUESTION: {question}",
            f"GOLD_ANSWER: {gold}",
            f"PREDICTED_ANSWER: {pred}",
            "JSON:",
        ]
    )
    raw = llm.generate(prompt, temperature=0.0)
    try:
        obj = json.loads(_extract_json_obj(raw))
    except Exception:
        obj = {
            "correct_binary": 0,
            "score_0_100": 0,
            "reasoning": f"parse_error: {raw[:240]}",
        }
    out = {
        "correct_binary": int(obj.get("correct_binary", 0)),
        "score_0_100": int(obj.get("score_0_100", 0)),
        "reasoning": str(obj.get("reasoning", "")),
        "judge_raw": raw[:4000],
    }
    out["correct_binary"] = 1 if out["correct_binary"] == 1 else 0
    out["score_0_100"] = max(0, min(100, out["score_0_100"]))
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Grade method answers and normalize metrics.")
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--dataset_jsonl", default="")
    ap.add_argument("--answer_field", default="auto")
    ap.add_argument("--question_field", default="question")
    ap.add_argument("--gold_field", default="gold_answer")
    ap.add_argument("--judge_backend", choices=["gemini", "openrouter"], default="openrouter")
    ap.add_argument("--judge_model", default="qwen/qwen3.5-397b-a17b")
    ap.add_argument("--openrouter_base_url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--openrouter_http_referer", default="")
    ap.add_argument("--openrouter_app_title", default="")
    ap.add_argument("--timeout_sec", type=int, default=300)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress_every", type=int, default=25)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_false", dest="resume")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.timeout_sec <= 0:
        raise ValueError("--timeout_sec must be > 0.")
    if args.retries < 0:
        raise ValueError("--retries must be >= 0.")
    if args.start_index < 0:
        raise ValueError("--start_index must be >= 0.")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0.")
    if args.progress_every < 0:
        raise ValueError("--progress_every must be >= 0.")


def load_gold_by_qid(dataset_jsonl: str, question_field: str, gold_field: str) -> Dict[str, Dict[str, str]]:
    if not dataset_jsonl:
        return {}
    path = Path(dataset_jsonl)
    if not path.exists():
        raise FileNotFoundError(f"dataset_jsonl not found: {path}")
    out: Dict[str, Dict[str, str]] = {}
    for row in iter_jsonl(path):
        qid = str(row.get("question_id", "")).strip()
        if not qid:
            continue
        out[qid] = {
            "question": str(row.get(question_field, "")),
            "gold_answer": str(row.get(gold_field, "")),
        }
    return out


def row_key(row: Dict[str, Any]) -> Tuple[str, str, int]:
    qid = str(row.get("question_id", "")).strip()
    variant = str(row.get("variant", "")).strip()
    budget = int(row.get("memory_budget_tokens", 0) or 0)
    return (qid, variant, budget)


def is_done_row(row: Dict[str, Any]) -> bool:
    if "accuracy_binary" not in row:
        return False
    if row.get("grading_error"):
        return False
    return True


def load_done_keys(path: Path) -> Set[Tuple[str, str, int]]:
    if not path.exists():
        return set()
    done: Set[Tuple[str, str, int]] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if is_done_row(row):
                k = row_key(row)
                if k[0]:
                    done.add(k)
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


def aggregate_scored_output(path: Path) -> Dict[str, Any]:
    stats = {
        "rows_written_total": 0,
        "rows_scored_total": 0,
        "rows_grading_error_total": 0,
        "judge_calls": 0,
        "judge_input_tokens": 0,
        "judge_output_tokens": 0,
        "judge_wall_time_sec": 0.0,
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
            if is_done_row(row):
                stats["rows_scored_total"] += 1
            if row.get("grading_error"):
                stats["rows_grading_error_total"] += 1
            judge_usage = row.get("judge_lm_usage") if isinstance(row.get("judge_lm_usage"), dict) else {}
            stats["judge_calls"] += _as_int(judge_usage.get("calls"))
            stats["judge_input_tokens"] += _as_int(judge_usage.get("input_tokens"))
            stats["judge_output_tokens"] += _as_int(judge_usage.get("output_tokens"))
            stats["judge_wall_time_sec"] += _as_float(judge_usage.get("wall_time_sec"))
    return stats


def pick_answer_field(row: Dict[str, Any], configured: str) -> str:
    if configured != "auto":
        return configured
    if "model_answer" in row:
        return "model_answer"
    if "pred_answer" in row:
        return "pred_answer"
    return "model_answer"


def extract_usage_totals(row: Dict[str, Any]) -> Dict[str, float]:
    if all(k in row for k in ("total_lm_calls", "total_lm_input_tokens", "total_lm_output_tokens")):
        calls = int(row.get("total_lm_calls", 0) or 0)
        in_tok = int(row.get("total_lm_input_tokens", 0) or 0)
        out_tok = int(row.get("total_lm_output_tokens", 0) or 0)
        wall = _as_float(row.get("total_lm_wall_time_sec"))
        if wall <= 0.0:
            wall = (
                _as_float(row.get("query_gen_lm_wall_time_sec"))
                + _as_float(row.get("update_lm_wall_time_sec"))
                + _as_float(row.get("answer_lm_wall_time_sec"))
            )
        return {
            "calls": calls,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
            "wall_time_sec": wall,
        }

    rlm_usage_totals = row.get("rlm_usage_totals")
    if isinstance(rlm_usage_totals, dict):
        calls = int(rlm_usage_totals.get("calls", 0) or 0)
        in_tok = int(rlm_usage_totals.get("input_tokens", 0) or 0)
        out_tok = int(rlm_usage_totals.get("output_tokens", 0) or 0)
        total_tok = int(rlm_usage_totals.get("total_tokens", in_tok + out_tok) or (in_tok + out_tok))
        wall = _as_float(row.get("total_lm_wall_time_sec"))
        if wall <= 0.0:
            wall = _as_float(row.get("lm_call_wall_time_sec"))
        return {
            "calls": calls,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": total_tok,
            "wall_time_sec": wall,
        }

    query_gen_usage = row.get("query_gen_lm_usage") if isinstance(row.get("query_gen_lm_usage"), dict) else {}
    update_usage = row.get("update_lm_usage") if isinstance(row.get("update_lm_usage"), dict) else {}
    answer_usage = row.get("answer_lm_usage") if isinstance(row.get("answer_lm_usage"), dict) else {}
    calls = (
        int(query_gen_usage.get("calls", 0) or 0)
        + int(update_usage.get("calls", 0) or 0)
        + int(answer_usage.get("calls", 0) or 0)
    )
    in_tok = (
        int(query_gen_usage.get("input_tokens", 0) or 0)
        + int(update_usage.get("input_tokens", 0) or 0)
        + int(answer_usage.get("input_tokens", 0) or 0)
    )
    out_tok = (
        int(query_gen_usage.get("output_tokens", 0) or 0)
        + int(update_usage.get("output_tokens", 0) or 0)
        + int(answer_usage.get("output_tokens", 0) or 0)
    )
    wall = (
        _as_float(query_gen_usage.get("wall_time_sec"))
        + _as_float(update_usage.get("wall_time_sec"))
        + _as_float(answer_usage.get("wall_time_sec"))
    )
    return {
        "calls": calls,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
        "wall_time_sec": wall,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    load_dotenv()

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    if not in_path.exists():
        raise FileNotFoundError(f"Input jsonl not found: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_started = time.time()

    gold_by_qid = load_gold_by_qid(args.dataset_jsonl, args.question_field, args.gold_field)
    rows_all = list(iter_jsonl(in_path))
    rows = rows_all[args.start_index :]
    if args.limit > 0:
        rows = rows[: args.limit]

    done_keys = load_done_keys(out_path) if args.resume else set()
    mode = "a" if args.resume else "w"

    counter = TokenCounter("cl100k_base")
    judge_llm: Optional[Any] = None
    if not args.dry_run:
        if args.judge_backend == "gemini":
            judge_llm = GeminiClient(
                model=args.judge_model,
                retry_policy=RetryPolicy(retries=args.retries),
                timeout_sec=args.timeout_sec,
            )
        else:
            judge_llm = OpenRouterClient(
                model=args.judge_model,
                retry_policy=RetryPolicy(retries=args.retries),
                timeout_sec=args.timeout_sec,
                token_counter=counter,
                base_url=args.openrouter_base_url,
                http_referer=args.openrouter_http_referer,
                app_title=args.openrouter_app_title,
            )

    processed = 0
    with out_path.open(mode, encoding="utf-8") as fout:
        for row in rows:
            key = row_key(row)
            if key[0] and key in done_keys:
                continue

            answer_field = pick_answer_field(row, args.answer_field)
            pred = str(row.get(answer_field, ""))
            qid = str(row.get("question_id", "")).strip()
            question = str(row.get(args.question_field, ""))
            gold = str(row.get(args.gold_field, ""))
            if qid and qid in gold_by_qid:
                if not question:
                    question = gold_by_qid[qid]["question"]
                if not gold:
                    gold = gold_by_qid[qid]["gold_answer"]

            grading_error = ""
            if not question or not gold:
                grading_error = "missing_question_or_gold_answer"
                judged = {
                    "correct_binary": 0,
                    "score_0_100": 0,
                    "reasoning": grading_error,
                    "judge_raw": "",
                }
            elif args.dry_run:
                judged = {
                    "correct_binary": 0,
                    "score_0_100": 0,
                    "reasoning": "DRY_RUN",
                    "judge_raw": "",
                }
                judge_usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "wall_time_sec": 0.0}
            else:
                try:
                    if judge_llm is None:
                        raise RuntimeError("Judge model is not initialized.")
                    judge_before = (
                        judge_llm.total_calls,
                        judge_llm.total_input_tokens,
                        judge_llm.total_output_tokens,
                        judge_llm.total_wall_time_sec,
                    )
                    judged = judge_answer(judge_llm, question=question, gold=gold, pred=pred)
                    judge_after = (
                        judge_llm.total_calls,
                        judge_llm.total_input_tokens,
                        judge_llm.total_output_tokens,
                        judge_llm.total_wall_time_sec,
                    )
                    judge_usage = {
                        "calls": judge_after[0] - judge_before[0],
                        "input_tokens": judge_after[1] - judge_before[1],
                        "output_tokens": judge_after[2] - judge_before[2],
                        "wall_time_sec": round(judge_after[3] - judge_before[3], 6),
                    }
                except Exception as exc:  # noqa: BLE001
                    grading_error = str(exc)
                    judged = {
                        "correct_binary": 0,
                        "score_0_100": 0,
                        "reasoning": f"judge_error: {exc}",
                        "judge_raw": "",
                    }
                    judge_usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "wall_time_sec": 0.0}
            if not question or not gold:
                judge_usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "wall_time_sec": 0.0}

            usage = extract_usage_totals(row)
            memory_text = str(row.get("memory_text", ""))
            if row.get("memory_tokens") is not None:
                memory_used_tokens = int(row.get("memory_tokens", 0) or 0)
            else:
                memory_used_tokens = counter.count(memory_text)

            row["accuracy_binary"] = int(judged["correct_binary"])
            row["accuracy_score_0_100"] = int(judged["score_0_100"])
            row["accuracy_reasoning"] = str(judged["reasoning"])
            row["judge_reasoning"] = str(judged["reasoning"])
            row["judge_raw"] = str(judged.get("judge_raw", ""))
            row["grading_error"] = grading_error
            row["query_similarity"] = None
            row["memory_used_tokens"] = memory_used_tokens
            row["total_lm_calls"] = usage["calls"]
            row["total_lm_input_tokens"] = usage["input_tokens"]
            row["total_lm_output_tokens"] = usage["output_tokens"]
            row["total_lm_tokens"] = usage["total_tokens"]
            row["total_lm_wall_time_sec"] = round(_as_float(usage["wall_time_sec"]), 6)
            row["judge_backend"] = args.judge_backend if not args.dry_run else ""
            row["judge_model"] = args.judge_model if not args.dry_run else ""
            row["judge_lm_usage"] = judge_usage
            row["judge_called"] = bool(judge_usage["calls"] > 0)
            row["judge_lm_wall_time_sec"] = round(_as_float(judge_usage["wall_time_sec"]), 6)
            row["total_lm_calls_with_judge"] = usage["calls"] + judge_usage["calls"]
            row["total_lm_input_tokens_with_judge"] = (
                usage["input_tokens"] + judge_usage["input_tokens"]
            )
            row["total_lm_output_tokens_with_judge"] = (
                usage["output_tokens"] + judge_usage["output_tokens"]
            )
            row["total_lm_tokens_with_judge"] = (
                row["total_lm_input_tokens_with_judge"] + row["total_lm_output_tokens_with_judge"]
            )
            row["total_lm_wall_time_sec_with_judge"] = round(
                _as_float(usage["wall_time_sec"]) + _as_float(judge_usage["wall_time_sec"]),
                6,
            )
            row["answer_field_used"] = answer_field
            row["answer_exact_match"] = bool(exact_match(pred, gold))

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            flush_jsonl_handle(fout)
            if key[0]:
                done_keys.add(key)
            processed += 1
            if args.progress_every > 0 and processed % args.progress_every == 0:
                print(
                    f"[progress] processed={processed} last_qid={qid} "
                    f"acc={row['accuracy_binary']} score={row['accuracy_score_0_100']}",
                    flush=True,
                )

    judge_calls = judge_llm.total_calls if judge_llm else 0
    judge_in = judge_llm.total_input_tokens if judge_llm else 0
    judge_out = judge_llm.total_output_tokens if judge_llm else 0
    judge_wall = judge_llm.total_wall_time_sec if judge_llm else 0.0
    totals = aggregate_scored_output(out_path)
    print(
        "[done] "
        f"rows_scored_this_run={processed} rows_scored_total={totals['rows_scored_total']} "
        f"judge_calls_this_run={judge_calls} judge_tokens_in_this_run={judge_in} "
        f"judge_tokens_out_this_run={judge_out} judge_lm_wall_time_sec_this_run={judge_wall:.3f}",
        flush=True,
    )
    finished = time.time()
    manifest = {
        "in_jsonl": str(in_path),
        "out_jsonl": str(out_path),
        "dataset_jsonl": args.dataset_jsonl,
        "answer_field": args.answer_field,
        "question_field": args.question_field,
        "gold_field": args.gold_field,
        "judge_backend": args.judge_backend,
        "judge_model": args.judge_model,
        "openrouter_base_url": args.openrouter_base_url,
        "openrouter_http_referer": args.openrouter_http_referer,
        "openrouter_app_title": args.openrouter_app_title,
        "timeout_sec": args.timeout_sec,
        "retries": args.retries,
        "start_index": args.start_index,
        "limit": args.limit,
        "resume": bool(args.resume),
        "dry_run": bool(args.dry_run),
        "tokenizer": "tiktoken/cl100k_base",
        "rows_scored": totals["rows_scored_total"],
        "rows_scored_this_run": processed,
        "rows_written_total": totals["rows_written_total"],
        "rows_grading_error_total": totals["rows_grading_error_total"],
        "judge_usage_totals": {
            "calls": totals["judge_calls"],
            "input_tokens": totals["judge_input_tokens"],
            "output_tokens": totals["judge_output_tokens"],
            "total_tokens": totals["judge_input_tokens"] + totals["judge_output_tokens"],
            "wall_time_sec": round(float(totals["judge_wall_time_sec"]), 6),
        },
        "judge_usage_this_run": {
            "calls": judge_calls,
            "input_tokens": judge_in,
            "output_tokens": judge_out,
            "total_tokens": judge_in + judge_out,
            "wall_time_sec": round(float(judge_wall), 6),
        },
        "run_started_unix": run_started,
        "run_finished_unix": finished,
        "runtime_sec": round(finished - run_started, 3),
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] wrote manifest to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
