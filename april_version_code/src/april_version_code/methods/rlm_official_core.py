#!/usr/bin/env python3
"""Official RLM runner

    This file is the shared RLM core used by the prompt-doc experiment. It remains close to the
    original implementation so the April package reproduces the same recursive language model
    behavior we have been evaluating already."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import tiktoken
from dotenv import load_dotenv
from rlm import RLM
from rlm.clients.gemini import GeminiClient as RLMSGeminiClient
from rlm.clients.openai import OpenAIClient as RLMSOpenAIClient
from rlm.environments.local_repl import LocalREPL
from rlm.utils.prompts import RLM_SYSTEM_PROMPT
from april_version_code.common import metadata as row_metadata


ORIG_EXECUTE = LocalREPL.execute_code
ORIG_RLMS_GEMINI_COMPLETION = RLMSGeminiClient.completion
ORIG_RLMS_OPENAI_COMPLETION = RLMSOpenAIClient.completion

LAST_MEMORY_STATE_TOKENS = 0
LAST_EXECUTE_CALLS = 0
LAST_MEMORY_STATE_PRESENT = False
LAST_MEMORY_STATE_TEXT = ""
TOTAL_RLM_GEMINI_CALL_WALL_TIME_SEC = 0.0


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


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_answer(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s\-\./:]", "", s)
    return s


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


EXPLICIT_ANSWER_LINE_RE = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(?:\*\*|__|\*)?\s*"
    r"(?:final(?:\s+answer|_answer)?|answer)"
    r"\s*(?:\*\*|__|\*)?\s*[:\-]\s*(.+?)\s*$"
)
MAX_EXTRACT_ANSWER_WORDS = 64
MAX_LINE_FALLBACK_CHARS = 400
LONG_RESPONSE_CHARS = 2000


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


def clean_answer_candidate(text: str) -> str:
    t = str(text or "")
    t = t.replace("```", " ")
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return ""
    # Strip common boilerplate prefixes.
    t = re.sub(
        r"(?i)^(the answer is|final answer is|answer is|answer|final(?:\s+answer|_answer)?)\s*[:\-]?\s*",
        "",
        t,
    ).strip()
    # Remove repeated marker chains and meta text that often follows extracted answers.
    marker = (
        re.search(r"(?i)(?:final(?:\s+answer|_answer)?|answer)\s*:", t[1:])
        if len(t) > 1
        else None
    )
    if marker:
        t = t[: marker.start() + 1].strip()
    tail = re.search(
        r"(?i)(the previous|based on the provided|i have gathered|i can now provide|reasoning:|analysis:)",
        t,
    )
    if tail and tail.start() > 0:
        t = t[: tail.start()].strip()
    t = t.strip(" \t\r\n\"'`*_")
    words = t.split()
    if len(words) > MAX_EXTRACT_ANSWER_WORDS:
        t = " ".join(words[:MAX_EXTRACT_ANSWER_WORDS]).strip()
    return t


def extract_final_answer(raw_response: str) -> Tuple[str, str, str]:
    text = str(raw_response or "").strip()
    if not text:
        return "", "empty_response", "empty_response"
    if text.lower().startswith("error:"):
        short = re.sub(r"\s+", " ", text)[:400]
        return "", "error_response", f"rlm_error_response: {short}"

    finals = [m.group(1).strip() for m in EXPLICIT_ANSWER_LINE_RE.finditer(text)]
    for cand_raw in reversed(finals):
        cand = clean_answer_candidate(cand_raw)
        if cand:
            return cand, "explicit_final_marker", ""
    if finals:
        return "", "explicit_final_marker", "empty_after_final_marker"

    if len(text) > LONG_RESPONSE_CHARS:
        first_line = text.splitlines()[0] if text.splitlines() else text
        cand = clean_answer_candidate(first_line)
        if cand and len(cand) <= MAX_LINE_FALLBACK_CHARS:
            return cand, "first_line_fallback_long_response", ""
        return "", "long_response_without_final_marker", "long_response_without_final_marker"

    lines = [clean_answer_candidate(line) for line in text.splitlines() if line.strip()]
    for cand in lines:
        if cand and len(cand) <= MAX_LINE_FALLBACK_CHARS:
            return cand, "line_fallback", ""

    cand = clean_answer_candidate(text)
    if cand:
        return cand, "full_response_fallback", ""
    return "", "empty_after_cleaning", "empty_after_cleaning"


def is_completed_row(row: Dict[str, Any]) -> bool:
    if str(row.get("runtime_error", "")).strip():
        return False
    if not str(row.get("model_answer", "")).strip():
        return False
    return True


def row_key(
    question_id: str,
    variant: str,
    memory_budget_tokens: int,
) -> Tuple[str, str, int]:
    budget_key = int(memory_budget_tokens) if variant == "statecap" else 0
    return (question_id, variant, budget_key)


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
            if not is_completed_row(row):
                continue
            qid = str(row.get("question_id", "")).strip()
            if not qid:
                continue
            variant = str(row.get("variant", "official")).strip() or "official"
            budget = int(row.get("memory_budget_tokens", 0) or 0)
            done.add(row_key(qid, variant, budget))
    return done


def aggregate_output_totals(path: Path) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "rows_written_total": 0,
        "rows_completed_total": 0,
        "rows_runtime_error_total": 0,
        "total_lm_calls": 0,
        "total_lm_input_tokens": 0,
        "total_lm_output_tokens": 0,
        "total_lm_wall_time_sec": 0.0,
        "variants": {},
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
            if is_completed_row(row):
                stats["rows_completed_total"] += 1
            if str(row.get("runtime_error", "")).strip():
                stats["rows_runtime_error_total"] += 1
            stats["total_lm_calls"] += _as_int(row.get("total_lm_calls"))
            stats["total_lm_input_tokens"] += _as_int(row.get("total_lm_input_tokens"))
            stats["total_lm_output_tokens"] += _as_int(row.get("total_lm_output_tokens"))
            lm_wall = row.get("total_lm_wall_time_sec")
            if lm_wall is None:
                lm_wall = row.get("lm_call_wall_time_sec")
            stats["total_lm_wall_time_sec"] += _as_float(lm_wall)
            variant = str(row.get("variant", "unknown") or "unknown")
            per_variant = stats["variants"].setdefault(
                variant,
                {
                    "rows_written": 0,
                    "rows_completed": 0,
                    "runtime_error_rows": 0,
                    "total_lm_wall_time_sec": 0.0,
                },
            )
            per_variant["rows_written"] += 1
            if is_completed_row(row):
                per_variant["rows_completed"] += 1
            if str(row.get("runtime_error", "")).strip():
                per_variant["runtime_error_rows"] += 1
            per_variant["total_lm_wall_time_sec"] += _as_float(lm_wall)
    stats["total_lm_tokens"] = stats["total_lm_input_tokens"] + stats["total_lm_output_tokens"]
    return stats


def extract_rlm_usage_totals(usage: Dict[str, Any]) -> Dict[str, int]:
    calls = 0
    input_tokens = 0
    output_tokens = 0
    model_usage = usage.get("model_usage_summaries") or {}
    if isinstance(model_usage, dict):
        for _, summary in model_usage.items():
            if not isinstance(summary, dict):
                continue
            calls += int(summary.get("total_calls", 0) or 0)
            input_tokens += int(summary.get("total_input_tokens", 0) or 0)
            output_tokens += int(summary.get("total_output_tokens", 0) or 0)
    return {
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def resolve_backend_api_key(backend: str) -> str:
    if backend == "gemini":
        key = (os.getenv("GEMINI_API_KEY") or "").strip()
        if key:
            return key
        key = (os.getenv("GOOGLE_API_KEY") or "").strip()
        if key:
            return key
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in environment.")
    if backend == "openrouter":
        key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
        if key:
            return key
        raise RuntimeError("Set OPENROUTER_API_KEY in environment.")
    raise RuntimeError(f"Unsupported backend for API key resolution: {backend}")


def install_state_cap(counter: TokenCounter, cap_tokens: int, state_var: str) -> None:
    def capped_execute(self: LocalREPL, code: str):  # type: ignore[override]
        global LAST_MEMORY_STATE_TOKENS, LAST_EXECUTE_CALLS, LAST_MEMORY_STATE_PRESENT, LAST_MEMORY_STATE_TEXT
        LAST_EXECUTE_CALLS += 1
        result = ORIG_EXECUTE(self, code)
        try:
            if state_var in self.locals:
                LAST_MEMORY_STATE_PRESENT = True
                txt = str(self.locals.get(state_var, "") or "")
                txt = counter.truncate(txt, cap_tokens, strategy="head")
                self.locals[state_var] = txt
                LAST_MEMORY_STATE_TEXT = txt
                LAST_MEMORY_STATE_TOKENS = counter.count(txt)
            else:
                LAST_MEMORY_STATE_PRESENT = False
                LAST_MEMORY_STATE_TEXT = ""
                LAST_MEMORY_STATE_TOKENS = 0
        except Exception:
            LAST_MEMORY_STATE_PRESENT = False
            LAST_MEMORY_STATE_TEXT = ""
            LAST_MEMORY_STATE_TOKENS = 0
        return result

    LocalREPL.execute_code = capped_execute  # type: ignore[assignment]


def uninstall_state_cap() -> None:
    LocalREPL.execute_code = ORIG_EXECUTE  # type: ignore[assignment]


def install_rlm_backend_call_timer(backend: str) -> None:
    def timed_completion(self: RLMSGeminiClient, prompt: Any, model: str | None = None):  # type: ignore[override]
        global TOTAL_RLM_GEMINI_CALL_WALL_TIME_SEC
        call_started = time.perf_counter()
        try:
            return ORIG_RLMS_GEMINI_COMPLETION(self, prompt, model=model)
        finally:
            TOTAL_RLM_GEMINI_CALL_WALL_TIME_SEC += max(0.0, time.perf_counter() - call_started)

    def timed_openai_completion(self: RLMSOpenAIClient, prompt: Any, model: str | None = None):  # type: ignore[override]
        global TOTAL_RLM_GEMINI_CALL_WALL_TIME_SEC
        call_started = time.perf_counter()
        try:
            return ORIG_RLMS_OPENAI_COMPLETION(self, prompt, model=model)
        finally:
            TOTAL_RLM_GEMINI_CALL_WALL_TIME_SEC += max(0.0, time.perf_counter() - call_started)

    if backend == "gemini":
        RLMSGeminiClient.completion = timed_completion  # type: ignore[assignment]
    elif backend == "openrouter":
        RLMSOpenAIClient.completion = timed_openai_completion  # type: ignore[assignment]
    else:
        raise RuntimeError(f"Unsupported backend timer install: {backend}")


def uninstall_rlm_backend_call_timer(backend: str) -> None:
    if backend == "gemini":
        RLMSGeminiClient.completion = ORIG_RLMS_GEMINI_COMPLETION  # type: ignore[assignment]
    elif backend == "openrouter":
        RLMSOpenAIClient.completion = ORIG_RLMS_OPENAI_COMPLETION  # type: ignore[assignment]


def completion_with_retries(
    rlm_obj: RLM,
    prompt: Any,
    root_prompt: str,
    retries: int,
    sleep_sec: float = 2.0,
):
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            comp = rlm_obj.completion(prompt=prompt, root_prompt=root_prompt)
            if comp is None:
                raise RuntimeError("RLM returned None completion object.")
            return comp
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(sleep_sec * (attempt + 1))
                continue
            break
    raise RuntimeError(f"RLM completion failed after retries: {last_error}")


def build_context_payload(
    docs: Sequence[Dict[str, Any]],
    counter: TokenCounter,
    max_doc_tokens: int,
    doc_truncate_strategy: str,
    max_doc_chars: int,
) -> Tuple[List[str], List[str], int, int]:
    payload: List[str] = []
    used_doc_ids: List[str] = []
    char_truncations = 0
    token_truncations = 0
    for d in docs:
        doc_id = str(d.get("doc_id", "")).strip()
        if not doc_id:
            continue
        txt = str(d.get("text", "") or "")
        if max_doc_chars > 0 and len(txt) > max_doc_chars:
            txt = txt[:max_doc_chars]
            char_truncations += 1
        if max_doc_tokens > 0 and counter.count(txt) > max_doc_tokens:
            txt = counter.truncate(txt, max_doc_tokens, strategy=doc_truncate_strategy)
            token_truncations += 1
        block_parts = [f"doc_id: {doc_id}", "text:", txt]
        block = "\n".join(block_parts)
        payload.append(block)
        used_doc_ids.append(doc_id)
    return payload, used_doc_ids, char_truncations, token_truncations


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="RLM baseline runner with official and memory-capped variants."
    )
    ap.add_argument("--dataset_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--run_log_jsonl", default="")
    ap.add_argument("--variant", choices=["official", "statecap", "both"], default="both")
    ap.add_argument("--memory_budget_tokens", type=int, default=2000)
    ap.add_argument("--state_variable_name", default="memory_state")
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
    if args.variant in {"statecap", "both"}:
        if args.memory_budget_tokens <= 0:
            raise ValueError("--memory_budget_tokens must be > 0 for statecap/both variants.")
    else:
        if args.memory_budget_tokens < 0:
            raise ValueError("--memory_budget_tokens must be >= 0 for official variant.")
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
    if not args.state_variable_name.strip():
        raise ValueError("--state_variable_name must be non-empty.")
    if args.backend == "openrouter" and not args.openrouter_base_url.strip():
        raise ValueError("--openrouter_base_url must be non-empty for openrouter backend.")


def get_variants(arg_variant: str) -> List[str]:
    if arg_variant == "both":
        return ["official", "statecap"]
    return [arg_variant]


def get_rlms_metadata() -> Dict[str, Any]:
    try:
        dist = importlib.metadata.distribution("rlms")
    except importlib.metadata.PackageNotFoundError:
        return {"package": "rlms", "installed": False}
    meta = dist.metadata
    project_urls = meta.get_all("Project-URL") or []
    return {
        "package": "rlms",
        "installed": True,
        "version": dist.version,
        "summary": meta.get("Summary", ""),
        "author_email": meta.get("Author-email", ""),
        "project_urls": project_urls,
    }


def write_run_log(path: Optional[Path], event: str, payload: Dict[str, Any]) -> None:
    if path is None:
        return
    row = {"ts": time.time(), "event": event, **payload}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    global LAST_MEMORY_STATE_TOKENS, LAST_EXECUTE_CALLS, LAST_MEMORY_STATE_PRESENT, LAST_MEMORY_STATE_TEXT
    global TOTAL_RLM_GEMINI_CALL_WALL_TIME_SEC
    args = parse_args()
    validate_args(args)
    load_dotenv()

    dataset_path = Path(args.dataset_jsonl)
    out_path = Path(args.out_jsonl)
    run_log_path = Path(args.run_log_jsonl) if args.run_log_jsonl else None
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset jsonl not found: {dataset_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if run_log_path:
        run_log_path.parent.mkdir(parents=True, exist_ok=True)

    rows_all = list(iter_jsonl(dataset_path))
    rows = rows_all[args.start_index :]
    if args.limit > 0:
        rows = rows[: args.limit]

    done_keys = load_done_keys(out_path) if args.resume else set()
    mode = "a" if args.resume else "w"
    if not args.resume and run_log_path and run_log_path.exists():
        run_log_path.unlink()

    token_counter = TokenCounter("cl100k_base")
    variants = get_variants(args.variant)
    if not args.dry_run:
        backend_api_key = resolve_backend_api_key(args.backend)
        install_rlm_backend_call_timer(args.backend)
    else:
        backend_api_key = ""
    TOTAL_RLM_GEMINI_CALL_WALL_TIME_SEC = 0.0

    processed = 0
    total_targets = len(rows) * len(variants)

    try:
        with out_path.open(mode, encoding="utf-8") as fout:
            for variant in variants:
                if variant == "statecap":
                    install_state_cap(
                        counter=token_counter,
                        cap_tokens=args.memory_budget_tokens,
                        state_var=args.state_variable_name,
                    )
                try:
                    if not args.dry_run:
                        system_prompt = None
                        if variant == "statecap":
                            cap_addendum = (
                                "ADDITIONAL CONSTRAINT FOR THIS RUN:\n"
                                f"Maintain a string variable named {args.state_variable_name} in the REPL as retained working memory.\n"
                                f"After each update, keep {args.state_variable_name} at most {args.memory_budget_tokens} tokens.\n"
                                f"Use {args.state_variable_name} as your compressed evidence buffer before final answer."
                            )
                            system_prompt = RLM_SYSTEM_PROMPT + "\n\n" + cap_addendum

                        rlm_kwargs: Dict[str, Any] = {
                            "backend": args.backend,
                            "backend_kwargs": {"model_name": args.model, "api_key": backend_api_key},
                            "environment": "local",
                            "max_depth": args.max_depth,
                            "max_iterations": args.max_iterations,
                            "verbose": bool(args.verbose),
                        }
                        if args.backend == "openrouter":
                            rlm_kwargs["backend_kwargs"]["base_url"] = args.openrouter_base_url
                        if system_prompt is not None:
                            rlm_kwargs["custom_system_prompt"] = system_prompt
                        rlm = RLM(**rlm_kwargs)
                    else:
                        rlm = None

                    for row in rows:
                        qid = str(row.get("question_id", "")).strip()
                        if not qid:
                            continue
                        key = row_key(qid, variant, args.memory_budget_tokens)
                        if key in done_keys:
                            continue

                        LAST_MEMORY_STATE_TOKENS = 0
                        LAST_EXECUTE_CALLS = 0
                        LAST_MEMORY_STATE_PRESENT = False
                        LAST_MEMORY_STATE_TEXT = ""

                        question = str(row.get("question", ""))
                        gold_answer = str(row.get("gold_answer", ""))
                        docs = list(row.get("docs") or row.get("stream_docs") or [])
                        if args.max_docs_per_query > 0:
                            docs = docs[: args.max_docs_per_query]

                        (
                            context_payload,
                            context_doc_ids,
                            doc_char_truncations,
                            doc_token_truncations,
                        ) = build_context_payload(
                            docs=docs,
                            counter=token_counter,
                            max_doc_tokens=args.max_doc_tokens,
                            doc_truncate_strategy=args.doc_truncate_strategy,
                            max_doc_chars=args.max_doc_chars,
                        )
                        if not context_payload:
                            context_payload = ["doc_id: \ntext:\nEmpty context."]

                        started = time.time()
                        model_answer = ""
                        raw_model_answer = ""
                        answer_extraction_mode = ""
                        answer_extraction_error = ""
                        usage: Dict[str, Any] = {}
                        runtime_error = ""
                        skip_reason = ""
                        lm_call_wall_before = TOTAL_RLM_GEMINI_CALL_WALL_TIME_SEC
                        lm_call_wall_time_sec = 0.0

                        write_run_log(
                            run_log_path,
                            "example_start",
                            {
                                "question_id": qid,
                                "variant": variant,
                                "memory_budget_tokens": args.memory_budget_tokens
                                if variant == "statecap"
                                else 0,
                                "num_context_docs_loaded": len(context_payload),
                                "doc_token_truncations": doc_token_truncations,
                            },
                        )

                        if args.dry_run:
                            raw_model_answer = "DRY_RUN"
                            model_answer = "DRY_RUN"
                            answer_extraction_mode = "dry_run"
                        else:
                            try:
                                if rlm is None:
                                    raise RuntimeError("RLM object is not initialized.")
                                completion_obj = completion_with_retries(
                                    rlm_obj=rlm,
                                    prompt=context_payload,
                                    root_prompt=question,
                                    retries=args.completion_retries,
                                )
                                raw_model_answer = str(
                                    getattr(completion_obj, "response", "") or ""
                                ).strip()
                                (
                                    model_answer,
                                    answer_extraction_mode,
                                    answer_extraction_error,
                                ) = extract_final_answer(raw_model_answer)
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
                            float(TOTAL_RLM_GEMINI_CALL_WALL_TIME_SEC - lm_call_wall_before),
                        )

                        latency_sec = round(time.time() - started, 3)
                        is_exact = exact_match(model_answer, gold_answer) if model_answer else False
                        usage_totals = extract_rlm_usage_totals(usage)
                        stream_doc_tokens = [token_counter.count(str(d.get("text", ""))) for d in docs]

                        out_row = {
                            "question_id": qid,
                            "question": question,
                            "gold_answer": gold_answer,
                            "variant": variant,
                            "memory_budget_tokens": args.memory_budget_tokens if variant == "statecap" else 0,
                            "state_variable_name": args.state_variable_name if variant == "statecap" else "",
                            "model": args.model,
                            "max_depth": args.max_depth,
                            "max_iterations": args.max_iterations,
                            "model_answer": model_answer,
                            "raw_model_answer": raw_model_answer,
                            "answer_extraction_mode": answer_extraction_mode,
                            "answer_extraction_error": answer_extraction_error,
                            "is_exact_match": bool(is_exact),
                            "row_failed": bool(runtime_error),
                            "update_attempt_count": 0,
                            "update_step_count": 0,
                            "latency_sec": latency_sec,
                            "runtime_error": runtime_error,
                            "skip_reason": skip_reason,
                            "num_stream_docs": len(docs),
                            "num_context_docs_loaded": len(context_payload),
                            "context_doc_ids": context_doc_ids,
                            "doc_truncate_strategy": args.doc_truncate_strategy,
                            "max_doc_tokens": args.max_doc_tokens,
                            "doc_char_truncations": doc_char_truncations,
                            "doc_token_truncations": doc_token_truncations,
                            "stream_doc_tokens": stream_doc_tokens,
                            "stream_total_tokens": sum(stream_doc_tokens),
                            "rlm_usage": usage,
                            "rlm_usage_totals": usage_totals,
                            "total_lm_calls": usage_totals["calls"],
                            "total_lm_input_tokens": usage_totals["input_tokens"],
                            "total_lm_output_tokens": usage_totals["output_tokens"],
                            "total_lm_tokens": usage_totals["total_tokens"],
                            "lm_call_wall_time_sec": round(float(lm_call_wall_time_sec), 6),
                            "total_lm_wall_time_sec": round(float(lm_call_wall_time_sec), 6),
                            "answer_tokens": token_counter.count(model_answer),
                            "memory_text": LAST_MEMORY_STATE_TEXT if variant == "statecap" else "",
                            "memory_tokens": LAST_MEMORY_STATE_TOKENS if variant == "statecap" else 0,
                            "execute_calls": LAST_EXECUTE_CALLS if variant == "statecap" else 0,
                            "memory_state_present": (
                                bool(LAST_MEMORY_STATE_PRESENT) if variant == "statecap" else False
                            ),
                            "memory_utilization_ratio": (
                                float(LAST_MEMORY_STATE_TOKENS) / float(args.memory_budget_tokens)
                                if variant == "statecap" and args.memory_budget_tokens > 0
                                else 0.0
                            ),
                            "dry_run": bool(args.dry_run),
                        }
                        row_metadata.attach_sample_metadata(out_row, row)
                        fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                        done_keys.add(key)

                        write_run_log(
                            run_log_path,
                            "example_done",
                            {
                                "question_id": qid,
                                "variant": variant,
                                "memory_budget_tokens": out_row["memory_budget_tokens"],
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
                                f"[progress] processed={processed}/{total_targets} variant={variant} "
                                f"qid={qid} total_lm_tokens={usage_totals['total_tokens']}",
                                flush=True,
                            )
                finally:
                    if variant == "statecap":
                        uninstall_state_cap()
    finally:
        if not args.dry_run:
            uninstall_rlm_backend_call_timer(args.backend)

    totals = aggregate_output_totals(out_path)
    manifest = {
        "dataset_jsonl": str(dataset_path),
        "out_jsonl": str(out_path),
        "run_log_jsonl": str(run_log_path) if run_log_path else "",
        "variant_arg": args.variant,
        "variants_executed": variants,
        "memory_budget_tokens": args.memory_budget_tokens,
        "state_variable_name": args.state_variable_name,
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
        "official_fidelity": {
            "official_variant_uses_default_system_prompt": True,
            "official_variant_calls": f"RLM(backend={args.backend}).completion(prompt=context_payload, root_prompt=question)",
            "context_payload_format": "List[str] docs (doc_id/text)",
            "doc_cap_policy": {
                "max_doc_tokens": args.max_doc_tokens,
                "doc_truncate_strategy": args.doc_truncate_strategy,
                "max_doc_chars_deprecated": args.max_doc_chars,
            },
            "statecap_delta": (
                "statecap adds prompt addendum for memory_state and hard truncates LocalREPL variable "
                f"'{args.state_variable_name}' after each execute_code."
            ),
            "statecap_caveat": (
                "Statecap constrains retained REPL variable state, but does not prove strict parity with "
                "streaming-memory observation constraints."
            ),
        },
        "rlms_metadata": get_rlms_metadata(),
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
