from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from synthetic_fs_env import ANSWER_PREFIX, load_index, normalize_answer


def call_chat_model(
    *,
    backend: str,
    model: str,
    base_url: str,
    api_key_env: str,
    messages: Sequence[dict[str, str]],
    response_json: bool = False,
    max_output_tokens: int | None = None,
    temperature: float = 0.0,
    max_retries: int = 6,
) -> str:
    api_key = os.getenv(api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key in env var {api_key_env}")

    for attempt in range(max_retries):
        try:
            if backend == "openrouter":
                payload: dict[str, Any] = {
                    "model": model,
                    "messages": list(messages),
                    "temperature": temperature,
                }
                if max_output_tokens is not None and max_output_tokens > 0:
                    payload["max_tokens"] = max_output_tokens
                if response_json:
                    payload["response_format"] = {"type": "json_object"}
                req = urllib.request.Request(
                    base_url.rstrip("/") + "/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=180) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                choices = body.get("choices") or []
                if not choices:
                    raise RuntimeError(f"Response missing choices: {body}")
                content = (choices[0].get("message") or {}).get("content", "")
                return str(content).strip()

            if backend == "gemini":
                model_path = model if model.startswith("models/") else f"models/{model}"
                system_texts = [
                    m["content"]
                    for m in messages
                    if m.get("role") == "system" and m.get("content")
                ]
                contents = []
                for message in messages:
                    role = message.get("role", "user")
                    if role == "system":
                        continue
                    contents.append(
                        {
                            "role": "model" if role == "assistant" else "user",
                            "parts": [{"text": message.get("content", "")}],
                        }
                    )
                payload = {
                    "contents": contents,
                    "generationConfig": {"temperature": temperature},
                }
                if max_output_tokens is not None and max_output_tokens > 0:
                    payload["generationConfig"]["maxOutputTokens"] = max_output_tokens
                if response_json:
                    payload["generationConfig"]["responseMimeType"] = "application/json"
                if system_texts:
                    payload["systemInstruction"] = {
                        "parts": [{"text": "\n\n".join(system_texts)}]
                    }
                req = urllib.request.Request(
                    f"{base_url.rstrip('/')}/{model_path}:generateContent",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": api_key,
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=180) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                candidates = body.get("candidates") or []
                if not candidates:
                    raise RuntimeError(f"Response missing candidates: {body}")
                parts = ((candidates[0].get("content") or {}).get("parts") or [])
                return "".join(str(part.get("text", "")) for part in parts).strip()

            raise ValueError(f"Unsupported backend: {backend}")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code not in {408, 409, 429, 500, 502, 503, 504} or attempt == max_retries - 1:
                raise RuntimeError(f"HTTP error {e.code}: {body[:500]}") from e
        except urllib.error.URLError:
            if attempt == max_retries - 1:
                raise
        time.sleep(min(60.0, 2.0 * (2**attempt)))
    raise RuntimeError("unreachable")


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            line for line in cleaned.splitlines() if not line.strip().startswith("```")
        ).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def extract_answer(text: str) -> str:
    stripped = text.strip()
    for line in stripped.splitlines():
        line = line.strip()
        if line.startswith(ANSWER_PREFIX):
            return line[len(ANSWER_PREFIX) :].strip()
    return stripped


def read_doc(datum: dict[str, Any], file_info: dict[str, Any]) -> str:
    path = Path(datum["agent_query_dir"]) / file_info["relative_path"]
    return path.read_text(encoding="utf-8", errors="replace")


def stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256("::".join([str(seed), *parts]).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def summarize_stream(
    *,
    datum: dict[str, Any],
    sample_idx: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rng = random.Random(stable_seed(args.seed, datum["question_id"], str(sample_idx)))
    files = list(datum["files"])
    rng.shuffle(files)
    running_summary = ""
    doc_summaries: list[dict[str, str]] = []
    total_doc_chars = 0

    for file_info in files:
        doc_text = read_doc(datum, file_info)
        total_doc_chars += len(doc_text)
        truncated_doc = doc_text[: args.doc_max_chars]
        if args.question_aware_summary:
            question_block = f"Target question:\n{datum['question']}\n\n"
        else:
            question_block = ""
        prompt = (
            "You are maintaining a compact memory for a streaming QA baseline. "
            "You see raw documents one at a time. Update the running summary using only supported facts. "
            "Return compact JSON with keys doc_summary and running_summary.\n\n"
            f"{question_block}"
            f"Current running summary:\n{running_summary or '(empty)'}\n\n"
            "Next document metadata:\n"
            f"doc_id: {file_info.get('doc_id', '')}\n"
            f"url: {file_info.get('url', '')}\n"
            f"path: {file_info.get('relative_path', '')}\n\n"
            f"Next document text:\n{truncated_doc}"
        )
        response = call_chat_model(
            backend=args.model_backend,
            model=args.model,
            base_url=args.model_base_url,
            api_key_env=args.model_api_key_env,
            messages=[
                {
                    "role": "system",
                    "content": "Return only valid compact JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            response_json=True,
            max_output_tokens=args.summary_max_output_tokens,
            temperature=args.temperature,
        )
        try:
            parsed = parse_json_object(response)
            doc_summary = str(parsed.get("doc_summary", "")).strip()
            running_summary = str(parsed.get("running_summary", "")).strip() or running_summary
        except Exception:
            doc_summary = response.strip()
            running_summary = (running_summary + "\n" + doc_summary).strip()
        doc_summaries.append(
            {
                "doc_id": str(file_info.get("doc_id", "")),
                "path": str(file_info.get("relative_path", "")),
                "summary": doc_summary,
            }
        )

    concat_doc_summaries = "\n\n".join(
        f"[doc_id={item['doc_id']} path={item['path']}]\n{item['summary']}"
        for item in doc_summaries
    )
    answer_context = (
        "[Final running summary]\n"
        f"{running_summary}\n\n"
        "[Per-document summaries]\n"
        f"{concat_doc_summaries}"
    )
    answer_prompt = (
        "Answer the target question using only the summaries below. "
        f"Return exactly one line in the format: {ANSWER_PREFIX} <answer>\n\n"
        f"Question:\n{datum['question']}\n\n"
        f"Summaries:\n{answer_context}"
    )
    answer_text = call_chat_model(
        backend=args.model_backend,
        model=args.model,
        base_url=args.model_base_url,
        api_key_env=args.model_api_key_env,
        messages=[
            {
                "role": "system",
                "content": "You answer questions from provided summaries only.",
            },
            {"role": "user", "content": answer_prompt},
        ],
        response_json=False,
        max_output_tokens=args.answer_max_output_tokens,
        temperature=args.temperature,
    )
    prediction = extract_answer(answer_text)
    exact_match = float(normalize_answer(prediction) == normalize_answer(datum["gold_answer"]))
    judge_score = 0.0
    judge_used = 0.0
    if exact_match == 0.0:
        judge_used = 1.0
        judge_prompt = (
            "You are judging whether a predicted answer should count as correct for a question. "
            "Return only JSON in the form {\"correct\": 0 or 1}.\n\n"
            f"Question: {datum['question']}\n"
            f"Gold answer: {datum['gold_answer']}\n"
            f"Predicted answer: {prediction}\n\n"
            "Count semantically equivalent answers as correct even if formatting differs. "
            "Be strict about factual mismatch."
        )
        judge_response = call_chat_model(
            backend=args.judge_backend,
            model=args.judge_model,
            base_url=args.judge_base_url,
            api_key_env=args.judge_api_key_env,
            messages=[{"role": "user", "content": judge_prompt}],
            response_json=True,
            max_output_tokens=args.judge_max_output_tokens,
            temperature=0.0,
        )
        try:
            judge_score = 1.0 if float(parse_json_object(judge_response).get("correct", 0)) >= 1.0 else 0.0
        except Exception:
            judge_score = 0.0
    correct = max(exact_match, judge_score)
    return {
        "question_id": datum["question_id"],
        "sample_idx": sample_idx,
        "question": datum["question"],
        "gold_answer": datum["gold_answer"],
        "prediction": prediction,
        "raw_answer_text": answer_text,
        "exact_match": exact_match,
        "judge_used": judge_used,
        "judge_score": judge_score,
        "correct": correct,
        "num_docs": len(files),
        "doc_order": [str(f.get("doc_id", "")) for f in files],
        "raw_doc_chars_read": total_doc_chars,
        "approx_raw_doc_tokens_read": total_doc_chars / 4.0,
        "final_summary": running_summary,
        "doc_summaries": doc_summaries,
    }


def load_existing(path: Path) -> set[tuple[str, int]]:
    seen: set[tuple[str, int]] = set()
    if not path.exists():
        return seen
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            seen.add((str(row["question_id"]), int(row["sample_idx"])))
    return seen


def write_outputs(out_dir: Path, records: list[dict[str, Any]], k: int) -> None:
    by_qid: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_qid.setdefault(str(record["question_id"]), []).append(record)

    per_question_path = out_dir / f"streaming_summary_pass_at_{k}_per_question.csv"
    with per_question_path.open("w", newline="") as f:
        fieldnames = [
            "question_id",
            "num_samples",
            "pass_at_k_correct",
            "pass_at_k_exact",
            "mean_correct",
            "mean_exact_match",
            "mean_raw_doc_tokens_read",
            "mean_num_docs",
            "predictions",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for qid, rows in sorted(by_qid.items(), key=lambda kv: int(kv[0])):
            writer.writerow(
                {
                    "question_id": qid,
                    "num_samples": len(rows),
                    "pass_at_k_correct": float(any(r["correct"] > 0.0 for r in rows)),
                    "pass_at_k_exact": float(any(r["exact_match"] > 0.0 for r in rows)),
                    "mean_correct": sum(float(r["correct"]) for r in rows) / len(rows),
                    "mean_exact_match": sum(float(r["exact_match"]) for r in rows) / len(rows),
                    "mean_raw_doc_tokens_read": sum(
                        float(r["approx_raw_doc_tokens_read"]) for r in rows
                    )
                    / len(rows),
                    "mean_num_docs": sum(float(r["num_docs"]) for r in rows) / len(rows),
                    "predictions": "; ".join(str(r["prediction"]) for r in rows),
                }
            )

    summary = {
        "k": k,
        "questions": len(by_qid),
        "questions_with_at_least_k": sum(1 for rows in by_qid.values() if len(rows) >= k),
        "mean_trajectory_correct": sum(float(r["correct"]) for r in records) / len(records),
        "mean_trajectory_exact_match": sum(float(r["exact_match"]) for r in records)
        / len(records),
        "pass_at_k_correct": sum(
            float(any(r["correct"] > 0.0 for r in rows)) for rows in by_qid.values()
        )
        / len(by_qid),
        "pass_at_k_exact": sum(
            float(any(r["exact_match"] > 0.0 for r in rows)) for rows in by_qid.values()
        )
        / len(by_qid),
        "mean_raw_doc_tokens_read": sum(
            float(r["approx_raw_doc_tokens_read"]) for r in records
        )
        / len(records),
        "mean_num_docs": sum(float(r["num_docs"]) for r in records) / len(records),
    }
    (out_dir / f"streaming_summary_pass_at_{k}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-jsonl", default="../tinker_fs_qa/train_q50_nonexcluded_fs/index.jsonl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model-backend", default="openrouter")
    parser.add_argument("--model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--model-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--model-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--judge-backend", default="gemini")
    parser.add_argument("--judge-model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--judge-base-url", default="https://generativelanguage.googleapis.com/v1beta")
    parser.add_argument("--judge-api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--judge-max-output-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--doc-max-chars", type=int, default=24000)
    parser.add_argument("--summary-max-output-tokens", type=int, default=900)
    parser.add_argument("--answer-max-output-tokens", type=int, default=128)
    parser.add_argument("--question-aware-summary", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / f"streaming_summary_pass_at_{args.k}_trajectories.jsonl"
    rows = load_index(Path(args.index_jsonl))
    if args.limit > 0:
        rows = rows[: args.limit]

    seen = load_existing(traj_path)
    with traj_path.open("a") as f:
        for datum in rows:
            for sample_idx in range(args.k):
                key = (str(datum["question_id"]), sample_idx)
                if key in seen:
                    continue
                record = summarize_stream(datum=datum, sample_idx=sample_idx, args=args)
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                f.flush()
                seen.add(key)
                print(
                    "qid",
                    datum["question_id"],
                    "sample",
                    sample_idx,
                    "correct",
                    record["correct"],
                    "exact",
                    record["exact_match"],
                    flush=True,
                )

    records = []
    with traj_path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    write_outputs(out_dir, records, args.k)
    print(json.dumps(json.loads((out_dir / f"streaming_summary_pass_at_{args.k}_summary.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
