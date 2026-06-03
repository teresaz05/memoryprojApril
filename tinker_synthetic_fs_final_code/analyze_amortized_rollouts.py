from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _metrics(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for step in row.get("steps", []) or []:
        metrics = step.get("metrics") or {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                out[key] = float(value)
    out.update(row.get("trajectory_metrics") or {})
    return out


def _construction_tokens(row: dict[str, Any]) -> float:
    total = 0.0
    for step in row.get("steps", []) or []:
        total += float(step.get("ob_len") or 0.0)
        total += float(step.get("ac_len") or 0.0)
    return total


def summarize_run(run_dir: Path) -> dict[str, Any]:
    rollout = run_dir / "iteration_000000" / "eval_test_rollout_summaries.jsonl"
    rows = [json.loads(line) for line in rollout.open(encoding="utf-8") if line.strip()]
    if not rows:
        raise ValueError(f"No rollout rows in {rollout}")

    question_total = 0.0
    correct_sum = 0.0
    exact_sum = 0.0
    construction_costs_per_question = []
    answer_tokens = []
    synthetic_tokens = []
    synthetic_reads = []
    raw_reads = []
    active_files = []
    maturity = []
    builder_turns = []

    for row in rows:
        m = _metrics(row)
        nq = float(m.get("amortization_num_questions", 1.0) or 1.0)
        correct = float(m.get("amortization_correct_sum", m.get("correct", 0.0)) or 0.0)
        exact = float(m.get("amortization_exact_sum", m.get("exact_match", 0.0)) or 0.0)
        question_total += nq
        correct_sum += correct
        exact_sum += exact
        construction_costs_per_question.append(_construction_tokens(row) / max(1.0, nq))
        answer_tokens.append(float(m.get("answerer_total_tokens_read", 0.0) or 0.0))
        synthetic_tokens.append(float(m.get("answerer_synthetic_tokens_read", 0.0) or 0.0))
        synthetic_reads.append(float(m.get("synthetic_read_count", 0.0) or 0.0))
        raw_reads.append(float(m.get("raw_doc_read_count", 0.0) or 0.0))
        active_files.append(float(m.get("num_active_files", 0.0) or 0.0))
        maturity.append(float(m.get("filesystem_maturity_score", 0.0) or 0.0))
        builder_turns.append(float(m.get("builder_turns_seen", 0.0) or 0.0))

    return {
        "run_name": run_dir.name,
        "trajectories": len(rows),
        "questions_scored": int(question_total),
        "amortized_correct": correct_sum / max(1.0, question_total),
        "amortized_exact": exact_sum / max(1.0, question_total),
        "construction_tokens_per_query": _safe_mean(construction_costs_per_question),
        "answer_tokens_per_query": _safe_mean(answer_tokens),
        "synthetic_tokens_per_query": _safe_mean(synthetic_tokens),
        "synthetic_files_read_per_query": _safe_mean(synthetic_reads),
        "raw_files_read_per_query": _safe_mean(raw_reads),
        "active_synthetic_files": _safe_mean(active_files),
        "filesystem_maturity_score": _safe_mean(maturity),
        "builder_turns": _safe_mean(builder_turns),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out-csv", type=Path)
    args = parser.parse_args()

    summaries = [summarize_run(path) for path in args.run_dirs]
    fieldnames = list(summaries[0].keys())
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)
        print(f"wrote {args.out_csv}")
    else:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
