from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _final_metrics(row: dict[str, Any]) -> dict[str, float]:
    steps = row.get("steps") or []
    if not steps:
        return {}
    metrics = steps[-1].get("metrics") or {}
    return {str(key): float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def _load_qids(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    qids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qids.append(str(row.get("question_id", "")))
    return qids


def _summary_paths(run_dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for run_dir in run_dirs:
        matches = sorted(run_dir.glob("iteration_*/eval_test_rollout_summaries.jsonl"))
        if not matches:
            raise FileNotFoundError(f"No eval_test_rollout_summaries.jsonl found under {run_dir}")
        paths.extend(matches)
    return paths


def _metric(metrics: dict[str, float], name: str) -> float:
    return float(metrics.get(name, 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute trajectory-level pass@k from rollout summaries.")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Run directories to combine.")
    parser.add_argument("--k", type=int, default=4, help="Number of trajectories per question to score.")
    parser.add_argument("--eval-index-jsonl", type=Path, default=None, help="Optional eval index for qid labels.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Optional output directory.")
    args = parser.parse_args()

    qids = _load_qids(args.eval_index_jsonl)
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    order: list[tuple[int, int]] = []

    for source_idx, path in enumerate(_summary_paths(args.run_dirs)):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                metrics = _final_metrics(row)
                if not metrics:
                    continue
                batch_index = int(metrics.get("training_batch_index", 1.0))
                group_idx = int(row.get("group_idx", 0))
                key = (batch_index, group_idx)
                if key not in grouped:
                    order.append(key)
                grouped[key].append(
                    {
                        "source_idx": source_idx,
                        "path": str(path),
                        "traj_idx": int(row.get("traj_idx", len(grouped[key]))),
                        "metrics": metrics,
                    }
                )

    per_question: list[dict[str, Any]] = []
    for ordinal, key in enumerate(sorted(order)):
        trajectories = sorted(grouped[key], key=lambda item: (item["source_idx"], item["traj_idx"]))[: args.k]
        if not trajectories:
            continue
        metrics_list = [item["metrics"] for item in trajectories]
        correct_values = [_metric(metrics, "correct") for metrics in metrics_list]
        exact_values = [_metric(metrics, "exact_match") for metrics in metrics_list]
        synthetic_reads = [_metric(metrics, "synthetic_read_count") for metrics in metrics_list]
        synthetic_tokens = [_metric(metrics, "answerer_synthetic_tokens_read") for metrics in metrics_list]
        total_tokens = [_metric(metrics, "answerer_total_tokens_read") for metrics in metrics_list]
        row = {
            "question_ordinal": ordinal,
            "question_id": qids[ordinal] if ordinal < len(qids) else "",
            "batch_index": key[0],
            "group_idx": key[1],
            "n_trajectories": len(trajectories),
            "pass_at_k_correct": float(any(value > 0.0 for value in correct_values)),
            "pass_at_k_exact": float(any(value > 0.0 for value in exact_values)),
            "mean_correct": sum(correct_values) / len(correct_values),
            "mean_exact_match": sum(exact_values) / len(exact_values),
            "mean_synthetic_read_count": sum(synthetic_reads) / len(synthetic_reads),
            "mean_answerer_synthetic_tokens_read": sum(synthetic_tokens) / len(synthetic_tokens),
            "mean_answerer_total_tokens_read": sum(total_tokens) / len(total_tokens),
            "trajectory_correct_values": ";".join(f"{value:.6g}" for value in correct_values),
            "trajectory_exact_values": ";".join(f"{value:.6g}" for value in exact_values),
        }
        per_question.append(row)

    if not per_question:
        raise SystemExit("No completed trajectory metrics found.")

    complete = [row for row in per_question if int(row["n_trajectories"]) >= args.k]
    denom_rows = complete if complete else per_question
    summary = {
        "k": args.k,
        "questions": len(per_question),
        "questions_with_at_least_k": len(complete),
        "pass_at_k_correct": sum(row["pass_at_k_correct"] for row in denom_rows) / len(denom_rows),
        "pass_at_k_exact": sum(row["pass_at_k_exact"] for row in denom_rows) / len(denom_rows),
        "mean_trajectory_correct": sum(row["mean_correct"] for row in denom_rows) / len(denom_rows),
        "mean_trajectory_exact_match": sum(row["mean_exact_match"] for row in denom_rows) / len(denom_rows),
        "mean_synthetic_read_count": sum(row["mean_synthetic_read_count"] for row in denom_rows) / len(denom_rows),
        "mean_answerer_synthetic_tokens_read": sum(row["mean_answerer_synthetic_tokens_read"] for row in denom_rows) / len(denom_rows),
        "mean_answerer_total_tokens_read": sum(row["mean_answerer_total_tokens_read"] for row in denom_rows) / len(denom_rows),
    }

    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / f"pass_at_{args.k}_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with (args.out_dir / f"pass_at_{args.k}_per_question.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_question[0].keys()))
            writer.writeheader()
            writer.writerows(per_question)


if __name__ == "__main__":
    main()
