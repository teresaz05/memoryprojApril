from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SIZE_METRICS = [
    "correct",
    "exact_match",
    "synthetic_read_count",
    "answerer_synthetic_chars_read",
    "answerer_synthetic_tokens_read",
    "answerer_total_tokens_read",
    "total_synthetic_files",
    "num_active_files",
    "num_active_clusters",
    "num_active_merges",
    "total_operations",
    "turns_per_episode",
    "total_ob_tokens",
    "total_ac_tokens",
    "builder_turns_seen",
    "answerer_steps",
    "answerer_evaluated",
    "step_tool_errors",
    "step_tool_errors_create_cluster",
    "step_tool_errors_create_clusters",
]


def load_qids(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    qids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qids.append(str(row.get("question_id", "")))
    return qids


def final_metrics(row: dict[str, Any]) -> dict[str, float]:
    steps = row.get("steps") or []
    if not steps:
        return {}
    metrics = steps[-1].get("metrics") or {}
    return {
        str(key): float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }


def rollout_path(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("iteration_*/eval_test_rollout_summaries.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No eval_test_rollout_summaries.jsonl under {run_dir}")
    return matches[-1]


def read_trajectories(run_dir: Path, k: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    order: list[tuple[int, int]] = []
    path = rollout_path(run_dir)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        metrics = final_metrics(row)
        if not metrics:
            continue
        batch_index = int(metrics.get("training_batch_index", 1.0))
        group_idx = int(row.get("group_idx", 0))
        key = (batch_index, group_idx)
        if key not in grouped:
            order.append(key)
        grouped[key].append(
            {
                "batch_index": batch_index,
                "group_idx": group_idx,
                "traj_idx": int(row.get("traj_idx", len(grouped[key]))),
                "metrics": metrics,
            }
        )
    trajectories: list[dict[str, Any]] = []
    for question_ordinal, key in enumerate(sorted(order)):
        selected = sorted(grouped[key], key=lambda item: item["traj_idx"])[:k]
        for item in selected:
            item["question_ordinal"] = question_ordinal
            trajectories.append(item)
    return trajectories


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(label: str, trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    metrics_list = [item["metrics"] for item in trajectories]
    question_groups: dict[int, list[dict[str, float]]] = defaultdict(list)
    for item in trajectories:
        question_groups[int(item["question_ordinal"])].append(item["metrics"])

    row: dict[str, Any] = {
        "label": label,
        "questions": len(question_groups),
        "trajectories": len(trajectories),
        "pass_at_k_correct": mean(
            [float(any(m.get("correct", 0.0) > 0.0 for m in group)) for group in question_groups.values()]
        ),
        "pass_at_k_exact": mean(
            [float(any(m.get("exact_match", 0.0) > 0.0 for m in group)) for group in question_groups.values()]
        ),
    }
    for metric in SIZE_METRICS:
        row[f"mean_{metric}"] = mean([float(m.get(metric, 0.0)) for m in metrics_list])
        row[f"max_{metric}"] = max([float(m.get(metric, 0.0)) for m in metrics_list], default=0.0)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare rollout size/trajectory metrics across runs.")
    parser.add_argument("--k", type=int, default=2, help="Trajectories per question to include.")
    parser.add_argument("--eval-index-jsonl", type=Path, default=None, help="Optional qid labels.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "RUN_DIR"),
        required=True,
        help="Run label and run directory. Can be repeated.",
    )
    args = parser.parse_args()

    qids = load_qids(args.eval_index_jsonl)
    summaries: list[dict[str, Any]] = []
    per_traj_rows: list[dict[str, Any]] = []

    for label, run_dir_str in args.run:
        run_dir = Path(run_dir_str)
        trajectories = read_trajectories(run_dir, args.k)
        summaries.append(summarize(label, trajectories))
        for item in trajectories:
            metrics = item["metrics"]
            ordinal = int(item["question_ordinal"])
            row: dict[str, Any] = {
                "label": label,
                "question_ordinal": ordinal,
                "question_id": qids[ordinal] if ordinal < len(qids) else "",
                "batch_index": item["batch_index"],
                "group_idx": item["group_idx"],
                "traj_idx": item["traj_idx"],
            }
            for metric in SIZE_METRICS:
                row[metric] = float(metrics.get(metric, 0.0))
            per_traj_rows.append(row)

    print(json.dumps(summaries, indent=2, sort_keys=True))

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "rollout_size_summary.json").write_text(
            json.dumps(summaries, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if per_traj_rows:
            with (args.out_dir / "rollout_size_per_trajectory.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(per_traj_rows[0].keys()))
                writer.writeheader()
                writer.writerows(per_traj_rows)


if __name__ == "__main__":
    main()
