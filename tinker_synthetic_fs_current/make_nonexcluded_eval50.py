#!/usr/bin/env python3
"""Build a 50-question eval split after deleting excluded questions.

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-index", default="train_q830_fs/index.jsonl")
    parser.add_argument("--old-eval-index", default="train_q50_fs/index.jsonl")
    parser.add_argument("--excluded", default="excluded100.jsonl")
    parser.add_argument("--out-dir", default="train_q50_nonexcluded_fs")
    parser.add_argument("--target-size", type=int, default=50)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def qid(row: dict[str, Any]) -> str:
    return str(row["question_id"])


def main() -> None:
    args = parse_args()
    train_index = Path(args.train_index)
    old_eval_index = Path(args.old_eval_index)
    excluded_path = Path(args.excluded)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_rows(train_index)
    old_eval_rows = load_rows(old_eval_index)
    excluded_qids = {qid(row) for row in load_rows(excluded_path)}

    kept_eval = [row for row in old_eval_rows if qid(row) not in excluded_qids]
    kept_qids = {qid(row) for row in kept_eval}
    fillers = [
        row
        for row in train_rows
        if qid(row) not in excluded_qids and qid(row) not in kept_qids
    ]

    needed = args.target_size - len(kept_eval)
    if needed < 0:
        raise SystemExit(
            f"Old eval has {len(kept_eval)} non-excluded rows, above target size {args.target_size}."
        )
    if len(fillers) < needed:
        raise SystemExit(f"Need {needed} filler rows, but only found {len(fillers)}.")

    new_eval = kept_eval + fillers[:needed]
    new_qids = {qid(row) for row in new_eval}
    if len(new_eval) != args.target_size or len(new_qids) != args.target_size:
        raise SystemExit("Corrected eval split does not have the requested unique size.")
    if new_qids & excluded_qids:
        raise SystemExit(f"Corrected eval still overlaps excluded: {sorted(new_qids & excluded_qids)}")

    index_path = out_dir / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for row in new_eval:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    train_qids = {qid(row) for row in train_rows}
    manifest = {
        "source": "old eval split with excluded questions deleted, filled from training index",
        "train_index": str(train_index.resolve()),
        "old_eval_index": str(old_eval_index.resolve()),
        "excluded": str(excluded_path.resolve()),
        "out_dir": str(out_dir.resolve()),
        "index_jsonl": str(index_path.resolve()),
        "target_size": args.target_size,
        "num_examples": len(new_eval),
        "old_eval_count": len(old_eval_rows),
        "old_eval_excluded_overlap_count": len({qid(row) for row in old_eval_rows} & excluded_qids),
        "old_eval_kept_after_excluded_deleted": len(kept_eval),
        "replacement_count": needed,
        "replacement_qids": [qid(row) for row in fillers[:needed]],
        "excluded_overlap_count": 0,
        "train_rows_after_excluded_and_corrected_eval": len(train_qids - excluded_qids - new_qids),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {index_path}")
    print(f"old_eval_count={len(old_eval_rows)}")
    print(f"old_eval_excluded_overlap={manifest['old_eval_excluded_overlap_count']}")
    print(f"kept_old_eval={len(kept_eval)}")
    print(f"replacement_count={needed}")
    print(f"new_eval_count={len(new_eval)}")
    print("new_eval_excluded_overlap=0")
    print(f"train_rows_after_excluded_and_corrected_eval={manifest['train_rows_after_excluded_and_corrected_eval']}")


if __name__ == "__main__":
    main()
