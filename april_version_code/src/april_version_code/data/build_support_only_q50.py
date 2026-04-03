#!/usr/bin/env python3
"""Build the q50 subset by taking the first 50 rows of the package's q100 dataset.

The existing q50 dataset was created by taking the first 50 rows of the already-sorted q100
support-only dataset. This small builder keeps that exact rule explicit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PACKAGE_ROOT / "data" / "derived" / "support_only" / "support_only_q100_manydocs.jsonl"
DEFAULT_OUTPUT = PACKAGE_ROOT / "data" / "derived" / "support_only" / "support_only_q50_manydocs.jsonl"
DEFAULT_QIDS = PACKAGE_ROOT / "data" / "derived" / "support_only" / "support_only_q50_manydocs_selected_qids.json"
DEFAULT_MANIFEST = PACKAGE_ROOT / "data" / "derived" / "support_only" / "support_only_q50_manydocs.manifest.json"


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def support_doc_count(row: Dict[str, Any]) -> int:
    value = row.get("num_support_docs")
    if value is not None:
        try:
            return int(value)
        except Exception:
            pass
    return len(list(row.get("docs") or []))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the q50 support-only dataset by taking the first 50 rows of the q100 dataset."
    )
    parser.add_argument("--in-jsonl", default=str(DEFAULT_INPUT))
    parser.add_argument("--out-jsonl", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--out-selected-qids-json", default=str(DEFAULT_QIDS))
    parser.add_argument("--out-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--num-queries", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    qids_path = Path(args.out_selected_qids_json)
    manifest_path = Path(args.out_manifest)

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    rows_all = list(iter_jsonl(in_path))
    selected = rows_all[: args.num_queries]
    if len(selected) < args.num_queries:
        raise ValueError(f"Only found {len(selected)} rows in {in_path}, need {args.num_queries}.")

    selected_qids = [str(row.get("question_id", "")).strip() for row in selected]
    support_counts = [support_doc_count(row) for row in selected]

    write_jsonl(out_path, selected)
    qids_path.parent.mkdir(parents=True, exist_ok=True)
    qids_path.write_text(json.dumps(selected_qids, indent=2), encoding="utf-8")

    manifest = {
        "source_dataset": str(in_path),
        "selection_policy": "first_50_rows_of_q100_sorted_by_support_doc_count_desc_then_question_id_asc",
        "num_rows": len(selected),
        "support_doc_count_stats": {
            "min": min(support_counts) if support_counts else 0,
            "median": median(support_counts) if support_counts else 0,
            "mean": mean(support_counts) if support_counts else 0.0,
            "max": max(support_counts) if support_counts else 0,
        },
        "output_dataset": str(out_path),
        "selected_qids_json": str(qids_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[done] wrote dataset to {out_path}")
    print(f"[done] wrote selected qids to {qids_path}")


if __name__ == "__main__":
    main()
