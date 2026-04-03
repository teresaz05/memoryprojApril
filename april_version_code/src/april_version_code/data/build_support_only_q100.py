#!/usr/bin/env python3
"""Build the q100 support-only BrowseComp+ subset used by the current experiment matrix.

This is the same selection logic as the original helper script, but the defaults point at the
copied data inside ``april_version_code/data`` so the package is self-contained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PACKAGE_ROOT / "data" / "source" / "browsecomp_plus" / "controlled_support_only_shuffled_q799.jsonl"
DEFAULT_OUTPUT = PACKAGE_ROOT / "data" / "derived" / "support_only" / "support_only_q100_manydocs.jsonl"
DEFAULT_QIDS = PACKAGE_ROOT / "data" / "derived" / "support_only" / "support_only_q100_manydocs_selected_qids.json"
DEFAULT_MANIFEST = PACKAGE_ROOT / "data" / "derived" / "support_only" / "support_only_q100_manydocs.manifest.json"


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield dictionary rows from a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write rows back to JSONL, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def support_doc_count(row: Dict[str, Any]) -> int:
    """Use the precomputed count when available, otherwise count docs directly."""
    value = row.get("num_support_docs")
    if value is not None:
        return safe_int(value, 0)
    return len(list(row.get("docs") or []))


def qid_sort_key(row: Dict[str, Any]) -> Any:
    """Sort numeric question ids numerically and all other ids lexicographically."""
    qid = str(row.get("question_id", "")).strip()
    return int(qid) if qid.isdigit() else qid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the support-only 100-query BrowseComp+ subset used by the April package. "
            "Rows must have at least four support docs, and rows with more support docs are "
            "preferred first."
        )
    )
    parser.add_argument("--in-jsonl", default=str(DEFAULT_INPUT))
    parser.add_argument("--out-jsonl", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--out-selected-qids-json", default=str(DEFAULT_QIDS))
    parser.add_argument("--out-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--min-support-docs", type=int, default=4)
    parser.add_argument("--num-queries", type=int, default=100)
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
    eligible = [row for row in rows_all if support_doc_count(row) >= args.min_support_docs]
    if len(eligible) < args.num_queries:
        raise ValueError(
            f"Requested {args.num_queries} queries but only found {len(eligible)} eligible rows "
            f"with support_doc_count >= {args.min_support_docs}."
        )

    # The selection policy matches the original experiment setup exactly.
    selected = sorted(
        eligible,
        key=lambda row: (-support_doc_count(row), qid_sort_key(row)),
    )[: args.num_queries]

    selected_qids = [str(row.get("question_id", "")).strip() for row in selected]
    support_counts = [support_doc_count(row) for row in selected]

    write_jsonl(out_path, selected)
    qids_path.parent.mkdir(parents=True, exist_ok=True)
    qids_path.write_text(json.dumps(selected_qids, indent=2), encoding="utf-8")

    manifest = {
        "source_jsonl": str(in_path),
        "out_jsonl": str(out_path),
        "out_selected_qids_json": str(qids_path),
        "selection_policy": (
            "support-only rows from q799 with support_doc_count >= min_support_docs, "
            "sorted by support_doc_count descending and question_id ascending, then top num_queries."
        ),
        "min_support_docs": args.min_support_docs,
        "num_queries_requested": args.num_queries,
        "num_rows_source": len(rows_all),
        "num_rows_eligible": len(eligible),
        "num_rows_selected": len(selected),
        "selected_support_doc_count_min": min(support_counts) if support_counts else 0,
        "selected_support_doc_count_max": max(support_counts) if support_counts else 0,
        "selected_support_doc_count_mean": mean(support_counts) if support_counts else 0.0,
        "selected_support_doc_count_median": median(support_counts) if support_counts else 0.0,
        "selected_question_ids": selected_qids,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[done] wrote dataset to {out_path}")
    print(f"[done] wrote selected qids to {qids_path}")
    print(f"[done] eligible_rows={len(eligible)} selected_rows={len(selected)}")


if __name__ == "__main__":
    main()
