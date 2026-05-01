from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Materialize a support-doc QA dataset as true filesystem examples for Tinker RL. "
            "Each question gets its own directory of read-only document files plus an index.jsonl."
        )
    )
    ap.add_argument("--source_jsonl", required=True, help="Support-only source dataset JSONL.")
    ap.add_argument("--out_dir", required=True, help="Output directory for the filesystem dataset.")
    ap.add_argument("--limit", type=int, default=0, help="Optional max number of questions to export.")
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    return ap.parse_args()


def safe_slug(text: str) -> str:
    chars = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            chars.append(ch)
        else:
            chars.append("_")
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "doc"


def render_doc_file(doc: dict[str, Any]) -> str:
    header = [
        f"DOC_ID: {doc.get('doc_id', '')}",
        f"URL: {doc.get('url', '')}",
        f"IS_EVIDENCE: {bool(doc.get('is_evidence', False))}",
        f"IS_GOLD: {bool(doc.get('is_gold', False))}",
        f"IS_NEGATIVE: {bool(doc.get('is_negative', False))}",
        "",
    ]
    return "\n".join(header) + str(doc.get("text", "")).strip() + "\n"


def main() -> None:
    args = parse_args()
    source = Path(args.source_jsonl)
    out_dir = Path(args.out_dir)
    examples_dir = out_dir / "examples"
    index_path = out_dir / "index.jsonl"
    manifest_path = out_dir / "manifest.json"

    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output directory already exists: {out_dir}. Use --overwrite to replace it.")
        shutil.rmtree(out_dir)

    examples_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with source.open("r", encoding="utf-8") as src, index_path.open("w", encoding="utf-8") as idx:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["question_id"])
            question = str(row["question"]).strip()
            gold_answer = str(row["gold_answer"]).strip()
            docs = list(row.get("docs", []))

            example_dir = examples_dir / f"question_{qid}"
            docs_dir = example_dir / "docs"
            docs_dir.mkdir(parents=True, exist_ok=True)

            file_manifest: list[dict[str, Any]] = []
            for idx_doc, doc in enumerate(docs, start=1):
                suffix = safe_slug(str(doc.get("doc_id", "")))
                file_name = f"doc_{idx_doc:03d}__{suffix}.txt"
                rel_path = f"docs/{file_name}"
                (docs_dir / file_name).write_text(render_doc_file(doc), encoding="utf-8")
                file_manifest.append(
                    {
                        "relative_path": rel_path,
                        "doc_id": doc.get("doc_id"),
                        "url": doc.get("url"),
                        "is_evidence": bool(doc.get("is_evidence", False)),
                        "is_gold": bool(doc.get("is_gold", False)),
                        "is_negative": bool(doc.get("is_negative", False)),
                    }
                )

            example_record = {
                "question_id": qid,
                "question": question,
                "gold_answer": gold_answer,
                "example_dir": str(example_dir),
                "docs_dir": str(docs_dir),
                "num_docs": len(docs),
                "dataset_type": row.get("dataset_type", "support_only"),
                "files": file_manifest,
            }
            idx.write(json.dumps(example_record, ensure_ascii=False) + "\n")
            written += 1
            if args.limit and written >= args.limit:
                break

    manifest = {
        "source_jsonl": str(source),
        "out_dir": str(out_dir),
        "index_jsonl": str(index_path),
        "num_examples": written,
        "layout": {
            "per_question_dir": "examples/question_<qid>/docs/*.txt",
            "tool_visible_files": "docs/*.txt only",
            "gold_answers_stored_only_in_index": True,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
