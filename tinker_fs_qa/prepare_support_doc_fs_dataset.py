from __future__ import annotations
"""Build the on-disk dataset layout (so plain python)

Its job is:
1. Read a support-only BrowseComp+ JSONL file.
2. Create an agent-visible directory of document files per question.
3. Create a separate privileged directory per question that stores the question text, gold answer, and a manifest (metadata)
4. Write one index.jsonl file that the RL training code can use later.

So the split is:
- agent_data/: files the model is allowed to read through tools
- privileged_data/: supervision/evaluation files the model should NOT read
"""

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
    # better naming
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
    # each agent-visible file contains:
    # - lightweight metadata at the top
    # - then the raw document text
    header = [
        f"DOC_ID: {doc.get('doc_id', '')}",
        f"URL: {doc.get('url', '')}",
        "",
    ]
    return "\n".join(header) + str(doc.get("text", "")).strip() + "\n"


def main() -> None:
    # creates the per-question directories
    args = parse_args()
    source = Path(args.source_jsonl)
    out_dir = Path(args.out_dir)
    # agent_data is what the model can read through tools.
    agent_dir = out_dir / "agent_data"
    # privileged_data is only for the trainer/evaluator.
    privileged_dir = out_dir / "privileged_data"
    index_path = out_dir / "index.jsonl"
    manifest_path = out_dir / "manifest.json"

    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output directory already exists: {out_dir}. Use --overwrite to replace it.")
        shutil.rmtree(out_dir)

    agent_dir.mkdir(parents=True, exist_ok=True)
    privileged_dir.mkdir(parents=True, exist_ok=True)

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

            # one per-question directory visible to the agent,
            # and one separate per-question privileged directory.
            agent_query_dir = agent_dir / qid
            privileged_query_dir = privileged_dir / qid
            agent_query_dir.mkdir(parents=True, exist_ok=True)
            privileged_query_dir.mkdir(parents=True, exist_ok=True)

            file_manifest: list[dict[str, Any]] = []
            used_names: set[str] = set()
            for idx_doc, doc in enumerate(docs, start=1):
                suffix = safe_slug(str(doc.get("doc_id", "")))
                file_name = f"doc_{idx_doc:03d}__{suffix}.txt"
                if file_name in used_names:
                    file_name = f"doc_{idx_doc:03d}__{suffix}_{idx_doc}.txt"
                used_names.add(file_name)
                rel_path = file_name
                # Only the document text file goes into the agent-visible area.
                (agent_query_dir / file_name).write_text(render_doc_file(doc), encoding="utf-8")
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

            # the query and answer live only in privileged_data.
            (privileged_query_dir / "query.txt").write_text(question + "\n", encoding="utf-8")
            (privileged_query_dir / "answer.txt").write_text(gold_answer + "\n", encoding="utf-8")
            per_query_manifest = {
                "question_id": qid,
                "query_file": "query.txt",
                "answer_file": "answer.txt",
                "agent_dir": str(agent_query_dir),
                "documents": file_manifest,
            }
            (privileged_query_dir / "manifest.json").write_text(
                json.dumps(per_query_manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            # index.jsonl is the bridge between data prep and RL training
            example_record = {
                "question_id": qid,
                "agent_query_dir": str(agent_query_dir),
                "privileged_query_dir": str(privileged_query_dir),
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
            "agent_data": "agent_data/<qid>/*.txt",
            "privileged_data": "privileged_data/<qid>/{query.txt,answer.txt,manifest.json}",
            "tool_visible_files": "agent_data/<qid>/*.txt only",
            "gold_answers_stored_outside_agent_dir": True,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
