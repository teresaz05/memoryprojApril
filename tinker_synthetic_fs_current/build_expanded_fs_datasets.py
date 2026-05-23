from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build expanded Synthetic-FS training datasets from current BrowseComp+, "
            "DeepDive trajectories, and synthetic BrowseComp+ clusters."
        )
    )
    ap.add_argument(
        "--current-jsonl",
        default="../data/browsecomp_plus_support_only_all_q830.jsonl",
        help="Current BrowseComp+ support-only source JSONL.",
    )
    ap.add_argument("--deepdive-jsonl", required=True, help="DeepDive annotated trajectory JSONL.")
    ap.add_argument("--synthetic-json", required=True, help="Synthetic BrowseComp+ clustered JSON.")
    ap.add_argument("--out-root", default="../tinker_fs_qa", help="Root directory for generated datasets.")
    ap.add_argument("--overwrite", action="store_true", help="Replace generated output directories/files.")
    return ap.parse_args()


def safe_slug(text: str, max_len: int = 120) -> str:
    chars: list[str] = []
    for ch in text:
        chars.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return (slug or "doc")[:max_len]


def url_slug(url: str, fallback: str) -> str:
    parsed = urlparse((url or "").strip())
    parts: list[str] = []
    if parsed.netloc:
        parts.append(parsed.netloc)
    path_parts = [part for part in parsed.path.split("/") if part]
    if path_parts:
        parts.extend(path_parts[-3:])
    if parsed.query:
        parts.append(parsed.query[:40])
    return safe_slug("__".join(parts) if parts else fallback)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def normalize_current(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied["question_id"] = str(row["question_id"])
        copied["source"] = "browsecomp_plus_current"
        normalized.append(copied)
    return normalized


def normalize_deepdive(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        query_id = str(row["query_id"])
        docs: list[dict[str, Any]] = []
        for doc in row.get("evidence_docs", []):
            docs.append(
                {
                    "doc_id": f"deepdive_{query_id}_evidence_{doc.get('docid', len(docs))}",
                    "text": str(doc.get("text", "")).strip(),
                    "url": str(doc.get("url", "")),
                    "title": str(doc.get("title", "")),
                    "is_evidence": True,
                    "is_gold": bool(doc.get("is_gold", False)),
                    "is_negative": False,
                }
            )
        for doc in row.get("negative_docs", []):
            docs.append(
                {
                    "doc_id": f"deepdive_{query_id}_negative_{doc.get('docid', len(docs))}",
                    "text": str(doc.get("text", "")).strip(),
                    "url": str(doc.get("url", "")),
                    "title": str(doc.get("title", "")),
                    "is_evidence": False,
                    "is_gold": False,
                    "is_negative": True,
                }
            )
        normalized.append(
            {
                "question_id": f"deepdive_{query_id}",
                "question": str(row["query"]).strip(),
                "gold_answer": str(row["answer"]).strip(),
                "dataset_type": "deepdive_annotated_web_search",
                "source": "deepdive",
                "source_question_id": query_id,
                "docs": docs,
            }
        )
    return normalized


def normalize_synthetic_questions(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    normalized: list[dict[str, Any]] = []
    for cluster in obj.get("clusters", []):
        cluster_id = int(cluster["cluster_id"])
        for question_idx, question in enumerate(cluster.get("questions", [])):
            docs = [
                {
                    "doc_id": str(doc.get("doc_id", f"cluster_{cluster_id}_doc_{doc_idx}")),
                    "text": str(doc.get("text", "")).strip(),
                    "url": str(doc.get("url", "")),
                    "title": str(doc.get("title", "")),
                    "is_evidence": bool(doc.get("is_evidence", False)),
                    "is_gold": bool(doc.get("is_gold", False)),
                    "is_negative": bool(doc.get("is_negative", False)),
                }
                for doc_idx, doc in enumerate(question.get("docs", []))
            ]
            normalized.append(
                {
                    "question_id": f"synthetic_c{cluster_id:03d}_q{question_idx:03d}",
                    "question": str(question["question"]).strip(),
                    "gold_answer": str(question["gold_answer"]).strip(),
                    "dataset_type": "synthetic_browsecomp_plus_question",
                    "source": "synthetic_browsecomp_plus",
                    "source_cluster_id": cluster_id,
                    "source_question_index": question_idx,
                    "docs": docs,
                }
            )
    return normalized


def normalize_synthetic_cluster_prompts(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    normalized: list[dict[str, Any]] = []
    for cluster in obj.get("clusters", []):
        cluster_id = int(cluster["cluster_id"])
        questions = list(cluster.get("questions", []))
        if not questions:
            continue

        # Docs are duplicated across questions in the source JSON; use the first question's docs
        # as the shared cluster document set.
        first_docs = questions[0].get("docs", [])
        docs = [
            {
                "doc_id": str(doc.get("doc_id", f"cluster_{cluster_id}_doc_{doc_idx}")),
                "text": str(doc.get("text", "")).strip(),
                "url": str(doc.get("url", "")),
                "title": str(doc.get("title", "")),
                "is_evidence": bool(doc.get("is_evidence", False)),
                "is_gold": bool(doc.get("is_gold", False)),
                "is_negative": bool(doc.get("is_negative", False)),
            }
            for doc_idx, doc in enumerate(first_docs)
        ]
        question_lines = [
            f"{idx}. {str(question['question']).strip()}" for idx, question in enumerate(questions, start=1)
        ]
        answer_lines = [
            f"{idx}. {str(question['gold_answer']).strip()}" for idx, question in enumerate(questions, start=1)
        ]
        normalized.append(
            {
                "question_id": f"synthetic_cluster_{cluster_id:03d}_multiq",
                "question": (
                    "Answer each of the following questions using the shared document set. "
                    "Return one numbered answer per question.\n\n" + "\n".join(question_lines)
                ),
                "gold_answer": "\n".join(answer_lines),
                "dataset_type": "synthetic_browsecomp_plus_multi_question_prompt",
                "source": "synthetic_browsecomp_plus_multiq",
                "source_cluster_id": cluster_id,
                "num_questions_in_prompt": len(questions),
                "docs": docs,
            }
        )
    return normalized


def render_doc_file(doc: dict[str, Any]) -> str:
    header = [
        f"DOC_ID: {doc.get('doc_id', '')}",
        f"URL: {doc.get('url', '')}",
    ]
    title = str(doc.get("title", "")).strip()
    if title:
        header.append(f"TITLE: {title}")
    header.append("")
    return "\n".join(header) + str(doc.get("text", "")).strip() + "\n"


def materialize_fs_dataset(rows: list[dict[str, Any]], out_dir: Path, overwrite: bool) -> dict[str, Any]:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {out_dir}. Use --overwrite.")
        shutil.rmtree(out_dir)

    agent_dir = out_dir / "agent_data"
    privileged_dir = out_dir / "privileged_data"
    agent_dir.mkdir(parents=True, exist_ok=True)
    privileged_dir.mkdir(parents=True, exist_ok=True)

    index_path = out_dir / "index.jsonl"
    source_counts: Counter[str] = Counter()
    doc_counts: Counter[str] = Counter()
    qid_seen: set[str] = set()

    with index_path.open("w", encoding="utf-8") as idx:
        for row in rows:
            qid = str(row["question_id"])
            if qid in qid_seen:
                raise ValueError(f"Duplicate question_id in materialized dataset: {qid}")
            qid_seen.add(qid)

            source = str(row.get("source", "unknown"))
            source_counts[source] += 1
            question = str(row["question"]).strip()
            answer = str(row["gold_answer"]).strip()
            docs = list(row.get("docs", []))

            agent_query_dir = agent_dir / safe_slug(qid, max_len=180)
            privileged_query_dir = privileged_dir / safe_slug(qid, max_len=180)
            agent_query_dir.mkdir(parents=True, exist_ok=True)
            privileged_query_dir.mkdir(parents=True, exist_ok=True)

            file_manifest: list[dict[str, Any]] = []
            used_names: set[str] = set()
            for doc_idx, doc in enumerate(docs, start=1):
                fallback = f"{doc.get('doc_id', f'doc_{doc_idx:03d}')}_{doc.get('title', '')}"
                file_stem = url_slug(str(doc.get("url", "")), fallback=str(fallback))
                file_name = f"{file_stem}.txt"
                if file_name in used_names:
                    file_name = f"{file_stem}_{doc_idx:03d}.txt"
                used_names.add(file_name)

                (agent_query_dir / file_name).write_text(render_doc_file(doc), encoding="utf-8")
                is_evidence = bool(doc.get("is_evidence", False))
                is_gold = bool(doc.get("is_gold", False))
                is_negative = bool(doc.get("is_negative", False))
                doc_counts["evidence" if is_evidence else "non_evidence"] += 1
                doc_counts["gold" if is_gold else "non_gold"] += 1
                doc_counts["negative" if is_negative else "non_negative"] += 1
                file_manifest.append(
                    {
                        "relative_path": file_name,
                        "doc_id": doc.get("doc_id"),
                        "url": doc.get("url", ""),
                        "is_evidence": is_evidence,
                        "is_gold": is_gold,
                        "is_negative": is_negative,
                    }
                )

            (privileged_query_dir / "query.txt").write_text(question + "\n", encoding="utf-8")
            (privileged_query_dir / "answer.txt").write_text(answer + "\n", encoding="utf-8")

            per_query_manifest = {
                "question_id": qid,
                "source": source,
                "query_file": "query.txt",
                "answer_file": "answer.txt",
                "agent_dir": str(agent_query_dir),
                "documents": file_manifest,
            }
            for key in ("source_question_id", "source_cluster_id", "source_question_index", "num_questions_in_prompt"):
                if key in row:
                    per_query_manifest[key] = row[key]
            (privileged_query_dir / "manifest.json").write_text(
                json.dumps(per_query_manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            index_record = {
                "question_id": qid,
                "agent_query_dir": str(agent_query_dir),
                "privileged_query_dir": str(privileged_query_dir),
                "num_docs": len(docs),
                "dataset_type": row.get("dataset_type", "support_only"),
                "source": source,
                "files": file_manifest,
            }
            for key in ("source_question_id", "source_cluster_id", "source_question_index", "num_questions_in_prompt"):
                if key in row:
                    index_record[key] = row[key]
            idx.write(json.dumps(index_record, ensure_ascii=False) + "\n")

    manifest = {
        "out_dir": str(out_dir),
        "index_jsonl": str(index_path),
        "num_examples": len(rows),
        "source_counts": dict(source_counts),
        "doc_counts": dict(doc_counts),
        "layout": {
            "agent_data": "agent_data/<qid>/*.txt",
            "privileged_data": "privileged_data/<qid>/{query.txt,answer.txt,manifest.json}",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def write_source(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> dict[str, Any]:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output source already exists: {path}. Use --overwrite.")
    count = write_jsonl(path, rows)
    return {
        "path": str(path),
        "num_examples": count,
        "source_counts": dict(Counter(str(row.get("source", "unknown")) for row in rows)),
    }


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    sources_dir = out_root / "expanded_sources"

    current_rows = normalize_current(Path(args.current_jsonl))
    deepdive_rows = normalize_deepdive(Path(args.deepdive_jsonl))
    synthetic_question_rows = normalize_synthetic_questions(Path(args.synthetic_json))
    synthetic_multiq_rows = normalize_synthetic_cluster_prompts(Path(args.synthetic_json))

    source_outputs = {
        "current": current_rows,
        "deepdive": deepdive_rows,
        "synthetic_questions": synthetic_question_rows,
        "synthetic_multiq": synthetic_multiq_rows,
        "current_plus_deepdive": current_rows + deepdive_rows,
        "current_plus_synthetic_multiq": current_rows + synthetic_multiq_rows,
        "current_plus_deepdive_plus_synthetic_questions": current_rows + deepdive_rows + synthetic_question_rows,
        "current_plus_deepdive_plus_synthetic_multiq": current_rows + deepdive_rows + synthetic_multiq_rows,
    }

    summary: dict[str, Any] = {"sources": {}, "fs_datasets": {}}
    for name, rows in source_outputs.items():
        summary["sources"][name] = write_source(sources_dir / f"{name}.jsonl", rows, args.overwrite)

    fs_specs = {
        "train_q830_plus_deepdive_fs": source_outputs["current_plus_deepdive"],
        "train_q830_plus_synthetic_multiq_fs": source_outputs["current_plus_synthetic_multiq"],
        "train_q830_plus_deepdive_plus_synthetic_questions_fs": source_outputs[
            "current_plus_deepdive_plus_synthetic_questions"
        ],
        "train_q830_plus_deepdive_plus_synthetic_multiq_fs": source_outputs[
            "current_plus_deepdive_plus_synthetic_multiq"
        ],
    }
    for dirname, rows in fs_specs.items():
        summary["fs_datasets"][dirname] = materialize_fs_dataset(rows, out_root / dirname, args.overwrite)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
