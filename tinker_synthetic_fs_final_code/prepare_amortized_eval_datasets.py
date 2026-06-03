from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Any


DEFAULT_BRIGHT_DOMAINS = [
    "biology",
    "economics",
    "robotics",
    "stackoverflow",
    "leetcode",
]


def _slug(text: str, fallback: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned[:90] or fallback


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _doc_filename(idx: int, doc: dict[str, Any]) -> str:
    doc_id = str(doc.get("doc_id", f"doc_{idx:03d}"))
    return f"doc_{idx:03d}__{_slug(doc_id, f'doc_{idx:03d}')}.txt"


def _build_cluster_row(
    *,
    out_dir: Path,
    dataset_type: str,
    domain: str,
    cluster_id: str,
    cluster_index: int,
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    if not questions:
        raise ValueError(f"{domain} {cluster_id}: empty questions")

    docs = list(questions[0].get("docs") or [])
    if not docs:
        raise ValueError(f"{domain} {cluster_id}: first question has no docs")

    qid = f"{domain}_cluster_{cluster_index:04d}"
    agent_dir = (out_dir / "agent_data" / qid).resolve()
    privileged_dir = (out_dir / "privileged_data" / qid).resolve()
    agent_dir.mkdir(parents=True, exist_ok=True)
    privileged_dir.mkdir(parents=True, exist_ok=True)

    first_q = questions[0]
    _write_text(privileged_dir / "query.txt", str(first_q.get("question", "")))
    _write_text(privileged_dir / "answer.txt", str(first_q.get("gold_answer", "")))

    file_rows: list[dict[str, Any]] = []
    for idx, doc in enumerate(docs, start=1):
        rel = _doc_filename(idx, doc)
        text = str(doc.get("text", ""))
        doc_id = str(doc.get("doc_id", f"doc_{idx:03d}"))
        _write_text(agent_dir / rel, f"doc_id: {doc_id}\n\n{text}")
        file_rows.append(
            {
                "relative_path": rel,
                "doc_id": doc_id,
                "is_gold": bool(doc.get("is_gold", False)),
                "is_evidence": bool(doc.get("is_evidence", False)),
                "is_negative": bool(doc.get("is_negative", False)),
            }
        )

    amortization_questions = [
        {
            "question_id": f"{qid}_q{qidx:02d}",
            "question": str(q.get("question", "")),
            "gold_answer": str(q.get("gold_answer", "")),
            "question_type": str(q.get("question_type", "")),
            "estimated_difficulty": str(q.get("estimated_difficulty", "")),
        }
        for qidx, q in enumerate(questions, start=1)
    ]

    return {
        "question_id": qid,
        "agent_query_dir": str(agent_dir),
        "privileged_query_dir": str(privileged_dir),
        "num_docs": len(file_rows),
        "num_questions": len(amortization_questions),
        "dataset_type": dataset_type,
        "domain": domain,
        "cluster_id": cluster_id,
        "files": file_rows,
        "amortization_questions": amortization_questions,
    }


def prepare_bright(zip_path: Path, out_root: Path, domains: list[str]) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        available = {
            name.split("/")[-2]: name
            for name in zf.namelist()
            if name.endswith("_synthetic-data.json") and "/BRIGHT-" in name
        }
        missing = [domain for domain in domains if domain not in available]
        if missing:
            raise SystemExit(f"Missing BRIGHT domains in zip: {', '.join(missing)}")

        for domain in domains:
            data = json.loads(zf.read(available[domain]).decode("utf-8"))
            out_dir = out_root / domain
            out_dir.mkdir(parents=True, exist_ok=True)
            index_path = out_dir / "index.jsonl"
            rows = []
            for idx, cluster in enumerate(data.get("clusters", [])):
                rows.append(
                    _build_cluster_row(
                        out_dir=out_dir,
                        dataset_type=f"bright_amortized_{domain}",
                        domain=domain,
                        cluster_id=str(cluster.get("cluster_id", f"cluster_{idx}")),
                        cluster_index=idx,
                        questions=list(cluster.get("questions") or []),
                    )
                )
            with index_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            print(f"wrote {index_path} rows={len(rows)}")


def prepare_browsecomp(json_path: Path, out_dir: Path) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"
    rows = []
    for idx, cluster in enumerate(data.get("clusters", [])):
        rows.append(
            _build_cluster_row(
                out_dir=out_dir,
                dataset_type="browsecomp_plus_amortized",
                domain="browsecomp_plus",
                cluster_id=str(cluster.get("cluster_id", f"cluster_{idx}")),
                cluster_index=idx,
                questions=list(cluster.get("questions") or []),
            )
        )
    with index_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {index_path} rows={len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bright-zip", type=Path)
    parser.add_argument("--bright-out", type=Path)
    parser.add_argument("--domains", nargs="+", default=DEFAULT_BRIGHT_DOMAINS)
    parser.add_argument("--browsecomp-json", type=Path)
    parser.add_argument("--browsecomp-out", type=Path)
    args = parser.parse_args()

    if args.bright_zip:
        if not args.bright_out:
            raise SystemExit("--bright-out is required with --bright-zip")
        prepare_bright(args.bright_zip, args.bright_out, args.domains)
    if args.browsecomp_json:
        if not args.browsecomp_out:
            raise SystemExit("--browsecomp-out is required with --browsecomp-json")
        prepare_browsecomp(args.browsecomp_json, args.browsecomp_out)
    if not args.bright_zip and not args.browsecomp_json:
        raise SystemExit("Provide --bright-zip and/or --browsecomp-json")


if __name__ == "__main__":
    main()
