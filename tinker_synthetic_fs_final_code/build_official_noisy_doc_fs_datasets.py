from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parents[2]
DEFAULT_OFFICIAL = WORKSPACE_ROOT / "BrowseComp-Plus" / "data" / "browsecomp_plus_decrypted.jsonl"
DEFAULT_TINKER_FS = ROOT.parent / "tinker_fs_qa"
DEFAULT_TRAIN_INDEX = DEFAULT_TINKER_FS / "train_q830_fs" / "index.jsonl"
DEFAULT_EVAL_INDEX = DEFAULT_TINKER_FS / "train_q50_nonexcluded_fs" / "index.jsonl"
DEFAULT_EXCLUDED = DEFAULT_TINKER_FS / "excluded100.jsonl"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Materialize official BrowseComp+ noisy-document filesystem datasets "
            "for the current train/eval qid split."
        )
    )
    ap.add_argument("--official-jsonl", default=str(DEFAULT_OFFICIAL))
    ap.add_argument("--train-index-jsonl", default=str(DEFAULT_TRAIN_INDEX))
    ap.add_argument("--eval-index-jsonl", default=str(DEFAULT_EVAL_INDEX))
    ap.add_argument("--excluded-qids-jsonl", default=str(DEFAULT_EXCLUDED))
    ap.add_argument(
        "--train-qids-file",
        default="",
        help=(
            "Optional explicit train qid file. Supports JSON, JSONL, or one qid per line. "
            "If omitted, train qids are train-index minus eval qids minus excluded qids."
        ),
    )
    ap.add_argument(
        "--eval-qids-file",
        default="",
        help="Optional explicit eval qid file. Supports JSON, JSONL, or one qid per line.",
    )
    ap.add_argument("--out-root", default=str(DEFAULT_TINKER_FS))
    ap.add_argument("--max-docs", type=int, default=50)
    ap.add_argument(
        "--noise-seed",
        type=int,
        default=20260523,
        help="Seed for deterministic per-question noisy-document sampling/order.",
    )
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def qid_from(row: dict[str, Any]) -> str:
    return str(row.get("question_id") or row.get("query_id") or row.get("qid") or "").strip()


def load_index_qids(path: Path) -> list[str]:
    return [qid for qid in (qid_from(row) for row in iter_jsonl(path)) if qid]


def qids_from_json_obj(obj: Any) -> list[str]:
    qids: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            qid = qid_from(value)
            if qid:
                qids.append(qid)
                return
            for key in ("qids", "query_ids", "question_ids", "selected_qids", "questions", "items", "rows"):
                if key in value:
                    visit(value[key])
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, (str, int)):
            text = str(value).strip()
            if text:
                qids.append(text)

    visit(obj)
    return qids


def load_qids_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        return [qid for qid in (qid_from(row) for row in iter_jsonl(path)) if qid]
    if path.suffix == ".json":
        return qids_from_json_obj(json.loads(text))
    return [line.strip().split()[0] for line in text.splitlines() if line.strip()]


def unique_preserve_order(qids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for qid in qids:
        qid = str(qid).strip()
        if qid and qid not in seen:
            seen.add(qid)
            out.append(qid)
    return out


def safe_slug(text: str, max_len: int = 120) -> str:
    chars = [ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text]
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


def normalize_official_docs(row: dict[str, Any]) -> list[dict[str, Any]]:
    docs_by_id: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []

    def upsert(raw_doc: dict[str, Any], role: str) -> None:
        doc_id = str(raw_doc.get("docid") or raw_doc.get("doc_id") or "").strip()
        text = str(raw_doc.get("text") or "").strip()
        if not doc_id or not text:
            return
        if doc_id not in docs_by_id:
            docs_by_id[doc_id] = {
                "doc_id": doc_id,
                "text": text,
                "url": str(raw_doc.get("url") or ""),
                "is_gold": False,
                "is_evidence": False,
                "is_negative": False,
                "source_roles": [],
            }
            ordered_ids.append(doc_id)
        doc = docs_by_id[doc_id]
        if role == "gold":
            doc["is_gold"] = True
        elif role == "evidence":
            doc["is_evidence"] = True
        elif role == "negative":
            doc["is_negative"] = True
        if role not in doc["source_roles"]:
            doc["source_roles"].append(role)

    for field_name, role_name in (("gold_docs", "gold"), ("evidence_docs", "evidence"), ("negative_docs", "negative")):
        for raw_doc in row.get(field_name) or []:
            if isinstance(raw_doc, dict):
                upsert(raw_doc, role_name)

    return [docs_by_id[doc_id] for doc_id in ordered_ids]


def split_docs(docs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    support = [doc for doc in docs if bool(doc.get("is_gold")) or bool(doc.get("is_evidence"))]
    noise = [
        doc
        for doc in docs
        if bool(doc.get("is_negative")) and not (bool(doc.get("is_gold")) or bool(doc.get("is_evidence")))
    ]
    return support, noise


def rng_for_qid(seed: int, qid: str, purpose: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{qid}:{purpose}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], byteorder="big"))


def docs_support_first_random_noise(
    docs: list[dict[str, Any]],
    *,
    qid: str,
    max_docs: int | None,
    noise_seed: int,
) -> list[dict[str, Any]]:
    support, noise = split_docs(docs)
    if max_docs is not None and len(support) > max_docs:
        raise ValueError(f"QID {qid} has {len(support)} support docs, above max_docs={max_docs}.")

    shuffled_noise = list(noise)
    rng_for_qid(noise_seed, qid, "noise_order").shuffle(shuffled_noise)
    if max_docs is None:
        return support + shuffled_noise

    remaining = max(0, max_docs - len(support))
    return support + shuffled_noise[:remaining]


def render_doc_file(doc: dict[str, Any]) -> str:
    header = [
        f"DOC_ID: {doc.get('doc_id', '')}",
        f"URL: {doc.get('url', '')}",
        "",
    ]
    return "\n".join(header) + str(doc.get("text", "")).strip() + "\n"


def stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "median": 0, "mean": 0, "max": 0}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": round(statistics.mean(values), 4),
        "max": max(values),
    }


@dataclass
class DatasetWriter:
    out_dir: Path
    dataset_type: str
    split_name: str
    source_jsonl: Path
    max_docs: int | None
    noise_seed: int
    qids_expected: set[str]
    agent_dir: Path = field(init=False)
    privileged_dir: Path = field(init=False)
    index_path: Path = field(init=False)
    index_handle: Any = field(init=False, default=None)
    qids_written: list[str] = field(default_factory=list)
    doc_counts: list[int] = field(default_factory=list)
    support_counts: list[int] = field(default_factory=list)
    noise_counts: list[int] = field(default_factory=list)

    def open(self, overwrite: bool) -> None:
        if self.out_dir.exists():
            if not overwrite:
                raise FileExistsError(f"Output directory already exists: {self.out_dir}. Use --overwrite.")
            shutil.rmtree(self.out_dir)
        self.agent_dir = self.out_dir / "agent_data"
        self.privileged_dir = self.out_dir / "privileged_data"
        self.index_path = self.out_dir / "index.jsonl"
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        self.privileged_dir.mkdir(parents=True, exist_ok=True)
        self.index_handle = self.index_path.open("w", encoding="utf-8")

    def close(self) -> None:
        if self.index_handle:
            self.index_handle.close()
            self.index_handle = None
        self.write_manifest()

    def write_row(self, row: dict[str, Any], docs: list[dict[str, Any]]) -> None:
        qid = str(row["query_id"]).strip()
        question = str(row.get("query") or "").strip()
        answer = str(row.get("answer") or "").strip()
        support_docs, noise_docs = split_docs(docs)

        agent_query_dir = self.agent_dir / safe_slug(qid, max_len=180)
        privileged_query_dir = self.privileged_dir / safe_slug(qid, max_len=180)
        agent_query_dir.mkdir(parents=True, exist_ok=True)
        privileged_query_dir.mkdir(parents=True, exist_ok=True)

        file_manifest: list[dict[str, Any]] = []
        used_names: set[str] = set()
        for doc_idx, doc in enumerate(docs, start=1):
            fallback = str(doc.get("doc_id") or f"doc_{doc_idx:03d}")
            file_stem = f"doc_{doc_idx:03d}__{url_slug(str(doc.get('url', '')), fallback=fallback)}"
            file_name = f"{file_stem}.txt"
            if file_name in used_names:
                file_name = f"{file_stem}__{doc_idx}.txt"
            used_names.add(file_name)
            (agent_query_dir / file_name).write_text(render_doc_file(doc), encoding="utf-8")
            file_manifest.append(
                {
                    "relative_path": file_name,
                    "doc_id": doc.get("doc_id"),
                    "url": doc.get("url"),
                    "is_gold": bool(doc.get("is_gold", False)),
                    "is_evidence": bool(doc.get("is_evidence", False)),
                    "is_negative": bool(doc.get("is_negative", False)),
                    "source_roles": list(doc.get("source_roles", [])),
                }
            )

        (privileged_query_dir / "query.txt").write_text(question + "\n", encoding="utf-8")
        (privileged_query_dir / "answer.txt").write_text(answer + "\n", encoding="utf-8")
        (privileged_query_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "question_id": qid,
                    "query_file": "query.txt",
                    "answer_file": "answer.txt",
                    "agent_dir": str(agent_query_dir),
                    "documents": file_manifest,
                    "num_support_docs": len(support_docs),
                    "num_noise_docs": len(noise_docs),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        self.index_handle.write(
            json.dumps(
                {
                    "question_id": qid,
                    "agent_query_dir": str(agent_query_dir),
                    "privileged_query_dir": str(privileged_query_dir),
                    "num_docs": len(docs),
                    "num_support_docs": len(support_docs),
                    "num_noise_docs": len(noise_docs),
                    "dataset_type": self.dataset_type,
                    "files": file_manifest,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        self.qids_written.append(qid)
        self.doc_counts.append(len(docs))
        self.support_counts.append(len(support_docs))
        self.noise_counts.append(len(noise_docs))

    def write_manifest(self) -> None:
        written = set(self.qids_written)
        missing = sorted(self.qids_expected - written, key=lambda x: (len(x), x))
        manifest = {
            "dataset_type": self.dataset_type,
            "split_name": self.split_name,
            "source_jsonl": str(self.source_jsonl),
            "out_dir": str(self.out_dir),
            "index_jsonl": str(self.index_path),
            "max_docs": self.max_docs,
            "noise_seed": self.noise_seed,
            "num_examples": len(self.qids_written),
            "expected_qids": len(self.qids_expected),
            "missing_qids": missing,
            "doc_count_stats": stats(self.doc_counts),
            "support_doc_count_stats": stats(self.support_counts),
            "noise_doc_count_stats": stats(self.noise_counts),
            "support_docs_always_first": True,
            "noisy_docs_randomized_per_qid": True,
            "agent_visible_labels": False,
        }
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def make_writers(
    out_root: Path,
    official_jsonl: Path,
    max_docs: int,
    noise_seed: int,
    train_qids: set[str],
    eval_qids: set[str],
) -> list[DatasetWriter]:
    return [
        DatasetWriter(
            out_dir=out_root / "train_official_noisy_docs_max50_fs",
            dataset_type="official_browsecomp_plus_noisy_docs_max50",
            split_name="train",
            source_jsonl=official_jsonl,
            max_docs=max_docs,
            noise_seed=noise_seed,
            qids_expected=train_qids,
        ),
        DatasetWriter(
            out_dir=out_root / "heldout50_official_noisy_docs_max50_fs",
            dataset_type="official_browsecomp_plus_noisy_docs_max50",
            split_name="heldout50",
            source_jsonl=official_jsonl,
            max_docs=max_docs,
            noise_seed=noise_seed,
            qids_expected=eval_qids,
        ),
        DatasetWriter(
            out_dir=out_root / "train_official_noisy_docs_all_fs",
            dataset_type="official_browsecomp_plus_noisy_docs_all",
            split_name="train",
            source_jsonl=official_jsonl,
            max_docs=None,
            noise_seed=noise_seed,
            qids_expected=train_qids,
        ),
        DatasetWriter(
            out_dir=out_root / "heldout50_official_noisy_docs_all_fs",
            dataset_type="official_browsecomp_plus_noisy_docs_all",
            split_name="heldout50",
            source_jsonl=official_jsonl,
            max_docs=None,
            noise_seed=noise_seed,
            qids_expected=eval_qids,
        ),
    ]


def main() -> None:
    args = parse_args()
    official_jsonl = Path(args.official_jsonl).expanduser().resolve()
    train_index = Path(args.train_index_jsonl).expanduser().resolve()
    eval_index = Path(args.eval_index_jsonl).expanduser().resolve()
    excluded_path = Path(args.excluded_qids_jsonl).expanduser().resolve() if args.excluded_qids_jsonl else None
    out_root = Path(args.out_root).expanduser().resolve()

    if not official_jsonl.exists() or official_jsonl.stat().st_size == 0:
        raise FileNotFoundError(f"Official BrowseComp+ JSONL is missing or empty: {official_jsonl}")
    if args.max_docs <= 0:
        raise ValueError("--max-docs must be positive for the capped dataset.")

    eval_qids_ordered = (
        unique_preserve_order(load_qids_file(Path(args.eval_qids_file).expanduser()))
        if args.eval_qids_file
        else unique_preserve_order(load_index_qids(eval_index))
    )
    excluded_qids = set(load_index_qids(excluded_path)) if excluded_path and excluded_path.exists() else set()
    if args.train_qids_file:
        train_qids_ordered = unique_preserve_order(load_qids_file(Path(args.train_qids_file).expanduser()))
    else:
        eval_set = set(eval_qids_ordered)
        train_qids_ordered = [
            qid
            for qid in unique_preserve_order(load_index_qids(train_index))
            if qid not in eval_set and qid not in excluded_qids
        ]

    train_qids = set(train_qids_ordered)
    eval_qids = set(eval_qids_ordered)
    overlap = train_qids & eval_qids
    if overlap:
        raise ValueError(f"Train/eval qids overlap: {sorted(overlap)[:20]}")
    excluded_overlap = train_qids & excluded_qids
    if excluded_overlap:
        raise ValueError(f"Train qids overlap excluded qids: {sorted(excluded_overlap)[:20]}")

    writers = make_writers(out_root, official_jsonl, args.max_docs, args.noise_seed, train_qids, eval_qids)
    writer_by_key = {
        ("train", "max50"): writers[0],
        ("eval", "max50"): writers[1],
        ("train", "all"): writers[2],
        ("eval", "all"): writers[3],
    }

    for writer in writers:
        writer.open(overwrite=args.overwrite)

    selected = train_qids | eval_qids
    source_seen: set[str] = set()
    try:
        for row in iter_jsonl(official_jsonl):
            qid = str(row.get("query_id") or "").strip()
            if qid not in selected:
                continue
            source_seen.add(qid)
            all_docs = normalize_official_docs(row)
            randomized_all_docs = docs_support_first_random_noise(
                all_docs,
                qid=qid,
                max_docs=None,
                noise_seed=args.noise_seed,
            )
            capped_docs = docs_support_first_random_noise(
                all_docs,
                qid=qid,
                max_docs=args.max_docs,
                noise_seed=args.noise_seed,
            )
            if qid in train_qids:
                writer_by_key[("train", "max50")].write_row(row, capped_docs)
                writer_by_key[("train", "all")].write_row(row, randomized_all_docs)
            elif qid in eval_qids:
                writer_by_key[("eval", "max50")].write_row(row, capped_docs)
                writer_by_key[("eval", "all")].write_row(row, randomized_all_docs)
    finally:
        for writer in writers:
            writer.close()

    missing = sorted(selected - source_seen, key=lambda x: (len(x), x))
    if missing:
        raise RuntimeError(f"Official BrowseComp+ source missing selected qids: {missing[:50]}")

    summary = {
        "official_jsonl": str(official_jsonl),
        "train_qids": len(train_qids),
        "eval_qids": len(eval_qids),
        "excluded_qids": len(excluded_qids),
        "max_docs": args.max_docs,
        "noise_seed": args.noise_seed,
        "outputs": [str(writer.out_dir / "index.jsonl") for writer in writers],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
