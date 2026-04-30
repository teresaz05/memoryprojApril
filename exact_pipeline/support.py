#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

from llm_backends import TokenCounter

class QwenEmbedder:
    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to import sentence-transformers for Qwen embedding. "
                "Install with: pip install sentence-transformers"
            ) from exc
        try:
            self.model = SentenceTransformer(
                model_name,
                trust_remote_code=True,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load embedding model '{model_name}'. "
                "Check model availability and network/cache."
            ) from exc

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(vecs, dtype=np.float32)

def format_doc_for_prompt(doc: Dict[str, Any]) -> str:
    parts = [
        f"doc_id: {doc.get('doc_id', '')}",
        "text:",
        str(doc.get("text", "")),
    ]
    return "\n".join(parts)

def format_doc_chunk_for_prompt(docs_chunk: Sequence[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for i, doc in enumerate(docs_chunk, start=1):
        blocks.append(f"[DOC_{i}]\n{format_doc_for_prompt(doc)}")
    return "\n\n".join(blocks)

def normalize_candidate_query(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"^\d+[\)\.\-\:\s]+", "", s)
    s = s.strip("`\"' ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_fallback_queries(warm_docs: Sequence[Dict[str, Any]], needed: int) -> List[str]:
    out: List[str] = []
    for idx, doc in enumerate(warm_docs, start=1):
        did = str(doc.get("doc_id", "")).strip() or f"doc_{idx}"
        out.extend(
            [
                f"What is the main claim in document {did}?",
                f"Which people, places, or organizations are central in document {did}?",
                f"What dated events or numeric facts are stated in document {did}?",
            ]
        )
        if len(out) >= needed:
            break
    while len(out) < needed:
        n = len(out) + 1
        out.append(f"What key factual relationship is stated in the warm-start documents ({n})?")
    return out[:needed]

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # embeddings are normalized in QwenEmbedder, but keep safe fallback
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def truncate_to_budget(
    text: str,
    counter: TokenCounter,
    budget_tokens: int,
    truncate_strategy: str,
) -> Tuple[str, bool]:
    text = (text or "").strip()
    tokens = counter.count(text)
    if tokens <= budget_tokens:
        return text, False
    return counter.truncate(text, budget_tokens, strategy=truncate_strategy), True

def write_jsonl_row(handle: Any, payload: Dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

def flush_jsonl_handle(handle: Any) -> None:
    handle.flush()
    try:
        os.fsync(handle.fileno())
    except (AttributeError, OSError, ValueError):
        pass
