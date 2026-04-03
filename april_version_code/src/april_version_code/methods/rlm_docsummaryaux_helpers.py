"""Small helper functions shared by docsummaryaux-based RLM variants.

These helpers were extracted from the older filesystem-docpair RLM runner so the new
prompt-doc variant can stay focused on the parts that are actually specific to prompt-doc
context construction.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from april_version_code.methods import rlm_official_core as rbase


def build_truncated_doc_block(
    doc: Dict[str, Any],
    counter: rbase.TokenCounter,
    max_doc_tokens: int,
    doc_truncate_strategy: str,
    max_doc_chars: int,
) -> Tuple[str, bool, bool]:
    """Return a raw-document block after applying the same truncation policy as the old runner."""
    doc_id = str(doc.get("doc_id", "")).strip()
    text = str(doc.get("text", "") or "")
    char_truncated = False
    token_truncated = False
    if max_doc_chars > 0 and len(text) > max_doc_chars:
        text = text[:max_doc_chars]
        char_truncated = True
    if max_doc_tokens > 0 and counter.count(text) > max_doc_tokens:
        text = counter.truncate(text, max_doc_tokens, strategy=doc_truncate_strategy)
        token_truncated = True
    block = "\n".join([f"doc_id: {doc_id}", "text:", text])
    return block, char_truncated, token_truncated


def build_companion_block(doc_id: str, source_row: Optional[Dict[str, Any]]) -> str:
    """Return the synthetic prompt document that holds summary + cluster-bank text for one doc."""
    lines = [
        f"doc_id: {doc_id}__clusters_summary",
        f"source_doc_id: {doc_id}",
        "",
    ]
    if source_row is None:
        lines.append("No derived summary or cluster-bank text was available for this source document.")
        return "\n".join(lines).strip()

    summary_text = str(source_row.get("source_doc_summary", "") or "").strip()
    cluster_bank_text = str(source_row.get("cluster_bank_text", "") or "").strip()
    if summary_text:
        lines.extend(["DOCUMENT_SUMMARY:", summary_text, ""])
    if cluster_bank_text:
        lines.extend(["DOCUMENT_CLUSTER_BANKS:", cluster_bank_text, ""])
    if not summary_text and not cluster_bank_text:
        lines.append("No derived summary or cluster-bank text was available for this source document.")
    return "\n".join(lines).strip()
