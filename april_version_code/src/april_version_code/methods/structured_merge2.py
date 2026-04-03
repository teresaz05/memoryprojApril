"""Configuration for the structured merge2 experiment."""

from __future__ import annotations

from pathlib import Path


EXPERIMENT_NAME = 'structured_merge2'
SUMMARY_MODE = 'per_doc_structured_cluster_banks_merge2'


def layer1_result_stub(doc_cluster_style: str = 'titled') -> str:
    return f'oracle_doc_structured_cluster_bank_concat_{doc_cluster_style}_support_only_Nauto_Mfree'


def merged_result_stub(doc_cluster_style: str = 'titled') -> str:
    return f'oracle_doc_structured_cluster_bank_merge2_concat_{doc_cluster_style}_support_only_Nauto_Mfree'


def build_default_run_dir(package_root: Path, model_slug: str, timestamp: str) -> Path:
    return package_root / 'runs' / 'support_only_q50' / model_slug / EXPERIMENT_NAME / timestamp
