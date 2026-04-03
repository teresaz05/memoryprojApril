"""Configuration for the prose merge2 experiment.

The heavy implementation lives in ``cluster_bank_core.py``. This module exists so the package
has a readable, experiment-specific home for the constants and path conventions that define the
prose merge2 run.
"""

from __future__ import annotations

from pathlib import Path


EXPERIMENT_NAME = 'prose_merge2'
SUMMARY_MODE = 'per_doc_full_cluster_banks_merge2'


def layer1_result_stub(doc_cluster_style: str = 'titled') -> str:
    """Return the filename stem for the layer-1 prose cluster-bank output."""
    return f'oracle_doc_cluster_bank_concat_{doc_cluster_style}_support_only_Nauto_Mfree'


def merged_result_stub(doc_cluster_style: str = 'titled') -> str:
    """Return the filename stem for the merge2 prose cluster-bank output."""
    return f'oracle_doc_cluster_bank_merge2_concat_{doc_cluster_style}_support_only_Nauto_Mfree'


def build_default_run_dir(package_root: Path, model_slug: str, timestamp: str) -> Path:
    """Place runs under a readable experiment/model/timestamp hierarchy."""
    return package_root / 'runs' / 'support_only_q50' / model_slug / EXPERIMENT_NAME / timestamp
