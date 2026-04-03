"""Helpers for copying sample metadata from input rows into output rows.

The current experiment matrix does not rely heavily on sampling metadata, but the original
BrowseCompV2 runners pass these keys through whenever they are present. Keeping that logic
in a dedicated module makes the copied experiment code easier to read.
"""

from __future__ import annotations

from typing import Any, Dict

# These are the only sample-level fields the original runners intentionally preserve.
PASSTHROUGH_SAMPLE_KEYS = ("sample_seed", "sample_tag", "sample_index")


def sample_metadata_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the metadata keys that should survive into downstream outputs."""
    out: Dict[str, Any] = {}
    for key in PASSTHROUGH_SAMPLE_KEYS:
        if key not in row:
            continue
        value = row.get(key)
        if value is None or value == "":
            continue
        # Integer fields should stay numeric when possible so downstream analysis stays clean.
        if key in {"sample_seed", "sample_index"}:
            try:
                value = int(value)
            except Exception:
                pass
        out[key] = value
    return out


def attach_sample_metadata(payload: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    """Mutate ``payload`` in place with any sample metadata carried by ``row``."""
    payload.update(sample_metadata_from_row(row))
    return payload
