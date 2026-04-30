#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict


PASSTHROUGH_SAMPLE_KEYS = ("sample_seed", "sample_tag", "sample_index")


def sample_metadata_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in PASSTHROUGH_SAMPLE_KEYS:
        if key not in row:
            continue
        value = row.get(key)
        if value is None or value == "":
            continue
        if key in {"sample_seed", "sample_index"}:
            try:
                value = int(value)
            except Exception:
                pass
        out[key] = value
    return out


def attach_sample_metadata(payload: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    payload.update(sample_metadata_from_row(row))
    return payload
