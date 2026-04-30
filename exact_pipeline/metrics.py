from __future__ import annotations


def exact_match(pred: str, gold: str) -> bool:
    pred = (pred or '').strip().lower()
    gold = (gold or '').strip().lower()
    return bool(pred) and pred == gold
