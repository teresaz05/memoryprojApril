"""Small token-counting helpers shared by human-facing wrapper code.

The large copied experiment cores still carry their original token utilities because that is
the safest way to preserve behavior. This helper module exists for the lightweight wrapper
scripts and for any future cleanup work.
"""

from __future__ import annotations

import re

import tiktoken


class TokenCounter:
    """Thin convenience wrapper around a tiktoken encoder."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoder = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        """Count tokens without treating any content as a special token."""
        return len(self._encoder.encode(text or "", disallowed_special=()))

    def truncate(self, text: str, max_tokens: int, strategy: str = "head") -> str:
        """Truncate text using the same head/middle/tail strategies used in the runners."""
        if max_tokens <= 0:
            return ""
        tokens = self._encoder.encode(text or "", disallowed_special=())
        if len(tokens) <= max_tokens:
            return text
        if strategy == "tail":
            keep = tokens[-max_tokens:]
        elif strategy == "middle":
            front = max_tokens // 2
            back = max_tokens - front
            keep = tokens[:front] + tokens[-back:]
        else:
            keep = tokens[:max_tokens]
        return self._encoder.decode(keep)


def slugify_text(value: str) -> str:
    """Return a filesystem-friendly slug for model names and run-directory labels."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip())
    cleaned = cleaned.strip("-._")
    return cleaned or "default"
