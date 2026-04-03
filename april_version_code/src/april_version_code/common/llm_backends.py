"""Shared helpers for model/backend selection in the local wrapper scripts.

The copied experiment cores still validate backend credentials themselves. These helpers keep
the outer scripts readable and centralize the few environment conventions we expect a human to
interact with directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

from april_version_code.common.token_utils import slugify_text


def load_local_env(package_root: Path) -> None:
    """Load ``.env`` if it exists next to the package root.

    The experiment cores also call ``load_dotenv`` internally. Loading it here is harmless and
    makes failures easier to understand when a wrapper exits before the core runner starts.
    """
    env_path = package_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)


def require_backend_credentials(backends: Iterable[str]) -> None:
    """Raise a clear error if a requested backend is missing its expected API key."""
    normalized = {str(backend).strip() for backend in backends if str(backend).strip()}
    if 'openrouter' in normalized and not (os.getenv('OPENROUTER_API_KEY') or '').strip():
        raise RuntimeError('OPENROUTER_API_KEY must be set for the openrouter backend.')
    if 'gemini' in normalized:
        gemini_key = (os.getenv('GEMINI_API_KEY') or '').strip()
        google_key = (os.getenv('GOOGLE_API_KEY') or '').strip()
        if not gemini_key and not google_key:
            raise RuntimeError('GEMINI_API_KEY or GOOGLE_API_KEY must be set for the gemini backend.')


def model_slug(model_name: str) -> str:
    """Convert a user-facing model identifier into a stable directory name."""
    return slugify_text(model_name.replace('/', '-').replace(':', '-'))
