"""Helpers shared by the user-facing wrapper scripts.

These helpers keep the top-level scripts short and explicit. The wrappers are intentionally
simple: they build readable defaults, call the copied experiment cores, and optionally run a
final grading pass.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Iterator


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def timestamp_now() -> str:
    """Return the timestamp format used throughout the package's run directories."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


@contextmanager
def temporary_argv(argv: Iterable[str]) -> Iterator[None]:
    """Temporarily replace ``sys.argv`` so copied script-style modules can be called directly."""
    old_argv = sys.argv[:]
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = old_argv


def run_script_main(entry_name: str, main_fn: Callable[[], None], argv: Iterable[str]) -> None:
    """Execute a copied ``main()`` function as if it were called from the command line."""
    with temporary_argv([entry_name, *argv]):
        main_fn()
