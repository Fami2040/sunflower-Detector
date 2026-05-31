"""Script entry bootstrap: repo root on ``sys.path`` and ``scripts._path`` side effects."""

from __future__ import annotations

import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_repo_root_on_syspath() -> None:
    """Insert repository root at the front of ``sys.path`` if missing."""
    root_s = str(_repo_root())
    if root_s not in sys.path:
        sys.path.insert(0, root_s)


def bootstrap_repo_imports() -> None:
    """
    Standard import bootstrap for top-level ``scripts/*.py`` entrypoints.

    Ensures the repo root is on ``sys.path``, then imports ``scripts._path`` (or
    sibling ``_path`` when executed as ``python scripts/foo.py``).
    """
    ensure_repo_root_on_syspath()
    try:
        from scripts import _path  # type: ignore # noqa: F401
    except Exception:
        import _path  # type: ignore # noqa: F401


def prepend_repo_root_for_script(script_file: str | Path) -> None:
    """Insert repo root derived from ``scripts/<name>.py`` before importing ``harchoc``."""
    script_path = Path(script_file).resolve()
    repo_root = script_path.parent.parent
    root_s = str(repo_root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)


def bootstrap_script_file(script_file: str | Path) -> None:
    """Prepend repo root for ``scripts/<name>.py``, then run :func:`bootstrap_repo_imports`."""
    prepend_repo_root_for_script(script_file)
    bootstrap_repo_imports()
