from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root_on_syspath() -> None:
    """
    Allow running scripts both as:
      - python scripts/foo.py
      - python -m scripts.foo

    When executed as a file, Python sets sys.path[0] to scripts/, which makes
    `import harchoc` fail. We fix that by adding the repo root to sys.path.
    """
    repo_root = Path(__file__).resolve().parents[1]
    s = str(repo_root)
    if s not in sys.path:
        sys.path.insert(0, s)


# Apply on import to keep all scripts consistent.
ensure_repo_root_on_syspath()

