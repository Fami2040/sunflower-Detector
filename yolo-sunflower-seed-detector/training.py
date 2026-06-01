"""
Legacy entrypoint — delegates to ``scripts/train.py``.

Prefer: ``mamba run -n harchoc python scripts/train.py --config configs/experiments/train_yolov8m_baseline.json``
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from scripts import _path  # type: ignore # noqa: F401
    except Exception:
        import scripts._path as _path  # type: ignore # noqa: F401

    from scripts.train import main as train_main

    return int(train_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
