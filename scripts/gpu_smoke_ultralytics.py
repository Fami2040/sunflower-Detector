"""Deprecated: use ``python scripts/check_gpu.py smoke-ultralytics``."""

from __future__ import annotations

import sys
from pathlib import Path

_r = Path(__file__).resolve().parent.parent
(str(_r) not in sys.path) and sys.path.insert(0, str(_r))

from scripts._common_cli import eprint

_DEPRECATION = (
    "Deprecated: use `mamba run -n harchoc python scripts/check_gpu.py smoke-ultralytics` "
    "(this shim exits 2)."
)


def main(argv: list[str] | None = None) -> int:
    eprint(_DEPRECATION)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
