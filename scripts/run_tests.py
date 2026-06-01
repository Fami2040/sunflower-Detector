#!/usr/bin/env python3
"""CI-parity unittest discover with one-shot summary + failure index."""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from harchoc.script_entry import bootstrap_repo_imports

bootstrap_repo_imports()

from harchoc.unittest_report import main

if __name__ == "__main__":
    raise SystemExit(main())
