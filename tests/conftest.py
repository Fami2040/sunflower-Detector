import os
import sys
from pathlib import Path


# Ensure repo-root imports work in lightweight CI.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Scripts gate on the harchoc conda env; unit tests run under base/python CI images.
os.environ.setdefault("HARCHOC_ALLOW_BASE_PYTHON", "1")

