"""Deprecated: use ``python scripts/experiment.py dataset-root`` instead."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from harchoc.script_entry import bootstrap_repo_imports

bootstrap_repo_imports()


def main() -> None:
    warnings.warn(
        "scripts/archive/dataset_from_manifest.py is deprecated; use: "
        "python scripts/experiment.py dataset-root",
        DeprecationWarning,
        stacklevel=1,
    )
    from harchoc.datasets import dataset_root_from_manifest

    name = os.getenv("DATASET_NAME", "sunflower-cvat-1093")
    print(dataset_root_from_manifest(dataset_name=name))


if __name__ == "__main__":
    main()
