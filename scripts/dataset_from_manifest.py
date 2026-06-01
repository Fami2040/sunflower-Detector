from __future__ import annotations

import os

import sys; from pathlib import Path; _r = Path(__file__).resolve().parent.parent; (str(_r) not in sys.path) and sys.path.insert(0, str(_r)); from harchoc.script_entry import bootstrap_repo_imports; bootstrap_repo_imports()
from harchoc.datasets import dataset_root_from_manifest


def main() -> None:
    name = os.getenv("DATASET_NAME", "sunflower-cvat-1093")
    root = dataset_root_from_manifest(dataset_name=name)
    print(str(root))


if __name__ == "__main__":
    main()

