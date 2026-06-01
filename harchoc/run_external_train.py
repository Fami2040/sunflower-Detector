"""Entry wrapper for external DETR train scripts (keeps vendor trees pristine).

Usage (also under ``torch.distributed.run``)::

    python harchoc/run_external_train.py <stack> <repo> <train_script> [train.py args...]

For ``stack=deim``, applies ``harchoc.deim_tv_compat`` before executing ``train_script``.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(
            "usage: run_external_train.py <stack> <repo> <train_script> [args...]"
        )
    stack = sys.argv[1]
    repo = Path(sys.argv[2]).resolve()
    script = sys.argv[3]
    train_argv = sys.argv[4:]
    repo_root = Path(__file__).resolve().parents[1]
    for p in (str(repo_root), str(repo)):
        if p not in sys.path:
            sys.path.insert(0, p)
    if stack == "deim":
        from harchoc.deim_tv_compat import apply_deim_torchvision_compat

        apply_deim_torchvision_compat()
    script_path = repo / script
    if not script_path.is_file():
        raise SystemExit(f"missing train script: {script_path}")
    sys.argv = [str(script_path), *train_argv]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
