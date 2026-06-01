from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def require_conda_env(*, env_name: str | None = None) -> None:
    """
    Fail fast if a script is executed outside the intended conda/mamba env.

    This prevents accidental `python -m pip install ...` into base/system Python
    when the repo expects `mamba run -n <env> ...` or `conda activate <env>`.
    """
    if env_name is None:
        from harchoc.ml_env import default_mamba_env

        env_name = default_mamba_env()
    if os.getenv("HARCHOC_ALLOW_BASE_PYTHON", "").strip() in {"1", "true", "yes"}:
        return

    conda_prefix = (os.getenv("CONDA_PREFIX") or "").strip()
    exe = Path(sys.executable).resolve()

    # If conda isn't active, this is almost certainly base/system Python.
    if not conda_prefix:
        raise SystemExit(
            "Refusing to run outside a conda env.\n"
            f"python: {exe}\n"
            "Run via `mamba run -n harchoc python ...` or `conda activate harchoc`.\n"
            "Override with HARCHOC_ALLOW_BASE_PYTHON=1."
        )

    prefix_path = Path(conda_prefix).resolve()
    expected_suffix = Path("envs") / env_name
    in_expected_env = prefix_path.name == env_name or str(prefix_path).endswith(str(expected_suffix))
    exe_in_prefix = str(exe).startswith(str(prefix_path) + os.sep)
    if not (in_expected_env and exe_in_prefix):
        raise SystemExit(
            "Refusing to run in the wrong conda env.\n"
            f"CONDA_PREFIX: {prefix_path}\n"
            f"python: {exe}\n"
            f"Expected env: {env_name}\n"
            "Run via `mamba run -n harchoc python ...` or `conda activate harchoc`.\n"
            "Override with HARCHOC_ALLOW_BASE_PYTHON=1."
        )


def add_dataset_args(p: argparse.ArgumentParser, *, suppress_defaults: bool = False) -> None:
    from harchoc.experiment_cli import add_dataset_args as _add

    _add(p, suppress_defaults=suppress_defaults)


def extend_dataset_argv(
    argv: list[str],
    *,
    manifest: str,
    default_dataset_name: str,
    dataset_env: dict[str, str] | None = None,
) -> list[str]:
    """Append manifest/default-dataset-name and optional dataset overrides."""
    out = list(argv)
    out.extend(["--manifest", manifest, "--default-dataset-name", default_dataset_name])
    if dataset_env:
        if "DATASET_NAME" in dataset_env:
            out.extend(["--dataset-name", dataset_env["DATASET_NAME"]])
        if "DATASET_ROOT" in dataset_env:
            out.extend(["--dataset-root", dataset_env["DATASET_ROOT"]])
        if "YOLO_DATA_YAML" in dataset_env:
            out.extend(["--yolo-data-yaml", dataset_env["YOLO_DATA_YAML"]])
    return out


def add_dry_run_arg(p: argparse.ArgumentParser, *, suppress_defaults: bool = False) -> None:
    if suppress_defaults:
        p.add_argument(
            "--dry-run",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Parse args and exit before heavy computation / I/O.",
        )
    else:
        p.add_argument(
            "--dry-run",
            action="store_true",
            help="Parse args and exit before heavy computation / I/O.",
        )


def is_quiet() -> bool:
    return os.getenv("HARCHOC_QUIET", "").strip().lower() in ("1", "true", "yes")


def cli_print(msg: str) -> None:
    """Stdout helper; suppressed when HARCHOC_QUIET=1 (used in unit tests)."""
    if is_quiet():
        return
    print(msg)


def read_json(path: str | Path) -> Any:
    from harchoc.json_io import load_json

    return load_json(path)


def read_json_dict(path: str | Path) -> dict[str, Any]:
    from harchoc.json_io import load_json_dict

    return load_json_dict(path)


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", "utf-8")
    return out_path


def require_existing_dir(path: str | Path, *, what: str, hint: str | None = None) -> Path:
    p = Path(path)
    if p.is_dir():
        return p
    extra = f"\nHint: {hint}" if hint else ""
    raise SystemExit(
        f"{what} does not exist: {p}\n"
        f"Fix by setting DATASET_ROOT or YOLO_DATA_YAML, or updating data/manifest.json.{extra}"
    )


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


EXAMPLE_GT_JSON = "data/examples/gt.json"
EXAMPLE_PREDS_JSON = "data/examples/preds.json"


def add_light_mode_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--light",
        action="store_true",
        help="Use tracked example GT/preds under data/examples/ (no model inference).",
    )


def resolve_light_gt_preds(
    *,
    repo_root: Path,
    gt_json: str,
    preds_json: str,
) -> tuple[Path, Path]:
    gt_path = Path(gt_json).expanduser() if (gt_json or "").strip() else (repo_root / EXAMPLE_GT_JSON)
    preds_path = Path(preds_json).expanduser() if (preds_json or "").strip() else (repo_root / EXAMPLE_PREDS_JSON)
    if not gt_path.is_absolute():
        gt_path = (repo_root / gt_path).resolve()
    else:
        gt_path = gt_path.resolve()
    if not preds_path.is_absolute():
        preds_path = (repo_root / preds_path).resolve()
    else:
        preds_path = preds_path.resolve()
    if not gt_path.is_file():
        raise SystemExit(f"--light requires GT JSON at {gt_path}")
    if not preds_path.is_file():
        raise SystemExit(f"--light requires preds JSON at {preds_path}")
    return gt_path, preds_path

