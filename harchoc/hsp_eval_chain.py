"""Shared HSP eval stage argv builders and subprocess runners."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from harchoc.hsp_export_protocol import (
    DEFAULT_EXPORT_MAX_DET,
    DEFAULT_SPLIT_FILE,
    EXPORT_DEVICE,
    eval_export_cli_flags,
)
from harchoc.ml_env import run_repo_python

DEFAULT_LOCKED_CONF_FROM = "reports/hsp/threshold_val.json"


def infer_smoke_eval_backend(weights: str | Path) -> str:
    return "supergradients" if Path(weights).suffix.lower() == ".pth" else "ultralytics"


def extract_count_mae(error_json_path: str | Path) -> tuple[float | None, dict[str, Any] | None]:
    p = Path(error_json_path)
    if not p.is_file():
        return None, None
    obj = json.loads(p.read_text(encoding="utf-8"))
    cm = obj.get("counting_metrics") or {}
    mae = cm.get("mae")
    if mae is None:
        return None, None
    return float(mae), cm.get("mae_ci")


def hsp_eval_prefix_paths(repo_root: Path, run_name: str, out_dir: str) -> dict[str, Path]:
    rr = repo_root.resolve()
    prefix = rr / out_dir / run_name
    return {
        "gt": prefix.with_name(prefix.name + "_gt.json"),
        "preds": prefix.with_name(prefix.name + "_preds.json"),
        "eval": prefix.with_name(prefix.name + "_eval.json"),
        "error": prefix.with_name(prefix.name + "_error.json"),
    }


def _repo_relative(path: str | Path, repo_root: Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(p.resolve())


def build_ultralytics_export_argv(
    *,
    repo_root: Path,
    run_name: str,
    weights: str | Path,
    out_dir: str,
    max_det: int = DEFAULT_EXPORT_MAX_DET,
    imgsz: int = 1280,
    split_file: str = DEFAULT_SPLIT_FILE,
) -> list[str]:
    rr = repo_root.resolve()
    prefix = str((rr / out_dir / run_name).relative_to(rr))
    gt_json = f"{prefix}_gt.json"
    preds_json = f"{prefix}_preds.json"
    eval_json = f"{prefix}_eval.json"
    return [
        "scripts/eval.py",
        "--weights",
        str(weights),
        "--split-file",
        split_file,
        "--export-only",
        "--export-gt-json",
        gt_json,
        "--export-preds-json",
        preds_json,
        *eval_export_cli_flags(max_det=max_det, device=EXPORT_DEVICE),
        "--imgsz",
        str(imgsz),
        "--out",
        eval_json,
    ]


def build_error_analysis_argv(
    gt_json: str | Path,
    preds_json: str | Path,
    locked_conf_from: str,
    out_json: str | Path,
    *,
    repo_root: Path | None = None,
) -> list[str]:
    rr = (repo_root or Path.cwd()).resolve()
    return [
        "scripts/error_analysis.py",
        "--gt-json",
        _repo_relative(gt_json, rr),
        "--preds-json",
        _repo_relative(preds_json, rr),
        "--locked-conf-from",
        locked_conf_from,
        "--out",
        _repo_relative(out_json, rr),
    ]


def build_ultralytics_hsp_stages(
    *,
    repo_root: Path,
    run_name: str,
    weights: str | Path,
    locked_conf_from: str = DEFAULT_LOCKED_CONF_FROM,
    out_dir: str,
    max_det: int = DEFAULT_EXPORT_MAX_DET,
    imgsz: int = 1280,
) -> list[tuple[str, list[str]]]:
    paths = hsp_eval_prefix_paths(repo_root, run_name, out_dir)
    export_argv = build_ultralytics_export_argv(
        repo_root=repo_root,
        run_name=run_name,
        weights=weights,
        out_dir=out_dir,
        max_det=max_det,
        imgsz=imgsz,
    )
    error_argv = build_error_analysis_argv(
        paths["gt"],
        paths["preds"],
        locked_conf_from,
        paths["error"],
        repo_root=repo_root,
    )
    return [
        ("eval_export", export_argv),
        ("error_analysis", error_argv),
    ]


def hsp_eval_artifacts_verified(repo_root: Path, *, run_name: str, out_dir: str) -> bool:
    err = hsp_eval_prefix_paths(repo_root, run_name, out_dir)["error"]
    if not err.is_file():
        return False
    mae, _ = extract_count_mae(err)
    return mae is not None


def run_stages(
    stages: list[tuple[str, list[str]]],
    *,
    repo_root: Path,
    dry_run: bool = False,
    on_stage: Callable[[str, list[str]], None] | None = None,
    env: dict[str, str] | None = None,
    skip_stage: Callable[[str], bool] | None = None,
) -> None:
    rr = repo_root.resolve()
    run_env = {**dict(os.environ), **(env or {})}
    for stage_id, argv in stages:
        if on_stage is not None:
            on_stage(stage_id, argv)
        if dry_run:
            from harchoc.ml_env import repo_python_cmd

            print(f"# {stage_id}: {' '.join(repo_python_cmd(argv))}")
            continue
        if skip_stage is not None and skip_stage(stage_id):
            continue
        proc = run_repo_python(argv, repo_root=rr, env=run_env)
        if proc.returncode != 0:
            raise RuntimeError(f"hsp eval stage {stage_id!r} failed: exit {proc.returncode}")


def ultralytics_hsp_artifact_paths(repo_root: Path, *, run_name: str, out_dir: str) -> dict[str, str]:
    paths = hsp_eval_prefix_paths(repo_root, run_name, out_dir)
    return {k: str(v.resolve()) for k, v in paths.items()}


def run_hsp_eval_chain(
    *,
    repo_root: str | Path,
    run_name: str,
    weights: str | Path,
    locked_conf_from: str = DEFAULT_LOCKED_CONF_FROM,
    out_dir: str,
    max_det: int = DEFAULT_EXPORT_MAX_DET,
    imgsz: int = 1280,
    backend: str = "ultralytics",
    dry_run: bool = False,
    on_stage: Callable[[str, list[str]], None] | None = None,
    env: dict[str, str] | None = None,
    model_id: str = "yolo_nas_s",
) -> dict[str, Any]:
    rr = Path(repo_root).resolve()
    picked = str(backend).strip().lower()
    if picked == "auto":
        picked = infer_smoke_eval_backend(Path(weights))
    if picked == "supergradients":
        from harchoc.supergradients_eval import run_sg_hsp_eval_chain

        return run_sg_hsp_eval_chain(
            repo_root=rr,
            run_name=run_name,
            weights=weights,
            locked_conf_from=locked_conf_from,
            out_dir=out_dir,
            max_det=max_det,
            model_id=model_id,
            dry_run=dry_run,
            on_stage=on_stage,
            env=env,
        )
    if picked == "external":
        raise ValueError("external backend uses eval_hsp_for_bench, not run_hsp_eval_chain")

    stages = build_ultralytics_hsp_stages(
        repo_root=rr,
        run_name=run_name,
        weights=weights,
        locked_conf_from=locked_conf_from,
        out_dir=out_dir,
        max_det=max_det,
        imgsz=imgsz,
    )
    prefix_paths = hsp_eval_prefix_paths(rr, run_name, out_dir)

    def _skip(stage_id: str) -> bool:
        if stage_id == "eval_export":
            return prefix_paths["gt"].is_file() and prefix_paths["preds"].is_file()
        if stage_id == "error_analysis":
            return hsp_eval_artifacts_verified(rr, run_name=run_name, out_dir=out_dir)
        return False

    run_stages(
        stages,
        repo_root=rr,
        dry_run=dry_run,
        on_stage=on_stage,
        env=env,
        skip_stage=_skip,
    )
    arts = ultralytics_hsp_artifact_paths(rr, run_name=run_name, out_dir=out_dir)
    return {
        "eval_json": arts["eval"],
        "error_json": arts["error"],
        "gt_json": arts["gt"],
        "preds_json": arts["preds"],
    }
