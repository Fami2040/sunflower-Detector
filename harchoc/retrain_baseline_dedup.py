"""Retrain dedup vs zoo matrix baselines (tensor-identical checkpoints, redundant close_mosaic)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from harchoc.aug_smoke_runner import resolve_train_weights
from harchoc.hsp_eval_chain import hsp_eval_prefix_paths
from harchoc.json_io import load_json_dict
from harchoc.schemas import with_schema_version

RETRAIN_DEDUP_SCHEMA = "retrain_baseline_dedup.v1"
DEFAULT_BASELINE_RUNS_DIR = "runs/hsp_zoo"
DEFAULT_BASELINE_HSP_OUT_DIR = "reports/hsp"
DEFAULT_MINIMAL_AUG = "configs/aug/robustness_minimal.yaml"


def _dedup_enabled(job: dict[str, Any], defaults: dict[str, Any] | None) -> bool:
    defs = defaults or {}
    if job.get("dedup_baseline") is False:
        return False
    return bool(
        job.get("dedup_baseline_run_name")
        or job.get("dedup_baseline_weights")
        or defs.get("dedup_baseline_run_name")
        or defs.get("dedup_baseline_weights")
        or defs.get("dedup_baseline_runs_dir")
    )


def _model_stem_from_train_config(repo_root: Path, train_config: str) -> str | None:
    from harchoc.train_config import resolve_train_config_extends

    p = repo_root / train_config
    if not p.is_file():
        return None
    raw = load_json_dict(p)
    merged = resolve_train_config_extends(raw, repo_root=repo_root, config_path=p)
    model = str(merged.get("model") or "").strip()
    if not model:
        return None
    return Path(model).stem.replace(".pt", "")


def infer_zoo_baseline_run_name(repo_root: Path, train_config: str, *, epochs: int = 100) -> str | None:
    stem = _model_stem_from_train_config(repo_root, train_config)
    if not stem:
        return None
    return f"{stem}_e{epochs}_s0"


def resolve_dedup_baseline_weights(
    job: dict[str, Any],
    *,
    repo_root: Path,
    defaults: dict[str, Any] | None = None,
) -> Path | None:
    """Zoo matrix ``best.pt`` for tensor dedup (explicit path or ``runs/hsp_zoo/{run}``)."""
    defs = defaults or {}
    explicit = job.get("dedup_baseline_weights") or defs.get("dedup_baseline_weights")
    if explicit and str(explicit).strip():
        p = Path(str(explicit).strip())
        if not p.is_absolute():
            p = repo_root / p
        return p.resolve() if p.is_file() else None

    run_name = str(
        job.get("dedup_baseline_run_name")
        or defs.get("dedup_baseline_run_name")
        or ""
    ).strip()
    if not run_name:
        tc = str(job.get("train_config") or "").strip()
        if tc:
            inferred = infer_zoo_baseline_run_name(repo_root, tc)
            if inferred:
                run_name = inferred
    if not run_name:
        return None

    runs_dir = str(job.get("dedup_baseline_runs_dir") or defs.get("dedup_baseline_runs_dir") or DEFAULT_BASELINE_RUNS_DIR)
    w = resolve_train_weights(repo_root=repo_root, run_name=run_name, out_dir=runs_dir)
    if w is not None:
        return w
    return resolve_train_weights(repo_root=repo_root, run_name=run_name)


def resolve_dedup_baseline_hsp_run_name(job: dict[str, Any], *, defaults: dict[str, Any] | None = None) -> str:
    defs = defaults or {}
    return str(
        job.get("dedup_baseline_hsp_run_name")
        or job.get("dedup_baseline_run_name")
        or defs.get("dedup_baseline_hsp_run_name")
        or defs.get("dedup_baseline_run_name")
        or ""
    ).strip()


def checkpoint_max_tensor_diff(path_a: Path, path_b: Path) -> float | None:
    """Max abs diff over shared ``model`` state dict tensors; ``None`` if incomparable."""
    import torch

    def _state_dict(ckpt: object) -> dict[str, Any]:
        if isinstance(ckpt, dict) and "model" in ckpt:
            m = ckpt["model"]
            if hasattr(m, "state_dict"):
                return m.state_dict()
        if hasattr(ckpt, "state_dict"):
            return ckpt.state_dict()
        return {}

    try:
        ck_a = torch.load(path_a, map_location="cpu", weights_only=False)
        ck_b = torch.load(path_b, map_location="cpu", weights_only=False)
    except Exception:
        return None

    sd_a = _state_dict(ck_a)
    sd_b = _state_dict(ck_b)
    if not sd_a or not sd_b:
        return None

    max_d = 0.0
    compared = 0
    for key, ta in sd_a.items():
        if key not in sd_b:
            continue
        tb = sd_b[key]
        if not hasattr(ta, "shape") or ta.shape != tb.shape:
            return None
        d = (ta.float() - tb.float()).abs().max().item()
        max_d = max(max_d, d)
        compared += 1
    if compared == 0:
        return None
    return max_d


def checkpoints_tensor_identical(
    path_a: Path | str,
    path_b: Path | str,
    *,
    atol: float = 0.0,
) -> bool:
    pa, pb = Path(path_a), Path(path_b)
    if not pa.is_file() or not pb.is_file():
        return False
    if pa.resolve() == pb.resolve():
        return True
    diff = checkpoint_max_tensor_diff(pa, pb)
    if diff is None:
        return False
    return diff <= atol


def recipe_matches_zoo_baseline_ignoring_close_mosaic(
    job: dict[str, Any],
    *,
    repo_root: Path,
    defaults: dict[str, Any] | None = None,
) -> bool:
    """True when job train recipe matches zoo ``train_bench_base`` + minimal aug except ``close_mosaic``."""
    from harchoc.train_config import resolve_train_config_extends

    tc = str(job.get("train_config") or "").strip()
    if not tc:
        return False
    stem = _model_stem_from_train_config(repo_root, tc)
    if not stem:
        return False

    bench_path = repo_root / "configs/experiments/train_bench_base.json"
    if not bench_path.is_file():
        return False
    zoo_raw = resolve_train_config_extends(
        {
            "extends": "configs/experiments/train_bench_base.json",
            "model": f"{stem}.pt",
            "batch": 1,
            "epochs": 100,
            "aug_config": DEFAULT_MINIMAL_AUG,
        },
        repo_root=repo_root,
        config_path=bench_path,
    )
    def _fp_ignore_close_mosaic(cfg: dict[str, Any]) -> str:
        import hashlib
        import json

        from harchoc.train_config import effective_train_aug_merged, EFFECTIVE_RECIPE_KEYS

        merged = effective_train_aug_merged(cfg, repo_root=repo_root)
        subset = {k: merged.get(k) for k in EFFECTIVE_RECIPE_KEYS if k != "close_mosaic"}
        payload = json.dumps(subset, sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    job_cfg = resolve_train_config_extends(
        load_json_dict(repo_root / tc), repo_root=repo_root, config_path=repo_root / tc
    )
    return _fp_ignore_close_mosaic(job_cfg) == _fp_ignore_close_mosaic(zoo_raw)


def should_pre_skip_redundant_train(
    job: dict[str, Any],
    *,
    repo_root: Path,
    baseline: Path,
    defaults: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Skip GPU train when manifest requests it and recipe is zoo-parity close_mosaic-only."""
    if not bool(job.get("dedup_pre_skip_train", (defaults or {}).get("dedup_pre_skip_train"))):
        return False, ""
    if not baseline.is_file():
        return False, ""
    if not recipe_matches_zoo_baseline_ignoring_close_mosaic(job, repo_root=repo_root, defaults=defaults):
        return False, ""
    return True, f"zoo_bench_redundant_close_mosaic_only (baseline {baseline.name})"


def seed_run_weights_from_baseline(
    *,
    baseline: Path,
    run_name: str,
    repo_root: Path,
) -> Path:
    """Copy baseline ``best.pt`` to Ultralytics run dirs so downstream resolve_train_weights works."""
    dests = [
        repo_root / "runs" / run_name / "weights" / "best.pt",
        repo_root / "runs" / "detect" / "runs" / run_name / "weights" / "best.pt",
    ]
    written: Path | None = None
    for dest in dests:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(baseline, dest)
        written = dest.resolve()
    if written is None:
        raise RuntimeError(f"failed to seed weights for run {run_name!r}")
    return written


def baseline_hsp_eval_artifacts(
    *,
    repo_root: Path,
    baseline_hsp_run_name: str,
    out_dir: str = DEFAULT_BASELINE_HSP_OUT_DIR,
) -> dict[str, str]:
    paths = hsp_eval_prefix_paths(repo_root, baseline_hsp_run_name, out_dir)
    return {k: str(v.resolve()) for k, v in paths.items()}


def build_dedup_summary_overlay(
    *,
    job: dict[str, Any],
    job_context: dict[str, Any],
    baseline: Path,
) -> dict[str, Any]:
    baseline_run = resolve_dedup_baseline_hsp_run_name(job) or ""
    return with_schema_version(
        {
            "baseline_run_name": baseline_run,
            "baseline_weights": str(baseline.resolve()),
            "train_skipped": bool(job_context.get("dedup_pre_skip_train")),
            "eval_skipped": bool(job_context.get("dedup_eval_skipped")),
            "tensor_identical": bool(job_context.get("dedup_baseline_identical")),
            "pre_skip_reason": job_context.get("dedup_pre_skip_reason"),
            "max_tensor_diff": job_context.get("dedup_max_tensor_diff"),
        },
        schema_version=RETRAIN_DEDUP_SCHEMA,
    )


def annotate_train_compare_dedup_skips(
    jobs: list[dict[str, Any]],
    *,
    repo_root: Path,
    defaults: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Mark ``train_compare`` jobs skipped when target weights already exist and match baseline tensors.
    """
    defs = defaults or {}
    out: list[dict[str, Any]] = []
    for job in jobs:
        j = dict(job)
        if str(j.get("kind") or "") != "train_compare" or not _dedup_enabled(j, defs):
            out.append(j)
            continue
        baseline = resolve_dedup_baseline_weights(j, repo_root=repo_root, defaults=defs)
        run_name = str(j.get("run_name") or "").strip()
        if not baseline or not run_name:
            out.append(j)
            continue
        existing = resolve_train_weights(repo_root=repo_root, run_name=run_name)
        if existing and checkpoints_tensor_identical(existing, baseline):
            j["skip"] = True
            j["skip_reason"] = (
                f"train_compare tensor-identical to baseline {baseline} (skip retrain+HSP eval)"
            )
        out.append(j)
    return out
