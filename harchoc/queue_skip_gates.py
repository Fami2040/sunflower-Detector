"""Unified GPU queue completion gates (summary, HSP eval artifacts, zoo matrix)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harchoc.aug_smoke_runner import (
    DEFAULT_OUT_DIR,
    extract_count_mae,
    finalize_smoke_job,
    hsp_eval_artifacts_verified,
    hsp_eval_prefix_paths,
    resolve_train_weights,
)

_HSP_EVAL_JOB_KINDS = frozenset(
    {
        "aug_smoke",
        "aug_sweep_15",
        "aug_sweep_100",
        "rtdetr_smoke",
        "train_compare",
        "amp_smoke",
        "sg_smoke",
    }
)


def job_summary_path(job: dict[str, Any]) -> str | None:
    skip_if = job.get("skip_if") or {}
    if skip_if.get("summary"):
        return str(skip_if["summary"])
    if job.get("summary_path"):
        return str(job["summary_path"])
    run_name = str(job.get("run_name") or job.get("id") or "")
    if not run_name:
        return None
    out_dir = str(job.get("eval_out_dir") or DEFAULT_OUT_DIR)
    if str(job.get("kind") or "") == "amp_smoke":
        out_dir = str(job.get("eval_out_dir") or "reports/hsp")
    return f"{out_dir}/{run_name}_summary.json"


def job_hsp_out_dir(job: dict[str, Any]) -> str:
    if str(job.get("kind") or "") == "amp_smoke":
        return str(job.get("eval_out_dir") or "reports/hsp")
    return str(job.get("eval_out_dir") or DEFAULT_OUT_DIR)


def job_needs_hsp_eval(job: dict[str, Any]) -> bool:
    kind = str(job.get("kind") or "")
    if kind not in _HSP_EVAL_JOB_KINDS:
        return False
    if job.get("skip_eval"):
        return False
    return True


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _matrix_group_bench_configs(repo_root: Path, matrix_group: str) -> list[Any]:
    from harchoc.bench_config import is_bench_row_config, load_bench_config

    if not matrix_group.strip():
        return []
    want = {matrix_group.strip()}
    bench_dir = repo_root / "configs" / "bench"
    paths = sorted(bench_dir.glob("*.yaml")) + sorted(bench_dir.glob("*.json"))
    out: list[Any] = []
    seen: set[str] = set()
    for p in paths:
        if not is_bench_row_config(p):
            continue
        cfg = load_bench_config(p)
        if not want.intersection(set(cfg.groups)):
            continue
        key = str(cfg.path.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(cfg)
    return out


def _matrix_run_has_test_mae(repo_root: Path, run: dict[str, Any]) -> bool:
    mae = run.get("test_count_mae")
    if mae is not None:
        return True
    err = run.get("error_test_report") or (run.get("test_eval") or {}).get("error_json")
    if not err:
        return False
    err_p = Path(str(err))
    if not err_p.is_absolute():
        err_p = repo_root / err_p
    mae_val, _ = extract_count_mae(err_p) if err_p.is_file() else (None, None)
    return mae_val is not None


def matrix_train_verified(
    repo_root: Path,
    train_out: str | Path,
    matrix_group: str,
) -> tuple[bool, str]:
    """True when matrix_train.json has ok weights + test MAE for every row in matrix_group."""
    path = (repo_root / str(train_out)).resolve()
    if not path.is_file():
        return False, ""
    try:
        doc = _read_json(path)
    except Exception:
        return False, ""
    if str(doc.get("schema_version") or "") != "benchmark_matrix_train.v1":
        return False, ""

    configs = _matrix_group_bench_configs(repo_root, matrix_group)
    if not configs:
        return False, ""

    runs = doc.get("runs") or []
    by_config = {str(r.get("config_path") or ""): r for r in runs if isinstance(r, dict)}
    by_name = {str(r.get("name") or ""): r for r in runs if isinstance(r, dict)}

    for cfg in configs:
        run = by_config.get(str(cfg.path.resolve())) or by_name.get(str(cfg.name))
        if not isinstance(run, dict):
            return False, ""
        if str(run.get("status") or "") != "ok":
            return False, ""
        if not run.get("weights"):
            return False, ""
        if not _matrix_run_has_test_mae(repo_root, run):
            return False, ""

    return True, f"matrix_train verified: {path} ({matrix_group}, {len(configs)} rows)"


def ensure_hsp_summary_from_artifacts(
    job: dict[str, Any],
    *,
    repo_root: Path,
) -> bool:
    """Write missing summary JSON from verified HSP eval artifacts. Returns True if complete."""
    summary_rel = job_summary_path(job)
    if not summary_rel:
        return False
    summary_p = repo_root / summary_rel
    if summary_p.is_file():
        try:
            obj = _read_json(summary_p)
            if obj.get("status") == "complete" and obj.get("test_count_mae") is not None:
                return True
        except Exception:
            pass

    run_name = str(job.get("run_name") or "")
    if not run_name:
        return False
    out_dir = job_hsp_out_dir(job)
    if not hsp_eval_artifacts_verified(repo_root, run_name=run_name, out_dir=out_dir):
        return False

    weights = resolve_train_weights(repo_root=repo_root, run_name=run_name)
    if weights is None:
        return False

    train_config = str(job.get("train_config") or "")
    smoke_id = str(
        job.get("smoke_id")
        or (job.get("backlog") or [None])[0]
        or job.get("id")
        or run_name
    )
    payload = finalize_smoke_job(
        repo_root=repo_root,
        run_name=run_name,
        train_config=train_config,
        weights=weights,
        summary_path=summary_rel,
        smoke_id=smoke_id,
        locked_conf_from=str(
            (job.get("skip_if") or {}).get("locked_conf_from")
            or job.get("locked_conf_from")
            or "reports/hsp/threshold_val.json"
        ),
        out_dir=out_dir,
        arch_ticket=",".join(job.get("backlog") or []),
    patch_index=(
        str(job.get("kind") or "") in ("aug_smoke", "aug_sweep_15")
        and bool(job.get("smoke_id"))
        and not bool(job.get("eval_only"))
    ),
        refresh_leaderboard=False,
    )
    return payload.get("status") == "complete" and payload.get("test_count_mae") is not None


def existing_bench_run_weights(cfg: Any, runs_dir: Path) -> Path | None:
    """Return best.pt/pth under runs_dir/{bench_run_name}/weights/ when present."""
    from harchoc.bench_config import _bench_run_name

    run_dir = runs_dir / _bench_run_name(cfg)
    for rel in ("weights/best.pt", "weights/best.pth"):
        p = run_dir / rel
        if p.is_file():
            return p.resolve()
    return None


def matrix_run_is_complete(repo_root: Path, run: dict[str, Any]) -> bool:
    if str(run.get("status") or "") != "ok":
        return False
    if not run.get("weights"):
        return False
    return _matrix_run_has_test_mae(repo_root, run)
