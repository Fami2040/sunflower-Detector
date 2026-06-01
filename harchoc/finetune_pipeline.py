"""Finetune planning + outcome wiring: weak-tray audit, TIDE hints, HSP counting, canonical gate."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from harchoc.finetune_tray_audit import build_weak_tray_plan, write_weak_tray_plan
from harchoc.hsp_eval_chain import DEFAULT_LOCKED_CONF_FROM as HSP_LOCKED_CONF
from harchoc.hsp_eval_chain import build_error_analysis_argv, extract_count_mae

DEFAULT_LOCKED_CONF_FROM = HSP_LOCKED_CONF
from harchoc.config_coerce import child_dict, pick_int
from harchoc.json_io import load_json_dict
from harchoc.schemas import with_schema_version

FINETUNE_OUTCOME_SCHEMA = "finetune_outcome.v1"

DEFAULT_GLOBAL_MAE_REF = 61.3
DEFAULT_CANONICAL_GATE_PCT = 0.10
DEFAULT_WEAK_PLAN_PATH = "reports/domains/weak_tray_plan.json"
LEGACY_WEAK_PLAN_PATH = "reports/transfer/weak_tray_plan.json"
DEFAULT_TIDE_SUMMARY_PATH = "reports/hsp/tide_bucket_summary.json"
DEFAULT_DOMAIN_COUNT_MAE_PATH = "reports/domains/domain_count_mae.json"
DEFAULT_DOMAIN_EVAL_PATH = "reports/domains/domain_eval.json"
DEFAULT_FINETUNE_BASE_WEIGHTS = "models/best2.pt"
DEFAULT_FINETUNE_OUT_DIR = "runs/transfer"
THRESHOLD_TEST_LOCKED_PATH = "reports/hsp/threshold_test_locked.json"
DUAL_METRIC_PATH = "reports/hsp/dual_metric.json"
FINETUNE_QUEUE_OUT_TEMPLATE = "reports/transfer/finetune_queue_{job_id}.json"

FINETUNE_TRAY_STAGE_TRAIN_CONFIG = {
    1: "configs/experiments/finetune_tray_stage1.json",
    2: "configs/experiments/finetune_tray_stage2.json",
}
FINETUNE_TRAY_STAGE_TRANSFER_CONFIG = {
    1: "configs/transfer/finetune_stage1.yaml",
    2: "configs/transfer/finetune_stage2.yaml",
}


def finetune_train_config_for_stage(job: dict[str, Any], stage: int) -> str:
    return str(job.get("train_config") or FINETUNE_TRAY_STAGE_TRAIN_CONFIG[stage])


def finetune_transfer_config_for_stage(job: dict[str, Any], stage: int) -> str:
    return str(job.get("transfer_config") or FINETUNE_TRAY_STAGE_TRANSFER_CONFIG[stage])


def resolve_finetune_base_weights(
    job: dict[str, Any],
    *,
    stage: int,
    defaults: dict[str, Any] | None = None,
) -> str:
    """Queue/live argv: stage 1 uses best2.pt unless overridden; stage 2 uses stage-1 best.pt."""
    defs = defaults or {}
    explicit = job.get("base_weights") or defs.get("base_weights")
    if explicit:
        return str(explicit)
    if stage == 2:
        stage1_run = str(
            job.get("stage1_run_name") or job.get("base_weights_from_run") or ""
        ).strip()
        if not stage1_run:
            tray = str(job.get("tray_key") or "").strip()
            if tray:
                stage1_run = f"finetune_{tray}_s1"
        if stage1_run:
            out_dir = str(
                job.get("out_dir") or defs.get("finetune_out_dir") or DEFAULT_FINETUNE_OUT_DIR
            )
            return f"{out_dir.rstrip('/')}/{stage1_run}/weights/best.pt"
    return DEFAULT_FINETUNE_BASE_WEIGHTS


def resolve_repo_path(repo_root: Path, raw: str | Path) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    return p


def resolve_weak_plan_rel(
    job: dict[str, Any],
    *,
    defaults: dict[str, Any] | None = None,
    prefer_out: bool = False,
) -> str:
    """Canonical relative path for weak_tray_plan (manifest job + defaults)."""
    defs = defaults or {}
    keys = ("weak_plan_out", "weak_plan") if prefer_out else ("weak_plan", "weak_plan_out")
    for key in keys:
        raw = job.get(key)
        if raw and str(raw).strip():
            return str(raw).strip()
    raw_def = defs.get("weak_plan")
    if raw_def and str(raw_def).strip():
        return str(raw_def).strip()
    return DEFAULT_WEAK_PLAN_PATH


def load_weak_tray_plan(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "recommended_tray_keys": []}
    return load_json_dict(path)


def weak_tray_plan_actionable(path: Path) -> bool:
    """True when weak_tray_plan.v1 exists with at least one recommended tray key."""
    if not path.is_file():
        return False
    try:
        plan = load_weak_tray_plan(path)
    except Exception:
        return False
    keys = [str(k).strip() for k in (plan.get("recommended_tray_keys") or []) if str(k).strip()]
    return bool(keys)


def mirror_weak_plan_legacy_compat(repo_root: Path) -> bool:
    """Symlink legacy transfer path to canonical domains plan when audit wrote domains only."""
    canonical = resolve_repo_path(repo_root, DEFAULT_WEAK_PLAN_PATH)
    if not weak_tray_plan_actionable(canonical):
        return False
    legacy = resolve_repo_path(repo_root, LEGACY_WEAK_PLAN_PATH)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    if legacy.exists() or legacy.is_symlink():
        try:
            if legacy.resolve() == canonical.resolve():
                return True
        except OSError:
            pass
        if legacy.is_file() and not legacy.is_symlink():
            return False
        legacy.unlink(missing_ok=True)
    rel = os.path.relpath(canonical, legacy.parent)
    legacy.symlink_to(rel)
    return True


def tray_keys_from_weak_plan(
    plan: dict[str, Any],
    *,
    top_n: int = 1,
    fallback: list[str] | None = None,
) -> list[str]:
    keys = [str(k).strip() for k in (plan.get("recommended_tray_keys") or []) if str(k).strip()]
    if keys:
        return keys[: max(1, int(top_n))]
    return list(fallback or [])


def tray_baseline_mae(plan: dict[str, Any], tray_key: str) -> float | None:
    for rec in plan.get("top_trays") or []:
        if isinstance(rec, dict) and str(rec.get("tray_key") or "") == tray_key:
            mae = rec.get("count_mae")
            return float(mae) if mae is not None else None
    return None


def ensure_weak_tray_plan(
    *,
    repo_root: Path,
    weak_plan_path: Path,
    count_mae_path: Path,
    domain_eval_path: Path,
    global_mae: float,
    top_k: int = 3,
    write: bool = True,
) -> dict[str, Any]:
    """Build or refresh weak_tray_plan.v1 from domain MAE artifacts."""
    payload = build_weak_tray_plan(
        count_mae_path=count_mae_path if count_mae_path.is_file() else None,
        domain_eval_path=domain_eval_path if domain_eval_path.is_file() else None,
        top_k=top_k,
        global_mae=global_mae,
    )
    if write:
        write_weak_tray_plan(weak_plan_path, payload)
        canonical = resolve_repo_path(repo_root, DEFAULT_WEAK_PLAN_PATH)
        if weak_plan_path.resolve() == canonical.resolve():
            mirror_weak_plan_legacy_compat(repo_root)
    return payload


def finetune_tide_guidance(
    tide_summary_path: Path | None,
    *,
    tray_key: str | None = None,
) -> dict[str, Any]:
    """
  Map global TIDE bucket mass to finetune policy hints (playbook Phase 3).

  High Loc+Bkg → tray_adapt; dominant Miss → recall/max_det before finetune.
  """
    if tide_summary_path is None or not tide_summary_path.is_file():
        return {
            "status": "missing",
            "source": str(tide_summary_path) if tide_summary_path else None,
            "recommended_train_mode": "tray_adapt",
            "notes": "No tide_bucket_summary.json; default tray_adapt when --tray-key set.",
        }
    raw = load_json_dict(tide_summary_path)
    buckets = child_dict(raw, "buckets")
    miss = int(buckets.get("Miss") or 0)
    loc = int(buckets.get("Loc") or 0)
    bkg = int(buckets.get("Bkg") or 0)
    cls = int(buckets.get("Cls") or 0)
    loc_bkg = loc + bkg
    n_errors = int(raw.get("n_errors") or 0) or (miss + loc + bkg + cls)
    dominant = raw.get("dominant_bucket")
    loc_cls_ratio = raw.get("loc_plus_bkg_over_cls_ratio")

    recommended_mode = "tray_adapt"
    notes: list[str] = []
    if n_errors and miss / n_errors > 0.45:
        recommended_mode = "defer_finetune"
        notes.append("Dominant Miss: try max_det / recall before tray_adapt.")
    elif loc_bkg >= cls and loc_bkg >= miss:
        recommended_mode = "tray_adapt"
        notes.append("Loc+Bkg mass high: tray_adapt appropriate (FINETUNE_WEAK_TRAYS § Phase 3).")
    elif dominant == "Miss":
        recommended_mode = "defer_finetune"
        notes.append(f"dominant_bucket={dominant!r}: prefer count levers over finetune.")

    return {
        "status": "ok",
        "source": str(tide_summary_path),
        "tray_key": tray_key,
        "dominant_bucket": dominant,
        "loc_plus_bkg_over_cls_ratio": loc_cls_ratio,
        "buckets": buckets,
        "recommended_train_mode": recommended_mode,
        "notes": "; ".join(notes) if notes else None,
    }


def merge_finetune_eval_section(
    eval_section: dict[str, Any] | None,
    *,
    locked_conf_from: str,
    hsp_counting: bool,
) -> dict[str, Any]:
    """Default HSP eval knobs for tray before/after (locked conf counting MAE)."""
    out: dict[str, Any] = dict(eval_section) if isinstance(eval_section, dict) else {}
    if locked_conf_from and not out.get("locked_conf_from"):
        out["locked_conf_from"] = locked_conf_from
    if hsp_counting:
        out["hsp_counting"] = True
    if out.get("max_det") is None:
        out["max_det"] = 3000
    if out.get("device") is None:
        env_dev = (os.environ.get("HARCHOC_EXPORT_DEVICE") or "").strip()
        out["device"] = env_dev if env_dev else "cpu"
    if out.get("export_device") is None:
        out["export_device"] = out.get("device", "cpu")
    return out


def hsp_transfer_paths(
    reports_dir: Path,
    *,
    phase: str,
    role: str,
    tray_key: str | None,
) -> dict[str, Path]:
    """Per-role HSP artifact paths under the finetune reports dir."""
    stem = f"hsp_{phase}_{role}"
    if tray_key:
        stem = f"{stem}_{tray_key}"
    base = reports_dir / stem
    return {
        "prefix": base,
        "gt": base.with_name(base.name + "_gt.json"),
        "preds": base.with_name(base.name + "_preds.json"),
        "eval": base.with_name(base.name + "_eval.json"),
        "error": base.with_name(base.name + "_error.json"),
    }


def metrics_from_eval_json(eval_path: Path) -> dict[str, Any]:
    if not eval_path.is_file():
        return {}
    try:
        obj = load_json_dict(eval_path)
    except Exception:
        return {}
    out: dict[str, Any] = {}
    if isinstance(obj.get("mAP50"), (int, float)):
        out["mAP50"] = float(obj["mAP50"])
    if isinstance(obj.get("mAP50_95"), (int, float)):
        out["mAP50_95"] = float(obj["mAP50_95"])
    cm = obj.get("counting_metrics")
    if isinstance(cm, dict) and cm.get("mae") is not None:
        out["count_mae"] = float(cm["mae"])
    elif obj.get("count_mae") is not None:
        out["count_mae"] = float(obj["count_mae"])
    return out


def role_metrics_from_phase_paths(
    phase_paths: dict[str, Any] | None,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Normalize tray_eval before/after paths into per-tray count MAE summaries."""
    if not phase_paths:
        return {}
    rr = repo_root.resolve()
    summary: dict[str, Any] = {}

    def _normalize_metrics_blob(blob: Any) -> dict[str, Any]:
        if isinstance(blob, dict) and ("count_mae" in blob or "eval_json" in blob):
            return dict(blob)
        eval_p = Path(str(blob))
        if not eval_p.is_absolute():
            eval_p = (rr / eval_p).resolve()
        err_p = eval_p.parent / eval_p.name.replace("_eval.json", "_error.json")
        mae, _ = extract_count_mae(err_p) if err_p.is_file() else (None, None)
        m = metrics_from_eval_json(eval_p)
        if mae is None:
            mae = m.get("count_mae")
        return {
            "eval_json": str(eval_p),
            "error_json": str(err_p) if err_p.is_file() else None,
            "count_mae": mae,
            "mAP50": m.get("mAP50"),
        }

    for key, val in phase_paths.items():
        if key == "test":
            summary["canonical_test"] = _normalize_metrics_blob(val)
            continue
        if not isinstance(val, dict):
            continue
        tray_summary: dict[str, Any] = {}
        for role, role_val in val.items():
            tray_summary[role] = _normalize_metrics_blob(role_val)
        summary[key] = tray_summary
    return summary


def check_canonical_gate(
    *,
    before_mae: float | None,
    after_mae: float | None,
    global_mae: float,
    gate_pct: float,
) -> dict[str, Any]:
    limit = float(global_mae) * (1.0 + float(gate_pct))
    ok = True
    detail: str | None = None
    if after_mae is None:
        ok = False
        detail = "canonical test MAE missing after finetune"
    elif after_mae > limit:
        ok = False
        detail = f"after MAE {after_mae:.2f} > gate {limit:.2f} (+{100*gate_pct:.0f}% vs {global_mae})"
    return {
        "global_mae_reference": global_mae,
        "gate_pct": gate_pct,
        "limit_mae": limit,
        "before_mae": before_mae,
        "after_mae": after_mae,
        "passed": ok,
        "detail": detail,
    }


def tray_holdout_deltas(
    *,
    before: dict[str, Any],
    after: dict[str, Any],
    tray_keys: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tk in tray_keys:
        b = (before.get(tk) or {}).get("tray") if isinstance(before.get(tk), dict) else {}
        a = (after.get(tk) or {}).get("tray") if isinstance(after.get(tk), dict) else {}
        b_mae = b.get("count_mae") if isinstance(b, dict) else None
        a_mae = a.get("count_mae") if isinstance(a, dict) else None
        delta = None
        if b_mae is not None and a_mae is not None:
            delta = float(a_mae) - float(b_mae)
        rows.append(
            {
                "tray_key": tk,
                "count_mae_before": b_mae,
                "count_mae_after": a_mae,
                "delta_mae": delta,
                "improved": delta < 0 if delta is not None else None,
            }
        )
    return rows


def build_finetune_outcome(
    *,
    tray_keys: list[str],
    tray_eval_before: dict[str, Any] | None,
    tray_eval_after: dict[str, Any] | None,
    repo_root: Path,
    global_mae: float,
    gate_pct: float,
    weak_plan: dict[str, Any] | None,
    tide_guidance: dict[str, Any] | None,
) -> dict[str, Any]:
    before_m = role_metrics_from_phase_paths(tray_eval_before, repo_root=repo_root)
    after_m = role_metrics_from_phase_paths(tray_eval_after, repo_root=repo_root)
    canon_b = (before_m.get("canonical_test") or {}).get("count_mae")
    canon_a = (after_m.get("canonical_test") or {}).get("count_mae")
    gate = check_canonical_gate(
        before_mae=canon_b if isinstance(canon_b, (int, float)) else None,
        after_mae=canon_a if isinstance(canon_a, (int, float)) else None,
        global_mae=global_mae,
        gate_pct=gate_pct,
    )
    deltas = tray_holdout_deltas(before=before_m, after=after_m, tray_keys=tray_keys)
    baselines: dict[str, float | None] = {}
    if weak_plan:
        for tk in tray_keys:
            baselines[tk] = tray_baseline_mae(weak_plan, tk)

    return with_schema_version(
        {
            "tray_keys": tray_keys,
            "tray_holdout": deltas,
            "domain_baseline_mae": baselines,
            "canonical_gate": gate,
            "tide_guidance": tide_guidance,
            "metrics_before": before_m,
            "metrics_after": after_m,
        },
        schema_version=FINETUNE_OUTCOME_SCHEMA,
    )


def finetune_queue_out_path(job_id: str) -> str:
    return FINETUNE_QUEUE_OUT_TEMPLATE.format(job_id=job_id)


def resolve_finetune_locked_conf(
    repo_root: Path,
    *,
    override: str | None = None,
) -> str:
    """Prefer test-locked threshold sweep; else dual_metric inputs; else val locked default."""
    if override and str(override).strip():
        return str(override).strip()
    rr = repo_root.resolve()
    test_locked = rr / THRESHOLD_TEST_LOCKED_PATH
    if test_locked.is_file():
        return THRESHOLD_TEST_LOCKED_PATH
    dm_path = rr / DUAL_METRIC_PATH
    if dm_path.is_file():
        inputs = child_dict(load_json_dict(dm_path), "inputs")
        for key in ("sweep_val", "threshold_val", "sweep_test"):
            rel = inputs.get(key)
            if rel and (rr / str(rel)).is_file():
                return str(rel)
    return DEFAULT_LOCKED_CONF_FROM


def build_finetune_queue_argv(
    job: dict[str, Any],
    *,
    repo_root: Path,
    defaults: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Argv tail for ``scripts/finetune.py`` from a gpu_queue ``finetune_tray`` job."""
    defs = defaults or {}
    job_id = str(job.get("id") or "finetune_tray")
    tray_key = str(job.get("tray_key") or "").strip()
    if not tray_key:
        raise ValueError(f"finetune_tray job {job_id!r}: tray_key is required")
    stage = pick_int(job.get("stage"), default=1)
    if stage not in (1, 2):
        raise ValueError(f"finetune_tray job {job_id!r}: stage must be 1 or 2, got {stage!r}")

    weak_plan = resolve_weak_plan_rel(job, defaults=defs)
    base_weights = resolve_finetune_base_weights(job, stage=stage, defaults=defs)
    locked = resolve_finetune_locked_conf(
        repo_root,
        override=str(job.get("locked_conf_from") or defs.get("locked_conf_from") or "") or None,
    )
    out = str(
        job.get("out")
        or job.get("finetune_out")
        or defs.get("finetune_out")
        or finetune_queue_out_path(job_id)
    )
    run_name = str(
        job.get("run_name")
        or job.get("name")
        or f"finetune_{tray_key.replace('-', '_')}_s{stage}"
    )
    out_dir = str(job.get("out_dir") or defs.get("finetune_out_dir") or DEFAULT_FINETUNE_OUT_DIR)

    train_config = finetune_train_config_for_stage(job, stage)
    transfer_config = finetune_transfer_config_for_stage(job, stage)
    tide_summary = str(
        job.get("tide_summary") or defs.get("tide_summary") or DEFAULT_TIDE_SUMMARY_PATH
    )

    argv: list[str] = [
        "--tray-eval",
        "--tray-key",
        tray_key,
        "--stage",
        str(stage),
        "--config",
        train_config,
        "--transfer-config",
        transfer_config,
        "--train-mode",
        str(job.get("train_mode") or "tray_adapt"),
        "--base-weights",
        base_weights,
        "--out",
        out,
        "--name",
        run_name,
        "--out-dir",
        out_dir,
        "--locked-conf-from",
        locked,
        "--tide-summary",
        tide_summary,
        "--hsp-counting",
    ]

    weak_p = resolve_repo_path(repo_root, weak_plan)
    use_weak = bool(job.get("from_weak_plan")) or weak_tray_plan_actionable(weak_p)
    if use_weak:
        argv.append("--from-weak-plan")
        argv.extend(["--weak-plan", weak_plan])
    if bool(job.get("audit_trays")):
        argv.append("--audit-trays")
    if bool(job.get("debug") or defs.get("finetune_debug")):
        argv.append("--debug")
    if dry_run:
        argv.append("--dry-run")
    if job.get("dataset_root"):
        argv.extend(["--dataset-root", str(job["dataset_root"])])
    if job.get("dataset_name"):
        argv.extend(["--dataset-name", str(job["dataset_name"])])
    if job.get("manifest"):
        argv.extend(["--manifest", str(job["manifest"])])
    return argv


def apply_debug_transfer_overrides(transfer: dict[str, Any]) -> dict[str, Any]:
    """Short train for pipeline debugging (2 epochs, low patience)."""
    out = dict(transfer)
    out["epochs"] = 2
    out["patience"] = 2
    out["debug"] = True
    return out
