"""Merge eval, threshold sweep, and error-analysis JSON into a manuscript table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harchoc.json_io import load_json_dict
from harchoc.schemas import with_schema_version
from harchoc.strict_ml import strict_ml_enabled
from harchoc.threshold_lock import load_locked_conf

# Manuscript / reviewer: val metrics are for checkpoint selection, not generalization.
# Val≈0.97 vs test≈0.79 mAP gap narrative: docs/manuscript/val_test_map_gap.md + reports/hsp/split_drift_p0.json
VAL_SPLIT_METRIC_LABEL = "in-training early-stop split (not generalization)"
TEST_SPLIT_METRIC_LABEL = "held-out manuscript split (generalization)"


def split_metric_role_label(split: str) -> str:
    """Human-readable split role for tables and dual_metric_report.v1 rows."""
    role = split.strip().lower()
    if role == "val":
        return VAL_SPLIT_METRIC_LABEL
    if role == "test":
        return TEST_SPLIT_METRIC_LABEL
    return role


def _split_role_from_eval(doc: dict[str, Any]) -> str | None:
    target = doc.get("eval_target")
    if isinstance(target, dict):
        role = target.get("split_role")
        if isinstance(role, str) and role.strip():
            return role.strip().lower()
    return None


def extract_detection_metrics(eval_doc: dict[str, Any]) -> dict[str, Any]:
    """mAP50 / mAP50-95 from eval_run.v1 (or compatible)."""
    out: dict[str, Any] = {}
    for key in ("mAP50", "mAP50_95"):
        v = eval_doc.get(key)
        if v is not None:
            out[key] = float(v)
    return out


def _map_overlay_path(eval_path: str | Path) -> Path | None:
    """Sibling ``eval_val.json`` → ``eval_val_map.json`` when present."""
    p = Path(eval_path).expanduser()
    candidate = p.with_name(f"{p.stem}_map{p.suffix}")
    return candidate if candidate.is_file() else None


def resolve_detection_metrics(
    eval_doc: dict[str, Any],
    *,
    eval_path: str | None = None,
    map_path: str | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Primary eval doc; optional map overlay when mAP is null (export-only)."""
    det = extract_detection_metrics(eval_doc)
    if det:
        return det, None

    resolved_map = (str(map_path).strip() or None) if map_path else None
    if resolved_map is None and eval_path:
        auto = _map_overlay_path(eval_path)
        if auto is not None:
            resolved_map = str(auto)

    if not resolved_map:
        return {}, None

    overlay = extract_detection_metrics(load_json_dict(resolved_map))
    if not overlay:
        return {}, None
    return overlay, resolved_map


def _compact_counting_block(cm: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("mae", "rmse", "rrmse", "n_images"):
        if key in cm and cm[key] is not None:
            out[key] = cm[key] if key == "n_images" else float(cm[key])
    if isinstance(cm.get("mae_ci"), dict):
        out["mae_ci"] = cm["mae_ci"]
    return out


def extract_counting_metrics(error_doc: dict[str, Any]) -> dict[str, Any]:
    """Counting block from error_analysis_summary.v1."""
    cm = error_doc.get("counting_metrics")
    if not isinstance(cm, dict):
        return {}
    return _compact_counting_block(cm)


def extract_locked_counting_metrics(sweep_doc: dict[str, Any] | None) -> dict[str, Any]:
    """Counting block from threshold_sweep_run.v1 ``locked.counting_metrics``."""
    if not isinstance(sweep_doc, dict):
        return {}
    locked = sweep_doc.get("locked")
    if not isinstance(locked, dict):
        return {}
    cm = locked.get("counting_metrics")
    if not isinstance(cm, dict):
        return {}
    return _compact_counting_block(cm)


def resolve_counting_metrics(
    *,
    error_doc: dict[str, Any],
    sweep_doc: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Prefer ``locked.counting_metrics`` on sweep JSON when both sources exist."""
    locked = extract_locked_counting_metrics(sweep_doc)
    error = extract_counting_metrics(error_doc)
    if locked:
        return locked, "locked"
    if error:
        return error, "error_analysis"
    return {}, None


def _row_from_sweep_block(block: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(block, dict):
        return None
    row = block.get("row")
    if not isinstance(row, dict):
        return None
    keys = (
        "conf_thr",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "fp_per_image",
    )
    compact = {k: row[k] for k in keys if k in row}
    return compact or None


def extract_operating_point(
    *,
    sweep_val: dict[str, Any],
    sweep_test: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    sweep_val_path: str | None = None,
) -> dict[str, Any]:
    """Selected conf on val; locked conf/row on test when available."""
    selected_block = sweep_val.get("selected")
    selected_row = _row_from_sweep_block(selected_block if isinstance(selected_block, dict) else None)
    selected_conf: float | None = None
    if selected_row and selected_row.get("conf_thr") is not None:
        selected_conf = float(selected_row["conf_thr"])

    locked_conf: float | None = None
    locked_source: str | None = None
    locked_row: dict[str, Any] | None = None

    if sweep_test is not None:
        locked_block = sweep_test.get("locked")
        if isinstance(locked_block, dict):
            locked_source = locked_block.get("source")
            if isinstance(locked_source, str):
                locked_source = locked_source.strip() or None
            locked_row = _row_from_sweep_block(locked_block)
            if locked_row and locked_row.get("conf_thr") is not None:
                locked_conf = float(locked_row["conf_thr"])

    if locked_conf is None and selected_conf is not None:
        locked_conf = selected_conf

    if locked_conf is None and sweep_val_path:
        try:
            locked_conf = load_locked_conf(sweep_val_path)
        except ValueError as exc:
            if warnings is not None:
                warnings.append(str(exc))

    out: dict[str, Any] = {
        "selected_conf": selected_conf,
        "locked_conf": locked_conf,
        "locked_source": locked_source,
    }
    if selected_row is not None:
        out["val_selected_row"] = selected_row
    if locked_row is not None:
        out["test_locked_row"] = locked_row
    return out


def _detection_empty_warnings(
    rows: list[dict[str, Any]],
    *,
    inputs: dict[str, str],
) -> list[str]:
    """Warn when eval paths were merged but a split has no mAP in its detection row."""
    warnings: list[str] = []
    for row in rows:
        split = row.get("split")
        if not isinstance(split, str):
            continue
        det = row.get("detection")
        if isinstance(det, dict) and det:
            continue
        eval_key = f"eval_{split}"
        if eval_key not in inputs and not any(
            k.startswith("eval_") and split in k for k in inputs
        ):
            continue
        warnings.append(
            f"{split}: eval JSON provided but detection metrics are empty "
            f"(run mAP eval or pass --eval-{split}-map / sibling eval_*_map.json)"
        )
    return warnings


def _finalize_report_status(
    payload: dict[str, Any], *, warnings: list[str]
) -> None:
    """Always attach warnings; downgrade status when detection rows are incomplete."""
    payload["warnings"] = list(warnings)
    if not warnings:
        return
    payload["status"] = "partial" if strict_ml_enabled() else payload.get("status", "ok")


def build_table_row(
    *,
    split: str,
    detection: dict[str, Any],
    counting: dict[str, Any],
    operating_conf: float | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "split": split,
        "split_role_label": split_metric_role_label(split),
        "detection": detection,
        "counting": counting,
    }
    if operating_conf is not None:
        row["operating_conf"] = float(operating_conf)
    return row


def build_dual_metric_report(
    *,
    eval_val: dict[str, Any],
    eval_test: dict[str, Any],
    sweep_val: dict[str, Any],
    error_val: dict[str, Any],
    error_test: dict[str, Any],
    sweep_test: dict[str, Any] | None = None,
    inputs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble dual_metric_report.v1 from upstream artifact dicts."""
    input_paths = dict(inputs or {})
    warnings: list[str] = []
    operating = extract_operating_point(
        sweep_val=sweep_val,
        sweep_test=sweep_test,
        warnings=warnings,
        sweep_val_path=input_paths.get("sweep_val"),
    )
    selected_conf = operating.get("selected_conf")
    locked_conf = operating.get("locked_conf")

    val_counting, val_counting_source = resolve_counting_metrics(error_doc=error_val)
    test_counting, test_counting_source = resolve_counting_metrics(
        error_doc=error_test, sweep_doc=sweep_test
    )

    rows = [
        build_table_row(
            split="val",
            detection=extract_detection_metrics(eval_val),
            counting=val_counting,
            operating_conf=float(selected_conf) if selected_conf is not None else None,
        ),
        build_table_row(
            split="test",
            detection=extract_detection_metrics(eval_test),
            counting=test_counting,
            operating_conf=float(locked_conf) if locked_conf is not None else None,
        ),
    ]
    counting_sources: dict[str, str | None] = {
        "val": val_counting_source,
        "test": test_counting_source,
    }

    warnings.extend(_detection_empty_warnings(rows, inputs=input_paths))
    payload: dict[str, Any] = {
        "status": "ok",
        "inputs": input_paths,
        "metric_roles": {
            "val": VAL_SPLIT_METRIC_LABEL,
            "test": TEST_SPLIT_METRIC_LABEL,
        },
        "operating_point": operating,
        "counting_sources": counting_sources,
        "rows": rows,
    }
    _finalize_report_status(payload, warnings=warnings)
    return with_schema_version(payload, schema_version="dual_metric_report.v1")


def build_dry_run_report(*, out: str, inputs: dict[str, str]) -> dict[str, Any]:
    placeholder_row = {
        "split": "val",
        "split_role_label": VAL_SPLIT_METRIC_LABEL,
        "detection": {"mAP50": None, "mAP50_95": None},
        "counting": {"mae": None, "mae_ci": None},
        "operating_conf": None,
    }
    payload: dict[str, Any] = {
        "status": "dry-run",
        "script": "dual_metric",
        "out": str(Path(out)),
        "inputs": inputs,
        "warnings": [],
        "operating_point": {
            "selected_conf": None,
            "locked_conf": None,
            "locked_source": None,
        },
        "metric_roles": {
            "val": VAL_SPLIT_METRIC_LABEL,
            "test": TEST_SPLIT_METRIC_LABEL,
        },
        "rows": [
            placeholder_row,
            {
                **placeholder_row,
                "split": "test",
                "split_role_label": TEST_SPLIT_METRIC_LABEL,
            },
        ],
    }
    return with_schema_version(payload, schema_version="dual_metric_report.v1")


def resolve_eval_by_split(
    *,
    eval_val_path: str | None,
    eval_test_path: str | None,
    eval_paths: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Resolve val/test eval docs from explicit paths or --eval list."""
    paths: dict[str, str] = {}
    if eval_val_path:
        paths["eval_val"] = eval_val_path
    if eval_test_path:
        paths["eval_test"] = eval_test_path

    by_role: dict[str, dict[str, Any]] = {}
    for p in eval_paths:
        doc = load_json_dict(p)
        role = _split_role_from_eval(doc)
        if role not in {"val", "test"}:
            raise ValueError(
                f"eval JSON must declare eval_target.split_role val|test: {p} (got {role!r})"
            )
        if role in by_role:
            raise ValueError(f"Duplicate eval for split {role!r}: {p}")
        by_role[role] = doc
        paths[f"eval_{role}"] = p

    val_doc = by_role.get("val")
    test_doc = by_role.get("test")
    if eval_val_path and val_doc is None:
        val_doc = load_json_dict(eval_val_path)
    if eval_test_path and test_doc is None:
        test_doc = load_json_dict(eval_test_path)

    if val_doc is None or test_doc is None:
        missing = [s for s, d in (("val", val_doc), ("test", test_doc)) if d is None]
        raise ValueError(
            "Need eval JSON for val and test (use --eval-val/--eval-test or two --eval files "
            f"with split_role). Missing: {', '.join(missing)}"
        )
    out_paths = {
        k: v
        for k, v in paths.items()
        if k in {"eval_val", "eval_test"}
    }
    if eval_val_path:
        out_paths["eval_val"] = eval_val_path
    if eval_test_path:
        out_paths["eval_test"] = eval_test_path
    return val_doc, test_doc, out_paths


def _enrich_eval_with_map_overlay(
    eval_doc: dict[str, Any],
    *,
    eval_path: str | None,
    map_path: str | None,
) -> tuple[dict[str, Any], str | None]:
    det, map_used = resolve_detection_metrics(
        eval_doc, eval_path=eval_path, map_path=map_path
    )
    if not map_used:
        return eval_doc, None
    enriched = dict(eval_doc)
    enriched.update(det)
    return enriched, map_used


def merge_dual_metric_from_paths(
    *,
    eval_val: str | None = None,
    eval_test: str | None = None,
    eval_val_map: str | None = None,
    eval_test_map: str | None = None,
    eval_paths: list[str] | None = None,
    sweep_val: str,
    sweep_test: str | None = None,
    error_val: str,
    error_test: str,
) -> dict[str, Any]:
    val_eval, test_eval, eval_input_paths = resolve_eval_by_split(
        eval_val_path=eval_val,
        eval_test_path=eval_test,
        eval_paths=list(eval_paths or []),
    )
    val_path = eval_input_paths.get("eval_val") or eval_val
    test_path = eval_input_paths.get("eval_test") or eval_test

    val_eval, val_map_used = _enrich_eval_with_map_overlay(
        val_eval, eval_path=val_path, map_path=eval_val_map
    )
    test_eval, test_map_used = _enrich_eval_with_map_overlay(
        test_eval, eval_path=test_path, map_path=eval_test_map
    )

    inputs: dict[str, str] = {
        **eval_input_paths,
        "sweep_val": sweep_val,
        "error_val": error_val,
        "error_test": error_test,
    }
    if val_map_used:
        inputs["eval_val_map"] = val_map_used
    if test_map_used:
        inputs["eval_test_map"] = test_map_used
    if sweep_test:
        inputs["sweep_test"] = sweep_test

    return build_dual_metric_report(
        eval_val=val_eval,
        eval_test=test_eval,
        sweep_val=load_json_dict(sweep_val),
        sweep_test=load_json_dict(sweep_test) if sweep_test else None,
        error_val=load_json_dict(error_val),
        error_test=load_json_dict(error_test),
        inputs=inputs,
    )
