"""Aggregate reviewer-2 mAP@0.5 claims from on-disk HSP artifacts (no new reports/)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harchoc.dual_metric_report import extract_detection_metrics
from harchoc.json_io import load_json_dict
from harchoc.schemas import with_schema_version

SCHEMA_VERSION = "reviewer2_map50_computed.v1"
MAP50_RTOL = 1e-6

# Explicit manuscript / audit literals (see reports/reviewer2_audit_map50.md).
MANUSCRIPT_DOCX_TEST_MAP50 = 0.793
UNIT_FIXTURE_TEST_MAP50 = 0.793
PEAK_VAL_TRAINING_MAP50 = 0.97
LEGACY_INTERNAL_TEST_MAP50 = 0.79
LOCKED_CONF_DISPLAY = 0.15


def _round3(v: float | None) -> float | None:
    if v is None:
        return None
    return round(float(v), 3)


def _row(
    *,
    source: str,
    split: str,
    map50: float | None,
    label: str,
    notes: str = "",
    path: str | None = None,
) -> dict[str, Any]:
    return {
        "source": source,
        "split": split,
        "mAP50": map50,
        "mAP50_display": (f"{_round3(map50):.3f}" if map50 is not None else None),
        "label": label,
        "notes": notes,
        "path": path,
    }


def _test_row_from_dual_metric(dm: dict[str, Any]) -> dict[str, Any] | None:
    rows = dm.get("rows")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and str(row.get("split", "")).strip().lower() == "test":
            return row
    return None


def _counting_from_dual_metric_row(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    block = row.get("counting")
    return block if isinstance(block, dict) else {}


def _val_row_from_dual_metric(dm: dict[str, Any]) -> dict[str, Any] | None:
    rows = dm.get("rows")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and str(row.get("split", "")).strip().lower() == "val":
            return row
    return None


def build_reviewer2_map50_computed(
    *,
    repo_root: Path,
    eval_test_map: str = "reports/hsp/eval_test_map.json",
    dual_metric: str = "reports/hsp/dual_metric.json",
    eval_val: str = "reports/hsp/eval_val.json",
    eval_test: str = "reports/hsp/eval_test.json",
    rerun_eval_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build reviewer2_map50_computed.v1 from canonical HSP JSON.

    ``rerun_eval_doc``: optional fresh eval_run.v1 from ``scripts/eval.py`` (compare only).
    """
    root = repo_root.resolve()
    etm_path = (root / eval_test_map).resolve()
    dm_path = (root / dual_metric).resolve()
    ev_path = (root / eval_val).resolve()
    et_path = (root / eval_test).resolve()

    missing: list[str] = []
    for p, label in (
        (etm_path, eval_test_map),
        (dm_path, dual_metric),
        (ev_path, eval_val),
        (et_path, eval_test),
    ):
        if not p.is_file():
            missing.append(label)

    etm = load_json_dict(etm_path) if etm_path.is_file() else {}
    dm = load_json_dict(dm_path) if dm_path.is_file() else {}
    ev = load_json_dict(ev_path) if ev_path.is_file() else {}
    et = load_json_dict(et_path) if et_path.is_file() else {}

    hsp_map50 = etm.get("mAP50")
    hsp_map5095 = etm.get("mAP50_95")
    test_row = _test_row_from_dual_metric(dm)
    val_row = _val_row_from_dual_metric(dm)
    dm_test_det = (
        extract_detection_metrics(test_row["detection"])
        if test_row and isinstance(test_row.get("detection"), dict)
        else {}
    )
    dm_test_count = _counting_from_dual_metric_row(test_row)
    dm_val_count = _counting_from_dual_metric_row(val_row)

    dm_test_map50 = dm_test_det.get("mAP50")
    cross: dict[str, Any] = {
        "tolerance_rtol": MAP50_RTOL,
        "dual_metric_test_map50_matches_eval_test_map": None,
        "rerun_eval_map50_matches_eval_test_map": None,
    }
    if hsp_map50 is not None and dm_test_map50 is not None:
        cross["dual_metric_test_map50_matches_eval_test_map"] = abs(
            float(hsp_map50) - float(dm_test_map50)
        ) <= MAP50_RTOL
    if rerun_eval_doc is not None:
        rerun_m50 = rerun_eval_doc.get("mAP50")
        if hsp_map50 is not None and rerun_m50 is not None:
            cross["rerun_eval_map50_matches_eval_test_map"] = abs(
                float(hsp_map50) - float(rerun_m50)
            ) <= MAP50_RTOL
            cross["rerun_eval_mAP50"] = float(rerun_m50)
            cross["rerun_eval_mAP50_display"] = f"{_round3(float(rerun_m50)):.3f}"

    export_only_null = {
        "eval_val_export_only": bool(ev.get("export_only")) if ev else None,
        "eval_val_mAP50_null": ev.get("mAP50") is None if ev else None,
        "eval_test_export_only": bool(et.get("export_only")) if et else None,
        "eval_test_mAP50_null": et.get("mAP50") is None if et else None,
    }

    evidence = [
        _row(
            source="reports/plants-4336582.docx",
            split="test",
            map50=MANUSCRIPT_DOCX_TEST_MAP50,
            label="NARRATIVE_ONLY",
            notes="Submitted manuscript; no matching eval JSON in repo",
        ),
        _row(
            source=eval_test_map,
            split="test",
            map50=float(hsp_map50) if hsp_map50 is not None else None,
            label="VERIFIED" if hsp_map50 is not None else "MISSING",
            notes="HSP eval.py → Ultralytics val(split=val) on generated eval_data.yaml",
            path=str(etm_path.relative_to(root)) if etm_path.is_file() else eval_test_map,
        ),
        _row(
            source=f"{dual_metric} → test.detection",
            split="test",
            map50=float(dm_test_map50) if dm_test_map50 is not None else None,
            label="VERIFIED" if dm_test_map50 is not None else "MISSING",
            notes="Merged from eval_test_map",
            path=str(dm_path.relative_to(root)) if dm_path.is_file() else dual_metric,
        ),
        _row(
            source="tests/test_dual_metric_report.py",
            split="test",
            map50=UNIT_FIXTURE_TEST_MAP50,
            label="NARRATIVE_ONLY",
            notes="Synthetic fixture; not production metric",
        ),
        _row(
            source="training logs (Ultralytics)",
            split="val",
            map50=PEAK_VAL_TRAINING_MAP50,
            label="NARRATIVE_ONLY",
            notes="Early-stop peak val mAP; not in HSP JSON",
        ),
        _row(
            source="legacy internal docs (~0.79)",
            split="test",
            map50=LEGACY_INTERNAL_TEST_MAP50,
            label="STALE",
            notes="training_tech_scan etc.; no reports/**/*.json at 0.79 for best2 test",
        ),
    ]

    status = "ok"
    if missing:
        status = "incomplete"
    if cross.get("dual_metric_test_map50_matches_eval_test_map") is False:
        status = "mismatch"

    payload: dict[str, Any] = {
        "status": status,
        "missing_inputs": missing,
        "hsp_canonical": {
            "path": eval_test_map,
            "mAP50": float(hsp_map50) if hsp_map50 is not None else None,
            "mAP50_display": (
                f"{_round3(float(hsp_map50)):.3f}" if hsp_map50 is not None else None
            ),
            "mAP50_95": float(hsp_map5095) if hsp_map5095 is not None else None,
            "export_only": etm.get("export_only"),
            "imgsz": etm.get("imgsz"),
            "max_det": etm.get("max_det"),
            "device": etm.get("device"),
            "weights": etm.get("weights"),
            "split_source": (
                (etm.get("split_source") or {}).get("path")
                if isinstance(etm.get("split_source"), dict)
                else etm.get("split_source")
            ),
            "generated_data_yaml": etm.get("generated_data_yaml"),
        },
        "evidence_table": evidence,
        "counting_at_locked_conf": {
            "locked_conf": LOCKED_CONF_DISPLAY,
            "test_mae": dm_test_count.get("mae"),
            "test_mae_display": (
                round(float(dm_test_count["mae"]), 1)
                if dm_test_count.get("mae") is not None
                else None
            ),
            "test_rrmse": dm_test_count.get("rrmse"),
            "test_rrmse_display": (
                round(float(dm_test_count["rrmse"]), 3)
                if dm_test_count.get("rrmse") is not None
                else None
            ),
            "val_mae": dm_val_count.get("mae"),
            "val_mae_display": (
                round(float(dm_val_count["mae"]), 1)
                if dm_val_count.get("mae") is not None
                else None
            ),
            "n_images_test": dm_test_count.get("n_images"),
        },
        "val_test_gap_narrative": {
            "peak_val_training_mAP50": PEAK_VAL_TRAINING_MAP50,
            "held_out_test_ranking_mAP50_hsp": _round3(
                float(hsp_map50) if hsp_map50 is not None else None
            ),
            "manuscript_docx_test_mAP50": MANUSCRIPT_DOCX_TEST_MAP50,
            "legacy_internal_test_mAP50": LEGACY_INTERNAL_TEST_MAP50,
            "summary": (
                "Peak val mAP (~0.97) is training early-stop only; "
                "canonical HSP test ranking mAP is ~0.18; docx 0.793 has no JSON; "
                "count MAE ~61.3 @ conf 0.15 is orthogonal."
            ),
        },
        "cross_checks": {**cross, **export_only_null},
        "inputs": {
            "eval_test_map": eval_test_map,
            "dual_metric": dual_metric,
            "eval_val": eval_val,
            "eval_test": eval_test,
        },
    }
    return with_schema_version(payload, schema_version=SCHEMA_VERSION)


def reviewer2_map50_reproduce_commands(*, rerun_eval: bool = False) -> dict[str, str]:
    agg = "mamba run -n harchoc python scripts/experiment.py reviewer2-map50"
    full = (
        "mamba run -n harchoc python scripts/experiment.py reviewer2-map50 --rerun-eval"
    )
    map_cpu = (
        "mamba run -n harchoc python scripts/experiment.py map-cpu "
        "--device cpu --out reports/hsp/eval_test_map.json"
    )
    return {
        "aggregate_from_artifacts": agg,
        "aggregate_with_cpu_rerun": full,
        "map_cpu_only": map_cpu,
    }
