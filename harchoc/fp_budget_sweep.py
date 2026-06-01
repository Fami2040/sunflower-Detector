"""Constraint-sweep ablation for count-first val selection / FP budget (P1-FP-BUDGET)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harchoc.detection_match import image_ids_union, match_counts_for_threshold
from harchoc.schemas import with_schema_version
from harchoc.threshold_protocol import (
    counting_metrics_at_conf,
    select_min_count_mae,
    select_operating_point,
)

FP_BUDGET_SWEEP_SCHEMA = "fp_budget_sweep.v1"

# Dense-tray HSP val: ~217 FP/img @ F1-max locked conf; grid spans tighter budgets.
DEFAULT_FP_BUDGET_GRID: tuple[float, ...] = (25.0, 50.0, 100.0, 150.0, 200.0, 217.0, 250.0, 300.0)


def _metrics(tp: int, fp: int, fn: int, *, n_images: int, conf_thr: float) -> dict[str, Any]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    fp_per_image = (fp / n_images) if n_images else 0.0
    return {
        "conf_thr": float(conf_thr),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fp_per_image": fp_per_image,
    }


def build_sweep_rows(
    *,
    gt: Any,
    preds: Any,
    thresholds: list[float],
    iou_thr: float,
    category_aware: bool,
    n_images: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for thr in thresholds:
        counts = match_counts_for_threshold(
            gt=gt,
            preds=preds,
            conf_thr=thr,
            iou_thr=float(iou_thr),
            category_aware=category_aware,
        )
        rows.append(_metrics(counts["tp"], counts["fp"], counts["fn"], n_images=n_images, conf_thr=thr))
    return rows


def _enrich_row(
    row: dict[str, Any] | None,
    *,
    gt: Any,
    preds: Any,
    iou_thr: float,
    category_aware: bool,
) -> dict[str, Any] | None:
    if row is None:
        return None
    conf = float(row.get("conf_thr", 0.0))
    counting = counting_metrics_at_conf(
        gt=gt,
        preds=preds,
        conf_thr=conf,
        iou_thr=float(iou_thr),
        category_aware=category_aware,
    )
    out = dict(row)
    out["count_mae"] = float(counting.get("mae", 0.0))
    out["counting_metrics"] = counting
    return out


def _selection_entry(
    *,
    mode: str,
    row: dict[str, Any] | None,
    constraints: dict[str, Any] | None = None,
    feasible: bool | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {"mode": mode, "selected": row}
    if constraints is not None:
        entry["constraints"] = constraints
    if feasible is not None:
        entry["feasible"] = feasible
    return entry


def build_fp_budget_sweep_payload(
    *,
    gt: Any,
    preds: Any,
    rows: list[dict[str, Any]],
    iou_thr: float,
    category_aware: bool,
    n_images: int,
    fp_budget_grid: list[float] | None = None,
    sweep_val_path: str | None = None,
    gt_json: str | None = None,
    preds_json: str | None = None,
    split_role: str = "val",
    split_hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare count-first, F1-max, and max_fp_per_image constraint picks on val sweep rows."""
    grid = list(fp_budget_grid if fp_budget_grid is not None else DEFAULT_FP_BUDGET_GRID)

    min_count_row = select_min_count_mae(
        rows,
        gt=gt,
        preds=preds,
        iou_thr=float(iou_thr),
        category_aware=category_aware,
    )
    best_f1_row = select_operating_point(rows, mode="best_f1")

    selection_comparison = [
        _selection_entry(
            mode="min_count_mae",
            row=_enrich_row(min_count_row, gt=gt, preds=preds, iou_thr=iou_thr, category_aware=category_aware),
        ),
        _selection_entry(
            mode="best_f1",
            row=_enrich_row(best_f1_row, gt=gt, preds=preds, iou_thr=iou_thr, category_aware=category_aware),
        ),
    ]

    fp_budget_entries: list[dict[str, Any]] = []
    for cap in grid:
        sel = select_operating_point(rows, mode="constraints", max_fp_per_image=float(cap))
        fp_budget_entries.append(
            {
                "max_fp_per_image": float(cap),
                "feasible": sel is not None,
                "selected": _enrich_row(sel, gt=gt, preds=preds, iou_thr=iou_thr, category_aware=category_aware),
            }
        )

    reference: dict[str, Any] | None = None
    if best_f1_row is not None:
        reference = {
            "mode": "best_f1",
            "conf_thr": float(best_f1_row.get("conf_thr", 0.0)),
            "fp_per_image": float(best_f1_row.get("fp_per_image", 0.0)),
        }
        if sweep_val_path:
            reference["source"] = sweep_val_path

    payload: dict[str, Any] = {
        "status": "ok",
        "inputs": {
            "gt_json": gt_json,
            "preds_json": preds_json,
            "sweep_val": sweep_val_path,
        },
        "eval_target": {
            "split_role": split_role,
            "hints": split_hints or {},
        },
        "match": {"iou": float(iou_thr), "category_aware": category_aware},
        "images": {"n": n_images},
        "n_sweep_rows": len(rows),
        "reference": reference,
        "selection_comparison": selection_comparison,
        "fp_budget_grid": fp_budget_entries,
    }
    return with_schema_version(payload, schema_version=FP_BUDGET_SWEEP_SCHEMA)


def build_dry_run_fp_budget_sweep(*, out: str, fp_budget_grid: list[float] | None = None) -> dict[str, Any]:
    grid = list(fp_budget_grid if fp_budget_grid is not None else DEFAULT_FP_BUDGET_GRID)
    return with_schema_version(
        {
            "status": "dry-run",
            "out": out,
            "fp_budget_grid_values": grid,
            "selection_modes": ["min_count_mae", "best_f1", "constraints"],
        },
        schema_version=FP_BUDGET_SWEEP_SCHEMA,
    )


def load_sweep_rows_and_match(sweep_path: str | Path) -> tuple[list[dict[str, Any]], float, bool, dict[str, Any]]:
    """Read sweep rows and match settings from threshold_sweep_run.v1 JSON."""
    obj = json.loads(Path(sweep_path).expanduser().read_text("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {sweep_path}")
    raw_rows = obj.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"No rows in sweep JSON: {sweep_path}")
    match = obj.get("match")
    if not isinstance(match, dict):
        raise ValueError(f"No match block in sweep JSON: {sweep_path}")
    iou = float(match.get("iou", 0.5))
    category_aware = bool(match.get("category_aware", True))
    return raw_rows, iou, category_aware, obj


def write_fp_budget_sweep(path: str | Path, payload: dict[str, Any]) -> Path:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return out


def _row_metric(row: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not row:
        return default
    val = row.get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _fmt_conf(conf: float) -> str:
    return f"{conf:.2f}"


def _fmt_num(val: float, *, digits: int = 1) -> str:
    return f"{val:.{digits}f}"


def load_locked_operating_point(locked_path: str | Path) -> dict[str, Any] | None:
    """Read locked row + counting metrics from threshold_sweep_run.v1 (test lock artifact)."""
    obj = json.loads(Path(locked_path).expanduser().read_text("utf-8"))
    if not isinstance(obj, dict):
        return None
    locked = obj.get("locked")
    if not isinstance(locked, dict):
        return None
    row = locked.get("row")
    if not isinstance(row, dict):
        return None
    out = dict(row)
    counting = locked.get("counting_metrics")
    if isinstance(counting, dict):
        out["count_mae"] = float(counting.get("mae", 0.0))
        out["counting_metrics"] = counting
    source = locked.get("source")
    if source:
        out["locked_source"] = str(source)
    return out


def format_fp_budget_manuscript_md(
    payload: dict[str, Any],
    *,
    locked_row: dict[str, Any] | None = None,
    title: str | None = None,
) -> str:
    """Manuscript-ready markdown: selection modes, constraint grid, pick recommendation."""
    from harchoc.config_coerce import child_dict

    eval_target = child_dict(payload, "eval_target")
    split_role = str(eval_target.get("split_role") or "val")
    n_images = int(child_dict(payload, "images").get("n") or 0)
    match = child_dict(payload, "match")
    iou = _row_metric(match, "iou", 0.5)
    inputs = child_dict(payload, "inputs")
    sweep_src = inputs.get("sweep_val") or inputs.get("sweep_from") or "—"

    heading = title or f"FP budget constraint sweep ({split_role})"
    lines: list[str] = [
        f"# {heading}",
        "",
        f"**Split:** {split_role} (n={n_images}) · **Match:** IoU {_fmt_num(iou, digits=1)}, category-aware",
        f"**Sweep rows:** `{sweep_src}` ({payload.get('n_sweep_rows', '—')} conf steps)",
        "",
    ]

    lines.extend(["## Selection comparison", "", "| Mode | conf | FP/img | Count MAE | F1 |", "|------|------|--------|-----------|-----|"])
    for entry in payload.get("selection_comparison") or []:
        if not isinstance(entry, dict):
            continue
        mode = str(entry.get("mode") or "")
        sel = entry.get("selected") if isinstance(entry.get("selected"), dict) else None
        if sel is None:
            lines.append(f"| {mode} | — | — | — | — |")
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    mode,
                    _fmt_conf(_row_metric(sel, "conf_thr")),
                    _fmt_num(_row_metric(sel, "fp_per_image")),
                    _fmt_num(_row_metric(sel, "count_mae")),
                    _fmt_num(_row_metric(sel, "f1"), digits=3),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Constraint grid (max FP/image cap)", "", "| Cap | conf | FP/img | Count MAE | F1 |", "|-----|------|--------|-----------|-----|"])
    for entry in payload.get("fp_budget_grid") or []:
        if not isinstance(entry, dict):
            continue
        cap = _row_metric(entry, "max_fp_per_image")
        if not entry.get("feasible"):
            lines.append(f"| {cap:.0f} | — | — | — | infeasible |")
            continue
        sel = entry.get("selected") if isinstance(entry.get("selected"), dict) else None
        if sel is None:
            lines.append(f"| {cap:.0f} | — | — | — | — |")
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{cap:.0f}",
                    _fmt_conf(_row_metric(sel, "conf_thr")),
                    _fmt_num(_row_metric(sel, "fp_per_image")),
                    _fmt_num(_row_metric(sel, "count_mae")),
                    _fmt_num(_row_metric(sel, "f1"), digits=3),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Manuscript pick"])
    if locked_row:
        locked_conf = _row_metric(locked_row, "conf_thr")
        locked_mae = _row_metric(locked_row, "count_mae")
        locked_fp = _row_metric(locked_row, "fp_per_image")
        locked_f1 = _row_metric(locked_row, "f1")
        lines.append(
            f"**Primary (locked):** val-selected conf **{_fmt_conf(locked_conf)}** applied unchanged on "
            f"{split_role} — count MAE **{_fmt_num(locked_mae)}**, FP/img **{_fmt_num(locked_fp)}**, "
            f"F1 **{_fmt_num(locked_f1, digits=3)}**. Threshold chosen on val (`min_count_mae`); "
            f"test reports the same conf only (no re-selection on test)."
        )
    else:
        min_row = None
        for entry in payload.get("selection_comparison") or []:
            if isinstance(entry, dict) and entry.get("mode") == "min_count_mae":
                sel = entry.get("selected")
                if isinstance(sel, dict):
                    min_row = sel
                    break
        if min_row:
            lines.append(
                f"**Count-first on {split_role}:** conf **{_fmt_conf(_row_metric(min_row, 'conf_thr'))}**, "
                f"MAE **{_fmt_num(_row_metric(min_row, 'count_mae'))}**, "
                f"FP/img **{_fmt_num(_row_metric(min_row, 'fp_per_image'))}**."
            )

    best_f1_row = None
    for entry in payload.get("selection_comparison") or []:
        if isinstance(entry, dict) and entry.get("mode") == "best_f1":
            sel = entry.get("selected")
            if isinstance(sel, dict):
                best_f1_row = sel
                break
    if locked_row and best_f1_row:
        mae_delta = _row_metric(best_f1_row, "count_mae") - _row_metric(locked_row, "count_mae")
        if mae_delta > 0.5:
            lines.append(
                f"F1-max on {split_role} would raise count MAE by **+{_fmt_num(mae_delta)}** "
                f"(conf {_fmt_conf(_row_metric(best_f1_row, 'conf_thr'))}) vs locked point — "
                "supports count-first selection over detection F1 alone."
            )

    tight_caps: list[tuple[float, dict[str, Any]]] = []
    for entry in payload.get("fp_budget_grid") or []:
        if not isinstance(entry, dict) or not entry.get("feasible"):
            continue
        cap = _row_metric(entry, "max_fp_per_image")
        sel = entry.get("selected")
        if isinstance(sel, dict) and cap <= 200.0:
            tight_caps.append((cap, sel))
    if tight_caps and locked_row:
        worst = max(tight_caps, key=lambda x: _row_metric(x[1], "count_mae"))
        cap, sel = worst
        mae_penalty = _row_metric(sel, "count_mae") - _row_metric(locked_row, "count_mae")
        if mae_penalty > 1.0:
            lines.append(
                f"Tighter FP cap **{cap:.0f}/img** (conf {_fmt_conf(_row_metric(sel, 'conf_thr'))}) "
                f"raises count MAE by **+{_fmt_num(mae_penalty)}** vs locked — dense-tray counting "
                "favors the val-locked operating point over strict FP budgets."
            )

    lines.append("")
    return "\n".join(lines)


def write_fp_budget_manuscript_summary(
    path: str | Path,
    payload: dict[str, Any],
    *,
    locked_row: dict[str, Any] | None = None,
    title: str | None = None,
) -> Path:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        format_fp_budget_manuscript_md(payload, locked_row=locked_row, title=title),
        encoding="utf-8",
    )
    return out


def run_fp_budget_sweep(
    *,
    out: str,
    gt_json: str,
    preds_json: str,
    gt: Any,
    preds: Any,
    sweep_from: str | None = None,
    fp_budget_grid: list[float] | None = None,
    iou_thr: float | None = None,
    category_aware: bool | None = None,
    split_role: str = "val",
    split_hints: dict[str, Any] | None = None,
    locked_conf_from: str | None = None,
    summary_out: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build constraint-sweep ablation on val/test exports (CPU-only; no inference)."""
    if dry_run:
        return build_dry_run_fp_budget_sweep(out=out, fp_budget_grid=fp_budget_grid)

    sweep_val_path = (sweep_from or "").strip() or None
    rows: list[dict[str, Any]]
    match_iou = float(iou_thr if iou_thr is not None else 0.5)
    cat_aware = bool(category_aware if category_aware is not None else True)

    if sweep_val_path:
        rows, match_iou, cat_aware, _ = load_sweep_rows_and_match(sweep_val_path)

    image_ids = image_ids_union(gt, preds)
    n_images = len(image_ids)

    if not sweep_val_path:
        thresholds = [max(0.0, min(1.0, x)) for x in _default_thresholds()]
        rows = build_sweep_rows(
            gt=gt,
            preds=preds,
            thresholds=thresholds,
            iou_thr=match_iou,
            category_aware=cat_aware,
            n_images=n_images,
        )

    payload = build_fp_budget_sweep_payload(
        gt=gt,
        preds=preds,
        rows=rows,
        iou_thr=match_iou,
        category_aware=cat_aware,
        n_images=n_images,
        fp_budget_grid=fp_budget_grid,
        sweep_val_path=sweep_val_path,
        gt_json=gt_json or None,
        preds_json=preds_json or None,
        split_role=split_role,
        split_hints=split_hints,
    )
    write_fp_budget_sweep(out, payload)
    summary_path = (summary_out or "").strip() or None
    if summary_path:
        locked_row = load_locked_operating_point(locked_conf_from) if locked_conf_from else None
        write_fp_budget_manuscript_summary(summary_path, payload, locked_row=locked_row)
    return payload


def _default_thresholds(*, tmin: float = 0.05, tmax: float = 0.95, steps: int = 19) -> list[float]:
    if steps <= 1:
        return [float(tmin)]
    if tmax < tmin:
        tmin, tmax = tmax, tmin
    return [tmin + (tmax - tmin) * (i / (steps - 1)) for i in range(steps)]
