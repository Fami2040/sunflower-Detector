"""Val→lock→test guardrails, IoU grid helpers, and counting metrics for threshold sweeps."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Literal

from harchoc.strict_ml import capture_failure, fail_or_warn

SplitRole = Literal["train", "val", "test", "unknown"]
SelectMode = Literal["best_f1", "constraints", "min_count_mae"]


def _best_f1(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for row in rows:
        if best is None:
            best = row
            continue
        if float(row.get("f1", 0.0)) > float(best.get("f1", 0.0)):
            best = row
            continue
        if math.isclose(float(row.get("f1", 0.0)), float(best.get("f1", 0.0))) and float(row.get("conf_thr", 0.0)) < float(
            best.get("conf_thr", 0.0)
        ):
            best = row
    return best


def select_operating_point(
    rows: list[dict[str, Any]],
    *,
    mode: SelectMode = "best_f1",
    min_recall: float | None = None,
    min_precision: float | None = None,
    max_fp_per_image: float | None = None,
) -> dict[str, Any] | None:
    """
    Choose an operating point (a row) from the sweep.

    - mode=best_f1: maximize F1 (tie-break by lower conf_thr)
    - mode=constraints: filter rows by constraints, then maximize F1 (tie-break by lower conf_thr)
    - mode=min_count_mae: use select_min_count_mae (requires gt/preds; not supported here)

    Returns None if mode=constraints and no row satisfies all constraints.
    """
    if mode == "min_count_mae":
        raise ValueError("min_count_mae selection requires gt/preds; use select_min_count_mae()")
    if mode == "best_f1":
        return _best_f1(rows)

    feas: list[dict[str, Any]] = []
    for r in rows:
        if min_recall is not None and float(r.get("recall", 0.0)) < float(min_recall):
            continue
        if min_precision is not None and float(r.get("precision", 0.0)) < float(min_precision):
            continue
        if max_fp_per_image is not None and float(r.get("fp_per_image", float("inf"))) > float(max_fp_per_image):
            continue
        feas.append(r)
    return _best_f1(feas) if feas else None


def _role_from_name(name: str) -> SplitRole | None:
    n = name.lower()
    if n in ("train", "val", "test"):
        return n  # type: ignore[return-value]
    if "test" in n and "latest" not in n:
        return "test"
    if "val" in n or "valid" in n:
        return "val"
    if "train" in n:
        return "train"
    return None


def _split_paths(*, repo_root: Path, dataset_root: Path | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    roots: list[Path] = [repo_root]
    if dataset_root is not None:
        roots.append(dataset_root)
    for root in roots:
        d = root / "data" / "splits"
        if not d.is_dir():
            continue
        for role in ("train", "val", "test"):
            p = d / f"{role}.txt"
            if p.is_file():
                out[role] = p
    return out


def _image_paths_from_gt(gt: Any) -> set[str]:
    from harchoc.detection_match import _iter_image_records, _image_id_from_record

    paths: set[str] = set()
    for rec in _iter_image_records(gt):
        fn = rec.get("file_name")
        if fn:
            paths.add(str(fn).replace("\\", "/"))
        iid = _image_id_from_record(rec)
        if iid:
            paths.add(str(iid).replace("\\", "/"))
    return paths


def _normalize_rel(p: str) -> str:
    return p.replace("\\", "/").lstrip("./")


def infer_split_role(
    *,
    gt_json: str | None = None,
    preds_json: str | None = None,
    split_file: str | None = None,
    gt: Any | None = None,
    repo_root: Path | None = None,
    dataset_root: Path | None = None,
) -> tuple[SplitRole, dict[str, Any]]:
    """
    Infer train/val/test role from split file path, export JSON paths, or overlap with tracked splits.
    """
    hints: dict[str, Any] = {}

    if split_file and str(split_file).strip():
        sf = Path(split_file).expanduser()
        hints["split_file"] = str(sf)
        role = _role_from_name(sf.stem)
        if role is not None:
            hints["method"] = "split_file"
            return role, hints

    for label, path in (("gt_json", gt_json), ("preds_json", preds_json)):
        if not path or not str(path).strip():
            continue
        p = Path(path).expanduser()
        hints[label] = str(p)
        role = _role_from_name(p.stem) or _role_from_name(p.parent.name)
        if role is not None:
            hints["method"] = f"{label}_path"
            return role, hints

    if gt is not None and repo_root is not None:
        img_paths = _image_paths_from_gt(gt)
        if img_paths:
            splits = _split_paths(repo_root=repo_root, dataset_root=dataset_root)
            scores: dict[str, int] = {}
            for role, sp in splits.items():
                with capture_failure(f"read split list {sp}") as cap:
                    from harchoc.splits_io import read_split_list

                    entries = read_split_list(sp, missing_ok=True)
                if cap.failed:
                    fail_or_warn(f"{cap.context}: {cap.exc_type}: {cap.exc_msg}")
                    continue
                norm_entries = {_normalize_rel(str(e)) for e in entries}
                hit = sum(
                    1
                    for ip in img_paths
                    if _normalize_rel(ip) in norm_entries
                    or any(_normalize_rel(ip).endswith(e) for e in norm_entries)
                )
                if hit:
                    scores[role] = hit
            if scores:
                best = max(scores, key=lambda k: scores[k])
                hints["method"] = "split_overlap"
                hints["overlap_hits"] = scores
                return best, hints  # type: ignore[return-value]

    hints["method"] = "unknown"
    return "unknown", hints


def tuning_active(
    *,
    locked_conf_from: str | None,
    iou_grid: list[float] | None,
    calibrate: str = "none",
) -> bool:
    """True when this run would tune thresholds (not val-locked reporting on test)."""
    if (locked_conf_from or "").strip():
        return False
    if iou_grid:
        return True
    if str(calibrate or "none") != "none":
        return True
    return True  # conf sweep + --select always tunes unless locked


def check_tuning_guardrails(
    split_role: SplitRole,
    *,
    locked_conf_from: str | None,
    iou_grid: list[float] | None,
    calibrate: str = "none",
    allow_test_tuning: bool = False,
) -> list[str]:
    """
    Return warning messages; raise SystemExit via enforce_tuning_guardrails when test tuning is forbidden.
    """
    if allow_test_tuning or split_role != "test":
        return []
    if not tuning_active(locked_conf_from=locked_conf_from, iou_grid=iou_grid, calibrate=calibrate):
        return []
    return [
        "Threshold tuning (--select, --iou-grid, or --calibrate) must not run on the test split. "
        "Sweep on val, then report on test with --locked-conf-from <val_sweep.json> (no re-selection)."
    ]


def enforce_tuning_guardrails(
    split_role: SplitRole,
    *,
    locked_conf_from: str | None,
    iou_grid: list[float] | None,
    calibrate: str = "none",
    allow_test_tuning: bool = False,
) -> None:
    msgs = check_tuning_guardrails(
        split_role,
        locked_conf_from=locked_conf_from,
        iou_grid=iou_grid,
        calibrate=calibrate,
        allow_test_tuning=allow_test_tuning,
    )
    if msgs:
        raise SystemExit(msgs[0])


def build_iou_grid(
    *,
    iou: float,
    iou_grid: list[float] | None = None,
    iou_min: float | None = None,
    iou_max: float | None = None,
    iou_steps: int | None = None,
) -> list[float] | None:
    """Return explicit IoU grid values, or None when only a single --iou is used."""
    if iou_grid:
        return sorted({max(0.0, min(1.0, float(x))) for x in iou_grid})
    if iou_min is not None or iou_max is not None or iou_steps is not None:
        lo = float(iou_min if iou_min is not None else iou)
        hi = float(iou_max if iou_max is not None else iou)
        steps = int(iou_steps if iou_steps is not None else 5)
        if steps <= 1:
            return [lo]
        if hi < lo:
            lo, hi = hi, lo
        return [lo + (hi - lo) * (i / (steps - 1)) for i in range(steps)]
    return None


def select_min_count_mae(
    rows: list[dict[str, Any]],
    *,
    gt: Any,
    preds: Any,
    iou_thr: float = 0.5,
    category_aware: bool = True,
) -> dict[str, Any] | None:
    """Pick conf_thr minimizing per-image count MAE (tie-break: lower conf_thr)."""
    best: dict[str, Any] | None = None
    best_mae = float("inf")
    for row in rows:
        conf = float(row.get("conf_thr", 0.0))
        m = counting_metrics_at_conf(
            gt=gt,
            preds=preds,
            conf_thr=conf,
            iou_thr=float(iou_thr),
            category_aware=category_aware,
        )
        mae = float(m.get("mae", float("inf")))
        if best is None or mae < best_mae or (
            math.isclose(mae, best_mae) and conf < float(best.get("conf_thr", 1.0))
        ):
            best = {**row, "count_mae": mae}
            best_mae = mae
    return best


def select_best_iou_from_grid(
    grid_results: list[dict[str, Any]],
) -> tuple[float, dict[str, Any] | None]:
    """Pick IoU with best selected F1; tie-break toward lower IoU."""
    best_iou = float(grid_results[0]["iou"])
    best_row: dict[str, Any] | None = grid_results[0].get("selected_row")
    best_f1 = float((best_row or {}).get("f1", 0.0))
    for rec in grid_results[1:]:
        row = rec.get("selected_row")
        f1 = float((row or {}).get("f1", 0.0))
        iou_v = float(rec["iou"])
        if f1 > best_f1 or (f1 == best_f1 and iou_v < best_iou):
            best_iou = iou_v
            best_row = row
            best_f1 = f1
    return best_iou, best_row


def counting_metrics_at_conf(
    *,
    gt: Any,
    preds: Any,
    conf_thr: float,
    iou_thr: float = 0.5,
    category_aware: bool = True,
) -> dict[str, Any]:
    """Per-image count MAE/rRMSE at a fixed confidence (matches error_analysis counting)."""
    from harchoc.detection_match import per_image_detection_counts

    per_image = per_image_detection_counts(
        gt=gt,
        preds=preds,
        conf_thr=float(conf_thr),
        category_aware=category_aware,
    )
    return aggregate_counting_metrics(per_image)


def aggregate_counting_metrics(per_image: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from harchoc.stats_ci import ci_for_values

    errs: list[float] = []
    abs_errs: list[float] = []
    gt_counts: list[int] = []
    for rec in per_image.values():
        n_gt = int(rec.get("n_gt") or 0)
        n_pred = int(rec.get("n_pred") or 0)
        gt_counts.append(n_gt)
        e = float(n_pred - n_gt)
        errs.append(e)
        abs_errs.append(abs(e))
    if not errs:
        return {"mae": 0.0, "rmse": 0.0, "rrmse": None, "n_images": 0}
    mae = sum(abs_errs) / len(abs_errs)
    rmse = (sum(e * e for e in errs) / len(errs)) ** 0.5
    mean_gt = sum(gt_counts) / len(gt_counts) if gt_counts else 0.0
    rrmse = (rmse / mean_gt) if mean_gt > 0 else None
    out: dict[str, Any] = {"mae": mae, "rmse": rmse, "rrmse": rrmse, "n_images": len(errs)}
    mae_ci = ci_for_values(abs_errs, stat="mean")
    if mae_ci is not None:
        out["mae_ci"] = mae_ci.to_json()
    err_ci = ci_for_values(errs, stat="mean")
    if err_ci is not None:
        out["signed_error_mean_ci"] = err_ci.to_json()
    return out


def resolve_dataset_root_for_splits(
    *,
    repo_root: Path,
    dataset_root: Path | None,
) -> Path | None:
    if dataset_root is not None and dataset_root.is_dir():
        return dataset_root
    env = (os.getenv("DATASET_ROOT") or "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            return p
    return None
