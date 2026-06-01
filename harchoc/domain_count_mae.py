"""Per-tray counting MAE at val-locked conf (domain_eval.v1 merge)."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Callable

from harchoc.hsp_export_protocol import EXPORT_CONF, EXPORT_IOU
from harchoc.json_io import load_json_dict
from harchoc.schemas import with_schema_version
from harchoc.threshold_lock import load_locked_conf

DOMAIN_COUNT_MAE_SCHEMA = "domain_count_mae.v1"


def match_settings_from_threshold_json(path: str | Path) -> tuple[float, bool]:
    obj = load_json_dict(path)
    match = obj.get("match") if isinstance(obj, dict) else None
    if not isinstance(match, dict):
        return EXPORT_IOU, True
    iou = float(match.get("iou", EXPORT_IOU))
    category_aware = bool(match.get("category_aware", True))
    return iou, category_aware


def counting_mae_from_eval_doc(doc: dict[str, Any]) -> dict[str, Any] | None:
    if doc.get("count_mae") is not None:
        return {
            "count_mae": float(doc["count_mae"]),
            "count_n_images": int((doc.get("counting_metrics") or {}).get("n_images") or 0),
            "locked_conf": doc.get("locked_conf"),
            "counting_metrics": doc.get("counting_metrics"),
        }
    cm = doc.get("counting_metrics")
    if isinstance(cm, dict) and cm.get("mae") is not None:
        return {
            "count_mae": float(cm["mae"]),
            "count_n_images": int(cm.get("n_images") or 0),
            "locked_conf": doc.get("locked_conf"),
            "counting_metrics": cm,
        }
    return None


def merge_count_fields_into_metrics(
    metrics: dict[str, Any] | None,
    count_block: dict[str, Any],
) -> dict[str, Any]:
    out = dict(metrics or {})
    out["count_mae"] = count_block["count_mae"]
    out["count_n_images"] = count_block.get("count_n_images")
    if count_block.get("locked_conf") is not None:
        out["locked_conf"] = count_block["locked_conf"]
    if count_block.get("counting_metrics") is not None:
        out["counting_metrics"] = count_block["counting_metrics"]
    return out


def summarize_tray_count_mae(
    domains: list[Any],
    *,
    locked_conf_from: str,
    locked_conf: float,
) -> dict[str, Any]:
    maes: list[float] = []
    per_tray: list[dict[str, Any]] = []
    n_ok = 0
    n_missing = 0
    for rec in domains:
        if not isinstance(rec, dict):
            continue
        key = str(rec.get("tray_key") or "")
        metrics = rec.get("metrics")
        if not isinstance(metrics, dict) or metrics.get("count_mae") is None:
            n_missing += 1
            continue
        mae = float(metrics["count_mae"])
        maes.append(mae)
        n_ok += 1
        per_tray.append(
            {
                "tray_key": key,
                "count_mae": mae,
                "count_n_images": metrics.get("count_n_images"),
                "mAP50": metrics.get("mAP50"),
            }
        )
    summary: dict[str, Any] = {
        "locked_conf_from": str(locked_conf_from),
        "locked_conf": float(locked_conf),
        "n_trays_with_count_mae": n_ok,
        "n_trays_missing_count_mae": n_missing,
    }
    if maes:
        summary["count_mae_mean"] = statistics.fmean(maes)
        summary["count_mae_median"] = statistics.median(maes)
        summary["count_mae_min"] = min(maes)
        summary["count_mae_max"] = max(maes)
        summary["count_mae_stdev"] = statistics.pstdev(maes) if len(maes) > 1 else 0.0
    else:
        summary["count_mae_mean"] = None
        summary["count_mae_median"] = None
        summary["count_mae_min"] = None
        summary["count_mae_max"] = None
        summary["count_mae_stdev"] = None
    summary["per_tray"] = sorted(per_tray, key=lambda x: x["tray_key"])
    return summary


def build_domain_count_mae_sidecar(
    *,
    domain_eval: dict[str, Any],
    summary: dict[str, Any],
    weights: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "ok" if summary.get("n_trays_with_count_mae") else "empty",
        "script": "eval_domains",
        "source": "domain_eval.v1",
        "weights": str(weights),
        "summary": summary,
        "domains": summary.get("per_tray") or [],
    }
    if domain_eval.get("tray_eval_summary") is not None:
        payload["tray_eval_summary"] = domain_eval["tray_eval_summary"]
    return with_schema_version(payload, schema_version=DOMAIN_COUNT_MAE_SCHEMA)


def apply_count_mae_summary_to_domain_eval(
    payload: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    payload["count_mae_summary"] = summary
    if summary.get("n_trays_with_count_mae"):
        notes = str(payload.get("notes") or "").strip()
        extra = (
            f"Per-tray count MAE @ locked conf {summary['locked_conf']:.4g} "
            f"({summary['n_trays_with_count_mae']} trays; "
            f"mean {summary.get('count_mae_mean')})."
        )
        payload["notes"] = f"{notes} {extra}".strip() if notes else extra
    return payload


TrayCountResult = tuple[str, int, dict[str, Any] | None]


def run_tray_count_mae_eval(
    *,
    tray_key: str,
    weights: str,
    domains_dir: str | Path,
    reports_dir: str | Path,
    locked_conf_from: str,
    split: str = "test",
    device: str = "cpu",
    export_conf: float = EXPORT_CONF,
    imgsz: int | None = 1280,
    max_det: int | None = 3000,
    manifest: str = "data/manifest.json",
    default_dataset_name: str = "sunflower",
    dataset_name: str | None = None,
    dataset_root: str | None = None,
    yolo_data_yaml: str | None = None,
    eval_main: Callable[[list[str] | None], int] | None = None,
) -> TrayCountResult:
    from harchoc.domain_eval_loop import build_tray_eval_argv, domain_split_file

    split_path = Path(domain_split_file(tray_key=tray_key, split=split, domains_dir=domains_dir))
    if not split_path.is_file():
        return tray_key, 1, None

    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    safe_key = tray_key.replace("/", "_")
    out_path = reports / f"domain_count_{split}_{safe_key}.json"
    gt_out = reports / f"domain_gt_{split}_{safe_key}.json"
    preds_out = reports / f"domain_preds_{split}_{safe_key}.json"

    argv = build_tray_eval_argv(
        weights=weights,
        tray_key=tray_key,
        domains_dir=domains_dir,
        split=split,
        out=str(out_path),
        locked_conf_from=locked_conf_from,
        device=device,
        imgsz=imgsz,
        max_det=max_det,
        manifest=manifest,
        default_dataset_name=default_dataset_name,
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        yolo_data_yaml=yolo_data_yaml,
    )
    argv.extend(
        [
            "--export-only",
            "--export-gt-json",
            str(gt_out),
            "--export-preds-json",
            str(preds_out),
            "--export-conf",
            str(float(export_conf)),
            "--export-device",
            str(device),
        ]
    )

    if eval_main is None:
        from scripts.eval import main as eval_main_fn

        eval_main = eval_main_fn

    rc = int(eval_main(argv))
    if rc != 0 or not out_path.is_file():
        return tray_key, rc, None

    doc = load_json_dict(out_path)
    block = counting_mae_from_eval_doc(doc)
    if block is None:
        return tray_key, rc, None
    block["count_eval_out"] = str(out_path)
    return tray_key, rc, block


def merge_tray_count_mae_results(
    payload: dict[str, Any],
    results: list[TrayCountResult],
    *,
    locked_conf_from: str,
    locked_conf: float,
) -> dict[str, Any]:
    domains = payload.get("domains")
    if not isinstance(domains, list):
        return payload

    for tray_key, rc, block in results:
        for rec in domains:
            if not isinstance(rec, dict) or str(rec.get("tray_key")) != tray_key:
                continue
            if block is not None:
                rec["metrics"] = merge_count_fields_into_metrics(rec.get("metrics"), block)
                rec["count_mae_status"] = "ok"
            else:
                rec["count_mae_status"] = "failed" if rc != 0 else "skipped"
            rec["count_mae_returncode"] = rc

    summary = summarize_tray_count_mae(domains, locked_conf_from=locked_conf_from, locked_conf=locked_conf)
    apply_count_mae_summary_to_domain_eval(payload, summary)
    payload["locked_conf_from"] = str(locked_conf_from)
    return payload
