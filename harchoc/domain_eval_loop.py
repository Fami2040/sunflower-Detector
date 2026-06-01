"""Per-tray domain eval via ``scripts.eval`` (CPU-friendly minimal GPU)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from harchoc.domain_eval import domain_split_file
from harchoc.json_io import load_json_dict


def metrics_from_eval_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    doc = load_json_dict(path)
    mAP50 = doc.get("mAP50")
    mAP50_95 = doc.get("mAP50_95")
    from harchoc.domain_count_mae import counting_mae_from_eval_doc

    count_block = counting_mae_from_eval_doc(doc)
    if mAP50 is None and mAP50_95 is None and count_block is None:
        return None
    out: dict[str, Any] = {"eval_out": str(path)}
    if mAP50 is not None:
        out["mAP50"] = mAP50
    if mAP50_95 is not None:
        out["mAP50_95"] = mAP50_95
    if count_block is not None:
        out.update({k: v for k, v in count_block.items() if k != "counting_metrics"})
        if count_block.get("counting_metrics") is not None:
            out["counting_metrics"] = count_block["counting_metrics"]
    return out


def build_tray_eval_argv(
    *,
    weights: str,
    tray_key: str,
    domains_dir: str | Path,
    split: str,
    out: str | Path,
    locked_conf_from: str | None,
    device: str,
    imgsz: int | None,
    max_det: int | None,
    manifest: str,
    default_dataset_name: str,
    dataset_name: str | None,
    dataset_root: str | None,
    yolo_data_yaml: str | None,
) -> list[str]:
    split_path = Path(domain_split_file(tray_key=tray_key, split=split, domains_dir=domains_dir))
    argv = [
        "--weights",
        str(weights),
        "--split-file",
        str(split_path),
        "--out",
        str(out),
        "--device",
        str(device),
    ]
    if imgsz is not None:
        argv.extend(["--imgsz", str(int(imgsz))])
    if max_det is not None:
        argv.extend(["--max-det", str(int(max_det))])
    if locked_conf_from:
        argv.extend(["--locked-conf-from", str(locked_conf_from)])
    if manifest:
        argv.extend(["--manifest", str(manifest)])
    if default_dataset_name:
        argv.extend(["--default-dataset-name", str(default_dataset_name)])
    if dataset_name:
        argv.extend(["--dataset-name", str(dataset_name)])
    if dataset_root:
        argv.extend(["--dataset-root", str(dataset_root)])
    if yolo_data_yaml:
        argv.extend(["--yolo-data-yaml", str(yolo_data_yaml)])
    return argv


def run_tray_domain_eval(
    *,
    tray_key: str,
    weights: str,
    domains_dir: str | Path,
    reports_dir: str | Path,
    split: str = "test",
    locked_conf_from: str | None = None,
    device: str = "cpu",
    imgsz: int | None = 1280,
    max_det: int | None = 3000,
    manifest: str = "data/manifest.json",
    default_dataset_name: str = "sunflower",
    dataset_name: str | None = None,
    dataset_root: str | None = None,
    yolo_data_yaml: str | None = None,
    eval_main: Callable[[list[str] | None], int] | None = None,
) -> tuple[int, dict[str, Any] | None]:
    """
    Run ``eval.py`` on ``{domains_dir}/{split}_{tray_key}.txt``; return exit code and metrics dict.
    """
    split_path = Path(domain_split_file(tray_key=tray_key, split=split, domains_dir=domains_dir))
    if not split_path.is_file():
        return 1, None

    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    out_path = reports / f"domain_eval_{split}_{tray_key.replace('/', '_')}.json"

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

    if eval_main is None:
        from scripts.eval import main as eval_main_fn

        eval_main = eval_main_fn

    rc = int(eval_main(argv))
    metrics = metrics_from_eval_json(out_path) if rc == 0 else None
    return rc, metrics


def _apply_one_tray_to_domain_eval(
    domains: list[Any],
    *,
    tray_key: str,
    metrics: dict[str, Any] | None,
    eval_rc: int,
) -> bool:
    """Update one tray record; return True if metrics were applied."""
    applied = False
    for rec in domains:
        if not isinstance(rec, dict):
            continue
        if str(rec.get("tray_key")) != tray_key:
            continue
        if metrics is not None:
            rec["metrics"] = metrics
            rec["eval_status"] = "ok"
            applied = True
        else:
            rec["metrics"] = None
            rec["eval_status"] = "failed" if eval_rc != 0 else "skipped"
        rec["eval_returncode"] = eval_rc
    return applied


def apply_tray_metrics_to_domain_eval(
    payload: dict[str, Any],
    *,
    tray_key: str,
    metrics: dict[str, Any] | None,
    eval_rc: int,
) -> dict[str, Any]:
    domains = payload.get("domains")
    if not isinstance(domains, list):
        return payload
    applied = _apply_one_tray_to_domain_eval(
        domains, tray_key=tray_key, metrics=metrics, eval_rc=eval_rc
    )
    if applied:
        payload["status"] = "partial"
    return payload


TrayEvalResult = tuple[str, int, dict[str, Any] | None]


def run_all_tray_domain_evals(
    tray_keys: list[str],
    *,
    weights: str,
    domains_dir: str | Path,
    reports_dir: str | Path,
    split: str = "test",
    locked_conf_from: str | None = None,
    device: str = "cpu",
    imgsz: int | None = 1280,
    max_det: int | None = 3000,
    manifest: str = "data/manifest.json",
    default_dataset_name: str = "sunflower",
    dataset_name: str | None = None,
    dataset_root: str | None = None,
    yolo_data_yaml: str | None = None,
    eval_main: Callable[[list[str] | None], int] | None = None,
) -> list[TrayEvalResult]:
    results: list[TrayEvalResult] = []
    for key in tray_keys:
        rc, metrics = run_tray_domain_eval(
            tray_key=key,
            weights=weights,
            domains_dir=domains_dir,
            reports_dir=reports_dir,
            split=split,
            locked_conf_from=locked_conf_from,
            device=device,
            imgsz=imgsz,
            max_det=max_det,
            manifest=manifest,
            default_dataset_name=default_dataset_name,
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            yolo_data_yaml=yolo_data_yaml,
            eval_main=eval_main,
        )
        results.append((key, rc, metrics))
    return results


def merge_tray_eval_results_into_domain_eval(
    payload: dict[str, Any],
    results: list[TrayEvalResult],
    *,
    device: str | None = None,
) -> dict[str, Any]:
    domains = payload.get("domains")
    if not isinstance(domains, list):
        return payload

    ok = 0
    failed = 0
    for tray_key, eval_rc, metrics in results:
        _apply_one_tray_to_domain_eval(
            domains, tray_key=tray_key, metrics=metrics, eval_rc=eval_rc
        )
        if metrics is not None:
            ok += 1
        else:
            failed += 1

    n = len(results)
    if n == 0:
        payload["status"] = payload.get("status", "scaffold")
    elif ok == n:
        payload["status"] = "ok"
    elif ok > 0:
        payload["status"] = "partial"
    else:
        payload["status"] = "failed"

    summary: dict[str, Any] = {"n_trays": n, "n_ok": ok, "n_failed": failed}
    if device is not None:
        summary["device"] = device
    payload["tray_eval_summary"] = summary
    return payload
