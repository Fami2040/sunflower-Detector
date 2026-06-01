"""Deploy (SAHI slice) vs manuscript (HSP full-frame locked conf) parity report."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from harchoc.deploy_filters import DeployFilterConfig, filter_object_predictions
from harchoc.eval_export import iter_split_image_paths, ultralytics_results_to_detections
from harchoc.sahi_infer import SahiSliceConfig, load_ultralytics_detection_model, model_confidence_min_from_env, run_sliced_prediction
from harchoc.schemas import with_schema_version

DEPLOY_HSP_PARITY_SCHEMA = "deploy_hsp_parity.v1"
HSP_FULLFRAME_IOU = 0.3
HSP_FULLFRAME_MAX_DET = 3000

CountFn = Callable[[str], dict[str, int]]
PerImageBuilder = Callable[[list[str], float], list[dict[str, Any]]]


def _locked_conf_value(locked_conf_from: str | None) -> float | None:
    if locked_conf_from and Path(locked_conf_from).is_file():
        from harchoc.threshold_lock import load_locked_conf

        return float(load_locked_conf(locked_conf_from))
    raw = os.getenv("HARCHOC_LOCKED_CONF", "").strip()
    if raw:
        return float(raw)
    json_path = os.getenv("HARCHOC_LOCKED_CONF_JSON", "").strip()
    if json_path and Path(json_path).is_file():
        from harchoc.threshold_lock import load_locked_conf

        return float(load_locked_conf(json_path))
    return None


def class_count_dict(*, developed: int, aborted: int) -> dict[str, int]:
    return {
        "developed": int(developed),
        "aborted": int(aborted),
        "total": int(developed) + int(aborted),
    }


def counts_from_category_ids(category_ids: list[int]) -> dict[str, int]:
    developed = sum(1 for c in category_ids if int(c) == 0)
    aborted = sum(1 for c in category_ids if int(c) == 1)
    return class_count_dict(developed=developed, aborted=aborted)


def sample_image_paths_from_split(
    *,
    split_file: Path,
    dataset_root: Path,
    n: int,
) -> list[str]:
    """First *n* existing absolute image paths from a split list file."""
    if n <= 0:
        return []
    paths: list[str] = []
    for _img_id, img_path, _file_name in iter_split_image_paths(
        split_file, dataset_root=dataset_root
    ):
        if not img_path.is_file():
            continue
        paths.append(str(img_path.resolve()))
        if len(paths) >= n:
            break
    return paths


def _infer_device() -> str:
    try:
        import torch

        fd = os.getenv("FORCE_DEVICE", "").lower()
        if fd == "cuda" and torch.cuda.is_available():
            return "cuda"
        if fd == "cpu":
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def count_sahi_deploy(
    image_path: str,
    *,
    weights: str | Path,
    device: str | None = None,
) -> dict[str, int]:
    """SAHI sliced inference + deploy post-filters (telegram_bot / run_infer_once path)."""
    dev = device or _infer_device()
    model_path = str(weights)
    detection_model = load_ultralytics_detection_model(
        model_path,
        device=dev,
        confidence_threshold=model_confidence_min_from_env(),
    )
    result = run_sliced_prediction(image_path, detection_model, SahiSliceConfig.from_env())
    filtered = filter_object_predictions(
        result.object_prediction_list,
        DeployFilterConfig.resolve(),
    )
    return counts_from_category_ids([int(p.category.id) for p in filtered])


def count_hsp_fullframe_locked(
    image_path: str,
    *,
    weights: str | Path,
    locked_conf: float,
    device: str | None = None,
    iou: float = HSP_FULLFRAME_IOU,
    max_det: int = HSP_FULLFRAME_MAX_DET,
) -> dict[str, int]:
    """Full-frame Ultralytics predict at manuscript locked conf (run_infer_once --fullframe-export)."""
    from ultralytics import YOLO  # type: ignore

    dev = device or _infer_device()
    model = YOLO(str(weights))
    res_list = model.predict(
        str(image_path),
        conf=float(locked_conf),
        iou=float(iou),
        max_det=int(max_det),
        verbose=False,
        device=dev,
    )
    res = res_list[0] if res_list else None
    dets = ultralytics_results_to_detections(res) if res is not None else []
    return counts_from_category_ids([int(d["category_id"]) for d in dets])


def build_per_image_parity_rows(
    image_paths: list[str],
    *,
    locked_conf: float,
    weights: str | Path,
    device: str | None = None,
    sahi_count_fn: CountFn | None = None,
    fullframe_count_fn: CountFn | None = None,
) -> list[dict[str, Any]]:
    sahi_fn = sahi_count_fn or (
        lambda p: count_sahi_deploy(p, weights=weights, device=device)
    )
    hsp_fn = fullframe_count_fn or (
        lambda p: count_hsp_fullframe_locked(
            p, weights=weights, locked_conf=locked_conf, device=device
        )
    )
    rows: list[dict[str, Any]] = []
    for path in image_paths:
        sahi = sahi_fn(path)
        hsp = hsp_fn(path)
        rows.append(
            {
                "image_path": path,
                "locked_conf": float(locked_conf),
                "sahi_count": sahi,
                "hsp_fullframe_locked": hsp,
                "delta_total": int(sahi["total"]) - int(hsp["total"]),
            }
        )
    return rows


def build_deploy_hsp_parity_payload(
    *,
    locked_conf_from: str | None = None,
    image_paths: list[str] | None = None,
    per_image: list[dict[str, Any]] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    deploy_default = DeployFilterConfig.from_env()
    deploy_resolved = DeployFilterConfig.resolve()
    sahi = SahiSliceConfig.from_env()
    locked = _locked_conf_value(locked_conf_from)

    deploy_conf = {
        "conf_thr_fertilized": deploy_default.conf_thr_fertilized,
        "conf_thr_unfertilized": deploy_default.conf_thr_unfertilized,
        "model_confidence_min": model_confidence_min_from_env(),
        "source": "env CONF_THR_FERTILIZED / CONF_THR_UNFERTILIZED",
    }
    resolved_conf = {
        "conf_thr_fertilized": deploy_resolved.conf_thr_fertilized,
        "conf_thr_unfertilized": deploy_resolved.conf_thr_unfertilized,
        "uniform_if_locked": deploy_resolved.conf_thr_fertilized
        == deploy_resolved.conf_thr_unfertilized,
        "source": (
            "HARCHOC_LOCKED_CONF*"
            if DeployFilterConfig.from_locked_env() is not None
            else "env (same as deploy_default)"
        ),
    }
    hsp_locked = {
        "conf": locked,
        "locked_conf_from": locked_conf_from,
        "manuscript_operating_point": "~0.15 typical (val min_count_mae)",
    }

    comparison = {
        "deploy_vs_hsp_locked_delta_fert": (
            None
            if locked is None
            else round(deploy_default.conf_thr_fertilized - locked, 6)
        ),
        "deploy_vs_hsp_locked_delta_unfert": (
            None
            if locked is None
            else round(deploy_default.conf_thr_unfertilized - locked, 6)
        ),
        "resolved_matches_locked": (
            locked is not None
            and abs(deploy_resolved.conf_thr_fertilized - locked) < 1e-9
        ),
        "gap_summary": (
            "Deploy SAHI uses lower per-class slice thresholds (often 0.04–0.06); "
            "HSP eval uses val-locked conf on full-frame export (~0.15). "
            "Set HARCHOC_LOCKED_CONF or --locked-conf-from for deploy filter parity."
        ),
    }

    paths = image_paths or []
    payload: dict[str, Any] = {
        "status": "ok",
        "deploy_conf": deploy_conf,
        "deploy_conf_resolved": resolved_conf,
        "hsp_locked_conf": hsp_locked,
        "sahi_slice": {
            "slice_size": sahi.slice_size,
            "overlap": sahi.overlap,
            "nms_iou": sahi.nms_iou,
        },
        "comparison": comparison,
        "image_sample_count": len(paths),
        "image_paths": paths,
    }
    if per_image:
        payload["per_image"] = per_image
        deltas = [int(r.get("delta_total") or 0) for r in per_image]
        payload["image_sample_summary"] = {
            "n_compared": len(per_image),
            "mean_delta_total": round(sum(deltas) / len(deltas), 4) if deltas else None,
            "mean_abs_delta_total": round(sum(abs(d) for d in deltas) / len(deltas), 4)
            if deltas
            else None,
        }
    if notes:
        payload["notes"] = notes
    return with_schema_version(payload, schema_version=DEPLOY_HSP_PARITY_SCHEMA)


def resolve_parity_image_sample(
    *,
    sample_images: int,
    split_file: str | Path,
    dataset_root: str | Path | None,
    locked_conf_from: str | None,
    weights: str | Path,
    manifest_path: str | None = None,
    dataset_name: str | None = None,
    yolo_data_yaml: str | None = None,
    default_dataset_name: str | None = None,
    sahi_count_fn: CountFn | None = None,
    fullframe_count_fn: CountFn | None = None,
    device: str | None = None,
) -> tuple[list[str], list[dict[str, Any]] | None, str | None]:
    """
    Resolve split paths and optionally run per-image SAHI vs full-frame counts.

    Returns (image_paths, per_image_rows, skip_note). *per_image_rows* is None when
    sampling was skipped (missing split, locked conf, weights, or torch).
    """
    if sample_images <= 0:
        return [], None, None

    split_path = Path(split_file)
    if not split_path.is_file():
        return [], None, f"split file not found: {split_path}"

    root: Path | None = None
    if dataset_root:
        root = Path(dataset_root).resolve()
        if not root.is_dir():
            return [], None, f"dataset root not found: {root}"

    if root is None:
        try:
            from harchoc.datasets import resolve_dataset

            from harchoc.config_coerce import optional_str

            spec = resolve_dataset(
                manifest_path=optional_str(manifest_path) or "data/manifest.json",
                default_dataset_name=optional_str(default_dataset_name) or "sunflower-cvat-2500",
                dataset_name=optional_str(dataset_name),
                dataset_root=None,
                yolo_data_yaml=optional_str(yolo_data_yaml),
            )
            root = Path(spec.root).resolve()
        except Exception as exc:
            return [], None, f"dataset not resolved: {exc}"

    paths = sample_image_paths_from_split(
        split_file=split_path,
        dataset_root=root,
        n=sample_images,
    )
    if not paths:
        return [], None, f"no existing images in {split_path} under {root}"

    locked = _locked_conf_value(locked_conf_from)
    if locked is None:
        return paths, None, "locked conf unavailable (set --locked-conf-from or HARCHOC_LOCKED_CONF*)"

    weights_path = Path(weights)
    if not weights_path.is_file():
        return paths, None, f"weights not found: {weights_path}"

    if sahi_count_fn is None or fullframe_count_fn is None:
        try:
            import torch  # noqa: F401
        except ImportError:
            return paths, None, "torch not installed; skipping image sample inference"

    per_image = build_per_image_parity_rows(
        paths,
        locked_conf=float(locked),
        weights=weights_path,
        device=device,
        sahi_count_fn=sahi_count_fn,
        fullframe_count_fn=fullframe_count_fn,
    )
    return paths, per_image, None


def write_deploy_hsp_parity(
    out: str | Path,
    *,
    locked_conf_from: str | None = None,
    image_paths: list[str] | None = None,
    per_image: list[dict[str, Any]] | None = None,
    notes: str | None = None,
) -> Path:
    from scripts._common_cli import write_json

    payload = build_deploy_hsp_parity_payload(
        locked_conf_from=locked_conf_from,
        image_paths=image_paths,
        per_image=per_image,
        notes=notes,
    )
    return Path(write_json(str(out), payload))
