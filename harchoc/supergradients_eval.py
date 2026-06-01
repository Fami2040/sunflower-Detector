from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from harchoc.data_yaml import read_class_names
from harchoc.supergradients_train import _repo_splits_dir, materialize_yolo_staging


def eval_test_for_bench(
    *,
    weights: str | Path,
    dataset_root: Path,
    eval_out: Path | None = None,
    max_det: int | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    """
  Run SuperGradients validation on the repo test split; return mAP50 / mAP50-95 when available.
    """
    weights_path = Path(weights).resolve()
    if not weights_path.is_file():
        return {
            "status": "failed",
            "reason": "weights_not_found",
            "split": "test",
            "weights": str(weights_path),
        }

    test_split = _repo_splits_dir() / "test.txt"
    if not test_split.is_file():
        return {
            "status": "failed",
            "reason": "missing_test_split",
            "split": "test",
        }

    try:
        from super_gradients.training import Trainer
        from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_val
        from super_gradients.training import models
    except ImportError as exc:
        return {
            "status": "skipped",
            "reason": f"missing_dependency:super_gradients ({exc})",
            "split": "test",
        }

    classes = read_class_names(dataset_root=dataset_root)
    staging_root = Path(tempfile.mkdtemp(prefix="sg_yolo_test_"))
    try:
        layout = materialize_yolo_staging(
            dataset_root=dataset_root,
            staging_root=staging_root,
            train_split=test_split,
            val_split=test_split,
        )
        test_loader = coco_detection_yolo_format_val(
            dataset_params={
                "data_dir": str(staging_root),
                "images_dir": layout["val_images"],
                "labels_dir": layout["val_labels"],
                "classes": classes,
            },
            dataloader_params={"batch_size": 1, "num_workers": 2},
        )

        model_id = "yolo_nas_s"
        model = models.get(model_id, num_classes=len(classes), pretrained_weights="coco")
        trainer = Trainer(experiment_name="sg_eval", ckpt_root_dir=str(staging_root / "ckpt"))
        trainer.load_checkpoint(str(weights_path))
        metrics_raw = trainer.test(model=model, test_loader=test_loader, test_metrics_list=None) or {}

        map50, map50_95 = _extract_map_metrics(metrics_raw)
        payload: dict[str, Any] = {
            "status": "ok",
            "split": "test",
            "mAP50": map50,
            "mAP50_95": map50_95,
            "backend": "supergradients",
            "weights": str(weights_path),
            "config_path": config_path,
            "metrics_raw": _json_safe(metrics_raw),
        }
        if max_det is not None:
            payload["max_det_requested"] = max_det
        if eval_out is not None:
            eval_out.parent.mkdir(parents=True, exist_ok=True)
            eval_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", "utf-8")
            payload["eval_out"] = str(eval_out)
        return payload
    except Exception as exc:
        return {
            "status": "failed",
            "reason": str(exc),
            "exc_type": type(exc).__name__,
            "split": "test",
            "weights": str(weights_path),
        }
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def _unwrap_sg_prediction(raw: Any) -> Any:
    if raw is None:
        return None
    pred = getattr(raw, "prediction", None)
    if pred is not None:
        return pred
    try:
        return raw[0]
    except Exception:
        return raw


def _row_to_floats(row: Any) -> list[float]:
    try:
        vals = row.tolist()
    except Exception:
        vals = row
    return [float(v) for v in vals]


def sg_prediction_to_detections(prediction: Any, *, max_det: int) -> list[dict[str, Any]]:
    """Convert SuperGradients predict() output to HSP preds JSON detection entries."""
    pred = _unwrap_sg_prediction(prediction)
    if pred is None:
        return []
    bboxes = getattr(pred, "bboxes_xyxy", None)
    if bboxes is None:
        return []
    scores = getattr(pred, "confidence", None)
    if scores is None:
        scores = getattr(pred, "scores", None)
    labels = getattr(pred, "labels", None)
    detections: list[dict[str, Any]] = []
    n = len(bboxes)
    for i in range(n):
        coords = _row_to_floats(bboxes[i])
        if len(coords) < 4:
            continue
        x1, y1, x2, y2 = coords[:4]
        score = float(_row_to_floats(scores[i])[0]) if scores is not None else None
        cat = int(_row_to_floats(labels[i])[0]) if labels is not None else 0
        detections.append(
            {
                "bbox": [x1, y1, x2, y2],
                "category_id": cat,
                "score": score,
            }
        )
    detections.sort(key=lambda d: float(d.get("score") or 0.0), reverse=True)
    if max_det > 0:
        detections = detections[: int(max_det)]
    return detections


def export_hsp_gt_preds_json(
    *,
    weights: str | Path,
    split_file: Path,
    dataset_root: Path,
    gt_out: Path,
    preds_out: Path,
    model_id: str = "yolo_nas_s",
    conf: float = 0.001,
    iou: float = 0.3,
    max_det: int = 3000,
    device: str | None = None,
) -> dict[str, Any]:
    """
    Export GT + SuperGradients predictions for HSP error_analysis (not scripts/eval.py).
    """
    from harchoc.eval_export import build_gt_export, iter_split_image_paths

    weights_path = Path(weights).resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"weights not found: {weights_path}")

    gt_obj = build_gt_export(split_file=split_file, dataset_root=dataset_root)
    gt_out.parent.mkdir(parents=True, exist_ok=True)
    preds_out.parent.mkdir(parents=True, exist_ok=True)
    gt_out.write_text(json.dumps(gt_obj, indent=2) + "\n", encoding="utf-8")

    try:
        from super_gradients.training import Trainer
        from super_gradients.training import models
    except ImportError as exc:
        raise RuntimeError(f"missing_dependency:super_gradients ({exc})") from exc

    classes = read_class_names(dataset_root=dataset_root)
    model = models.get(model_id, num_classes=len(classes), pretrained_weights="coco")
    staging_root = Path(tempfile.mkdtemp(prefix="sg_hsp_export_"))
    try:
        ckpt_dir = staging_root / "ckpt"
        trainer = Trainer(experiment_name="sg_hsp_export", ckpt_root_dir=str(ckpt_dir))
        trainer.load_checkpoint(str(weights_path))
        if device:
            model = model.to(device)

        images: list[dict[str, Any]] = []
        for img_id, img_path, file_name in iter_split_image_paths(split_file, dataset_root=dataset_root):
            if not img_path.is_file():
                continue
            raw = model.predict(str(img_path), conf=float(conf), iou=float(iou), fp16=False)
            dets = sg_prediction_to_detections(raw, max_det=max_det)
            images.append({"image_id": img_id, "file_name": file_name, "detections": dets})
        preds_obj = {"images": images}
        preds_out.write_text(json.dumps(preds_obj, indent=2) + "\n", encoding="utf-8")
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    return {
        "gt_json": str(gt_out.resolve()),
        "preds_json": str(preds_out.resolve()),
        "n_images": len(gt_obj.get("images") or []),
        "export_conf": conf,
        "export_iou": iou,
        "export_max_det": max_det,
        "backend": "supergradients",
        "weights": str(weights_path),
        "model_id": model_id,
    }


def run_sg_hsp_eval_chain(
    *,
    repo_root: str | Path,
    run_name: str,
    weights: str | Path,
    locked_conf_from: str,
    out_dir: str,
    max_det: int = 3000,
    model_id: str = "yolo_nas_s",
    dry_run: bool = False,
    on_stage: Callable[[str, list[str]], None] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """SuperGradients HSP chain: SG export + error_analysis (no scripts/eval.py)."""
    rr = Path(repo_root).resolve()
    prefix = f"{out_dir}/{run_name}"
    gt_json = rr / f"{prefix}_gt.json"
    preds_json = rr / f"{prefix}_preds.json"
    error_json = rr / f"{prefix}_error.json"
    split_file = rr / "data/splits/test.txt"
    error_argv = [
        "scripts/error_analysis.py",
        "--gt-json",
        str(gt_json.relative_to(rr)),
        "--preds-json",
        str(preds_json.relative_to(rr)),
        "--locked-conf-from",
        locked_conf_from,
        "--out",
        str(error_json.relative_to(rr)),
    ]
    if on_stage is not None:
        on_stage(
            "sg_export",
            [
                "harchoc.supergradients_eval.export_hsp_gt_preds_json",
                "--weights",
                str(weights),
                "--model-id",
                model_id,
            ],
        )
        on_stage("error_analysis", error_argv)
    if dry_run:
        return {
            "eval_json": str((rr / f"{prefix}_eval.json").resolve()),
            "error_json": str(error_json.resolve()),
            "gt_json": str(gt_json.resolve()),
            "preds_json": str(preds_json.resolve()),
        }

    from harchoc.datasets import resolve_dataset

    spec = resolve_dataset(manifest_path=rr / "data/manifest.json", default_dataset_name="sunflower")
    export_hsp_gt_preds_json(
        weights=weights,
        split_file=split_file,
        dataset_root=spec.root,
        gt_out=gt_json,
        preds_out=preds_json,
        model_id=model_id,
        max_det=max_det,
        device=(env or {}).get("HARCHOC_EXPORT_DEVICE") or os.environ.get("HARCHOC_EXPORT_DEVICE") or "cpu",
    )
    run_env = {**dict(os.environ), **(env or {})}
    mamba_env = os.environ.get("HARCHOC_MAMBA_ENV", "harchoc")
    cmd = ["mamba", "run", "-n", mamba_env, "python", *error_argv]
    proc = subprocess.run(cmd, cwd=str(rr), env=run_env)
    if proc.returncode != 0:
        raise RuntimeError(f"sg hsp eval stage error_analysis failed: exit {proc.returncode}")
    return {
        "eval_json": str((rr / f"{prefix}_eval.json").resolve()),
        "error_json": str(error_json.resolve()),
        "gt_json": str(gt_json.resolve()),
        "preds_json": str(preds_json.resolve()),
    }


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _extract_map_metrics(metrics: Any) -> tuple[float | None, float | None]:
    if not isinstance(metrics, dict):
        return None, None
    for k50, k95 in (
        ("mAP@0.50", "mAP@0.50:0.95"),
        ("map50", "map"),
        ("mAP50", "mAP50-95"),
    ):
        v50 = metrics.get(k50)
        v95 = metrics.get(k95)
        if isinstance(v50, (int, float)):
            m50 = float(v50)
            m95 = float(v95) if isinstance(v95, (int, float)) else None
            return m50, m95
    return None, None
