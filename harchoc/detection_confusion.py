"""3×3 detection confusion matrix (classes 0, 1, background) at a locked operating point.

Paths (pick one):

- **CPU / frozen preds:** ``confusion_matrix_from_exports(gt, preds, …)`` — same matcher as
  ``error_analysis_core.analyze_errors``; used by ``error_analysis.py --confusion-matrix-out``,
  ``experiment.py reviewer2-confusion``, and ``eval.py --confusion-from-exports``.
- **GPU streaming:** ``confusion_matrix_streaming`` / ``confusion_matrix_multi_split`` — Ultralytics
  ``predict`` over split images (default device ``cuda``); ``eval.py --confusion-matrix-only`` without
  ``--confusion-from-exports``.

IoU: §11 / error-analysis tables use **0.5**; HSP counting / ``best2_*_confusion.json`` use val-locked
conf with match IoU **0.3** (``resolve_match_settings`` + ``EXPORT_IOU``).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from harchoc.detection_match import (
    _extract_boxes,
    _filter_preds_by_conf,
    _index_records_by_image_id,
)
from harchoc.hsp_export_protocol import DEFAULT_EXPORT_MAX_DET, EXPORT_CONF, EXPORT_IOU
from harchoc.instance_match import classify_unmatched_prediction, find_same_class_tp_match

if TYPE_CHECKING:
    from harchoc.strict_ml import StrictWarnings

BG = 2
CLASS_LABELS = ["developed (0)", "aborted (1)", "background"]
DETECTION_CONFUSION_MATRIX_V1 = "detection_confusion_matrix.v1"


def _default_streaming_device(device: str | int | None) -> str | int:
    if device is not None:
        return device
    return (os.getenv("HARCHOC_EXPORT_DEVICE") or "").strip() or "cuda"


def _predict_batch_size() -> int:
    raw = (os.getenv("HARCHOC_PREDICT_BATCH") or "").strip()
    if not raw:
        return 1
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


@dataclass
class ConfusionMatrixAccumulator:
    """Greedy IoU matcher aligned with ``error_analysis_core.analyze_errors``."""

    matrix: list[list[int]] = field(default_factory=lambda: [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    stats: dict[str, int] = field(
        default_factory=lambda: {"tp": 0, "cls_confusion": 0, "dupe": 0, "fp": 0, "fn": 0}
    )
    n_images: int = 0

    def update_image(
        self,
        gt_annotations: list[dict[str, Any]],
        detections: list[dict[str, Any]],
        *,
        conf_thr: float,
        iou_thr: float,
        iou_bg_thr: float = 0.1,
    ) -> None:
        gt_boxes = _extract_boxes({"annotations": gt_annotations}, key="annotations")
        pr_filt = _filter_preds_by_conf(
            _extract_boxes({"detections": detections}, key="detections"),
            conf_thr,
        )
        gt_used = [False] * len(gt_boxes)

        for p in pr_filt:
            pc = int(p["category_id"])
            if pc not in (0, 1):
                continue

            best_i = find_same_class_tp_match(p, gt_boxes, gt_used, iou_thr=iou_thr)
            if best_i >= 0:
                gt_used[best_i] = True
                gc = int(gt_boxes[best_i]["category_id"])
                self.matrix[gc][pc] += 1
                self.stats["tp"] += 1
                continue

            outcome, confused_gt = classify_unmatched_prediction(
                p, gt_boxes, gt_used, iou_thr=iou_thr, iou_bg_thr=iou_bg_thr
            )
            if outcome == "dupe":
                self.matrix[BG][pc] += 1
                self.stats["dupe"] += 1
            elif outcome == "cls_confusion" and confused_gt is not None:
                gc = int(gt_boxes[confused_gt]["category_id"])
                self.matrix[gc][pc] += 1
                self.stats["cls_confusion"] += 1
            else:
                self.matrix[BG][pc] += 1
                self.stats["fp"] += 1

        for i, g in enumerate(gt_boxes):
            if not gt_used[i]:
                gc = int(g["category_id"])
                if gc in (0, 1):
                    self.matrix[gc][BG] += 1
                    self.stats["fn"] += 1

        self.n_images += 1

    def to_payload(
        self,
        *,
        conf_thr: float,
        iou_thr: float,
        split_role: str | None = None,
        weights: str | None = None,
        export_conf: float | None = None,
        export_iou: float | None = None,
        export_device: str | None = None,
        runtime_s: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": DETECTION_CONFUSION_MATRIX_V1,
            "labels": list(CLASS_LABELS),
            "matrix": self.matrix,
            "stats": dict(self.stats),
            "match": {"conf": float(conf_thr), "iou": float(iou_thr)},
            "n_images": self.n_images,
            "gt_instances": {
                "developed": sum(self.matrix[0]),
                "aborted": sum(self.matrix[1]),
            },
            "row_normalized": _row_normalized(self.matrix),
        }
        if split_role:
            payload["split_role"] = split_role
        if weights:
            payload["weights"] = weights
        if export_conf is not None:
            payload["export_conf"] = float(export_conf)
        if export_iou is not None:
            payload["export_iou"] = float(export_iou)
        if export_device is not None:
            payload["export_device"] = export_device
        if runtime_s is not None:
            payload["runtime_s"] = float(runtime_s)
        return payload


def _row_normalized(matrix: list[list[int]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    keys = ("pred_0", "pred_1", "pred_bg")
    for ri, label in enumerate(CLASS_LABELS[:2]):
        row_sum = sum(matrix[ri]) or 1
        out[label] = {
            keys[ci]: 100.0 * matrix[ri][ci] / row_sum for ci in range(3)
        }
    return out


def confusion_matrix_from_exports(
    gt: dict[str, Any],
    preds: dict[str, Any],
    *,
    conf_thr: float = 0.15,
    iou_thr: float = 0.5,
    iou_bg_thr: float = 0.1,
) -> ConfusionMatrixAccumulator:
    acc = ConfusionMatrixAccumulator()
    gt_by_img = _index_records_by_image_id(gt)
    preds_by_img = _index_records_by_image_id(preds)
    for img_id in sorted(set(gt_by_img) | set(preds_by_img)):
        gt_rec = gt_by_img.get(img_id, {"annotations": []})
        pr_rec = preds_by_img.get(img_id, {"detections": []})
        acc.update_image(
            list(gt_rec.get("annotations") or []),
            list(pr_rec.get("detections") or []),
            conf_thr=conf_thr,
            iou_thr=iou_thr,
            iou_bg_thr=iou_bg_thr,
        )
    return acc


# Backward-compatible alias
build_matrix_from_exports = confusion_matrix_from_exports


def resolve_match_settings(
    *,
    conf: float,
    iou: float,
    locked_conf_from: str | None = None,
) -> tuple[float, float]:
    """Operating-point conf/IoU for confusion matrix (defaults to HSP val lock when set)."""
    locked = (locked_conf_from or "").strip()
    if not locked:
        return float(conf), float(iou)
    from harchoc.domain_count_mae import match_settings_from_threshold_json
    from harchoc.threshold_lock import load_locked_conf

    match_iou, _category_aware = match_settings_from_threshold_json(locked)
    return float(load_locked_conf(locked)), float(match_iou)


def format_confusion_matrix_text(
    matrix: list[list[int]],
    stats: dict[str, int],
    *,
    title: str = "Detection confusion matrix",
) -> str:
    lines = [
        title,
        f"{'GT / Pred':16} {'pred 0':>12} {'pred 1':>12} {'pred bg':>12}",
    ]
    for ri, label in enumerate(CLASS_LABELS):
        lines.append(
            f"{label:16} {matrix[ri][0]:12,} {matrix[ri][1]:12,} {matrix[ri][2]:12,}"
        )
    lines.append(f"GT instances: developed={sum(matrix[0]):,}  aborted={sum(matrix[1]):,}")
    lines.append("Row-normalized (% of each GT class):")
    for ri, label in enumerate(CLASS_LABELS[:2]):
        row_sum = sum(matrix[ri]) or 1
        lines.append(
            f"  {label}: pred0={100 * matrix[ri][0] / row_sum:.1f}%  "
            f"pred1={100 * matrix[ri][1] / row_sum:.1f}%  "
            f"pred_bg={100 * matrix[ri][2] / row_sum:.1f}%"
        )
    bg_fps = matrix[BG][0] + matrix[BG][1]
    lines.append(f"Background-row detections (FP + dupe): {bg_fps:,}")
    lines.append(f"Matcher stats: {stats}")
    return "\n".join(lines)


def split_path_for_role(repo_root: Path, split_role: str) -> Path:
    return (repo_root / "data" / "splits" / f"{split_role.strip()}.txt").resolve()


def confusion_matrix_out_path(out_prefix: Path, split_role: str) -> Path:
    """``{prefix}_{split}_confusion.json`` when *out_prefix* has no ``.json`` suffix."""
    p = out_prefix
    if p.suffix.lower() == ".json":
        stem = p.with_suffix("")
        return stem.parent / f"{stem.name}_{split_role}_confusion.json"
    return p.parent / f"{p.name}_{split_role}_confusion.json"


def parse_confusion_matrix_splits(raw: str, *, repo_root: Path) -> dict[str, Path]:
    """Parse comma-separated split roles or JSON map of role→split file."""
    text = (raw or "").strip()
    if not text:
        return {}
    if text.startswith("{"):
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("confusion-matrix-splits JSON must be an object")
        return {str(k): Path(v).expanduser().resolve() for k, v in obj.items()}
    out: dict[str, Path] = {}
    for part in text.split(","):
        role = part.strip()
        if not role:
            continue
        out[role] = split_path_for_role(repo_root, role)
    return out


def confusion_matrix_streaming(
    *,
    weights: Path,
    split_file: Path,
    dataset_root: Path,
    conf_thr: float,
    iou_thr: float,
    export_conf: float = EXPORT_CONF,
    export_iou: float = EXPORT_IOU,
    max_det: int = DEFAULT_EXPORT_MAX_DET,
    device: str | int | None = None,
    imgsz: int | None = None,
    iou_bg_thr: float = 0.1,
    model: Any | None = None,
    strict_warnings: StrictWarnings | None = None,
) -> tuple[ConfusionMatrixAccumulator, float]:
    """
    One pass: GT labels from disk, predict, accumulate matrix (no preds JSON).

    Pass a pre-loaded *model* to avoid reloading weights (multi-split runs).
    """
    from harchoc.data_yaml import labels_path_for_image
    from harchoc.eval_export import (
        iter_split_image_paths,
        load_gt_annotations,
        read_image_size,
        ultralytics_results_to_detections,
    )

    from ultralytics import YOLO  # type: ignore

    owned_model = model is None
    if owned_model:
        model = YOLO(str(weights))
    if model is None:
        raise RuntimeError("detection model is required")

    acc = ConfusionMatrixAccumulator()
    device_resolved = _default_streaming_device(device)
    batch_size = _predict_batch_size()
    predict_kwargs: dict[str, Any] = {
        "conf": float(export_conf),
        "iou": float(export_iou),
        "max_det": int(max_det),
        "verbose": False,
    }
    predict_kwargs["device"] = device_resolved
    if imgsz is not None:
        predict_kwargs["imgsz"] = int(imgsz)

    entries = [
        (img_id, img_path, file_name)
        for img_id, img_path, file_name in iter_split_image_paths(split_file, dataset_root=dataset_root)
        if img_path.is_file()
    ]

    t0 = time.perf_counter()
    for chunk_start in range(0, len(entries), batch_size):
        chunk = entries[chunk_start : chunk_start + batch_size]
        paths = [str(item[1]) for item in chunk]
        predict_kwargs["batch"] = len(paths)
        res_list = model.predict(paths, **predict_kwargs)
        if not isinstance(res_list, list):
            res_list = [res_list]
        for (_img_id, img_path, _file_name), res in zip(chunk, res_list):
            w, h = read_image_size(img_path, strict_warnings=strict_warnings)
            label_path = labels_path_for_image(dataset_root=dataset_root, image_path=img_path)
            gt_anns = load_gt_annotations(label_path=label_path, img_w=w, img_h=h)
            dets = ultralytics_results_to_detections(res, strict_warnings=strict_warnings)
            acc.update_image(
                gt_anns,
                dets,
                conf_thr=float(conf_thr),
                iou_thr=float(iou_thr),
                iou_bg_thr=float(iou_bg_thr),
            )

    return acc, time.perf_counter() - t0


def confusion_matrix_multi_split(
    *,
    weights: Path,
    splits: dict[str, Path],
    dataset_root: Path,
    conf_thr: float,
    iou_thr: float,
    export_conf: float = EXPORT_CONF,
    export_iou: float = EXPORT_IOU,
    max_det: int = DEFAULT_EXPORT_MAX_DET,
    device: str | int | None = None,
    imgsz: int | None = None,
    iou_bg_thr: float = 0.1,
    strict_warnings: StrictWarnings | None = None,
) -> dict[str, tuple[ConfusionMatrixAccumulator, float]]:
    """Load YOLO once; accumulate a confusion matrix per split name."""
    from ultralytics import YOLO  # type: ignore

    model = YOLO(str(weights))
    results: dict[str, tuple[ConfusionMatrixAccumulator, float]] = {}
    for split_role, split_file in splits.items():
        acc, runtime_s = confusion_matrix_streaming(
            weights=weights,
            split_file=split_file,
            dataset_root=dataset_root,
            conf_thr=conf_thr,
            iou_thr=iou_thr,
            export_conf=export_conf,
            export_iou=export_iou,
            max_det=max_det,
            device=device,
            imgsz=imgsz,
            iou_bg_thr=iou_bg_thr,
            model=model,
            strict_warnings=strict_warnings,
        )
        results[split_role] = (acc, runtime_s)
    return results


# Backward-compatible alias for eval.py call sites during migration
run_streaming_confusion_matrix = confusion_matrix_streaming
