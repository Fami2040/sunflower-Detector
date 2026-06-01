"""Export GT labels and model predictions to JSON for threshold_sweep / error_analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from harchoc.data_yaml import labels_path_for_image
from harchoc.hsp_export_protocol import (
    DEFAULT_EXPORT_MAX_DET,
    EXPORT_CONF,
    EXPORT_IOU,
)

if TYPE_CHECKING:
    from harchoc.strict_ml import StrictWarnings


def _image_id_from_path(path: Path) -> str:
    return path.stem


def iter_split_image_paths(
    split_file: Path, *, dataset_root: Path
) -> list[tuple[str, Path, str]]:
    """
    Return (image_id, absolute_image_path, file_name_for_json) per split line.
    file_name is repo-relative when possible, else absolute.
    """
    out: list[tuple[str, Path, str]] = []
    root = dataset_root.resolve()
    for ln in split_file.read_text("utf-8", errors="ignore").splitlines():
        rel = ln.strip()
        if not rel or rel.startswith("#"):
            continue
        p = Path(rel)
        if not p.is_absolute():
            p = root / p
        p = p.resolve()
        img_id = _image_id_from_path(p)
        try:
            file_name = str(p.relative_to(root))
        except ValueError:
            file_name = str(p)
        out.append((img_id, p, file_name))
    return out


def yolo_label_line_to_xyxy(
    line: str, *, img_w: int, img_h: int
) -> tuple[int, tuple[float, float, float, float]] | None:
    line = line.strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
    except ValueError:
        return None
    bw, bh = w * img_w, h * img_h
    x1 = cx * img_w - bw / 2.0
    y1 = cy * img_h - bh / 2.0
    x2 = x1 + bw
    y2 = y1 + bh
    return cls, (x1, y1, x2, y2)


def load_gt_annotations(
    *, label_path: Path, img_w: int, img_h: int
) -> list[dict[str, Any]]:
    if not label_path.is_file():
        return []
    anns: list[dict[str, Any]] = []
    for line in label_path.read_text("utf-8", errors="ignore").splitlines():
        parsed = yolo_label_line_to_xyxy(line, img_w=img_w, img_h=img_h)
        if parsed is None:
            continue
        cls, bbox = parsed
        anns.append({"bbox": list(bbox), "category_id": cls})
    return anns


def _read_image_size_header(img_path: Path) -> tuple[int, int] | None:
    """Best-effort PNG IHDR / JPEG SOF parse when PIL cannot open the file."""
    import struct

    suf = img_path.suffix.lower()
    if suf == ".png":
        with img_path.open("rb") as f:
            if f.read(8) != b"\x89PNG\r\n\x1a\n":
                return None
            f.read(4)
            if f.read(4) != b"IHDR":
                return None
            w = struct.unpack(">I", f.read(4))[0]
            h = struct.unpack(">I", f.read(4))[0]
            return int(w), int(h)
    if suf in {".jpg", ".jpeg"}:
        with img_path.open("rb") as f:
            if f.read(2) != b"\xff\xd8":
                return None
            while True:
                b = f.read(1)
                if not b:
                    break
                if b != b"\xff":
                    continue
                while True:
                    m = f.read(1)
                    if not m:
                        return None
                    if m != b"\xff":
                        break
                marker = m[0]
                if marker in {0xD9, 0xDA}:
                    break
                seglen_bytes = f.read(2)
                if len(seglen_bytes) != 2:
                    break
                seglen = struct.unpack(">H", seglen_bytes)[0]
                if seglen < 2:
                    break
                if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                    f.read(1)
                    h = struct.unpack(">H", f.read(2))[0]
                    w = struct.unpack(">H", f.read(2))[0]
                    return int(w), int(h)
                f.seek(seglen - 2, 1)
    return None


def read_image_size(
    img_path: Path,
    *,
    strict_warnings: StrictWarnings | None = None,
) -> tuple[int, int]:
    """
    Return (width, height) from PIL, else PNG/JPEG header parse, else (1, 1).

    On PIL failure: records ``pil_image_open_failed``; strict mode raises before
  fallback; otherwise tries header parse, then (1, 1) for finite YOLO xyxy.
    """
    from harchoc.strict_ml import capture_failure

    with capture_failure(f"open image {img_path}") as cap:
        from PIL import Image  # type: ignore

        with Image.open(img_path) as im:
            return int(im.size[0]), int(im.size[1])
    if not cap.failed:
        raise RuntimeError("read_image_size: unexpected capture_failure state")
    msg = cap.exc_msg or "unknown error"
    if strict_warnings is not None:
        strict_warnings.warn(
            "pil_image_open_failed",
            msg,
            path=str(img_path),
            exc_type=cap.exc_type,
            raise_if_strict=True,
        )
    if img_path.is_file():
        header = _read_image_size_header(img_path)
        if header is not None:
            return header
    return 1, 1


def build_gt_export(
    *,
    split_file: Path,
    dataset_root: Path,
    strict_warnings: StrictWarnings | None = None,
) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    for img_id, img_path, file_name in iter_split_image_paths(split_file, dataset_root=dataset_root):
        if not img_path.is_file():
            continue
        w, h = read_image_size(img_path, strict_warnings=strict_warnings)
        label_path = labels_path_for_image(dataset_root=dataset_root, image_path=img_path)
        images.append(
            {
                "image_id": img_id,
                "file_name": file_name,
                "annotations": load_gt_annotations(label_path=label_path, img_w=w, img_h=h),
            }
        )
    return {"images": images}


def _box_row_to_xyxy(row: Any) -> tuple[float, float, float, float]:
    try:
        vals = row.tolist()
    except Exception:
        vals = row
    coords = [float(v) for v in vals]
    if len(coords) < 4:
        raise ValueError(f"expected >=4 box coords, got {len(coords)}")
    return coords[0], coords[1], coords[2], coords[3]


def ultralytics_results_to_detections(
    result: Any,
    *,
    strict_warnings: StrictWarnings | None = None,
) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = getattr(boxes, "xyxy", None)
    cls = getattr(boxes, "cls", None)
    conf = getattr(boxes, "conf", None)
    if xyxy is None:
        return []
    detections: list[dict[str, Any]] = []
    n = len(xyxy)
    for i in range(n):
        row = xyxy[i]
        try:
            x1, y1, x2, y2 = _box_row_to_xyxy(row)
        except Exception as ex:
            if strict_warnings is not None:
                strict_warnings.warn(
                    "ultralytics_box_parse",
                    str(ex),
                    index=i,
                    raise_if_strict=False,
                )
            continue
        cat = int(cls[i].item()) if cls is not None else 0
        score = float(conf[i].item()) if conf is not None else None
        detections.append(
            {
                "bbox": [x1, y1, x2, y2],
                "category_id": cat,
                "score": score,
            }
        )
    return detections


def run_predict_export(
    *,
    model: Any,
    split_file: Path,
    dataset_root: Path,
    conf: float,
    iou: float,
    max_det: int,
    device: str | int | None = None,
    strict_warnings: StrictWarnings | None = None,
) -> dict[str, Any]:
    entries = iter_split_image_paths(split_file, dataset_root=dataset_root)
    if not entries:
        return {"images": []}

    predict_kwargs: dict[str, Any] = {
        "conf": float(conf),
        "iou": float(iou),
        "max_det": int(max_det),
        "verbose": False,
        "batch": 1,
    }
    if device is not None:
        predict_kwargs["device"] = device

    # One image per predict() call: passing the full split list in a single call
    # can OOM on 8 GiB GPUs (Ultralytics holds peak VRAM across the run).
    images: list[dict[str, Any]] = []
    for img_id, img_path, file_name in entries:
        res_list = model.predict(str(img_path), **predict_kwargs)
        res = res_list[0] if res_list else None
        dets = (
            ultralytics_results_to_detections(res, strict_warnings=strict_warnings)
            if res is not None
            else []
        )
        images.append(
            {
                "image_id": img_id,
                "file_name": file_name,
                "detections": dets,
            }
        )
    return {"images": images}


def export_gt_preds_json(
    *,
    split_file: Path,
    dataset_root: Path,
    weights: Path,
    gt_out: Path,
    preds_out: Path,
    conf: float = EXPORT_CONF,
    iou: float = EXPORT_IOU,
    max_det: int = DEFAULT_EXPORT_MAX_DET,
    device: str | int | None = None,
    strict_warnings: StrictWarnings | None = None,
) -> dict[str, Any]:
    """Write GT and prediction JSON files; return paths and image counts."""
    gt_obj = build_gt_export(
        split_file=split_file,
        dataset_root=dataset_root,
        strict_warnings=strict_warnings,
    )
    gt_out.parent.mkdir(parents=True, exist_ok=True)
    preds_out.parent.mkdir(parents=True, exist_ok=True)
    gt_out.write_text(
        __import__("json").dumps(gt_obj, indent=2) + "\n",
        encoding="utf-8",
    )

    from ultralytics import YOLO  # type: ignore

    model = YOLO(str(weights))
    preds_obj = run_predict_export(
        model=model,
        split_file=split_file,
        dataset_root=dataset_root,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
        strict_warnings=strict_warnings,
    )
    preds_out.write_text(
        __import__("json").dumps(preds_obj, indent=2) + "\n",
        encoding="utf-8",
    )
    out: dict[str, Any] = {
        "gt_json": str(gt_out.resolve()),
        "preds_json": str(preds_out.resolve()),
        "n_images": len(gt_obj.get("images") or []),
        "export_conf": conf,
        "export_iou": iou,
        "export_max_det": max_det,
        "export_device": device,
    }
    if strict_warnings is not None and strict_warnings.items:
        out["strict_warnings"] = strict_warnings.as_list()
    return out
