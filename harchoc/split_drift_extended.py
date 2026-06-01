"""Extended split-drift proxies: per-class means, bbox area quantiles, tray density."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from harchoc.domain_tags import tray_key_from_stem
from harchoc.label_stats import label_path_for_image
from harchoc.splits_io import read_split_list, resolve_split_entry


def _quantiles(
    xs: list[float], *, qs: tuple[float, ...] = (0.25, 0.5, 0.75)
) -> dict[str, float | int | None]:
    if not xs:
        return {"n": 0, "q25": None, "q50": None, "q75": None}
    ordered = sorted(xs)
    n = len(ordered)

    def _at(q: float) -> float:
        if n == 1:
            return float(ordered[0])
        idx = q * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)

    out: dict[str, float | int | None] = {"n": int(n)}
    for q in qs:
        key = f"q{int(round(q * 100))}"
        out[key] = _at(q)
    return out


def _int_stats(xs: list[int]) -> dict[str, float | int | None]:
    if not xs:
        return {"n": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "n": int(len(xs)),
        "min": int(min(xs)),
        "max": int(max(xs)),
        "mean": float(statistics.fmean(xs)),
        "median": float(statistics.median(xs)),
    }


def _parse_yolo_box_dims(label_path: Path) -> list[tuple[int, float, float]]:
    """Return (class_id, w_norm, h_norm) for each YOLO box line."""
    if not label_path.is_file():
        return []
    out: list[tuple[int, float, float]] = []
    for ln in label_path.read_text("utf-8", errors="replace").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        toks = s.split()
        if len(toks) < 5:
            continue
        try:
            cls = int(float(toks[0]))
            w = float(toks[3])
            h = float(toks[4])
        except (ValueError, TypeError):
            continue
        if w <= 0 or h <= 0:
            continue
        out.append((cls, w, h))
    return out


def _safe_ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or a == 0:
        return None
    return float(b) / float(a)


def _per_class_mean_ratios(
    a: dict[str, float], b: dict[str, float]
) -> dict[str, float | None]:
    keys = set(a) | set(b)
    return {k: _safe_ratio(a.get(k), b.get(k)) for k in sorted(keys, key=lambda x: int(x) if x.isdigit() else x)}


def _tray_jaccard(a_trays: dict[str, int], b_trays: dict[str, int]) -> float | None:
    sa = set(a_trays)
    sb = set(b_trays)
    if not sa and not sb:
        return None
    union = sa | sb
    if not union:
        return None
    return float(len(sa & sb)) / float(len(union))


def _load_catalog_tray_keys(catalog_path: Path | None) -> dict[str, Any] | None:
    if catalog_path is None or not catalog_path.is_file():
        return None
    try:
        from harchoc.domain_eval import tray_keys_from_catalog_blob

        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        keys = tray_keys_from_catalog_blob(raw)
        return {"catalog_path": str(catalog_path), "n_trays": len(keys), "tray_keys": keys[:20]}
    except Exception:
        return {"catalog_path": str(catalog_path), "available": False}


def collect_split_extended_stats(
    *,
    dataset_root: Path,
    split_list: list[str],
    limit: int = 0,
) -> dict[str, Any]:
    """
    Per-split extended proxies from split list image paths.

    Requires describe_split image readers (deferred import).
    """
    from scripts.describe_split import _infer_label_path, _read_image_size

    root = dataset_root.resolve()
    imgs = list(split_list)
    if limit > 0:
        imgs = imgs[: int(limit)]

    per_image_class_counts: list[dict[str, int]] = []
    bbox_areas_px: list[float] = []
    tray_image_counts: dict[str, int] = defaultdict(int)
    n_images = 0
    n_read_errors = 0

    for rel in imgs:
        rel_s = str(rel).strip()
        if not rel_s or rel_s.startswith("#"):
            continue
        try:
            img_path = resolve_split_entry(rel_s, dataset_root=root)
        except Exception:
            n_read_errors += 1
            continue
        if not img_path.is_file():
            n_read_errors += 1
            continue
        try:
            w, h = _read_image_size(img_path)
        except Exception:
            n_read_errors += 1
            continue

        n_images += 1
        tray_image_counts[tray_key_from_stem(img_path.stem)] += 1

        try:
            rel_to_root = img_path.resolve().relative_to(root)
            lbl = label_path_for_image(root, rel_to_root)
        except ValueError:
            lbl = _infer_label_path(dataset_root=root, image_path=img_path)

        cls_counts: dict[str, int] = defaultdict(int)
        for cls, bw, bh in _parse_yolo_box_dims(lbl):
            k = str(cls)
            cls_counts[k] = cls_counts.get(k, 0) + 1
            bbox_areas_px.append(float(bw) * float(bh) * float(w) * float(h))
        per_image_class_counts.append(dict(cls_counts))

    per_class_mean: dict[str, float] = {}
    if n_images > 0:
        totals: dict[str, int] = defaultdict(int)
        for row in per_image_class_counts:
            for k, v in row.items():
                totals[k] += int(v)
        per_class_mean = {k: float(v) / float(n_images) for k, v in sorted(totals.items(), key=lambda kv: int(kv[0]))}

    return {
        "n_images_scanned": int(n_images),
        "n_read_errors": int(n_read_errors),
        "per_class_boxes_per_image_mean": per_class_mean,
        "bbox_area_px_quantiles": _quantiles(bbox_areas_px),
        "images_per_tray": {
            "n_trays": int(len(tray_image_counts)),
            "per_tray_counts": _int_stats(list(tray_image_counts.values())),
            "tray_keys": sorted(tray_image_counts.keys()),
        },
    }


def extended_pairwise_comparison(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    a_cls = dict(a.get("per_class_boxes_per_image_mean") or {})
    b_cls = dict(b.get("per_class_boxes_per_image_mean") or {})
    a_q = a.get("bbox_area_px_quantiles") if isinstance(a.get("bbox_area_px_quantiles"), dict) else {}
    b_q = b.get("bbox_area_px_quantiles") if isinstance(b.get("bbox_area_px_quantiles"), dict) else {}
    a_tray = a.get("images_per_tray") if isinstance(a.get("images_per_tray"), dict) else {}
    b_tray = b.get("images_per_tray") if isinstance(b.get("images_per_tray"), dict) else {}
    a_pt = a_tray.get("per_tray_counts") if isinstance(a_tray.get("per_tray_counts"), dict) else {}
    b_pt = b_tray.get("per_tray_counts") if isinstance(b_tray.get("per_tray_counts"), dict) else {}

    a_keys = list(a_tray.get("tray_keys") or [])
    b_keys = list(b_tray.get("tray_keys") or [])

    return {
        "per_class_boxes_per_image_mean_ratio": _per_class_mean_ratios(
            {k: float(v) for k, v in a_cls.items()},
            {k: float(v) for k, v in b_cls.items()},
        ),
        "bbox_area_px_q50_ratio": _safe_ratio(a_q.get("q50"), b_q.get("q50")),
        "bbox_area_px_q25_ratio": _safe_ratio(a_q.get("q25"), b_q.get("q25")),
        "bbox_area_px_q75_ratio": _safe_ratio(a_q.get("q75"), b_q.get("q75")),
        "images_per_tray_mean_ratio": _safe_ratio(a_pt.get("mean"), b_pt.get("mean")),
        "tray_key_jaccard": _tray_jaccard({k: 1 for k in a_keys}, {k: 1 for k in b_keys}),
    }


def build_extended_drift_block(
    *,
    dataset_root: Path,
    split_lists: dict[str, list[str] | None],
    catalog_path: Path | None = None,
    scan_limit: int = 0,
) -> dict[str, Any]:
    """Build ``extended`` payload for split_drift reports."""
    per_split: dict[str, Any] = {}
    for name in ("train", "val", "test"):
        lst = split_lists.get(name)
        per_split[name] = collect_split_extended_stats(
            dataset_root=dataset_root,
            split_list=list(lst) if isinstance(lst, list) else [],
            limit=scan_limit,
        )

    comparisons: dict[str, Any] = {
        "train_vs_val": extended_pairwise_comparison(per_split["train"], per_split["val"]),
        "train_vs_test": extended_pairwise_comparison(per_split["train"], per_split["test"]),
        "val_vs_test": extended_pairwise_comparison(per_split["val"], per_split["test"]),
    }

    block: dict[str, Any] = {
        "per_split": per_split,
        "comparisons": comparisons,
    }
    catalog = _load_catalog_tray_keys(catalog_path)
    if catalog is not None:
        block["catalog"] = catalog
    return block
