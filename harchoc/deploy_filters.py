"""Shared SAHI/object-prediction post-filters for deploy paths (bot + run_infer_once)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class DeployFilterConfig:
    conf_thr_fertilized: float = 0.06
    conf_thr_unfertilized: float = 0.04
    unfert_dedup: bool = True
    unfert_dedup_center_ratio: float = 1.4
    unfert_dedup_min_pix: float = 2.0
    unfert_vs_fert_suppress: bool = True
    unfert_vs_fert_iou: float = 0.99
    unfert_tip_on_seed_suppress: bool = True
    unfert_fert_area_ratio_min: float = 1.35
    unfert_fert_expand_px: float = 4.0

    @classmethod
    def with_uniform_conf(cls, conf: float) -> DeployFilterConfig:
        base = cls.from_env()
        return DeployFilterConfig(
            conf_thr_fertilized=float(conf),
            conf_thr_unfertilized=float(conf),
            unfert_dedup=base.unfert_dedup,
            unfert_dedup_center_ratio=base.unfert_dedup_center_ratio,
            unfert_dedup_min_pix=base.unfert_dedup_min_pix,
            unfert_vs_fert_suppress=base.unfert_vs_fert_suppress,
            unfert_vs_fert_iou=base.unfert_vs_fert_iou,
            unfert_tip_on_seed_suppress=base.unfert_tip_on_seed_suppress,
            unfert_fert_area_ratio_min=base.unfert_fert_area_ratio_min,
            unfert_fert_expand_px=base.unfert_fert_expand_px,
        )

    @classmethod
    def from_locked_env(cls) -> DeployFilterConfig | None:
        """Optional manuscript locked conf via HARCHOC_LOCKED_CONF or HARCHOC_LOCKED_CONF_JSON."""
        raw = os.getenv("HARCHOC_LOCKED_CONF", "").strip()
        if raw:
            return cls.with_uniform_conf(float(raw))
        json_path = os.getenv("HARCHOC_LOCKED_CONF_JSON", "").strip()
        if json_path:
            from harchoc.threshold_lock import load_locked_conf

            return cls.with_uniform_conf(load_locked_conf(json_path))
        return None

    @classmethod
    def resolve(cls) -> DeployFilterConfig:
        locked = cls.from_locked_env()
        if locked is not None:
            return locked
        return cls.from_env()

    @classmethod
    def from_env(cls) -> DeployFilterConfig:
        legacy = os.getenv("CONF_THR", "").strip()
        conf_fert = float(
            os.getenv("CONF_THR_FERTILIZED", legacy if legacy else "0.06")
        )
        conf_unfert = float(
            os.getenv("CONF_THR_UNFERTILIZED", legacy if legacy else "0.04")
        )
        return cls(
            conf_thr_fertilized=conf_fert,
            conf_thr_unfertilized=conf_unfert,
            unfert_dedup=os.getenv("UNFERT_DEDUP", "true").lower() == "true",
            unfert_dedup_center_ratio=float(os.getenv("UNFERT_DEDUP_CENTER_RATIO", "1.4")),
            unfert_dedup_min_pix=float(os.getenv("UNFERT_DEDUP_MIN_PIX", "2.0")),
            unfert_vs_fert_suppress=os.getenv("UNFERT_VS_FERT_SUPPRESS", "true").lower() == "true",
            unfert_vs_fert_iou=float(os.getenv("UNFERT_VS_FERT_IOU", "0.99")),
            unfert_tip_on_seed_suppress=os.getenv("UNFERT_TIP_ON_SEED_SUPPRESS", "true").lower()
            == "true",
            unfert_fert_area_ratio_min=float(os.getenv("UNFERT_FERT_AREA_RATIO_MIN", "1.35")),
            unfert_fert_expand_px=float(os.getenv("UNFERT_FERT_EXPAND_PX", "4")),
        )


def _bbox_iou(pred_a: Any, pred_b: Any) -> float:
    a, b = pred_a.bbox, pred_b.bbox
    ax1, ay1, ax2, ay2 = float(a.minx), float(a.miny), float(a.maxx), float(a.maxy)
    bx1, by1, bx2, by2 = float(b.minx), float(b.miny), float(b.maxx), float(b.maxy)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def _bbox_area(pred: Any) -> float:
    b = pred.bbox
    return max(
        1e-6,
        (float(b.maxx) - float(b.minx)) * (float(b.maxy) - float(b.miny)),
    )


def _unfert_center_inside_fert(unfert_pred: Any, fert_pred: Any, *, expand_px: float = 0.0) -> bool:
    u, f = unfert_pred.bbox, fert_pred.bbox
    cx = (float(u.minx) + float(u.maxx)) * 0.5
    cy = (float(u.miny) + float(u.maxy)) * 0.5
    ex = float(expand_px)
    return (float(f.minx) - ex) <= cx <= (float(f.maxx) + ex) and (float(f.miny) - ex) <= cy <= (
        float(f.maxy) + ex
    )


def _suppress_unfert_vs_fert(unfert_pred: Any, fert_pred: Any, cfg: DeployFilterConfig) -> bool:
    if _bbox_iou(unfert_pred, fert_pred) >= cfg.unfert_vs_fert_iou:
        return True
    if cfg.unfert_tip_on_seed_suppress:
        af = _bbox_area(fert_pred)
        au = _bbox_area(unfert_pred)
        if af >= au * cfg.unfert_fert_area_ratio_min and _unfert_center_inside_fert(
            unfert_pred, fert_pred, expand_px=cfg.unfert_fert_expand_px
        ):
            return True
    return False


def filter_object_predictions(
    preds: Sequence[Any],
    cfg: DeployFilterConfig | None = None,
) -> list[Any]:
    """Per-class threshold, suppress unfert vs fert overlap, then unfert de-dup."""
    cfg = cfg or DeployFilterConfig.from_env()
    fert_list: list[Any] = []
    unfert_candidates: list[Any] = []

    for p in preds:
        cls_id = int(p.category.id)
        score = float(p.score.value)
        thr = cfg.conf_thr_fertilized if cls_id == 0 else cfg.conf_thr_unfertilized
        if score < thr:
            continue
        if cls_id == 0:
            fert_list.append(p)
        else:
            unfert_candidates.append(p)

    if cfg.unfert_vs_fert_suppress and fert_list and unfert_candidates:
        kept_unfert: list[Any] = []
        for u in unfert_candidates:
            if not any(_suppress_unfert_vs_fert(u, f, cfg) for f in fert_list):
                kept_unfert.append(u)
        unfert_candidates = kept_unfert

    if not cfg.unfert_dedup or not unfert_candidates:
        return fert_list + unfert_candidates

    def _center_and_size(pred: Any) -> tuple[float, float, float, float]:
        b = pred.bbox
        x1, y1, x2, y2 = float(b.minx), float(b.miny), float(b.maxx), float(b.maxy)
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5, w, h

    deduped: list[Any] = []
    for p in sorted(unfert_candidates, key=lambda x: float(x.score.value), reverse=True):
        cx, cy, w, h = _center_and_size(p)
        is_dup = False
        for k in deduped:
            kx, ky, kw, kh = _center_and_size(k)
            scale = min(w, h, kw, kh)
            radius = max(cfg.unfert_dedup_min_pix, cfg.unfert_dedup_center_ratio * scale)
            if ((cx - kx) ** 2 + (cy - ky) ** 2) ** 0.5 <= radius:
                is_dup = True
                break
        if not is_dup:
            deduped.append(p)

    return fert_list + deduped
