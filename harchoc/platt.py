"""Detection score calibration: isotonic and Platt (sigmoid) paths.

**Isotonic** (``calibrate=isotonic``): ``sklearn.isotonic.IsotonicRegression`` when
sklearn is importable (full ``harchoc`` env); otherwise PAVA in ``harchoc.isotonic``
(CI-light runs without sklearn).

**Platt** (``calibrate=platt``): ``sklearn.linear_model.LogisticRegression`` on raw
scores (sigmoid / Platt scaling) when sklearn is available; otherwise
``scipy.optimize.minimize`` L-BFGS-B on negative log-likelihood (same ``PlattModel``
API, no new deps in minimal CI).

Callers: ``scripts/threshold_sweep.py`` via ``apply_calibration_to_preds`` only — do
not duplicate fit/apply logic in scripts.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Iterable

from harchoc.detection_match import _extract_boxes, _index_records_by_image_id, _iou_xyxy


@dataclass(frozen=True)
class PlattModel:
    a: float
    b: float

    def predict_one(self, score: float) -> float:
        z = self.a * float(score) + self.b
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def predict(self, scores: Iterable[float]) -> list[float]:
        return [self.predict_one(s) for s in scores]


def _fit_platt_model(
    *, scores: list[float], targets: list[float]
) -> tuple[PlattModel, dict[str, Any]]:
    """Prefer sklearn sigmoid fit in full env; scipy NLL fallback for CI-light runs."""
    xs = [float(s) for s in scores]
    ys = [float(t) for t in targets]
    if not xs:
        return PlattModel(a=1.0, b=0.0), {"calibrator": "identity"}
    if len(set(ys)) <= 1:
        return PlattModel(a=1.0, b=0.0), {"calibrator": "identity"}

    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore

        lr = LogisticRegression(solver="lbfgs", max_iter=1000)
        lr.fit([[x] for x in xs], ys)
        coef: Any = lr.coef_
        intercept: Any = lr.intercept_
        a = float(coef.reshape(-1)[0])
        b = float(intercept.reshape(-1)[0])
        return PlattModel(a=a, b=b), {"calibrator": "sklearn_platt"}
    except Exception:
        from scipy.optimize import minimize  # type: ignore

        def nll(params: list[float]) -> float:
            a, b = params
            loss = 0.0
            for x, y in zip(xs, ys, strict=False):
                p = PlattModel(a=a, b=b).predict_one(x)
                p = min(1.0 - 1e-12, max(1e-12, p))
                loss -= y * math.log(p) + (1.0 - y) * math.log(1.0 - p)
            return loss

        res = minimize(nll, x0=[1.0, 0.0], method="L-BFGS-B")
        a, b = float(res.x[0]), float(res.x[1])
        return PlattModel(a=a, b=b), {"calibrator": "scipy_platt"}


def fit_platt(*, scores: Iterable[float], targets: Iterable[float]) -> PlattModel:
    model, _meta = _fit_platt_model(
        scores=[float(s) for s in scores],
        targets=[float(t) for t in targets],
    )
    return model


def _fit_isotonic_scores(
    *, scores: list[float], targets: list[int]
) -> tuple[list[float], dict[str, Any]]:
    """Prefer sklearn isotonic in full env; PAVA fallback for CI-light runs."""
    xs = [float(s) for s in scores]
    ys = [float(t) for t in targets]
    try:
        from sklearn.isotonic import IsotonicRegression  # type: ignore

        reg = IsotonicRegression(out_of_bounds="clip")
        reg.fit(xs, ys)
        return [float(x) for x in reg.predict(xs)], {"calibrator": "sklearn_isotonic"}
    except Exception:
        from harchoc.isotonic import fit_isotonic_pava

        model = fit_isotonic_pava(scores=xs, targets=ys)
        return model.predict(xs), {"calibrator": "isotonic_pava"}


def collect_detection_calibration_pairs(
    *,
    gt: Any,
    preds: Any,
    iou_thr: float = 0.5,
    category_aware: bool = True,
) -> tuple[list[float], list[int]]:
    """Score + 0/1 label: pred matches a GT box at iou_thr (greedy per image)."""
    gt_by_img = _index_records_by_image_id(gt)
    preds_by_img = _index_records_by_image_id(preds)
    scores: list[float] = []
    labels: list[int] = []

    for img_id in sorted(set(gt_by_img.keys()) | set(preds_by_img.keys())):
        gt_rec = gt_by_img.get(img_id, {"annotations": []})
        pr_rec = preds_by_img.get(img_id, {"detections": []})
        gt_boxes = _extract_boxes(gt_rec, key="annotations")
        pr_boxes = _extract_boxes(pr_rec, key="detections")
        gt_used = [False] * len(gt_boxes)
        for p in sorted(pr_boxes, key=lambda x: float(x.get("score") or 0.0), reverse=True):
            sc = p.get("score")
            if sc is None:
                continue
            best_i, best_iou = -1, 0.0
            for i, g in enumerate(gt_boxes):
                if gt_used[i]:
                    continue
                if category_aware and int(p["category_id"]) != int(g["category_id"]):
                    continue
                iou = _iou_xyxy(p["bbox"], g["bbox"])
                if iou >= iou_thr and iou > best_iou:
                    best_i, best_iou = i, iou
            if best_i >= 0:
                gt_used[best_i] = True
                labels.append(1)
            else:
                labels.append(0)
            scores.append(float(sc))
    return scores, labels


def apply_calibration_to_preds(
    preds: Any,
    *,
    calibrate: str,
    gt: Any | None = None,
    iou_thr: float = 0.5,
    category_aware: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """
    Return a deep-copied preds object with calibrated detection scores.
    calibrate: none | isotonic | platt
    """
    mode = (calibrate or "none").strip().lower()
    meta: dict[str, Any] = {"mode": mode}
    if mode == "none":
        return preds, meta
    if gt is None:
        raise ValueError("calibration requires gt")

    scores, labels = collect_detection_calibration_pairs(
        gt=gt, preds=preds, iou_thr=iou_thr, category_aware=category_aware
    )
    meta["n_pairs"] = len(scores)
    if not scores:
        return copy.deepcopy(preds), {**meta, "note": "no scores to calibrate"}

    if mode == "isotonic":
        new_scores, iso_meta = _fit_isotonic_scores(scores=scores, targets=labels)
        meta.update(iso_meta)
    elif mode == "platt":
        model, platt_meta = _fit_platt_model(scores=scores, targets=[float(x) for x in labels])
        meta.update(platt_meta)
        new_scores = model.predict(scores)
    else:
        raise ValueError(f"unknown calibrate mode: {calibrate}")

    out = copy.deepcopy(preds)
    idx = 0
    gt_by_img = _index_records_by_image_id(gt)
    preds_by_img = _index_records_by_image_id(out)
    for img_id in sorted(set(gt_by_img.keys()) | set(preds_by_img.keys())):
        rec = preds_by_img.get(img_id, {"detections": []})
        dets = rec.get("detections") or []
        if not isinstance(dets, list):
            continue
        for d in sorted(dets, key=lambda x: float(x.get("score") or 0.0), reverse=True):
            if not isinstance(d, dict) or d.get("score") is None:
                continue
            d["score"] = float(new_scores[idx])
            idx += 1
    return out, meta


def apply_score_map_to_preds(preds: Any, score_map: list[float]) -> Any:
    """Apply precomputed scores in detection iteration order."""
    out = copy.deepcopy(preds)
    i = 0
    for rec in _index_records_by_image_id(out).values():
        dets = rec.get("detections") or []
        if not isinstance(dets, list):
            continue
        for d in dets:
            if isinstance(d, dict) and d.get("score") is not None:
                d["score"] = float(score_map[i])
                i += 1
    return out
