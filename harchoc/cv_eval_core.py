from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from harchoc.stats_ci import ci_for_values


def kfold_assign(items: list[str], *, folds: int, seed: int) -> list[list[str]]:
    if folds < 2:
        raise ValueError("folds must be >= 2")
    rnd = random.Random(int(seed))
    shuffled = list(items)
    rnd.shuffle(shuffled)
    out: list[list[str]] = [[] for _ in range(folds)]
    for i, item in enumerate(shuffled):
        out[i % folds].append(item)
    return out


def load_fold_metric(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object in {path}")
    return obj


def _extract_scalar(doc: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for k in keys:
        if k in doc and doc[k] is not None:
            try:
                return float(doc[k])
            except (TypeError, ValueError):
                continue
        parts = k.split(".")
        cur: Any = doc
        ok = True
        for p in parts:
            if not isinstance(cur, dict) or p not in cur:
                ok = False
                break
            cur = cur[p]
        if ok and cur is not None:
            try:
                return float(cur)
            except (TypeError, ValueError):
                continue
    return None


def aggregate_fold_metrics(
  fold_docs: list[dict[str, Any]],
) -> dict[str, Any]:
    map50 = []
    map5095 = []
    mae = []
    for d in fold_docs:
        v = _extract_scalar(d, ("mAP50", "metrics.mAP50", "test_eval.mAP50"))
        if v is not None:
            map50.append(v)
        v = _extract_scalar(d, ("mAP50_95", "metrics.mAP50_95", "test_eval.mAP50_95"))
        if v is not None:
            map5095.append(v)
        v = _extract_scalar(
            d,
            (
                "counting_metrics.mae",
                "mae",
                "count_mae",
            ),
        )
        if v is not None:
            mae.append(v)

    out: dict[str, Any] = {"n_folds": len(fold_docs)}
    if map50:
        ci = ci_for_values(map50)
        out["mAP50"] = {"values": map50, "ci": ci.to_json() if ci else None}
    if map5095:
        ci = ci_for_values(map5095)
        out["mAP50_95"] = {"values": map5095, "ci": ci.to_json() if ci else None}
    if mae:
        ci = ci_for_values(mae)
        out["count_mae"] = {"values": mae, "ci": ci.to_json() if ci else None}
    return out
