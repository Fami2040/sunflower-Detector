from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class DriftAcceptanceConfig:
    """Thresholds for split-drift pairwise comparisons."""

    ks_pvalue_warn: float = 0.05
    ks_pvalue_fail: float = 0.01
    class_jsd_nats_warn: float = 0.1
    class_jsd_nats_fail: float = 0.25
    class_dist_l1_warn: float = 0.15
    class_dist_l1_fail: float = 0.35
    pairs: tuple[str, ...] = ("train_vs_val", "val_vs_test", "train_vs_test")

    @classmethod
    def from_json_file(cls, path: Path) -> DriftAcceptanceConfig:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("acceptance config must be a JSON object")
        kwargs: dict[str, Any] = {}
        for field in (
            "ks_pvalue_warn",
            "ks_pvalue_fail",
            "class_jsd_nats_warn",
            "class_jsd_nats_fail",
            "class_dist_l1_warn",
            "class_dist_l1_fail",
        ):
            if field in raw:
                kwargs[field] = float(raw[field])
        if "pairs" in raw and isinstance(raw["pairs"], list):
            kwargs["pairs"] = tuple(str(x) for x in raw["pairs"])
        return cls(**kwargs)

    def to_json(self) -> dict[str, Any]:
        return {
            "ks_pvalue_warn": self.ks_pvalue_warn,
            "ks_pvalue_fail": self.ks_pvalue_fail,
            "class_jsd_nats_warn": self.class_jsd_nats_warn,
            "class_jsd_nats_fail": self.class_jsd_nats_fail,
            "class_dist_l1_warn": self.class_dist_l1_warn,
            "class_dist_l1_fail": self.class_dist_l1_fail,
            "pairs": list(self.pairs),
        }


Severity = Literal["ok", "warn", "fail"]


def _ks_severity(pvalue: float | None, *, cfg: DriftAcceptanceConfig) -> Severity | None:
    if pvalue is None:
        return None
    if pvalue < cfg.ks_pvalue_fail:
        return "fail"
    if pvalue < cfg.ks_pvalue_warn:
        return "warn"
    return "ok"


def _metric_severity(
    value: float | None,
    *,
    warn: float,
    fail: float,
    higher_is_worse: bool = True,
) -> Severity | None:
    if value is None:
        return None
    v = float(value)
    if higher_is_worse:
        if v >= fail:
            return "fail"
        if v >= warn:
            return "warn"
        return "ok"
    if v <= fail:
        return "fail"
    if v <= warn:
        return "warn"
    return "ok"


def evaluate_pair(
    pair_name: str,
    comparison: dict[str, Any],
    *,
    cfg: DriftAcceptanceConfig,
) -> dict[str, Any]:
    reasons: list[dict[str, Any]] = []
    worst: Severity = "ok"

    def _bump(sev: Severity | None) -> None:
        nonlocal worst
        if sev == "fail":
            worst = "fail"
        elif sev == "warn" and worst != "fail":
            worst = "warn"

    labels = comparison.get("labels") if isinstance(comparison.get("labels"), dict) else {}
    images = comparison.get("images") if isinstance(comparison.get("images"), dict) else {}

    jsd = labels.get("class_jsd_nats")
    sev = _metric_severity(
        float(jsd) if jsd is not None else None,
        warn=cfg.class_jsd_nats_warn,
        fail=cfg.class_jsd_nats_fail,
    )
    if sev:
        reasons.append({"metric": "class_jsd_nats", "value": jsd, "severity": sev})
        _bump(sev)

    l1 = labels.get("class_dist_l1")
    sev = _metric_severity(
        float(l1) if l1 is not None else None,
        warn=cfg.class_dist_l1_warn,
        fail=cfg.class_dist_l1_fail,
    )
    if sev:
        reasons.append({"metric": "class_dist_l1", "value": l1, "severity": sev})
        _bump(sev)

    for key in ("width_ks", "height_ks"):
        ks = images.get(key) if isinstance(images.get(key), dict) else None
        if not ks or not ks.get("available"):
            continue
        pv = ks.get("pvalue")
        sev = _ks_severity(float(pv) if pv is not None else None, cfg=cfg)
        if sev:
            reasons.append({"metric": key, "pvalue": pv, "severity": sev})
            _bump(sev)

    bks = labels.get("boxes_per_image_ks")
    if isinstance(bks, dict) and bks.get("available"):
        pv = bks.get("pvalue")
        sev = _ks_severity(float(pv) if pv is not None else None, cfg=cfg)
        if sev:
            reasons.append({"metric": "boxes_per_image_ks", "pvalue": pv, "severity": sev})
            _bump(sev)

    return {"pair": pair_name, "status": worst, "reasons": reasons}


def evaluate_acceptance(
    report: dict[str, Any],
    *,
    cfg: DriftAcceptanceConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or DriftAcceptanceConfig()
    comps = report.get("comparisons") if isinstance(report.get("comparisons"), dict) else {}
    per_pair: dict[str, Any] = {}
    overall: Severity = "ok"
    for pair in cfg.pairs:
        comp = comps.get(pair)
        if not isinstance(comp, dict):
            continue
        ev = evaluate_pair(pair, comp, cfg=cfg)
        per_pair[pair] = ev
        if ev["status"] == "fail":
            overall = "fail"
        elif ev["status"] == "warn" and overall != "fail":
            overall = "warn"
    return {
        "status": overall,
        "config": cfg.to_json(),
        "pairs": per_pair,
    }
