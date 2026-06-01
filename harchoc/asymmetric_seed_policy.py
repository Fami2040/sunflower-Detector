"""Load and validate ``asymmetric_seed_policy.v1`` eval policy JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harchoc.schemas import require_schema_version
from harchoc.sunflower_dataset import CLASS_ID_ABORTED, CLASS_ID_DEVELOPED, CLASS_NAMES_DICT

ASYMMETRIC_SEED_POLICY_V1 = "asymmetric_seed_policy.v1"


def validate_asymmetric_seed_policy(payload: dict[str, Any]) -> None:
    require_schema_version(payload, expected=ASYMMETRIC_SEED_POLICY_V1)

    classes = payload.get("classes")
    if not isinstance(classes, dict):
        raise ValueError("classes must be a dict")
    for cid in (str(CLASS_ID_DEVELOPED), str(CLASS_ID_ABORTED)):
        if classes.get(cid) != CLASS_NAMES_DICT[int(cid)]:
            raise ValueError(f"classes[{cid!r}] must be {CLASS_NAMES_DICT[int(cid)]!r}")

    prevalence = payload.get("prevalence")
    if not isinstance(prevalence, dict):
        raise ValueError("prevalence must be a dict")
    splits = prevalence.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("prevalence.splits must be a dict")
    for split_name in ("train", "val", "test"):
        row = splits.get(split_name)
        if not isinstance(row, dict):
            raise ValueError(f"prevalence.splits[{split_name!r}] must be a dict")
        counts = row.get("class_counts")
        if not isinstance(counts, dict):
            raise ValueError(f"prevalence.splits[{split_name!r}].class_counts must be a dict")
        for cid in ("0", "1"):
            if cid not in counts:
                raise ValueError(f"prevalence.splits[{split_name!r}].class_counts missing {cid!r}")
        dev_frac = row.get("developed_fraction")
        abort_frac = row.get("aborted_fraction")
        if not isinstance(dev_frac, (int, float)) or not isinstance(abort_frac, (int, float)):
            raise ValueError(f"prevalence.splits[{split_name!r}] missing developed/aborted fractions")
        total = int(counts["0"]) + int(counts["1"])
        if total <= 0:
            raise ValueError(f"prevalence.splits[{split_name!r}] class_counts sum to zero")
        expected_dev = int(counts["0"]) / float(total)
        if abs(float(dev_frac) - expected_dev) > 0.001:
            raise ValueError(
                f"prevalence.splits[{split_name!r}].developed_fraction inconsistent with class_counts"
            )

    eval_policy = payload.get("eval_policy")
    if not isinstance(eval_policy, dict):
        raise ValueError("eval_policy must be a dict")
    if eval_policy.get("primary_split") != "test":
        raise ValueError("eval_policy.primary_split must be 'test'")
    split_file = eval_policy.get("split_file")
    if not isinstance(split_file, str) or not split_file.strip():
        raise ValueError("eval_policy.split_file must be a non-empty string")
    metrics = eval_policy.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("eval_policy.metrics must be a dict")
    if metrics.get("primary") != "total_count_mae":
        raise ValueError("eval_policy.metrics.primary must be 'total_count_mae'")


def load_asymmetric_seed_policy(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    obj = json.loads(p.read_text(encoding="utf-8"))
    validate_asymmetric_seed_policy(obj)
    return obj
