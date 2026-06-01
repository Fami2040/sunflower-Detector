"""Field contracts for ``error_analysis_summary.v1`` / ``error_analysis_report.v1``."""

from __future__ import annotations

from typing import Any

from harchoc.schemas import require_schema_version
from harchoc.tide_summary import FP_BUCKET_KEYS

ERROR_ANALYSIS_SUMMARY_V1 = "error_analysis_summary.v1"
ERROR_ANALYSIS_REPORT_V1 = "error_analysis_report.v1"

ERROR_ANALYSIS_SHARED_FIELDS: tuple[str, ...] = (
    "ambiguous_summary",
    "ambiguous_fp_crosstab",
    "tide_bucket_summary",
    "counting_metrics",
    "counting_metrics_excl_ambiguous_band",
    "error_taxonomy",
    "fp_breakdown",
)


def validate_ambiguous_fp_crosstab(obj: Any) -> None:
    """Cross-tab ambiguous detections × FP buckets (background/localization/cls/dupe)."""
    if not isinstance(obj, dict):
        raise TypeError("ambiguous_fp_crosstab must be a dict")
    band = obj.get("conf_band")
    if not isinstance(band, list) or len(band) < 2:
        raise ValueError("ambiguous_fp_crosstab.conf_band must be [lo, hi]")
    by_bucket = obj.get("by_bucket")
    if not isinstance(by_bucket, dict):
        raise ValueError("ambiguous_fp_crosstab.by_bucket must be a dict")
    for key in FP_BUCKET_KEYS:
        row = by_bucket.get(key)
        if not isinstance(row, dict):
            raise ValueError(f"ambiguous_fp_crosstab.by_bucket[{key!r}] must be a dict")
        for side in ("ambiguous", "not_ambiguous"):
            if side not in row:
                raise ValueError(
                    f"ambiguous_fp_crosstab.by_bucket[{key!r}] missing {side!r}"
                )
    totals = obj.get("totals")
    if not isinstance(totals, dict):
        raise ValueError("ambiguous_fp_crosstab.totals must be a dict")
    for key in ("n_predictions", "n_ambiguous", "n_ambiguous_among_fp_buckets"):
        if key not in totals:
            raise ValueError(f"ambiguous_fp_crosstab.totals missing {key!r}")


def validate_error_analysis_payload(
    payload: dict[str, Any],
    *,
    schema_version: str,
) -> None:
    """Light guard for committed error_analysis JSON shapes."""
    require_schema_version(payload, expected=schema_version)
    for field in ERROR_ANALYSIS_SHARED_FIELDS:
        if field not in payload:
            raise ValueError(f"missing required field {field!r}")
    validate_ambiguous_fp_crosstab(payload["ambiguous_fp_crosstab"])
