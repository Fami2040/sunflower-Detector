from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Sunflower CVAT-1093 frozen splits (describe_split on full dataset, 2026-05).
SUNFLOWER_DOCUMENTED_PEAK_GT_BOXES_PER_IMAGE = 1015

# Ultralytics RT-DETR decoder default unless overridden in train kwargs.
ULTRALYTICS_RTDETR_DEFAULT_NUM_QUERIES = 300


def is_rtdetr_model(model: str | None) -> bool:
    if not model or not str(model).strip():
        return False
    stem = Path(str(model).strip()).stem.lower()
    return "rtdetr" in stem or stem.startswith("rt-detr")


def _coerce_int(v: object, *, field: str, path: str) -> int:
    try:
        return int(v)  # type: ignore[arg-type]
    except Exception as exc:
        raise SystemExit(f"Invalid {field} in {path} (expected int, got {v!r})") from exc


def _env_peak_override() -> int | None:
    raw = os.getenv("HARCHOC_RTDETR_PEAK_GT_BOXES_PER_IMAGE", "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
    except Exception as exc:
        raise SystemExit(
            f"Invalid HARCHOC_RTDETR_PEAK_GT_BOXES_PER_IMAGE={raw!r} (expected int)"
        ) from exc
    if v <= 0:
        raise SystemExit(
            f"HARCHOC_RTDETR_PEAK_GT_BOXES_PER_IMAGE must be > 0 (got {raw!r})"
        )
    return v


def rtdetr_fields_from_train_json(raw: dict[str, Any], *, path: str) -> dict[str, int | bool]:
    """
    Read RT-DETR cap fields from a flat train_bench_*.json document.

    ``documented_peak_gt_boxes_per_image`` defaults to the repo constant (1015).
    ``num_queries`` defaults to Ultralytics RT-DETR (300).
    """
    num_queries = ULTRALYTICS_RTDETR_DEFAULT_NUM_QUERIES
    if raw.get("num_queries") is not None:
        num_queries = _coerce_int(raw["num_queries"], field="num_queries", path=path)

    peak_env = _env_peak_override()
    peak = peak_env if peak_env is not None else SUNFLOWER_DOCUMENTED_PEAK_GT_BOXES_PER_IMAGE
    if peak_env is None and raw.get("documented_peak_gt_boxes_per_image") is not None:
        peak = _coerce_int(
            raw["documented_peak_gt_boxes_per_image"],
            field="documented_peak_gt_boxes_per_image",
            path=path,
        )

    accept = bool(raw.get("accept_rtdetr_query_truncation"))
    return {
        "num_queries": num_queries,
        "documented_peak_gt_boxes_per_image": peak,
        "accept_rtdetr_query_truncation": accept,
    }


def rtdetr_query_cap_message(*, num_queries: int, peak_gt: int) -> str:
    return (
        f"RT-DETR num_queries={num_queries} is below documented peak GT boxes/image={peak_gt}; "
        "decoder query slots truncate dense trays. Set accept_rtdetr_query_truncation=true in the "
        "committed train_bench JSON after review, or raise num_queries in train kwargs."
    )


def rtdetr_eval_max_det(num_queries: int) -> int:
    """Eval/infer ``max_det`` for RT-DETR must match decoder query slots."""
    return int(num_queries)


def rtdetr_infer_max_det_mismatch_message(
    *,
    infer_max_det: int,
    num_queries: int,
    cfg_path: str,
) -> str:
    expected = rtdetr_eval_max_det(num_queries)
    return (
        f"RT-DETR infer.max_det={infer_max_det} must match num_queries={expected} "
        f"in {cfg_path} (decoder query slots cap predictions; YOLO max_det=3000 does not apply)."
    )


def validate_rtdetr_infer_max_det(
    *,
    model: str | None,
    infer_max_det: int | None,
    train_json: dict[str, Any] | None,
    train_json_path: str,
    cfg_path: str,
    fail: bool = True,
) -> list[str]:
    """Ensure bench ``infer.max_det`` equals RT-DETR ``num_queries`` from train JSON."""
    if not is_rtdetr_model(model):
        return []
    if infer_max_det is None:
        return []
    raw = train_json if isinstance(train_json, dict) else {}
    fields = rtdetr_fields_from_train_json(raw, path=train_json_path)
    expected = rtdetr_eval_max_det(int(fields["num_queries"]))
    if int(infer_max_det) == expected:
        return []
    msg = rtdetr_infer_max_det_mismatch_message(
        infer_max_det=int(infer_max_det),
        num_queries=expected,
        cfg_path=cfg_path,
    )
    if fail and not _warn_only_env():
        raise SystemExit(msg)
    return [msg]


def validate_rtdetr_query_cap(
    *,
    model: str | None,
    train_json: dict[str, Any] | None,
    train_json_path: str,
    fail: bool = True,
) -> list[str]:
    """
    Return warning strings when num_queries < documented peak GT boxes/image.
    When ``fail`` is true and truncation is not explicitly accepted, raise SystemExit.
    """
    if not is_rtdetr_model(model):
        return []
    raw = train_json if isinstance(train_json, dict) else {}
    fields = rtdetr_fields_from_train_json(raw, path=train_json_path)
    num_queries = int(fields["num_queries"])
    peak = int(fields["documented_peak_gt_boxes_per_image"])
    if num_queries >= peak:
        return []

    msg = rtdetr_query_cap_message(num_queries=num_queries, peak_gt=peak)
    if fields["accept_rtdetr_query_truncation"]:
        return [msg]

    if fail and not _warn_only_env():
        raise SystemExit(msg)
    return [msg]


def _warn_only_env() -> bool:
    v = os.getenv("HARCHOC_RTDETR_QUERY_CAP", "").strip().lower()
    return v in ("warn", "warning", "0", "false", "no")
