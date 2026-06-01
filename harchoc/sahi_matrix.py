"""SAHI matrix eval protocol (plan scaffold; GPU eval not implemented yet)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harchoc.bench_config import BenchConfig
from harchoc.sahi_infer import SahiSliceConfig

SAHI_MATRIX_EVAL_V1 = "sahi_matrix_eval.v1"
SAHI_EVAL_NOT_IMPLEMENTED = "sahi_matrix_eval scaffold: GPU eval not implemented"


@dataclass(frozen=True)
class SahiEvalParams:
    slice_size: int
    overlap: float
    nms_iou: float
    conf_fertilized: float | None = None
    conf_unfertilized: float | None = None
    label: str | None = None

    def to_json(self) -> dict[str, object]:
        out: dict[str, object] = {
            "slice_size": self.slice_size,
            "overlap": self.overlap,
            "nms_iou": self.nms_iou,
        }
        if self.conf_fertilized is not None:
            out["conf_fertilized"] = self.conf_fertilized
        if self.conf_unfertilized is not None:
            out["conf_unfertilized"] = self.conf_unfertilized
        if self.label:
            out["label"] = self.label
        return out

    def to_slice_config(self) -> SahiSliceConfig:
        return SahiSliceConfig(
            slice_size=self.slice_size,
            overlap=self.overlap,
            nms_iou=self.nms_iou,
        )

    def run_suffix(self) -> str:
        if self.label and str(self.label).strip():
            safe = str(self.label).strip().replace(" ", "_")
            return f"sahi_{safe}"
        return f"sahi_{self.slice_size}_{self.overlap}_{self.nms_iou}"


def deploy_default_sahi_params() -> SahiEvalParams:
    """Deploy parity defaults (``run_infer_once.py`` / ``SahiSliceConfig.from_env()``)."""
    return SahiEvalParams(
        slice_size=500,
        overlap=0.35,
        nms_iou=0.50,
        conf_fertilized=0.06,
        conf_unfertilized=0.04,
        label="deploy_default",
    )


def _coerce_float(raw: object, *, field: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise SystemExit(f"Invalid SAHI {field}: expected number, got {raw!r}")
    return float(raw)


def _coerce_int(raw: object, *, field: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise SystemExit(f"Invalid SAHI {field}: expected int, got {raw!r}")
    return int(raw)


def parse_sahi_eval_params(raw: object) -> SahiEvalParams:
    if not isinstance(raw, dict):
        raise SystemExit(f"Invalid SAHI row: expected object, got {type(raw).__name__}")
    obj = {str(k): v for k, v in raw.items()}
    slice_size = _coerce_int(obj.get("slice_size", 500), field="slice_size")
    overlap = _coerce_float(obj.get("overlap", 0.35), field="overlap")
    nms_iou = _coerce_float(obj.get("nms_iou", 0.50), field="nms_iou")
    conf_fert_raw = obj.get("conf_fertilized", obj.get("conf_fert"))
    conf_unfert_raw = obj.get("conf_unfertilized", obj.get("conf_unfert"))
    conf_fert = _coerce_float(conf_fert_raw, field="conf_fertilized") if conf_fert_raw is not None else None
    conf_unfert = (
        _coerce_float(conf_unfert_raw, field="conf_unfertilized") if conf_unfert_raw is not None else None
    )
    label_raw = obj.get("label")
    label = str(label_raw).strip() if label_raw is not None and str(label_raw).strip() else None
    validate_sahi_eval_params(
        SahiEvalParams(slice_size, overlap, nms_iou, conf_fert, conf_unfert, label)
    )
    return SahiEvalParams(slice_size, overlap, nms_iou, conf_fert, conf_unfert, label)


def validate_sahi_eval_params(params: SahiEvalParams) -> None:
    if params.slice_size <= 0:
        raise SystemExit(f"SAHI slice_size must be > 0 (got {params.slice_size})")
    if not (0.0 <= params.overlap < 1.0):
        raise SystemExit(f"SAHI overlap must be in [0, 1) (got {params.overlap})")
    if not (0.0 < params.nms_iou <= 1.0):
        raise SystemExit(f"SAHI nms_iou must be in (0, 1] (got {params.nms_iou})")
    for name, val in (
        ("conf_fertilized", params.conf_fertilized),
        ("conf_unfertilized", params.conf_unfertilized),
    ):
        if val is not None and not (0.0 <= val <= 1.0):
            raise SystemExit(f"SAHI {name} must be in [0, 1] (got {val})")


def parse_sahi_rows_config(rows: object | None) -> list[SahiEvalParams]:
    if rows is None:
        return [deploy_default_sahi_params()]
    if not isinstance(rows, list) or not rows:
        raise SystemExit("sahi_rows must be a non-empty list of SAHI param objects")
    return [parse_sahi_eval_params(row) for row in rows]


def _flat_sahi_row_from_infer(infer: dict[str, object]) -> dict[str, object] | None:
    prefix = "sahi_"
    flat = {str(k)[len(prefix) :]: v for k, v in infer.items() if str(k).startswith(prefix)}
    if flat:
        return flat
    if infer.get("tiling") != "sahi":
        return None
    row: dict[str, object] = {}
    for key in ("slice_size", "overlap", "nms_iou", "conf_fertilized", "conf_unfert", "conf_unfertilized", "label"):
        if key in infer:
            row[key] = infer[key]
    return row or None


def sahi_params_from_infer(infer: dict[str, object]) -> SahiEvalParams | None:
    tiling = infer.get("tiling")
    sahi_raw = infer.get("sahi")
    if isinstance(sahi_raw, dict):
        return parse_sahi_eval_params(sahi_raw)
    flat = _flat_sahi_row_from_infer(infer)
    if flat is not None:
        return parse_sahi_eval_params(flat)
    if sahi_raw is None and tiling != "sahi":
        return None
    if sahi_raw is not None and not isinstance(sahi_raw, dict):
        raise SystemExit(f"Invalid infer.sahi: expected object, got {type(sahi_raw).__name__}")
    return deploy_default_sahi_params()


def resolve_sahi_rows_for_bench(
    cfg: BenchConfig,
    *,
    matrix_rows: list[SahiEvalParams] | None,
    sahi_eval: bool,
) -> list[SahiEvalParams]:
    per_bench = sahi_params_from_infer(cfg.infer)
    if per_bench is not None:
        return [per_bench]
    if not sahi_eval:
        return []
    return list(matrix_rows or [deploy_default_sahi_params()])


def expand_bench_configs_with_sahi(
    configs: list[BenchConfig],
    *,
    matrix_rows: list[SahiEvalParams] | None,
    sahi_eval: bool,
) -> list[tuple[BenchConfig, SahiEvalParams | None]]:
    expanded: list[tuple[BenchConfig, SahiEvalParams | None]] = []
    for cfg in configs:
        rows = resolve_sahi_rows_for_bench(cfg, matrix_rows=matrix_rows, sahi_eval=sahi_eval)
        if not rows:
            expanded.append((cfg, None))
            continue
        for row in rows:
            expanded.append((cfg, row))
    return expanded


def sahi_eval_status(*, dry_run: bool, would_eval: bool, sahi_eval: bool) -> tuple[str, str | None]:
    if not sahi_eval:
        return ("not_run" if dry_run or (not would_eval) else "planned", None)
    if dry_run or not would_eval:
        return ("not_run", SAHI_EVAL_NOT_IMPLEMENTED if not dry_run else None)
    return ("skipped", SAHI_EVAL_NOT_IMPLEMENTED)


def sahi_matrix_metadata(
    cfg: BenchConfig,
    sahi: SahiEvalParams | None,
) -> dict[str, object]:
    infer_tiling = cfg.infer.get("tiling")
    return {
        "eval_protocol": "sahi" if sahi is not None else "ultralytics_val",
        "tiling": str(infer_tiling) if infer_tiling is not None else None,
        "sahi": sahi.to_json() if sahi is not None else None,
    }
