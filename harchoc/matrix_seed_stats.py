from __future__ import annotations

import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from harchoc.dual_metric_report import extract_counting_metrics, extract_locked_counting_metrics
from harchoc.json_io import load_json_dict
from harchoc.schemas import with_schema_version

MATRIX_SEED_STATS_V1 = "matrix_seed_stats.v1"
BENCHMARK_MATRIX_SEED_STATS_V1 = "benchmark_matrix_seed_stats.v1"

RUN_ARTIFACT_KEYS: tuple[str, ...] = (
    "error_report",
    "error_test_report",
    "error_analysis_report",
    "threshold_locked",
    "threshold_test_locked",
)


def _model_key_from_run(run: dict[str, Any]) -> str:
    name = str(run.get("name") or run.get("model") or "")
    run_name = str(run.get("run_name") or "")
    stem = run_name
    m = re.match(r"^(.+)_e\d+_s\d+$", run_name)
    if m:
        stem = m.group(1)
    return name or stem or "unknown"


def _metric_value(run: dict[str, Any], key: str) -> float | None:
    if run.get(key) is not None:
        try:
            return float(run[key])
        except (TypeError, ValueError):
            pass
    te = run.get("test_eval")
    if isinstance(te, dict) and te.get(key) is not None:
        try:
            return float(te[key])
        except (TypeError, ValueError):
            pass
    return None


def parse_count_mae_json_arg(spec: str) -> tuple[str | None, str]:
    """
    Parse ``run_name=path`` or bare ``path`` from ``--count-mae-json``.

    Returns ``(run_name_or_none, path)``.
    """
    text = spec.strip()
    if "=" in text:
        left, right = text.split("=", 1)
        run_name = left.strip() or None
        path = right.strip()
        if not path:
            raise ValueError(f"invalid --count-mae-json (empty path): {spec!r}")
        return run_name, path
    if not text:
        raise ValueError(f"invalid --count-mae-json (empty): {spec!r}")
    return None, text


def parse_count_mae_json_args(specs: list[str]) -> dict[str, str]:
    """Build ``run_name -> path`` from repeatable CLI specs."""
    out: dict[str, str] = {}
    for spec in specs:
        run_name, path = parse_count_mae_json_arg(spec)
        if run_name:
            out[run_name] = path
        else:
            out.setdefault("__bare__", path)
    return out


def count_mae_from_doc(doc: dict[str, Any]) -> tuple[float | None, str | None]:
    """Read count MAE from error-analysis or threshold-locked JSON."""
    locked = extract_locked_counting_metrics(doc)
    if locked.get("mae") is not None:
        return float(locked["mae"]), "threshold_locked"
    error = extract_counting_metrics(doc)
    if error.get("mae") is not None:
        return float(error["mae"]), "error_report"
    raw = doc.get("count_mae")
    if raw is not None:
        try:
            return float(raw), "count_mae"
        except (TypeError, ValueError):
            pass
    return None, None


def load_count_mae_from_artifact(
    path: str | Path,
    *,
    repo_root: Path | None = None,
) -> tuple[float | None, str | None]:
    """Load MAE from ``error_*_report.json`` or ``threshold_*_locked.json`` when present."""
    p = Path(path).expanduser()
    if not p.is_absolute() and repo_root is not None:
        candidate = (repo_root / p).resolve()
        if candidate.is_file():
            p = candidate
    if not p.is_file():
        return None, None
    doc = load_json_dict(p)
    mae, source = count_mae_from_doc(doc)
    if mae is None:
        return None, None
    return mae, source


def _resolve_path(path: str, *, repo_root: Path | None) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute() and repo_root is not None:
        p = (repo_root / p).resolve()
    return p


def artifact_path_candidates_for_run(
    run: dict[str, Any],
    *,
    count_mae_paths: dict[str, str] | None = None,
) -> list[tuple[str, Path]]:
    """Ordered ``(kind, path)`` candidates for per-run count MAE artifacts."""
    rn = str(run.get("run_name") or "")
    out: list[tuple[str, Path]] = []
    if count_mae_paths and rn and rn in count_mae_paths:
        out.append(("cli", _resolve_path(count_mae_paths[rn], repo_root=None)))

    for key in RUN_ARTIFACT_KEYS:
        raw = run.get(key)
        if isinstance(raw, str) and raw.strip():
            out.append((key, _resolve_path(raw.strip(), repo_root=None)))

    artifacts = run.get("artifacts")
    if isinstance(artifacts, dict):
        for key in RUN_ARTIFACT_KEYS:
            raw = artifacts.get(key)
            if isinstance(raw, str) and raw.strip():
                out.append((f"artifacts.{key}", _resolve_path(raw.strip(), repo_root=None)))

    seen: set[Path] = set()
    deduped: list[tuple[str, Path]] = []
    for kind, path in out:
        if path in seen:
            continue
        seen.add(path)
        deduped.append((kind, path))
    return deduped


def resolve_count_mae_for_run(
    run: dict[str, Any],
    *,
    count_mae_paths: dict[str, str] | None = None,
    repo_root: Path | None = None,
) -> tuple[float | None, str | None, str | None]:
    """
    Return ``(mae, source, artifact_path)`` from the first readable candidate.
    """
    if count_mae_paths and repo_root is not None:
        rn = str(run.get("run_name") or "")
        if rn and rn in count_mae_paths:
            mae, source = load_count_mae_from_artifact(
                count_mae_paths[rn], repo_root=repo_root
            )
            if mae is not None:
                return mae, source, str(_resolve_path(count_mae_paths[rn], repo_root=repo_root))

    for kind, path in artifact_path_candidates_for_run(run, count_mae_paths=count_mae_paths):
        resolved = path if path.is_absolute() or repo_root is None else _resolve_path(str(path), repo_root=repo_root)
        mae, source = load_count_mae_from_artifact(resolved, repo_root=repo_root)
        if mae is not None:
            src = source or kind
            return mae, src, str(resolved)
    return None, None, None


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, None
    return mean, statistics.stdev(values)


def compare_runs_by_seed(
    train_doc: dict[str, Any],
    *,
    count_mae_paths: dict[str, str] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """
    Group benchmark_matrix_train runs by model key; compare mAP and count MAE across seeds.
    """
    runs = train_doc.get("runs")
    if not isinstance(runs, list):
        return {"status": "error", "reason": "missing runs list"}

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        if not isinstance(r, dict) or r.get("status") != "ok":
            continue
        by_model[_model_key_from_run(r)].append(r)

    all_count_mae: list[float] = []
    models_out: dict[str, Any] = {}
    for model, group in sorted(by_model.items()):
        seeds: list[dict[str, Any]] = []
        map50_vals: list[float] = []
        map5095_vals: list[float] = []
        count_mae_vals: list[float] = []
        for r in group:
            rn = str(r.get("run_name") or "")
            sm = re.search(r"_s(\d+)$", rn)
            seed = int(sm.group(1)) if sm else None
            m50 = _metric_value(r, "mAP50")
            m95 = _metric_value(r, "mAP50_95")
            count_mae, count_src, count_path = resolve_count_mae_for_run(
                r,
                count_mae_paths=count_mae_paths,
                repo_root=repo_root,
            )
            seed_row: dict[str, Any] = {
                "run_name": rn,
                "seed": seed,
                "mAP50": m50,
                "mAP50_95": m95,
                "count_mae": count_mae,
            }
            if count_src is not None:
                seed_row["count_mae_source"] = count_src
            if count_path is not None:
                seed_row["count_mae_artifact"] = count_path
            seeds.append(seed_row)
            if m50 is not None:
                map50_vals.append(m50)
            if m95 is not None:
                map5095_vals.append(m95)
            if count_mae is not None:
                count_mae_vals.append(count_mae)
                all_count_mae.append(count_mae)

        ks_block: dict[str, Any] | None = None
        if len(map50_vals) >= 2:
            try:
                from scipy.stats import ks_2samp  # type: ignore

                a, b = map50_vals[:1], map50_vals[1:]
                if len(b) >= 1:
                    r = ks_2samp(a, b)
                    ks_block = {
                        "metric": "mAP50",
                        "statistic": float(r.statistic),
                        "pvalue": float(r.pvalue),
                        "n_a": len(a),
                        "n_b": len(b),
                    }
            except ImportError:
                ks_block = {"available": False, "reason": "missing scipy"}

        c_mean, c_std = _mean_std(count_mae_vals)
        models_out[model] = {
            "n_runs": len(group),
            "seeds": seeds,
            "mAP50_values": map50_vals,
            "mAP50_95_values": map5095_vals,
            "ks_mAP50_first_vs_rest": ks_block,
            "count_mae_values": count_mae_vals,
            "n_count_mae": len(count_mae_vals),
            "count_mae_mean": c_mean,
            "count_mae_std": c_std,
        }

    global_mean, global_std = _mean_std(all_count_mae)
    return {
        "status": "ok",
        "n_models": len(models_out),
        "n_count_mae": len(all_count_mae),
        "count_mae_mean": global_mean,
        "count_mae_std": global_std,
        "models": models_out,
    }


def build_matrix_seed_stats_v1(
    train_doc: dict[str, Any],
    *,
    source_train_out: str | Path,
    count_mae_paths: dict[str, str] | None = None,
    repo_root: Path | None = None,
    schema_version: str = MATRIX_SEED_STATS_V1,
) -> dict[str, Any]:
    stats = compare_runs_by_seed(
        train_doc,
        count_mae_paths=count_mae_paths,
        repo_root=repo_root,
    )
    return with_schema_version(
        {
            "source_train_out": str(source_train_out),
            **stats,
        },
        schema_version=schema_version,
    )


def build_dry_run_matrix_seed_stats_v1(
    *,
    out: str | Path,
    source_train_out: str | Path | None = None,
    train_doc: dict[str, Any] | None = None,
    count_mae_paths: dict[str, str] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Emit ``matrix_seed_stats.v1`` for ``--dry-run``.

    When ``train_doc`` is provided (e.g. test fixtures), compute MAE aggregates;
    otherwise top-level ``count_mae_mean`` / ``count_mae_std`` are ``null``.
    """
    if train_doc is not None and source_train_out is not None:
        payload = build_matrix_seed_stats_v1(
            train_doc,
            source_train_out=source_train_out,
            count_mae_paths=count_mae_paths,
            repo_root=repo_root,
        )
        payload["status"] = "dry-run"
        return payload

    return with_schema_version(
        {
            "status": "dry-run",
            "out": str(out),
            "source_train_out": str(source_train_out) if source_train_out else None,
            "n_models": 0,
            "n_count_mae": 0,
            "count_mae_mean": None,
            "count_mae_std": None,
            "models": {},
            "notes": "Provide --train-out (and optional --count-mae-json) to preview MAE aggregates.",
        },
        schema_version=MATRIX_SEED_STATS_V1,
    )
