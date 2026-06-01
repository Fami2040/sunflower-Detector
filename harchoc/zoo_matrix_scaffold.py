"""Generate and validate zoo matrix bench YAML + train_bench JSON from matrix_rows manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harchoc.bench_config import load_bench_config
from harchoc.yaml_minimal import parse_minimal_yaml


MANIFEST_SCHEMA = "zoo_matrix_rows.v1"
DEFAULT_MANIFEST = "configs/zoo/matrix_rows.v1.json"


@dataclass
class ZooScaffoldReport:
    manifest_path: Path
    mode: str
    bench_generated: int = 0
    bench_updated: int = 0
    bench_skipped: int = 0
    train_generated: int = 0
    train_updated: int = 0
    train_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": "zoo_matrix_scaffold_report.v1",
            "manifest_path": str(self.manifest_path),
            "mode": self.mode,
            "bench_generated": self.bench_generated,
            "bench_updated": self.bench_updated,
            "bench_skipped": self.bench_skipped,
            "train_generated": self.train_generated,
            "train_updated": self.train_updated,
            "train_skipped": self.train_skipped,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_matrix_rows_manifest(path: Path | None = None) -> dict[str, Any]:
    p = path or (_repo_root() / DEFAULT_MANIFEST)
    p = p.resolve()
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Expected object in {p}")
    if raw.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"Expected schema_version={MANIFEST_SCHEMA!r} in {p}")
    rows = raw.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected non-empty rows[] in {p}")
    return raw


def _row_groups(row: dict[str, Any]) -> tuple[str, ...]:
    g = row.get("groups")
    if isinstance(g, list):
        return tuple(str(x).strip() for x in g if str(x).strip())
    rules = row.get("group_rules")
    if isinstance(rules, dict):
        out: list[str] = []
        for key in ("always", "zoo_core", "zoo_scale", "sota_2026", "sota_deim", "family_scales"):
            v = rules.get(key)
            if isinstance(v, list):
                out.extend(str(x).strip() for x in v if str(x).strip())
        return tuple(out)
    return ()


def _infer_block(row: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    from harchoc.config_coerce import child_dict

    base = child_dict(defaults, "infer")
    overlay = child_dict(row, "infer")
    out = {str(k): v for k, v in base.items()}
    out.update({str(k): v for k, v in overlay.items()})
    return out


def _bench_yaml_text(row: dict[str, Any], defaults: dict[str, Any]) -> str:
    bench_file = str(row["bench_file"])
    name = str(row.get("name") or f"bench_{Path(bench_file).stem}")
    backend = str(row["backend"])
    groups = ", ".join(_row_groups(row))
    infer = _infer_block(row, defaults)
    lines = [
        "include: _defaults.yaml",
        f"name: {name}",
        f"backend: {backend}",
    ]
    if backend == "ultralytics":
        lines.append(f"model: {row['model']}")
    elif backend == "external":
        sid = str(row.get("source_id") or row.get("model_id") or row["id"])
        lines.append(f"source_id: {sid}")
        lines.append(f"model_id: {str(row.get('model_id') or sid)}")
    elif backend == "supergradients":
        lines.append(f"model_id: {row['model_id']}")
    lines.append(f"groups: {groups}")
    lines.append("infer:")
    for k, v in infer.items():
        if k == "imgsz" and v == defaults.get("infer", {}).get("imgsz"):
            continue
        if v in ("none", None):
            lines.append(f'  {k}: "none"')
        else:
            lines.append(f"  {k}: {v}")
    notes = row.get("notes")
    if isinstance(notes, str) and notes.strip():
        lines.append("notes: >")
        for part in notes.strip().splitlines():
            lines.append(f"  {part.strip()}")
    else:
        stem = str(row.get("train_config_stem") or row["id"])
        lines.append("notes: >")
        lines.append(f"  Matrix zoo row {row['id']}. Train via train_bench_{stem}.json.")
    return "\n".join(lines) + "\n"


def _train_json_object(row: dict[str, Any]) -> dict[str, Any] | None:
    if not row.get("scaffold_train", True):
        return None
    backend = str(row["backend"])
    if backend == "external":
        return None
    stem = str(row.get("train_config_stem") or row["id"])
    obj: dict[str, Any] = {
        "extends": "configs/experiments/train_bench_base.json",
        "batch": 1,
        "cache": False,
        "notes": f"Matrix/bench training recipe for {stem} @ 1280 (batch=1 on 8GB-class GPUs).",
    }
    if backend == "ultralytics":
        obj["model"] = str(row["model"])
    elif backend == "supergradients":
        obj["model_id"] = str(row["model_id"])
    train_overlay = row.get("train_overlay")
    if isinstance(train_overlay, dict):
        obj.update(train_overlay)
    return obj


def _normalize_groups_tuple(groups: tuple[str, ...]) -> frozenset[str]:
    return frozenset(groups)


def _infer_values_equal(want: object, have: object) -> bool:
    if want == have:
        return True
    if want in ("none", None) and have in ("none", None):
        return True
    return False


def _bench_matches_manifest(cfg_path: Path, row: dict[str, Any], defaults: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    cfg = load_bench_config(cfg_path)
    want_groups = _normalize_groups_tuple(_row_groups(row))
    have_groups = _normalize_groups_tuple(cfg.groups)
    if want_groups != have_groups:
        errs.append(
            f"{cfg_path.name}: groups {sorted(have_groups)} != manifest {sorted(want_groups)}"
        )
    want_backend = str(row["backend"])
    from harchoc.bench_config import select_backend

    if select_backend(cfg) != want_backend:
        errs.append(f"{cfg_path.name}: backend {select_backend(cfg)!r} != {want_backend!r}")
    if want_backend == "ultralytics" and cfg.model != str(row.get("model")):
        errs.append(f"{cfg_path.name}: model {cfg.model!r} != {row.get('model')!r}")
    if want_backend == "external":
        sid = str(row.get("source_id") or row["id"])
        if (cfg.source_id or cfg.model_id) != sid:
            errs.append(f"{cfg_path.name}: source_id mismatch (want {sid!r})")
    want_infer = _infer_block(row, defaults)
    for k, v in want_infer.items():
        if not _infer_values_equal(v, cfg.infer.get(k)):
            errs.append(f"{cfg_path.name}: infer.{k}={cfg.infer.get(k)!r} != {v!r}")
    for k in ("epochs", "patience", "seed"):
        dv = defaults.get(k)
        cv = getattr(cfg, k)
        if cv != dv:
            errs.append(f"{cfg_path.name}: {k}={cv!r} != default {dv!r}")
    return errs


def _train_matches_manifest(train_path: Path, row: dict[str, Any], *, repo_root: Path) -> list[str]:
    if not row.get("scaffold_train", True) or str(row["backend"]) == "external":
        return []
    stem = str(row.get("train_config_stem") or row["id"])
    from harchoc.bench_config import (
        _RUNTIME_TRAIN_BENCH_COMMITTED_STEMS,
        _load_bench_train_raw,
        load_bench_config,
    )

    if stem not in _RUNTIME_TRAIN_BENCH_COMMITTED_STEMS:
        bench_file = repo_root / "configs" / "bench" / str(row["bench_file"])
        if not bench_file.is_file():
            return [f"missing bench for runtime train check: {bench_file}"]
        cfg = load_bench_config(bench_file)
        try:
            raw = _load_bench_train_raw(cfg)
        except FileNotFoundError:
            return [f"runtime train config unavailable for {stem}"]
    else:
        if not train_path.is_file():
            return [f"missing train JSON: {train_path}"]
        raw = json.loads(train_path.read_text(encoding="utf-8"))
        if raw.get("extends"):
            from harchoc.train_config import load_train_config_json

            raw = load_train_config_json(train_path, repo_root=repo_root)
    expected = _train_json_object(row)
    if expected is None:
        return []
    check_keys = ("model", "model_id", "batch")
    if stem in _RUNTIME_TRAIN_BENCH_COMMITTED_STEMS:
        check_keys = ("extends", "model", "model_id", "batch")
    for key in check_keys:
        if key in expected and raw.get(key) != expected.get(key):
            return [f"{train_path.name}: {key}={raw.get(key)!r} != {expected.get(key)!r}"]
    return []


def scaffold_zoo_matrix(
    *,
    repo_root: Path | None = None,
    manifest_path: Path | None = None,
    write: bool = True,
) -> ZooScaffoldReport:
    root = repo_root or _repo_root()
    manifest = load_matrix_rows_manifest(manifest_path)
    mpath = (manifest_path or (root / DEFAULT_MANIFEST)).resolve()
    from harchoc.config_coerce import as_dict

    defaults = as_dict(manifest.get("defaults"))
    report = ZooScaffoldReport(manifest_path=mpath, mode="scaffold" if write else "scaffold_dry")

    bench_dir = root / "configs" / "bench"
    train_dir = root / "configs" / "experiments"

    for raw_row in manifest["rows"]:
        if not isinstance(raw_row, dict):
            report.errors.append("row is not an object")
            continue
        row: dict[str, Any] = raw_row
        bench_file = bench_dir / str(row["bench_file"])
        scaffold_bench = bool(row.get("scaffold_bench", True))

        if scaffold_bench and write:
            text = _bench_yaml_text(row, defaults)
            if bench_file.is_file():
                existing = bench_file.read_text(encoding="utf-8")
                if existing.strip() != text.strip():
                    bench_file.write_text(text, encoding="utf-8")
                    report.bench_updated += 1
                else:
                    report.bench_skipped += 1
            else:
                bench_file.write_text(text, encoding="utf-8")
                report.bench_generated += 1
        elif scaffold_bench:
            report.bench_skipped += 1
        else:
            report.bench_skipped += 1
            if not bench_file.is_file():
                report.errors.append(f"missing hand-authored bench YAML: {bench_file}")

        train_obj = _train_json_object(row)
        stem = str(row.get("train_config_stem") or row["id"])
        from harchoc.bench_config import _RUNTIME_TRAIN_BENCH_COMMITTED_STEMS

        if (
            train_obj is not None
            and write
            and row.get("scaffold_train", True)
            and stem in _RUNTIME_TRAIN_BENCH_COMMITTED_STEMS
        ):
            train_path = train_dir / f"train_bench_{stem}.json"
            text = json.dumps(train_obj, indent=2) + "\n"
            if train_path.is_file():
                existing = train_path.read_text(encoding="utf-8")
                if existing.strip() != text.strip():
                    train_path.write_text(text, encoding="utf-8")
                    report.train_updated += 1
                else:
                    report.train_skipped += 1
            else:
                train_path.write_text(text, encoding="utf-8")
                report.train_generated += 1
        elif train_obj is not None:
            report.train_skipped += 1
        else:
            report.train_skipped += 1

    return report


def validate_zoo_matrix(
    *,
    repo_root: Path | None = None,
    manifest_path: Path | None = None,
) -> ZooScaffoldReport:
    root = repo_root or _repo_root()
    manifest = load_matrix_rows_manifest(manifest_path)
    mpath = (manifest_path or (root / DEFAULT_MANIFEST)).resolve()
    from harchoc.config_coerce import as_dict

    defaults = as_dict(manifest.get("defaults"))
    report = ZooScaffoldReport(manifest_path=mpath, mode="validate")

    bench_dir = root / "configs" / "bench"
    train_dir = root / "configs" / "experiments"
    manifest_bench_names = {str(r["bench_file"]) for r in manifest["rows"] if isinstance(r, dict)}

    for raw_row in manifest["rows"]:
        if not isinstance(raw_row, dict):
            report.errors.append("row is not an object")
            continue
        row: dict[str, Any] = raw_row
        bench_file = bench_dir / str(row["bench_file"])
        if not bench_file.is_file():
            report.errors.append(f"missing bench YAML: {bench_file}")
            continue
        report.errors.extend(_bench_matches_manifest(bench_file, row, defaults))

        stem = str(row.get("train_config_stem") or row["id"])
        train_path = train_dir / f"train_bench_{stem}.json"
        report.errors.extend(_train_matches_manifest(train_path, row, repo_root=root))

    extra = sorted(
        p.name
        for p in bench_dir.glob("*.yaml")
        if p.name not in manifest_bench_names and not p.name.startswith("_")
    )
    if extra:
        report.warnings.append(f"bench YAML not in manifest: {', '.join(extra)}")

    if report.errors:
        report.bench_skipped = len(manifest["rows"])
    else:
        report.bench_skipped = len(manifest["rows"])

    return report


def group_counts_from_manifest(manifest_path: Path | None = None) -> dict[str, int]:
    manifest = load_matrix_rows_manifest(manifest_path)
    counts: dict[str, int] = {}
    for raw_row in manifest["rows"]:
        if not isinstance(raw_row, dict):
            continue
        for g in _row_groups(raw_row):
            counts[g] = counts.get(g, 0) + 1
    return counts
