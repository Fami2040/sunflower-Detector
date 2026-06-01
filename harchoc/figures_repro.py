"""Manuscript figure reproduction: journal style, run manifest, file audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harchoc.figure_style import FIGURE_DPI
from harchoc.model_zoo import file_sha256
from harchoc.schemas import with_schema_version

FIGURES_REPRO_MANIFEST_SCHEMA = "figures_repro_manifest.v1"

MANUSCRIPT_FIGURE_IDS: tuple[str, ...] = (
    "fig_concept",
    "fig_pr_curve",
    "fig_error_taxonomy",
    "fig_gradcam_panel",
    "fig_ambiguous_panel",
    "fig_split_drift",
)

# Catalog aligned with scripts/make_figures._plan_figures (audit / gap tracking).
def _rel_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


FIGURE_CATALOG: tuple[dict[str, str], ...] = (
    {"id": "fig_concept", "notes": "Pipeline concept diagram / overview (CPU)."},
    {"id": "fig_pr_curve", "notes": "PR/F1 vs confidence (threshold sweep)."},
    {"id": "fig_error_taxonomy", "notes": "FP taxonomy examples / qualitative panel (CPU)."},
    {"id": "fig_gradcam_panel", "notes": "Grad-CAM overlays on FP crops (optional GPU)."},
    {"id": "fig_ambiguous_panel", "notes": "Ambiguous detections (low conf / high pred IoU)."},
    {"id": "fig_split_drift", "notes": "Train/val/test drift summary plots."},
)


def default_figures_repro_fields(
    *,
    bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve paths from manuscript_repro_bundle artifacts when present."""
    arts = (bundle or {}).get("artifacts") if isinstance(bundle, dict) else {}
    arts = arts if isinstance(arts, dict) else {}
    return {
        "out_dir": "reports/figures",
        "meta_out": "reports/figures/run.json",
        "manifest_out": "reports/figures/manifest.json",
        "split_drift_report": str(arts.get("split_drift") or "reports/hsp/split_drift_p0.json"),
        "threshold_csv": "reports/hsp/threshold_val.csv",
        "threshold_json": str(arts.get("threshold_val") or "reports/hsp/threshold_val.json"),
        "error_report": str(
            arts.get("error_test_report")
            or arts.get("error_test")
            or "reports/hsp/error_test_report.json"
        ),
        "figure": "all",
        "panel_size": 12,
        "journal_style": True,
    }


def load_figures_repro_bundle(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    obj = json.loads(p.read_text(encoding="utf-8"))
    if obj.get("schema_version") != "figures_repro_bundle.v1":
        raise ValueError(f"unsupported figures repro bundle schema: {obj.get('schema_version')!r}")
    return obj


def figures_repro_fields_from_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    ms = bundle.get("manuscript")
    ms_bundle: dict[str, Any] | None = None
    if isinstance(ms, dict):
        ms_bundle = ms
    elif isinstance(ms, str) and ms.strip():
        from harchoc.manuscript_repro import load_manuscript_repro_bundle

        ms_bundle = load_manuscript_repro_bundle(ms)
    base = default_figures_repro_fields(bundle=ms_bundle)
    run = bundle.get("run") or bundle.get("figures-repro")
    if isinstance(run, dict):
        for key in (
            "out_dir",
            "meta_out",
            "manifest_out",
            "split_drift_report",
            "threshold_csv",
            "threshold_json",
            "error_report",
            "weights",
            "figure",
            "panel_size",
            "journal_style",
        ):
            if key in run and run[key] is not None:
                base[key] = run[key]
    return base


def iter_rendered_paths(rendered: dict[str, Any]) -> list[tuple[str, str]]:
    """Return (figure_id, path) pairs from make_figures rendered block."""
    out: list[tuple[str, str]] = []
    for fid, entry in rendered.items():
        if not isinstance(entry, dict):
            continue
        op = entry.get("out_path")
        if op:
            out.append((fid, str(op)))
        files = entry.get("files")
        if isinstance(files, list):
            for f in files:
                if f:
                    out.append((fid, str(f)))
    return out


def _dedupe_paths(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for fid, path in pairs:
        key = str(Path(path))
        if key in seen:
            continue
        seen.add(key)
        out.append((fid, path))
    return out


def collect_figure_paths(
    run_payload: dict[str, Any],
    *,
    repo_root: Path,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    rendered = run_payload.get("rendered")
    if isinstance(rendered, dict):
        pairs.extend(iter_rendered_paths(rendered))
    figures = run_payload.get("figures")
    if isinstance(figures, list):
        for fig in figures:
            if not isinstance(fig, dict):
                continue
            fid = str(fig.get("id") or "")
            paths = fig.get("paths")
            if isinstance(paths, list):
                for p in paths:
                    if p:
                        pairs.append((fid, str(p)))
    resolved: list[tuple[str, str]] = []
    for fid, raw in pairs:
        p = Path(raw)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        resolved.append((fid, str(p)))
    return _dedupe_paths(resolved)


def inspect_figure_file(
    path: Path,
    *,
    figure_id: str,
    journal_style: bool,
    rendered_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "figure_id": figure_id,
        "path": str(path),
        "dpi": FIGURE_DPI if journal_style else 120,
    }
    if not path.is_file():
        entry["status"] = "missing"
        entry["size_bytes"] = 0
        return entry
    entry["status"] = "ok"
    entry["size_bytes"] = path.stat().st_size
    entry["sha256"] = file_sha256(path)
    suffix = path.suffix.lower()
    if suffix in (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"):
        try:
            from PIL import Image  # type: ignore

            with Image.open(path) as im:
                entry["pixel_size"] = [int(im.size[0]), int(im.size[1])]
        except Exception as exc:
            entry["pixel_read_error"] = str(exc)
    if isinstance(rendered_entry, dict):
        fs = rendered_entry.get("figsize_inches")
        if isinstance(fs, (list, tuple)) and len(fs) >= 2:
            entry["figsize_inches"] = [float(fs[0]), float(fs[1])]
        px = rendered_entry.get("pixel_size")
        if isinstance(px, (list, tuple)) and len(px) >= 2 and "pixel_size" not in entry:
            entry["pixel_size"] = [int(px[0]), int(px[1])]
    return entry


def audit_figure_catalog(
    *,
    repo_root: Path,
    run_payload: dict[str, Any],
) -> dict[str, Any]:
    on_disk: dict[str, list[str]] = {}
    for fid, path in collect_figure_paths(run_payload, repo_root=repo_root):
        on_disk.setdefault(fid, []).append(path)
    missing_ids = [fid for fid in MANUSCRIPT_FIGURE_IDS if fid not in on_disk]
    rendered = run_payload.get("rendered")
    skipped: list[dict[str, str]] = []
    if isinstance(rendered, dict):
        for fid in MANUSCRIPT_FIGURE_IDS:
            ent = rendered.get(fid)
            if isinstance(ent, dict) and ent.get("status") == "skipped":
                skipped.append({"id": fid, "reason": str(ent.get("reason") or "")})
    return {
        "catalog": list(FIGURE_CATALOG),
        "expected_ids": list(MANUSCRIPT_FIGURE_IDS),
        "present_ids": sorted(on_disk.keys()),
        "missing_ids": missing_ids,
        "skipped": skipped,
        "paths_by_id": on_disk,
    }


def build_figures_repro_manifest(
    *,
    repo_root: Path,
    run_payload: dict[str, Any],
    journal_style: bool,
    out_dir: str | Path,
    run_json: str | Path,
) -> dict[str, Any]:
    rendered = run_payload.get("rendered")
    rendered = rendered if isinstance(rendered, dict) else {}
    files: list[dict[str, Any]] = []
    for fid, path_str in collect_figure_paths(run_payload, repo_root=repo_root):
        ent = rendered.get(fid)
        rend_ent = ent if isinstance(ent, dict) else None
        files.append(
            inspect_figure_file(
                Path(path_str),
                figure_id=fid,
                journal_style=journal_style,
                rendered_entry=rend_ent,
            )
        )
    validation = validate_manifest_files(files)
    payload: dict[str, Any] = {
        "script": "figures-repro",
        "journal_style": journal_style,
        "figure_dpi": FIGURE_DPI if journal_style else 120,
        "font_family": "sans-serif",
        "out_dir": str(Path(out_dir)),
        "run_json": str(Path(run_json)),
        "run_schema_version": run_payload.get("schema_version"),
        "audit": audit_figure_catalog(repo_root=repo_root, run_payload=run_payload),
        "files": files,
        "validation": validation,
    }
    return with_schema_version(payload, schema_version=FIGURES_REPRO_MANIFEST_SCHEMA)


def validate_manifest_files(files: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    for ent in files:
        path = ent.get("path")
        fid = ent.get("figure_id")
        if ent.get("status") != "ok":
            errors.append(f"{fid}: missing file {path}")
            continue
        size = int(ent.get("size_bytes") or 0)
        if size <= 0:
            errors.append(f"{fid}: zero-byte file {path}")
    return {"ok": not errors, "errors": errors}


def validate_required_figures(
    manifest: dict[str, Any],
    *,
    require_ids: tuple[str, ...] = MANUSCRIPT_FIGURE_IDS,
) -> list[str]:
    """Return error strings when a required figure id has no ok on-disk file."""
    by_id: dict[str, list[dict[str, Any]]] = {}
    for ent in manifest.get("files") or []:
        if not isinstance(ent, dict):
            continue
        fid = str(ent.get("figure_id") or "")
        by_id.setdefault(fid, []).append(ent)
    errors: list[str] = []
    audit = manifest.get("audit")
    skipped_ids: set[str] = set()
    if isinstance(audit, dict):
        for row in audit.get("skipped") or []:
            if isinstance(row, dict) and row.get("id"):
                skipped_ids.add(str(row["id"]))
    for fid in require_ids:
        if fid in skipped_ids:
            continue
        rows = by_id.get(fid) or []
        if not any(r.get("status") == "ok" and int(r.get("size_bytes") or 0) > 0 for r in rows):
            errors.append(f"required figure not rendered: {fid}")
    return errors


def figures_repro_reproduce_commands(
    *,
    bundle_path: str = "configs/experiments/figures_repro.json",
) -> dict[str, list[str]]:
    ci = (
        "PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 "
        f"python scripts/experiment.py --config {bundle_path} figures-repro"
    )
    gpu = f"mamba run -n harchoc python scripts/experiment.py --config {bundle_path} figures-repro"
    return {
        "ci_safe_dry_run": [f"{ci} --dry-run"],
        "local_full": [gpu],
        "gradcam_only": [
            "mamba run -n harchoc python scripts/experiment.py gradcam --weights models/best2.pt"
        ],
    }


def run_figures_repro(
    fields: dict[str, Any],
    *,
    repo_root: Path,
    dry_run: bool = False,
) -> int:
    from harchoc.experiment_argv import argv_for_figures_repro
    from harchoc.manuscript_repro import _format_cmd
    from scripts._common_cli import write_json

    argv = argv_for_figures_repro(fields)
    manifest_out = str(fields.get("manifest_out") or "reports/figures/manifest.json")
    meta_out = str(fields.get("meta_out") or "reports/figures/run.json")
    journal_style = bool(fields.get("journal_style", True))

    if dry_run:
        print("# figures-repro")
        print(_format_cmd(["scripts/make_figures.py", *argv], mamba=True))
        print(f"# would write {manifest_out} (figures_repro_manifest.v1)")
        return 0

    from scripts.make_figures import main as make_figures_main

    rc = make_figures_main(argv)
    if rc != 0:
        return rc

    meta_path = Path(meta_out)
    if not meta_path.is_absolute():
        meta_path = (repo_root / meta_path).resolve()
    if not meta_path.is_file():
        print(f"ERROR: missing run manifest {meta_path}", file=__import__("sys").stderr)
        return 1

    run_payload = json.loads(meta_path.read_text(encoding="utf-8"))
    manifest = build_figures_repro_manifest(
        repo_root=repo_root,
        run_payload=run_payload,
        journal_style=journal_style,
        out_dir=str(fields.get("out_dir") or "reports/figures"),
        run_json=_rel_path(meta_path, repo_root),
    )
    figure_sel = str(fields.get("figure") or "all").strip()
    require_ids: tuple[str, ...] = (
        MANUSCRIPT_FIGURE_IDS if figure_sel == "all" else (figure_sel,)
    )
    extra_errors = validate_required_figures(manifest, require_ids=require_ids)
    if extra_errors:
        manifest["validation"]["ok"] = False
        manifest["validation"]["errors"] = list(manifest["validation"].get("errors") or []) + extra_errors

    out_manifest = Path(manifest_out)
    if not out_manifest.is_absolute():
        out_manifest = (repo_root / out_manifest).resolve()
    write_json(str(out_manifest), manifest)
    print(f"Wrote {out_manifest}")
    n_ok = sum(1 for f in manifest.get("files") or [] if f.get("status") == "ok")
    print(f"Figures on disk: {n_ok} file(s) across {len(manifest.get('audit', {}).get('present_ids', []))} id(s)")
    if not manifest["validation"]["ok"]:
        for err in manifest["validation"]["errors"]:
            print(f"ERROR: {err}", file=__import__("sys").stderr)
        return 1
    return 0
