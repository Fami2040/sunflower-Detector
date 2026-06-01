"""Publication preflight: reviewer2 → figures → tables → aug → backlog narrative."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable

from harchoc.json_io import write_json
from harchoc.repro_chain import (
    format_repro_cmd,
    hsp_test_exports_present,
    overall_step_status,
    reproduce_command_block,
    step_record,
    utc_now_iso,
)
from harchoc.reviewer2_repro import (
    build_reviewer2_repro_chain,
    load_reviewer2_repro_bundle,
    run_reviewer2_repro_chain,
)

PREFLIGHT_MANIFEST_SCHEMA = "manuscript_preflight_manifest.v1"

DEFAULT_STEP_ORDER: tuple[str, ...] = (
    "reviewer2_repro",
    "figures_repro",
    "tables_repro",
    "docx_repro",
    "aug_compare",
    "backlog_narrative",
)


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _cfg_section(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    return _mapping(cfg.get(key))


def preflight_config(ms_bundle: dict[str, Any]) -> dict[str, Any]:
    return _mapping(ms_bundle.get("manuscript_preflight"))


def build_preflight_manifest_skeleton(
    *,
    ms_bundle_path: str,
    dry_run: bool,
    step_ids: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema_version": PREFLIGHT_MANIFEST_SCHEMA,
        "generated_at": utc_now_iso(),
        "bundle": ms_bundle_path,
        "dry_run": dry_run,
        "steps": {sid: step_record(status="pending") for sid in step_ids},
        "overall_status": "pending",
    }


def write_preflight_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["generated_at"] = utc_now_iso()
    manifest["overall_status"] = overall_step_status(manifest.get("steps") or {})
    write_json(path, manifest)


def _resolve_reviewer2_bundle(ms_bundle: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    r2_path = str(
        _cfg_section(cfg, "reviewer2").get("bundle")
        or _mapping(ms_bundle.get("post_zoo_reviewer2")).get("bundle")
        or "configs/experiments/reviewer2_repro.json"
    ).strip()
    return load_reviewer2_repro_bundle(r2_path)


def _resolve_figures_fields(ms_bundle: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    from harchoc.figures_repro import (
        default_figures_repro_fields,
        figures_repro_fields_from_bundle,
        load_figures_repro_bundle,
    )

    fig_cfg = _cfg_section(cfg, "figures")
    bundle_path = str(fig_cfg.get("bundle") or "configs/experiments/figures_repro.json").strip()
    try:
        fig_bundle = load_figures_repro_bundle(bundle_path)
        fields = figures_repro_fields_from_bundle(fig_bundle)
    except Exception:
        fields = default_figures_repro_fields(bundle=ms_bundle)
    if isinstance(fig_cfg, dict):
        for key, val in fig_cfg.items():
            if key != "bundle" and val is not None:
                fields[key] = val
    return fields


def _resolve_tables_fields(ms_bundle: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    from harchoc.manuscript_tables import (
        DEFAULT_AUG_INDEX,
        DEFAULT_AUG_OUT,
        DEFAULT_DUAL_METRIC,
        DEFAULT_MATRIX_ROWS,
        DEFAULT_MATRIX_TRAIN,
        DEFAULT_MODEL_LABEL,
        DEFAULT_OUT_DIR,
        DEFAULT_TOP_N,
    )

    tables_cfg = _cfg_section(cfg, "tables")
    art = _mapping(ms_bundle.get("artifacts"))
    return {
        "out_dir": tables_cfg.get("out_dir") or DEFAULT_OUT_DIR,
        "dual_metric": tables_cfg.get("dual_metric") or art.get("dual_metric") or DEFAULT_DUAL_METRIC,
        "matrix_train": tables_cfg.get("matrix_train") or DEFAULT_MATRIX_TRAIN,
        "matrix_rows": tables_cfg.get("matrix_rows") or DEFAULT_MATRIX_ROWS,
        "aug_index": tables_cfg.get("aug_index") or DEFAULT_AUG_INDEX,
        "aug_out_dir": tables_cfg.get("aug_out_dir") or DEFAULT_AUG_OUT,
        "model_label": tables_cfg.get("model_label") or DEFAULT_MODEL_LABEL,
        "matrix_group": tables_cfg.get("matrix_group") or "zoo_yolo_only",
        "top_n": int(tables_cfg.get("top_n") or DEFAULT_TOP_N),
        "latex": bool(tables_cfg.get("latex", False)),
    }


def _resolve_aug_fields(cfg: dict[str, Any]) -> dict[str, Any]:
    aug_cfg = _cfg_section(cfg, "aug_compare")
    return {
        "index": aug_cfg.get("index") or "configs/experiments/aug_smoke_index.json",
        "out_dir": aug_cfg.get("out_dir") or "reports/aug_smoke",
        "write_figure": aug_cfg.get("write_figure", True),
    }


def _resolve_backlog_fields(cfg: dict[str, Any]) -> dict[str, Any]:
    bn = _cfg_section(cfg, "backlog_narrative")
    return {
        "backlog": bn.get("backlog") or "backlog.md",
        "out_md": bn.get("out_md") or "reports/manuscript/narrative_from_backlog.md",
        "out_json": bn.get("out_json") or "reports/manuscript/backlog_narrative.json",
    }


def run_manuscript_preflight(
    ms_bundle: dict[str, Any],
    *,
    repo_root: str | Path,
    ms_bundle_path: str = "configs/experiments/manuscript_repro_bundle.json",
    dry_run: bool = False,
    on_step: Callable[[str], None] | None = None,
) -> int:
    rr = Path(repo_root).expanduser().resolve()
    cfg = preflight_config(ms_bundle)
    manifest_rel = str(cfg.get("manifest") or "reports/manuscript/preflight_manifest.json")
    manifest_path = (rr / manifest_rel).resolve()

    raw_steps = cfg.get("steps")
    step_ids: tuple[str, ...] = tuple(raw_steps) if isinstance(raw_steps, list) else DEFAULT_STEP_ORDER

    manifest = build_preflight_manifest_skeleton(
        ms_bundle_path=ms_bundle_path,
        dry_run=dry_run,
        step_ids=step_ids,
    )
    write_preflight_manifest(manifest_path, manifest)

    r2_cfg = _cfg_section(cfg, "reviewer2")
    skip_r2_missing = bool(r2_cfg.get("skip_if_missing_hsp_exports", True))

    def _run_step(step_id: str) -> int:
        if on_step is not None:
            on_step(step_id)
        started = utc_now_iso()
        steps = manifest["steps"]
        steps[step_id] = step_record(status="running", started_at=started)
        write_preflight_manifest(manifest_path, manifest)

        try:
            if step_id == "reviewer2_repro":
                return _run_reviewer2_step(
                    rr,
                    ms_bundle,
                    r2_cfg=r2_cfg,
                    skip_if_missing=skip_r2_missing,
                    dry_run=dry_run,
                    steps=steps,
                    started=started,
                )
            if step_id == "figures_repro":
                return _run_figures_step(rr, ms_bundle, cfg, dry_run=dry_run, steps=steps, started=started)
            if step_id == "tables_repro":
                return _run_tables_step(rr, ms_bundle, cfg, dry_run=dry_run, steps=steps, started=started)
            if step_id == "docx_repro":
                return _run_docx_step(rr, cfg, dry_run=dry_run, steps=steps, started=started)
            if step_id == "aug_compare":
                return _run_aug_step(rr, cfg, dry_run=dry_run, steps=steps, started=started)
            if step_id == "backlog_narrative":
                return _run_backlog_step(rr, cfg, dry_run=dry_run, steps=steps, started=started)
            raise ValueError(f"unknown preflight step: {step_id!r}")
        finally:
            write_preflight_manifest(manifest_path, manifest)

    exit_rc = 0
    for step_id in step_ids:
        rc = _run_step(step_id)
        if rc != 0:
            exit_rc = rc
            break

    print(f"Wrote {manifest_path.relative_to(rr) if manifest_path.is_relative_to(rr) else manifest_path}")
    print(f"preflight overall_status={manifest['overall_status']}")
    return exit_rc


def _run_reviewer2_step(
    rr: Path,
    ms_bundle: dict[str, Any],
    *,
    r2_cfg: dict[str, Any],
    skip_if_missing: bool,
    dry_run: bool,
    steps: dict[str, Any],
    started: str,
) -> int:
    r2_bundle = _resolve_reviewer2_bundle(ms_bundle, {"reviewer2": r2_cfg})
    hsp = r2_bundle.get("hsp_artifacts") or {}
    exports_ok = hsp_test_exports_present(rr, hsp)

    if not exports_ok and skip_if_missing and not dry_run:
        msg = "skipped reviewer2-repro: missing HSP test GT/preds exports"
        warnings.warn(msg, stacklevel=2)
        print(f"WARNING: {msg}", file=__import__("sys").stderr)
        steps["reviewer2_repro"] = step_record(
            status="skipped",
            started_at=started,
            finished_at=utc_now_iso(),
            message=msg,
        )
        return 0

    if dry_run:
        print("# reviewer2_repro")
        for sid, argv in build_reviewer2_repro_chain(r2_bundle, repo_root=rr, global_dry_run=True):
            print(f"# {sid}")
            mamba = sid != "reviewer2_paste_check"
            print(format_repro_cmd(argv, mamba=mamba))
        steps["reviewer2_repro"] = step_record(
            status="dry_run",
            started_at=started,
            finished_at=utc_now_iso(),
        )
        return 0

    rc = run_reviewer2_repro_chain(r2_bundle, repo_root=rr, dry_run=False)
    outs = r2_bundle.get("outputs") or {}
    steps["reviewer2_repro"] = step_record(
        status="ok" if rc == 0 else "failed",
        started_at=started,
        finished_at=utc_now_iso(),
        artifacts=[str(v) for v in outs.values() if v],
    )
    return rc


def _run_figures_step(
    rr: Path,
    ms_bundle: dict[str, Any],
    cfg: dict[str, Any],
    *,
    dry_run: bool,
    steps: dict[str, Any],
    started: str,
) -> int:
    from harchoc.figures_repro import run_figures_repro

    fields = _resolve_figures_fields(ms_bundle, cfg)
    if dry_run:
        print("# figures_repro")
    rc = run_figures_repro(fields, repo_root=rr, dry_run=dry_run)
    steps["figures_repro"] = step_record(
        status="dry_run" if dry_run else ("ok" if rc == 0 else "failed"),
        started_at=started,
        finished_at=utc_now_iso(),
        artifacts=[
            str(fields.get("manifest_out") or "reports/figures/manifest.json"),
            str(fields.get("meta_out") or "reports/figures/run.json"),
        ],
    )
    return rc


def _run_docx_step(
    rr: Path,
    cfg: dict[str, Any],
    *,
    dry_run: bool,
    steps: dict[str, Any],
    started: str,
) -> int:
    from harchoc.manuscript_docx_repro import run_manuscript_docx_repro

    docx_cfg = _cfg_section(cfg, "docx_repro")
    out_dir = str(docx_cfg.get("out_dir") or "reports/manuscript/docx")
    if dry_run:
        print("# docx-repro")
        print(format_repro_cmd(["scripts/experiment.py", "manuscript-docx-repro"], mamba=False))
        steps["docx_repro"] = step_record(
            status="dry_run",
            started_at=started,
            finished_at=utc_now_iso(),
            artifacts=[f"{out_dir}/catalog.json"],
        )
        return 0

    payload = run_manuscript_docx_repro(
        rr,
        out_dir=out_dir,
        confusion_path=str(docx_cfg.get("confusion") or "reports/hsp/best2_test_confusion.json"),
        dual_metric_path=str(docx_cfg.get("dual_metric") or "reports/hsp/dual_metric.json"),
        counting_path=str(docx_cfg.get("counting") or "reports/reviewer2_counting_metrics_computed.json"),
        threshold_csv=str(docx_cfg.get("threshold_csv") or "reports/hsp/threshold_val.csv"),
        preds_test=str(docx_cfg.get("preds_test") or "reports/hsp/preds_test.json"),
        gt_test=str(docx_cfg.get("gt_test") or "reports/hsp/gt_test.json"),
        split_file=str(docx_cfg.get("split_file") or "data/splits/test.txt"),
        training_csv=str(docx_cfg.get("training_csv") or "runs/detect/runs/hsp_zoo/yolov8m_e100_s0/results.csv"),
        dry_run=False,
    )
    artifacts = [f"{out_dir}/catalog.json", f"{out_dir}/README.md"]
    for key in ("figures", "tables"):
        block = payload.get(key) or {}
        for ent in block.values():
            if isinstance(ent, dict) and ent.get("path"):
                artifacts.append(str(ent["path"]))
    steps["docx_repro"] = step_record(
        status="ok",
        started_at=started,
        finished_at=utc_now_iso(),
        artifacts=artifacts,
    )
    print(f"docx-repro catalog: {out_dir}/catalog.json")
    return 0


def _run_tables_step(
    rr: Path,
    ms_bundle: dict[str, Any],
    cfg: dict[str, Any],
    *,
    dry_run: bool,
    steps: dict[str, Any],
    started: str,
) -> int:
    from harchoc.manuscript_tables import build_tables_repro_dry_run, write_manuscript_tables

    fields = _resolve_tables_fields(ms_bundle, cfg)
    out_dir = str(fields["out_dir"])
    kwargs = {
        "dual_metric_path": str(fields["dual_metric"]),
        "matrix_train_path": str(fields["matrix_train"]),
        "matrix_rows_path": str(fields["matrix_rows"]),
        "aug_index_path": str(fields["aug_index"]),
        "aug_out_dir": str(fields["aug_out_dir"]),
        "model_label": str(fields["model_label"]),
        "matrix_group": str(fields["matrix_group"]),
        "top_n": int(fields["top_n"]),
    }
    if dry_run:
        print("# tables-repro")
        print(
            format_repro_cmd(
                ["scripts/experiment.py", "tables-repro", "--out-dir", out_dir],
                mamba=False,
            )
        )
        payload = build_tables_repro_dry_run(repo_root=rr, out_dir=out_dir, **kwargs)
        manifest_path = rr / out_dir / "tables_manifest.json"
        write_json(manifest_path, payload)
        steps["tables_repro"] = step_record(
            status="dry_run",
            started_at=started,
            finished_at=utc_now_iso(),
            artifacts=[str(manifest_path.relative_to(rr))],
        )
        return 0

    written = write_manuscript_tables(
        repo_root=rr,
        out_dir=out_dir,
        write_latex=bool(fields.get("latex")),
        **kwargs,
    )
    manifest = written.get("tables_manifest.json")
    artifacts = [str(p.relative_to(rr)) for p in written.values() if p]
    steps["tables_repro"] = step_record(
        status="ok",
        started_at=started,
        finished_at=utc_now_iso(),
        artifacts=artifacts,
    )
    if manifest:
        print(f"tables-repro manifest: {manifest.relative_to(rr)}")
    return 0


def _run_aug_step(
    rr: Path,
    cfg: dict[str, Any],
    *,
    dry_run: bool,
    steps: dict[str, Any],
    started: str,
) -> int:
    from harchoc.aug_comparative import write_aug_comparative_analysis

    fields = _resolve_aug_fields(cfg)
    out_dir = str(fields["out_dir"])
    if dry_run:
        print("# aug-compare")
        print(format_repro_cmd(["scripts/experiment.py", "aug-compare"], mamba=False))
        steps["aug_compare"] = step_record(
            status="dry_run",
            started_at=started,
            finished_at=utc_now_iso(),
            artifacts=[
                f"{out_dir}/comparative_analysis.json",
                f"{out_dir}/comparative_analysis.md",
            ],
        )
        return 0

    paths = write_aug_comparative_analysis(
        repo_root=rr,
        index_path=str(fields["index"]),
        out_dir=out_dir,
        write_figure=bool(fields.get("write_figure", True)),
    )
    artifacts = [str(p.relative_to(rr)) for p in paths.values()]
    steps["aug_compare"] = step_record(
        status="ok",
        started_at=started,
        finished_at=utc_now_iso(),
        artifacts=artifacts,
    )
    for label, path in paths.items():
        print(f"Wrote aug-compare {label}: {path.relative_to(rr)}")
    return 0


def _run_backlog_step(
    rr: Path,
    cfg: dict[str, Any],
    *,
    dry_run: bool,
    steps: dict[str, Any],
    started: str,
) -> int:
    from harchoc.backlog_narrative import run_backlog_narrative

    fields = _resolve_backlog_fields(cfg)
    if dry_run:
        print("# backlog-narrative")
        print(format_repro_cmd(["scripts/experiment.py", "backlog-narrative"], mamba=False))
        steps["backlog_narrative"] = step_record(
            status="dry_run",
            started_at=started,
            finished_at=utc_now_iso(),
        )
        return 0
    rc = run_backlog_narrative(rr, fields)
    steps["backlog_narrative"] = step_record(
        status="ok" if rc == 0 else "failed",
        started_at=started,
        finished_at=utc_now_iso(),
        artifacts=[str(fields["out_md"]), str(fields["out_json"])],
    )
    return rc


def manuscript_preflight_reproduce_commands(
    *,
    bundle_path: str = "configs/experiments/manuscript_repro_bundle.json",
) -> dict[str, list[str]]:
    ci = (
        "PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 "
        f"python scripts/experiment.py --config {bundle_path} manuscript-preflight"
    )
    gpu = f"mamba run -n harchoc python scripts/experiment.py manuscript-preflight"
    return reproduce_command_block(
        ci_dry_run=[f"{ci} --dry-run"],
        local=[gpu, f"{gpu} --dry-run"],
        extra={
            "repro_alias": [f"mamba run -n harchoc python scripts/experiment.py repro --stage preflight"],
            "post_zoo_only": [
                f"mamba run -n harchoc python scripts/experiment.py repro --stage post-zoo",
            ],
        },
    )


def run_publication_pipeline(
    ms_bundle: dict[str, Any],
    *,
    repo_root: str | Path,
    ms_bundle_path: str,
    dry_run: bool,
    include_hsp: bool,
    skip_gpu_check: bool = False,
    include_test_map: bool = False,
) -> int:
    """HSP repro (optional) then manuscript preflight — single entry for ``repro --stage full``."""
    from harchoc.manuscript_repro import run_manuscript_repro_chain

    rr = Path(repo_root).expanduser().resolve()
    if include_hsp:
        rc = run_manuscript_repro_chain(
            ms_bundle,
            repo_root=rr,
            dry_run=dry_run,
            skip_gpu_check=skip_gpu_check,
            include_test_map=include_test_map,
        )
        if rc != 0:
            return rc
    return run_manuscript_preflight(
        ms_bundle,
        repo_root=rr,
        ms_bundle_path=ms_bundle_path,
        dry_run=dry_run,
    )
