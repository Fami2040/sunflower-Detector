from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import sys

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from harchoc.script_entry import bootstrap_repo_imports

bootstrap_repo_imports()

from harchoc.experiment_config import load_config_json, merge_experiment_config, script_section_from_config
from harchoc.hsp_weights import HSP_DETECTION_WEIGHTS
from scripts._common_cli import add_dataset_args, add_dry_run_arg, write_json


def _ns_to_dict(ns: argparse.Namespace) -> dict[str, Any]:
    return {k: v for k, v in vars(ns).items() if v is not argparse.SUPPRESS}


_SUBCOMMAND_RUN_KIND: dict[str, str] = {
    "benchmark": "benchmark_matrix",
    "cv-eval": "cv_eval",
}


def _merged_for_subcommand(
    *,
    config_obj: dict[str, Any],
    subcommand: str,
    cli_fields: dict[str, Any],
    force_dry_run: bool,
) -> dict[str, Any]:
    cfg = config_obj.get(subcommand)
    if not isinstance(cfg, dict):
        alt_key = _SUBCOMMAND_RUN_KIND.get(subcommand, subcommand.replace("-", "_"))
        if alt_key != subcommand:
            cfg = config_obj.get(alt_key)
    cfg_obj = cfg if isinstance(cfg, dict) else {}
    if not cfg_obj:
        section = _SUBCOMMAND_RUN_KIND.get(subcommand, subcommand.replace("-", "_"))
        cfg_obj = script_section_from_config(config_obj, section)
    merged = merge_experiment_config(config=cfg_obj, cli=cli_fields)
    if force_dry_run:
        merged["dry_run"] = True
    return merged


def _run_gradcam(fields: dict[str, Any]) -> int:
    from harchoc.experiment_argv import argv_for_gradcam
    from harchoc.manuscript_repro import _format_cmd

    argv = argv_for_gradcam(fields)
    if bool(fields.get("dry_run")):
        print("# gradcam")
        print(_format_cmd(["scripts/make_figures.py", *argv], mamba=True))
        return 0
    from scripts.make_figures import main as legacy_main

    return legacy_main(argv)


def _run_figures_repro(fields: dict[str, Any]) -> int:
    from harchoc.figures_repro import (
        figures_repro_fields_from_bundle,
        load_figures_repro_bundle,
        run_figures_repro,
    )

    repo_root = Path(__file__).resolve().parents[1]
    merged = dict(fields)
    bundle_path = str(merged.get("bundle") or "").strip()
    if bundle_path:
        bundle = load_figures_repro_bundle(bundle_path)
        merged = {**figures_repro_fields_from_bundle(bundle), **merged}
    elif merged.get("schema_version") == "figures_repro_bundle.v1":
        merged = {**figures_repro_fields_from_bundle(merged), **{k: v for k, v in fields.items() if k != "schema_version"}}
    return run_figures_repro(merged, repo_root=repo_root, dry_run=bool(merged.get("dry_run")))


def _run_deploy_parity(fields: dict[str, Any]) -> int:
    from harchoc.deploy_hsp_parity import (
        build_deploy_hsp_parity_payload,
        resolve_parity_image_sample,
        write_deploy_hsp_parity,
    )
    from harchoc.experiment_argv import argv_for_deploy_parity
    from harchoc.manuscript_repro import _format_cmd

    argv = argv_for_deploy_parity(fields)
    out = str(fields.get("out") or "reports/hsp/deploy_hsp_parity.json")
    locked = fields.get("locked_conf_from") or "reports/hsp/threshold_val.json"
    sample_n = int(fields.get("sample_images") or 0)
    split_file = str(fields.get("split_file") or "data/splits/test.txt")
    weights = str(fields.get("weights") or HSP_DETECTION_WEIGHTS)
    if bool(fields.get("dry_run")):
        print("# deploy-parity")
        print(_format_cmd(["scripts/experiment.py", "deploy-parity", *argv], mamba=False))
        payload = build_deploy_hsp_parity_payload(locked_conf_from=str(locked))
        if sample_n > 0:
            print(f"# would sample up to {sample_n} images from {split_file}")
        print(f"# would write {out} status={payload.get('status')}")
        return 0

    image_paths: list[str] | None = None
    per_image = None
    notes: str | None = None
    if sample_n > 0:
        image_paths, per_image, skip_note = resolve_parity_image_sample(
            sample_images=sample_n,
            split_file=split_file,
            dataset_root=fields.get("dataset_root"),
            locked_conf_from=str(locked),
            weights=weights,
            manifest_path=fields.get("manifest"),
            dataset_name=fields.get("dataset_name"),
            yolo_data_yaml=fields.get("yolo_data_yaml"),
            default_dataset_name=fields.get("default_dataset_name"),
        )
        if skip_note:
            notes = skip_note

    write_deploy_hsp_parity(
        out,
        locked_conf_from=str(locked),
        image_paths=image_paths,
        per_image=per_image,
        notes=notes,
    )
    print(f"Wrote {out}")
    if per_image:
        print(f"Compared {len(per_image)} image(s) (SAHI vs full-frame @ locked conf)")
    elif sample_n > 0 and notes:
        print(f"Image sample skipped: {notes}")
    return 0


def _run_finetune_tray(fields: dict[str, Any]) -> int:
    from harchoc.experiment_argv import argv_for_finetune
    from harchoc.manuscript_repro import _format_cmd
    from scripts.finetune import main as finetune_main

    argv = argv_for_finetune(fields)
    if bool(fields.get("dry_run")):
        print("# finetune-tray")
        print(_format_cmd(["scripts/experiment.py", "finetune-tray", *argv], mamba=True))
    return int(finetune_main(argv))


def _run_domain_tray_audit(fields: dict[str, Any]) -> int:
    from harchoc.finetune_tray_audit import build_weak_tray_plan, write_weak_tray_plan
    from harchoc.manuscript_repro import _format_cmd

    out = str(fields.get("out") or "reports/domains/weak_tray_plan.json")
    top_k = int(fields.get("top_k") or 3)
    global_mae = fields.get("global_mae")
    global_ref = float(global_mae) if global_mae is not None else 61.3

    if bool(fields.get("dry_run")):
        print("# domain-tray-audit")
        print(
            _format_cmd(
                [
                    "scripts/experiment.py",
                    "domain-tray-audit",
                    "--out",
                    out,
                    "--top-k",
                    str(top_k),
                ],
                mamba=False,
            )
        )
        payload = build_weak_tray_plan(
            count_mae_path=Path(str(fields.get("count_mae") or "reports/domains/domain_count_mae.json")),
            domain_eval_path=Path(str(fields.get("domain_eval") or "reports/domains/domain_eval.json")),
            top_k=top_k,
            global_mae=global_ref,
        )
        write_weak_tray_plan(Path(out), payload)
        print(f"Wrote dry-run plan {out}")
        return 0

    payload = build_weak_tray_plan(
        count_mae_path=Path(str(fields.get("count_mae") or "reports/domains/domain_count_mae.json")),
        domain_eval_path=Path(str(fields.get("domain_eval") or "reports/domains/domain_eval.json")),
        top_k=top_k,
        global_mae=global_ref,
    )
    write_weak_tray_plan(Path(out), payload)
    print(f"Wrote {out}")
    keys = payload.get("recommended_tray_keys") or []
    if keys:
        print(f"Recommended finetune tray_key(s): {', '.join(keys)}")
    return 0 if payload.get("status") in ("ok", "pending", "empty") else 1


def _run_map_cpu(fields: dict[str, Any]) -> int:
    from harchoc.experiment_argv import argv_for_map_cpu
    from harchoc.manuscript_repro import _format_cmd

    argv = argv_for_map_cpu(fields)
    if bool(fields.get("dry_run")):
        print("# map-cpu")
        print(_format_cmd(["scripts/eval.py", *argv], mamba=True))
        return 0
    from scripts.eval import main as legacy_main

    return legacy_main(argv)


def _run_reviewer2_map50(fields: dict[str, Any]) -> int:
    from harchoc.experiment_argv import argv_for_map_cpu
    from harchoc.json_io import load_json_dict
    from harchoc.manuscript_repro import _format_cmd
    from harchoc.reviewer2_map50_report import (
        build_reviewer2_map50_computed,
        reviewer2_map50_reproduce_commands,
    )
    from scripts._common_cli import require_conda_env, write_json

    repo_root = Path(__file__).resolve().parents[1]
    out = str(fields.get("out") or "reports/reviewer2_map50_computed.json")
    cmds = reviewer2_map50_reproduce_commands(rerun_eval=bool(fields.get("rerun_eval")))

    if bool(fields.get("dry_run")):
        print("# reviewer2-map50")
        print(f"# aggregate: {cmds['aggregate_from_artifacts']}")
        if bool(fields.get("rerun_eval")):
            print("# rerun-eval via map-cpu:")
            print(_format_cmd(["scripts/eval.py", *argv_for_map_cpu(fields)], mamba=True))
        print(f"# would write {out}")
        return 0

    rerun_doc: dict[str, Any] | None = None
    if bool(fields.get("rerun_eval")):
        require_conda_env()
        from scripts.eval import main as legacy_main

        eval_out = str(fields.get("eval_test_map") or "reports/hsp/eval_test_map.json")
        rerun_argv = argv_for_map_cpu({**fields, "out": eval_out})
        rc = legacy_main(rerun_argv)
        if rc != 0:
            return rc
        rerun_doc = load_json_dict(repo_root / eval_out)

    payload = build_reviewer2_map50_computed(
        repo_root=repo_root,
        eval_test_map=str(fields.get("eval_test_map") or "reports/hsp/eval_test_map.json"),
        dual_metric=str(fields.get("dual_metric") or "reports/hsp/dual_metric.json"),
        eval_val=str(fields.get("eval_val") or "reports/hsp/eval_val.json"),
        eval_test=str(fields.get("eval_test") or "reports/hsp/eval_test.json"),
        rerun_eval_doc=rerun_doc,
    )
    write_json(out, payload)
    print(f"Wrote {out} status={payload.get('status')}")
    hsp = (payload.get("hsp_canonical") or {}).get("mAP50_display")
    if hsp:
        print(f"Canonical test mAP50 (HSP): {hsp}")
    return 0 if payload.get("status") == "ok" else 1


def _run_tune_sahi(fields: dict[str, Any]) -> int:
    from harchoc.experiment_argv import argv_for_tune_sahi
    from harchoc.manuscript_repro import _format_cmd

    argv = argv_for_tune_sahi(fields)
    if bool(fields.get("dry_run")):
        print("# tune-sahi")
        print(_format_cmd(["scripts/experiment.py", "tune-sahi", *argv], mamba=True))
        return 0
    raise SystemExit(
        "tune-sahi live grid was removed with tune_sahi_params.py; "
        "use run_infer_once.py or experiment.py deploy-parity for deploy tuning."
    )


def _run_reviewer2_confusion(fields: dict[str, Any]) -> int:
    from harchoc.manuscript_repro import _format_cmd
    from harchoc.reviewer2_confusion_tide import DEFAULT_OUT, run_reviewer2_confusion_tide
    from scripts._common_cli import cli_print, write_json

    repo_root = Path(__file__).resolve().parents[1]
    out = str(fields.get("out") or DEFAULT_OUT)
    if bool(fields.get("dry_run")):
        payload = run_reviewer2_confusion_tide(
            repo_root=repo_root,
            out=out,
            gt_json=str(fields.get("gt_json") or "").strip() or None,
            preds_json=str(fields.get("preds_json") or "").strip() or None,
            locked_conf_from=str(fields.get("locked_conf_from") or "").strip() or None,
            error_report=str(fields.get("error_report") or "").strip() or None,
            tide_summary=str(fields.get("tide_summary") or "").strip() or None,
            confusion_iou_0_3=str(fields.get("confusion_iou_0_3") or "").strip() or None,
            dry_run=True,
        )
        write_json(out, payload)
        print("# reviewer2-confusion")
        print(_format_cmd(["scripts/experiment.py", "reviewer2-confusion", "--out", out], mamba=False))
        cli_print(f"Wrote {out}")
        return 0

    payload = run_reviewer2_confusion_tide(
        repo_root=repo_root,
        out=out,
        gt_json=str(fields.get("gt_json") or "").strip() or None,
        preds_json=str(fields.get("preds_json") or "").strip() or None,
        locked_conf_from=str(fields.get("locked_conf_from") or "").strip() or None,
        error_report=str(fields.get("error_report") or "").strip() or None,
        tide_summary=str(fields.get("tide_summary") or "").strip() or None,
        confusion_iou_0_3=str(fields.get("confusion_iou_0_3") or "").strip() or None,
        dry_run=False,
    )
    write_json(out, payload)
    parity = payload.get("parity") or {}
    cli_print(f"Wrote {out}")
    cli_print(f"parity: {parity}")
    audit = payload.get("audit") or []
    failed = [row for row in audit if not row.get("match")]
    if failed:
        cli_print(f"audit mismatches: {len(failed)}")
        return 1
    return 0


def _run_fp_budget_sweep(fields: dict[str, Any]) -> int:
    from harchoc.fp_budget_sweep import run_fp_budget_sweep
    from harchoc.manuscript_repro import _format_cmd
    from harchoc.threshold_protocol import enforce_tuning_guardrails, infer_split_role, resolve_dataset_root_for_splits
    from scripts._common_cli import cli_print, read_json, require_conda_env, resolve_light_gt_preds, write_json

    require_conda_env()
    out = str(fields.get("out") or "reports/hsp/fp_budget_sweep.json")
    if bool(fields.get("dry_run")):
        payload = run_fp_budget_sweep(
            out=out,
            gt_json="",
            preds_json="",
            gt={},
            preds={},
            fp_budget_grid=list(fields.get("fp_budget_grid") or []),
            dry_run=True,
        )
        write_json(out, payload)
        print("# fp-budget-sweep")
        print(_format_cmd(["scripts/experiment.py", "fp-budget-sweep", "--out", out], mamba=False))
        cli_print(f"Wrote {out}")
        return 0

    repo_root = Path(__file__).resolve().parents[1]
    gt_json = str(fields.get("gt_json") or "").strip()
    preds_json = str(fields.get("preds_json") or "").strip()
    if bool(fields.get("light")):
        gt_path, preds_path = resolve_light_gt_preds(repo_root=repo_root, gt_json=gt_json, preds_json=preds_json)
        gt_json, preds_json = str(gt_path), str(preds_path)

    gt_obj = read_json(gt_json) if gt_json else None
    preds_obj = read_json(preds_json) if preds_json else None
    if gt_obj is None or preds_obj is None:
        raise SystemExit("fp-budget-sweep requires --gt-json and --preds-json (or --light)")

    ds_root = resolve_dataset_root_for_splits(repo_root=repo_root, dataset_root=None)
    split_role, split_hints = infer_split_role(
        gt_json=gt_json,
        preds_json=preds_json,
        split_file=(str(fields.get("split_file") or "").strip() or None),
        gt=gt_obj,
        repo_root=repo_root,
        dataset_root=ds_root,
    )
    sweep_from = (str(fields.get("sweep_from") or "").strip() or None)
    locked_from = (str(fields.get("locked_conf_from") or "").strip() or None)
    enforce_tuning_guardrails(
        split_role,
        locked_conf_from=sweep_from or locked_from,
        iou_grid=None,
        calibrate="none",
    )

    grid_raw = fields.get("fp_budget_grid")
    fp_grid = list(grid_raw) if isinstance(grid_raw, list) and grid_raw else None
    iou_raw = fields.get("iou")
    iou_thr = float(iou_raw) if iou_raw is not None else None

    payload = run_fp_budget_sweep(
        out=out,
        gt_json=gt_json,
        preds_json=preds_json,
        gt=gt_obj,
        preds=preds_obj,
        sweep_from=(str(fields.get("sweep_from") or "").strip() or None),
        fp_budget_grid=fp_grid,
        iou_thr=iou_thr,
        split_role=split_role,
        split_hints=split_hints,
        locked_conf_from=(str(fields.get("locked_conf_from") or "").strip() or None),
        summary_out=(str(fields.get("summary_out") or "").strip() or None),
    )
    cli_print(f"Wrote {out}")
    summary_out = (str(fields.get("summary_out") or "").strip() or None)
    if summary_out:
        cli_print(f"Wrote {summary_out}")
    return 0


def _run_reviewer2_repro(fields: dict[str, Any]) -> int:
    from harchoc.reviewer2_repro import load_reviewer2_repro_bundle, run_reviewer2_repro_chain

    repo_root = Path(__file__).resolve().parents[1]
    bundle_path = str(
        fields.get("bundle") or "configs/experiments/reviewer2_repro.json"
    ).strip()
    bundle = load_reviewer2_repro_bundle(bundle_path)
    return run_reviewer2_repro_chain(
        bundle,
        repo_root=repo_root,
        dry_run=bool(fields.get("dry_run")),
    )


def _run_manuscript_preflight(fields: dict[str, Any]) -> int:
    from harchoc.manuscript_preflight import run_manuscript_preflight
    from harchoc.manuscript_repro import load_manuscript_repro_bundle

    repo_root = Path(__file__).resolve().parents[1]
    bundle_path = str(
        fields.get("bundle") or "configs/experiments/manuscript_repro_bundle.json"
    ).strip()
    ms_bundle = load_manuscript_repro_bundle(bundle_path)
    return run_manuscript_preflight(
        ms_bundle,
        repo_root=repo_root,
        ms_bundle_path=bundle_path,
        dry_run=bool(fields.get("dry_run")),
    )


def _run_aug_compare(fields: dict[str, Any]) -> int:
    from harchoc.aug_comparative import write_aug_comparative_analysis
    from harchoc.manuscript_repro import _format_cmd

    repo_root = Path(__file__).resolve().parents[1]
    index_path = str(fields.get("index") or "configs/experiments/aug_smoke_index.json")
    out_dir = str(fields.get("out_dir") or "reports/aug_smoke")
    if bool(fields.get("dry_run")):
        print("# aug-compare")
        print(_format_cmd(["scripts/experiment.py", "aug-compare"], mamba=False))
        return 0
    paths = write_aug_comparative_analysis(
        repo_root=repo_root,
        index_path=index_path,
        out_dir=out_dir,
        write_figure=not bool(fields.get("no_figure")),
    )
    for label, path in paths.items():
        print(f"Wrote {label}: {path}")
    return 0


def _run_tables_repro(fields: dict[str, Any]) -> int:
    from harchoc.manuscript_repro import _format_cmd
    from harchoc.manuscript_tables import (
        DEFAULT_OUT_DIR,
        build_tables_repro_dry_run,
        write_manuscript_tables,
    )
    from scripts._common_cli import write_json

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = str(fields.get("out_dir") or DEFAULT_OUT_DIR)
    kwargs = {
        "dual_metric_path": str(fields.get("dual_metric") or "reports/hsp/dual_metric.json"),
        "matrix_train_path": str(fields.get("matrix_train") or "reports/hsp/matrix_train.json"),
        "matrix_rows_path": str(fields.get("matrix_rows") or "configs/zoo/matrix_rows.v1.json"),
        "aug_index_path": str(
            fields.get("aug_index") or "configs/experiments/aug_smoke_index.json"
        ),
        "aug_out_dir": str(fields.get("aug_out_dir") or "reports/aug_smoke"),
        "model_label": str(fields.get("model_label") or "models/best2.pt"),
        "matrix_group": str(fields.get("matrix_group") or "zoo_yolo_only"),
        "top_n": int(fields.get("top_n") or 10),
        "aug_leaderboard_json": str(fields.get("aug_leaderboard_json") or "").strip() or None,
    }
    if bool(fields.get("dry_run")):
        payload = build_tables_repro_dry_run(repo_root=repo_root, out_dir=out_dir, **kwargs)
        manifest_path = repo_root / out_dir / "tables_manifest.json"
        write_json(manifest_path, payload)
        print("# tables-repro")
        print(_format_cmd(["scripts/experiment.py", "tables-repro", "--out-dir", out_dir], mamba=False))
        for line in payload.get("would_write") or []:
            print(f"#   {line}")
        if payload.get("warnings"):
            for w in payload["warnings"]:
                print(f"# warn: {w}")
        print(f"Wrote {manifest_path.relative_to(repo_root)} (dry-run)")
        return 0

    written = write_manuscript_tables(
        repo_root=repo_root,
        out_dir=out_dir,
        write_latex=bool(fields.get("latex")),
        **kwargs,
    )
    manifest = written.get("tables_manifest.json")
    print(f"Wrote {len(written)} files under {out_dir}")
    if manifest:
        rel = (
            manifest.relative_to(repo_root)
            if manifest.is_relative_to(repo_root)
            else manifest
        )
        print(f"Manifest: {rel}")
    return 0


def _run_repro(fields: dict[str, Any]) -> int:
    from harchoc.manuscript_preflight import run_manuscript_preflight, run_publication_pipeline
    from harchoc.manuscript_repro import load_manuscript_repro_bundle, run_manuscript_repro_chain
    from harchoc.reviewer2_repro import load_reviewer2_repro_bundle, run_reviewer2_repro_chain

    repo_root = Path(__file__).resolve().parents[1]
    stage = str(fields.get("stage") or "hsp").strip().lower()
    dry = bool(fields.get("dry_run"))
    bundle_path = str(
        fields.get("bundle") or "configs/experiments/manuscript_repro_bundle.json"
    ).strip()
    ms_bundle = load_manuscript_repro_bundle(bundle_path)

    if stage == "preflight":
        return run_manuscript_preflight(
            ms_bundle,
            repo_root=repo_root,
            ms_bundle_path=bundle_path,
            dry_run=dry,
        )

    if stage == "post-zoo":
        r2_path = str((ms_bundle.get("post_zoo_reviewer2") or {}).get("bundle") or "").strip()
        r2_bundle = load_reviewer2_repro_bundle(
            r2_path or "configs/experiments/reviewer2_repro.json"
        )
        return run_reviewer2_repro_chain(r2_bundle, repo_root=repo_root, dry_run=dry)

    if stage == "hsp":
        return run_manuscript_repro_chain(
            ms_bundle,
            repo_root=repo_root,
            dry_run=dry,
            skip_gpu_check=bool(fields.get("skip_gpu_check")),
            include_test_map=bool(fields.get("include_test_map")),
        )

    if stage == "full":
        return run_publication_pipeline(
            ms_bundle,
            repo_root=repo_root,
            ms_bundle_path=bundle_path,
            dry_run=dry,
            include_hsp=True,
            skip_gpu_check=bool(fields.get("skip_gpu_check")),
            include_test_map=bool(fields.get("include_test_map")),
        )

    raise ValueError(f"unknown repro --stage {stage!r} (expected hsp, post-zoo, preflight, or full)")


def _run_reviewer_counting(fields: dict[str, Any]) -> int:
    from harchoc.experiment_argv import argv_for_reviewer_counting
    from harchoc.manuscript_repro import _format_cmd
    from harchoc.reviewer_counting_report import run_reviewer_counting

    norm = dict(fields)
    out_path = str(norm.get("out") or "reports/reviewer2_counting_metrics_computed.json")

    if bool(norm.get("dry_run")):
        print("# reviewer-counting")
        print(_format_cmd(["scripts/experiment.py", *argv_for_reviewer_counting(norm)], mamba=True))
        payload = run_reviewer_counting({**norm, "dry_run": True})
        write_json(out_path, payload)
        print(f"Wrote dry-run plan {out_path}")
        return 0

    payload = run_reviewer_counting(norm)
    write_json(out_path, payload)
    print(f"Wrote {out_path}")
    mae = (payload.get("pooled") or {}).get("mae")
    if mae is not None:
        print(f"test count MAE @ locked conf: {mae:.4f}")
    return 0


def _run_dual_metric(fields: dict[str, Any]) -> int:
    from harchoc.dual_metric_report import build_dry_run_report, merge_dual_metric_from_paths
    from harchoc.experiment_argv import argv_for_dual_metric_bundle

    fields = argv_for_dual_metric_bundle(fields)
    out_path = str(fields.get("out") or "reports/dual_metric/report.json")
    eval_paths = fields.get("eval")
    eval_list: list[str] = list(eval_paths) if isinstance(eval_paths, list) else []

    inputs = {
        "eval_val": str(fields.get("eval_val") or ""),
        "eval_test": str(fields.get("eval_test") or ""),
        "eval_val_map": str(fields.get("eval_val_map") or ""),
        "eval_test_map": str(fields.get("eval_test_map") or ""),
        "sweep_val": str(fields.get("sweep") or ""),
        "sweep_test": str(fields.get("sweep_test") or ""),
        "error_val": str(fields.get("error_val") or ""),
        "error_test": str(fields.get("error_test") or ""),
    }

    if bool(fields.get("dry_run")):
        payload = build_dry_run_report(out=out_path, inputs=inputs)
        write_json(out_path, payload)
        return 0

    sweep_val = str(fields.get("sweep") or "").strip()
    error_val = str(fields.get("error_val") or "").strip()
    error_test = str(fields.get("error_test") or "").strip()
    if not sweep_val or not error_val or not error_test:
        raise SystemExit(
            "dual-metric requires --sweep, --error-val, and --error-test "
            "(plus val/test eval via --eval-val/--eval-test or --eval ×2)"
        )

    payload = merge_dual_metric_from_paths(
        eval_val=(str(fields.get("eval_val")).strip() or None) if fields.get("eval_val") else None,
        eval_test=(str(fields.get("eval_test")).strip() or None) if fields.get("eval_test") else None,
        eval_val_map=(str(fields.get("eval_val_map")).strip() or None)
        if fields.get("eval_val_map")
        else None,
        eval_test_map=(str(fields.get("eval_test_map")).strip() or None)
        if fields.get("eval_test_map")
        else None,
        eval_paths=eval_list,
        sweep_val=sweep_val,
        sweep_test=(str(fields.get("sweep_test")).strip() or None) if fields.get("sweep_test") else None,
        error_val=error_val,
        error_test=error_test,
    )
    write_json(out_path, payload)
    return 0


def _run_hpo(fields: dict[str, Any]) -> int:
    """
    Budgeted HPO planner.

    Intentionally lightweight: generates trial train configs and writes a JSON report.
    Execution of training loops is intentionally out-of-scope here (use the generated
    train configs with `scripts/train.py` / `scripts/benchmark_matrix.py`).
    """
    from harchoc.hpo_search import plan_hpo_trials

    out_path = str(fields.get("out") or "reports/hpo/hpo_plan.json")
    base_train_config = str(fields.get("base_train_config") or "").strip()
    if not base_train_config:
        raise SystemExit("hpo requires --base-train-config (path to train config JSON)")

    trials = int(fields.get("trials") or 20)
    seed = int(fields.get("seed") or 0)
    space_raw = fields.get("space")
    if space_raw is None:
        raise SystemExit("hpo requires --space (JSON object or path to JSON file)")

    if isinstance(space_raw, str):
        import json
        from pathlib import Path

        s = space_raw.strip()
        p = Path(s).expanduser()
        space = json.loads(p.read_text("utf-8")) if p.is_file() else json.loads(s)
    elif isinstance(space_raw, dict):
        space = space_raw
    else:
        raise SystemExit("--space must be a JSON object or a JSON string/path")

    payload = plan_hpo_trials(
        base_train_config=base_train_config,
        space=space,
        trials=trials,
        seed=seed,
    )
    write_json(out_path, payload)
    return 0


_DATASET_INHERIT_KEYS = (
    "manifest",
    "default_dataset_name",
    "dataset_name",
    "dataset_root",
    "yolo_data_yaml",
)


def _inherit_dataset_fields(*, dataset_cfg: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
    merged = dict(fields)
    for key in _DATASET_INHERIT_KEYS:
        if key not in merged and dataset_cfg.get(key) is not None:
            merged[key] = dataset_cfg[key]
    return merged


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Unified experiment entrypoint.")
    p.add_argument(
        "--config",
        action="append",
        default=[],
        help="Inline JSON object or path to JSON file. Can be repeated; later entries override earlier.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Force dry-run for the selected subcommand.",
    )

    sp = p.add_subparsers(dest="cmd", required=True)

    # `splits` legacy args
    ps = sp.add_parser("splits", help="Create train/val/test split lists.")
    add_dataset_args(ps, suppress_defaults=True)
    add_dry_run_arg(ps, suppress_defaults=True)
    ps.add_argument("--mode", choices=["from-folders", "random"], default=argparse.SUPPRESS)
    ps.add_argument("--out-dir", default=argparse.SUPPRESS)
    ps.add_argument("--ext", action="append", default=argparse.SUPPRESS)
    ps.add_argument("--glob", default=argparse.SUPPRESS)
    ps.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    ps.add_argument("--val-frac", type=float, default=argparse.SUPPRESS)
    ps.add_argument("--test-frac", type=float, default=argparse.SUPPRESS)

    # `describe` legacy args
    pd = sp.add_parser("describe", help="Describe dataset split stats.")
    add_dataset_args(pd, suppress_defaults=True)
    add_dry_run_arg(pd, suppress_defaults=True)
    pd.add_argument("--split", choices=["train", "val", "test"], default=argparse.SUPPRESS)
    pd.add_argument("--split-file", action="append", default=argparse.SUPPRESS)
    pd.add_argument("--out", default=argparse.SUPPRESS)

    pdr = sp.add_parser(
        "dataset-root",
        help="Print resolved dataset root from data/manifest.json.",
    )
    pdr.add_argument("--dataset-name", default=argparse.SUPPRESS)

    # `eval` legacy args
    pe = sp.add_parser("eval", help="Evaluate a trained model.")
    add_dataset_args(pe, suppress_defaults=True)
    add_dry_run_arg(pe, suppress_defaults=True)
    pe.add_argument("--weights", default=argparse.SUPPRESS)
    pe.add_argument("--split-file", default=argparse.SUPPRESS)
    pe.add_argument("--out", default=argparse.SUPPRESS)

    pts = sp.add_parser("threshold-sweep", help="Val confidence sweep; lock conf for test (HSP).")
    add_dataset_args(pts, suppress_defaults=True)
    add_dry_run_arg(pts, suppress_defaults=True)
    pts.add_argument("--weights", default=argparse.SUPPRESS)
    pts.add_argument("--out", default=argparse.SUPPRESS)
    pts.add_argument("--gt-json", default=argparse.SUPPRESS)
    pts.add_argument("--preds-json", default=argparse.SUPPRESS)
    pts.add_argument("--locked-conf-from", default=argparse.SUPPRESS)
    pts.add_argument("--select", default=argparse.SUPPRESS)
    pts.add_argument("--split-file", default=argparse.SUPPRESS)

    pea = sp.add_parser("error-analysis", help="Error analysis + optional TIDE/confusion from HSP JSON.")
    add_dataset_args(pea, suppress_defaults=True)
    add_dry_run_arg(pea, suppress_defaults=True)
    pea.add_argument("--weights", default=argparse.SUPPRESS)
    pea.add_argument("--out", default=argparse.SUPPRESS)
    pea.add_argument("--report", default=argparse.SUPPRESS)
    pea.add_argument("--gt-json", default=argparse.SUPPRESS)
    pea.add_argument("--preds-json", default=argparse.SUPPRESS)
    pea.add_argument("--locked-conf-from", default=argparse.SUPPRESS)

    psd = sp.add_parser("split-drift", help="Train/val/test proxy drift report (HSP P0).")
    add_dataset_args(psd, suppress_defaults=True)
    add_dry_run_arg(psd, suppress_defaults=True)
    psd.add_argument("--splits-dir", default=argparse.SUPPRESS)
    psd.add_argument("--out", default=argparse.SUPPRESS)
    psd.add_argument("--with-ks", action="store_true", default=argparse.SUPPRESS)
    psd.add_argument("--extended", action="store_true", default=argparse.SUPPRESS)

    # `benchmark` legacy args
    pb = sp.add_parser("benchmark", help="Run benchmark matrix harness.")
    add_dataset_args(pb, suppress_defaults=True)
    # keep name aligned with legacy file: bench-config, bench-dir, ...
    pb.add_argument("--bench-config", action="append", default=argparse.SUPPRESS)
    pb.add_argument("--bench-dir", default=argparse.SUPPRESS)
    pb.add_argument("--pattern", default=argparse.SUPPRESS)
    pb.add_argument("--limit", type=int, default=argparse.SUPPRESS)
    pb.add_argument("--out", default=argparse.SUPPRESS)
    pb.add_argument("--no-train", action="store_true", default=argparse.SUPPRESS)
    pb.add_argument("--no-eval", action="store_true", default=argparse.SUPPRESS)
    pb.add_argument("--eval-out", default=argparse.SUPPRESS)
    pb.add_argument("--no-dry-run", action="store_true", default=argparse.SUPPRESS)
    pb.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS)

    # `train` args (new)
    pt = sp.add_parser("train", help="Train a detection model with reproducible outputs.")
    add_dataset_args(pt, suppress_defaults=True)
    add_dry_run_arg(pt, suppress_defaults=True)
    pt.add_argument("--config", default=argparse.SUPPRESS)
    pt.add_argument("--out-dir", default=argparse.SUPPRESS)
    pt.add_argument("--name", default=argparse.SUPPRESS)

    ph = sp.add_parser("hpo", help="Plan a budget-capped hyperparameter search.")
    ph.add_argument("--out", default=argparse.SUPPRESS)
    ph.add_argument("--base-train-config", default=argparse.SUPPRESS)
    ph.add_argument("--space", default=argparse.SUPPRESS)
    ph.add_argument("--trials", type=int, default=argparse.SUPPRESS)
    ph.add_argument("--seed", type=int, default=argparse.SUPPRESS)

    pr = sp.add_parser(
        "repro",
        help="Run the HSP manuscript reproducibility chain from manuscript_repro_bundle.json.",
    )
    pr.add_argument(
        "--bundle",
        default=argparse.SUPPRESS,
        help="Path to manuscript_repro_bundle.v1 JSON (default: configs/experiments/manuscript_repro_bundle.json).",
    )
    add_dry_run_arg(pr, suppress_defaults=True)
    pr.add_argument(
        "--skip-gpu-check",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Skip scripts/check_gpu.py at chain start.",
    )
    pr.add_argument(
        "--include-test-map",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Append optional test mAP eval + dual-metric regen (GPU; may OOM on 8 GiB).",
    )
    pr.add_argument(
        "--stage",
        default=argparse.SUPPRESS,
        choices=("hsp", "post-zoo", "preflight", "full"),
        help="hsp=P0 chain; post-zoo=reviewer2 only; preflight=figures/tables/aug/narrative/reviewer2; full=HSP+preflight.",
    )

    pdocx = sp.add_parser(
        "manuscript-docx-repro",
        help="Reproduce docx Figures 1–6 & Tables 1–3 from HSP JSON (journal style, CPU).",
    )
    add_dry_run_arg(pdocx, suppress_defaults=True)
    pdocx.add_argument("--out-dir", default=argparse.SUPPRESS)

    pmp = sp.add_parser(
        "manuscript-preflight",
        help="Publication preflight: reviewer2 → figures → tables → docx → aug → narrative.",
    )
    add_dry_run_arg(pmp, suppress_defaults=True)
    pmp.add_argument(
        "--bundle",
        default=argparse.SUPPRESS,
        help="manuscript_repro_bundle.v1 JSON (default configs/experiments/manuscript_repro_bundle.json).",
    )

    pr2rep = sp.add_parser(
        "reviewer2-repro",
        help="Post-zoo reviewer-2 chain: counting → mAP50 → confusion → paste-check (CPU).",
    )
    add_dry_run_arg(pr2rep, suppress_defaults=True)
    pr2rep.add_argument(
        "--bundle",
        default=argparse.SUPPRESS,
        help="reviewer2_repro_bundle.v1 JSON (default configs/experiments/reviewer2_repro.json).",
    )

    pcv = sp.add_parser(
        "cv-eval",
        help="K-fold split planning and fold-metric aggregation (delegates to cv_eval.py).",
    )
    add_dataset_args(pcv, suppress_defaults=True)
    add_dry_run_arg(pcv, suppress_defaults=True)
    pcv.add_argument("--weights", default=argparse.SUPPRESS)
    pcv.add_argument("--folds", type=int, default=argparse.SUPPRESS)
    pcv.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    pcv.add_argument("--splits-dir", default=argparse.SUPPRESS)
    pcv.add_argument("--fold-metrics", action="append", default=argparse.SUPPRESS)
    pcv.add_argument("--out", default=argparse.SUPPRESS)
    pcv.add_argument("--write-fold-splits", default=argparse.SUPPRESS)

    pgc = sp.add_parser(
        "gradcam",
        help="Grad-CAM FP crop panel (delegates to make_figures.py --figure fig_gradcam_panel).",
    )
    add_dry_run_arg(pgc, suppress_defaults=True)
    pgc.add_argument("--out-dir", default=argparse.SUPPRESS)
    pgc.add_argument("--meta-out", default=argparse.SUPPRESS)
    pgc.add_argument("--error-report", default=argparse.SUPPRESS)
    pgc.add_argument("--weights", default=argparse.SUPPRESS)
    pgc.add_argument("--panel-size", type=int, default=argparse.SUPPRESS)

    pfr = sp.add_parser(
        "figures-repro",
        help="Regenerate all manuscript figures (journal style) + figures manifest JSON.",
    )
    add_dry_run_arg(pfr, suppress_defaults=True)
    pfr.add_argument(
        "--bundle",
        default=argparse.SUPPRESS,
        help="figures_repro_bundle.v1 JSON (default configs/experiments/figures_repro.json).",
    )
    pfr.add_argument("--out-dir", default=argparse.SUPPRESS)
    pfr.add_argument("--meta-out", default=argparse.SUPPRESS)
    pfr.add_argument("--manifest-out", default=argparse.SUPPRESS)
    pfr.add_argument("--split-drift-report", default=argparse.SUPPRESS)
    pfr.add_argument("--threshold-csv", default=argparse.SUPPRESS)
    pfr.add_argument("--threshold-json", default=argparse.SUPPRESS)
    pfr.add_argument("--error-report", default=argparse.SUPPRESS)
    pfr.add_argument("--weights", default=argparse.SUPPRESS)
    pfr.add_argument("--panel-size", type=int, default=argparse.SUPPRESS)
    pfr.add_argument("--figure", default=argparse.SUPPRESS)
    pfr.add_argument(
        "--journal-style",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )

    pdp = sp.add_parser(
        "deploy-parity",
        help="Deploy SAHI conf vs HSP locked conf side-by-side JSON (R-SCI-2).",
    )
    add_dataset_args(pdp, suppress_defaults=True)
    add_dry_run_arg(pdp, suppress_defaults=True)
    pdp.add_argument(
        "--locked-conf-from",
        default=argparse.SUPPRESS,
        help="Val threshold JSON for manuscript locked conf.",
    )
    pdp.add_argument("--out", default=argparse.SUPPRESS)
    pdp.add_argument(
        "--sample-images",
        type=int,
        default=argparse.SUPPRESS,
        help="Compare SAHI vs full-frame counts on N test-split images.",
    )
    pdp.add_argument("--split-file", default=argparse.SUPPRESS)
    pdp.add_argument("--weights", default=argparse.SUPPRESS)

    pft = sp.add_parser(
        "finetune-tray",
        help="Tray-targeted fine-tune from best2 (tray_adapt splits + before/after eval).",
    )
    add_dataset_args(pft, suppress_defaults=True)
    add_dry_run_arg(pft, suppress_defaults=True)
    pft.add_argument("--base-weights", default=argparse.SUPPRESS)
    pft.add_argument("--config", default=argparse.SUPPRESS)
    pft.add_argument("--transfer-config", default=argparse.SUPPRESS)
    pft.add_argument("--name", default=argparse.SUPPRESS)
    pft.add_argument("--out-dir", default=argparse.SUPPRESS)
    pft.add_argument("--out", default=argparse.SUPPRESS)
    pft.add_argument("--train-mode", default=argparse.SUPPRESS)
    pft.add_argument("--stage", type=int, default=argparse.SUPPRESS)
    pft.add_argument(
        "--tray-eval",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    pft.add_argument("--tray-key", action="append", default=argparse.SUPPRESS)
    pft.add_argument("--tray-catalog", default=argparse.SUPPRESS)
    pft.add_argument("--domains-dir", default=argparse.SUPPRESS)
    pft.add_argument("--splits-dir", default=argparse.SUPPRESS)

    pda = sp.add_parser(
        "domain-tray-audit",
        help="Rank weak trays from domain_count_mae.json (CPU; finetune planning).",
    )
    add_dry_run_arg(pda, suppress_defaults=True)
    pda.add_argument(
        "--out",
        default=argparse.SUPPRESS,
        help="weak_tray_plan.v1 JSON (default reports/domains/weak_tray_plan.json).",
    )
    pda.add_argument("--count-mae", default=argparse.SUPPRESS)
    pda.add_argument("--domain-eval", default=argparse.SUPPRESS)
    pda.add_argument("--top-k", type=int, default=argparse.SUPPRESS)
    pda.add_argument("--global-mae", type=float, default=argparse.SUPPRESS)

    pr2m = sp.add_parser(
        "reviewer2-map50",
        help="Aggregate reviewer-2 mAP@0.5 claims into reports/reviewer2_map50_computed.json.",
    )
    add_dry_run_arg(pr2m, suppress_defaults=True)
    pr2m.add_argument(
        "--out",
        default=argparse.SUPPRESS,
        help="Output JSON (default: reports/reviewer2_map50_computed.json).",
    )
    pr2m.add_argument("--eval-test-map", default=argparse.SUPPRESS)
    pr2m.add_argument("--dual-metric", default=argparse.SUPPRESS)
    pr2m.add_argument("--eval-val", default=argparse.SUPPRESS)
    pr2m.add_argument("--eval-test", default=argparse.SUPPRESS)
    pr2m.add_argument(
        "--rerun-eval",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Re-run scripts/eval.py (map-cpu settings) and cross-check mAP50 vs on-disk eval_test_map.",
    )
    add_dataset_args(pr2m, suppress_defaults=True)

    pcm = sp.add_parser(
        "map-cpu",
        help="Test-split mAP eval on CPU (SCI-MAP-CPU; delegates to eval.py).",
    )
    add_dataset_args(pcm, suppress_defaults=True)
    add_dry_run_arg(pcm, suppress_defaults=True)
    pcm.add_argument("--weights", default=argparse.SUPPRESS)
    pcm.add_argument("--split-file", default=argparse.SUPPRESS)
    pcm.add_argument("--imgsz", type=int, default=argparse.SUPPRESS)
    pcm.add_argument("--max-det", type=int, default=argparse.SUPPRESS)
    pcm.add_argument("--device", default=argparse.SUPPRESS)
    pcm.add_argument("--out", default=argparse.SUPPRESS)

    pts = sp.add_parser(
        "tune-sahi",
        help="Deploy SAHI slice/conf grid vs manual GT (dry-run argv only; live grid removed).",
    )
    add_dry_run_arg(pts, suppress_defaults=True)
    pts.add_argument(
        "--image",
        default=argparse.SUPPRESS,
        help="Reference head image (default: test_sunflower_tune.png).",
    )

    prc = sp.add_parser(
        "reviewer-counting",
        help="Pooled test/val count MAE, per-class totals, relative-error stats (CPU from GT/preds JSON).",
    )
    add_dry_run_arg(prc, suppress_defaults=True)
    prc.add_argument("--out", default=argparse.SUPPRESS)
    prc.add_argument(
        "--locked-conf-from",
        default=argparse.SUPPRESS,
        help="Val threshold_sweep_run.v1 for locked conf (default reports/hsp/threshold_val.json).",
    )
    prc.add_argument("--gt-test", default=argparse.SUPPRESS)
    prc.add_argument("--preds-test", default=argparse.SUPPRESS)
    prc.add_argument("--gt-val", default=argparse.SUPPRESS)
    prc.add_argument("--preds-val", default=argparse.SUPPRESS)
    prc.add_argument("--eval-test", default=argparse.SUPPRESS)
    prc.add_argument("--dual-metric", default=argparse.SUPPRESS)
    prc.add_argument("--split-file", default=argparse.SUPPRESS)
    prc.add_argument("--weights", default=argparse.SUPPRESS)

    pdm = sp.add_parser(
        "dual-metric",
        help="Merge eval, threshold sweep, and error-analysis JSON into a manuscript table.",
    )
    add_dry_run_arg(pdm, suppress_defaults=True)
    pdm.add_argument("--out", default=argparse.SUPPRESS)
    pdm.add_argument("--eval-val", default=argparse.SUPPRESS)
    pdm.add_argument("--eval-test", default=argparse.SUPPRESS)
    pdm.add_argument(
        "--eval-val-map",
        default=argparse.SUPPRESS,
        help="Optional eval_run.v1 with mAP when --eval-val is export-only.",
    )
    pdm.add_argument(
        "--eval-test-map",
        default=argparse.SUPPRESS,
        help="Optional eval_run.v1 with mAP when --eval-test is export-only.",
    )
    pdm.add_argument(
        "--eval",
        action="append",
        default=argparse.SUPPRESS,
        help="eval_run.v1 path; split_role val|test (use twice or with --eval-val/--eval-test).",
    )
    pdm.add_argument("--sweep", default=argparse.SUPPRESS, help="Val threshold_sweep_run.v1 JSON.")
    pdm.add_argument("--sweep-test", default=argparse.SUPPRESS, help="Optional test sweep with locked block.")
    pdm.add_argument("--error-val", default=argparse.SUPPRESS)
    pdm.add_argument("--error-test", default=argparse.SUPPRESS)

    pr2 = sp.add_parser(
        "reviewer2-confusion",
        help="§11 confusion + TIDE bucket audit JSON from HSP test exports (CPU; no re-inference).",
    )
    add_dry_run_arg(pr2, suppress_defaults=True)
    pr2.add_argument("--out", default=argparse.SUPPRESS)
    pr2.add_argument("--gt-json", default=argparse.SUPPRESS)
    pr2.add_argument("--preds-json", default=argparse.SUPPRESS)
    pr2.add_argument("--locked-conf-from", default=argparse.SUPPRESS)
    pr2.add_argument("--error-report", default=argparse.SUPPRESS)
    pr2.add_argument("--tide-summary", default=argparse.SUPPRESS)
    pr2.add_argument(
        "--confusion-iou-0-3",
        default=argparse.SUPPRESS,
        help="Stored IoU 0.3 matrix JSON (default reports/hsp/best2_test_confusion.json).",
    )

    pfb = sp.add_parser(
        "fp-budget-sweep",
        help="Val constraint-sweep ablation: count-first vs F1-max vs max_fp_per_image grid (P1-FP-BUDGET).",
    )
    add_dataset_args(pfb, suppress_defaults=True)
    add_dry_run_arg(pfb, suppress_defaults=True)
    pfb.add_argument("--out", default=argparse.SUPPRESS)
    pfb.add_argument("--gt-json", default=argparse.SUPPRESS)
    pfb.add_argument("--preds-json", default=argparse.SUPPRESS)
    pfb.add_argument(
        "--sweep-from",
        default=argparse.SUPPRESS,
        help="Reuse rows from threshold_sweep_run.v1 (e.g. reports/hsp/threshold_val.json).",
    )
    pfb.add_argument("--split-file", default=argparse.SUPPRESS)
    pfb.add_argument("--iou", type=float, default=argparse.SUPPRESS)
    pfb.add_argument(
        "--fp-budget-grid",
        type=float,
        action="append",
        default=argparse.SUPPRESS,
        help="Repeatable max_fp_per_image caps (default grid in harchoc.fp_budget_sweep).",
    )
    pfb.add_argument("--light", action="store_true", default=argparse.SUPPRESS)

    pvas = sp.add_parser(
        "validate-aug-smoke",
        help="Validate aug_smoke_index.json vs runtime train/aug configs (CI-safe, no ML deps).",
    )
    pvas.add_argument(
        "--index",
        default=argparse.SUPPRESS,
        help="aug_smoke_index.v1 JSON (default configs/experiments/aug_smoke_index.json).",
    )

    pr2 = sp.add_parser(
        "reviewer2-paste-check",
        help="Validate reviewer2 paste sources, SOTA inventory vs zoo matrix, docx claim gaps (CI-safe).",
    )
    pr2.add_argument("--docx", default=argparse.SUPPRESS)
    pr2.add_argument("--checklist", default=argparse.SUPPRESS)
    pr2.add_argument("--inventory-md", default=argparse.SUPPRESS)
    pr2.add_argument("--inventory-json", default=argparse.SUPPRESS)
    pr2.add_argument("--matrix-rows", default=argparse.SUPPRESS)
    pr2.add_argument("--out", default=argparse.SUPPRESS)
    add_dry_run_arg(pr2, suppress_defaults=True)

    pbn = sp.add_parser(
        "backlog-narrative",
        help="IMRaD-ish manuscript narrative from backlog.md (CPU, no LLM).",
    )
    pbn.add_argument(
        "--backlog",
        default=argparse.SUPPRESS,
        help="backlog.md path (default: repo backlog.md).",
    )
    pbn.add_argument(
        "--out-md",
        default=argparse.SUPPRESS,
        help="Markdown output (default: reports/manuscript/narrative_from_backlog.md).",
    )
    pbn.add_argument(
        "--out-json",
        default=argparse.SUPPRESS,
        help="JSON output (default: reports/manuscript/backlog_narrative.json).",
    )
    add_dry_run_arg(pbn, suppress_defaults=True)

    pal = sp.add_parser(
        "aug-leaderboard",
        help="Regenerate reports/aug_smoke/leaderboard.md from index + summaries (P2-AUG-RANK-REPORT).",
    )
    pal.add_argument(
        "--index",
        default=argparse.SUPPRESS,
        help="aug_smoke_index.v1 JSON (default configs/experiments/aug_smoke_index.json).",
    )
    pal.add_argument(
        "--out-dir",
        default=argparse.SUPPRESS,
        help="Output directory (default reports/aug_smoke).",
    )
    pal.add_argument(
        "--no-json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Skip leaderboard.json.",
    )

    ptr = sp.add_parser(
        "tables-repro",
        help="Publication tables (headline, aug top-N, zoo_core) under reports/manuscript/tables (CPU).",
    )
    add_dry_run_arg(ptr, suppress_defaults=True)
    ptr.add_argument(
        "--out-dir",
        default=argparse.SUPPRESS,
        help="Output directory (default reports/manuscript/tables).",
    )
    ptr.add_argument("--dual-metric", default=argparse.SUPPRESS)
    ptr.add_argument("--matrix-train", default=argparse.SUPPRESS)
    ptr.add_argument("--matrix-rows", default=argparse.SUPPRESS)
    ptr.add_argument("--aug-index", default=argparse.SUPPRESS)
    ptr.add_argument("--aug-out-dir", default=argparse.SUPPRESS)
    ptr.add_argument(
        "--aug-leaderboard-json",
        default=argparse.SUPPRESS,
        help="Optional aug_smoke_leaderboard.v1 JSON (default reports/aug_smoke/leaderboard.json).",
    )
    ptr.add_argument("--model-label", default=argparse.SUPPRESS)
    ptr.add_argument("--matrix-group", default=argparse.SUPPRESS)
    ptr.add_argument("--top-n", type=int, default=argparse.SUPPRESS)
    ptr.add_argument(
        "--latex",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Also write optional .tex fragments per table.",
    )

    pnts = sp.add_parser(
        "now-todos-smoke",
        help="Run testable smoke checks for now_todos.md (CPU default; GPU with HARCHOC_RUN_GPU_SMOKE=1).",
    )
    pnts.add_argument(
        "--bundle",
        default="configs/experiments/now_todos_smoke_bundle.json",
        help="now_todos_smoke_bundle.v1 JSON.",
    )
    pnts.add_argument(
        "--stage",
        default="cpu",
        choices=("verify", "cpu", "gpu", "all"),
        help="Stage group to run (default cpu).",
    )

    pac = sp.add_parser(
        "aug-compare",
        help="Aug comparative analysis JSON, figure, and narrative from index + summaries (CPU-only).",
    )
    pac.add_argument(
        "--index",
        default=argparse.SUPPRESS,
        help="aug_smoke_index.v1 JSON (default configs/experiments/aug_smoke_index.json).",
    )
    pac.add_argument(
        "--out-dir",
        default=argparse.SUPPRESS,
        help="Output directory (default reports/aug_smoke).",
    )
    pac.add_argument(
        "--no-figure",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Skip fig_aug_mae_comparison.png.",
    )
    add_dry_run_arg(pac, suppress_defaults=True)

    args = p.parse_args(argv)

    # Merge all --config entries (left-to-right).
    config_obj: dict[str, Any] = {}
    for raw in args.config:
        cfg = load_config_json(raw)
        config_obj = merge_experiment_config(config=config_obj, cli=cfg)

    dataset_cfg = config_obj.get("dataset")
    dataset_cfg_obj = dataset_cfg if isinstance(dataset_cfg, dict) else {}

    cmd = str(args.cmd)
    cli_fields = _ns_to_dict(args)
    # Remove top-level keys not relevant to underlying scripts.
    cli_fields.pop("cmd", None)
    cli_fields.pop("config", None)
    force_dry_run = bool(args.dry_run)
    cli_fields.pop("dry_run", None)  # global dry-run handled below

    merged_fields = _merged_for_subcommand(
        config_obj=config_obj,
        subcommand=cmd,
        cli_fields=cli_fields,
        force_dry_run=force_dry_run,
    )
    merged_fields = _inherit_dataset_fields(dataset_cfg=dataset_cfg_obj, fields=merged_fields)

    if cmd == "splits":
        from harchoc.experiment_argv import argv_for_splits
        from scripts.make_splits import main as legacy_main

        return legacy_main(argv_for_splits(merged_fields))
    if cmd == "describe":
        from harchoc.experiment_argv import argv_for_describe
        from scripts.describe_split import main as legacy_main

        return legacy_main(argv_for_describe(merged_fields))
    if cmd == "eval":
        from harchoc.experiment_argv import argv_for_eval
        from scripts.eval import main as legacy_main

        return legacy_main(argv_for_eval(merged_fields))
    if cmd == "threshold-sweep":
        from harchoc.experiment_argv import argv_for_threshold_sweep
        from scripts.threshold_sweep import main as legacy_main

        return legacy_main(argv_for_threshold_sweep(merged_fields))
    if cmd == "error-analysis":
        from harchoc.experiment_argv import argv_for_error_analysis
        from scripts.error_analysis import main as legacy_main

        return legacy_main(argv_for_error_analysis(merged_fields))
    if cmd == "split-drift":
        from harchoc.experiment_argv import argv_for_split_drift
        from scripts.split_drift import main as legacy_main

        return legacy_main(argv_for_split_drift(merged_fields))
    if cmd == "benchmark":
        from harchoc.experiment_argv import argv_for_benchmark
        from scripts.benchmark_matrix import main as legacy_main

        return legacy_main(argv_for_benchmark(merged_fields))
    if cmd == "train":
        from harchoc.experiment_argv import argv_for_train
        from scripts.train import main as legacy_main

        return legacy_main(argv_for_train(merged_fields))
    if cmd == "hpo":
        return _run_hpo(merged_fields)
    if cmd == "reviewer-counting":
        return _run_reviewer_counting(merged_fields)
    if cmd == "dual-metric":
        return _run_dual_metric(merged_fields)
    if cmd == "reviewer2-confusion":
        return _run_reviewer2_confusion(merged_fields)
    if cmd == "fp-budget-sweep":
        return _run_fp_budget_sweep(merged_fields)
    if cmd == "cv-eval":
        from harchoc.experiment_argv import argv_for_cv_eval
        from scripts.cv_eval import main as legacy_main

        return legacy_main(argv_for_cv_eval(merged_fields))
    if cmd == "gradcam":
        return _run_gradcam(merged_fields)
    if cmd == "figures-repro":
        return _run_figures_repro(merged_fields)
    if cmd == "deploy-parity":
        return _run_deploy_parity(merged_fields)
    if cmd == "finetune-tray":
        return _run_finetune_tray(merged_fields)
    if cmd == "domain-tray-audit":
        return _run_domain_tray_audit(merged_fields)
    if cmd == "map-cpu":
        return _run_map_cpu(merged_fields)
    if cmd == "reviewer2-map50":
        return _run_reviewer2_map50(merged_fields)
    if cmd == "tune-sahi":
        return _run_tune_sahi(merged_fields)
    if cmd == "reviewer2-repro":
        return _run_reviewer2_repro(merged_fields)
    if cmd == "manuscript-docx-repro":
        from harchoc.manuscript_docx_repro import run_manuscript_docx_repro

        repo_root = Path(__file__).resolve().parents[1]
        out_dir = str(merged_fields.get("out_dir") or "reports/manuscript/docx")
        if bool(merged_fields.get("dry_run")):
            run_manuscript_docx_repro(repo_root, out_dir=out_dir, dry_run=True)
            return 0
        run_manuscript_docx_repro(repo_root, out_dir=out_dir, dry_run=False)
        print(f"Wrote {out_dir}/catalog.json and figures/tables (see README.md)")
        return 0
    if cmd == "manuscript-preflight":
        return _run_manuscript_preflight(merged_fields)
    if cmd == "repro":
        return _run_repro(merged_fields)
    if cmd == "validate-aug-smoke":
        from harchoc.aug_smoke_train import validate_aug_smoke_configs

        repo_root = Path(__file__).resolve().parents[1]
        index_arg = merged_fields.get("index") or cli_fields.get("index")
        errors = validate_aug_smoke_configs(
            repo_root,
            index_path=str(index_arg) if index_arg else None,
        )
        if errors:
            for err in errors:
                print(f"ERROR: {err}", file=sys.stderr)
            return 1
        print("aug_smoke config validation OK")
        return 0
    if cmd == "reviewer2-paste-check":
        from harchoc.manuscript_repro import _format_cmd
        from harchoc.reviewer2_paste_check import run_reviewer2_paste_check

        repo_root = Path(__file__).resolve().parents[1]
        out_path = str(merged_fields.get("out") or "reports/reviewer2_paste_check.json")
        paste_argv = ["scripts/experiment.py", "reviewer2-paste-check", "--out", out_path]
        if bool(merged_fields.get("dry_run")):
            print("# reviewer2-paste-check")
            print(_format_cmd(paste_argv, mamba=False))
        report = run_reviewer2_paste_check(
            repo_root,
            docx_path=str(merged_fields.get("docx") or "reports/plants-4336582.docx"),
            checklist_path=str(
                merged_fields.get("checklist") or "reports/_llm/docx_paste_checklist.md"
            ),
            inventory_md=str(
                merged_fields.get("inventory_md") or "reports/_llm/sota_inventory.md"
            ),
            inventory_json=str(
                merged_fields.get("inventory_json") or "reports/reviewer2_sota_inventory.json"
            ),
            matrix_rows_path=str(
                merged_fields.get("matrix_rows") or "configs/zoo/matrix_rows.v1.json"
            ),
            out_path=out_path,
            drift_md_path=str(
                merged_fields.get("drift_md")
                or "reports/manuscript/docx_vs_submission.md"
            ),
            strict_docx=bool(merged_fields.get("strict_docx")),
            write_drift_md=not bool(merged_fields.get("no_drift_md")),
        )
        summary = report["summary"]
        print(
            f"manuscript parity check: {report['status']} "
            f"(pass={summary['pass']} warn={summary['warn']} fail={summary['fail']})"
        )
        print(f"Wrote {report['out']}")
        if report.get("docx_gaps"):
            print(f"docx gaps: {len(report['docx_gaps'])}")
        if bool(merged_fields.get("dry_run")):
            return 0
        return 1 if summary.get("fail") else 0
    if cmd == "backlog-narrative":
        from harchoc.backlog_narrative import run_backlog_narrative

        return run_backlog_narrative(Path(__file__).resolve().parents[1], merged_fields)
    if cmd == "aug-leaderboard":
        from harchoc.aug_smoke_leaderboard import write_aug_smoke_leaderboard

        repo_root = Path(__file__).resolve().parents[1]
        paths = write_aug_smoke_leaderboard(
            repo_root=repo_root,
            index_path=str(
                merged_fields.get("index")
                or cli_fields.get("index")
                or "configs/experiments/aug_smoke_index.json"
            ),
            out_dir=str(
                merged_fields.get("out_dir")
                or cli_fields.get("out_dir")
                or "reports/aug_smoke"
            ),
            write_json=not bool(merged_fields.get("no_json") or cli_fields.get("no_json")),
        )
        for label, path in paths.items():
            print(f"Wrote {label}: {path}")
        return 0
    if cmd == "tables-repro":
        return _run_tables_repro(merged_fields)
    if cmd == "now-todos-smoke":
        from harchoc.now_todos_smoke import run_now_todos_smoke

        repo_root = Path(__file__).resolve().parents[1]
        bundle = str(merged_fields.get("bundle") or cli_fields.get("bundle") or "")
        stage = str(merged_fields.get("stage") or cli_fields.get("stage") or "cpu")
        payload, rc = run_now_todos_smoke(
            repo_root,
            bundle_path=bundle or "configs/experiments/now_todos_smoke_bundle.json",
            stage_group=stage,
        )
        print(
            f"now-todos-smoke [{stage}]: {payload['overall_status']} "
            f"(ok={payload['n_ok']} skip={payload['n_skip']} fail={payload['n_fail']})"
        )
        for row in payload.get("stages") or []:
            print(f"  {row['id']}: {row['status']} — {row['detail'][:120]}")
        return rc
    if cmd == "aug-compare":
        return _run_aug_compare(merged_fields)
    if cmd == "dataset-root":
        from harchoc.datasets import dataset_root_from_manifest

        name = str(
            merged_fields.get("dataset_name")
            or os.environ.get("DATASET_NAME", "sunflower-cvat-1093")
        )
        print(dataset_root_from_manifest(dataset_name=name))
        return 0
    raise SystemExit(f"Unknown subcommand: {cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
