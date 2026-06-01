"""Typed argv builders for experiment.py subcommands (narrow DRY scope)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harchoc.hsp_weights import HSP_DETECTION_WEIGHTS


def _bool_flag(name: str, v: object) -> list[str]:
    return [name] if bool(v) else []


def _opt(name: str, v: object) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str) and not v.strip():
        return []
    return [name, str(v)]


def _opt_repeat(name: str, vs: object) -> list[str]:
    if not isinstance(vs, list):
        return []
    out: list[str] = []
    for v in vs:
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        out.extend([name, s])
    return out


def _argv_dataset(fields: dict[str, Any], *, full: bool = True) -> list[str]:
    out: list[str] = []
    out += _opt("--manifest", fields.get("manifest"))
    out += _opt("--default-dataset-name", fields.get("default_dataset_name"))
    if full:
        out += _opt("--dataset-name", fields.get("dataset_name"))
        out += _opt("--dataset-root", fields.get("dataset_root"))
        out += _opt("--yolo-data-yaml", fields.get("yolo_data_yaml"))
    return out


def argv_for_describe(fields: dict[str, Any]) -> list[str]:
    """Argv tail for ``scripts/describe_split.py`` via ``experiment.py describe``."""
    out: list[str] = []
    out += _argv_dataset(fields)
    out += _bool_flag("--dry-run", fields.get("dry_run"))
    out += _opt("--split", fields.get("split"))
    out += _opt_repeat("--split-file", fields.get("split_file"))
    out += _opt("--out", fields.get("out"))
    return out


def argv_for_train(fields: dict[str, Any]) -> list[str]:
    """Argv tail for ``scripts/train.py`` via ``experiment.py train``."""
    out: list[str] = []
    out += _argv_dataset(fields)
    out += _bool_flag("--dry-run", fields.get("dry_run"))
    out += _opt("--config", fields.get("config"))
    out += _opt("--out-dir", fields.get("out_dir"))
    out += _opt("--name", fields.get("name"))
    out += _opt("--aug-config", fields.get("aug_config"))
    out += _bool_flag("--skip-eval", fields.get("skip_eval"))
    out += _opt("--eval-out", fields.get("eval_out"))
    return out


def argv_for_splits(fields: dict[str, Any]) -> list[str]:
    """Argv tail for ``scripts/make_splits.py`` via ``experiment.py splits``."""
    out: list[str] = []
    out += _argv_dataset(fields, full=False)
    out += _bool_flag("--dry-run", fields.get("dry_run"))
    out += _opt("--mode", fields.get("mode"))
    out += _opt("--out-dir", fields.get("out_dir"))
    out += _opt_repeat("--ext", fields.get("ext"))
    out += _opt("--glob", fields.get("glob"))
    out += _opt("--seed", fields.get("seed"))
    out += _opt("--val-frac", fields.get("val_frac"))
    out += _opt("--test-frac", fields.get("test_frac"))
    return out


def argv_for_benchmark(fields: dict[str, Any]) -> list[str]:
    """Argv tail for ``scripts/benchmark_matrix.py`` via ``experiment.py benchmark``."""
    out: list[str] = []
    out += _argv_dataset(fields, full=False)

    dry_run = bool(fields.get("dry_run", True))
    if dry_run:
        out += ["--dry-run"]
    else:
        out += ["--no-dry-run"]

    out += _opt_repeat("--bench-config", fields.get("bench_config"))
    out += _opt("--bench-dir", fields.get("bench_dir"))
    out += _opt("--pattern", fields.get("pattern"))
    out += _opt("--limit", fields.get("limit"))
    out += _opt("--out", fields.get("out"))
    out += _bool_flag("--no-train", fields.get("no_train"))
    out += _bool_flag("--no-eval", fields.get("no_eval"))
    out += _opt("--eval-out", fields.get("eval_out"))
    out += _bool_flag("--sahi-eval", fields.get("sahi_eval"))
    return out


def argv_for_threshold_sweep(fields: dict[str, Any]) -> list[str]:
    """Argv tail for ``scripts/threshold_sweep.py`` via ``experiment.py threshold-sweep``."""
    out: list[str] = []
    out += _argv_dataset(fields)
    out += _bool_flag("--dry-run", fields.get("dry_run"))
    out += _bool_flag("--light", fields.get("light"))
    out += _opt("--weights", fields.get("weights"))
    out += _opt("--out", fields.get("out"))
    out += _opt("--csv-out", fields.get("csv_out"))
    out += _opt("--gt-json", fields.get("gt_json"))
    out += _opt("--preds-json", fields.get("preds_json"))
    out += _opt("--split-file", fields.get("split_file"))
    out += _opt("--locked-conf-from", fields.get("locked_conf_from"))
    out += _opt("--sweep-from", fields.get("sweep_from"))
    out += _opt("--select", fields.get("select"))
    out += _opt("--iou", fields.get("iou"))
    out += _opt("--min", fields.get("tmin"))
    out += _opt("--max", fields.get("tmax"))
    out += _opt("--steps", fields.get("steps"))
    out += _opt("--fixed-conf", fields.get("fixed_conf"))
    out += _opt("--fp-budget-sweep-out", fields.get("fp_budget_sweep_out"))
    out += _opt_repeat("--fp-budget-grid", fields.get("fp_budget_grid"))
    out += _bool_flag("--allow-test-tuning", fields.get("allow_test_tuning"))
    out += _bool_flag("--class-agnostic", fields.get("class_agnostic"))
    out += _bool_flag("--calibration-metrics", fields.get("calibration_metrics"))
    out += _opt("--calibrate", fields.get("calibrate"))
    return out


def argv_for_error_analysis(fields: dict[str, Any]) -> list[str]:
    """Argv tail for ``scripts/error_analysis.py`` via ``experiment.py error-analysis``."""
    out: list[str] = []
    out += _argv_dataset(fields)
    out += _bool_flag("--dry-run", fields.get("dry_run"))
    out += _opt("--weights", fields.get("weights"))
    out += _opt("--out", fields.get("out"))
    out += _opt("--report", fields.get("report"))
    out += _opt("--gt-json", fields.get("gt_json"))
    out += _opt("--preds-json", fields.get("preds_json"))
    out += _opt("--locked-conf-from", fields.get("locked_conf_from"))
    out += _opt("--conf", fields.get("conf"))
    out += _opt("--iou", fields.get("iou"))
    out += _opt("--iou-bg", fields.get("iou_bg"))
    out += _opt("--tide-out", fields.get("tide_out"))
    out += _opt("--confusion-matrix-out", fields.get("confusion_matrix_out"))
    out += _bool_flag("--export-fp-crops", fields.get("export_fp_crops"))
    out += _opt("--fp-crops-dir", fields.get("fp_crops_dir"))
    return out


def argv_for_split_drift(fields: dict[str, Any]) -> list[str]:
    """Argv tail for ``scripts/split_drift.py`` via ``experiment.py split-drift``."""
    out: list[str] = []
    out += _argv_dataset(fields)
    out += _bool_flag("--dry-run", fields.get("dry_run"))
    out += _opt("--splits-dir", fields.get("splits_dir"))
    out += _opt("--out", fields.get("out"))
    out += _bool_flag("--with-ks", fields.get("with_ks"))
    out += _opt("--ks-limit", fields.get("ks_limit"))
    out += _opt("--acceptance-config", fields.get("acceptance_config"))
    out += _bool_flag("--emit-plots", fields.get("emit_plots"))
    out += _opt("--plots-dir", fields.get("plots_dir"))
    out += _bool_flag("--extended", fields.get("extended"))
    out += _opt("--extended-limit", fields.get("extended_limit"))
    out += _opt("--catalog", fields.get("catalog"))
    return out


def argv_for_eval(fields: dict[str, Any]) -> list[str]:
    out: list[str] = []
    out += _argv_dataset(fields)
    out += _bool_flag("--dry-run", fields.get("dry_run"))
    out += _opt("--weights", fields.get("weights"))
    out += _opt("--split-file", fields.get("split_file"))
    out += _opt("--out", fields.get("out"))
    out += _opt("--imgsz", fields.get("imgsz"))
    out += _opt("--max-det", fields.get("max_det"))
    out += _opt("--export-gt-json", fields.get("export_gt_json"))
    out += _opt("--export-preds-json", fields.get("export_preds_json"))
    out += _opt("--export-conf", fields.get("export_conf"))
    out += _opt("--locked-conf-from", fields.get("locked_conf_from"))
    out += _bool_flag("--export-only", fields.get("export_only"))
    out += _opt("--export-device", fields.get("export_device"))
    out += _opt("--device", fields.get("device"))
    return out


def argv_for_cv_eval(fields: dict[str, Any]) -> list[str]:
    """Argv tail for ``scripts/cv_eval.py`` via ``experiment.py cv-eval``."""
    out: list[str] = []
    out += _argv_dataset(fields)
    out += _bool_flag("--dry-run", fields.get("dry_run"))
    out += _opt("--weights", fields.get("weights"))
    out += _opt("--folds", fields.get("folds"))
    out += _opt("--seed", fields.get("seed"))
    out += _opt("--splits-dir", fields.get("splits_dir"))
    fold_metrics = fields.get("fold_metrics")
    if fold_metrics is None:
        metrics_list: list[str] = []
    elif isinstance(fold_metrics, list):
        metrics_list = [str(x) for x in fold_metrics if str(x).strip()]
    else:
        metrics_list = [str(fold_metrics)] if str(fold_metrics).strip() else []
    out += _opt_repeat("--fold-metrics", metrics_list)
    out += _opt("--out", fields.get("out"))
    out += _opt("--write-fold-splits", fields.get("write_fold_splits"))
    return out


def argv_for_gradcam(fields: dict[str, Any]) -> list[str]:
    """Argv tail for ``scripts/make_figures.py`` via ``experiment.py gradcam``."""
    out: list[str] = []
    out += _opt("--out-dir", fields.get("out_dir") or "reports/figures")
    out += _opt("--meta-out", fields.get("meta_out") or "reports/figures/run.json")
    out += _opt(
        "--error-report",
        fields.get("error_report") or "reports/hsp/error_test_report.json",
    )
    out += _opt("--weights", fields.get("weights") or HSP_DETECTION_WEIGHTS)
    out += _opt("--panel-size", fields.get("panel_size") or 12)
    out += _opt("--figure", fields.get("figure") or "fig_gradcam_panel")
    return out


def argv_for_figures_repro(fields: dict[str, Any]) -> list[str]:
    """Argv tail for ``scripts/make_figures.py`` via ``experiment.py figures-repro``."""
    from harchoc.figures_repro import default_figures_repro_fields

    defaults = default_figures_repro_fields()
    out: list[str] = []
    out += _opt("--out-dir", fields.get("out_dir") or defaults["out_dir"])
    out += _opt("--meta-out", fields.get("meta_out") or defaults["meta_out"])
    out += _opt(
        "--split-drift-report",
        fields.get("split_drift_report") or defaults["split_drift_report"],
    )
    out += _opt("--threshold-csv", fields.get("threshold_csv") or defaults["threshold_csv"])
    out += _opt("--threshold-json", fields.get("threshold_json") or defaults["threshold_json"])
    if "error_report" in fields:
        err = str(fields.get("error_report") or "").strip()
        if err:
            out += _opt("--error-report", err)
    else:
        out += _opt("--error-report", defaults["error_report"])
    out += _opt("--weights", fields.get("weights"))
    out += _opt("--panel-size", fields.get("panel_size") or defaults["panel_size"])
    out += _opt("--figure", fields.get("figure") or defaults["figure"])
    if fields.get("journal_style") is False:
        out.append("--no-journal-style")
    else:
        out.append("--journal-style")
    return out


def argv_for_finetune(fields: dict[str, Any]) -> list[str]:
    """Argv tail for ``scripts/finetune.py`` via ``experiment.py finetune-tray``."""
    out: list[str] = []
    out += _argv_dataset(fields)
    out += _bool_flag("--dry-run", fields.get("dry_run"))
    out += _opt("--base-weights", fields.get("base_weights") or HSP_DETECTION_WEIGHTS)
    out += _opt("--config", fields.get("config"))
    out += _opt("--transfer-config", fields.get("transfer_config"))
    out += _opt("--name", fields.get("name"))
    out += _opt("--out-dir", fields.get("out_dir"))
    out += _opt("--out", fields.get("out"))
    out += _opt("--train-mode", fields.get("train_mode"))
    stage = fields.get("stage")
    if stage is not None:
        out += _opt("--stage", int(stage))
    if fields.get("tray_eval") is False:
        out.append("--no-tray-eval")
    out += _opt_repeat("--tray-key", fields.get("tray_key") or fields.get("tray_keys"))
    out += _opt("--tray-catalog", fields.get("tray_catalog"))
    out += _opt("--domains-dir", fields.get("domains_dir"))
    out += _opt("--splits-dir", fields.get("splits_dir"))
    out += _bool_flag("--from-weak-plan", fields.get("from_weak_plan"))
    out += _bool_flag("--audit-trays", fields.get("audit_trays"))
    out += _opt("--weak-plan", fields.get("weak_plan"))
    out += _opt("--locked-conf-from", fields.get("locked_conf_from"))
    out += _opt("--global-mae-ref", fields.get("global_mae_ref"))
    out += _opt("--canonical-gate-pct", fields.get("canonical_gate_pct"))
    out += _opt("--tide-summary", fields.get("tide_summary"))
    out += _bool_flag("--debug", fields.get("debug"))
    if fields.get("hsp_counting") is False:
        out.append("--no-hsp-counting")
    return out


def argv_for_domain_tray_audit(fields: dict[str, Any]) -> list[str]:
    """Fields consumed by ``experiment.py domain-tray-audit`` (in-process, no argv)."""
    return []


def argv_for_deploy_parity(fields: dict[str, Any]) -> list[str]:
    """Logical argv for ``experiment.py deploy-parity`` (writes JSON via deploy_hsp_parity)."""
    out: list[str] = []
    out += _argv_dataset(fields)
    out += _opt("--locked-conf-from", fields.get("locked_conf_from"))
    out += _opt("--out", fields.get("out") or "reports/hsp/deploy_hsp_parity.json")
    out += _opt("--split-file", fields.get("split_file") or "data/splits/test.txt")
    out += _opt("--weights", fields.get("weights") or HSP_DETECTION_WEIGHTS)
    sample = fields.get("sample_images")
    if sample is not None and int(sample) > 0:
        out += _opt("--sample-images", int(sample))
    return out


def argv_for_tune_sahi(fields: dict[str, Any]) -> list[str]:
    """Argv tail for ``experiment.py tune-sahi`` (dry-run planning only)."""
    image = fields.get("image") or "test_sunflower_tune.png"
    return [str(image)]


def argv_for_map_cpu(fields: dict[str, Any]) -> list[str]:
    """Argv tail for ``scripts/eval.py`` via ``experiment.py map-cpu`` (SCI-MAP-CPU)."""
    out: list[str] = []
    out += _argv_dataset(fields)
    out += _opt("--weights", fields.get("weights") or HSP_DETECTION_WEIGHTS)
    out += _opt("--split-file", fields.get("split_file") or "data/splits/test.txt")
    out += _opt("--imgsz", fields.get("imgsz") or 1280)
    out += _opt("--max-det", fields.get("max_det") or 3000)
    out += _opt("--device", fields.get("device") or "cpu")
    out += _opt("--out", fields.get("out") or "reports/hsp/eval_test_map.json")
    return out


def argv_for_dual_metric_bundle(fields: dict[str, Any]) -> dict[str, Any]:
    """Return normalized field dict for dual-metric (used by experiment.py)."""
    eval_paths = fields.get("eval")
    if eval_paths is None:
        eval_list: list[str] = []
    elif isinstance(eval_paths, list):
        eval_list = [str(x) for x in eval_paths if str(x).strip()]
    else:
        eval_list = [str(eval_paths)] if str(eval_paths).strip() else []

    return {
        "dry_run": bool(fields.get("dry_run")),
        "out": fields.get("out"),
        "eval": eval_list,
        "eval_val": fields.get("eval_val"),
        "eval_test": fields.get("eval_test"),
        "eval_val_map": fields.get("eval_val_map"),
        "eval_test_map": fields.get("eval_test_map"),
        "sweep": fields.get("sweep"),
        "sweep_test": fields.get("sweep_test"),
        "error_val": fields.get("error_val"),
        "error_test": fields.get("error_test"),
    }


def argv_for_dual_metric(fields: dict[str, Any]) -> list[str]:
    """Argv tail for ``scripts/experiment.py dual-metric`` (subcommand + flags)."""
    norm = argv_for_dual_metric_bundle(fields)
    out: list[str] = ["dual-metric"]
    out += _bool_flag("--dry-run", norm.get("dry_run"))
    out += _opt("--out", norm.get("out"))
    out += _opt("--eval-val", norm.get("eval_val"))
    out += _opt("--eval-test", norm.get("eval_test"))
    out += _opt("--eval-val-map", norm.get("eval_val_map"))
    out += _opt("--eval-test-map", norm.get("eval_test_map"))
    out += _opt_repeat("--eval", norm.get("eval"))
    out += _opt("--sweep", norm.get("sweep"))
    out += _opt("--sweep-test", norm.get("sweep_test"))
    out += _opt("--error-val", norm.get("error_val"))
    out += _opt("--error-test", norm.get("error_test"))
    return out


def dual_metric_fields_from_bundle_art(
    art: dict[str, Any], *, include_test_map: bool = False
) -> dict[str, Any]:
    """Field dict for dual-metric from manuscript_repro_bundle ``artifacts``."""
    fields: dict[str, Any] = {
        "eval_val": art["eval_val"],
        "eval_test": art["eval_test"],
        "sweep": art["threshold_val"],
        "sweep_test": art["threshold_test_locked"],
        "error_val": art["error_val"],
        "error_test": art["error_test"],
        "out": art["dual_metric"],
    }
    if include_test_map:
        fields["eval_test_map"] = art["eval_test_map"]
    return fields


def argv_for_fp_budget_sweep(fields: dict[str, Any]) -> list[str]:
    """Argv tail for ``scripts/experiment.py fp-budget-sweep``."""
    out: list[str] = ["fp-budget-sweep"]
    out += _argv_dataset(fields)
    out += _bool_flag("--dry-run", fields.get("dry_run"))
    out += _bool_flag("--light", fields.get("light"))
    out += _opt("--out", fields.get("out"))
    out += _opt("--gt-json", fields.get("gt_json"))
    out += _opt("--preds-json", fields.get("preds_json"))
    out += _opt("--sweep-from", fields.get("sweep_from"))
    out += _opt("--split-file", fields.get("split_file"))
    out += _opt("--locked-conf-from", fields.get("locked_conf_from"))
    out += _opt("--summary-out", fields.get("summary_out"))
    out += _opt("--iou", fields.get("iou"))
    out += _opt_repeat("--fp-budget-grid", fields.get("fp_budget_grid"))
    return out


def argv_for_validate_splits(fields: dict[str, Any]) -> list[str]:
    """Argv tail for ``scripts/validate_splits.py`` via ``experiment.py validate-splits``."""
    out: list[str] = []
    out += _argv_dataset(fields, full=True)
    out += _bool_flag("--dry-run", fields.get("dry_run"))
    out += _opt("--splits-dir", fields.get("splits_dir"))
    out += _bool_flag("--require-test", fields.get("require_test"))
    out += _bool_flag("--check-rtdetr-query-cap", fields.get("check_rtdetr_query_cap"))
    if fields.get("num_queries") is not None:
        out += ["--num-queries", str(int(fields["num_queries"]))]
    if fields.get("documented_peak_gt_boxes") is not None:
        out += ["--documented-peak-gt-boxes", str(int(fields["documented_peak_gt_boxes"]))]
    out += _bool_flag("--audit-leakage", fields.get("audit_leakage"))
    out += _opt("--audit-leakage-out", fields.get("audit_leakage_out"))
    out += _opt("--group-key", fields.get("group_key"))
    return out


def argv_for_reviewer_counting(fields: dict[str, Any]) -> list[str]:
    """Argv tail for ``scripts/experiment.py reviewer-counting``."""
    out: list[str] = ["reviewer-counting"]
    out += _bool_flag("--dry-run", fields.get("dry_run"))
    out += _opt("--out", fields.get("out"))
    out += _opt("--locked-conf-from", fields.get("locked_conf_from"))
    out += _opt("--gt-test", fields.get("gt_test"))
    out += _opt("--preds-test", fields.get("preds_test"))
    out += _opt("--gt-val", fields.get("gt_val"))
    out += _opt("--preds-val", fields.get("preds_val"))
    out += _opt("--eval-test", fields.get("eval_test"))
    out += _opt("--dual-metric", fields.get("dual_metric"))
    out += _opt("--split-file", fields.get("split_file"))
    out += _opt("--weights", fields.get("weights"))
    return out


def argv_for_repro_steps(
    bundle: dict[str, Any],
    *,
    repo_root: str | Path | None = None,
    skip_gpu_check: bool = False,
    include_test_map: bool = False,
) -> list[tuple[str, list[str]]]:
    """Build (step_id, argv) steps from manuscript_repro_bundle.v1."""
    from harchoc.manuscript_repro import build_manuscript_repro_chain

    return build_manuscript_repro_chain(
        bundle,
        repo_root=repo_root,
        skip_gpu_check=skip_gpu_check,
        include_test_map=include_test_map,
    )


def argv_for_reviewer2_repro(fields: dict[str, Any]) -> list[str]:
    """Argv tail for ``scripts/experiment.py reviewer2-repro``."""
    out: list[str] = ["reviewer2-repro"]
    out += _bool_flag("--dry-run", fields.get("dry_run"))
    out += _opt("--bundle", fields.get("bundle"))
    return out
