from __future__ import annotations

import argparse
from pathlib import Path

import sys; from pathlib import Path; _r = Path(__file__).resolve().parent.parent; (str(_r) not in sys.path) and sys.path.insert(0, str(_r)); from harchoc.script_entry import bootstrap_repo_imports; bootstrap_repo_imports()
import json

from harchoc.ambiguous_panel import plan_ambiguous_panel, render_ambiguous_mosaic, select_ambiguous_panel_entries, load_fp_crop_rows
from harchoc.concept_diagram import emit_concept_diagram
from harchoc.gradcam_panel import plan_gradcam_panel, render_gradcam_mosaic, select_panel_entries, load_fp_crop_entries
from harchoc.hsp_weights import HSP_DETECTION_WEIGHTS
from harchoc.script_scaffold import build_plan_dry_run_payload, build_plan_scaffold_payload
from harchoc.split_drift_plots import emit_split_drift_plots
from harchoc.threshold_plots import emit_pr_f1_vs_conf_plot
from scripts._common_cli import add_dry_run_arg, cli_print, write_json



def _load_prior_rendered(meta_out: str) -> dict[str, object]:
    path = Path(meta_out)
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    prior = obj.get("rendered")
    return dict(prior) if isinstance(prior, dict) else {}


_RENDERED_STATUSES = frozenset({"ok", "partial", "skipped", "failed"})


def _merge_figure_statuses(
    figures: list[dict[str, object]], rendered: dict[str, object]
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for fig in figures:
        fid = str(fig.get("id") or "")
        entry = dict(fig)
        r = rendered.get(fid)
        if isinstance(r, dict):
            status = r.get("status")
            if isinstance(status, str) and status in _RENDERED_STATUSES:
                entry["status"] = status
            elif r.get("files") or r.get("out_path"):
                entry["status"] = "partial" if r.get("reason") else "ok"
            else:
                entry["status"] = "partial"
            if r.get("reason"):
                entry["reason"] = r["reason"]
            files = r.get("files")
            if isinstance(files, list) and files:
                entry["paths"] = files
            elif r.get("out_path"):
                entry["paths"] = [str(r["out_path"])]
        out.append(entry)
    return out


def _plan_figures() -> list[dict[str, object]]:
    return [
        {"id": "fig_concept", "status": "ok", "notes": "Pipeline concept diagram / overview (CPU)."},
        {"id": "fig_pr_curve", "status": "todo", "notes": "PR/ROC curves and threshold selection."},
        {"id": "fig_error_taxonomy", "status": "todo", "notes": "FP taxonomy examples / qualitative panel."},
        {"id": "fig_gradcam_panel", "status": "todo", "notes": "Grad-CAM overlays on FP crops from error_analysis."},
        {"id": "fig_ambiguous_panel", "status": "todo", "notes": "Ambiguous detections (low conf / high pred IoU)."},
        {"id": "fig_split_drift", "status": "todo", "notes": "Train/val/test drift summary plots."},
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Regenerate manuscript figures (scaffold).")
    add_dry_run_arg(p)
    p.add_argument("--out-dir", default="figures", help="Where to write generated figures.")
    p.add_argument("--meta-out", default="reports/figures/run.json", help="Where to write a small run manifest.")
    p.add_argument(
        "--error-report",
        default="",
        help="error_analysis report.json with fp_crops (for fig_gradcam_panel real mode).",
    )
    p.add_argument("--weights", default=HSP_DETECTION_WEIGHTS, help="HSP detection weights for Grad-CAM overlay on FP crops.")
    p.add_argument("--panel-size", type=int, default=12, help="Max panels in fig_gradcam_panel.")
    p.add_argument(
        "--figure",
        choices=[
            "all",
            "fig_concept",
            "fig_gradcam_panel",
            "fig_ambiguous_panel",
            "fig_error_taxonomy",
            "fig_split_drift",
            "fig_pr_curve",
        ],
        default="all",
        help="Which figure(s) to render in non-dry-run mode.",
    )
    p.add_argument(
        "--split-drift-report",
        default="reports/split_drift/report.json",
        help="split_drift JSON for fig_split_drift.",
    )
    p.add_argument(
        "--threshold-csv",
        default="reports/hsp/threshold_val.csv",
        help="threshold_sweep CSV for fig_pr_curve.",
    )
    p.add_argument(
        "--threshold-json",
        default="",
        help="Optional threshold_sweep JSON (best_f1.conf_thr marker on fig_pr_curve).",
    )
    p.add_argument(
        "--journal-style",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Journal fonts, 300 DPI, and A/B/C panel labels (default: on).",
    )
    args = p.parse_args(argv)

    plan = {"schema_version": "figures_run.v1", "script": "make_figures", "figures": _plan_figures()}
    report_path = (args.error_report or "").strip() or None
    gradcam_plan = plan_gradcam_panel(report_path=report_path, max_panels=int(args.panel_size))
    plan["fig_gradcam_panel"] = gradcam_plan
    ambiguous_plan = plan_ambiguous_panel(report_path=report_path, max_panels=int(args.panel_size))
    plan["fig_ambiguous_panel"] = ambiguous_plan

    if args.dry_run:
        meta_out = write_json(
            args.meta_out,
            build_plan_dry_run_payload(plan, out_dir=args.out_dir, meta_out=args.meta_out),
        )
        cli_print(f"Wrote {meta_out}")
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered: dict[str, object] = _load_prior_rendered(args.meta_out)
    if args.figure in ("all", "fig_concept"):
        rendered["fig_concept"] = emit_concept_diagram(
            out_path=out_dir / "fig_concept.png",
            journal_style=bool(args.journal_style),
        )

    if args.figure in ("all", "fig_pr_curve"):
        best_conf: float | None = None
        thr_json = (args.threshold_json or "").strip()
        if thr_json:
            thr_path = Path(thr_json)
            if thr_path.is_file():
                thr_obj = json.loads(thr_path.read_text(encoding="utf-8"))
                best = thr_obj.get("best_f1") if isinstance(thr_obj.get("best_f1"), dict) else {}
                if best.get("conf_thr") is not None:
                    best_conf = float(best["conf_thr"])
        rendered["fig_pr_curve"] = emit_pr_f1_vs_conf_plot(
            args.threshold_csv,
            out_path=out_dir / "threshold" / "pr_f1_vs_conf.png",
            best_conf=best_conf,
            journal_style=bool(args.journal_style),
        )

    if args.figure in ("all", "fig_split_drift"):
        drift_path = Path(args.split_drift_report)
        if drift_path.is_file():
            drift_report = json.loads(drift_path.read_text(encoding="utf-8"))
            drift_out = out_dir / "split_drift"
            rendered["fig_split_drift"] = emit_split_drift_plots(
                drift_report,
                out_dir=drift_out,
                journal_style=bool(args.journal_style),
            )
        else:
            rendered["fig_split_drift"] = {
                "status": "skipped",
                "reason": f"report not found: {drift_path}",
            }

    if args.figure in ("all", "fig_gradcam_panel") and report_path:
        entries = select_panel_entries(load_fp_crop_entries(report_path), max_panels=int(args.panel_size))
        weights = (args.weights or "").strip() or HSP_DETECTION_WEIGHTS
        gc_result = render_gradcam_mosaic(
            entries=entries,
            out_path=out_dir / "fig_gradcam_panel.png",
            weights=weights,
            journal_style=bool(args.journal_style),
        )
        rendered["fig_gradcam_panel"] = {
            **gc_result,
            "device": "gpu" if weights else "cpu",
        }

    if args.figure in ("all", "fig_error_taxonomy") and report_path:
        tax_entries = select_panel_entries(
            load_fp_crop_entries(report_path), max_panels=int(args.panel_size)
        )
        tax_result = render_gradcam_mosaic(
            entries=tax_entries,
            out_path=out_dir / "fig_error_taxonomy.png",
            weights=None,
            suptitle="False-positive taxonomy (TIDE-aligned crop examples)",
            journal_style=bool(args.journal_style),
        )
        rendered["fig_error_taxonomy"] = {**tax_result, "device": "cpu"}

    if args.figure in ("all", "fig_ambiguous_panel") and report_path:
        amb_plan = plan["fig_ambiguous_panel"]
        conf_band = amb_plan.get("conf_band") if isinstance(amb_plan, dict) else []
        amb_entries = select_ambiguous_panel_entries(
            load_fp_crop_rows(report_path),
            conf_band=list(conf_band) if isinstance(conf_band, list) else None,
            max_panels=int(args.panel_size),
        )
        amb_result = render_ambiguous_mosaic(
            entries=amb_entries,
            out_path=out_dir / "fig_ambiguous_panel.png",
            journal_style=bool(args.journal_style),
        )
        rendered["fig_ambiguous_panel"] = {**amb_result, "device": "cpu"}

    plan["figures"] = _merge_figure_statuses(plan["figures"], rendered)
    payload = build_plan_scaffold_payload(
        plan,
        out_dir=out_dir,
        rendered=rendered,
        notes=(
            "Real mode: fig_concept is CPU-only (no inputs). "
            "Pass --error-report from error_analysis --export-fp-crops; "
            "--threshold-csv for fig_pr_curve. Journal style (300 DPI, panel labels) is on by default; "
            "use --no-journal-style to disable."
        ),
    )
    if args.journal_style:
        payload["journal_style"] = True
    meta_out = write_json(args.meta_out, payload)
    cli_print(f"Wrote {meta_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
