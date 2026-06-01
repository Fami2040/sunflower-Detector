"""Reproduce plants-4336582.docx figures & tables from HSP artifacts (one journal style)."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harchoc.detection_match import per_image_detection_counts
from harchoc.figure_style import (
    FIGURE_DPI,
    add_panel_label,
    panel_label,
    prepare_matplotlib,
    savefig_kwargs,
)
from harchoc.json_io import load_json_dict, write_json
from harchoc.manuscript_tables import (
    FOOTNOTE_LOCKED_CONF,
    FOOTNOTE_TEST_SPLIT,
    build_headline_rows,
    fmt_conf,
    fmt_mae,
    fmt_map,
    render_headline_md,
)
from harchoc.schemas import with_schema_version

DOCX_REPRO_SCHEMA = "manuscript_docx_repro.v1"
DEFAULT_OUT_DIR = "reports/manuscript/docx"
CLASS_COLORS = {0: "#2ca02c", 1: "#d62728"}  # developed green, aborted red
CLASS_NAMES = {0: "developed", 1: "aborted"}


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mpl():
    prepare_matplotlib(journal_style=True)
    import matplotlib.pyplot as plt  # type: ignore

    return plt


def plot_confusion_matrix(
    matrix: list[list[int]],
    labels: list[str],
    *,
    out_path: Path,
    normalized: bool = False,
    title: str,
) -> dict[str, Any]:
    plt = _mpl()
    import numpy as np  # type: ignore

    data = np.array(matrix, dtype=float)
    if normalized:
        row_sums = data.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        data = data / row_sums * 100.0
        fmt = ".1f"
        cbar_label = "% of GT row"
    else:
        fmt = "d"
        cbar_label = "Count"

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(data, cmap="Blues", aspect="auto")
    pred_labels = ["pred 0", "pred 1", "pred bg"]
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(pred_labels, rotation=25, ha="right")
    ax.set_yticklabels(labels)
    for i in range(3):
        for j in range(3):
            val = data[i, j]
            text = f"{int(val)}" if not normalized else f"{val:.1f}"
            color = "white" if val > (data.max() * 0.55) else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    add_panel_label(ax, panel_label(0))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, **savefig_kwargs(journal_style=True))
    plt.close(fig)
    return {"status": "ok", "path": str(out_path), "normalized": normalized}


def plot_figure_06_panels(csv_path: Path, *, out_path: Path, locked_conf: float) -> dict[str, Any]:
    plt = _mpl()
    rows: list[dict[str, float]] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                {
                    "conf_thr": float(row["conf_thr"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1": float(row["f1"]),
                }
            )
    if not rows:
        return {"status": "skipped", "reason": "empty threshold csv"}

    conf = [r["conf_thr"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))
    panels = [
        (axes[0, 0], "precision", "recall", "Precision–recall (val sweep)"),
        (axes[0, 1], "conf_thr", "f1", "F1 vs confidence"),
        (axes[1, 0], "conf_thr", "precision", "Precision vs confidence"),
        (axes[1, 1], "conf_thr", "recall", "Recall vs confidence"),
    ]
    for ax, xk, yk, title in panels:
        xs = [r[xk] for r in rows]
        ys = [r[yk] for r in rows]
        ax.plot(xs, ys, color="#1f77b4", linewidth=1.25)
        if xk == "conf_thr":
            ax.axvline(locked_conf, color="#888", linestyle="--", linewidth=0.9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=8)
        ax.grid(True, alpha=0.25)
    for i, ax in enumerate(axes.flat):
        add_panel_label(ax, panel_label(i))
    fig.suptitle(
        f"HSP val threshold sweep (locked test conf {locked_conf:.2f})",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, **savefig_kwargs(journal_style=True))
    plt.close(fig)
    return {"status": "ok", "path": str(out_path)}


def plot_detection_example(
    *,
    dataset_root: Path,
    split_file: Path,
    preds_path: Path,
    out_path: Path,
    conf_thr: float,
    max_boxes: int = 400,
) -> dict[str, Any]:
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except ImportError as ex:
        return {"status": "skipped", "reason": f"PIL missing: {ex}"}

    from harchoc.datasets import resolve_dataset
    from harchoc.eval_export import iter_split_image_paths

    root = resolve_dataset(dataset_root=str(dataset_root)).root
    entries = iter_split_image_paths(split_file, dataset_root=root)
    if not entries:
        return {"status": "skipped", "reason": "empty split"}
    img_id, img_path, _ = entries[0]
    preds = load_json_dict(preds_path)
    pred_recs = {str(r["image_id"]): r for r in preds.get("images") or [] if isinstance(r, dict)}
    rec = pred_recs.get(str(img_id)) or pred_recs.get(img_id)
    if not rec:
        return {"status": "skipped", "reason": f"no preds for {img_id}"}

    im = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    n = 0
    for det in rec.get("detections") or []:
        score = det.get("score")
        if score is not None and float(score) < conf_thr:
            continue
        bbox = det.get("bbox") or []
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        cat = int(det.get("category_id", 0))
        color = CLASS_COLORS.get(cat, "#ffffff")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        n += 1
        if n >= max_boxes:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, dpi=(FIGURE_DPI, FIGURE_DPI))
    return {"status": "ok", "path": str(out_path), "image_id": img_id, "boxes_drawn": n}


def plot_training_curves(results_csv: Path, *, out_path: Path) -> dict[str, Any]:
    if not results_csv.is_file():
        return {"status": "skipped", "reason": f"missing {results_csv}"}
    plt = _mpl()
    epochs: list[int] = []
    series: dict[str, list[float]] = {}
    with results_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            epochs.append(int(float(row.get("epoch", len(epochs)))))
            for key in row:
                if key == "epoch":
                    continue
                try:
                    series.setdefault(key, []).append(float(row[key]))
                except (TypeError, ValueError):
                    pass
    if not epochs:
        return {"status": "skipped", "reason": "empty results.csv"}

    pick = [k for k in series if "loss" in k.lower() or "map" in k.lower()][:6]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for key in pick:
        ax.plot(epochs, series[key], label=key, linewidth=1.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    ax.set_title("Training metrics (Ultralytics results.csv)")
    ax.legend(fontsize=6, ncol=2)
    add_panel_label(ax, panel_label(0))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, **savefig_kwargs(journal_style=True))
    plt.close(fig)
    return {"status": "ok", "path": str(out_path), "columns": pick}


def _error_bins(per_image: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    bins = [
        ("<2%", 0.0, 2.0),
        ("2–5%", 2.0, 5.0),
        ("5–10%", 5.0, 10.0),
        ("10–20%", 10.0, 20.0),
        (">20%", 20.0, 1e9),
    ]
    counts = [0] * len(bins)
    n = 0
    for rec in per_image.values():
        gt = int(rec.get("n_gt") or 0)
        if gt <= 0:
            continue
        err_pct = abs(int(rec.get("n_pred") or 0) - gt) / gt * 100.0
        n += 1
        for i, (_, lo, hi) in enumerate(bins):
            if lo <= err_pct < hi:
                counts[i] += 1
                break
    rows = []
    for (label, _, _), c in zip(bins, counts):
        pct = (c / n * 100.0) if n else 0.0
        rows.append({"bin": label, "n_images": c, "pct": pct})
    return rows


def build_table_01_markdown(dual_metric: dict[str, Any], *, dual_metric_path: str) -> str:
    rows = build_headline_rows(dual_metric=dual_metric)
    md = render_headline_md(
        rows,
        dual_metric_path=dual_metric_path,
        title="Table 1. Object detection and counting (HSP reproduction)",
    )
    return md + "\n> Submitted docx cites mAP@0.5 ≈0.793; HSP canonical test mAP50 is ~0.18 — see `reviewer2_map50_computed.json`.\n"


def build_table_02_markdown(counting: dict[str, Any], *, locked_conf: float) -> str:
    rel = counting.get("per_image_relative_error_pct") or counting.get("test", {}).get("per_image_relative_error_pct") or {}
    pooled = counting.get("pooled") or counting.get("test", {}).get("pooled") or {}
    lines = [
        "# Table 2. Counting summary (HSP test reproduction)",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Locked confidence | {fmt_conf(locked_conf)} |",
        f"| Test images (n) | {counting.get('n_images') or pooled.get('n_images') or 109} |",
        f"| Count MAE | {fmt_mae(pooled.get('mae'))} |",
        f"| Mean relative error (%) | {rel.get('mean', '—')} |",
        f"| Median relative error (%) | {rel.get('median', '—')} |",
        f"| % images rel. error <2% | {rel.get('pct_below_2', '—')} |",
        "",
        "## Footnotes",
        "",
        f"1. {FOOTNOTE_LOCKED_CONF}",
        "2. Full test n=109; docx Table 2 cites n=50 blinded audit (not in repo).",
        "",
    ]
    return "\n".join(lines)


def build_table_03_markdown(bin_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Table 3. Distribution of relative counting error",
        "",
        "| Relative error bin | n images | % of test |",
        "|--------------------|----------:|----------:|",
    ]
    for r in bin_rows:
        lines.append(f"| {r['bin']} | {r['n_images']} | {r['pct']:.1f} |")
    lines.extend(["", "## Footnotes", "", f"1. {FOOTNOTE_LOCKED_CONF}", "2. Bins from per-image |pred−gt|/gt at locked conf."])
    return "\n".join(lines) + "\n"


def run_manuscript_docx_repro(
    repo_root: str | Path,
    *,
    out_dir: str = DEFAULT_OUT_DIR,
    confusion_path: str = "reports/hsp/best2_test_confusion.json",
    dual_metric_path: str = "reports/hsp/dual_metric.json",
    counting_path: str = "reports/reviewer2_counting_metrics_computed.json",
    threshold_csv: str = "reports/hsp/threshold_val.csv",
    preds_test: str = "reports/hsp/preds_test.json",
    gt_test: str = "reports/hsp/gt_test.json",
    split_file: str = "data/splits/test.txt",
    dataset_root: str | None = None,
    training_csv: str = "runs/detect/runs/hsp_zoo/yolov8m_e100_s0/results.csv",
    dry_run: bool = False,
) -> dict[str, Any]:
    rr = Path(repo_root).resolve()
    out = (rr / out_dir).resolve()
    fig_dir = out / "figures"
    tab_dir = out / "tables"

    catalog: dict[str, Any] = {
        "schema_version": DOCX_REPRO_SCHEMA,
        "generated_at": _utc(),
        "style": {"dpi": FIGURE_DPI, "module": "harchoc.figure_style"},
        "docx": "reports/plants-4336582.docx",
        "figures": {},
        "tables": {},
    }

    if dry_run:
        catalog["dry_run"] = True
        catalog["would_write"] = {
            "figures": [f"figure_{i:02d}_*.png" for i in range(1, 12)],
            "tables": ["table_01.md", "table_02.md", "table_03.md"],
            "readme": str(out / "README.md"),
        }
        write_json(out / "catalog.json", catalog)
        return catalog

    locked_conf = 0.15
    if (rr / dual_metric_path).is_file():
        dm = load_json_dict(rr / dual_metric_path)
        locked_conf = float((dm.get("operating_point") or {}).get("locked_conf") or locked_conf)

    # Figures 4–5 confusion
    if (rr / confusion_path).is_file():
        cm = load_json_dict(rr / confusion_path)
        mat = cm.get("matrix") or []
        labels = cm.get("labels") or ["developed (0)", "aborted (1)", "background"]
        r4 = plot_confusion_matrix(
            mat,
            labels,
            out_path=fig_dir / "figure_04_confusion_absolute.png",
            normalized=False,
            title="Confusion matrix (absolute counts)",
        )
        r5 = plot_confusion_matrix(
            mat,
            labels,
            out_path=fig_dir / "figure_05_confusion_normalized.png",
            normalized=True,
            title="Normalized confusion matrix (% of GT row)",
        )
        catalog["figures"]["Figure 4"] = {**r4, "docx_caption": "Confusion matrix (absolute counts)"}
        catalog["figures"]["Figure 5"] = {**r5, "docx_caption": "Normalized confusion matrix"}

    # Figure 6
    r6 = plot_figure_06_panels(
        rr / threshold_csv,
        out_path=fig_dir / "figure_06_metrics_panels.png",
        locked_conf=locked_conf,
    )
    catalog["figures"]["Figure 6"] = {**r6, "docx_caption": "Performance vs confidence (HSP val sweep)"}

    # Figure 1 detection example
    ds_root = dataset_root or "data/raw/extracted/dataset"
    r1 = plot_detection_example(
        dataset_root=rr / ds_root,
        split_file=rr / split_file,
        preds_path=rr / preds_test,
        out_path=fig_dir / "figure_01_detection_example.png",
        conf_thr=locked_conf,
    )
    catalog["figures"]["Figure 1"] = {**r1, "docx_caption": "Example detections (developed=green, aborted=red)"}

    # Figure 2 training curves (proxy run)
    r2 = plot_training_curves(rr / training_csv, out_path=fig_dir / "figure_02_training_curves.png")
    catalog["figures"]["Figure 2"] = {
        **r2,
        "docx_caption": "Training curves (yolov8m_e100_s0 proxy; replace with best2 run if available)",
    }

    # Tables
    tab_dir.mkdir(parents=True, exist_ok=True)
    dm = load_json_dict(rr / dual_metric_path) if (rr / dual_metric_path).is_file() else {}
    counting = load_json_dict(rr / counting_path) if (rr / counting_path).is_file() else {}
    gt = load_json_dict(rr / gt_test) if (rr / gt_test).is_file() else {}
    preds = load_json_dict(rr / preds_test) if (rr / preds_test).is_file() else {}
    per_img = per_image_detection_counts(gt=gt, preds=preds, conf_thr=locked_conf, category_aware=True)
    bin_rows = _error_bins(per_img)

    t1 = build_table_01_markdown(dm, dual_metric_path=dual_metric_path)
    t2 = build_table_02_markdown(counting, locked_conf=locked_conf)
    t3 = build_table_03_markdown(bin_rows)
    (tab_dir / "table_01_detection_metrics.md").write_text(t1, encoding="utf-8")
    (tab_dir / "table_02_counting_summary.md").write_text(t2, encoding="utf-8")
    (tab_dir / "table_03_error_bins.md").write_text(t3, encoding="utf-8")
    catalog["tables"]["Table 1"] = {"path": str(tab_dir / "table_01_detection_metrics.md")}
    catalog["tables"]["Table 2"] = {"path": str(tab_dir / "table_02_counting_summary.md")}
    catalog["tables"]["Table 3"] = {"path": str(tab_dir / "table_03_error_bins.md")}

    # README mapping
    readme = _build_readme(catalog)
    (out / "README.md").write_text(readme, encoding="utf-8")
    write_json(out / "catalog.json", catalog)
    return catalog


def _build_readme(catalog: dict[str, Any]) -> str:
    lines = [
        "# Manuscript docx reproduction (HSP data, journal style)",
        "",
        f"Generated: {catalog.get('generated_at')}",
        "",
        "Reproduces **quantitative** figures/tables aligned with `reports/plants-4336582.docx` "
        "from frozen HSP exports. Photos/setup figures (7–11) remain manual.",
        "",
        "## Figures",
        "",
        "| Docx | File | Status |",
        "|------|------|--------|",
    ]
    mapping = {
        "Figure 1": "figures/figure_01_detection_example.png",
        "Figure 2": "figures/figure_02_training_curves.png",
        "Figure 3": "— (dataset spatial panels: not automated)",
        "Figure 4": "figures/figure_04_confusion_absolute.png",
        "Figure 5": "figures/figure_05_confusion_normalized.png",
        "Figure 6": "figures/figure_06_metrics_panels.png",
        "Figure 7–11": "manual photos / CVAT / architecture",
    }
    for fig, path in mapping.items():
        ent = (catalog.get("figures") or {}).get(fig, {})
        st = ent.get("status", "manual")
        lines.append(f"| {fig} | `{path}` | {st} |")
    lines.extend(
        [
            "",
            "## Tables",
            "",
            "| Docx | File |",
            "|------|------|",
            "| Table 1 | `tables/table_01_detection_metrics.md` |",
            "| Table 2 | `tables/table_02_counting_summary.md` |",
            "| Table 3 | `tables/table_03_error_bins.md` |",
            "",
            "## Command",
            "",
            "```bash",
            "PYTHONPATH=. python scripts/experiment.py manuscript-docx-repro",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
