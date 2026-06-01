"""Manuscript pipeline overview figure (fig_concept / MS-EXPLAIN)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Fallback copy when HSP JSON artifacts are absent (matches p0_summary.md snapshot).
_DEFAULT_HEADLINES: dict[str, str] = {
    "locked_conf": "0.15",
    "val_mae": "71.0",
    "test_mae": "61.3",
    "match_iou": "0.3",
    "export_conf": "0.001",
    "max_det": "3000",
    "train_n": "875",
    "val_n": "109",
    "test_n": "109",
    "seed_ratio": "~55% dev / ~45% abrt",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def _fmt_conf(conf: float) -> str:
    return f"{conf:.2f}".rstrip("0").rstrip(".")


def _fmt_mae(mae: float) -> str:
    return f"{mae:.1f}"


def _min_count_mae_row(obj: dict[str, Any]) -> dict[str, Any] | None:
    for block in obj.get("selection_comparison") or []:
        if not isinstance(block, dict):
            continue
        if str(block.get("mode") or "") == "min_count_mae":
            sel = block.get("selected")
            if isinstance(sel, dict):
                return sel
    locked = obj.get("locked")
    if isinstance(locked, dict):
        row = locked.get("row")
        if isinstance(row, dict):
            return row
        cm = locked.get("counting_metrics")
        if isinstance(cm, dict) and locked.get("conf_thr") is not None:
            return {
                "conf_thr": locked.get("conf_thr"),
                "count_mae": cm.get("mae"),
            }
    sel = obj.get("selected")
    if isinstance(sel, dict):
        row = sel.get("row")
        if isinstance(row, dict):
            return row
    return None


def load_concept_headlines(*, repo_root: Path | None = None, hsp_dir: Path | None = None) -> dict[str, str]:
    """
    Load headline operating-point strings from HSP JSON when present.

    Prefers ``fp_budget_sweep*.json`` (min_count_mae); falls back to defaults.
    """
    root = repo_root or _repo_root()
    hsp = hsp_dir or (root / "reports" / "hsp")
    out = dict(_DEFAULT_HEADLINES)

    val_budget = _read_json(hsp / "fp_budget_sweep.json")
    test_budget = _read_json(hsp / "fp_budget_sweep_test.json")
    val_row = _min_count_mae_row(val_budget) if val_budget else None
    test_row = _min_count_mae_row(test_budget) if test_budget else None

    if val_row:
        conf = val_row.get("conf_thr")
        if conf is not None:
            out["locked_conf"] = _fmt_conf(float(conf))
        mae = val_row.get("count_mae")
        if mae is None and isinstance(val_row.get("counting_metrics"), dict):
            mae = val_row["counting_metrics"].get("mae")
        if mae is not None:
            out["val_mae"] = _fmt_mae(float(mae))

    if test_row:
        mae = test_row.get("count_mae")
        if mae is None and isinstance(test_row.get("counting_metrics"), dict):
            mae = test_row["counting_metrics"].get("mae")
        if mae is not None:
            out["test_mae"] = _fmt_mae(float(mae))
        conf = test_row.get("conf_thr")
        if conf is not None and "locked_conf" not in out:
            out["locked_conf"] = _fmt_conf(float(conf))

    thr_val = _read_json(hsp / "threshold_val.json")
    if thr_val:
        match = thr_val.get("match")
        if isinstance(match, dict) and match.get("iou") is not None:
            out["match_iou"] = str(match["iou"])

    return out


def emit_concept_diagram(
    *,
    out_path: str | Path,
    journal_style: bool = True,
    include_svg: bool = True,
    repo_root: Path | None = None,
    hsp_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Render HSP pipeline: splits → train → export → detection → val threshold lock → test MAE.

    CPU-only (matplotlib); optional headline numbers from ``reports/hsp/*.json``.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    headlines = load_concept_headlines(repo_root=repo_root, hsp_dir=hsp_dir)

    try:
        from harchoc.figure_style import add_panel_label, panel_label, prepare_matplotlib, savefig_kwargs

        prepare_matplotlib(journal_style=journal_style)
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # type: ignore
    except ImportError as ex:
        return {"status": "skipped", "reason": f"missing_dependency:matplotlib ({ex})"}

    conf = headlines["locked_conf"]
    val_mae = headlines["val_mae"]
    test_mae = headlines["test_mae"]
    match_iou = headlines["match_iou"]
    export_conf = headlines["export_conf"]
    max_det = headlines["max_det"]

    # Double-column width; taller canvas for 10–12 pt body text.
    fig_w, fig_h = (7.0, 5.75) if journal_style else (9.0, 6.5)
    title_fs = 12.0 if journal_style else 13.0
    box_title_fs = 10.5 if journal_style else 11.0
    box_body_fs = 9.5 if journal_style else 10.5
    note_fs = 8.5 if journal_style else 9.5
    band_fs = 9.0 if journal_style else 10.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def _box(
        xy: tuple[float, float],
        w: float,
        h: float,
        *,
        title: str,
        body: str,
        face: str,
        edge: str = "#333333",
        linestyle: str = "solid",
        linewidth: float = 1.0,
        title_fs_override: float | None = None,
        body_fs_override: float | None = None,
    ) -> FancyBboxPatch:
        patch = FancyBboxPatch(
            xy,
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=linewidth,
            edgecolor=edge,
            facecolor=face,
            linestyle=linestyle,
        )
        ax.add_patch(patch)
        cx = xy[0] + w / 2
        cy = xy[1] + h / 2
        tfs = title_fs_override or box_title_fs
        bfs = body_fs_override or box_body_fs
        n_lines = body.count("\n") + 1
        title_dy = 0.12 + 0.06 * max(0, n_lines - 3)
        ax.text(cx, cy + title_dy, title, ha="center", va="center", fontsize=tfs, fontweight="bold")
        ax.text(cx, cy - 0.18, body, ha="center", va="center", fontsize=bfs, linespacing=1.2)
        return patch

    def _arrow(
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        *,
        color: str = "#444444",
        linestyle: str = "solid",
    ) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.2,
                color=color,
                linestyle=linestyle,
                shrinkA=3,
                shrinkB=3,
            )
        )

    def _band_label(x: float, y: float, text: str, color: str = "#555555") -> None:
        ax.text(x, y, text, ha="left", va="center", fontsize=band_fs, fontweight="bold", color=color)

    ax.text(
        7.0,
        9.55,
        "HSP two-stage counting: val threshold lock → test count MAE",
        ha="center",
        va="center",
        fontsize=title_fs,
        fontweight="bold",
    )

    # --- Row 1: dataset ---
    _band_label(0.25, 8.85, "Data")
    _box(
        (0.35, 7.95),
        4.0,
        0.75,
        title="Frozen YOLO splits",
        body=(
            f"train {headlines['train_n']} · val {headlines['val_n']} · test {headlines['test_n']} images\n"
            f"classes: developed (0) · aborted (1)\n"
            f"asymmetric seeds: {headlines['seed_ratio']} (biological mix)"
        ),
        face="#f5f5f5",
    )
    _box(
        (4.65, 7.95),
        4.2,
        0.75,
        title="Eval policy",
        body=(
            "val: tune & lock global conf (no test peeking)\n"
            "test: headline count MAE only @ locked conf"
        ),
        face="#fafafa",
        edge="#666666",
    )

    _arrow(7.0, 7.92, 7.0, 7.55)

    # --- Row 2: train ---
    _band_label(0.25, 7.35, "Train")
    _box(
        (1.2, 6.35),
        5.2,
        0.85,
        title="Detector training",
        body="YOLO / RT-DETR · imgsz 1280\nval split early-stop → models/best2.pt",
        face="#e8f0fe",
        edge="#2a5ea8",
    )
    _arrow(7.0, 7.9, 3.8, 7.2)
    _arrow(3.8, 6.35, 3.8, 5.95)

    # --- Row 3: export ---
    _band_label(0.25, 5.75, "Export")
    _box(
        (0.9, 4.85),
        5.8,
        0.8,
        title="Low-confidence prediction export",
        body=(
            f"conf {export_conf} · NMS IoU 0.3 · max_det {max_det}\n"
            "full-frame @ 1280 (manuscript metrics path)"
        ),
        face="#ede7f6",
        edge="#5e35b1",
    )
    _arrow(3.8, 6.35, 3.8, 5.65)

    # --- Row 4: two stages (visual bands) ---
    _band_label(0.25, 4.35, "Inference & counting")
    stage_h = 1.35
    stage_y = 2.75
    _box(
        (0.35, stage_y),
        6.3,
        stage_h,
        title="Stage 1 · Detection",
        body=(
            "dense seed boxes per tray image\n"
            "two-class heads (developed / aborted)\n"
            "not SAHI for locked test MAE (full-frame)"
        ),
        face="#e3f2fd",
        edge="#1565c0",
        linewidth=1.3,
    )
    _box(
        (7.05, stage_y),
        6.5,
        stage_h,
        title="Stage 2 · Counting",
        body=(
            f"greedy match IoU {match_iou} · category-aware\n"
            "count-first: min val count MAE over conf grid\n"
            "(not F1-max alone — high FP budget on dense trays)"
        ),
        face="#e8f5e9",
        edge="#2e7d32",
        linewidth=1.3,
    )
    _arrow(6.68, stage_y + stage_h / 2, 7.03, stage_y + stage_h / 2)
    _arrow(3.8, 4.85, 3.5, 4.1)

    # --- Row 5: threshold lock flow ---
    _band_label(0.25, 2.35, "Operating point")
    th_h = 1.05
    th_y = 1.05
    _box(
        (0.35, th_y),
        4.0,
        th_h,
        title="threshold_sweep (val)",
        body=(
            f"select min_count_mae → conf {conf}\n"
            f"val count MAE {val_mae}"
        ),
        face="#fff3e0",
        edge="#ef6c00",
    )
    _box(
        (4.55, th_y),
        1.35,
        th_h,
        title="lock",
        body=f"conf\n{conf}",
        face="#ffffff",
        edge="#ef6c00",
        title_fs_override=10.0,
        body_fs_override=10.0,
    )
    _box(
        (6.15, th_y),
        3.5,
        th_h,
        title="test (locked)",
        body=(
            f"same conf {conf} · no re-tune\n"
            f"headline: count MAE {test_mae}"
        ),
        face="#c8e6c9",
        edge="#1b5e20",
        linewidth=1.4,
    )
    _arrow(4.38, th_y + th_h / 2, 4.52, th_y + th_h / 2, color="#ef6c00")
    _arrow(5.93, th_y + th_h / 2, 6.12, th_y + th_h / 2, color="#1b5e20")
    _arrow(3.5, 2.75, 2.35, 2.12)

    ax.text(
        10.2,
        th_y + th_h / 2,
        "← val only",
        ha="center",
        va="center",
        fontsize=note_fs,
        color="#ef6c00",
        style="italic",
    )
    ax.text(
        10.2,
        th_y + th_h / 2 - 0.35,
        "test frozen →",
        ha="center",
        va="center",
        fontsize=note_fs,
        color="#1b5e20",
        style="italic",
    )

    # --- Optional deploy (dashed branch) ---
    _box(
        (9.75, 4.55),
        3.9,
        1.55,
        title="Optional deploy gate",
        body=(
            "classifier.pt: sunflower vs other\n"
            "SAHI tiling + best2.pt (field path)\n"
            "dashed = production, not headline metric"
        ),
        face="#fff8e1",
        edge="#e65100",
        linestyle="dashed",
        linewidth=1.3,
    )
    _arrow(
        6.65,
        stage_y + stage_h,
        10.2,
        6.12,
        color="#e65100",
        linestyle="dashed",
    )
    ax.text(
        11.7,
        6.35,
        "optional",
        ha="center",
        va="center",
        fontsize=note_fs,
        color="#e65100",
        style="italic",
    )

    if journal_style:
        add_panel_label(ax, panel_label(0), x=-0.02, y=1.02, fontsize=11)

    fig.tight_layout(pad=0.4)
    kw = savefig_kwargs(journal_style=journal_style)
    fig.savefig(out, **kw)
    written = [str(out.resolve())]
    if include_svg:
        svg_path = out.with_suffix(".svg")
        fig.savefig(svg_path, format="svg", bbox_inches=kw.get("bbox_inches", "tight"), facecolor="white")
        written.append(str(svg_path.resolve()))
    plt.close(fig)

    try:
        from PIL import Image  # type: ignore

        with Image.open(out) as im:
            px_w, px_h = im.size
    except Exception:
        px_w, px_h = None, None

    return {
        "status": "ok",
        "out_path": str(out),
        "files": written,
        "device": "cpu",
        "figsize_inches": [fig_w, fig_h],
        "pixel_size": [px_w, px_h],
        "headlines": headlines,
    }
