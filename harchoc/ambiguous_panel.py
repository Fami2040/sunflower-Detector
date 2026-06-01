"""Ambiguous-detection panel helpers (low-conf band + localization FP crops)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_fp_crop_rows(report_path: str | Path) -> list[dict[str, Any]]:
    """Return all FP crop rows from an error_analysis report (any status)."""
    obj = json.loads(Path(report_path).expanduser().read_text("utf-8"))
    crops = obj.get("fp_crops")
    if not isinstance(crops, dict):
        return []
    results = crops.get("results")
    if not isinstance(results, list):
        return []
    return [r for r in results if isinstance(r, dict)]


def _in_conf_band(score: float | None, band: list[float]) -> bool:
    if score is None or len(band) < 2:
        return False
    lo, hi = float(band[0]), float(band[1])
    return lo <= float(score) <= hi


def select_ambiguous_panel_entries(
    entries: list[dict[str, Any]],
    *,
    conf_band: list[float] | None = None,
    max_panels: int = 12,
) -> list[dict[str, Any]]:
    """
    Pick localization-FP crops and low-confidence-band crops for fig_ambiguous_panel.

    Prefers exported crops (status=ok); falls back to entries without crop_path.
    """
    ok = [e for e in entries if e.get("status") == "ok" and e.get("crop_path")]
    pool = ok if ok else list(entries)
    if not pool:
        return []

    band = list(conf_band or [])
    loc: list[dict[str, Any]] = []
    low_conf: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []

    for ex in pool:
        et = str(ex.get("error_type") or "")
        sc = ex.get("score")
        if et == "localization":
            loc.append(ex)
        elif band and _in_conf_band(float(sc) if sc is not None else None, band):
            low_conf.append(ex)
        else:
            other.append(ex)

    for bucket in (loc, low_conf, other):
        bucket.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)

    picked: list[dict[str, Any]] = []
    half = max(1, max_panels // 2)
    for ex in loc[:half]:
        if len(picked) >= max_panels:
            break
        picked.append({**ex, "panel_tag": "localization_fp"})
    for ex in low_conf:
        if len(picked) >= max_panels:
            break
        if ex not in picked:
            picked.append({**ex, "panel_tag": "low_conf_band"})
    for ex in other:
        if len(picked) >= max_panels:
            break
        if ex not in picked:
            picked.append({**ex, "panel_tag": str(ex.get("error_type") or "fp")})
    return picked[:max_panels]


def plan_ambiguous_panel(
    *,
    report_path: str | Path | None,
    max_panels: int = 12,
) -> dict[str, Any]:
    """Dry-run plan for fig_ambiguous_panel."""
    entries: list[dict[str, Any]] = []
    conf_band: list[float] = []
    crosstab: dict[str, Any] | None = None
    if report_path:
        report = json.loads(Path(report_path).expanduser().read_text("utf-8"))
        amb = report.get("ambiguous_summary")
        if isinstance(amb, dict) and isinstance(amb.get("conf_band"), list):
            conf_band = [float(x) for x in amb["conf_band"]]
        xt = report.get("ambiguous_fp_crosstab")
        if isinstance(xt, dict):
            crosstab = xt
        entries = select_ambiguous_panel_entries(
            load_fp_crop_rows(report_path),
            conf_band=conf_band,
            max_panels=max_panels,
        )
    panels = [
        {
            "panel_tag": ex.get("panel_tag"),
            "error_type": ex.get("error_type"),
            "score": ex.get("score"),
            "crop_path": ex.get("crop_path"),
        }
        for ex in entries
    ]
    return {
        "max_panels": max_panels,
        "n_selected": len(panels),
        "conf_band": conf_band,
        "ambiguous_fp_crosstab": crosstab,
        "panels": panels,
        "report_path": str(Path(report_path).resolve()) if report_path else None,
    }


def render_ambiguous_mosaic(
    *,
    entries: list[dict[str, Any]],
    out_path: str | Path,
    journal_style: bool = True,
) -> dict[str, Any]:
    """Render a labeled mosaic from FP / low-conf crop PNGs."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        from harchoc.figure_style import add_panel_label, panel_label, prepare_matplotlib, savefig_kwargs

        prepare_matplotlib(journal_style=journal_style)
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.gridspec import GridSpec  # type: ignore
    except Exception as ex:
        return {"status": "skipped", "reason": f"matplotlib unavailable: {ex}"}

    if not entries:
        return {"status": "skipped", "reason": "no panel entries"}

    n = len(entries)
    cols = min(4, max(1, n))
    rows = (n + cols - 1) // cols
    cell = 2.2 if journal_style else 3.0
    fig = plt.figure(figsize=(cell * cols, cell * rows))
    gs = GridSpec(rows, cols, figure=fig, wspace=0.08, hspace=0.28)

    rendered = 0
    for i, ex in enumerate(entries):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        crop_path = Path(str(ex.get("crop_path") or ""))
        tag = str(ex.get("panel_tag") or ex.get("error_type") or "fp")
        score = float(ex.get("score") or 0.0)
        if not crop_path.is_file():
            ax.set_title(f"missing {tag}")
            ax.axis("off")
            continue
        try:
            from PIL import Image  # type: ignore

            with Image.open(crop_path) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_title(f"{tag}\ns={score:.2f}")
            rendered += 1
        except Exception:
            ax.set_title(tag)
        ax.axis("off")
        if journal_style:
            add_panel_label(ax, panel_label(i), x=-0.02, y=1.02, fontsize=10)

    fig.suptitle(
        "Ambiguous detections: localization FP + low-conf band",
        fontsize=10,
        y=1.02 if journal_style else 0.98,
    )
    fig.tight_layout()
    fig.savefig(out, **savefig_kwargs(journal_style=journal_style))
    plt.close(fig)
    status = "ok" if rendered else "partial"
    return {
        "status": status,
        "out_path": str(out.resolve()),
        "n_panels": n,
        "n_rendered": rendered,
    }
