from __future__ import annotations

from pathlib import Path
from typing import Any


def emit_split_drift_plots(
    report: dict[str, Any],
    *,
    out_dir: Path,
    journal_style: bool = True,
) -> dict[str, Any]:
    """
    Write simple drift summary PNGs from split_drift report JSON.
    Matplotlib is imported lazily; returns manifest of written paths.
    """
    try:
        from harchoc.figure_style import add_panel_label, panel_label, prepare_matplotlib, savefig_kwargs

        prepare_matplotlib(journal_style=journal_style)
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as ex:
        return {"status": "skipped", "reason": f"missing_dependency:matplotlib ({ex})"}

    out_dir.mkdir(parents=True, exist_ok=True)
    comps = report.get("comparisons") if isinstance(report.get("comparisons"), dict) else {}
    written: list[str] = []
    panel_idx = 0

    metrics = [
        ("labels", "class_dist_l1", "Class L1 distance"),
        ("labels", "class_jsd_nats", "Class JSD (nats)"),
    ]
    pairs = [k for k in ("train_vs_val", "val_vs_test", "train_vs_test") if k in comps]
    if not pairs:
        return {"status": "skipped", "reason": "no comparisons in report"}

    figsize = (3.4, 2.2) if journal_style else (6, 3)
    for section, key, title in metrics:
        vals = []
        labels = []
        for p in pairs:
            comp = comps.get(p)
            if not isinstance(comp, dict):
                continue
            sec = comp.get(section) if isinstance(comp.get(section), dict) else {}
            v = sec.get(key)
            if v is None:
                continue
            vals.append(float(v))
            labels.append(p)
        if not vals:
            continue
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(labels, vals)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        if journal_style:
            add_panel_label(ax, panel_label(panel_idx))
            panel_idx += 1
        fig.tight_layout()
        path = out_dir / f"{section}_{key}.png"
        fig.savefig(path, **savefig_kwargs(journal_style=journal_style))
        plt.close(fig)
        written.append(str(path))

    ks_pairs: list[tuple[str, float | None]] = []
    for p in pairs:
        comp = comps[p]
        img = comp.get("images") if isinstance(comp.get("images"), dict) else {}
        wks = img.get("width_ks") if isinstance(img.get("width_ks"), dict) else {}
        if wks.get("available") and wks.get("pvalue") is not None:
            ks_pairs.append((p, float(wks["pvalue"])))
    if ks_pairs:
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar([x[0] for x in ks_pairs], [x[1] for x in ks_pairs])
        ax.axhline(0.05, color="orange", linestyle="--", label="p=0.05")
        ax.set_title("Width KS p-value")
        ax.legend()
        if journal_style:
            add_panel_label(ax, panel_label(panel_idx))
        fig.tight_layout()
        path = out_dir / "width_ks_pvalues.png"
        fig.savefig(path, **savefig_kwargs(journal_style=journal_style))
        plt.close(fig)
        written.append(str(path))

    return {"status": "ok", "out_dir": str(out_dir), "files": written}
