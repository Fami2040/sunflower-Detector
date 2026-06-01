from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _read_threshold_rows(csv_path: Path) -> list[dict[str, float]]:
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
    return rows


def emit_pr_f1_vs_conf_plot(
    csv_path: str | Path,
    *,
    out_path: Path,
    best_conf: float | None = None,
    journal_style: bool = True,
) -> dict[str, Any]:
    """Plot precision, recall, and F1 vs confidence threshold from threshold_sweep CSV."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        return {"status": "skipped", "reason": f"csv not found: {csv_path}"}

    try:
        from harchoc.figure_style import add_panel_label, panel_label, prepare_matplotlib, savefig_kwargs

        prepare_matplotlib(journal_style=journal_style)
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as ex:
        return {"status": "skipped", "reason": f"missing_dependency:matplotlib ({ex})"}

    rows = _read_threshold_rows(csv_path)
    if not rows:
        return {"status": "skipped", "reason": "empty csv"}

    conf = [r["conf_thr"] for r in rows]
    if best_conf is None:
        best = max(rows, key=lambda r: r["f1"])
        best_conf = best["conf_thr"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.5, 2.5) if journal_style else (7, 4))
    ax.plot(conf, [r["precision"] for r in rows], label="precision")
    ax.plot(conf, [r["recall"] for r in rows], label="recall")
    ax.plot(conf, [r["f1"] for r in rows], label="f1")
    ax.axvline(best_conf, color="gray", linestyle="--", linewidth=1, label=f"best F1 @ {best_conf:.2f}")
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("metric")
    ax.set_title("Precision / recall / F1 vs confidence (val)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    if journal_style:
        add_panel_label(ax, panel_label(0))
    fig.tight_layout()
    fig.savefig(out_path, **savefig_kwargs(journal_style=journal_style))
    plt.close(fig)
    return {"status": "ok", "files": [str(out_path)], "best_conf": best_conf}
