"""Grad-CAM panel helpers for FP crops from error_analysis reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harchoc.strict_ml import append_ml_error, record_ml_failure


def load_fp_crop_entries(report_path: str | Path) -> list[dict[str, Any]]:
    """Return exported FP crop rows (status=ok) from an error_analysis report."""
    obj = json.loads(Path(report_path).expanduser().read_text("utf-8"))
    crops = obj.get("fp_crops")
    if not isinstance(crops, dict):
        return []
    results = crops.get("results")
    if not isinstance(results, list):
        return []
    return [r for r in results if isinstance(r, dict) and r.get("status") == "ok"]


def select_panel_entries(
    entries: list[dict[str, Any]],
    *,
    max_panels: int = 12,
) -> list[dict[str, Any]]:
    """Pick diverse error types first, then fill by score."""
    if len(entries) <= max_panels:
        return list(entries)
    by_type: dict[str, list[dict[str, Any]]] = {}
    for ex in entries:
        t = str(ex.get("error_type") or "fp")
        by_type.setdefault(t, []).append(ex)
    for t in by_type:
        by_type[t].sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)
    picked: list[dict[str, Any]] = []
    types = sorted(by_type.keys())
    while len(picked) < max_panels:
        progressed = False
        for t in types:
            bucket = by_type.get(t) or []
            if bucket:
                picked.append(bucket.pop(0))
                progressed = True
                if len(picked) >= max_panels:
                    break
        if not progressed:
            break
    return picked


def plan_gradcam_panel(
    *,
    report_path: str | Path | None,
    max_panels: int = 12,
) -> dict[str, Any]:
    """Build a dry-run / planning payload for fig_gradcam_panel."""
    entries: list[dict[str, Any]] = []
    if report_path:
        entries = select_panel_entries(load_fp_crop_entries(report_path), max_panels=max_panels)
    panels = [
        {
            "error_type": ex.get("error_type"),
            "score": ex.get("score"),
            "crop_path": ex.get("crop_path"),
            "image_path": ex.get("image_path"),
        }
        for ex in entries
    ]
    return {
        "max_panels": max_panels,
        "n_selected": len(panels),
        "panels": panels,
        "report_path": str(Path(report_path).resolve()) if report_path else None,
    }


def _gradcam_panel_status(
    *,
    weights: str | Path | None,
    gradcam_overlays: int,
) -> str:
    if weights and gradcam_overlays == 0:
        return "partial"
    return "ok"


def _get_yolo_net(weights: str | Path, cache: dict[str, Any]) -> Any:
    key = str(Path(weights).resolve())
    if key not in cache:
        from ultralytics import YOLO  # type: ignore

        cache[key] = YOLO(key).model
    return cache[key]


def _gradcam_scalar_target(out: Any, activations: list[Any]) -> Any:
    import torch  # type: ignore

    if isinstance(out, torch.Tensor):
        return out.sum()
    if isinstance(out, dict):
        scores = out.get("scores")
        if scores is not None and isinstance(scores, torch.Tensor):
            return scores.sum()
        tensors = [v for v in out.values() if isinstance(v, torch.Tensor)]
        if tensors:
            return sum(t.sum() for t in tensors)
    if isinstance(out, tuple):
        tensors = [o for o in out if isinstance(o, torch.Tensor)]
        if tensors:
            return sum(t.sum() for t in tensors)
    return activations[0].mean()


def render_gradcam_mosaic(
    *,
    entries: list[dict[str, Any]],
    out_path: str | Path,
    weights: str | Path | None = None,
    suptitle: str | None = None,
    journal_style: bool = True,
) -> dict[str, Any]:
    """
    Render a labeled mosaic from FP crops.

    When ``weights`` is set and torch/ultralytics are available, overlays a coarse
    Grad-CAM heatmap on each crop; otherwise saves crops with taxonomy captions only.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    gradcam_errors: list[dict[str, Any]] = []

    try:
        from harchoc.figure_style import add_panel_label, panel_label, prepare_matplotlib, savefig_kwargs

        prepare_matplotlib(journal_style=journal_style)
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.gridspec import GridSpec  # type: ignore
    except Exception as ex:
        return {
            "status": "skipped",
            "reason": f"matplotlib unavailable: {ex}",
            "gradcam_errors": gradcam_errors,
        }

    if not entries:
        return {
            "status": "skipped",
            "reason": "no crop entries",
            "gradcam_errors": gradcam_errors,
        }

    n = len(entries)
    cols = min(4, max(1, n))
    rows = (n + cols - 1) // cols
    cell = 2.2 if journal_style else 3.0
    fig = plt.figure(figsize=(cell * cols, cell * rows))
    gs = GridSpec(rows, cols, figure=fig, wspace=0.08, hspace=0.28)

    gradcam_ok = 0
    yolo_cache: dict[str, Any] = {}
    for i, ex in enumerate(entries):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        crop_path = Path(str(ex.get("crop_path") or ""))
        if not crop_path.is_file():
            ax.set_title(f"missing: {ex.get('error_type')}")
            ax.axis("off")
            continue
        try:
            from PIL import Image  # type: ignore

            with Image.open(crop_path) as im:
                rgb = im.convert("RGB")
            title = f"{ex.get('error_type')} s={float(ex.get('score') or 0.0):.2f}"
            if weights:
                try:
                    if _try_gradcam_overlay(
                        rgb,
                        ex,
                        weights,
                        panel_index=i,
                        errors=gradcam_errors,
                        yolo_cache=yolo_cache,
                    ):
                        gradcam_ok += 1
                        title += " +cam"
                except Exception as ex_cam:
                    record_ml_failure(gradcam_errors, panel_index=i, exc=ex_cam)
            ax.imshow(rgb)
            ax.set_title(title)
        except Exception as ex_panel:
            record_ml_failure(gradcam_errors, panel_index=i, exc=ex_panel)
            ax.set_title(str(ex.get("error_type")))
        ax.axis("off")
        if journal_style:
            add_panel_label(ax, panel_label(i), x=-0.02, y=1.02, fontsize=10)

    if suptitle:
        fig.suptitle(suptitle, fontsize=10, y=1.02 if journal_style else 0.98)
    fig.tight_layout()
    fig.savefig(out, **savefig_kwargs(journal_style=journal_style))
    plt.close(fig)
    status = _gradcam_panel_status(weights=weights, gradcam_overlays=gradcam_ok)
    return {
        "status": status,
        "out_path": str(out.resolve()),
        "n_panels": n,
        "gradcam_overlays": gradcam_ok,
        "gradcam_errors": gradcam_errors,
        "weights": str(weights) if weights else None,
    }


def _try_gradcam_overlay(
    rgb: Any,
    entry: dict[str, Any],
    weights: str | Path,
    *,
    panel_index: int = 0,
    errors: list[dict[str, Any]] | None = None,
    yolo_cache: dict[str, Any] | None = None,
) -> bool:
    """Best-effort Grad-CAM on crop; returns True when overlay applied."""
    err_log = errors if errors is not None else []

    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as ex:
        append_ml_error(err_log, panel_index=panel_index, exc=ex)
        return False

    image_path = entry.get("image_path")
    bbox = entry.get("bbox") or entry.get("crop_xyxy")
    if not image_path or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        append_ml_error(
            err_log,
            panel_index=panel_index,
            exc=ValueError("missing image_path or bbox for Grad-CAM"),
        )
        return False

    cache: dict[str, Any] = yolo_cache if yolo_cache is not None else {}
    hooks: list[Any] = []
    was_training = False
    net = None
    try:
        net = _get_yolo_net(weights, cache)
        was_training = bool(net.training)
        target_layer = net.model[-2]
        activations: list[Any] = []
        gradients: list[Any] = []

        def fwd_hook(_module: Any, _inp: Any, out: Any) -> None:
            activations.append(out)

        def bwd_hook(_module: Any, _gin: Any, gout: Any) -> None:
            grad = gout[0] if isinstance(gout, tuple) else gout
            if grad is not None:
                gradients.append(grad)

        hooks.append(target_layer.register_forward_hook(fwd_hook))
        hooks.append(target_layer.register_full_backward_hook(bwd_hook))

        with Image.open(str(image_path)) as im:
            full = im.convert("RGB")
        w, h = full.size
        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            append_ml_error(
                err_log,
                panel_index=panel_index,
                exc=ValueError(f"degenerate bbox: {(x1, y1, x2, y2)}"),
            )
            return False

        crop = full.crop((int(x1), int(y1), int(x2), int(y2)))
        arr = np.array(crop.resize((224, 224)), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
        tensor.requires_grad_(True)
        device = next(net.parameters()).device
        tensor = tensor.to(device)

        net.train()
        with torch.enable_grad():
            net.zero_grad(set_to_none=True)
            out = net(tensor)
            if not activations:
                append_ml_error(
                    err_log,
                    panel_index=panel_index,
                    exc=RuntimeError("no activations from target layer"),
                )
                return False
            score = _gradcam_scalar_target(out, activations)
            if not getattr(score, "requires_grad", False):
                append_ml_error(
                    err_log,
                    panel_index=panel_index,
                    exc=RuntimeError("Grad-CAM target does not require grad"),
                )
                return False
            score.backward(retain_graph=False)
        if not gradients:
            append_ml_error(
                err_log,
                panel_index=panel_index,
                exc=RuntimeError("no gradients from target layer backward hook"),
            )
            return False

        weights_cam = gradients[0].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights_cam * activations[0]).sum(dim=1, keepdim=True))
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_img = Image.fromarray((cam * 255).astype("uint8")).resize(crop.size)
        cam_arr = np.array(cam_img, dtype=np.float32) / 255.0
        base = np.array(crop, dtype=np.float32) / 255.0
        heat = np.clip(
            base * 0.5 + np.stack([cam_arr, cam_arr * 0.3, cam_arr * 0.1], axis=-1) * 0.5,
            0,
            1,
        )
        rgb.paste(Image.fromarray((heat * 255).astype("uint8")))
        return True
    except Exception as ex:
        record_ml_failure(err_log, panel_index=panel_index, exc=ex)
        return False
    finally:
        for hook in hooks:
            try:
                hook.remove()
            except Exception:
                pass
        if net is not None:
            try:
                net.train(was_training)
            except Exception:
                pass
