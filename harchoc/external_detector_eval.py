"""HSP test eval for external DETR checkpoints (RT-DETRv2 / D-FINE / DEIM)."""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from harchoc.aug_smoke_runner import extract_count_mae
from harchoc.detector_sources import entry_for_bench
from harchoc.external_detector_train import external_bench_availability
from harchoc.external_repos import resolve_external_repo_path, spec_for_train_stack
from harchoc.hsp_eval_chain import DEFAULT_LOCKED_CONF_FROM, build_error_analysis_argv
from harchoc.hsp_export_protocol import (
    DEFAULT_EXPORT_MAX_DET,
    EXPORT_CONF,
    EXPORT_IOU,
    export_result_meta,
)
from harchoc.ml_env import run_repo_python


@contextmanager
def _upstream_import_context(repo_dir: Path) -> Iterator[None]:
    repo_str = str(repo_dir.resolve())
    inserted = repo_str not in sys.path
    if inserted:
        sys.path.insert(0, repo_str)
    try:
        yield
    finally:
        if inserted:
            sys.path.remove(repo_str)


def _extract_model_state(checkpoint: dict[str, Any]) -> dict[str, Any]:
    ema = checkpoint.get("ema")
    if isinstance(ema, dict) and isinstance(ema.get("module"), dict):
        return ema["module"]
    model = checkpoint.get("model")
    if isinstance(model, dict):
        return model
    return checkpoint


def external_results_to_detections(
    labels: Any,
    boxes: Any,
    scores: Any,
    *,
    max_det: int,
) -> list[dict[str, Any]]:
    """Convert deploy-mode postprocessor output to HSP preds JSON detection entries."""
    try:
        lab_rows = labels.cpu().detach().numpy().reshape(-1)
        box_rows = boxes.cpu().detach().numpy().reshape(-1, 4)
        score_rows = scores.cpu().detach().numpy().reshape(-1)
    except Exception:
        return []

    detections: list[dict[str, Any]] = []
    n = min(len(lab_rows), len(box_rows), len(score_rows))
    for i in range(n):
        x1, y1, x2, y2 = (float(v) for v in box_rows[i][:4])
        detections.append(
            {
                "bbox": [x1, y1, x2, y2],
                "category_id": int(lab_rows[i]),
                "score": float(score_rows[i]),
            }
        )
    detections.sort(key=lambda d: float(d.get("score") or 0.0), reverse=True)
    if max_det > 0:
        detections = detections[: int(max_det)]
    return detections


def _build_deploy_model(
    *,
    repo_dir: Path,
    train_stack: str,
    config_path: Path,
    weights_path: Path,
    device: str,
) -> Any:
    import torch
    import torch.nn as nn

    with _upstream_import_context(repo_dir):
        if train_stack == "deim":
            from engine.core import YAMLConfig  # type: ignore[import-not-found]
        else:
            from src.core import YAMLConfig  # type: ignore[import-not-found]

        cfg = YAMLConfig(str(config_path))
        checkpoint = torch.load(str(weights_path), map_location="cpu")
        state = _extract_model_state(checkpoint if isinstance(checkpoint, dict) else {})
        cfg.model.load_state_dict(state, strict=False)

        class DeployModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()

            def forward(self, images: torch.Tensor, orig_target_sizes: torch.Tensor) -> tuple[Any, Any, Any]:
                outputs = self.model(images)
                return self.postprocessor(outputs, orig_target_sizes)

        return DeployModel().to(device).eval()


def export_hsp_gt_preds_json(
    *,
    source_id: str,
    model_id: str | None,
    weights: str | Path,
    split_file: Path,
    dataset_root: Path,
    gt_out: Path,
    preds_out: Path,
    config_path: Path,
    imgsz: int = 1280,
    max_det: int = 3000,
    device: str | None = None,
) -> dict[str, Any]:
    """Export GT + external-stack predictions for HSP error_analysis."""
    from harchoc.eval_export import build_gt_export, iter_split_image_paths

    entry = entry_for_bench(model_id=model_id, source_id=source_id)
    if entry is None:
        raise ValueError(f"unknown external source: {source_id!r}")

    weights_path = Path(weights).resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"weights not found: {weights_path}")

    spec = spec_for_train_stack(entry.train_stack)
    if spec is None:
        raise ValueError(f"unknown train_stack: {entry.train_stack}")
    repo = resolve_external_repo_path(entry.train_stack)
    if repo is None:
        raise FileNotFoundError(f"missing external repo for train_stack={entry.train_stack}")

    if not config_path.is_file():
        raise FileNotFoundError(f"missing eval config: {config_path}")

    gt_obj = build_gt_export(split_file=split_file, dataset_root=dataset_root)
    gt_out.parent.mkdir(parents=True, exist_ok=True)
    preds_out.parent.mkdir(parents=True, exist_ok=True)
    gt_out.write_text(json.dumps(gt_obj, indent=2) + "\n", encoding="utf-8")

    try:
        import torch
        import torchvision.transforms as T
        from PIL import Image
    except ImportError as exc:
        raise ImportError(f"missing_dependency:torch ({exc})") from exc

    dev = (device or os.environ.get("HARCHOC_EXPORT_DEVICE") or "cpu").strip()
    model = _build_deploy_model(
        repo_dir=repo,
        train_stack=entry.train_stack,
        config_path=config_path,
        weights_path=weights_path,
        device=dev,
    )
    transforms = T.Compose([T.Resize((int(imgsz), int(imgsz))), T.ToTensor()])

    images: list[dict[str, Any]] = []
    with torch.inference_mode():
        for img_id, img_path, file_name in iter_split_image_paths(split_file, dataset_root=dataset_root):
            if not img_path.is_file():
                continue
            im_pil = Image.open(img_path).convert("RGB")
            w, h = im_pil.size
            im_data = transforms(im_pil)[None].to(dev)
            orig_size = torch.tensor([[w, h]], dtype=torch.float32, device=dev)
            labels, boxes, scores = model(im_data, orig_size)
            dets = external_results_to_detections(labels, boxes, scores, max_det=max_det)
            images.append({"image_id": img_id, "file_name": file_name, "detections": dets})

    preds_obj = {"images": images}
    preds_out.write_text(json.dumps(preds_obj, indent=2) + "\n", encoding="utf-8")
    return {
        **export_result_meta(max_det=max_det),
        "gt_json": str(gt_out.resolve()),
        "preds_json": str(preds_out.resolve()),
        "n_images": len(images),
        "backend": "external",
        "weights": str(weights_path),
        "source_id": source_id,
        "train_stack": entry.train_stack,
        "config_path": str(config_path.resolve()),
    }


def _resolve_eval_config(*, run_dir: Path | None, entry: Any, repo: Path) -> Path | None:
    if run_dir is not None:
        overlay = (run_dir / "harchoc_train_overlay.yml").resolve()
        if overlay.is_file():
            return overlay
    upstream = (repo / entry.config_relpath).resolve()
    return upstream if upstream.is_file() else None


def _repo_relative_or_absolute(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _run_error_analysis(repo_root: Path, error_argv: list[str], env: dict[str, str]) -> int:
    proc = run_repo_python(error_argv, repo_root=repo_root, env=env)
    return int(proc.returncode)


def eval_hsp_for_bench(
    *,
    source_id: str,
    model_id: str | None,
    weights: str | Path,
    dataset_root: Path,
    run_dir: Path | None = None,
    eval_out: Path | None = None,
    max_det: int | None = None,
    imgsz: int = 1280,
    locked_conf_from: str = DEFAULT_LOCKED_CONF_FROM,
    repo_root: Path | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """HSP protocol for external rows: export test preds + error_analysis @ locked conf."""
    rr = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    weights_path = Path(weights).resolve()
    cap_max_det = int(max_det if max_det is not None else 3000)

    base_fail: dict[str, Any] = {
        "status": "failed",
        "split": "test",
        "backend": "external",
        "source_id": source_id,
        "weights": str(weights_path),
    }
    if not weights_path.is_file():
        return {**base_fail, "reason": "weights_not_found"}

    entry = entry_for_bench(model_id=model_id, source_id=source_id)
    if entry is None:
        return {**base_fail, "reason": "unknown_external_source"}

    available, missing_reason = external_bench_availability(model_id=model_id, source_id=source_id)
    if not available:
        return {
            "status": "skipped",
            "reason": missing_reason,
            "split": "test",
            "backend": "external",
            "source_id": source_id,
            "weights": str(weights_path),
        }

    repo = resolve_external_repo_path(entry.train_stack)
    assert repo is not None
    cfg_path = Path(config_path).resolve() if config_path else _resolve_eval_config(
        run_dir=run_dir, entry=entry, repo=repo
    )
    if cfg_path is None or not cfg_path.is_file():
        return {**base_fail, "reason": "missing_eval_config", "config_path": str(cfg_path)}

    from scripts.train import _resolve_test_split_file

    split_file = _resolve_test_split_file(repo_root=rr, dataset_root=dataset_root)
    if split_file is None:
        return {**base_fail, "reason": "missing_test_split"}

    run_name = run_dir.name if run_dir is not None else Path(weights_path).stem
    artifact_dir = (run_dir or (rr / "reports" / "hsp" / "external" / run_name)).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    gt_json = artifact_dir / "gt_test.json"
    preds_json = artifact_dir / "preds_test.json"
    error_json = artifact_dir / "error_test.json"
    eval_json = eval_out or (artifact_dir / "test_eval.json")

    try:
        export_hsp_gt_preds_json(
            source_id=source_id,
            model_id=model_id,
            weights=weights_path,
            split_file=split_file,
            dataset_root=dataset_root,
            gt_out=gt_json,
            preds_out=preds_json,
            config_path=cfg_path,
            imgsz=int(imgsz),
            max_det=cap_max_det,
            device=(os.environ.get("HARCHOC_EXPORT_DEVICE") or "cpu"),
        )
    except ImportError as exc:
        return {
            "status": "skipped",
            "reason": str(exc),
            "split": "test",
            "backend": "external",
            "source_id": source_id,
            "weights": str(weights_path),
        }
    except Exception as exc:
        return {
            **base_fail,
            "reason": str(exc),
            "exc_type": type(exc).__name__,
        }

    error_argv = build_error_analysis_argv(
        gt_json,
        preds_json,
        locked_conf_from,
        error_json,
        repo_root=rr,
    )
    env = dict(os.environ)
    rc = _run_error_analysis(rr, error_argv, env)
    if rc != 0:
        return {
            **base_fail,
            "reason": "error_analysis_failed",
            "returncode": rc,
            "gt_json": str(gt_json),
            "preds_json": str(preds_json),
            "error_json": str(error_json),
            "eval_out": str(eval_json),
        }

    mae, mae_ci = extract_count_mae(error_json)
    payload: dict[str, Any] = {
        "status": "ok" if mae is not None else "failed",
        "returncode": 0,
        "split": "test",
        "backend": "external",
        "source_id": source_id,
        "weights": str(weights_path),
        "locked_conf_from": locked_conf_from,
        "max_det": cap_max_det,
        "gt_json": str(gt_json.resolve()),
        "preds_json": str(preds_json.resolve()),
        "error_json": str(error_json.resolve()),
        "eval_out": str(eval_json.resolve()),
        "test_count_mae": mae,
        "test_count_mae_ci": mae_ci,
        "config_path": str(cfg_path),
    }
    eval_json.parent.mkdir(parents=True, exist_ok=True)
    eval_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if mae is None:
        payload["reason"] = "missing_count_mae"
        payload["status"] = "failed"
    return payload
