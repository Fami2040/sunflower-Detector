"""Post-train eval policy: device selection and GPU memory release after Ultralytics train."""

from __future__ import annotations

import gc
import os
from typing import Any


def post_train_eval_skipped(*, cli_skip: bool, eval_section: dict[str, Any] | None) -> bool:
    if cli_skip:
        return True
    if not isinstance(eval_section, dict):
        return False
    return bool(eval_section.get("skip"))


def resolve_post_train_eval_device(eval_section: dict[str, Any] | None) -> str | None:
    """
    Device for chained ``eval.py`` after ``train.py`` (val mAP + optional export).

    Precedence: config ``eval.device`` → ``HARCHOC_POST_TRAIN_EVAL_DEVICE`` →
    ``HARCHOC_EXPORT_DEVICE`` → auto (``cpu`` if <2 GiB CUDA free after train, else ``cuda``).
    """
    if isinstance(eval_section, dict):
        raw = eval_section.get("device")
        if raw is not None and str(raw).strip():
            return str(raw).strip()

    for key in ("HARCHOC_POST_TRAIN_EVAL_DEVICE", "HARCHOC_EXPORT_DEVICE"):
        v = (os.getenv(key) or "").strip()
        if v:
            return v

    return _auto_post_train_device()


def _auto_post_train_device() -> str:
    try:
        import torch  # type: ignore
    except Exception:
        return "cpu"
    if not bool(torch.cuda.is_available()):
        return "cpu"
    try:
        free_b, _total_b = torch.cuda.mem_get_info()
    except Exception:
        return "cuda"
    if int(free_b) < 2 * 1024**3:
        return "cpu"
    return "cuda"


def release_cuda_after_train(model: object | None) -> None:
    """Best-effort free GPU memory before loading weights again for eval."""
    del model
    gc.collect()
    try:
        import torch  # type: ignore

        if bool(torch.cuda.is_available()):
            torch.cuda.empty_cache()
    except Exception:
        pass


def restore_cuda_visible_devices_after_ultralytics_cpu(prior: str | None) -> None:
    """
    Ultralytics ``select_device('cpu')`` sets ``CUDA_VISIBLE_DEVICES=''``, which breaks
    a subsequent GPU train in the same Python process (matrix zoo rows).
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        return
    if prior is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = prior


def build_post_train_eval_argv(
    *,
    recorded_weights: str,
    eval_out: str,
    manifest: str,
    default_dataset_name: str,
    dataset_name: str | None,
    dataset_root: str | None,
    yolo_data_yaml: str | None,
    split_file: str | None,
    eval_section: dict[str, Any] | None,
    train_imgsz: int | None,
) -> list[str]:
    """Argv for ``scripts.eval.main`` after training."""
    argv: list[str] = [
        "--weights",
        recorded_weights,
        "--out",
        eval_out,
        "--manifest",
        manifest,
        "--default-dataset-name",
        default_dataset_name,
    ]
    if dataset_name:
        argv += ["--dataset-name", dataset_name]
    if dataset_root:
        argv += ["--dataset-root", dataset_root]
    if yolo_data_yaml:
        argv += ["--yolo-data-yaml", yolo_data_yaml]
    if split_file is not None:
        argv += ["--split-file", str(split_file)]

    section = eval_section if isinstance(eval_section, dict) else {}
    max_det = section.get("max_det")
    if max_det is not None:
        argv += ["--max-det", str(int(max_det))]

    imgsz = section.get("imgsz", train_imgsz)
    if imgsz is not None:
        argv += ["--imgsz", str(int(imgsz))]

    device = resolve_post_train_eval_device(section)
    if device:
        argv += ["--device", device, "--export-device", device]

    if bool(section.get("export_only")):
        argv.append("--export-only")

    export_gt = section.get("export_gt_json")
    export_preds = section.get("export_preds_json")
    if export_gt:
        argv += ["--export-gt-json", str(export_gt)]
    if export_preds:
        argv += ["--export-preds-json", str(export_preds)]

    locked = section.get("locked_conf_from")
    if locked:
        argv += ["--locked-conf-from", str(locked)]

    return argv
