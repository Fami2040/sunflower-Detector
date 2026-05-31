from __future__ import annotations

from typing import Any

from harchoc.gpu_probe import torch_cuda_payload, try_import_torch


def snapshot_from_train_cfg(train_cfg: dict[str, Any], *, include_torch: bool = True) -> dict[str, Any]:
    """
    Lightweight resource-ish metadata that is safe in CI.

    - Does not require torch; if torch is available, includes a small CUDA payload.
    - Intended for `meta.json` / matrix JSON provenance, not for live monitoring.
    """
    out: dict[str, Any] = {
        "device": train_cfg.get("device"),
        "workers": train_cfg.get("workers"),
        "batch": train_cfg.get("batch"),
        "imgsz": train_cfg.get("imgsz"),
        "amp": train_cfg.get("amp"),
        "nbs": train_cfg.get("nbs"),
    }

    if not include_torch:
        out["torch"] = {"available": None, "skipped": True}
        return out

    torch, _torchvision, err = try_import_torch()
    if err:
        out["torch"] = {"available": False, "import_error": err}
        return out
    assert torch is not None
    payload = torch_cuda_payload(torch)
    payload["available"] = True
    out["torch"] = payload
    return out

