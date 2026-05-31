from __future__ import annotations

import time
from typing import Any

from harchoc import strict_ml


def try_import_torch() -> tuple[object | None, object | None, str | None]:
    """Return ``(torch, torchvision, error)``. *error* is set when torch import fails."""
    try:
        import torch  # type: ignore
    except Exception as ex:
        return None, None, f"Failed to import torch: {ex}"

    try:
        import torchvision  # type: ignore
    except ImportError:
        torchvision = None

    return torch, torchvision, None


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    units = ["KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        x /= 1024.0
        if x < 1024.0:
            return f"{x:.2f} {u}"
    return f"{x:.2f} PiB"


def torch_cuda_payload(torch: object) -> dict[str, Any]:
    """CUDA availability and device details suitable for JSON reports."""
    t = torch
    cuda_available = bool(getattr(t.cuda, "is_available")())
    out: dict[str, Any] = {
        "torch_version": getattr(t, "__version__", "unknown"),
        "torch_cuda_version": getattr(getattr(t, "version", object()), "cuda", None),
        "cuda_available": cuda_available,
    }
    if not cuda_available:
        return out
    try:
        idx = int(getattr(t.cuda, "current_device")())
        name = str(getattr(t.cuda, "get_device_name")(idx))
        cap = getattr(t.cuda, "get_device_capability")(idx)
        props = getattr(t.cuda, "get_device_properties")(idx)
        total_mem = int(getattr(props, "total_memory"))
    except Exception as ex:
        out["device_error"] = str(ex)
        return out
    out.update(
        {
            "device_index": idx,
            "device_name": name,
            "device_capability": list(cap) if isinstance(cap, tuple) else cap,
            "total_memory_bytes": total_mem,
        }
    )
    # Optional: free/total memory (PyTorch 1.10+). Keep best-effort and non-fatal.
    with strict_ml.capture_failure("cuda_mem_get_info") as cap:
        if hasattr(t.cuda, "mem_get_info"):
            free_b, total_b = getattr(t.cuda, "mem_get_info")()
            out["free_memory_bytes"] = int(free_b)
            out["total_memory_bytes_cuda_api"] = int(total_b)
    if cap.failed:
        out["mem_get_info_error"] = f"{cap.exc_type}: {cap.exc_msg}"
    return out


def matmul_bench(torch: object, *, device: str = "cuda", n: int, iters: int) -> dict[str, Any]:
    """Small float16 matmul benchmark; returns elapsed seconds and TFLOPS."""
    t = torch

    a = t.randn((n, n), device=device, dtype=t.float16)
    b = t.randn((n, n), device=device, dtype=t.float16)

    for _ in range(3):
        _ = a @ b
    t.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = a @ b
    t.cuda.synchronize()
    elapsed_s = time.perf_counter() - start

    flops = 2.0 * (float(n) ** 3) * float(iters)
    tflops = flops / elapsed_s / 1e12 if elapsed_s > 0 else 0.0
    return {"n": n, "iters": iters, "elapsed_s": float(elapsed_s), "tflops": float(tflops)}
