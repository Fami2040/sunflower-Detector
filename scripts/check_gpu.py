from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from harchoc.script_entry import bootstrap_repo_imports

bootstrap_repo_imports()
from harchoc.gpu_probe import fmt_bytes, matmul_bench, torch_cuda_payload, try_import_torch
from scripts._common_cli import write_json


def _print_cuda_guidance() -> None:
    print("CUDA is not available via PyTorch on this machine.")
    print()
    print("Guidance:")
    print("- Ensure NVIDIA driver + CUDA runtime are installed (e.g. `nvidia-smi` works).")
    print("- Ensure you installed a CUDA-enabled PyTorch build (not CPU-only).")
    print("- In this repo, prefer the conda env in `envs/mamba-gpu.yml` (pytorch-cuda=12.1).")


def _build_payload(
    *,
    dry_run: bool,
    n: int,
    iters: int,
) -> tuple[dict[str, Any], int]:
    """Return JSON-serializable report and process exit code."""
    payload: dict[str, Any] = {
        "script": "check_gpu",
        "schema_version": "check_gpu.v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dry_run": dry_run,
        "torch": None,
        "torchvision_version": None,
        "bench": None,
    }

    torch, torchvision, err = try_import_torch()
    if err:
        payload["status"] = "missing_torch"
        payload["torch"] = {"import_error": err}
        return payload, 0 if dry_run else 2

    assert torch is not None
    tv_ver = getattr(torchvision, "__version__", "not-installed") if torchvision is not None else "not-installed"
    payload["torchvision_version"] = tv_ver
    info = torch_cuda_payload(torch)
    payload["torch"] = info

    if not info.get("cuda_available"):
        payload["status"] = "cuda_unavailable"
        return payload, 0

    if "device_error" in info:
        payload["status"] = "device_error"
        return payload, 0 if dry_run else 2

    if dry_run:
        payload["status"] = "ok"
        return payload, 0

    try:
        bench = matmul_bench(torch, device="cuda", n=n, iters=iters)
    except Exception as ex:
        payload["status"] = "bench_failed"
        payload["bench"] = {"error": str(ex)}
        return payload, 2

    payload["status"] = "ok"
    payload["bench"] = bench
    return payload, 0


def cmd_sanity(argv: list[str] | None = None) -> int:
    """Matmul + CUDA JSON report (formerly gpu_sanity.py)."""
    from scripts._common_cli import add_dry_run_arg, write_json

    p = argparse.ArgumentParser(description="GPU sanity checks with JSON output.")
    add_dry_run_arg(p)
    p.add_argument("--out", default="reports/gpu_sanity.json", help="Where to write JSON report.")
    p.add_argument("--n", type=int, default=1024, help="Matmul size (NxN).")
    p.add_argument("--iters", type=int, default=20, help="Number of matmuls to time.")
    args = p.parse_args(argv)

    payload: dict[str, Any] = {
        "script": "check_gpu",
        "subcommand": "sanity",
        "schema_version": "gpu_sanity.v1",
        "status": "dry-run" if args.dry_run else "ok",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": None,
        "bench": None,
    }

    torch, _, err = try_import_torch()
    if err:
        payload["status"] = "dry-run" if args.dry_run else "missing_torch"
        payload["torch"] = {"import_error": err}
        write_json(args.out, payload)
        return 0

    assert torch is not None
    payload["torch"] = torch_cuda_payload(torch)

    if (not args.dry_run) and bool(payload["torch"].get("cuda_available")):
        try:
            payload["bench"] = matmul_bench(torch, n=int(args.n), iters=int(args.iters))
        except Exception as ex:
            payload["status"] = "bench_failed"
            payload["bench"] = {"error": str(ex)}

    out = write_json(args.out, payload)
    if not args.dry_run:
        print(f"Wrote {out}")
    return 0


def cmd_smoke_ultralytics(argv: list[str] | None = None) -> int:
    """Ultralytics predict smoke (formerly gpu_smoke_ultralytics.py)."""
    import time
    from pathlib import Path

    from harchoc.run_metadata import collect_run_metadata
    from harchoc.schemas import with_schema_version
    from scripts._common_cli import add_dry_run_arg, write_json

    p = argparse.ArgumentParser(description="Ultralytics GPU smoke test.")
    add_dry_run_arg(p)
    p.add_argument("--out", default="reports/gpu_smoke_ultralytics.json")
    p.add_argument("--weights", default="", help="Weights path or model id.")
    p.add_argument("--img", action="append", default=[], help="Image path (repeatable).")
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    meta = collect_run_metadata(
        repo_root=repo_root,
        dataset_manifest=repo_root / "data" / "manifest.json",
        extra_files={
            "weights": str(args.weights),
            "images": str(args.img[0]) if args.img else "",
        },
    )

    if args.dry_run:
        out = write_json(
            args.out,
            with_schema_version(
                {
                    "status": "dry-run",
                    "script": "check_gpu",
                    "subcommand": "smoke-ultralytics",
                    "meta": meta,
                    "weights": str(args.weights),
                    "images": [str(x) for x in (args.img or [])],
                    "imgsz": int(args.imgsz),
                },
                schema_version="gpu_smoke_ultralytics.v1",
            ),
        )
        print(f"Wrote {out}")
        return 0

    if not (args.weights or "").strip():
        raise SystemExit("--weights is required.")
    if not args.img:
        raise SystemExit("Provide at least one --img path.")

    images = [str(Path(x).expanduser().resolve()) for x in args.img]
    for x in images:
        if not Path(x).is_file():
            raise SystemExit(f"Image does not exist: {x}")

    import torch  # type: ignore
    from ultralytics import YOLO  # type: ignore

    if not bool(torch.cuda.is_available()):
        raise SystemExit("CUDA not available according to torch.cuda.is_available().")

    t0 = time.perf_counter()
    model = YOLO(str(args.weights))
    _res = model.predict(images, imgsz=int(args.imgsz), device=0, verbose=False)
    runtime_s = float(time.perf_counter() - t0)

    payload = with_schema_version(
        {
            "status": "ok",
            "script": "check_gpu",
            "subcommand": "smoke-ultralytics",
            "meta": meta,
            "weights": str(args.weights),
            "images": images,
            "imgsz": int(args.imgsz),
            "runtime_s": runtime_s,
            "cuda_available": True,
            "torch_version": getattr(torch, "__version__", "unknown"),
        },
        schema_version="gpu_smoke_ultralytics.v1",
    )
    out = write_json(args.out, payload)
    print(f"Wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    from harchoc.ml_env import should_reexec_in_mamba_for_torch

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("check", "sanity", "smoke-ultralytics"):
        cmd = argv[0]
        rest = argv[1:]
        if cmd == "sanity":
            return cmd_sanity(rest)
        if cmd == "smoke-ultralytics":
            return cmd_smoke_ultralytics(rest)
        argv = rest

    if should_reexec_in_mamba_for_torch():
        from harchoc.ml_env import reexec_script_in_mamba_env

        reexec_script_in_mamba_env(argv)

    p = argparse.ArgumentParser(description="Quick GPU sanity check for PyTorch.")
    p.add_argument("--dry-run", action="store_true", help="Print environment info only; skip matmul benchmark.")
    p.add_argument("--json-out", metavar="PATH", help="Write JSON report (uses harchoc.gpu_probe).")
    p.add_argument("--n", type=int, default=2048, help="Matmul size (NxN).")
    p.add_argument("--iters", type=int, default=50, help="Number of matmuls to time.")
    p.add_argument(
        "--verify-deps",
        action="store_true",
        help=(
            "Run pip check + optional super_gradients / external DETR imports "
            "(HARCHOC_CHECK_SG=1, HARCHOC_CHECK_EXTERNAL=1)."
        ),
    )
    args = p.parse_args(argv)

    if args.verify_deps:
        import os as _os

        from harchoc.env_health import env_health_report

        env_name = _os.path.basename((_os.getenv("CONDA_PREFIX") or "harchoc").rstrip("/"))
        sg = _os.getenv("HARCHOC_CHECK_SG", "").strip() in ("1", "true", "yes")
        external = _os.getenv("HARCHOC_CHECK_EXTERNAL", "").strip() in ("1", "true", "yes")
        health = env_health_report(
            env=env_name,
            with_super_gradients=sg,
            with_external_detr=external,
        )
        print(json.dumps(health, indent=2))
        return 0 if health.get("status") == "ok" else 1

    if args.json_out:
        payload, code = _build_payload(dry_run=bool(args.dry_run), n=int(args.n), iters=int(args.iters))
        out = write_json(args.json_out, payload)
        print(f"Wrote {out}")
        return code

    torch, torchvision, err = try_import_torch()
    print(f"python: {platform.python_version()} ({platform.platform()})")
    if err:
        print(err)
        if args.dry_run:
            print()
            print("Install PyTorch first, then re-run:")
            print("- conda/mamba env: see `envs/mamba-gpu.yml` or `envs/mamba-cpu.yml`")
            return 0
        return 2

    assert torch is not None
    torch_version = getattr(torch, "__version__", "unknown")
    torchvision_version = getattr(torchvision, "__version__", "not-installed") if torchvision is not None else "not-installed"
    print(f"torch: {torch_version}")
    print(f"torchvision: {torchvision_version}")

    info = torch_cuda_payload(torch)
    print(f"torch.version.cuda: {info.get('torch_cuda_version')}")
    print(f"cuda_available: {info.get('cuda_available')}")

    if not info.get("cuda_available"):
        if args.dry_run:
            _print_cuda_guidance()
            return 0
        _print_cuda_guidance()
        return 0

    if "device_error" in info:
        print(f"Failed to query CUDA device details: {info['device_error']}")
        if args.dry_run:
            return 0
        return 2

    idx = info["device_index"]
    name = info["device_name"]
    cap = info["device_capability"]
    total_mem = info["total_memory_bytes"]
    print(f"device: cuda:{idx} ({name})")
    print(f"capability: {tuple(cap) if isinstance(cap, list) else cap}")
    print(f"total_memory: {fmt_bytes(int(total_mem))}")

    if args.dry_run:
        return 0

    try:
        bench = matmul_bench(torch, device="cuda", n=int(args.n), iters=int(args.iters))
    except Exception as ex:
        print(f"GPU matmul failed: {ex}")
        return 2

    print(f"matmul: n={args.n} iters={args.iters} elapsed_s={bench['elapsed_s']:.4f} tflops={bench['tflops']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
