from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from harchoc.coco_detection_export import materialize_coco_detection_tree
from harchoc.detector_sources import entry_for_bench
from harchoc.external_repos import (
    default_external_root,
    load_external_repo_specs,
    resolve_external_repo_path,
    spec_for_train_stack,
    validate_repo_layout,
)


def external_bench_availability(
    *,
    model_id: str | None,
    source_id: str | None,
) -> tuple[bool, str | None]:
    entry = entry_for_bench(model_id=model_id, source_id=source_id)
    if entry is None:
        return False, "unknown_external_source"
    if not entry.cache_path.is_file() or entry.cache_path.stat().st_size <= 0:
        return False, f"checkpoint_not_cached:{entry.source_id}"
    spec = spec_for_train_stack(entry.train_stack)
    if spec is None:
        return False, f"unknown_train_stack:{entry.train_stack}"
    repo = resolve_external_repo_path(entry.train_stack)
    if repo is None:
        return False, (
            f"missing_repo:{entry.train_stack} "
            "(run: python scripts/check_weights_cache.py --download --strict)"
        )
    issues = validate_repo_layout(spec, working_dir=repo)
    if issues:
        return False, f"invalid_repo:{entry.train_stack}:{issues[0]}"
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "missing_dependency:torch"
    return True, None


def write_train_overlay_yaml(
    *,
    out_path: Path,
    upstream_config: Path,
    coco_root: Path,
    output_dir: Path,
    epochs: int,
    imgsz: int,
    num_classes: int = 2,
) -> None:
    train_img = (coco_root / "images" / "train").resolve()
    val_img = (coco_root / "images" / "val").resolve()
    train_ann = (coco_root / "annotations" / "instances_train.json").resolve()
    val_ann = (coco_root / "annotations" / "instances_val.json").resolve()
    lines = [
        f"__include__: ['{upstream_config.as_posix()}']",
        f"output_dir: {output_dir.as_posix()}",
        f"epoches: {int(epochs)}",
        "remap_mscoco_category: false",
        f"num_classes: {int(num_classes)}",
        "train_dataloader:",
        "  dataset:",
        f"    img_folder: {train_img.as_posix()}",
        f"    ann_file: {train_ann.as_posix()}",
        "    return_masks: false",
        "  collate_fn:",
        f"    base_size: {int(imgsz)}",
        "val_dataloader:",
        "  dataset:",
        f"    img_folder: {val_img.as_posix()}",
        f"    ann_file: {val_ann.as_posix()}",
        "    return_masks: false",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _find_best_checkpoint(output_dir: Path) -> Path | None:
    patterns = ("best.pth", "best_stg1.pth", "last.pth")
    for pat in patterns:
        hits = sorted(output_dir.rglob(pat))
        if hits:
            return hits[-1]
    ckpts = sorted(output_dir.rglob("*.pth"))
    return ckpts[-1] if ckpts else None


def train_bench_run(
    *,
    source_id: str,
    model_id: str | None,
    dataset_root: Path,
    runs_dir: Path,
    run_name: str,
    epochs: int,
    imgsz: int,
    seed: int,
    batch: int = 1,
) -> dict[str, Any]:
    entry = entry_for_bench(model_id=model_id, source_id=source_id)
    if entry is None:
        return {"status": "failed", "reason": "unknown_external_source", "returncode": 1}

    available, reason = external_bench_availability(
        model_id=model_id, source_id=source_id
    )
    if not available:
        return {"status": "skipped", "reason": reason, "returncode": 0}

    spec = spec_for_train_stack(entry.train_stack)
    assert spec is not None
    repo = resolve_external_repo_path(entry.train_stack)
    assert repo is not None
    upstream_cfg = (repo / entry.config_relpath).resolve()
    if not upstream_cfg.is_file():
        return {
            "status": "failed",
            "reason": f"missing_upstream_config:{upstream_cfg}",
            "returncode": 1,
        }

    run_dir = (runs_dir / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    coco_root = run_dir / "coco_export"
    materialize_coco_detection_tree(dataset_root=dataset_root, out_root=coco_root)
    stack_out = run_dir / f"{entry.train_stack}_output"
    overlay = run_dir / "harchoc_train_overlay.yml"
    write_train_overlay_yaml(
        out_path=overlay,
        upstream_config=upstream_cfg,
        coco_root=coco_root,
        output_dir=stack_out,
        epochs=epochs,
        imgsz=imgsz,
    )

    nproc = max(1, int(os.getenv("HARCHOC_EXTERNAL_TRAIN_NPROC", "1")))
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        f"--master_port={int(os.getenv('HARCHOC_EXTERNAL_MASTER_PORT', '29500'))}",
        spec.train_script,
        "-c",
        str(overlay),
        "-t",
        str(entry.cache_path.resolve()),
        "--use-amp",
        f"--seed={int(seed)}",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    log_path = run_dir / "external_train.log"
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            cmd,
            cwd=str(repo),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        return {
            "status": "failed",
            "reason": "external_train_failed",
            "returncode": int(proc.returncode),
            "log_path": str(log_path),
            "command": cmd,
        }

    best = _find_best_checkpoint(stack_out)
    if best is None:
        return {
            "status": "failed",
            "reason": "no_checkpoint_after_train",
            "returncode": 1,
            "log_path": str(log_path),
            "output_dir": str(stack_out),
        }

    weights_link = run_dir / "best.pth"
    if not weights_link.exists():
        try:
            os.symlink(best, weights_link)
        except OSError:
            shutil.copy2(best, weights_link)

    return {
        "status": "ok",
        "returncode": 0,
        "weights": str(weights_link),
        "checkpoint": str(best),
        "log_path": str(log_path),
        "coco_export": str(coco_root),
        "train_stack": entry.train_stack,
        "source_id": entry.source_id,
        "repo_dir": str(repo),
    }
