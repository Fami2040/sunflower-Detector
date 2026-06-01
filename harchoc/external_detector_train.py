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
    for mod in _external_python_modules(entry.train_stack):
        try:
            __import__(mod)
        except ImportError:
            return False, f"missing_dependency:{mod}"
    return True, None


# Pip packages for bootstrap --with-external-detr (aligned with external/D-FINE/requirements.txt).
EXTERNAL_DETR_PIP_PACKAGES: tuple[str, ...] = (
    "faster-coco-eval>=1.6.6",
    "calflops",
    "transformers",
    "loguru",
    "gdown",  # DEIM Google Drive checkpoints (check_weights_cache --download)
)


def external_detr_python_modules() -> tuple[str, ...]:
    """Runtime imports required across external DETR train stacks (beyond torch)."""
    return ("faster_coco_eval", "calflops", "transformers", "loguru")


def _rtdetrv2_collate_scales(imgsz: int) -> tuple[int, ...]:
    """Scale RT-DETRv2 multiscale list from the default 640 anchor to ``imgsz``."""
    ref = (480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800)
    factor = int(imgsz) / 640.0
    out: list[int] = []
    for s in ref:
        v = int(round(s * factor / 32) * 32)
        out.append(max(32, v))
    return tuple(out)


def _external_python_modules(train_stack: str) -> tuple[str, ...]:
    """Runtime imports required before external train (beyond torch)."""
    common = ("faster_coco_eval",)
    if train_stack == "deim":
        return common + ("calflops", "transformers")
    if train_stack in ("dfine", "rtdetrv2_pytorch"):
        return common + ("loguru",)
    return common


def _scale_epoch_milestone(value: int, *, target_epochs: int, ref_epochs: int) -> int:
    """Scale a milestone epoch from upstream defaults to the matrix ``target_epochs``."""
    if ref_epochs <= 0:
        return max(1, min(target_epochs - 1, int(value)))
    scaled = int(round(value * target_epochs / ref_epochs))
    return max(1, min(target_epochs - 1, scaled))


def _epoch_field(train_stack: str) -> str:
    """D-FINE uses ``epochs``; DEIM / RT-DETRv2 use ``epoches`` (RT-DETR spelling)."""
    return "epochs" if train_stack == "dfine" else "epoches"


def _overlay_schedule(
    *,
    train_stack: str,
    upstream_config: Path,
    epochs: int,
) -> dict[str, Any]:
    """Return stack-specific schedule overrides for the train overlay YAML."""
    cfg_name = upstream_config.as_posix()
    if train_stack == "dfine":
        ref_epochs, stop = 80, 72
        return {
            "epoch_field": "epochs",
            "ref_epochs": ref_epochs,
            "stop_epoch": _scale_epoch_milestone(stop, target_epochs=epochs, ref_epochs=ref_epochs),
            "policy_epoch": _scale_epoch_milestone(stop, target_epochs=epochs, ref_epochs=ref_epochs),
        }
    if train_stack == "rtdetrv2_pytorch":
        ref_epochs, stop = 72, 71
        return {
            "epoch_field": "epoches",
            "ref_epochs": ref_epochs,
            "stop_epoch": _scale_epoch_milestone(stop, target_epochs=epochs, ref_epochs=ref_epochs),
            "policy_epoch": _scale_epoch_milestone(stop, target_epochs=epochs, ref_epochs=ref_epochs),
        }
    # DEIM — recipe differs for D-FINE vs RT-DETRv2 upstream configs.
    if "deim_rtdetrv2" in cfg_name or "deim_r50vd" in cfg_name:
        ref_epochs = 60
        stop, flat, no_aug = 58, 34, 2
        mixup = (4, 34)
        policy = (4, 34, 58)
    else:
        ref_epochs = 58
        stop, flat, no_aug = 50, 29, 8
        mixup = (4, 29)
        policy = (4, 29, 50)
    return {
        "epoch_field": "epoches",
        "ref_epochs": ref_epochs,
        "stop_epoch": _scale_epoch_milestone(stop, target_epochs=epochs, ref_epochs=ref_epochs),
        "flat_epoch": _scale_epoch_milestone(flat, target_epochs=epochs, ref_epochs=ref_epochs),
        "no_aug_epoch": max(1, _scale_epoch_milestone(no_aug, target_epochs=epochs, ref_epochs=ref_epochs)),
        "mixup_epochs": [
            _scale_epoch_milestone(mixup[0], target_epochs=epochs, ref_epochs=ref_epochs),
            _scale_epoch_milestone(mixup[1], target_epochs=epochs, ref_epochs=ref_epochs),
        ],
        "policy_epochs": [
            _scale_epoch_milestone(p, target_epochs=epochs, ref_epochs=ref_epochs) for p in policy
        ],
    }


# DEIM dense-augmentation configs (rt_deim.yml) use a 3-stage policy + Mosaic. On
# torchvision 0.21 that breaks ConvertPILImage when Compose passes (image, target,
# dataset). Use D-FINE-style transforms (no Mosaic, single policy epoch int).
_DEIM_SAFE_POLICY_OPS = (
    "RandomPhotometricDistort",
    "RandomZoomOut",
    "RandomIoUCrop",
)


def _deim_train_transform_lines(*, imgsz: int, sched: dict[str, Any]) -> list[str]:
    sz = int(imgsz)
    stop = int(sched["stop_epoch"])
    return [
        "    transforms:",
        "      ops:",
        "        - {type: RandomPhotometricDistort, p: 0.5}",
        "        - {type: RandomZoomOut, fill: 0}",
        "        - {type: RandomIoUCrop, p: 0.8}",
        "        - {type: SanitizeBoundingBoxes, min_size: 1}",
        "        - {type: RandomHorizontalFlip}",
        f"        - {{type: Resize, size: [{sz}, {sz}]}}",
        "        - {type: SanitizeBoundingBoxes, min_size: 1}",
        "        - {type: ConvertPILImage, dtype: 'float32', scale: True}",
        "        - {type: ConvertBoxes, fmt: 'cxcywh', normalize: True}",
        "      policy:",
        "        name: stop_epoch",
        f"        epoch: {stop}",
        f"        ops: [{', '.join(repr(x) for x in _DEIM_SAFE_POLICY_OPS)}]",
        "      mosaic_prob: 0",
    ]


def _yaml_config_for_stack(train_stack: str, repo: Path):
    """Return YAMLConfig class after prepending ``repo`` to sys.path."""
    import sys

    repo_str = str(repo.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    if train_stack == "deim":
        from engine.core.yaml_config import YAMLConfig

        return YAMLConfig
    if train_stack == "dfine":
        from src.core import YAMLConfig

        return YAMLConfig
    if train_stack == "rtdetrv2_pytorch":
        from src.core import YAMLConfig

        return YAMLConfig
    raise ValueError(f"unknown_train_stack:{train_stack}")


def _apply_stack_runtime_compat(train_stack: str, repo: Path) -> None:
    """Vendor-tree-safe hooks (monkeypatch) before importing an external stack."""
    if train_stack != "deim":
        return
    repo_root = Path(__file__).resolve().parents[1]
    root_s = str(repo_root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    repo_s = str(repo.resolve())
    if repo_s not in sys.path:
        sys.path.insert(0, repo_s)
    from harchoc.deim_tv_compat import apply_deim_torchvision_compat

    apply_deim_torchvision_compat()


def _verify_external_train_smoke_inprocess(
    *,
    overlay_path: Path,
    repo: Path,
    train_stack: str,
    epoch: int,
) -> tuple[bool, str | None]:
    _apply_stack_runtime_compat(train_stack, repo)
    yaml_config = _yaml_config_for_stack(train_stack, repo)
    cfg = yaml_config(str(overlay_path.resolve()))
    ds = cfg.train_dataloader.dataset
    ds.set_epoch(int(epoch))  # type: ignore[attr-defined]
    img, target = ds.load_item(0)  # type: ignore[attr-defined]
    out = ds._transforms(img, target, ds)  # type: ignore[attr-defined]
    if isinstance(out, tuple):
        if len(out) < 2:
            return False, f"{train_stack}_transform_short_tuple"
        out_img, out_target = out[0], out[1]
    else:
        out_img, out_target = out, None
    if out_target is None or out_img is None:
        return False, f"{train_stack}_transform_empty_output"
    return True, None


def verify_external_train_smoke(
    *,
    overlay_path: Path,
    repo: Path,
    train_stack: str,
    epoch: int = 0,
) -> tuple[bool, str | None]:
    """Build train dataloader and run one transform sample (fail-fast before subprocess).

    Runs in a child process with only ``repo`` on ``PYTHONPATH`` so DEIM / D-FINE /
    RT-DETRv2 ``@register()`` collate classes do not collide in one interpreter.
    """
    if not repo.is_dir():
        return False, f"missing_repo:{repo}"
    if os.getenv("HARCHOC_EXTERNAL_SMOKE_INPROCESS", "").strip() in ("1", "true", "yes"):
        try:
            return _verify_external_train_smoke_inprocess(
                overlay_path=overlay_path,
                repo=repo,
                train_stack=train_stack,
                epoch=epoch,
            )
        except ImportError as ex:
            return False, f"{train_stack}_import:{ex}"
        except ValueError as ex:
            return False, str(ex)
        except Exception as ex:
            return False, f"{train_stack}_dataloader:{type(ex).__name__}:{ex}"

    overlay_s = str(overlay_path.resolve())
    repo_s = str(repo.resolve())
    code = (
        "import sys\n"
        f"sys.path.insert(0, {repo_s!r})\n"
        "from pathlib import Path\n"
        "from harchoc.external_detector_train import _verify_external_train_smoke_inprocess\n"
        "ok, reason = _verify_external_train_smoke_inprocess(\n"
        f"    overlay_path=Path({overlay_s!r}),\n"
        f"    repo=Path({repo_s!r}),\n"
        f"    train_stack={train_stack!r},\n"
        f"    epoch={int(epoch)},\n"
        ")\n"
        "raise SystemExit(0 if ok else 1)\n"
    )
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(repo_root)
    env["HARCHOC_ALLOW_BASE_PYTHON"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return True, None
    detail = (proc.stderr or proc.stdout or "").strip().splitlines()
    tail = detail[-1] if detail else f"exit_{proc.returncode}"
    return False, f"{train_stack}_dataloader_smoke:{tail}"


def verify_deim_train_dataloader(
    *,
    overlay_path: Path,
    repo: Path,
    epoch: int = 0,
) -> tuple[bool, str | None]:
    """DEIM-only alias for ``verify_external_train_smoke``."""
    return verify_external_train_smoke(
        overlay_path=overlay_path,
        repo=repo,
        train_stack="deim",
        epoch=epoch,
    )


def build_train_cli_updates(
    *,
    train_stack: str,
    upstream_config: Path,
    epochs: int,
    imgsz: int,
) -> list[str]:
    """CLI ``-u`` overrides (merged after YAML) for epoch + spatial size."""
    sched = _overlay_schedule(
        train_stack=train_stack, upstream_config=upstream_config, epochs=epochs
    )
    field = sched["epoch_field"]
    updates = [
        f"{field}={int(epochs)}",
        f"eval_spatial_size=[{int(imgsz)}, {int(imgsz)}]",
    ]
    if train_stack == "rtdetrv2_pytorch":
        scales = ", ".join(str(s) for s in _rtdetrv2_collate_scales(int(imgsz)))
        updates.append(f"train_dataloader.collate_fn.scales=[{scales}]")
        updates.append(
            f"train_dataloader.collate_fn.stop_epoch={sched['stop_epoch']}"
        )
    elif train_stack == "deim":
        updates.append(f"train_dataloader.collate_fn.base_size={int(imgsz)}")
        updates.append(
            f"train_dataloader.collate_fn.stop_epoch={sched['stop_epoch']}"
        )
        mix0, mix1 = sched["mixup_epochs"]
        updates.append(f"train_dataloader.collate_fn.mixup_epochs=[{mix0}, {mix1}]")
        updates.append("train_dataloader.dataset.transforms.mosaic_prob=0")
    elif train_stack == "dfine":
        updates.append(f"train_dataloader.collate_fn.base_size={int(imgsz)}")
        updates.append(
            f"train_dataloader.collate_fn.stop_epoch={sched['stop_epoch']}"
        )
    return updates


def write_train_overlay_yaml(
    *,
    out_path: Path,
    upstream_config: Path,
    coco_root: Path,
    output_dir: Path,
    epochs: int,
    imgsz: int,
    train_stack: str,
    num_classes: int = 2,
    batch: int = 1,
) -> None:
    train_img = (coco_root / "images" / "train").resolve()
    val_img = (coco_root / "images" / "val").resolve()
    train_ann = (coco_root / "annotations" / "instances_train.json").resolve()
    val_ann = (coco_root / "annotations" / "instances_val.json").resolve()
    sz = int(imgsz)
    ep = int(epochs)
    sched = _overlay_schedule(
        train_stack=train_stack, upstream_config=upstream_config, epochs=ep
    )
    epoch_field = sched["epoch_field"]
    lines = [
        f"__include__: ['{upstream_config.as_posix()}']",
        f"output_dir: {output_dir.as_posix()}",
        f"{epoch_field}: {ep}",
        "remap_mscoco_category: false",
        f"num_classes: {int(num_classes)}",
        f"eval_spatial_size: [{sz}, {sz}]",
    ]
    if train_stack == "deim":
        lines.extend(
            [
                f"flat_epoch: {sched['flat_epoch']}",
                f"no_aug_epoch: {sched['no_aug_epoch']}",
            ]
        )
    lines.extend(
        [
            "train_dataloader:",
            "  dataset:",
            f"    img_folder: {train_img.as_posix()}",
            f"    ann_file: {train_ann.as_posix()}",
            "    return_masks: false",
        ]
    )
    if train_stack == "deim":
        mix0, mix1 = sched["mixup_epochs"]
        lines.extend(_deim_train_transform_lines(imgsz=sz, sched=sched))
        lines.extend(
            [
                "  collate_fn:",
                f"    base_size: {sz}",
                f"    stop_epoch: {sched['stop_epoch']}",
                f"    mixup_epochs: [{mix0}, {mix1}]",
                "  num_workers: 2",
            ]
        )
    elif train_stack == "dfine":
        lines.extend(
            [
                "    transforms:",
                "      policy:",
                f"        epoch: {sched['policy_epoch']}",
                "  collate_fn:",
                f"    base_size: {sz}",
                f"    stop_epoch: {sched['stop_epoch']}",
            ]
        )
    elif train_stack == "rtdetrv2_pytorch":
        scales = ", ".join(str(s) for s in _rtdetrv2_collate_scales(sz))
        lines.extend(
            [
                "    transforms:",
                "      policy:",
                f"        epoch: {sched['policy_epoch']}",
                "  collate_fn:",
                f"    scales: [{scales}]",
                f"    stop_epoch: {sched['stop_epoch']}",
            ]
        )
    lines.extend(
        [
            f"  total_batch_size: {int(batch)}",
            "val_dataloader:",
            "  dataset:",
            f"    img_folder: {val_img.as_posix()}",
            f"    ann_file: {val_ann.as_posix()}",
            "    return_masks: false",
            "    transforms:",
            "      ops:",
            f"        - {{type: Resize, size: [{sz}, {sz}]}}",
            "        - {type: ConvertPILImage, dtype: 'float32', scale: True}",
            f"  total_batch_size: {int(batch)}",
            "",
        ]
    )
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
        train_stack=entry.train_stack,
        batch=batch,
    )

    cli_updates = build_train_cli_updates(
        train_stack=entry.train_stack,
        upstream_config=upstream_cfg,
        epochs=int(epochs),
        imgsz=int(imgsz),
    )
    smoke_ok, smoke_reason = verify_external_train_smoke(
        overlay_path=overlay,
        repo=repo,
        train_stack=entry.train_stack,
        epoch=0,
    )
    if not smoke_ok:
        return {
            "status": "failed",
            "reason": smoke_reason or "external_train_smoke_failed",
            "returncode": 1,
            "overlay": str(overlay),
        }

    repo_root = Path(__file__).resolve().parents[1]
    nproc = max(1, int(os.getenv("HARCHOC_EXTERNAL_TRAIN_NPROC", "1")))
    train_entry = spec.train_script
    train_entry_args: list[str] = []
    if entry.train_stack == "deim":
        train_entry = str((repo_root / "harchoc" / "run_external_train.py").resolve())
        train_entry_args = [entry.train_stack, str(repo), spec.train_script]
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        f"--master_port={int(os.getenv('HARCHOC_EXTERNAL_MASTER_PORT', '29500'))}",
        train_entry,
        *train_entry_args,
        "-c",
        str(overlay),
        "-t",
        str(entry.cache_path.resolve()),
        "--use-amp",
        f"--seed={int(seed)}",
        "-u",
        *cli_updates,
    ]
    env = os.environ.copy()
    py_path = str(repo_root) + os.pathsep + str(repo)
    if env.get("PYTHONPATH"):
        py_path += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = py_path

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
