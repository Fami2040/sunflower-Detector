from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

_DATASET_MANIFEST = REPO_ROOT / "data" / "manifest.json"
_WEIGHTS_MANIFEST = REPO_ROOT / "data" / "weights" / "weights_manifest.json"
_BOOTSTRAP_CHECKLIST = REPO_ROOT / "configs" / "bootstrap_checklist.json"

_SG_PIN_EPILOG = """
SuperGradients (--with-super-gradients):
  Installs YOLO-NAS backend with pinned deps to avoid source builds:
    numpy>=1.24.2,<2  (onnxruntime breaks on numpy>=2)
    opencv-python-headless==4.11.0.86
    setuptools==81.0.0  (pkg_resources for SG)
    protobuf==3.20.3, onnx==1.15.0, onnxruntime==1.15.0
    super-gradients==3.7.1 (--no-deps)
  Re-run bootstrap with --with-super-gradients after upgrading requirements.txt
  if pip check reports numpy conflicts with super-gradients.
  Epilog re-pins numpy<2 and opencv-python-headless after data-gradients install.
"""


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check)


def _compute_capability() -> str | None:
    """
    Returns compute capability like "12.0" if available.
    """
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
        if p.returncode != 0:
            return None
        line = (p.stdout or "").strip().splitlines()
        return line[0].strip() if line else None
    except FileNotFoundError:
        return None


def _pip_install(env: str, args: list[str]) -> None:
    base = ["mamba", "run", "-n", env, "python", "-m", "pip", "install"]
    # Network robustness: reduce cache use and increase retries/timeouts.
    extra = [
        "--no-cache-dir",
        "--retries",
        os.getenv("PIP_RETRIES", "20"),
        "--timeout",
        os.getenv("PIP_TIMEOUT", "120"),
    ]
    _run(base + extra + args)


def _pip_uninstall(env: str, pkgs: list[str]) -> None:
    base = ["mamba", "run", "-n", env, "python", "-m", "pip", "uninstall", "-y"]
    _run(base + pkgs, check=False)


def _install_super_gradients(env: str) -> None:
    """
    Install the YOLO-NAS backend (SuperGradients) in a way that avoids
    source builds for onnx/pycocotools and works with numpy<2.
    """
    # onnxruntime wheels currently break with numpy>=2.
    _pip_install(env, ["numpy>=1.24.2,<2"])

    # Some deps (data-gradients) may pull in opencv-python; ensure we end up with headless.
    _pip_uninstall(env, ["opencv-python"])
    _pip_install(env, ["opencv-python-headless==4.11.0.86"])

    # SuperGradients 3.7.1 expects pkg_resources; keep a setuptools version that still ships it.
    _pip_install(env, ["setuptools==81.0.0"])

    # Keep onnx/onnxruntime aligned with SG pins; install wheels (no source builds).
    _pip_install(env, ["protobuf==3.20.3", "onnx==1.15.0", "onnxruntime==1.15.0"])

    # Remaining runtime deps SG expects (keep pins where SG pins).
    _pip_install(
        env,
        [
            "psutil>=5.8.0",
            "boto3>=1.17.15",
            "jsonschema>=3.2.0",
            "Deprecated>=1.2.11",
            "torchmetrics==0.8",
            "hydra-core>=1.2.0",
            "omegaconf",
            "einops==0.3.2",
            "rapidfuzz",
            "json-tricks==3.16.1",
            "onnxsim>=0.4.3,<1.0",
            "albumentations~=1.3",
            "data-gradients~=0.3.1",
            "pip-tools>=6.12.1",
            "sphinx~=4.0.2",
            "sphinx-rtd-theme",
            "stringcase>=1.2.0",
            "termcolor==1.1.0",
            "treelib==1.6.1",
        ],
    )

    # Install SG itself without re-resolving deps (which may try building onnx/pycocotools).
    _pip_install(env, ["--no-deps", "super-gradients==3.7.1"])

    # data-gradients (above) pulls opencv-python and can upgrade numpy to 2.x; re-pin last.
    _pip_uninstall(env, ["opencv-python"])
    _pip_install(env, ["numpy>=1.24.2,<2"])
    _pip_install(env, ["--no-deps", "opencv-python-headless==4.11.0.86"])


def _resolve_repo_relative(path: str | os.PathLike) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


def verify_dataset_manifest() -> list[str]:
    """Return human-readable issues for data/manifest.json extracted_paths."""
    issues: list[str] = []
    if not _DATASET_MANIFEST.is_file():
        issues.append(f"missing dataset manifest: {_DATASET_MANIFEST}")
        return issues
    obj = json.loads(_DATASET_MANIFEST.read_text(encoding="utf-8"))
    for ds in obj.get("datasets", []):
        name = str(ds.get("name") or "?")
        extracted = ds.get("extracted_paths") or []
        if not extracted:
            issues.append(f"dataset {name}: extracted_paths empty in {_DATASET_MANIFEST}")
            continue
        for raw in extracted:
            p = _resolve_repo_relative(str(raw))
            if not p.is_dir():
                issues.append(f"dataset {name}: extracted path missing: {p}")
    return issues


def verify_weights_manifest() -> list[str]:
    """Return human-readable issues for data/weights/weights_manifest.json cache files."""
    from harchoc.model_zoo import verify_weights_manifest as _verify

    return _verify(_WEIGHTS_MANIFEST)


def _load_bootstrap_checklist() -> dict:
    if not _BOOTSTRAP_CHECKLIST.is_file():
        raise FileNotFoundError(f"missing bootstrap checklist: {_BOOTSTRAP_CHECKLIST}")
    return json.loads(_BOOTSTRAP_CHECKLIST.read_text(encoding="utf-8"))


def print_post_install_checklist(*, env: str, with_super_gradients: bool) -> None:
    data = _load_bootstrap_checklist()
    print(f"\n{data.get('heading', 'Post-install checklist:')}")
    for cmd in data.get("commands", []):
        print(f"  {str(cmd).format(env=env)}")
    if with_super_gradients:
        for cmd in (data.get("when") or {}).get("with_super_gradients", []):
            print(f"  {str(cmd).format(env=env)}")
    footer = data.get("footer")
    if footer:
        print(f"  {footer}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="GPU-aware, robust environment bootstrap.",
        epilog=_SG_PIN_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--env", default="harchoc", help="Mamba env name to use/create.")
    p.add_argument(
        "--create",
        action="store_true",
        help="Create the env (python+pip only) before pip installing deps.",
    )
    p.add_argument(
        "--with-super-gradients",
        action="store_true",
        help=(
            "Install YOLO-NAS backend (super-gradients) with pinned numpy<2, onnx, "
            "onnxruntime, and headless OpenCV (see epilog)."
        ),
    )
    p.add_argument(
        "--verify-manifests",
        action="store_true",
        help=(
            "After install, verify data/manifest.json extracted_paths and "
            "data/weights/weights_manifest.json cache files exist."
        ),
    )
    args = p.parse_args(argv)

    cc = _compute_capability()
    print(f"compute_capability: {cc}")

    if args.create:
        # Create a minimal env first to avoid mamba's pip step failing mid-transaction.
        _run(["mamba", "create", "-n", args.env, "-y", "python=3.11", "pip"])

    # Install GPU torch if an NVIDIA GPU is present.
    # - sm_12x: use the project constraints for cu130
    # - otherwise: default to cu124 wheels (works for V100/cc 7.0 here)
    if cc:
        if cc.startswith("12."):
            constraints = REPO_ROOT / "constraints" / "torch-cu130.txt"
            _pip_install(
                args.env,
                [
                    "--extra-index-url",
                    "https://download.pytorch.org/whl/cu130",
                    "-c",
                    str(constraints),
                    "torch",
                    "torchvision",
                ],
            )
        else:
            _pip_install(
                args.env,
                [
                    "--index-url",
                    "https://download.pytorch.org/whl/cu124",
                    "torch",
                    "torchvision",
                ],
            )

    # Then install the rest of the repo requirements.
    req = REPO_ROOT / "requirements.txt"
    req_text = req.read_text(encoding="utf-8")
    req_lines = [
        line
        for line in req_text.splitlines()
        if line.strip() and not line.strip().startswith("#") and not line.strip().startswith("super-gradients")
    ]
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
        f.write("\n".join(req_lines) + "\n")
        req_path = f.name
    _pip_install(args.env, ["-r", req_path])

    if args.with_super_gradients:
        _install_super_gradients(args.env)

    print("Done. Sanity check:")
    _run(["mamba", "run", "-n", args.env, "python", str(REPO_ROOT / "scripts" / "check_gpu.py")])

    manifest_issues: list[str] = []
    if args.verify_manifests:
        manifest_issues.extend(verify_dataset_manifest())
        manifest_issues.extend(verify_weights_manifest())
        if manifest_issues:
            print("\nManifest verification issues:")
            for msg in manifest_issues:
                print(f"  - {msg}")
        else:
            print("\nManifest verification: OK (dataset + weights paths exist).")

    print_post_install_checklist(env=args.env, with_super_gradients=bool(args.with_super_gradients))

    from harchoc.env_health import env_health_report

    health = env_health_report(env=args.env, with_super_gradients=bool(args.with_super_gradients))
    print(f"\nEnv health: {health['status']}")
    pip = health.get("pip_check") if isinstance(health.get("pip_check"), dict) else {}
    if pip.get("ignored_sg_numpy"):
        print("  pip check: SG-related numpy warnings suppressed (expected after --with-super-gradients)")
    if pip.get("issues"):
        print("  pip check issues:")
        for ln in pip["issues"]:
            print(f"    - {ln}")
        print(f"  {health.get('remediation')}")
    sg = health.get("super_gradients")
    if isinstance(sg, dict) and sg.get("ok"):
        print(f"  super_gradients: OK ({sg.get('version')})")
    elif isinstance(sg, dict) and not sg.get("ok"):
        print(f"  super_gradients import failed: {sg.get('import_error')}")

    if manifest_issues:
        return 1
    if health.get("status") != "ok":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

