from __future__ import annotations

import argparse
import importlib.util
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys; from pathlib import Path; _r = Path(__file__).resolve().parent.parent; (str(_r) not in sys.path) and sys.path.insert(0, str(_r)); from harchoc.script_entry import bootstrap_repo_imports; bootstrap_repo_imports()
from harchoc.model_zoo import backend_availability
from harchoc.hsp_weights import HSP_DETECTION_WEIGHTS
from scripts._common_cli import write_json


@dataclass(frozen=True)
class PipelineEnv:
    detection_model: str
    classifier_model: str
    force_device: str | None


def _env() -> PipelineEnv:
    # Keep names consistent with existing runtime entrypoints.
    det = os.getenv("DETECTION_MODEL", HSP_DETECTION_WEIGHTS).strip()
    clf = os.getenv("CLASSIFIER_MODEL", "models/classifier.pt").strip()
    force_device = os.getenv("FORCE_DEVICE", "").strip().lower() or None
    return PipelineEnv(detection_model=det, classifier_model=clf, force_device=force_device)


def _module_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _detect_backend_availability() -> dict[str, object]:
    """
    Report runtime dependencies without importing them.
    This makes `--dry-run` CI-safe.
    """
    ultralytics_ok, _ = backend_availability("ultralytics")
    needed = {
        "torch": _module_available("torch"),
        "sahi": _module_available("sahi"),
        "ultralytics": ultralytics_ok,
        "cv2": _module_available("cv2"),
    }
    missing = sorted([k for k, ok in needed.items() if not ok])
    return {"modules": needed, "missing": missing}


def build_request_record(
    *,
    request_id: str,
    image_path: str,
    out_path: Path,
    dry_run: bool,
    now: str | None = None,
) -> dict[str, Any]:
    env = _env()
    generated_at = now or datetime.now(timezone.utc).isoformat()
    availability = _detect_backend_availability()

    # This is a contract for downstream tooling (plots, audit, regression tests),
    # not a mirror of any specific runtime. Keep it stable and explicit.
    payload: dict[str, Any] = {
        "schema_version": "pipeline_request.v1",
        "script": "pipeline_request",
        "generated_at": generated_at,
        "request_id": request_id,
        "input": {"image": image_path},
        "output": {"path": str(out_path)},
        "env": {
            "detection_model": env.detection_model,
            "classifier_model": env.classifier_model,
            "force_device": env.force_device,
        },
        "availability": availability,
        "status": "dry-run" if dry_run else "planned",
        "results": {
            "is_sunflower": None,
            "counts": {"developed": None, "aborted": None, "total": None},
            "timings_s": {"classifier": None, "detection": None, "postprocess": None, "total": None},
        },
        "skip_reason": (f"missing_dependency:{','.join(availability['missing'])}" if availability["missing"] else None),
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Per-request pipeline output writer (CI-safe via --dry-run).")
    p.add_argument("--image", required=True, help="Input image path (used for metadata only in --dry-run).")
    p.add_argument(
        "--out",
        default="outputs/{request_id}.json",
        help="Output JSON path template. Supports {request_id}.",
    )
    p.add_argument(
        "--request-id",
        default="",
        help="Request identifier. If omitted, a random uuid4 is used.",
    )
    p.add_argument("--dry-run", action="store_true", help="Write JSON contract only; no ML imports.")
    args = p.parse_args(argv)

    request_id = args.request_id.strip() or str(uuid.uuid4())
    out_str = str(args.out).replace("{request_id}", request_id)
    out_path = Path(out_str)

    payload = build_request_record(
        request_id=request_id,
        image_path=str(args.image),
        out_path=out_path.resolve(),
        dry_run=bool(args.dry_run),
    )
    write_json(out_path, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

