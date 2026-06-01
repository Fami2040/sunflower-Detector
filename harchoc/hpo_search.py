from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from harchoc.schemas import with_schema_version
from harchoc.train_config import load_train_config_json
from harchoc.training_budget import enforce_budget

HPO_SEARCH_SCHEMA = "hpo_search.v1"

ParamKind = Literal["int", "float", "categorical"]


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: ParamKind
    low: float | None = None
    high: float | None = None
    choices: tuple[object, ...] | None = None
    log: bool = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json_obj(raw: str) -> dict[str, Any]:
    s = str(raw).strip()
    if not s:
        return {}
    p = Path(s).expanduser()
    if p.is_file():
        from harchoc.json_io import load_json_dict

        return load_json_dict(p)
    return json.loads(s)


def _parse_space(space: dict[str, Any]) -> list[ParamSpec]:
    params = space.get("params")
    if not isinstance(params, dict) or not params:
        raise SystemExit("HPO space must include a non-empty object: {\"params\": {...}}")
    out: list[ParamSpec] = []
    for name, spec_any in params.items():
        if not isinstance(name, str) or not name.strip():
            raise SystemExit("HPO param names must be non-empty strings")
        if not isinstance(spec_any, dict):
            raise SystemExit(f"HPO param {name!r} spec must be an object")
        kind = str(spec_any.get("kind") or "").strip()
        if kind not in ("int", "float", "categorical"):
            raise SystemExit(
                f"HPO param {name!r} has invalid kind={kind!r} (expected int|float|categorical)"
            )
        log = bool(spec_any.get("log", False))
        if kind == "categorical":
            ch = spec_any.get("choices")
            if not isinstance(ch, list) or len(ch) < 1:
                raise SystemExit(
                    f"HPO param {name!r} categorical requires non-empty choices list"
                )
            out.append(ParamSpec(name=name, kind="categorical", choices=tuple(ch), log=log))
            continue

        low = spec_any.get("low")
        high = spec_any.get("high")
        if low is None or high is None:
            raise SystemExit(f"HPO param {name!r} requires low/high")
        try:
            low_f = float(low)
            high_f = float(high)
        except Exception as exc:
            raise SystemExit(f"HPO param {name!r} low/high must be numeric") from exc
        if not (high_f > low_f):
            raise SystemExit(
                f"HPO param {name!r} requires high > low (got low={low_f} high={high_f})"
            )
        out.append(
            ParamSpec(
                name=name,
                kind=kind,  # type: ignore[arg-type]
                low=low_f,
                high=high_f,
                log=log,
            )
        )
    return out


def _sample_one(rng: random.Random, spec: ParamSpec) -> object:
    if spec.kind == "categorical":
        assert spec.choices is not None
        return rng.choice(list(spec.choices))
    assert spec.low is not None and spec.high is not None
    if spec.log and spec.low <= 0:
        raise SystemExit(
            f"HPO param {spec.name!r} uses log sampling but low={spec.low} <= 0"
        )
    if spec.kind == "float":
        u = rng.random()
        if spec.log:
            # sample uniformly in log space
            import math

            lo = math.log(spec.low)
            hi = math.log(spec.high)
            return float(math.exp(lo + (hi - lo) * u))
        return float(spec.low + (spec.high - spec.low) * u)
    if spec.kind == "int":
        if spec.log:
            import math

            lo = math.log(spec.low)
            hi = math.log(spec.high)
            v = math.exp(lo + (hi - lo) * rng.random())
            return int(round(v))
        return int(rng.randint(int(spec.low), int(spec.high)))
    raise AssertionError(f"Unhandled param kind: {spec.kind}")


def _apply_params(base_train: dict[str, Any], sampled: dict[str, object]) -> dict[str, Any]:
    out = dict(base_train)
    for k, v in sampled.items():
        out[str(k)] = v
    return out


def plan_hpo_trials(
    *,
    base_train_config: str,
    space: dict[str, Any],
    trials: int,
    seed: int,
) -> dict[str, Any]:
    if trials <= 0:
        raise SystemExit(f"trials must be > 0 (got {trials})")
    repo = _repo_root()
    base_cfg = load_train_config_json(Path(base_train_config), repo_root=repo)
    base_train = {str(k): v for k, v in base_cfg.items() if k not in ("dataset", "eval")}
    params = _parse_space(space)

    rng = random.Random(int(seed))
    planned: list[dict[str, Any]] = []
    for i in range(trials):
        sampled: dict[str, object] = {p.name: _sample_one(rng, p) for p in params}
        train_cfg = _apply_params(base_train, sampled)
        # Enforce budget caps even for planning (so configs never exceed hard limits).
        epochs = int(train_cfg.get("epochs") or 0)
        imgsz = int(train_cfg.get("imgsz") or 0)
        batch = int(train_cfg.get("batch") or 0)
        enforce_budget(epochs=epochs, imgsz=imgsz, batch=batch)

        planned.append(
            {
                "trial_index": i,
                "seed": int(seed),
                "params": sampled,
                "train": train_cfg,
                "status": "planned",
            }
        )

    return with_schema_version(
        {
            "created_at_unix": int(time.time()),
            "repo_root": str(repo),
            "base_train_config": str(base_train_config),
            "space": space,
            "trials": planned,
            "budget": {
                "HARCHOC_MAX_EPOCHS": int(os.getenv("HARCHOC_MAX_EPOCHS", "500") or 500),
                "HARCHOC_MAX_IMGSZ": int(os.getenv("HARCHOC_MAX_IMGSZ", "2048") or 2048),
                "HARCHOC_MAX_BATCH": int(os.getenv("HARCHOC_MAX_BATCH", "16") or 16),
            },
        },
        schema_version=HPO_SEARCH_SCHEMA,
    )

