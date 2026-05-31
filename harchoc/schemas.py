from __future__ import annotations

from typing import Any


def require_schema_version(payload: dict[str, Any], *, expected: str) -> None:
    got = payload.get("schema_version")
    if got != expected:
        raise ValueError(f"schema_version mismatch: got={got!r} expected={expected!r}")


def with_schema_version(payload: dict[str, Any], *, schema_version: str) -> dict[str, Any]:
    """
    Return a shallow copy with a stable `schema_version`.

    If `schema_version` already exists and differs, raise. This is a small guard
    so scripts don't silently mix incompatible payload shapes under one filename.
    """
    if "schema_version" in payload and payload.get("schema_version") != schema_version:
        raise ValueError(
            f"Refusing to overwrite schema_version {payload.get('schema_version')!r} with {schema_version!r}"
        )
    out = dict(payload)
    out["schema_version"] = schema_version
    return out

