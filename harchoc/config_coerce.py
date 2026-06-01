"""Narrow JSON/YAML ``object`` values for static analysis and runtime."""

from __future__ import annotations

from typing import Any


def as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def child_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    """Return ``parent[key]`` when it is a dict, else ``{}``."""
    value = parent.get(key)
    return value if isinstance(value, dict) else {}


def coerce_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def coerce_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def as_str_list(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    return []


def as_float_list(value: object) -> list[float]:
    out: list[float] = []
    if isinstance(value, (list, tuple)):
        for item in value:
            parsed = coerce_float(item)
            if parsed is not None:
                out.append(parsed)
    return out


def as_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def pick_int(value: object, *, default: int) -> int:
    parsed = coerce_int(value)
    return parsed if parsed is not None else default


def pick_float(value: object, *, default: float) -> float:
    parsed = coerce_float(value)
    return parsed if parsed is not None else default


def pick_optional_int(value: object) -> int | None:
    return coerce_int(value)


def pick_optional_float(value: object) -> float | None:
    return coerce_float(value)


def split_lists_from_source(payload: dict[str, Any]) -> dict[str, list[str] | None]:
    src = as_dict(payload.get("source"))
    raw = as_dict(src.get("split_lists"))
    out: dict[str, list[str] | None] = {}
    for name in ("train", "val", "test"):
        entry = raw.get(name)
        if isinstance(entry, list):
            out[name] = [str(x) for x in entry]
        else:
            out[name] = None
    return out
