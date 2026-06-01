from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def load_json(path: str | Path) -> Any:
    """Load JSON from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_json_dict(path: str | Path) -> dict[str, Any]:
    """Load JSON and require a top-level object."""
    data = load_json(path)
    if not isinstance(data, dict):
        raise TypeError(f"expected JSON object at {path}, got {type(data).__name__}")
    return data


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    """Write JSON to disk (creates parent directories)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=indent) + "\n", encoding="utf-8")
    return p
