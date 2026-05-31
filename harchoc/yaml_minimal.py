from __future__ import annotations

from pathlib import Path


def _parse_scalar(raw: str) -> str | int | float | bool | None:
    s = raw.strip()
    if not s:
        return ""
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.lower() in ("null", "none"):
        return None
    if s and ((s[0] == s[-1]) and s[0] in ("'", '"')):
        s = s[1:-1]
    try:
        if "." in s:
            return float(s)
        return int(s)
    except (ValueError, TypeError):
        return s


def _is_mapping_value(v: object) -> bool:
    return isinstance(v, dict)


def parse_minimal_yaml(path: Path) -> dict[str, object]:
    """
    Minimal YAML subset parser (no PyYAML dependency).

    Supported:
    - top-level "key: value"
    - top-level "key:" starting a one-level nested mapping
    - one-level nested mappings (2-space or tab indented)
    - block scalars: "key: >" or "key: |" with indented lines
    """
    lines = path.read_text("utf-8").splitlines()
    out: dict[str, object] = {}
    current_map_key: str | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        if line.startswith(("  ", "\t")):
            if current_map_key is None:
                i += 1
                continue
            inner = line.lstrip()
            if ":" not in inner:
                i += 1
                continue
            k, rest = inner.split(":", 1)
            k = k.strip()
            rest = rest.strip()
            m = out.get(current_map_key)
            if not _is_mapping_value(m):
                m = {}
                out[current_map_key] = m
            assert isinstance(m, dict)
            m[k] = _parse_scalar(rest) if rest else ""
            i += 1
            continue

        if ":" not in line:
            i += 1
            continue

        key, rest = line.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        current_map_key = None

        if rest in (">", "|"):
            i += 1
            block: list[str] = []
            while i < len(lines):
                nxt = lines[i]
                if nxt.startswith(("  ", "\t")):
                    block.append(nxt.lstrip())
                    i += 1
                elif not nxt.strip():
                    block.append("")
                    i += 1
                else:
                    break
            out[key] = (
                " ".join([b for b in block if b != ""]).strip()
                if rest == ">"
                else "\n".join(block).rstrip()
            )
            continue

        if rest == "":
            out[key] = {}
            current_map_key = key
            i += 1
            continue

        out[key] = _parse_scalar(rest)
        i += 1
    return out


def safe_load(path: str | Path) -> dict[str, object]:
    """Load a YAML subset file into a top-level mapping (no PyYAML)."""
    return parse_minimal_yaml(Path(path))


def parse_minimal_yaml_flat(path: Path) -> dict[str, str]:
    """
    Flat top-level key loader for Ultralytics-style data.yaml.

    Ignores nested mappings and list lines; values remain strings.
    """
    out: dict[str, str] = {}
    for raw in path.read_text("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(("-", "[")):
            continue
        if ":" not in line:
            continue
        k, rest = line.split(":", 1)
        k = k.strip()
        v = rest.strip().strip("'\"")
        if v:
            out[k] = v
    return out


def parse_names_and_nc(path: Path) -> tuple[dict[int, str] | None, int | None]:
    """
    Load YOLO-style data.yaml fields:
    - nc: <int>
    - names: (mapping) {0: class0, 1: class1, ...}
    """
    nc: int | None = None
    names: dict[int, str] | None = None
    current_map_key: str | None = None
    for raw in path.read_text("utf-8").splitlines():
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith(("  ", "\t")):
            if current_map_key != "names":
                continue
            inner = line.lstrip()
            if ":" not in inner:
                continue
            k, v = inner.split(":", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            try:
                ki = int(k)
            except ValueError:
                continue
            if names is None:
                names = {}
            names[ki] = v
            continue

        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        current_map_key = None
        if key == "nc":
            try:
                nc = int(rest)
            except ValueError:
                nc = None
        elif key == "names" and rest == "":
            current_map_key = "names"
            if names is None:
                names = {}
    return names, nc
