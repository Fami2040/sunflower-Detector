from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable

_AUG_SUFFIXES: tuple[str, ...] = (
    "_aug",
    "_flip",
    "_rot",
    "_mirror",
    "_hflip",
    "_vflip",
    "_copy",
    "_duplicate",
    "_bright",
    "_dark",
    "_blur",
    "_noise",
    "_crop",
)

_TRAILING_INDEX_RE = re.compile(r"_\d+$")


def normalize_stem(stem: str) -> str:
    """Strip common augmentation suffixes and trailing ``_123`` indices from a stem."""
    s = stem.lower()
    changed = True
    while changed:
        changed = False
        for suf in _AUG_SUFFIXES:
            if s.endswith(suf):
                s = s[: -len(suf)]
                changed = True
        stripped = _TRAILING_INDEX_RE.sub("", s)
        if stripped != s:
            s = stripped
            changed = True
    return s


def parse_group_key_spec(spec: str) -> tuple[str, str | int | None]:
    """
    Parse ``--group-key`` values.

    Supported forms: ``stem``, ``parent``, ``prefix:N``, ``csv:PATH`` (or a ``.csv`` path).
    """
    s = (spec or "").strip()
    if not s:
        raise ValueError("group-key spec must be non-empty")
    if s == "stem":
        return ("stem", None)
    if s == "parent":
        return ("parent", None)
    if s.startswith("prefix:"):
        n_s = s.split(":", 1)[1].strip()
        if not n_s.isdigit() or int(n_s) < 1:
            raise ValueError(f"invalid prefix group-key: {spec!r}")
        return ("prefix", int(n_s))
    if s.startswith("csv:"):
        return ("csv", s[4:].strip())
    if s.endswith(".csv"):
        return ("csv", s)
    raise ValueError(f"unsupported group-key spec: {spec!r}")


def load_group_csv(csv_path: Path, *, dataset_root: Path | None = None) -> dict[str, str]:
    """
    Load ``image_key -> group_id`` from a CSV.

    Uses the first two columns (optional header ``image``/``path`` + ``group``).
    Keys are stored for rel path, basename, and stem lookups.
    """
    path = csv_path if csv_path.is_absolute() else (dataset_root or Path.cwd()) / csv_path
    if not path.is_file():
        raise FileNotFoundError(path)

    index: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        rows = [r for r in reader if r and any(c.strip() for c in r)]
    if not rows:
        return index

    start = 0
    h0 = rows[0][0].strip().lower() if rows[0] else ""
    if h0 in {"image", "path", "file", "filename", "rel_path", "rel"}:
        start = 1

    for row in rows[start:]:
        if len(row) < 2:
            continue
        key = row[0].strip()
        group = row[1].strip()
        if not key or not group:
            continue
        index[key] = group
        index[Path(key).name] = group
        index[Path(key).stem] = group
    return index


def resolve_group_id(rel_path: str | Path, *, spec: str, csv_index: dict[str, str] | None = None) -> str:
    """Map a dataset-relative image path to a group id."""
    kind, arg = parse_group_key_spec(spec)
    p = Path(rel_path)
    if kind == "stem":
        return p.stem
    if kind == "parent":
        parent = p.parent.name
        return parent if parent else p.stem
    if kind == "prefix":
        assert isinstance(arg, int)
        return p.stem[: arg] if len(p.stem) >= arg else p.stem
    if kind == "csv":
        assert isinstance(arg, str)
        if csv_index is None:
            raise ValueError("csv_index required for csv group-key")
        rel = p.as_posix()
        for candidate in (rel, p.name, p.stem):
            if candidate in csv_index:
                return csv_index[candidate]
        raise KeyError(f"no group for {rel_path!r} in csv index")
    raise ValueError(f"unsupported group-key kind: {kind!r}")


def assign_splits_by_group(
    items: list[str],
    *,
    group_for: Callable[[str], str],
    seed: int,
    val_frac: float,
    test_frac: float,
) -> dict[str, list[str]]:
    """Assign whole groups to train/val/test; images in one group never span splits."""
    if val_frac < 0 or test_frac < 0 or (val_frac + test_frac) >= 1.0:
        raise ValueError("require val_frac>=0, test_frac>=0, and val_frac+test_frac < 1")

    by_group: dict[str, list[str]] = defaultdict(list)
    for rel in items:
        by_group[group_for(rel)].append(rel)

    groups = sorted(by_group.keys())
    rnd = random.Random(seed)
    rnd.shuffle(groups)

    n = len(groups)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    n_train = n - n_val - n_test

    train_g = set(groups[:n_train])
    val_g = set(groups[n_train : n_train + n_val])
    test_g = set(groups[n_train + n_val :])

    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for g, rels in by_group.items():
        if g in train_g:
            out["train"].extend(sorted(rels))
        elif g in val_g:
            out["val"].extend(sorted(rels))
        elif g in test_g:
            out["test"].extend(sorted(rels))
        else:
            raise RuntimeError(f"group {g!r} unassigned")
    for k in out:
        out[k].sort()
    return out


def _stem_for_entry(rel: str | Path) -> str:
    return Path(rel).stem


def audit_split_leakage(
    splits: dict[str, list[str]],
    *,
    group_key_spec: str | None = None,
    csv_index: dict[str, str] | None = None,
    include_perceptual_hash: bool = False,
) -> dict[str, object]:
    """
    Detect cross-split stem / normalized-stem collisions and optional group leakage.

    Returns a JSON-serializable report section.
    """
    stem_to_splits: dict[str, set[str]] = defaultdict(set)
    norm_to_splits: dict[str, set[str]] = defaultdict(set)
    group_to_splits: dict[str, set[str]] = defaultdict(set)

    for split_name, entries in splits.items():
        for rel in entries:
            stem = _stem_for_entry(rel)
            stem_to_splits[stem].add(split_name)
            norm_to_splits[normalize_stem(stem)].add(split_name)
            if group_key_spec:
                try:
                    gid = resolve_group_id(rel, spec=group_key_spec, csv_index=csv_index)
                except KeyError:
                    gid = f"__missing__:{rel}"
                group_to_splits[gid].add(split_name)

    def _collisions(mapping: dict[str, set[str]]) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for key, split_set in sorted(mapping.items()):
            if len(split_set) > 1:
                out.append({"key": key, "splits": sorted(split_set)})
        return out

    stem_collisions = _collisions(stem_to_splits)
    norm_collisions = _collisions(norm_to_splits)
    group_collisions = _collisions(group_to_splits) if group_key_spec else []

    phash_section: dict[str, object]
    if include_perceptual_hash:
        phash_section = {
            "status": "stub",
            "message": "perceptual hash not implemented; install imagehash + Pillow to enable later",
        }
    else:
        phash_section = {"status": "skipped"}

    ok = not stem_collisions and not norm_collisions and not group_collisions
    return {
        "ok": ok,
        "n_images": sum(len(v) for v in splits.values()),
        "stem_collisions": stem_collisions,
        "normalized_stem_collisions": norm_collisions,
        "group_collisions": group_collisions,
        "perceptual_hash": phash_section,
    }


def splits_from_split_dir(splits_dir: Path) -> dict[str, list[str]]:
    """Load train/val/test lists from a splits directory (missing files -> empty)."""
    from harchoc.splits_io import read_split_list

    out: dict[str, list[str]] = {}
    for name in ("train", "val", "test"):
        p = splits_dir / f"{name}.txt"
        if p.exists():
            entries = read_split_list(p)
            assert isinstance(entries, list)
            out[name] = [str(e) for e in entries]
        else:
            out[name] = []
    return out
