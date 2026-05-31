from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_splits_dir() -> Path:
    """Tracked split lists (train/val/test.txt) at repo ``data/splits/``."""
    return repo_root() / "data" / "splits"


def resolve_splits_dir(*, dataset_root: Path, splits_dir: Path | str | None = None) -> Path:
    """
    Resolve split list directory: explicit path, ``dataset_root/<splits_dir>``,
    or repo ``data/splits/`` when lists live outside the dataset tree.
    """
    root = dataset_root.resolve()
    if splits_dir is not None:
        sd = Path(splits_dir)
        if sd.is_absolute():
            return sd
        under_ds = (root / sd).resolve()
        if under_ds.is_dir():
            return under_ds
        under_repo = (repo_root() / sd).resolve()
        if under_repo.is_dir():
            return under_repo
        return under_ds
    repo_sd = repo_splits_dir()
    if repo_sd.is_dir():
        return repo_sd
    return (root / "data" / "splits").resolve()


def iter_split_list_lines(lines: Iterable[str]) -> Iterator[str]:
    """Yield stripped, non-empty split entries; skip blanks and ``#`` comments."""
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        yield s


def read_split_list(
    path: Path,
    *,
    missing_ok: bool = False,
    as_paths: bool = False,
    errors: str = "strict",
) -> list[str] | list[Path]:
    """
    Read a split list file (one image path per line).

    When ``missing_ok`` is true and ``path`` does not exist, returns an empty list.
    Otherwise raises ``FileNotFoundError``.
    """
    if not path.exists():
        if missing_ok:
            return []
        raise FileNotFoundError(path)
    text = path.read_text("utf-8", errors=errors)
    entries = list(iter_split_list_lines(text.splitlines()))
    if as_paths:
        return [Path(s) for s in entries]
    return entries


def resolve_split_entry(entry: str | Path, *, dataset_root: Path) -> Path:
    """Resolve a split entry relative to ``dataset_root`` when not absolute."""
    p = Path(entry)
    if not p.is_absolute():
        p = dataset_root / p
    return p.resolve()


def abs_paths_from_split_file(
    *,
    split_source: Path,
    dataset_root: Path,
    errors: str = "ignore",
) -> list[str]:
    """Return resolved absolute path strings for entries in a split list file."""
    entries = read_split_list(split_source, errors=errors)
    assert isinstance(entries, list) and (not entries or isinstance(entries[0], str))
    return [str(resolve_split_entry(rel, dataset_root=dataset_root)) for rel in entries]


def materialize_abs_split_list(
    *,
    split_source: Path,
    dataset_root: Path,
    out_path: Path,
    errors: str = "ignore",
) -> Path:
    """
    Write a split list with absolute image paths (one per line).

    Matches the materialization used by ``train.py`` / ``eval.py`` for Ultralytics.
    """
    abs_lines = abs_paths_from_split_file(
        split_source=split_source, dataset_root=dataset_root, errors=errors
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(abs_lines) + ("\n" if abs_lines else ""), "utf-8")
    return out_path
