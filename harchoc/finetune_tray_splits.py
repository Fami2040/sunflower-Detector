"""Compose train/val split lists for tray-targeted fine-tune (no canonical test leakage)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from harchoc.domain_eval import domain_split_file
from harchoc.splits_io import read_split_list, resolve_split_entry

MERGED_FAMILY_SPLIT_SUFFIX = "_family.txt"

TrainMode = Literal["canonical", "tray_adapt", "lofo_pool"]

FINETUNE_SPLIT_PLAN_SCHEMA = "finetune_split_plan.v1"


@dataclass(frozen=True)
class FinetuneSplitPlan:
    train_split_file: Path
    val_split_file: Path
    train_mode: TrainMode
    tray_keys: list[str]
    n_train: int
    n_val: int
    canonical_test_file: Path
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FINETUNE_SPLIT_PLAN_SCHEMA,
            "train_mode": self.train_mode,
            "tray_keys": list(self.tray_keys),
            "train_split_file": str(self.train_split_file),
            "val_split_file": str(self.val_split_file),
            "n_train": self.n_train,
            "n_val": self.n_val,
            "canonical_test_file": str(self.canonical_test_file),
            "notes": self.notes,
        }


def _read_rel_paths(path: Path) -> list[str]:
    if not path.is_file():
        return []
    entries = read_split_list(path, missing_ok=True)
    assert isinstance(entries, list)
    return [str(x) for x in entries if str(x).strip()]


def _split_has_entries(path: Path) -> bool:
    return bool(_read_rel_paths(path))


def _family_split_stem(*, split: str, tray_key: str, path: Path) -> str | None:
    """Return tray suffix when ``path`` is ``{split}_{tray_key}`` or a family variant."""
    prefix = f"{split}_"
    stem = path.stem
    if not stem.startswith(prefix):
        return None
    suffix = stem[len(prefix) :]
    key = tray_key.strip()
    if suffix == key:
        return suffix
    if suffix.startswith(key + "-") or suffix.startswith(key + "_"):
        return suffix
    return None


def domain_split_paths_for_tray_adapt(
    tray_key: str,
    *,
    split: str,
    domains_dir: Path,
) -> list[Path]:
    """
    Domain split list files for tray_adapt.

    Uses exact ``{split}_{tray_key}.txt`` when non-empty; otherwise merges all
    non-empty family lists ``{split}_{tray_key}-*.txt`` / ``{split}_{tray_key}_*.txt``.
    """
    key = str(tray_key).strip()
    if not key:
        return []
    root = domains_dir.resolve()
    exact = root / f"{split}_{key}.txt"
    if _split_has_entries(exact):
        return [exact]

    family: list[Path] = []
    for path in sorted(root.glob(f"{split}_{key}*.txt")):
        if path == exact:
            continue
        if _family_split_stem(split=split, tray_key=key, path=path) is None:
            continue
        if _split_has_entries(path):
            family.append(path)
    return family


def tray_key_has_tray_adapt_splits(tray_key: str, *, domains_dir: Path) -> bool:
    """True when tray_adapt can assemble non-empty train (or val) domain lists."""
    train_paths = domain_split_paths_for_tray_adapt(tray_key, split="train", domains_dir=domains_dir)
    val_paths = domain_split_paths_for_tray_adapt(tray_key, split="val", domains_dir=domains_dir)
    return bool(train_paths or val_paths)


def ensure_domain_split_file(
    tray_key: str,
    *,
    split: str,
    domains_dir: Path,
    cache_dir: Path | None = None,
) -> Path | None:
    """
    Resolve a single split list path for eval/finetune (exact or merged family cache).
    """
    paths = domain_split_paths_for_tray_adapt(tray_key, split=split, domains_dir=domains_dir)
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0]

    cache = (cache_dir or (domains_dir / "_merged")).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    safe_key = tray_key.replace("/", "_")
    out = cache / f"{split}_{safe_key}{MERGED_FAMILY_SPLIT_SUFFIX}"
    merged: list[str] = []
    seen: set[str] = set()
    for path in paths:
        for rel in _read_rel_paths(path):
            if rel not in seen:
                seen.add(rel)
                merged.append(rel)
    if not merged:
        return None
    out.write_text("\n".join(merged) + "\n", encoding="utf-8")
    return out


def _canonical_test_abs_set(*, canonical_test: Path, dataset_root: Path) -> set[str]:
    out: set[str] = set()
    for rel in _read_rel_paths(canonical_test):
        out.add(str(resolve_split_entry(rel, dataset_root=dataset_root)))
    return out


def assert_no_test_leakage(
    *,
    train_entries: list[str],
    val_entries: list[str],
    canonical_test: Path,
    dataset_root: Path,
) -> None:
    forbidden = _canonical_test_abs_set(canonical_test=canonical_test, dataset_root=dataset_root)
    for label, entries in (("train", train_entries), ("val", val_entries)):
        for rel in entries:
            abs_p = str(resolve_split_entry(rel, dataset_root=dataset_root))
            if abs_p in forbidden:
                raise SystemExit(
                    f"Fine-tune {label} split leaks canonical test image: {rel}\n"
                    f"  resolved: {abs_p}\n"
                    f"  test list: {canonical_test}"
                )


def _write_split_list(path: Path, entries: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    unique: list[str] = []
    seen: set[str] = set()
    for e in entries:
        if e not in seen:
            seen.add(e)
            unique.append(e)
    path.write_text("\n".join(unique) + ("\n" if unique else ""), encoding="utf-8")
    return path


def compose_tray_adapt_splits(
    tray_keys: list[str],
    *,
    domains_dir: Path,
    splits_dir: Path,
    dataset_root: Path,
    work_dir: Path,
    canonical_test: Path | None = None,
) -> FinetuneSplitPlan:
    """
    Train on labeled train+val images for holdout tray(s); early-stop on tray val.

    Never includes ``data/splits/test.txt`` paths. Requires ``--write-domain-splits``.
    """
    if not tray_keys:
        raise SystemExit("tray_adapt requires at least one tray_key")

    test_file = canonical_test or (splits_dir / "test.txt")
    train_entries: list[str] = []
    val_entries: list[str] = []

    for key in tray_keys:
        train_paths = domain_split_paths_for_tray_adapt(
            key, split="train", domains_dir=domains_dir
        )
        val_paths = domain_split_paths_for_tray_adapt(key, split="val", domains_dir=domains_dir)
        t_train: list[str] = []
        t_val: list[str] = []
        for path in train_paths:
            t_train.extend(_read_rel_paths(path))
        for path in val_paths:
            t_val.extend(_read_rel_paths(path))
        if not t_train and not t_val:
            raise SystemExit(
                f"No domain split lists for tray {key!r}. Run:\n"
                f"  python scripts/eval_domains.py --write-domain-splits "
                f"--catalog reports/domains/catalog.json --domains-dir {domains_dir}"
            )
        train_entries.extend(t_train)
        train_entries.extend(t_val)
        if t_val:
            val_entries.extend(t_val)
        elif t_train:
            val_entries.extend(t_train)

    if not train_entries:
        raise SystemExit(f"tray_adapt produced empty train list for keys {tray_keys}")

    if not val_entries:
        val_entries = list(train_entries)

    assert_no_test_leakage(
        train_entries=train_entries,
        val_entries=val_entries,
        canonical_test=test_file,
        dataset_root=dataset_root,
    )

    work_dir.mkdir(parents=True, exist_ok=True)
    keys_tag = "_".join(k.replace("/", "_") for k in tray_keys[:3])
    if len(tray_keys) > 3:
        keys_tag += f"_plus{len(tray_keys) - 3}"
    train_out = _write_split_list(work_dir / f"finetune_train_{keys_tag}.txt", train_entries)
    val_out = _write_split_list(work_dir / f"finetune_val_{keys_tag}.txt", val_entries)

    return FinetuneSplitPlan(
        train_split_file=train_out,
        val_split_file=val_out,
        train_mode="tray_adapt",
        tray_keys=list(tray_keys),
        n_train=len(train_entries),
        n_val=len(val_entries),
        canonical_test_file=test_file.resolve(),
        notes="train=union(domain train+val per tray); val=domain val per tray (fallback train)",
    )


def compose_lofo_pool_splits(
    tray_keys: list[str],
    *,
    splits_dir: Path,
    dataset_root: Path,
    work_dir: Path,
    canonical_test: Path | None = None,
) -> FinetuneSplitPlan:
    """LOFO pool: canonical train/val with holdout ``tray_key`` images removed."""
    from harchoc.domain_tags import filter_split_entries_by_domains, tray_key_from_stem

    if not tray_keys:
        raise SystemExit("lofo_pool requires at least one tray_key")

    holdout = frozenset(tray_keys)
    test_file = canonical_test or (splits_dir / "test.txt")

    def _filter_split(name: str) -> list[str]:
        src = splits_dir / f"{name}.txt"
        entries = _read_rel_paths(src)
        kept: list[str] = []
        for rel in entries:
            stem = Path(rel).stem
            if tray_key_from_stem(stem) in holdout:
                continue
            kept.append(rel)
        return kept

    train_entries = _filter_split("train")
    val_entries = _filter_split("val")
    if not train_entries:
        raise SystemExit("lofo_pool train split empty after removing holdout trays")

    assert_no_test_leakage(
        train_entries=train_entries,
        val_entries=val_entries,
        canonical_test=test_file,
        dataset_root=dataset_root,
    )

    work_dir.mkdir(parents=True, exist_ok=True)
    keys_tag = "_".join(k.replace("/", "_") for k in tray_keys[:3])
    train_out = _write_split_list(work_dir / f"lofo_train_{keys_tag}.txt", train_entries)
    val_out = _write_split_list(work_dir / f"lofo_val_{keys_tag}.txt", val_entries)

    return FinetuneSplitPlan(
        train_split_file=train_out,
        val_split_file=val_out,
        train_mode="lofo_pool",
        tray_keys=list(tray_keys),
        n_train=len(train_entries),
        n_val=len(val_entries),
        canonical_test_file=test_file.resolve(),
        notes="canonical train/val minus holdout tray_key images",
    )


def resolve_finetune_split_plan(
    *,
    train_mode: str,
    tray_keys: list[str],
    domains_dir: Path,
    splits_dir: Path,
    dataset_root: Path,
    work_dir: Path,
) -> FinetuneSplitPlan | None:
    mode = (train_mode or "canonical").strip().lower()
    if mode == "canonical":
        return None
    if mode == "tray_adapt":
        return compose_tray_adapt_splits(
            tray_keys,
            domains_dir=domains_dir,
            splits_dir=splits_dir,
            dataset_root=dataset_root,
            work_dir=work_dir,
        )
    if mode == "lofo_pool":
        return compose_lofo_pool_splits(
            tray_keys,
            splits_dir=splits_dir,
            dataset_root=dataset_root,
            work_dir=work_dir,
        )
    raise SystemExit(f"Unknown finetune train_mode: {train_mode!r} (canonical|tray_adapt|lofo_pool)")
