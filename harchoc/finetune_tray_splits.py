"""Compose train/val split lists for tray-targeted fine-tune (no canonical test leakage)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from harchoc.domain_eval import domain_split_file
from harchoc.splits_io import read_split_list, resolve_split_entry

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
        train_tray = Path(domain_split_file(tray_key=key, split="train", domains_dir=domains_dir))
        val_tray = Path(domain_split_file(tray_key=key, split="val", domains_dir=domains_dir))
        t_train = _read_rel_paths(train_tray)
        t_val = _read_rel_paths(val_tray)
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
