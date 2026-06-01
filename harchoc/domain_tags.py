from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from harchoc.label_stats import count_yolo_boxes, label_path_for_image
from harchoc.splits_io import read_split_list, resolve_split_entry, resolve_splits_dir
from harchoc.split_leakage_audit import normalize_stem

# Tray/session id from CVAT-style names: 349-10-2, 3a2-2, 200-3-1
_TRAY_KEY_RE = re.compile(r"^([0-9a-z]+(?:-\d+)+)", re.IGNORECASE)


def tray_key_from_stem(stem: str) -> str:
    """
    Domain / tray tag from image stem (before augmentation suffixes).
    Falls back to normalized stem when pattern does not match.
    """
    raw = stem
    # Strip trailing aug markers common in this dataset (aug0, aug4, etc.)
    low = raw.lower()
    aug_idx = low.find("aug")
    if aug_idx > 0:
        raw = raw[:aug_idx].rstrip("_-.")
    m = _TRAY_KEY_RE.match(raw.replace("_", ""))
    if m:
        return m.group(1).lower()
    return normalize_stem(stem)


def class_counts_from_label_file(label_path: Path) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    if not label_path.is_file():
        return {}
    for ln in label_path.read_text("utf-8", errors="replace").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        toks = s.split()
        if not toks:
            continue
        try:
            cls = int(float(toks[0]))
        except ValueError:
            continue
        counts[str(cls)] += 1
    return dict(counts)


def catalog_domains_from_dataset(
    *,
    dataset_root: Path,
    splits_dir: Path | str | None = None,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, Any]:
    """
    Build domain catalog from YOLO labels under dataset_root (class tags = category_id per line).
    Groups images by tray_key derived from filename stems.
    """
    root = dataset_root.resolve()
    sd = resolve_splits_dir(dataset_root=root, splits_dir=splits_dir)

    by_domain: dict[str, dict[str, Any]] = {}
    split_membership: dict[str, set[str]] = {s: set() for s in splits}

    for split in splits:
        txt = sd / f"{split}.txt"
        if not txt.is_file():
            continue
        entries = read_split_list(txt, missing_ok=True)
        assert isinstance(entries, list)
        for rel in entries:
            rel_s = str(rel)
            img_path = resolve_split_entry(rel_s, dataset_root=root)
            stem = img_path.stem
            domain = tray_key_from_stem(stem)
            split_membership[split].add(domain)

            try:
                rel_to_root = img_path.resolve().relative_to(root)
            except ValueError:
                rel_to_root = Path(img_path.name)
            lbl = label_path_for_image(root, rel_to_root)
            cls_counts = class_counts_from_label_file(lbl)
            n_boxes = sum(cls_counts.values())

            rec = by_domain.setdefault(
                domain,
                {
                    "tray_key": domain,
                    "n_images": 0,
                    "n_boxes": 0,
                    "class_counts": defaultdict(int),
                    "splits": defaultdict(int),
                    "example_images": [],
                },
            )
            rec["n_images"] = int(rec["n_images"]) + 1
            rec["n_boxes"] = int(rec["n_boxes"]) + n_boxes
            for k, v in cls_counts.items():
                rec["class_counts"][k] += int(v)
            rec["splits"][split] += 1
            examples: list[str] = rec["example_images"]
            if len(examples) < 3:
                examples.append(rel_s)

    domains_out: list[dict[str, Any]] = []
    for key in sorted(by_domain.keys()):
        rec = by_domain[key]
        domains_out.append(
            {
                "tray_key": key,
                "n_images": rec["n_images"],
                "n_boxes": rec["n_boxes"],
                "class_counts": dict(rec["class_counts"]),
                "splits": dict(rec["splits"]),
                "example_images": list(rec["example_images"]),
            }
        )

    return {
        "status": "ok",
        "dataset_root": str(root),
        "splits_dir": str(sd),
        "n_domains": len(domains_out),
        "domains": domains_out,
        "split_domain_counts": {s: len(split_membership[s]) for s in splits},
    }


DOMAIN_METADATA_TAGS_SCHEMA = "domain_metadata_tags.v0"
DOMAIN_TAG_AXES = ("variety", "maturity", "lighting", "site")


def _normalize_tag_value(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    return s if s else None


def load_domain_tags_csv(csv_path: Path) -> tuple[dict[str, dict[str, str | None]], dict[str, Any]]:
    """
    Parse domain metadata CSV with columns: tray_key, variety, maturity, lighting, site.
    Returns normalized per_tray map (lowercase tray_key) and an import summary dict.
    """
    path = csv_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    per_tray: dict[str, dict[str, str | None]] = {}
    n_rows = 0
    n_skipped = 0
    duplicate_keys: list[str] = []

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return {}, {"csv_path": str(path), "status": "empty", "n_rows": 0}

        fields = {str(f).strip().lower(): str(f) for f in reader.fieldnames if f}
        if "tray_key" not in fields:
            raise ValueError(
                f"domain tags CSV missing tray_key column; got {list(reader.fieldnames)}"
            )

        for row in reader:
            n_rows += 1
            tk = _normalize_tag_value(str(row.get(fields["tray_key"], "") or "").lower())
            if not tk:
                n_skipped += 1
                continue

            tags: dict[str, str | None] = {}
            for axis in DOMAIN_TAG_AXES:
                col = fields.get(axis)
                tags[axis] = _normalize_tag_value(row.get(col)) if col else None

            if tk in per_tray:
                duplicate_keys.append(tk)
            per_tray[tk] = tags

    summary: dict[str, Any] = {
        "csv_path": str(path),
        "n_rows": n_rows,
        "n_trays": len(per_tray),
        "n_skipped_rows": n_skipped,
        "duplicate_tray_keys": sorted(set(duplicate_keys)),
    }
    return per_tray, summary


def tag_axes_from_per_tray(per_tray: dict[str, dict[str, str | None]]) -> dict[str, list[str] | None]:
    """Unique sorted tag values per axis; null when no values exist."""
    axes: dict[str, list[str] | None] = {}
    for axis in DOMAIN_TAG_AXES:
        vals = sorted({v for rec in per_tray.values() if (v := rec.get(axis)) is not None})
        axes[axis] = vals if vals else None
    return axes


def merge_domain_metadata_tags(
    *,
    csv_path: Path | None = None,
    catalog_tray_keys: set[str] | frozenset[str] | None = None,
    n_trays: int | None = None,
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build domain_metadata_tags for catalog JSON.
    When csv_path is set, merge rows into per_tray and set status to partial.
    """
    out: dict[str, Any] = dict(base) if base else domain_metadata_tags_scaffold(n_trays=n_trays)
    if n_trays is not None:
        out["n_trays_in_catalog"] = int(n_trays)

    if csv_path is None:
        return out

    imported, import_summary = load_domain_tags_csv(csv_path)
    per_tray: dict[str, dict[str, str | None]] = dict(out.get("per_tray") or {})
    per_tray.update(imported)
    out["per_tray"] = per_tray
    out["tag_axes"] = tag_axes_from_per_tray(per_tray)
    out["status"] = "partial" if per_tray else str(out.get("status", "scaffold"))
    out["import"] = import_summary

    if catalog_tray_keys is not None:
        catalog_set = {str(k).lower() for k in catalog_tray_keys if str(k).strip()}
        tagged = set(per_tray.keys())
        import_summary["n_catalog_trays"] = len(catalog_set)
        import_summary["n_catalog_trays_tagged"] = len(catalog_set & tagged)
        import_summary["n_catalog_trays_untagged"] = len(catalog_set - tagged)
        unknown = sorted(tagged - catalog_set)
        import_summary["unknown_tray_keys_in_csv"] = unknown
        import_summary["n_unknown_tray_keys_in_csv"] = len(unknown)

    out["notes"] = (
        "Per-tray variety/maturity/lighting/site merged from --import-domain-tags CSV. "
        "Tag axes list unique values when populated."
    )
    return out


def domain_metadata_tags_scaffold(*, n_trays: int | None = None) -> dict[str, Any]:
    """
    Honest scaffold for P1-DOMAIN-TAGS until acquisition metadata is linked to tray keys.
    Written into eval_domains catalog JSON; per_tray entries stay empty until metadata exists.
    """
    out: dict[str, Any] = {
        "schema_version": DOMAIN_METADATA_TAGS_SCHEMA,
        "status": "scaffold",
        "backlog_id": "P1-DOMAIN-TAGS",
        "notes": (
            "Variety, maturity, lighting, and site are not encoded in image stems or YOLO labels. "
            "Populate per_tray when metadata is available; tag_axes remain null (TBD) until then."
        ),
        "tag_axes": {
            "variety": None,
            "maturity": None,
            "lighting": None,
            "site": None,
        },
        "per_tray": {},
    }
    if n_trays is not None:
        out["n_trays_in_catalog"] = int(n_trays)
    return out


def attach_tray_tags_to_domains(
    domains: list[dict[str, Any]],
    domain_metadata_tags: dict[str, Any],
) -> list[dict[str, Any]]:
    """Copy per-tray tag dict onto each domain_eval domain record when tags exist."""
    per_tray = domain_metadata_tags.get("per_tray")
    if not isinstance(per_tray, dict) or not per_tray:
        return domains
    out: list[dict[str, Any]] = []
    for dom in domains:
        rec = dict(dom)
        key = str(rec.get("tray_key") or "").strip().lower()
        tags = per_tray.get(key)
        if isinstance(tags, dict) and tags:
            rec["tags"] = dict(tags)
        out.append(rec)
    return out


def filter_split_entries_by_domains(
    entries: list[str],
    *,
    dataset_root: Path,
    domains: set[str],
) -> list[str]:
    root = dataset_root.resolve()
    out: list[str] = []
    for rel in entries:
        img_path = resolve_split_entry(rel, dataset_root=root)
        if tray_key_from_stem(img_path.stem) in domains:
            out.append(rel)
    return out
