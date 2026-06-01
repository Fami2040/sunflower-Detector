# Dataset (Methods subsection)

## Corpus and imaging

We used a CVAT-annotated corpus of sunflower capitula imaged on a **benchtop tray** under **fixed indoor lighting** at a **single site**. Heads were **dried** before imaging. The public annotation share lists approximately **2500** images (share identifier `y9xGFqCW`). For modeling we used a frozen pool of **1093** images with seed-level bounding boxes.

This design supports high-throughput viability counting under controlled conditions. It does **not** establish field robustness across varieties, growth stages, natural illumination, or geographic sites.

## Annotations

Each seed instance received a YOLO-format axis-aligned box with class **developed** (id 0) or **aborted** (id 1). Coordinates are normalized to image width and height. Images are dense: on the held-out test split (*n* = 109) the mean ground-truth count is approximately **554** boxes per image.

## Splits and leakage control

Train, validation, and test lists (`875` / `109` / `109` images) were drawn by a reproducible random holdout (seed 0, 80% / 10% / 10%). List checksums are recorded in the manuscript reproducibility bundle.

- **Validation** is used only to select the global counting confidence (minimize count MAE).
- **Test** is held out for all headline metrics reported in the Results.

CVAT export folder names (`images/train`, `images/val`) do not define the modeling boundary; only the frozen split files do.

## Tray structure and optional metadata

Image file stems encode tray and session identifiers (for example tray keys such as `3a5-9`). Supplementary evaluation reports count MAE per tray slice to quantify acquisition-session variability. Optional CSV metadata (variety, maturity, lighting, site) can be merged for future multi-site studies; the headline manuscript numbers use the pooled test split unless stated otherwise.

## Provenance relative to upstream code

Detection weights (`best2.pt`) originate from the public [sunflower-Detector](https://github.com/Fami2040/sunflower-Detector) repository. **Labels, splits, and evaluation protocol in this study are fork-specific** and tied to the CVAT corpus above. The Kaggle hub dataset cited in some upstream drafts is **not** the corpus for the manuscript statistics reported here.

## Availability

Annotation share: https://izba-memes.ru/share/y9xGFqCW  

Operational install and directory layout for local reproduction are documented in `data/README.md` at the repository root (not part of this Methods text).
