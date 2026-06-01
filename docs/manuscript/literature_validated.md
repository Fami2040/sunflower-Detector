# Validated literature registry

Machine-readable source: [`literature_validated.json`](literature_validated.json) (`literature_validated.v1`).

**Checked:** 2026-05-29 · **Gap map:** [`reviewer_comments_backlog_gap.md`](reviewer_comments_backlog_gap.md)

## Reviewer-requested papers

| ID | DOI | Reviewer use | Alignment |
|----|-----|--------------|-----------|
| `yang2024_oct_tl` | [10.1371/journal.pone.0296175](https://doi.org/10.1371/journal.pone.0296175) | Domain adaptation (§12) | **Analogy** — OCT ensemble + transfer, not seed OD |
| `ren2025_scripta_interp` | [10.1016/j.scriptamat.2024.116350](https://doi.org/10.1016/j.scriptamat.2024.116350) | Explainability (§13) | **Framing** — interpretable ML; pair with Grad-CAM panel |
| `alshehri2025_uav` | [10.3389/fnbot.2025.1582995](https://doi.org/10.3389/fnbot.2025.1582995) | Two-stage deploy (§14) | **Analogy** — UAV action recognition; gate+detect vs HSP single-stage; **MS-DEPLOY-2STG** Done |
| `yao2025_hfuzzy` | [10.1109/TFUZZ.2025.3549791](https://doi.org/10.1109/TFUZZ.2025.3549791) | Boundary / ambiguous seeds (§15) | **Methods + Discussion analogy** — **graded trust on detections** (locked conf + low-conf score band + `ambiguous_summary` + FP taxonomy; `fig_ambiguous_panel`); stack step 3 with **MS-FUZZY-BOUND** Done, **P1-FP-BUDGET**, **P1-UNCERT-FP**; Yao is tabular fuzzy **regression**, not a 3rd detect class or relabel protocol |

## Crop-seed / training peers (repo anchors)

| ID | DOI | Architecture takeaway | Research doc |
|----|-----|----------------------|--------------|
| `grainnet2025` | [10.1186/s13007-025-01363-y](https://doi.org/10.1186/s13007-025-01363-y) | Counting MAE + EMA/background (cite-only; **ARCH-EMA-BG-SPIKE** Done) | [fp_taxonomy](../research/fp_taxonomy_literature.md) · [arch_ema_bg_spike](../research/arch_ema_bg_spike_literature.md) · [originality §3](originality_contribution_peers.md#3-peer-comparison-cite-table) |
| `lwcd_yolo2025` | [10.3390/agriculture15181968](https://doi.org/10.3390/agriculture15181968) | Mosaic off @ dense benchtop; **MS-ORIG** peer | [aug scan](../research/training_tech_scan_2026_augmentation.md) · [originality §3](originality_contribution_peers.md#3-peer-comparison-cite-table) |
| `tfa2020` | [arXiv:2003.06957](https://arxiv.org/abs/2003.06957) | Staged freeze finetune | [domain_shift](../research/domain_shift_transfer_literature.md) |
| `gandhi2025_yolov8_freeze` | [arXiv:2505.01016](https://arxiv.org/abs/2505.01016) | YOLOv8 freeze depth (preprint) | [domain_shift](../research/domain_shift_transfer_literature.md) |

## Phenotyping / sunflower (Related Work — **MS-LIT** Done)

| ID | DOI | §2 use | Research doc |
|----|-----|--------|--------------|
| `gwhd2020` | [10.34133/2020.3521832](https://doi.org/10.34133/2020.3521832) | Dense organ phenotyping benchmark | [fp_taxonomy](../research/fp_taxonomy_literature.md) |
| `iamchuen2026_sunflower_uav` | [10.3390/su18021026](https://doi.org/10.3390/su18021026) | Sunflower heads; conf/IoU grids | [domain_shift](../research/domain_shift_transfer_literature.md), [eval scan](../research/training_tech_scan_2026_eval_calibration.md) |
| `gulzar2025_sunflower_tl` | [10.55730/1300-0152.2763](https://doi.org/10.55730/1300-0152.2763) | Field TL / domain-shift survey | [domain_shift](../research/domain_shift_transfer_literature.md) |

**§2 outline (repo draft):** [`related_work_outline.md`](related_work_outline.md) · gap [§6](reviewer_comments_backlog_gap.md#6-manuscript-draft--related-work--literature-review-depth)

## Where to update next

When adding a cite: append JSON entry → link from one `docs/research/*.md` section → add or extend backlog row (`MS-*`, `ARCH-*`, `LIT-VALIDATE`).
