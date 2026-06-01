#!/usr/bin/env bash
# Assemble a zip for co-authors: manuscript prose, figures, curated metrics (no multi‑GB preds).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATE_TAG="${1:-$(date +%Y-%m-%d)}"
STAGE="${ROOT}/reports/_science_bundle_staging"
OUT_ZIP="${ROOT}/reports/sunflower_science_bundle_${DATE_TAG}.zip"
BUNDLE_NAME="sunflower_science_bundle_${DATE_TAG}"

rm -rf "${STAGE}"
mkdir -p "${STAGE}/${BUNDLE_NAME}"/{manuscript/tables,manuscript/docx,manuscript/lit_audit,figures,metrics,aug_smoke,domains,docs,reviewer_audits}

# --- Navigation ---
cp "${ROOT}/reports/manuscript/reviewer2_rebuttal_for_coauthor.md" \
  "${STAGE}/${BUNDLE_NAME}/"
cp "${ROOT}/reports/reviewer2.md" "${STAGE}/${BUNDLE_NAME}/"
cp "${ROOT}/reports/manuscript/response_to_reviewers.md" \
  "${STAGE}/${BUNDLE_NAME}/manuscript/"
cp "${ROOT}/reports/manuscript/literature_doi_audit_2026-06-01.md" \
  "${STAGE}/${BUNDLE_NAME}/manuscript/"
cp -r "${ROOT}/reports/manuscript/lit_audit/." \
  "${STAGE}/${BUNDLE_NAME}/manuscript/lit_audit/"

# Manuscript prose + tables (skip large json except tables_manifest)
for f in abstract.md dataset.md results_and_methods.md FRESHNESS.md docx_vs_submission.md README.md; do
  cp "${ROOT}/reports/manuscript/${f}" "${STAGE}/${BUNDLE_NAME}/manuscript/" 2>/dev/null || true
done
cp "${ROOT}/reports/manuscript/tables/"*.md "${STAGE}/${BUNDLE_NAME}/manuscript/tables/" 2>/dev/null || true
cp -r "${ROOT}/reports/manuscript/docx/." "${STAGE}/${BUNDLE_NAME}/manuscript/docx/"

# Figures + diagrams
cp -r "${ROOT}/reports/figures/." "${STAGE}/${BUNDLE_NAME}/figures/"

# Curated metrics (headline + zoo + errors; no preds/gt blobs)
METRICS=(
  dual_metric.json
  eval_test.json
  eval_test_map.json
  threshold_val.json
  threshold_test_locked.json
  error_test_report.json
  tide_bucket_summary.json
  tide_bucket_summary_val.json
  matrix_train.json
  matrix_plan.json
  split_drift_p0.json
  p0_summary.md
)
for m in "${METRICS[@]}"; do
  if [[ -f "${ROOT}/reports/hsp/${m}" ]]; then
    cp "${ROOT}/reports/hsp/${m}" "${STAGE}/${BUNDLE_NAME}/metrics/"
  fi
done

cp "${ROOT}/reports/aug_smoke/"*summary*.json "${STAGE}/${BUNDLE_NAME}/aug_smoke/" 2>/dev/null || true
cp "${ROOT}/reports/domains/domain_eval.json" "${STAGE}/${BUNDLE_NAME}/domains/" 2>/dev/null || true
cp "${ROOT}/reports/domains/catalog.json" "${STAGE}/${BUNDLE_NAME}/domains/" 2>/dev/null || true

DOCS=(
  ORIGIN_MAIN_AND_DATASET.md
  manuscript/val_test_map_gap.md
  manuscript/originality_contribution_peers.md
  manuscript/related_work_outline.md
  manuscript/literature_validated.md
  manuscript/literature_validated.json
  zoo_comparison_design.md
)
for d in "${DOCS[@]}"; do
  if [[ -f "${ROOT}/docs/${d}" ]]; then
    base="$(basename "${d}")"
    cp "${ROOT}/docs/${d}" "${STAGE}/${BUNDLE_NAME}/docs/${base}"
  fi
done

for j in reviewer2_counting_metrics_computed.json reviewer2_map50_computed.json \
  reviewer2_confusion_tide.json reviewer2_paste_check.json; do
  if [[ -f "${ROOT}/reports/${j}" ]]; then
    cp "${ROOT}/reports/${j}" "${STAGE}/${BUNDLE_NAME}/reviewer_audits/"
  fi
done

cat > "${STAGE}/${BUNDLE_NAME}/BUNDLE_README.md" <<'EOF'
# Sunflower Detector — science bundle (co-author)

**Start here:** `reviewer2_rebuttal_for_coauthor.md` (point-by-point rebuttal) and `manuscript/results_and_methods.md` (canonical numbers).

## Folder map

| Path | Contents |
|------|----------|
| `reviewer2.md` | Verbatim Reviewer 2 comments |
| `reviewer2_rebuttal_for_coauthor.md` | Draft responses + headline table |
| `manuscript/` | Abstract, Methods/Results, dataset, freshness, formal `response_to_reviewers.md` |
| `manuscript/literature_doi_audit_2026-06-01.md` | Verified DOI + honest `claim_fit` for all registry cites |
| `manuscript/lit_audit/` | Per-paper audit reports (11 entries) |
| `manuscript/tables/` | `zoo_core.md`, `headline_metrics.md`, aug smoke top-N |
| `manuscript/docx/` | Journal-style tables (markdown) + **300 DPI figures** (PNG) |
| `figures/` | Error taxonomy, ambiguous panel, concept diagram, split-drift plots, threshold PR curve |
| `metrics/` | Small JSON: `dual_metric.json`, `eval_test_map.json`, thresholds, zoo `matrix_train.json`, TIDE summary |
| `metrics/p0_summary.md` | One-page headline card |
| `aug_smoke/` | 100-epoch confirm summary JSON |
| `domains/` | Tray/session domain eval (`domain_eval.json`) |
| `docs/` | Provenance, val/test mAP gap, originality peers, related-work outline, `literature_validated.json`, zoo design |
| `reviewer_audits/` | Programmatic paste-check / counting / mAP audit JSON |

## Headline numbers (do not cite stale draft values)

- Test **count MAE 61.3** (95% CI 51.3–71.3) → `metrics/dual_metric.json`
- Test **mAP50 0.18** → `metrics/eval_test_map.json`
- Locked confidence **~0.15** → `metrics/threshold_test_locked.json`
- Zoo: `manuscript/tables/zoo_core.md` + `metrics/matrix_train.json`

## Figures for the paper

| Figure role | File |
|-------------|------|
| Detection example | `manuscript/docx/figures/figure_01_detection_example.png` |
| Training curves | `manuscript/docx/figures/figure_02_training_curves.png` |
| Confusion (abs / norm) | `figure_04_*`, `figure_05_*` |
| Metrics panels | `figure_06_metrics_panels.png` |
| FP taxonomy | `figures/fig_error_taxonomy.png` |
| Ambiguous boundaries | `figures/fig_ambiguous_panel.png` |
| Pipeline concept | `figures/fig_concept.png` / `.svg` |

## Reproduce on a GPU machine (not included in zip)

```bash
git clone https://github.com/Fami2040/sunflower-Detector.git
cd sunflower-Detector && git checkout pr/backlog-ci-dataset
mamba env create -f envs/mamba.yml -y
mamba run -n harchoc python scripts/bootstrap_env.py --create
# Set DATASET_ROOT per data/manifest.json
mamba run -n harchoc python scripts/experiment.py repro
mamba run -n harchoc python scripts/experiment.py manuscript-preflight
```

Full dataset weights are local/gitignored; see `docs/ORIGIN_MAIN_AND_DATASET.md`.

## Not in this zip (too large or local-only)

- Per-image prediction JSON (`preds_*.json`, 10–90 MB each)
- Submitted Word file `reports/plants-4336582.docx` (keep separately)
- Raw `runs/` training logs

Regenerate this bundle: `./scripts/build_science_friend_bundle.sh` from repo root.
EOF

rm -f "${OUT_ZIP}"
(cd "${STAGE}" && zip -rq "${OUT_ZIP}" "${BUNDLE_NAME}")
rm -rf "${STAGE}"
echo "Wrote ${OUT_ZIP}"
du -h "${OUT_ZIP}"
unzip -l "${OUT_ZIP}" | tail -5
