# Aug smoke GPU queue dedup — root cause notes

## Part 1 — Recipe fingerprint dedup

`_merge_aug_smoke_jobs` and `_filter_duplicate_train_recipes` drop or skip jobs whose **effective train recipe** (merged train JSON + aug YAML) already has a **complete** smoke with a verified summary. This catches training-equivalent arms such as **S0≡S1** (scaled `close_mosaic`) and manifest arms like **close15** (recipe duplicate of S0/S1). It does **not** compare test prediction files.

## Part 2 — Preds SHA dedup at manifest load

`load_gpu_queue_manifest` runs `_filter_duplicate_preds_sha` after recipe dedup. For `aug_smoke` train jobs, it maps each smoke to a test `preds_json` SHA (from complete summaries or `equivalence_classes.preds_sha256`) and sets `skip=true` when a **different** smoke_id would repeat an owner SHA (e.g. re-queued **S6** vs complete **S3**). Eval-only jobs (e.g. **S14**) are excluded from preds dedup.

**Verified 2026-05-30:** Re-adding **S6** / **S7** as `gpu_pending` in a fixture index and loading `gpu_queue_aug_pending.json` via `load_gpu_queue_manifest()` sets `skip=true` on both jobs. With production summaries on disk, `should_skip_job()` also returns `summary complete: …` before train.

### Mechanism

`load_gpu_queue_manifest()` runs `_filter_duplicate_train_recipes()` then `_filter_duplicate_preds_sha()` on the merged job list.

| Helper | Role |
|--------|------|
| `_complete_preds_sha_owners` | Maps `artifacts.preds_json.sha256` → first **complete** smoke with verified summary (count MAE present). S3 owns `41e79d287721…` because it precedes S6/S7 in the index. |
| `_index_preds_sha_by_smoke_id` | Maps smoke_id → `equivalence_classes.classes[].preds_sha256` (audit metadata). Keys are uppercased (`S6`, not `s6`). |
| `_job_preds_sha_for_dedup` | Resolves a pending train job’s known preds SHA: **(1)** `index_preds[sid]`, else **(2)** `skip_if.summary` on disk. Returns `None` for eval-only jobs. |
| `_filter_duplicate_preds_sha` | If resolved SHA is owned by another complete smoke → `skip=true`, `skip_reason=preds duplicate of complete smoke {owner} (sha=…)`. Also dedupes duplicate SHAs within one manifest. |

**S3 / S6 / S7 fingerprints (distinct recipes, identical preds):**

| Smoke | `job_train_recipe_fingerprint` (prefix) | test preds SHA (prefix) |
|-------|----------------------------------------|-------------------------|
| S3 | `ecb0637903bcbadf…` | `41e79d287721faf9…` (owner) |
| S6 | `aa3385baae9b69e3…` | `41e79d287721faf9…` |
| S7 | `c86fc0173db434df…` | `41e79d287721faf9…` |

Recipe dedup correctly keeps all three as distinct hypotheses; preds dedup skips S6/S7 once S3’s summary is verified.

### `_job_preds_sha_for_dedup` and `equivalence_classes`

When a pending smoke has **no** summary file yet, preds dedup still works if the index lists the smoke in `equivalence_classes` with `preds_sha256`. Lookup order in `_job_preds_sha_for_dedup`:

1. `sid in index_preds` → return class SHA (no disk read).
2. Else read `skip_if.summary` (or default `reports/aug_smoke/{sid}_summary.json`).
3. Else return `None` (job not preds-deduped).

Integration test `test_integration_gpu_pending_s6_skipped_preds_duplicate` disables `equivalence_classes` on the fixture index and still skips S6 via S3’s on-disk summary — preds dedup does not depend on audit metadata when a verified owner summary exists.

### “Lazy preemptive input cache comparison” — what existed before preds dedup

No function with that name exists. The phrase maps to **reactive skip paths that only fire after a prior run wrote artifacts**:

| Mechanism | When it skips | Why it failed for historical S6/S7 |
|-----------|---------------|-------------------------------------|
| `skip_if.summary` + `_summary_is_verified_complete` | Summary file exists with count MAE | S6/S7 had no summary until **after** GPU train |
| `skip_if.index_status: complete` | Index row already `complete` | Rows were `gpu_pending` when queued |
| `artifact_fingerprints` in summaries | Post-hoc audit in `finalize_smoke_job` | Written at job end, not consulted pre-train |
| Recipe fingerprint dedup | Same effective train+aug recipe | S6/S7 differ from S3 in erasing YAML — **by design** |
| Leaderboard `equivalence_classes` | Ranking / audit only (pre-fix) | Not wired to queue expansion until audit-only expand skip |

Preds SHA dedup closes the gap: it compares **outcome identity** (test preds file hash), not training recipe identity, using either verified owner summaries or index audit SHA.

### Wiring today (two layers)

1. **Manifest preds dedup** (`_filter_duplicate_preds_sha`) — always runs in `load_gpu_queue_manifest`.
2. **Audit-only expand skip** (`expand_aug_smoke_jobs_from_index` + `parse_equivalence_classes`) — marks non-canonical class members (S0, S6, S7, S13, …) at expansion with `audit-only equivalence class (canonical …; preds_sha256=…)`.

Both set `skip=true`; `should_skip_job()` surfaces `job.skip_reason` when manifest skip is set. Preds dedup remains the fallback when audit metadata is stripped from a fixture index but owner summaries exist.

### Tests

`tests/test_gpu_queue.py`: unit tests on `_filter_duplicate_preds_sha`, `_job_preds_sha_for_dedup`, and `GpuQueueDedupIntegrationTests` (manifest load with `include_equivalence_classes=False` for S6/S7 preds path).

## Part 3 — Why S6/S7 appeared in `gpu_queue_aug_pending` before preds dedup

Historical queue behavior (pre-2026-05-30 wiring):

1. **`gpu_queue_aug_pending.json`** sets `aug_smoke_from_index: true` and only lists `preflight`. Every `gpu_pending` row in [`aug_smoke_index.json`](../../configs/experiments/aug_smoke_index.json) is expanded by `expand_aug_smoke_jobs_from_index` — there was no audit-only filter at expansion time.

2. **S6** (`erasing=0`) and **S7** (`erasing=0.3`) were legitimate **hypothesis** arms when marked `gpu_pending`: they differ from **S3** in aug YAML and in `job_train_recipe_fingerprint`, so **recipe dedup did not remove them** (unlike **S0/S1**, which share an effective recipe).

3. **`aug_smoke_index_queue_parity`** requires every `gpu_pending` smoke to appear as an `aug_smoke_*` job in the expanded manifest, so the runner emitted runnable jobs for S6/S7 rather than omitting them.

4. **Preds equivalence was unknown until after training.** At 15 epochs, erasing is **inert** for this dataset: S3/S6/S7 share identical test preds (`preds_sha256=41e79d28…`, see `equivalence_classes` in the index). That was established by post-hoc SHA audit, not by recipe fingerprinting.

5. **Leaderboard** already treated S6/S7 as audit-only via `parse_equivalence_classes`, but the **GPU queue** scheduled them until preds dedup at manifest load (and now **audit-only skip at expand**) aligns queue behavior with ranking policy.

**Current fix:** `expand_aug_smoke_jobs_from_index` marks non-canonical members of `equivalence_classes` with `skip=true` and a `skip_reason` naming the canonical smoke and preds SHA, so audit arms are not scheduled even while still `gpu_pending` in the index for parity.
