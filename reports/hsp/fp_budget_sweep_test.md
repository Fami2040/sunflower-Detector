# FP budget constraint sweep (test)

**Split:** test (n=109) · **Match:** IoU 0.3, category-aware
**Sweep rows:** `reports/hsp/threshold_test_locked.json` (19 conf steps)

## Selection comparison

| Mode | conf | FP/img | Count MAE | F1 |
|------|------|--------|-----------|-----|
| min_count_mae | 0.15 | 193.6 | 61.3 | 0.610 |
| best_f1 | 0.10 | 256.5 | 79.4 | 0.616 |

## Constraint grid (max FP/image cap)

| Cap | conf | FP/img | Count MAE | F1 |
|-----|------|--------|-----------|-----|
| 25 | 0.65 | 16.9 | 474.6 | 0.197 |
| 50 | 0.60 | 34.6 | 419.3 | 0.290 |
| 100 | 0.35 | 96.4 | 253.1 | 0.479 |
| 150 | 0.25 | 130.2 | 166.5 | 0.550 |
| 200 | 0.15 | 193.6 | 61.3 | 0.610 |
| 217 | 0.15 | 193.6 | 61.3 | 0.610 |
| 250 | 0.15 | 193.6 | 61.3 | 0.610 |
| 300 | 0.10 | 256.5 | 79.4 | 0.616 |

## Manuscript pick
**Primary (locked):** val-selected conf **0.15** applied unchanged on test — count MAE **61.3**, FP/img **193.6**, F1 **0.610**. Threshold chosen on val (`min_count_mae`); test reports the same conf only (no re-selection on test).
F1-max on test would raise count MAE by **+18.1** (conf 0.10) vs locked point — supports count-first selection over detection F1 alone.
Tighter FP cap **25/img** (conf 0.65) raises count MAE by **+413.3** vs locked — dense-tray counting favors the val-locked operating point over strict FP budgets.
