# Table 2. Counting summary (HSP test reproduction)

| Metric | Value |
|--------|------:|
| Locked confidence | 0.15 |
| Test images (n) | 109 |
| Count MAE | 61.3 |
| Mean relative error (%) | 12.04 |
| Median relative error (%) | 9.34 |
| % images rel. error <2% | 13.8 |

## Footnotes

1. Operating confidence fixed on **val** (`min_count_mae` on `data/splits/val.txt`) and applied unchanged on **test** (`data/splits/test.txt`, *n*=109).
2. Full test n=109; docx Table 2 cites n=50 blinded audit (not in repo).
