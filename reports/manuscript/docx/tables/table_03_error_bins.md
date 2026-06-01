# Table 3. Distribution of relative counting error

| Relative error bin | n images | % of test |
|--------------------|----------:|----------:|
| <2% | 15 | 13.8 |
| 2–5% | 19 | 17.4 |
| 5–10% | 24 | 22.0 |
| 10–20% | 38 | 34.9 |
| >20% | 13 | 11.9 |

## Footnotes

1. Operating confidence fixed on **val** (`min_count_mae` on `data/splits/val.txt`) and applied unchanged on **test** (`data/splits/test.txt`, *n*=109).
2. Bins from per-image |pred−gt|/gt at locked conf.
