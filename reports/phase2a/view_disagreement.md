# Validation-only view-disagreement exploration

`std_view_probability` and `range_view_probability` describe dispersion of Model B probabilities across real radiographic views in a study. They are not uncertainty estimates and do not establish causation.

| Policy | Group | n | Median SD | Median range |
|---|---:|---:|---:|---:|
| P1 historical Youden-J | correct | 587 | 0.026919 | 0.047632 |
| P1 historical Youden-J | incorrect | 105 | 0.338128 | 0.622818 |
| P1 historical Youden-J | false_negative | 53 | 0.157535 | 0.310346 |
| P1 historical Youden-J | true_positive | 212 | 0.027509 | 0.049343 |
| P2 sensitivity >= 0.85 | correct | 567 | 0.024289 | 0.043287 |
| P2 sensitivity >= 0.85 | incorrect | 125 | 0.351001 | 0.654940 |
| P2 sensitivity >= 0.85 | false_negative | 39 | 0.069926 | 0.136076 |
| P2 sensitivity >= 0.85 | true_positive | 226 | 0.034897 | 0.062656 |
| P3 sensitivity >= 0.90 | correct | 544 | 0.021152 | 0.039242 |
| P3 sensitivity >= 0.90 | incorrect | 148 | 0.321390 | 0.570135 |
| P3 sensitivity >= 0.90 | false_negative | 26 | 0.045454 | 0.087246 |
| P3 sensitivity >= 0.90 | true_positive | 239 | 0.045396 | 0.083003 |

## Error transitions between policies

| From → to | FNs resolved | New FPs introduced |
|---|---:|---:|
| P1 historical Youden-J → P2 sensitivity >= 0.85 | 14 | 34 |
| P2 sensitivity >= 0.85 → P3 sensitivity >= 0.90 | 13 | 36 |

The same frozen image model and mean-pooling scores are used in every row. The analysis is restricted to the development validation set; it neither trains learned pooling nor uses the already-opened test set.
