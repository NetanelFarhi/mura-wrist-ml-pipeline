# Phase 2C geometry robustness diagnostic

This is a non-selection sensitivity check: the already-trained models were evaluated on the same internal-validation studies after controlled display-geometry changes. It cannot prove that a model used anatomy, markers, borders, or any specific clinical feature. It reports only whether scores/decisions changed under the transformations.

| model | display_variant | mean_absolute_probability_change | decision_flips_at_95_policy | decision_flip_percent_at_95_policy |
| --- | --- | --- | --- | --- |
| Frozen mean embedding | padding_midgray | 0.056076 | 58 | 8.393632 |
| Partial fine-tuned mean embedding | padding_midgray | 0.034299 | 39 | 5.643994 |
| Frozen mean embedding | outer_border_black_5pct | 0.022761 | 19 | 2.749638 |
| Partial fine-tuned mean embedding | outer_border_black_5pct | 0.023032 | 17 | 2.460203 |
| Frozen mean embedding | direct_resize | 0.076196 | 62 | 8.972504 |
| Partial fine-tuned mean embedding | direct_resize | 0.078062 | 74 | 10.709117 |
