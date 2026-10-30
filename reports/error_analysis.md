# Phase 1 descriptive error analysis

This report indexes observable prediction outcomes, raw model probabilities, and view counts. It does not establish a clinical cause for any error.

> **How to read the tables:** “Observable outcome counts” contains the full count for the 237-study reference evaluation. “Representative index categories” is a compact, deterministic index with at most 25 studies per category; its rows can overlap and are not complete error totals. For example, a high-score false negative is also included in the false-negative category. Use the observable outcome table for the complete false-negative and false-positive counts.
>
> **Score wording:** a “predicted-class score” is the model's uncalibrated raw probability for the class it predicted. It is not calibrated confidence, clinical confidence, or an uncertainty estimate.

## Frozen ResNet-18 embeddings plus Logistic Regression

Study-level operating threshold: `0.3543`.

This report is descriptive: it indexes observed labels, model probabilities, and view counts. It does not establish a clinical cause for any false negative or false positive.

## Observable outcome counts

| View grouping | Outcome | Studies |
|---|---|---:|
| multi_view | false_negative | 15 |
| multi_view | false_positive | 24 |
| multi_view | true_negative | 94 |
| multi_view | true_positive | 80 |
| single_view | false_negative | 2 |
| single_view | false_positive | 3 |
| single_view | true_negative | 19 |
| all_studies | all | 237 |

## Representative index categories (non-exhaustive subsets)

| Category | Indexed studies | Definition |
|---|---:|---|
| false_negatives | 17 | ABNORMAL study predicted NORMAL at the operating threshold. |
| false_positives | 25 | NORMAL study predicted ABNORMAL at the operating threshold. |
| high_predicted_class_score_false_negatives | 13 | False negative with raw predicted-class score at least 0.80 (abnormality probability at most 0.20); not calibrated confidence. |
| high_predicted_class_score_false_positives | 6 | False positive with raw predicted-class score at least 0.80 (abnormality probability at least 0.80); not calibrated confidence. |
| near_midpoint_score_predictions | 25 | Raw predicted-class score no greater than 0.60 (abnormality probability between 0.40 and 0.60); not a clinical uncertainty estimate. |
| single_view_errors | 5 | False negatives or false positives from one-view studies. |
| multi_view_errors | 25 | False negatives or false positives from multi-view studies. |
| single_view_studies | 24 | Representative one-view studies, regardless of outcome. |
| multi_view_studies | 25 | Representative multi-view studies, regardless of outcome. |

## Fine-tuned ResNet-18

Study-level operating threshold: `0.3860`.

This report is descriptive: it indexes observed labels, raw model probabilities, and view counts. It does not establish a clinical cause for any false negative or false positive.

## Observable outcome counts

| View grouping | Outcome | Studies |
|---|---|---:|
| multi_view | false_negative | 24 |
| multi_view | false_positive | 6 |
| multi_view | true_negative | 112 |
| multi_view | true_positive | 71 |
| single_view | false_negative | 2 |
| single_view | false_positive | 1 |
| single_view | true_negative | 21 |
| all_studies | all | 237 |

## Representative index categories (non-exhaustive subsets)

| Category | Indexed studies | Definition |
|---|---:|---|
| false_negatives | 25 | ABNORMAL study predicted NORMAL at the operating threshold. |
| false_positives | 7 | NORMAL study predicted ABNORMAL at the operating threshold. |
| high_predicted_class_score_false_negatives | 18 | False negative with raw predicted-class score at least 0.80 (abnormality probability at most 0.20); not calibrated confidence. |
| high_predicted_class_score_false_positives | 3 | False positive with raw predicted-class score at least 0.80 (abnormality probability at least 0.80); not calibrated confidence. |
| near_midpoint_score_predictions | 11 | Raw predicted-class score no greater than 0.60 (abnormality probability between 0.40 and 0.60); not a clinical uncertainty estimate. |
| single_view_errors | 3 | False negatives or false positives from one-view studies. |
| multi_view_errors | 25 | False negatives or false positives from multi-view studies. |
| single_view_studies | 24 | Representative one-view studies, regardless of outcome. |
| multi_view_studies | 25 | Representative multi-view studies, regardless of outcome. |
