# Phase 1 descriptive error analysis

This report indexes observable prediction outcomes, probabilities, and view counts. It does not establish a clinical cause for any error.

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

## Representative index categories

| Category | Indexed studies | Definition |
|---|---:|---|
| false_negatives | 17 | ABNORMAL study predicted NORMAL at the operating threshold. |
| false_positives | 25 | NORMAL study predicted ABNORMAL at the operating threshold. |
| high_confidence_false_negatives | 13 | False negative with predicted-NORMAL confidence at least 0.80. |
| high_confidence_false_positives | 6 | False positive with predicted-ABNORMAL confidence at least 0.80. |
| low_confidence_predictions | 25 | Probability confidence no greater than 0.60; this is not a clinical uncertainty estimate. |
| single_view_errors | 5 | False negatives or false positives from one-view studies. |
| multi_view_errors | 25 | False negatives or false positives from multi-view studies. |
| single_view_studies | 24 | Representative one-view studies, regardless of outcome. |
| multi_view_studies | 25 | Representative multi-view studies, regardless of outcome. |

## Fine-tuned ResNet-18

Study-level operating threshold: `0.3860`.

This report is descriptive: it indexes observed labels, model probabilities, and view counts. It does not establish a clinical cause for any false negative or false positive.

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

## Representative index categories

| Category | Indexed studies | Definition |
|---|---:|---|
| false_negatives | 25 | ABNORMAL study predicted NORMAL at the operating threshold. |
| false_positives | 7 | NORMAL study predicted ABNORMAL at the operating threshold. |
| high_confidence_false_negatives | 18 | False negative with predicted-NORMAL confidence at least 0.80. |
| high_confidence_false_positives | 3 | False positive with predicted-ABNORMAL confidence at least 0.80. |
| low_confidence_predictions | 11 | Probability confidence no greater than 0.60; this is not a clinical uncertainty estimate. |
| single_view_errors | 3 | False negatives or false positives from one-view studies. |
| multi_view_errors | 25 | False negatives or false positives from multi-view studies. |
| single_view_studies | 24 | Representative one-view studies, regardless of outcome. |
| multi_view_studies | 25 | Representative multi-view studies, regardless of outcome. |
