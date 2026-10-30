# Phase 1 baseline evaluation results

> **Historical official-valid reference only.** The official MURA `valid` split was opened once for the frozen Phase 1 baseline evaluation reported below. These values must not be used to choose or re-choose a later model, threshold, calibration method, architecture, or agent policy. See [TEST_SET_GOVERNANCE.md](TEST_SET_GOVERNANCE.md) for what later phases may and may not use.

All values are study-level results. Operating thresholds in the primary table were selected from validation predictions only; the final test data were used only to calculate the reported metrics.

## Validation-selected operating threshold

| Model | Threshold | AUROC | AUPRC | Recall | Sensitivity | Specificity | F1 | FN Rate | Studies | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Frozen ResNet-18 embeddings plus Logistic Regression | 0.3543 | 0.8756 | 0.8596 | 0.8247 | 0.8247 | 0.8071 | 0.7843 | 0.1753 | 237 | 113 | 27 | 17 | 80 |
| Fine-tuned ResNet-18 | 0.3860 | 0.8970 | 0.8937 | 0.7320 | 0.7320 | 0.9500 | 0.8114 | 0.2680 | 237 | 133 | 7 | 26 | 71 |

## Fixed threshold = 0.5

| Model | Threshold | AUROC | AUPRC | Recall | Sensitivity | Specificity | F1 | FN Rate | Studies | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Frozen ResNet-18 embeddings plus Logistic Regression | 0.5000 | 0.8756 | 0.8596 | 0.7320 | 0.7320 | 0.9143 | 0.7889 | 0.2680 | 237 | 128 | 12 | 26 | 71 |
| Fine-tuned ResNet-18 | 0.5000 | 0.8970 | 0.8937 | 0.7010 | 0.7010 | 0.9786 | 0.8095 | 0.2990 | 237 | 137 | 3 | 29 | 68 |

## AUROC bootstrap intervals

| Model | AUROC | 95% CI |
|---|---:|---:|
| Frozen ResNet-18 embeddings plus Logistic Regression | 0.8756 | [0.8249, 0.9201] |
| Fine-tuned ResNet-18 | 0.8970 | [0.8518, 0.9353] |

## Integrity/provenance evidence

Image-manifest split integrity check: passed.
