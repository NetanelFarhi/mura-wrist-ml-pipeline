# Results entry point

## Historical Phase-1 reference only

The following is the once-opened official MURA `valid` reference evaluation (237 wrist studies). Both thresholds were selected and frozen using the internal patient-level validation split before this evaluation. These values are preserved history; they are not an untouched evaluation for later Phase-2 model selection.

| Model | AUROC | AUPRC | Sensitivity | Specificity | F1 | False negatives |
|---|---:|---:|---:|---:|---:|---:|
| Frozen ImageNet ResNet-18 embeddings + Logistic Regression | 0.876 | 0.860 | 0.825 | 0.807 | 0.784 | 17 / 97 |
| Fine-tuned ResNet-18 | 0.897 | 0.894 | 0.732 | 0.950 | 0.811 | 26 / 97 |

The two models were evaluated on the same population. Fine-tuning improved ranking metrics, but its historical Youden-J operating point missed more abnormal studies while flagging fewer normal studies. This does not establish clinical performance or acute-fracture detection.

See [TEST_SET_GOVERNANCE.md](TEST_SET_GOVERNANCE.md) for permitted future use, [baseline_results.md](baseline_results.md) for historical detail, and the phase reports for explicitly labelled development/validation analyses.
