# MURA X Phase 1 summary

> **Current governance note:** the references below to a “sealed test” describe the frozen Phase-1 evaluation protocol. Official MURA `valid` was opened once for that reference result, so it is not untouched for later model, threshold, calibration, architecture, or agent selection. See [TEST_SET_GOVERNANCE.md](TEST_SET_GOVERNANCE.md).

## Status

**COMPLETE — the planned Phase 1 data audit, patient-safe split, Model A and Model B runs, frozen validation selections, sealed test evaluation, and descriptive error analysis have been executed.** This is a research baseline for MURA XR_WRIST NORMAL/ABNORMAL study classification, not a fracture-diagnosis or autonomous clinical-decision result.

## Dataset and split audit

The executed [XR_WRIST audit](wrist_data_audit.json) measured **3,474 patients, 3,697 studies, and 10,411 images**: 2,274 NORMAL and 1,423 ABNORMAL studies. The official MURA `valid` source was kept as the sealed final test set; train/validation were created from official training patients with seed 23 and requested validation fraction 0.20.

| Final split | Patients | Studies | Images | NORMAL studies | ABNORMAL studies |
|---|---:|---:|---:|---:|---:|
| Train | 2,614 | 2,768 | 7,805 | 1,707 | 1,061 |
| Validation | 653 | 692 | 1,947 | 427 | 265 |
| Test (official `valid`) | 207 | 237 | 659 | 140 | 97 |
| **Total** | **3,474** | **3,697** | **10,411** | **2,274** | **1,423** |

The audit found 0 duplicate paths, missing files, corrupt images, folder-label mismatches, inconsistent study labels, study/patient identity conflicts, patient overlap, study overlap, or unassigned studies. The final evaluator separately checked all 10,411 joined manifest rows and found no violations; its image-row split counts were 7,805 train, 1,947 validation, and 659 test.

## Leakage and integrity evidence

Software tests are reported separately from ML performance. The current synthetic suite completed with **17 passing tests**. It covers manifest/split contracts, patient and study isolation, label integrity, study grouping, validation-only thresholds, frozen-threshold handoff, test-leakage rejection, and report generation.

For both trained models, the persisted selection record states `selection_data: validation only` and `test_used_for_selection: false`. The final evaluator consumed each frozen validation threshold rather than selecting a new threshold on test data. A separate global provenance JSON was not supplied to the evaluator; the run therefore relies on the model-specific selection records, run configurations, split audit, and sealed prediction artifacts as its provenance evidence.

## Final sealed test results

All metrics below are study-level on the same 237-study sealed test set. AUROC confidence intervals are 95% stratified study-bootstrap percentile intervals with 2,000 replicates and seed 42.

### Validation-selected operating threshold

| Model | Threshold | AUROC (95% CI) | AUPRC | Recall / sensitivity | Specificity | F1 | FN rate | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen ResNet-18 embeddings + Logistic Regression | 0.3543 | 0.8756 (0.8249–0.9201) | 0.8596 | 0.8247 | 0.8071 | 0.7843 | 0.1753 | 113 / 27 / 17 / 80 |
| Fine-tuned ResNet-18 | 0.3860 | 0.8970 (0.8518–0.9353) | 0.8937 | 0.7320 | 0.9500 | 0.8114 | 0.2680 | 133 / 7 / 26 / 71 |

### Fixed threshold = 0.5

| Model | AUROC | AUPRC | Recall / sensitivity | Specificity | F1 | FN rate | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen ResNet-18 embeddings + Logistic Regression | 0.8756 | 0.8596 | 0.7320 | 0.9143 | 0.7889 | 0.2680 | 128 / 12 / 26 / 71 |
| Fine-tuned ResNet-18 | 0.8970 | 0.8937 | 0.7010 | 0.9786 | 0.8095 | 0.2990 | 137 / 3 / 29 / 68 |

The original machine-readable Phase 1 metric files were local-only run artifacts and are intentionally excluded from this repository by data-governance rules; they are therefore not linked here. The durable repository record of these historical study-level values is [baseline_results.md](baseline_results.md), together with the descriptive [error-analysis report](error_analysis.md) and this summary. No new evaluation was run to create this documentation clarification.

## Model A — frozen embeddings + Logistic Regression

ImageNet-pretrained ResNet-18 was used as a frozen feature extractor; image embeddings were mean-pooled at study level. Logistic Regression selected **C = 0.1** from `{0.1, 1.0, 10.0}` using validation AUROC only. Its validation-selected Youden-J threshold was **0.3542854529699628** (695 candidates). The final test result at that frozen threshold was AUROC 0.8756, AUPRC 0.8596, recall 0.8247, specificity 0.8071, F1 0.7843, and 17 false negatives.

## Model B — fine-tuned ResNet-18

The ImageNet-initialized ResNet-18 run used CUDA, seed 23, 224×224 preprocessing, auto class weighting (positive weight 1.6089), and validation AUROC monitoring. It ran **8 of 15 permitted epochs** and stopped early; the validation-best checkpoint was **epoch 4** with validation AUROC **0.9107**. Mean study-probability aggregation was selected on validation (AUROC 0.9107 versus 0.9018 for max). Its frozen Youden-J threshold was **0.3860229253768921** (695 candidates). At that threshold, its final test result was AUROC 0.8970, AUPRC 0.8937, recall 0.7320, specificity 0.9500, F1 0.8114, and 26 false negatives.

## Comparison and interpretation

Relative to Model A at each model's frozen validation threshold, Model B has higher point estimates for AUROC (+0.0214), AUPRC (+0.0341), specificity (+0.1429), and F1 (+0.0271). It has lower recall (−0.0928) and a higher false-negative rate (+0.0928): 26 rather than 17 false negatives. It also has 7 rather than 27 false positives.

Thus, fine-tuning improved the discrimination, specificity, and F1 point estimates in this one sealed-test evaluation, but it did not improve the false-negative outcome emphasized by the charter. The two per-model AUROC intervals overlap, and no paired difference confidence interval or hypothesis test was run. The evidence therefore does **not** support a claim of statistically significant improvement; it supports a measured operating-point trade-off that needs a pre-specified follow-up analysis.

## Descriptive false-negative and false-positive analysis

The generated [error-analysis report](error_analysis.md) indexes prediction outcomes, probabilities, and view counts only; it makes no clinical-causal claim.

- Model A: 17 false negatives and 27 false positives at its frozen threshold. The index contains 13 high predicted-class-score false negatives and 6 high predicted-class-score false positives; 2 false negatives and 3 false positives came from single-view studies.
- Model B: 26 false negatives and 7 false positives at its frozen threshold. The index contains 18 high predicted-class-score false negatives and 3 high predicted-class-score false positives; 2 false negatives and 1 false positive came from single-view studies.

These counts describe this test set and operating thresholds only. They do not establish why an abnormality was missed or whether either model would improve clinical outcomes.

## Compute and limitations

- The [actual-run environment report](environment.json) records CUDA with PyTorch 2.2.2 and a GTX 1660 Ti environment; it does not record a wall-clock runtime, so none is reported here.
- This is one controlled fine-tuning configuration and one frozen-feature baseline on a 237-study sealed test set. The result is not an external validation, calibration study, or deployment evaluation.
- MURA labels NORMAL/ABNORMAL studies, not fractures. No fracture diagnosis, localization, outcome improvement, or autonomous clinical decision claim is supported.
- Thresholds were optimized for validation Youden's J, not for a clinically justified sensitivity or false-negative operating target. The observed Model B false-negative trade-off should be addressed before treating its higher AUROC/AUPRC as a preferred triage configuration.
- Attention, ensembles, uncertainty modelling, Grad-CAM, RAG, Streamlit, and the agentic layer remain outside Phase 1.

## Continue decision

Phase 1 supports proceeding to a carefully pre-registered follow-up research stage, including a paired uncertainty comparison and a validation-defined operating point that explicitly addresses false-negative performance. It does not support clinical deployment or an autonomous/agentic clinical workflow. The Phase 1 experiment itself is complete; no further test-set tuning should be performed.
