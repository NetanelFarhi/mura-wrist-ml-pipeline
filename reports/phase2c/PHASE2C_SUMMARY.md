# Phase 2C summary — controlled study-level partial fine-tuning

## Scope and integrity

This is a DEVELOPMENT/INTERNAL-VALIDATION result only. The script accepted official MURA `train` rows only and used the existing patient-safe internal split. Official MURA `valid` was not loaded for training, model choice, threshold choice, calibration, or evaluation. Because official `valid` was observed historically in Phase 1, this report does not make an untouched-final-test claim.

Population checked: 9752 images, 3460 studies, 3267 patients; 2769 internal train and 691 internal validation studies.

## Models compared

- **Frozen mean embedding reference:** the existing fine-tuned Phase-1 ResNet-18 emits 512-D view embeddings; the encoder is frozen and a linear study head follows a masked mean over real views.
- **Partial fine-tuned mean embedding:** same study-level mean-embedding architecture, initialized from the same Phase-1 checkpoint; only ResNet `layer4` and the new study head are trainable.
- The positive-class weight was computed from internal-train studies only: 1.619678.

## Validation comparison

| model | auroc | auprc | at_sensitivity_85_threshold | at_sensitivity_85_sensitivity | at_sensitivity_85_specificity | at_sensitivity_85_precision | at_sensitivity_85_f1 | at_sensitivity_85_false_positive | at_sensitivity_85_false_negative | at_sensitivity_85_flagged_percent | at_sensitivity_90_threshold | at_sensitivity_90_sensitivity | at_sensitivity_90_specificity | at_sensitivity_90_precision | at_sensitivity_90_f1 | at_sensitivity_90_false_positive | at_sensitivity_90_false_negative | at_sensitivity_90_flagged_percent | at_sensitivity_95_threshold | at_sensitivity_95_sensitivity | at_sensitivity_95_specificity | at_sensitivity_95_precision | at_sensitivity_95_f1 | at_sensitivity_95_false_positive | at_sensitivity_95_false_negative | at_sensitivity_95_flagged_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frozen mean embedding | 0.914983 | 0.878652 | 0.641280 | 0.851301 | 0.848341 | 0.781570 | 0.814947 | 64 | 40 | 42.402315 | 0.334643 | 0.903346 | 0.710900 | 0.665753 | 0.766562 | 122 | 26 | 52.821997 | 0.138672 | 0.951673 | 0.585308 | 0.593968 | 0.731429 | 175 | 13 | 62.373372 |
| Partial fine-tuned mean embedding | 0.913767 | 0.873423 | 0.530749 | 0.851301 | 0.831754 | 0.763333 | 0.804921 | 71 | 40 | 43.415340 | 0.359029 | 0.903346 | 0.725118 | 0.676880 | 0.773885 | 116 | 26 | 51.953690 | 0.172869 | 0.951673 | 0.571090 | 0.585812 | 0.725212 | 181 | 13 | 63.241679 |

## Primary >=95% sensitivity policy

Each threshold below is the highest threshold that meets the pre-specified sensitivity floor on the same internal-validation split. Therefore these are development operating estimates, not independent performance evidence.

| model | at_sensitivity_95_threshold | at_sensitivity_95_sensitivity | at_sensitivity_95_specificity | at_sensitivity_95_precision | at_sensitivity_95_f1 | at_sensitivity_95_false_positive | at_sensitivity_95_false_negative | at_sensitivity_95_flagged_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frozen mean embedding | 0.138672 | 0.951673 | 0.585308 | 0.593968 | 0.731429 | 175 | 13 | 62.373372 |
| Partial fine-tuned mean embedding | 0.172869 | 0.951673 | 0.571090 | 0.585812 | 0.725212 | 181 | 13 | 63.241679 |

## Paired uncertainty

Candidate minus frozen reference, using 2,000 patient-cluster bootstrap replicates on the same validation studies:

| metric | difference_candidate_minus_baseline | ci_lower_95 | ci_upper_95 | n_bootstrap_requested | n_bootstrap_valid | seed | resampling_unit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AUROC | -0.001216 | -0.004395 | 0.002044 | 2000 | 2000 | 42 | patient cluster |
| AUPRC | -0.005229 | -0.019358 | 0.009981 | 2000 | 2000 | 42 | patient cluster |
| Specificity@95% | -0.014218 | -0.076402 | 0.092423 | 2000 | 2000 | 42 | patient cluster |

## Interpretation

The 95% CI for the specificity difference includes zero; a point estimate alone is not enough to claim that partial fine-tuning improved the 95% sensitivity policy.

The aggregate paired-error counts are in `error_summary.csv`; view-count and disagreement subgroups are descriptive in `subgroup_summary.csv`, including the internal-train tercile cutpoints used for disagreement. The non-selection padding/border/resize diagnostic is in `geometry_robustness_summary.csv`. Neither analysis establishes clinical causation. The historical Phase-2B attention result remains negative/inconclusive and was not retrained here.

## Error, subgroup, and robustness evidence

At their own >=95% sensitivity policies, partial fine-tuning fixed 8 validation errors but introduced 14; it fixed 2 false negatives and introduced 2. Both models therefore had 13 false negatives, while partial fine-tuning added 6 false positives.
The high-disagreement subgroup was essentially tied (0.732357 frozen vs 0.732750 partial AUROC); this exploratory subgroup does not justify further representation work. The 1-view, 2-view, and 3+-view rows are descriptive because the 1-view subgroup contains only 6 abnormal studies.
The geometry diagnostic found non-zero prediction/decision changes under display transformations for both models. This supports continued QC/shortcut investigation, but it does not demonstrate the clinical source of those changes or validate localization.

## Complexity and stopping rule

Frozen reference: 513 trainable / 11,177,025 total parameters; best epoch 16; runtime 9.6 seconds.
Partial model: 8,394,241 trainable / 11,177,025 total parameters; best epoch 1; runtime 2092.9 seconds.
Only the pre-registered reference and one partial fine-tuning run were performed. No hyperparameter sweep, new backbone, attention retraining, ensemble, Agentic AI, or test-set evaluation occurred.

## Decision gate

Partial fine-tuning did **not** provide a meaningful improvement: at >=95% sensitivity, specificity was 0.571090 versus 0.585308 for the frozen reference, and the paired confidence interval includes no improvement. Its ranking point estimates were also lower and the training curve showed an overfitting signal.
The ML core should therefore remain frozen at the simpler frozen mean-embedding reference. The single justified next project step is to advance the bounded deterministic safety policy and separately evaluated Agentic-AI layer, while treating input-QC/geometry sensitivity as a safety-evaluation requirement rather than as proof of a medical shortcut.

## Final status

- **PHASE 2C STATUS:** COMPLETE
- **MEANINGFUL IMPROVEMENT:** NO
- **ML CORE READY TO FREEZE:** YES
- **RECOMMENDED NEXT STEP:** Implement and evaluate the bounded deterministic safety policy and Agentic-AI design on a dedicated agent-evaluation set; do not continue architecture tuning on this already-used development split.
