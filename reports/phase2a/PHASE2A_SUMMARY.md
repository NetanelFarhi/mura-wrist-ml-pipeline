# Phase 2A summary — validation-only operating policy and calibration

## Scope and guardrails

Phase 2A used the already-trained Phase 1 fine-tuned ResNet-18 only. No image backbone was retrained, no attention/learned pooling, ensemble, Grad-CAM, Agentic AI, or Streamlit component was built. The opened Phase 1 test predictions were not read by the Phase 2A analysis scripts; all policy/calibration/error/disagreement results below use Model B’s saved 692-study validation predictions only.

The Phase 1.5 artifacts were verified before analysis: Model B validation AUROC is 0.910715, AUPRC is 0.884109, and the historical validation Youden-J threshold is 0.3860229253768921 (sensitivity 0.800000, specificity 0.878220). The exact sensitivity-constrained thresholds below reproduce the stored Phase 1.5 values.

## 1. Is the Phase 1 sensitivity problem largely an operating-threshold problem?

On the saved validation set, **substantially yes**: the same Model B ranking can be operated at 0.853 sensitivity or 0.902 sensitivity simply by lowering the decision threshold. This is a validation operating-policy finding, not evidence that ranking errors are absent or that the result will transfer to another cohort. It trades specificity and review workload for fewer false negatives.

## 2–3. Pre-registered operating policies and workload

Policies were fixed in `PHASE2A_PROTOCOL.md` before the new analysis. P2 and P3 choose the highest threshold that meets the pre-registered validation sensitivity floor. “Flagged” means predicted ABNORMAL/referred for review.

| Policy | Threshold | Sensitivity | Specificity | Precision | F1 | FN | FP | Flagged studies |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 historical Youden-J | 0.386023 | 0.800000 | 0.878220 | 0.803030 | 0.801512 | 53 | 52 | 264 / 692 (38.15%) |
| P2 sensitivity >=0.85 | 0.280313 | 0.852830 | 0.798595 | 0.724359 | 0.783362 | 39 | 86 | 312 / 692 (45.09%) |
| P3 sensitivity >=0.90 | 0.160984 | 0.901887 | 0.714286 | 0.662050 | 0.763578 | 26 | 122 | 361 / 692 (52.17%) |

Relative to P1, P2 resolves 14 validation false negatives and introduces 34 false positives. Moving P2 to P3 resolves a further 13 false negatives and introduces 36 further false positives. Thus P3 resolves 27 P1 false negatives at the cost of 70 additional normal studies flagged. This is a workload tradeoff, not a claim that any policy is clinically sufficient.

The 95% stratified study-bootstrap intervals (2,000 replicates, seed 23) show material sampling uncertainty:

| Policy | Sensitivity 95% CI | Specificity 95% CI |
|---|---:|---:|
| P2 sensitivity >=0.85 | 0.808–0.894 | 0.759–0.836 |
| P3 sensitivity >=0.90 | 0.864–0.940 | 0.670–0.756 |

See `operating_points.csv`, `operating_point_analysis.md`, `validation_error_transitions.csv`, and `workload_tradeoff.png`.

## 4–5. Calibration and temperature scaling

Raw sigmoid probabilities are model scores; they should not be called clinical “confidence.” The validation-only uncalibrated audit is:

| Version | AUROC | AUPRC | Brier | NLL | ECE (10 bins) |
|---|---:|---:|---:|---:|---:|
| Uncalibrated | 0.910715 | 0.884109 | 0.109157 | 0.373411 | 0.048050 |
| Temperature-scaled, 5-fold cross-fitted | 0.909275 | 0.882161 | 0.109844 | 0.361675 | 0.039236 |

Temperature was fit only on the other four validation folds for every held-out validation fold (temperatures 1.298–1.476). Therefore the reported scaled calibration metrics are cross-fitted rather than in-sample fit metrics. NLL and ECE improve modestly, while Brier is slightly worse; there is **no material, consistent calibration improvement** across all three measures. The tiny AUROC/AUPRC changes reflect cross-fitted fold-specific monotonic transformations and are not presented as ranking improvement. See `calibration_before.md` and `calibration_before.png`.

## 6. Cross-validation robustness

No CNN cross-validation was performed. A scientifically valid patient-level out-of-fold assessment would require retraining the fixed Phase 1 recipe separately in each patient-grouped fold; that is outside the Phase 2A no-retraining scope. In-sample train predictions are not used as a substitute for that experiment. The reported bootstrap CIs quantify uncertainty conditional on this one validation split, but they do **not** establish cross-split or external-cohort stability of the thresholds.

## 7. Validation error characterization

All errors are study-level validation outcomes under the policies above. Lowering the threshold makes 14 P1 false-negative studies positive by P2, and a further 13 by P3. The corresponding increased review load comes entirely from additional normal studies becoming positive (34 then 36). No medical explanation of individual errors was inferred.

## 8. Per-view predictions and disagreement

The frozen existing checkpoint was run in deterministic evaluation mode on development images only. Files saved:

- `models/resnet18/per_view_predictions_train.csv` — 7,805 real views.
- `models/resnet18/per_view_predictions_validation.csv` — 1,947 real views.
- `models/resnet18/train_view_probability_summary.csv` and `models/resnet18/val_view_probability_summary.csv` — mean, max, min, standard deviation, and range per study.

The validation recomputed mean view probabilities agree with saved Phase 1 study probabilities to a maximum absolute difference of `8.01e-06`; this is expected floating-point variation. Each export was checked to have exactly the manifest-declared number of views per study, with an actual image path for every row; padded views cannot appear as predictions.

At P1, correct validation study predictions have median view-probability standard deviation 0.026919 and median range 0.047632; incorrect predictions have 0.338128 and 0.622818 respectively. P1 false negatives have median SD 0.157535/range 0.310346 versus true positives 0.027509/0.049343. At P3, false negatives (0.045454/0.087246) are close to true positives (0.045396/0.083003). These are descriptive associations on one validation split, not uncertainty estimates or causal evidence.

The large P1 correct-vs-incorrect difference makes learned multi-view modelling a **justified Phase 2B research question**, but not an established solution. It should be tested in a separately specified experiment with patient-safe development/evaluation and without revisiting the opened Phase 1 test set for selection.

## 9. What remains unverified

- Generalization of P2/P3 to a fresh cohort or an untouched final test set.
- Patient-grouped cross-validation stability of the fixed Model B training recipe.
- Clinical appropriateness of 85% or 90% sensitivity and the resulting workload.
- Whether temperature scaling will improve calibration in a new cohort.
- Whether view disagreement improves decision-making or learned pooling; it is only associated with errors here.
- Any medical cause of false positives/false negatives.

## Testing

Added Phase 2A regression tests for rejecting test-prediction paths, sensitivity-constrained threshold selection, identity preservation, duplicate-study rejection, and rejection of padded/nonexistent views. The full suite passed: **22 passed**. These are software/integrity tests, not ML-performance evidence.

## Final status

PHASE 2A STATUS:
Complete. Validation-only analysis and development per-view export complete; no new model training or test-set selection performed.

ORIGINAL MODEL:
Validation AUROC: 0.910715
Validation AUPRC: 0.884109

85% SENSITIVITY POLICY:
Threshold: 0.280313
Sensitivity: 0.852830
Specificity: 0.798595
Precision: 0.724359
Flagged workload: 312 / 692 (45.09%)

90% SENSITIVITY POLICY:
Threshold: 0.160984
Sensitivity: 0.901887
Specificity: 0.714286
Precision: 0.662050
Flagged workload: 361 / 692 (52.17%)

CALIBRATION:
Brier before: 0.109157
Brier after: 0.109844
ECE before: 0.048050
ECE after: 0.039236

VIEW DISAGREEMENT:
On validation, incorrect P1 predictions show substantially larger median cross-view SD/range than correct predictions (0.338128/0.622818 vs 0.026919/0.047632). Exploratory association only.

MOST IMPORTANT FINDING:
The Model B sensitivity issue is materially adjustable through a validation-only operating threshold, but reaching 90% sensitivity flags just over half of validation studies and adds 70 false positives versus the historical Youden-J policy.

RECOMMENDED PHASE 2B:
Pre-register a patient-safe learned multi-view aggregation experiment, retaining the fixed operating-policy comparison and using a fresh untouched evaluation cohort. The observed disagreement signal justifies testing it; it does not justify assuming it will improve performance.
