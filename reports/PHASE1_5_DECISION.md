# What Phase 1 showed

Phase 1 remains unchanged.  All threshold selection in this Phase 1.5
analysis uses only the saved validation predictions (692 studies: 265
abnormal, 427 normal).  The saved test predictions are used only for
descriptive error and view-count analysis at the already frozen Phase 1
thresholds; they were not used to choose or revise a threshold.

| Model | Validation AUROC | Validation AUPRC | Frozen Phase 1 test AUROC | Frozen Phase 1 test AUPRC | Test sensitivity | Test specificity | Test FN / FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Model A — frozen ResNet-18 embeddings + logistic regression | 0.864 | 0.814 | 0.876 | 0.860 | 0.825 | 0.807 | 17 / 27 |
| Model B — fine-tuned ResNet-18 | 0.911 | 0.884 | 0.897 | 0.894 | 0.732 | 0.950 | 26 / 7 |

The reconstructed validation ROC and precision-recall curves are in
[PHASE1_5_KEY_FIGURE.png](team_share/PHASE1_5_KEY_FIGURE.png).  The
underlying curve points are in
[phase1_5_validation_curve_points.csv](phase1_5_validation_curve_points.csv).
On validation, Model B improves on Model A by 0.046 AUROC and 0.070 AUPRC.
That is a ranking result, not a guarantee of fewer false negatives at every
threshold.

# Threshold tradeoff

The following is a validation-only threshold table.  An asterisk marks the
existing frozen Phase 1 threshold; it is shown for comparison, not changed.
FN and FP are validation-study counts.  The full 0.1--0.9 grid is saved in
[phase1_5_validation_threshold_table.csv](phase1_5_validation_threshold_table.csv).

| Model | Threshold | Sensitivity | Specificity | Precision | F1 | FN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.10 | 0.928 | 0.482 | 0.527 | 0.672 | 19 | 221 |
| A | 0.20 | 0.887 | 0.604 | 0.582 | 0.703 | 30 | 169 |
| A | 0.30 | 0.849 | 0.703 | 0.639 | 0.729 | 40 | 127 |
| A | 0.3543* | 0.830 | 0.752 | 0.675 | 0.745 | 45 | 106 |
| A | 0.40 | 0.792 | 0.782 | 0.693 | 0.739 | 55 | 93 |
| A | 0.50 | 0.713 | 0.845 | 0.741 | 0.727 | 76 | 66 |
| A | 0.60 | 0.634 | 0.904 | 0.804 | 0.709 | 97 | 41 |
| A | 0.80 | 0.498 | 0.960 | 0.886 | 0.638 | 133 | 17 |
| B | 0.10 | 0.917 | 0.644 | 0.615 | 0.736 | 22 | 152 |
| B | 0.20 | 0.887 | 0.745 | 0.683 | 0.772 | 30 | 109 |
| B | 0.30 | 0.838 | 0.817 | 0.740 | 0.786 | 43 | 78 |
| B | 0.3860* | 0.800 | 0.878 | 0.803 | 0.802 | 53 | 52 |
| B | 0.40 | 0.781 | 0.883 | 0.805 | 0.793 | 58 | 50 |
| B | 0.50 | 0.725 | 0.939 | 0.881 | 0.795 | 73 | 26 |
| B | 0.60 | 0.642 | 0.960 | 0.909 | 0.752 | 95 | 17 |
| B | 0.80 | 0.581 | 0.984 | 0.957 | 0.723 | 111 | 7 |

Model B can achieve sensitivity >= 0.85 on validation.  The saved
validation threshold with the highest specificity among those meeting that
floor is 0.2803: sensitivity 0.853 (226/265), specificity 0.799
(341/427), precision 0.724, F1 0.783, FN 39, and FP 86.

Sensitivity >= 0.90 is also reachable on this validation split, at threshold
0.1610: sensitivity 0.902 (239/265), specificity 0.714 (305/427),
precision 0.662, F1 0.764, FN 26, and FP 122.  It costs 16.4 percentage
points of specificity and 70 additional FPs relative to Model B's frozen
validation operating point (0.878 specificity, 52 FPs).  It is therefore a
real validation tradeoff, not a free improvement, and its generalization
would need a pre-specified, untouched evaluation set.

# Matched operating point comparison

Comparing frozen thresholds alone is misleading because they choose different
sensitivity/specificity tradeoffs.  The table below independently selects, on
validation only, the highest-specificity threshold that reaches each
sensitivity floor.  Both models have the same achieved number of TPs at each
row.

| Sensitivity floor | Model A: threshold / specificity / FP | Model B: threshold / specificity / FP | Model B specificity advantage |
|---|---|---|---:|
| >= 0.80 | 0.3972 / 0.778 / 95 | 0.3860 / 0.878 / 52 | +10.1 pp |
| >= 0.85 | 0.2947 / 0.696 / 130 | 0.2803 / 0.799 / 86 | +10.3 pp |
| >= 0.90 | 0.1649 / 0.569 / 184 | 0.1610 / 0.714 / 122 | +14.5 pp |

The inverse comparison uses the highest-sensitivity validation threshold
that reaches each specificity floor.  Exact equality is not always possible
because scores are discrete; realized specificity is shown.

| Specificity floor | Model A: sensitivity / realized specificity / threshold | Model B: sensitivity / realized specificity / threshold | Model B sensitivity advantage |
|---|---|---|---:|
| >= 0.80 | 0.770 / 0.803 / 0.4451 | 0.849 / 0.801 / 0.2812 | +7.9 pp |
| >= 0.85 | 0.709 / 0.857 / 0.5145 | 0.815 / 0.855 / 0.3469 | +10.6 pp |
| >= 0.90 | 0.645 / 0.902 / 0.5963 | 0.755 / 0.906 / 0.4344 | +10.9 pp |
| >= 0.95 | 0.528 / 0.956 / 0.7700 | 0.698 / 0.951 / 0.5178 | +17.0 pp |

Model B is stronger in these validation-only matched comparisons.  Its lower
test sensitivity at the frozen threshold is therefore mainly an
operating-threshold issue: its selected point already favored specificity
(validation sensitivity/specificity 0.800/0.878) over recall.  This is not
an assertion that all Model B FNs are threshold-near or that ranking errors
do not exist.  The saved test error set has 14 FNs shared by both models and
12 Model-B-only FNs, so representation/ranking errors remain for some
studies.

All exact matched rows, including counts and F1, are in
[phase1_5_matched_operating_points.csv](phase1_5_matched_operating_points.csv).

# Error patterns

The complete saved-test error table is
[phase1_5_error_comparison.csv](phase1_5_error_comparison.csv).  It contains
all 59 studies misclassified by at least one model at the frozen Phase 1
thresholds, with study_id, true label, both probabilities, number of views,
decision agreement, signed margin (probability minus threshold), absolute
margin, and error role.  Absolute margin is a model-output distance from the
threshold, not a calibration or clinical-confidence claim.

| Frozen-test error relationship | Count |
|---|---:|
| Model A FN / FP | 17 / 27 |
| Model B FN / FP | 26 / 7 |
| Shared FNs / shared FPs | 14 / 4 |
| Model-A-only FNs / Model-B-only FNs | 3 / 12 |
| Model-A-only FPs / Model-B-only FPs | 23 / 3 |
| Agreement on predicted label | 196 / 237 studies (82.7%) |

Observable score and threshold-margin summaries are below.  These are
descriptive medians from the saved test errors only.

| Error type | Count | Median probability | Median absolute margin | Errors in 1 / 2 / 3+ views |
|---|---:|---:|---:|---|
| Model A FN | 17 | 0.080 | 0.270 | 2 / 3 / 12 |
| Model A FP | 27 | 0.490 | 0.130 | 3 / 5 / 19 |
| Model B FN | 26 | 0.110 | 0.280 | 2 / 3 / 21 |
| Model B FP | 7 | 0.860 | 0.480 | 1 / 2 / 4 |

Model B exchanges 20 fewer FPs for 9 more FNs at its frozen point.  Its
additional FNs are not all threshold-edge cases by this raw margin summary.
The 3+ view group contains 21 of Model B's 26 FNs versus 12 of Model A's 17,
but it also contains most test positives (81/97) and most studies.  No
medical or causal explanation is inferred from these records.

# Multi-view evidence

This is a descriptive saved-test analysis at the frozen validation-selected
thresholds.  Metrics are reported only where each class has at least 10
studies.  AUROC is threshold-independent; sensitivity and specificity use
the frozen thresholds.

| Views per study | Studies (normal / abnormal) | Model A: AUROC / sensitivity / specificity | Model B: AUROC / sensitivity / specificity |
|---|---:|---|---|
| 1 | 24 (22 / 2) | Not reported: only 2 abnormal studies | Not reported: only 2 abnormal studies |
| 2 | 44 (30 / 14) | 0.848 / 0.786 / 0.833 | 0.948 / 0.786 / 0.933 |
| 3+ | 169 (88 / 81) | 0.890 / 0.852 / 0.784 | 0.884 / 0.741 / 0.955 |

The full table, including support rules and error counts, is in
[phase1_5_view_stratified_performance.csv](phase1_5_view_stratified_performance.csv).
The two-view and 3+ view results differ, but the two-view group is small
(14 abnormal studies), the one-view group is not reportable, and these
groups may differ in more than view count.  The current saved Model B score
already uses mean aggregation; no saved comparison evaluates learned
multi-view aggregation.  Multi-view learning is a reasonable later research
question, but the present evidence is not strong enough to make it the next
experiment.

# Recommended next ML experiment

Run one validation-only, sensitivity-constrained operating-point experiment
for the already trained Model B.  Keep the backbone and aggregation fixed.
Before looking at a new held-out evaluation set, pre-specify the sensitivity
target (for example >= 0.85), choose the highest-specificity threshold
satisfying it using validation or nested validation folds, and include a
reliability/calibration diagnostic.  Do not re-select a threshold on the
completed Phase 1 test set or overwrite its result.

RECOMMENDED NEXT EXPERIMENT:

A. Threshold/calibration improvement for Model B, with a pre-specified
validation-only sensitivity target.

WHY:

Model B has the best validation ranking and is superior at matched sensitivity
and matched specificity.  It reaches 0.853 sensitivity with 0.799
specificity on validation, whereas Model A reaches the same sensitivity with
0.696 specificity.  This directly addresses the observed frozen-threshold
FN tradeoff without making an unsupported architecture claim.

EXPECTED BENEFIT:

Convert more of Model B's discrimination advantage into a sensitivity level
that is explicitly chosen for the intended operating policy while preserving
more specificity than Model A at the same validation sensitivity.

MAIN RISK:

A threshold optimized on one validation split may not transfer to a future
cohort; calibration and score distributions can shift.  Mitigate this with a
pre-specified target, nested validation or a fresh holdout, and one
untouched final evaluation.

EVIDENCE USED:

Saved Phase 1 validation predictions and frozen selection records; saved
Phase 1 test predictions only for descriptive errors and view counts; the
reconstructed validation ROC/PR curves and the validation-only matched
operating-point analysis in this directory.  No model was retrained, no
test threshold was tuned, and no saved Phase 1 test result was changed.
