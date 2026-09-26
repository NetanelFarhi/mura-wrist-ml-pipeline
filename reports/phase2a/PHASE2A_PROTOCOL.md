# Phase 2A protocol — fixed before analysis

## Scope

Phase 2A evaluates operating policies and calibration for the already-trained Phase 1 Model B (fine-tuned ImageNet ResNet-18 with mean study-probability pooling). It does not retrain an image model, change the backbone, implement learned multi-view pooling/attention, build an ensemble, or create an agent/UI.

The previously opened Phase 1 official-MURA-valid test set is excluded from every Phase 2A fitting, threshold-selection, calibration-selection, calibration-metric, error, and view-disagreement analysis. It is not read by the Phase 2A analysis scripts.

## Verified starting evidence

The saved Phase 1 Model B validation artifacts record AUROC 0.910715 and AUPRC 0.884109. The historical Youden-J threshold is 0.3860229253768921, with validation sensitivity 0.800000 and specificity 0.878220. The saved Phase 1.5 validation-only analysis reports approximately 0.2803 for sensitivity >=0.85 (specificity 0.799) and 0.1610 for sensitivity >=0.90 (specificity 0.714). Phase 2A re-derives the exact frozen thresholds from the saved validation predictions only.

## Pre-registered policies

1. **P1 historical Youden-J:** threshold `0.3860229253768921`; retained solely as the Phase 1 reference.
2. **P2 sensitivity >=0.85:** choose the highest validation threshold meeting sensitivity >=0.85.
3. **P3 sensitivity >=0.90:** choose the highest validation threshold meeting sensitivity >=0.90.

The two sensitivity floors are fixed before the analysis. False negatives are a material project outcome, so policy selection uses a sensitivity constraint and then maximizes specificity (equivalently chooses the highest feasible threshold), rather than optimizing Youden J again.

## Calibration protocol

Uncalibrated calibration is measured on all saved validation predictions. Temperature scaling is fitted and evaluated by deterministic five-fold stratified cross-fitting inside the validation set: each held-out fold receives a temperature fitted only on the other four folds. Thus the cross-fitted calibration metrics are not presented as in-sample fit metrics. Ranking metrics are reported for transparency, but temperature scaling is monotonic and is not claimed to improve ranking.

## Uncertainty and robustness protocol

Sensitivity and specificity confidence intervals use stratified study-level bootstrap resampling of the validation predictions (2,000 replicates; seed 23). This quantifies sampling uncertainty on the one existing validation split. No CNN cross-validation/retraining is performed: a valid out-of-fold model-performance cross-validation would require retraining the fixed Phase 1 recipe per patient-grouped fold, which is outside this no-retraining Phase 2A scope. No in-sample train predictions are used to make a generalization claim.

## Per-view and disagreement protocol

The frozen existing Model B checkpoint is run in deterministic evaluation mode on only final `train` and `val` studies to persist real-view probabilities. Padded positions are excluded using the view mask. Per-view results are descriptive only; they are not used to fit a learned pooling model or alter a decision policy. View-disagreement analyses are restricted to validation predictions, where study-level correctness is available without reusing the opened test set.
