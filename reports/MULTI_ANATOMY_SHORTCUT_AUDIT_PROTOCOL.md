# Multi-Anatomy Peripheral-Pixel Sensitivity Audit Protocol

## Purpose

This is a diagnostic check of the fine-tuned models, not a new model-selection
experiment. It evaluates whether pre-specified changes to synthetic padding,
the outer image border, or the four corners materially alter study-level
predictions on internal validation.

## Fixed population and safeguards

- Use only the seven official-MURA-train-derived internal-validation manifests.
- Never read official MURA `valid`, test labels, or a new evaluation split.
- Reuse each saved fine-tuned checkpoint and its already selected internal-
  validation 95% sensitivity threshold; do not retrain or choose a new
  threshold.
- Before an intervention is run, reproduce the saved canonical validation
  predictions from the checkpoint and exact deterministic preprocessing. Stop
  if the maximum absolute difference exceeds `1e-5`.
- Verify the checkpoint SHA-256 before and after the audit.

## Pre-specified interventions

1. **padding_midgray:** only synthetic square-padding pixels change from black
   to mid-gray before the normal resize/normalization path.
2. **outer_border_black_5pct:** set the outer 5% of the final 224×224 image to
   black.
3. **four_corner_mask_black_10pct:** set each 10%-sized corner of the final
   image to black.

The model output remains mean pooled across the real views of a study.

## Reported quantities

For every anatomy/intervention report mean/median/95th-percentile absolute
probability change, Spearman rank correlation with canonical predictions,
decision-change count and percent at the frozen threshold, newly introduced
false negatives/positives, and patient-cluster bootstrap 95% intervals for
mean probability change and decision-change percent.

## Interpretation boundary

This can measure sensitivity to peripheral pixels. It cannot identify a
specific L/R marker, prove shortcut learning, prove clinical localization, or
prove that a model is free of shortcut learning. Border/corner masks can also
remove anatomy, collimation, or acquisition content. No raw images or
study-identifying rows are included in the Git-safe report.
