# Phase 2C protocol — frozen before execution

## Question and scope

Can joint, study-level adaptation of the top ResNet-18 block improve a simple
mean-embedding multi-view classifier over a frozen encoder reference? This is
a **development/validation-only** experiment. It does not test a new image
backbone, attention, an ensemble, calibration method, or Agentic AI.

The historical official MURA `valid` population was opened for the frozen
Phase-1 reference and is prohibited from every Phase-2C input, decision, and
report. The only eligible population is official MURA `train`, XR_WRIST,
using the existing patient-safe split: 2,769 internal-train studies and 691
internal-validation studies. The manifests must contain `source_split=train`
only and have no overlapping patients.

## Fixed reference and primary experiment

The Phase-1 fine-tuned ResNet-18 checkpoint is the common initialization.
Because the historical Phase-2B frozen-embedding artifacts used an earlier
2,768/692 split, this phase first regenerates one compatible frozen
mean-embedding reference on the current 2,769/691 split. It is a fixed
reference reproduction, not a tuning search.

| Run | Encoder | Study classifier | Training rule |
|---|---|---|---|
| Frozen mean reference | All ResNet-18 blocks frozen | masked mean of 512-D real-view embeddings, then `Linear(512,1)` | fixed head-only recipe: AdamW, lr `1e-3`, weight decay `1e-4`, max 40 epochs, AUROC early stopping patience 7, seed 42 |
| Primary partial fine-tuning | `conv1`, `bn1`, `layer1`–`layer3` frozen; only `layer4` trainable | same masked mean embedding and a new `Linear(512,1)` study head | AdamW: `layer4` lr `1e-5`, head lr `1e-4`, weight decay `1e-4`, max 15 epochs, AUROC early stopping patience 4, seed 42 |

Both runs use study-level `BCEWithLogitsLoss` with the positive class weight
computed from internal-train studies only, the same deterministic image
transform (no new augmentation), and study batches of two. This keeps the
comparison focused on the pre-specified study-level trainability/package rather
than an augmentation change. Padding is masked before encoding and pooling;
every study must retain at least one real view. Frozen BatchNorm modules remain
in evaluation mode during the partial run.

## Primary evaluation policy

The primary project operating target is sensitivity **>=95%**. For each model,
the threshold is the highest internal-validation score that achieves that
target. At this policy the report will state specificity, precision, F1,
false positives, false negatives, and the percentage flagged for review.

AUROC and AUPRC are ranking measures and are reported separately. The prior
85% and 90% sensitivity policies are retained as descriptive secondary rows,
not as alternative selection criteria. The comparison is a 2,000-replicate
paired patient-cluster bootstrap of partial fine-tuning minus frozen reference
for AUROC, AUPRC, and specificity at the 95% policy. Bootstrap resampling
estimates uncertainty only; it does not select a threshold or access test data.

## Pre-specified decision rule

Partial fine-tuning is considered promising only if the paired evidence is
consistent with a practically useful specificity gain at >=95% sensitivity,
without a material loss of AUROC/AUPRC or a worse safety/robustness profile.
A small point-estimate change whose confidence interval crosses zero is
reported as effectively similar or inconclusive. If the primary run is valid
but does not meet this rule, no retry, hyperparameter sweep, or further
unfreezing will be performed in this phase.

## Additional descriptive checks

The report will show paired error counts, study view-count groups (1, 2, 3+),
and original view-probability disagreement groups whose boundaries are defined
from internal-train only. It will also rerun the existing padding/border/resize
diagnostic as a non-selection check. These analyses cannot establish clinical
causation or lesion localization.

All detailed predictions and checkpoints stay in the local ignored experiment
directory. Git receives aggregate tables, bootstrap summaries, training curves,
and code only; no MURA image pixels, patient/study IDs, paths, or raw model
weights are committed. Complexity reporting includes the shared ResNet encoder
for both models, even when it is frozen.
