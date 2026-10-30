# Multi-Anatomy Development Protocol

## Scope

This is a new development-only extension of the historical XR_WRIST project.
It covers the seven MURA anatomy categories: elbow, finger, forearm, hand,
humerus, shoulder, and wrist. The scientific task remains study-level
MURA-defined NORMAL versus ABNORMAL classification; it is not fracture
detection.

## Data governance

Only official MURA `train` is used. It is divided into internal training and
internal validation with a single global patient-level split. A patient who
appears in more than one anatomy remains on one side of the split. Study IDs
include anatomy, patient, and study directory so views from different body
parts cannot be pooled together.

Official MURA `valid` is not read by this extension. It was historically
opened for the wrist Phase-1 result, so it cannot be described as an untouched
advanced-model test set for later model selection.

## Baseline protocol

For each anatomy separately:

1. Use a frozen ImageNet-pretrained ResNet-18 to create a 512-dimensional
   embedding for every view.
2. Mean-pool the view embeddings to one study embedding.
3. Select Logistic Regression `C` from `{0.1, 1.0, 10.0}` by five-fold
   patient-grouped cross-validation inside internal training only.
4. Fit the selected Logistic Regression on the full internal-training split.
5. Select the highest internal-validation threshold reaching sensitivity at
   least 95% for that anatomy.
6. Report ranking metrics, threshold metrics, workload, and patient-cluster
   bootstrap confidence intervals on the internal validation set.

Every reported baseline result is development evidence only. No raw MURA
pixels, ZIP files, model weights, embeddings, or study predictions are
intended for Git.
