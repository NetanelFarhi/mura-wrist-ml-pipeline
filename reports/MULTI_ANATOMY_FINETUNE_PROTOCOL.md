# Multi-Anatomy Fine-Tuning Protocol

## Question

Does a study-level fine-tuned ResNet-18 improve over the frozen embedding
baseline for each MURA anatomy under the same internal-validation protocol?

## Frozen design

One model is trained independently for each anatomy. Each real view passes
through an ImageNet-initialized ResNet-18; the view abnormality probabilities
are mean-pooled to one study probability. This is mean probability pooling,
not learned attention or embedding pooling.

Training uses AdamW (`1e-4`, weight decay `1e-4`), batch size 4, up to 15
epochs, automatic class weighting from internal-training studies only, and
validation-AUROC early stopping with patience 4. The seed is 42. No search of
backbones, learning rates, or pooling rules is permitted in this run.

## Selection and reporting

All model selection occurs on the anatomy-specific internal validation split.
The operating threshold is the highest validation threshold achieving at
least 95% sensitivity. Each anatomy gets its own threshold. Report AUROC,
AUPRC, sensitivity, specificity, precision, F1, errors, workload, and
patient-cluster bootstrap intervals.

Official MURA `valid` is not used. The historical wrist official-valid result
remains preserved separately and is not an untouched advanced-model test set.
