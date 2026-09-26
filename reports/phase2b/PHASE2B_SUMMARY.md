# Phase 2B summary — frozen-encoder multi-view development experiment

## Result

On the existing patient-safe validation split only, gated attention did **not** outperform the simpler embedding-level baselines. This is a negative but useful result: with the Phase 1 ResNet-18 encoder frozen, learned attention over 512-D view embeddings did not improve AUROC, AUPRC, or specificity at the fixed high-sensitivity policies.

| Model | AUROC | AUPRC | Specificity @ >=85% sensitivity | Specificity @ >=90% sensitivity |
|---|---:|---:|---:|---:|
| Mean embedding | 0.917061 | 0.897481 | 0.822014 | 0.714286 |
| Max embedding | 0.916689 | 0.899164 | 0.817330 | 0.707260 |
| Gated attention | 0.916177 | 0.894082 | 0.812646 | 0.723653 |

At threshold 0.5, mean/max/attention had FN counts 45/46/48 respectively. Thus attention fixed no net false negatives relative to mean and introduced at least three net additional FNs at that shared reference threshold. It did have a slightly higher specificity at >=90% sensitivity than the simple embedding baselines, but it was lower at >=85%, lower in AUROC/AUPRC, and its isolated point estimate is not evidence that attention wins.

## High-disagreement and view-count evidence

Validation studies were split into equal-count thirds by the Phase 2A real-view probability range: low (231), medium (230), high (231). Attention AUROC versus mean embedding AUROC was 0.971845 vs 0.971929, 0.908758 vs 0.909555, and 0.767991 vs 0.774206 respectively. Attention therefore did not help the high-disagreement group in this frozen-encoder experiment.

Attention weights are saved per real view in `models/phase2b/attention_weights_validation.csv`; padding is masked before softmax and no padded item is exported. The attention analysis is descriptive only and is not lesion localization.

## Complexity

Mean/max heads each have 513 trainable parameters and trained in approximately 3.9/5.6 CPU seconds after embedding caching. Gated attention has 131,969 trainable pooling/head parameters and trained in approximately 9.9 CPU seconds. The shared ResNet-18 encoder was initialized from the Phase 1 checkpoint and frozen; its parameters are not included in these trainable-head counts. The added attention complexity is not justified by the observed validation results.

## Integrity and limitations

The official MURA `valid` set previously opened as Phase 1 test was not loaded by the extraction, training, or evaluation scripts. All results are DEVELOPMENT/VALIDATION results. No new image backbone, ensemble, Agentic AI, or UI was built. The bounded protocol allowed exactly mean embedding, max embedding, and frozen gated attention; no partial/end-to-end fine-tuning was run.

The original Phase-2B run did not complete paired inference. A subsequent documented closeout saved the 2,000-replicate paired validation bootstrap in [paired_comparison.md](paired_comparison.md): every confidence interval for AUROC, AUPRC, specificity at >=85% sensitivity, and specificity at >=90% sensitivity crosses zero. It therefore strengthens the original negative conclusion and does not establish statistical superiority for any pooling method. The requested comprehensive view-count metrics and all attention-specific regression assertions remain incomplete; this is still partial work.

## Answer to the research question

The current evidence does not show that learned gated attention exploits multiple views better than simple embedding pooling. The paired patient-bootstrap is now complete and compatible with no difference for every requested comparison. Do not move immediately to a larger architecture; first close the remaining attention-masking/view-count test gaps, then decide whether a controlled partial encoder fine-tuning experiment is warranted. A fresh untouched evaluation cohort is required before any generalization or clinical-performance claim.

PHASE 2B STATUS:
PARTIAL

MEAN EMBEDDING:
AUROC: 0.917061
AUPRC: 0.897481
Specificity @ >=85% sensitivity: 0.822014
Specificity @ >=90% sensitivity: 0.714286

MAX EMBEDDING:
AUROC: 0.916689
AUPRC: 0.899164
Specificity @ >=85% sensitivity: 0.817330
Specificity @ >=90% sensitivity: 0.707260

GATED ATTENTION:
AUROC: 0.916177
AUPRC: 0.894082
Specificity @ >=85% sensitivity: 0.812646
Specificity @ >=90% sensitivity: 0.723653

PAIRED IMPROVEMENT:
Not demonstrated; the saved 2,000-replicate paired-bootstrap CIs for every requested comparison cross zero. See `paired_comparison.md`.

FALSE NEGATIVES FIXED:
No net FN reduction versus mean embedding at threshold 0.5 (attention 48 vs mean 45).

NEW FALSE NEGATIVES:
At least 3 net additional versus mean embedding at threshold 0.5.

HIGH-DISAGREEMENT RESULT:
Attention AUROC 0.767991 vs mean 0.774206; no observed benefit.

ATTENTION JUSTIFIED:
NO

MOST IMPORTANT FINDING:
With the encoder frozen, attention adds complexity without a consistent validation benefit over simple embedding pooling.

RECOMMENDED NEXT STEP:
Close the remaining attention-specific test/view-count gaps, then review whether controlled partial encoder fine-tuning is scientifically warranted; do not use the opened Phase 1 test set.
