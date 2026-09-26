# Phase 2B protocol — frozen before training

## Hypothesis and scope

**H1.** A learned gated-attention study model over 512-dimensional view embeddings from the shared, frozen Phase 1 fine-tuned ResNet-18 encoder will improve study-level discrimination and/or the sensitivity-specificity trade-off over simple aggregation of independent views.

Primary comparison: gated attention versus **mean embedding pooling**. Secondary comparison: gated attention versus **max embedding pooling**. Phase 1 probability mean/max results are preserved as historical reference baselines but are not retrained.

This is a development experiment. It uses only the official MURA training population in the existing patient-safe final `train` (2,768 studies / 2,614 patients) and `val` (692 studies / 653 patients) partitions. The official MURA `valid` population, already opened as Phase 1 test, is prohibited from loading, fitting, selection, or evaluation in every Phase 2B script.

## Verified starting evidence

MURA v1.1 `XR_WRIST`: 3,474 patients, 3,697 studies, 10,411 images, 2,274 NORMAL and 1,423 ABNORMAL studies. View counts are 1: 300, 2: 705, 3: 2,138, 4: 509, and 5+: 45. Phase 1 Model B validation mean-probability AUROC/AUPRC are 0.910715/0.884109; saved max-probability AUROC is 0.901754. Phase 2A found a descriptive association between view-score disagreement and validation errors; this motivates but does not establish H1.

## Architecture and fixed training recipe

- Shared encoder: Phase 1 `best_resnet18_study_level.pt`; scalar `fc` removed to emit 512-D embedding/view. Encoder parameters frozen.
- Simple baselines: masked mean or masked coordinate-wise max embedding; single `Linear(512, 1)` study head.
- Attention: gated MIL pooling with `tanh(Vh) ⊙ sigmoid(Uh)`, scalar score, real-view-only softmax, weighted 512-D study embedding, and a single `Linear(512,1)` head. Attention hidden dimension: 128.
- Loss: study-level `BCEWithLogitsLoss(pos_weight=1.6088595664)`, matching the Phase 1 training-study class-ratio policy.
- Optimizer: AdamW, lr `1e-3`, weight decay `1e-4` for the new pooling/head parameters only.
- Scheduler: ReduceLROnPlateau (maximize validation AUROC; factor 0.5; patience 3; min lr 1e-6).
- Batch size: 32 studies of cached frozen embeddings.
- Maximum epochs: 40. Early stopping: validation AUROC, patience 7, min delta 0.0. Seed: 23 with deterministic PyTorch settings.

## Bounded experiment set and criteria

Exactly three main experiments: (1) mean embedding, (2) max embedding, (3) gated attention. No partial/end-to-end encoder fine-tuning is permitted in this Phase 2B run. The primary metric is validation AUROC; secondary metrics are AUPRC, validation metrics at the pre-registered historical 0.5 decision threshold, and highest specificity under validation sensitivity floors >=0.85 and >=0.90.

Attention is **promising**, not proven, only if paired validation uncertainty credibly supports at least one pre-registered benefit: AUROC/AUPRC beyond the strongest simple embedding baseline; higher specificity under one sensitivity constraint; fewer FNs at comparable workload; or a reproducible high-disagreement subgroup benefit. A tiny point-estimate difference whose paired CI includes no difference is a negative/inconclusive result. Stop after these three experiments and report negative results equally.

## Evaluation and integrity

All model/threshold selection and paired bootstrap comparisons use validation predictions only. Sensitivity-constrained thresholds are chosen as the highest validation threshold achieving the fixed target. Paired, patient-cluster bootstrap uses the same validation patients for every model. Attention weights are descriptive model internals, not lesion localization or medical explanation. Padded views must have zero attention and cannot enter a saved attention record.
