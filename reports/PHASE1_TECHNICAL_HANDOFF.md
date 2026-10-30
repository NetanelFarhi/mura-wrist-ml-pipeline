# Phase 1 technical handoff — MURA XR_WRIST

> **Historical local-artifact handoff:** this document records what was found
> during the original Phase 1 handoff. Several paths named below were local
> run artifacts at that time and are now deliberately excluded from Git under
> the project's data-governance rules. Their absence from the public repository
> does not change the historical values; it means they cannot be re-opened from
> Git. Read this together with [result lineage](../docs/RESULT_LINEAGE.md) and
> [test-set governance](TEST_SET_GOVERNANCE.md). It is not the current
> Checkpoint 1 result source.

## Audit boundary

This is a read-only handoff of artifacts already present in this repository. No model was retrained, no new experiment or threshold search was run, and no Phase 2 component was started while preparing this document. “NOT VERIFIED” means no persisted artifact establishes the requested fact.

The task is **study-level NORMAL/ABNORMAL classification** for MURA `XR_WRIST`. It is not a fracture-location, fracture-type, or autonomous clinical-decision result.

## 1. Dataset

- **Dataset:** Stanford MURA v1.1, indexed only for `XR_WRIST`.
- **Unit and anatomy:** wrist radiograph studies; one label and final prediction per study, with 1–8 radiographic views per study.
- **Total:** 3,474 patients; 3,697 studies; 10,411 images.
- **Labels:** 2,274 NORMAL studies and 1,423 ABNORMAL studies. MURA’s label is study-level abnormality, not a confirmed fracture label.

| Views/study | Studies | Share |
|---:|---:|---:|
| 1 | 300 | 8.11% |
| 2 | 705 | 19.07% |
| 3 | 2,138 | 57.83% |
| 4 | 509 | 13.77% |
| 5 | 28 | 0.76% |
| 6 | 10 | 0.27% |
| 7 | 5 | 0.14% |
| 8 | 2 | 0.05% |

| Final split | Patients | Studies | Images | NORMAL | ABNORMAL |
|---|---:|---:|---:|---:|---:|
| Train | 2,614 | 2,768 | 7,805 | 1,707 | 1,061 |
| Validation | 653 | 692 | 1,947 | 427 | 265 |
| Test | 207 | 237 | 659 | 140 | 97 |

**Split construction.** All official MURA `valid` studies were assigned to the sealed final `test` split. The internal validation set was selected only from official MURA `train` patients with requested fraction 0.20, seed 23, and deterministic patient-label-signature stratification. Therefore the split is patient-level, and the official MURA validation set was used as the project’s final test set.

**Integrity checks performed and result.** The audited manifest reported: 0 duplicate paths, missing files, corrupt images, folder-label mismatches, inconsistent labels within studies, study-to-patient conflicts, patient overlaps across final splits, study overlaps across final splits, and unassigned studies. The final evaluator also validated all 10,411 joined manifest rows (7,805 train / 1,947 validation / 659 test) with no violations.

Primary evidence: `reports/wrist_data_audit.json`, `reports/PHASE1_EDA_DATA_QUALITY.md`, and `data/manifests/*.csv`.

## 2. Preprocessing

- **Input:** 224 × 224.
- **Decode/channels:** each image decoded as grayscale (`L`) and replicated into three RGB channels.
- **Geometry:** aspect-preserving black padding to square, then bilinear resize to 224 × 224. No anatomy crop.
- **Normalization:** fixed ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`; no dataset-statistic fit.
- **Training augmentation:** random rotation ±7°, brightness jitter ±0.08, contrast jitter ±0.10.
- **Evaluation augmentation:** none; validation and test preprocessing deterministic.
- **Explicitly excluded:** horizontal flip, random crop, and noise injection.
- **Sampling:** no oversampling or sampler is recorded. Model B used automatic positive class weighting; Model A did not (`class_weight: null`).

The actual Model B `run_config.json` is authoritative over the earlier YAML template, which contains stale/non-executed affine/brightness values.

## 3. Model A — frozen ResNet-18 embeddings + Logistic Regression

- **Encoder:** torchvision ResNet-18, ImageNet `ResNet18_Weights.IMAGENET1K_V1`, cached as `resnet18-f37072fd.pth`.
- **Embedding layer/dimension:** the final ImageNet `fc` was replaced by identity; the pre-classifier embedding is 512-dimensional.
- **Frozen:** yes; the ResNet-18 encoder was frozen during extraction.
- **Study aggregation:** image embeddings were mean-pooled within each study (`mean_embedding`), then one logistic-regression probability was produced per study.
- **Classifier:** `StandardScaler` + scikit-learn `LogisticRegression(solver="liblinear", max_iter=2000, random_state=42, class_weight=None)`.
- **Hyperparameter selection:** C in `{0.1, 1.0, 10.0}`; each candidate fitted on training studies only and selected by validation AUROC. Scores were 0.864434, 0.833202, and 0.820229 respectively; selected `C=0.1`.

| Split / frozen operating point | AUROC | AUPRC | Sensitivity | Specificity | Precision | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation, threshold 0.354285 | 0.864434 | 0.813649 | 0.830189 | 0.751756 | 0.674847 | 0.744501 | 321 | 106 | 45 | 220 |
| Test, threshold 0.354285 | 0.875626 | 0.859639 | 0.824742 | 0.807143 | 0.747664 | 0.784314 | 113 | 27 | 17 | 80 |

The test AUROC 95% stratified-bootstrap CI is 0.824884–0.920107 (2,000 replicates; seed 42).

## 4. Model B — fine-tuned ResNet-18

- **Architecture:** ImageNet-initialized torchvision ResNet-18, original 1,000-class `fc` replaced by `Linear(512, 1)`; output is one logit per view.
- **Frozen/unfrozen layers:** no freeze operation is recorded; the optimizer receives `model.parameters()`, so the full backbone and replacement head were fine-tuned.
- **Study loss:** weighted `StudyProbabilityBCELoss`: sigmoid view probabilities are mean-aggregated to one probability/study before BCE.
- **Class weighting:** automatic weighting was enabled from training studies: NORMAL=1,707, ABNORMAL=1,061, positive weight `1707/1061 = 1.6088595664`.
- **Optimizer:** AdamW, learning rate `1e-4`, weight decay `1e-4`.
- **Scheduler:** `ReduceLROnPlateau`, mode=max, factor=0.5, patience=2, minimum learning rate `1e-6`.
- **Batch size/workers:** 8 studies/batch, 4 workers.
- **Seed:** 23, with deterministic PyTorch configuration.
- **Epochs/early stopping:** maximum 15; 8 completed; early stopping after 4 non-improving epochs with `min_delta=0.0`; best epoch 4 by validation mean-aggregation AUROC 0.910715.
- **Augmentation:** exactly the training preprocessing above; no evaluation augmentation.
- **View aggregation:** both mean and max **probability pooling** were evaluated on validation. Mean was selected by validation AUROC (mean 0.910715; max 0.901754; tie rule would prefer mean). Mean was also the training aggregation.

| Split / frozen operating point | AUROC | AUPRC | Sensitivity | Specificity | Precision | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation, threshold 0.386023 | 0.910715 | 0.884109 | 0.800000 | 0.878220 | 0.803030 | 0.801512 | 375 | 52 | 53 | 212 |
| Test, threshold 0.386023 | 0.896981 | 0.893707 | 0.731959 | 0.950000 | 0.910256 | 0.811429 | 133 | 7 | 26 | 71 |

The test AUROC 95% stratified-bootstrap CI is 0.851762–0.935348 (2,000 replicates; seed 42).

## 5. Threshold selection

Both thresholds were selected on the 692-study validation set and saved before final test metric computation. Both used the same **Youden J** objective (`sensitivity + specificity − 1`) over 695 candidates, with the saved tie rule “closest to 0.5, then lower threshold.” The model-specific records explicitly say `selection_data: validation only` and `test_used_for_selection: false`.

| Model | Frozen threshold | Validation objective | Validation sensitivity | Validation specificity |
|---|---:|---:|---:|---:|
| A | 0.3542854529699628 | J=0.581945 | 0.830189 | 0.751756 |
| B | 0.3860229253768921 | J=0.678220 | 0.800000 | 0.878220 |

Existing Phase 1.5 artifacts reconstruct a validation-only threshold table, without changing either Phase 1 threshold. For example, Model B reaches sensitivity >=0.85 at threshold 0.2803 with specificity 0.799, and >=0.90 at 0.1610 with specificity 0.714. These are retrospective validation analyses only, not new frozen choices and not test-tuned results. See `reports/PHASE1_5_DECISION.md` and `reports/phase1_5_validation_threshold_table.csv`.

## 6. Per-view / multi-view work already done

| Item | Status | Exact implementation/evidence |
|---|---|---|
| Per-view predictions | IMPLEMENTED, NOT SAVED separately | Model B generates one logit/probability per real view internally, then restores the padded study layout. Saved CSVs contain only study-level predictions; no standalone per-view prediction CSV exists. |
| Mean pooling | IMPLEMENTED / TESTED | Model A: mean 512-D embeddings. Model B: mean view probabilities; selected by validation AUROC. |
| Max pooling | IMPLEMENTED / TESTED | Model B max view-probability pooling was compared on validation (AUROC 0.901754); it was not selected. |
| Embedding pooling | IMPLEMENTED / TESTED | Model A mean-embedding pooling before Logistic Regression. No max embedding pooling artifact. |
| Probability pooling | IMPLEMENTED / TESTED | Model B mean and max probability pooling. |
| Learned pooling | NOT IMPLEMENTED | No learned aggregation parameters/model artifact. |
| Attention | NOT IMPLEMENTED | No attention module or artifact. |
| View disagreement | NOT IMPLEMENTED | No persisted per-view scores or disagreement metric/report. |

## 7. Calibration / uncertainty

| Item | Status |
|---|---|
| Probability calibration | NOT IMPLEMENTED |
| Temperature scaling | NOT IMPLEMENTED |
| Brier score | NOT IMPLEMENTED / NOT SAVED |
| Expected Calibration Error | NOT IMPLEMENTED / NOT SAVED |
| Reliability diagram | NOT IMPLEMENTED / NOT SAVED |
| Predictive entropy | NOT IMPLEMENTED |
| Model disagreement | NOT IMPLEMENTED as an uncertainty method |
| View disagreement | NOT IMPLEMENTED |
| Uncertainty score | NOT IMPLEMENTED |
| Selective prediction / human-deferral simulation | NOT IMPLEMENTED |

The descriptive error report uses probability distance/confidence labels, but explicitly says this is **not** a clinical uncertainty estimate; it is not calibration or uncertainty modelling.

## 8. Error analysis already performed

Phase 1 created descriptive study-level error reports for each model. They index labels, probabilities, view counts, false negatives/positives, high predicted-class-score error categories, low-score predictions, and representative single-/multi-view studies. These raw scores are not calibrated confidence or uncertainty estimates, and the reports do not infer medical causes.

At frozen Phase 1 test thresholds, Model A had 17 FN and 27 FP; Model B had 26 FN and 7 FP. The reports’ representative index is capped in some categories. The original evaluation JSONs were historical local artifacts and are not retained in Git; use the aggregate counts in [baseline_results.md](baseline_results.md) and [error_analysis.md](error_analysis.md) for the durable repository record.

Existing post-Phase-1 analysis (`reports/PHASE1_5_DECISION.md`, based only on saved predictions) additionally records 14 shared FNs, 4 shared FPs, 3 Model-A-only FNs, 12 Model-B-only FNs, 23 Model-A-only FPs, and 3 Model-B-only FPs. It also stratifies the frozen test predictions by views:

| Views | Support (normal/abnormal) | Model A: AUROC / sens / spec | Model B: AUROC / sens / spec |
|---|---|---|---|
| 1 | 24 (22/2) | Not reported: only 2 abnormal | Not reported: only 2 abnormal |
| 2 | 44 (30/14) | 0.848 / 0.786 / 0.833 | 0.948 / 0.786 / 0.933 |
| 3+ | 169 (88/81) | 0.890 / 0.852 / 0.784 | 0.884 / 0.741 / 0.955 |

This is descriptive only; it does not establish that view count causes performance differences.

## 9. Software tests

The persisted Phase 1 progress record states **17 synthetic software tests passed**. This is software/integrity evidence, not model-performance evidence. The test source contains these 17 tests:

1. Reject a study with inconsistent labels before splitting.
2. Audit reports measured data-quality findings.
3. Dataset filtering retains all views of a selected study.
4. Ragged study collation emits a real-view mask.
5. Official `valid` is sealed and train/validation are patient-safe/deterministic.
6. Manifest and split CSV round trips preserve their exact contracts.
7. Audit CLI writes manifest/split artifacts from a data-root alias.
8. A patient-safe manifest passes all integrity checks.
9. Patient overlap between train and validation is rejected.
10. Splitting one study’s images between train and test is rejected.
11. Inconsistent labels within a study are rejected.
12. Test split use for fit or threshold selection is rejected.
13. Valid recorded provenance passes the integrity/provenance check.
14. Threshold helper refuses a test source.
15. Test metrics use a validation-selected threshold.
16. Evaluation CLI writes reports from synthetic predictions without MURA.
17. Frozen-threshold test evaluation does not reselect a threshold on test.

The tests are in `tests/test_label_integrity.py`, `tests/test_study_grouping.py`, `tests/test_patient_split.py`, and `tests/test_no_test_leakage.py`. They were not re-run for this handoff.

## 10. Saved artifacts

All paths below are relative to the repository root unless absolute path is shown. `NOT SAVED` means no corresponding persisted artifact was found.

| Artifact | Path(s) |
|---|---|
| Model A trained classifier | `models/frozen_logistic/frozen_embedding_logistic.joblib` |
| Model B checkpoint | `models/resnet18/best_resnet18_study_level.pt` |
| Model A embeddings | `data/processed/embeddings_train.npz`, `embeddings_val.npz`, `embeddings_test.npz` plus their `.metadata.json` files |
| A validation/test predictions | `models/frozen_logistic/validation_predictions.csv`; `models/frozen_logistic/test_predictions.csv` |
| B validation/test predictions | `models/resnet18/validation_predictions.csv`; `models/resnet18/validation_predictions_mean.csv`; `models/resnet18/validation_predictions_max.csv`; `models/resnet18/test_predictions.csv` |
| Per-view prediction CSV | **NOT SAVED** |
| Study-level predictions | All prediction CSVs listed immediately above (each has `patient_id, study_id, label, num_views, abnormal_probability`) |
| Split manifests | `data/manifests/wrist_manifest.csv`; `data/manifests/split_manifest.csv` |
| Configs | `configs/data.yaml`; `configs/frozen_logistic.yaml`; `configs/resnet18_finetune.yaml`; `configs/evaluation_models.json` |
| Model selections/provenance | `models/frozen_logistic/selection.json`; `models/frozen_logistic/test_prediction_metadata.json`; `models/frozen_logistic/validation_metrics.json`; `models/resnet18/selection.json`; `models/resnet18/run_config.json`; `models/resnet18/training_history.json`; `models/resnet18/test_prediction_metadata.json`; `models/resnet18/validation_metrics.json` |
| Final metrics | Historical local-only `reports/metrics/` JSON/CSV artifacts; **NOT SAVED in Git** under current governance. Aggregate historical values are retained in `reports/baseline_results.md`, `reports/error_analysis.md`, and `reports/PHASE1_SUMMARY.md`. |
| Error artifacts | `reports/error_analysis.md`; all CSV/Markdown files under `reports/errors/` |
| Executed notebook evidence | Notebooks exist under `notebooks/`, but all six have zero saved outputs and zero execution counts: execution is **NOT VERIFIED** from notebook metadata. |
| Figures | No Phase 1 figure was saved (`reports/figures/.gitkeep` only). A later Phase 1.5 figure exists at `reports/team_share/PHASE1_5_KEY_FIGURE.png`. |
| Phase 1 reports | `reports/PHASE1_SUMMARY.md`; `reports/PHASE1_EDA_DATA_QUALITY.md`; `reports/PHASE1_SCOPE_AND_COMPLIANCE.md`; `reports/baseline_results.md`; `reports/PROGRESS.md`; `reports/wrist_data_audit.json`; this handoff |

## 11. Test-set status

- **Has the final test set been opened?** Yes. Test-prediction CSVs exist for both models, and final test evaluation artifacts exist.
- **How many times was it evaluated?** The progress artifact records one sole final study-level metric evaluation (`evaluate_phase1.py`). Test prediction generation occurred once per model after selections froze; their metadata say final metrics were not computed by the training scripts.
- **Decisions made before opening for final metrics:** split definition; Model A C; Model B best checkpoint/epoch; Model B mean vs max aggregation; and both operating thresholds. All are marked validation-only.
- **Observed after opening:** the test prediction CSVs; final metrics, bootstrap AUROC CIs, comparison table, and descriptive error reports.
- **Untouched for future model selection?** No. It can accurately be called a sealed/frozen **Phase 1 final test evaluation**, because model-selection records say it was not used for those Phase 1 choices. It cannot honestly be called untouched for future model selection, since its labels and results have already been observed; do not tune a next model or threshold against it.

## 12. Compute

- **Model B device:** CUDA.
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 6,144 MiB inventory.
- **PyTorch:** 2.2.2 recorded for Model B.
- **Model A device:** CUDA is recorded in embedding metadata.
- **Training duration, Model A:** NOT VERIFIED / not recorded.
- **Training duration, Model B:** NOT VERIFIED / not recorded.
- **Peak GPU memory:** NOT VERIFIED / not recorded. GPU capacity, not peak use, is recorded.

## 13. Final Phase 1 comparison

### Validation (selection set; selected threshold)

| Model | AUROC | AUPRC | Sensitivity | Specificity | F1 | FP | FN | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A — frozen ResNet-18 embeddings + Logistic Regression | 0.864434 | 0.813649 | 0.830189 | 0.751756 | 0.744501 | 106 | 45 | 0.354285 |
| B — fine-tuned ResNet-18 | 0.910715 | 0.884109 | 0.800000 | 0.878220 | 0.801512 | 52 | 53 | 0.386023 |

### Test (frozen validation-selected threshold)

| Model | AUROC | AUPRC | Sensitivity | Specificity | F1 | FP | FN | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A — frozen ResNet-18 embeddings + Logistic Regression | 0.875626 | 0.859639 | 0.824742 | 0.807143 | 0.784314 | 27 | 17 | 0.354285 |
| B — fine-tuned ResNet-18 | 0.896981 | 0.893707 | 0.731959 | 0.950000 | 0.811429 | 7 | 26 | 0.386023 |

## 14. Important work not yet implemented

- Learned multi-view aggregation or attention pooling.
- Stronger backbone (e.g., DenseNet/ConvNeXt).
- Probability calibration, temperature scaling, Brier/ECE/reliability diagrams.
- Predictive/model/view uncertainty, selective prediction, or deferral simulation.
- Ensemble.
- Grad-CAM or other explainability implementation.
- Agentic AI layer.
- Streamlit/UI/integration for incoming clinical images.
- External validation, paired model-comparison inference, or a fresh untouched test set for future selection.

## 15. Recommended next research step (do not execute here)

Run one **pre-specified validation-only sensitivity-constrained threshold/calibration experiment for the already trained Model B**; keep its backbone and mean aggregation fixed. Model B has higher validation ranking and wins the saved matched-operating-point comparisons, but its frozen Youden-J operating point favors specificity (validation 0.878) over sensitivity (0.800), contributing to 26 test FNs. The saved validation predictions show it can reach sensitivity >=0.85 with specificity 0.799, which is higher than Model A’s 0.696 at the same sensitivity floor. The main risk is that a threshold calibrated on this validation split may not transfer; use a pre-specified sensitivity target and a new untouched evaluation set rather than the already opened Phase 1 test set.
