# Protocol: fine-tuning depth ablation for ResNet-18 (Model B)

**Location:** `experiments/ablation_finetune_depth/PROTOCOL.md` (branch `ARTEM`)
**Status:** pre-registered **before the first run**. After the first run, changes go only to §13 "Deviation log".
**Date fixed:** 19.10.2026 (the commit timestamp is authoritative)
**Author:** Artem

## 1. Question

Does the number of trainable ResNet-18 layers change study-level performance of Model B when everything else is held constant?

Motivation: `notebooks/Hodaya_05_MURA_End_to_End__Fine_Tuning(Model_B).ipynb` (branch `main`) trains only `layer3`, `layer4` and `fc`. That run changed about eight factors at once, so the effect of freezing cannot be isolated from it.

## 2. Hypotheses

- **H0:** AUROC of partial fine-tuning does not differ from full fine-tuning.
- **H1 (two-sided):** it differs.
- **Expectation (not a hypothesis):** "no practically relevant difference" for arm S3, because only 6.1% of the parameters are frozen there.

## 3. Arms (the only factor that changes)

| Arm | Trainable layers | Trainable parameters | Share of 11,177,025 |
|---|---|---|---|
| S1 | `fc` | 513 | 0.005% |
| S2 | `layer4` + `fc` | 8,394,241 | 75.1% |
| S3 | `layer3` + `layer4` + `fc` (Hodaya's setup) | 10,493,953 | 93.9% |
| S4 | all layers (control, current Model B) | 11,177,025 | 100% |

Definition of "frozen":

1. Frozen parameters have `requires_grad = False` and are not passed to the optimizer.
2. BatchNorm layers in the frozen part are kept in `eval()` mode during training, so their running statistics do not change.
3. The arm, the number of trainable parameters and the BatchNorm mode are written to `run_metadata.json`.

## 4. Held constant across all arms

| Item | Value |
|---|---|
| Code | `notebooks/05_MURA_ResNet18_FineTune.ipynb` (ARTEM). The only change is a CONFIG parameter `trainable_layers` with values `fc`, `layer4+fc`, `layer3+layer4+fc`, `all`. The default is `all`, which behaves exactly like the current notebook |
| Data | `PREPROCESSED_WRIST/wrist_internal_train_clean.csv` (7,810 images, 2,769 studies, 2,614 patients); `wrist_internal_val_clean.csv` (1,942 images, 691 studies, 653 patients) |
| Image preprocessing | grayscale → black pad to square → 224×224 (bilinear) → RGB → ImageNet normalization |
| Augmentation | as in the 05 CONFIG: rotation 7°, translation 0.03, scale 0.95–1.05, brightness 0.05; no flips |
| Loss | weighted study-level BCE on the mean of view probabilities; ABNORMAL weight = n_normal / n_abnormal (1.620) |
| Optimization | AdamW (lr 1e-4, weight decay 1e-4, trainable parameters only); ReduceLROnPlateau (mode=max, factor 0.5, patience 2); up to 15 epochs; early stopping with patience 4 on validation AUROC (mean aggregation); 8 studies per batch; mixed precision (AMP) |
| Aggregation | mean or max chosen on validation, ties keep mean (as in 05) |
| Environment | Google Colab, T4 GPU |
| Seeds | 23, 42, 7 |

## 5. Runs

| Arm | seed 23 | seed 42 | seed 7 |
|---|---|---|---|
| S4 (existing) | `run_20261017T095546Z` | `run_20261016T121029Z` | `run_20261016T123127Z` |
| S3 | new | new | new |
| S2 | new | new | new |
| S1 | new | new | new |
| R0 (S4 reproducibility rerun) | new | — | — |

10 new runs in total, about 13 minutes each on a T4.

The existing S4 notebooks were checked on 19.10.2026: the code of `05_MURA_ResNet18_FineTune.ipynb` (seed 23) and of `seed_runs/…_seed42.ipynb` and `seed_runs/…_seed7.ipynb` differs only in the seed line, a non-functional regex form (`(^|/)` vs `(?:^|/)`) and the indentation of one line.

## 6. Reproducibility check (R0), run first

S4 with seed 23 is rerun on the modified code (`trainable_layers = all`). This checks two things at once: the code change did not alter the default behaviour, and the current Colab environment reproduces the 17.10.2026 result.

| R0 result | Decision |
|---|---|
| abs(AUROC(R0) − 0.9020) ≤ 0.005 | the existing S4 runs are used as the control |
| abs(AUROC(R0) − 0.9020) > 0.005 | all three S4 runs are repeated in the current environment and the old ones are excluded from the analysis |

Rationale for the tolerance: 0.005 is about 3.6 times the seed-to-seed SD of AUROC (≈ 0.0014).

## 7. Endpoints

- **Primary:** study-level AUROC on internal validation (691 studies), averaged over the three seeds within an arm.
- **Secondary:**
  - specificity at sensitivity ≥ 95% using the 06a rule (highest threshold with sensitivity ≥ 0.95; flag if p ≥ threshold);
  - FN, FP, share of studies flagged for review;
  - AUPRC;
  - best epoch;
  - train/validation loss gap at the best epoch;
  - time per epoch.

## 8. Analysis

1. **Primary comparison:** S3 vs S4. Comparisons of S1 and S2 with S4 are secondary and descriptive; they show how AUROC depends on the number of trainable parameters. This avoids a multiple-comparisons problem by design.
2. **Difference:** Δ = seed-averaged AUROC(Sx) − seed-averaged AUROC(S4).
3. **95% confidence interval:** paired patient-cluster bootstrap, 2,000 resamples, random generator seed 2026. In each resample, 653 patients are drawn with replacement, AUROC is computed for each of the six runs, averaged within each arm, and the difference is taken; percentile interval.
4. **Also reported:** seed-matched differences (S3 vs S4 for seeds 23, 42, 7) and the seed-to-seed SD within each arm.
5. **Data source:** only the `validation_predictions.csv` of each run.

## 9. Decision rule (S3 vs S4, applied in this order)

1. The 95% CI of Δ lies entirely within [−0.01, +0.01] → **"no practically relevant difference"**, even if the CI excludes 0.
2. Otherwise, the lower bound of the CI > 0 → **"S3 is better"**.
3. Otherwise, the upper bound of the CI < 0 → **"S4 is better"**.
4. Otherwise → **"inconclusive"**, go to §10.

The 0.01 margin is fixed in advance as the smallest practically relevant AUROC difference for this project. It is about 7 times the seed-to-seed SD.

## 10. Stage 2 (extension specified in advance)

Only if §9 gives "inconclusive":

- seeds 101 and 202 are added to all four arms (8 runs);
- the §8–§9 analysis is repeated once with 5 seeds per arm;
- the report states that a two-stage design was used.

No other extensions are allowed.

## 11. Limitations and rules

- **The official MURA `valid` split (locked test) is not used** (`reports/TEST_SET_GOVERNANCE.md`). All results are labelled "internal validation, development comparison".
- **The results do not enter the 06a selection.** Switching Model B to another arm requires a team decision, a new policy freeze in 06a, and then a single 06b run.
- **Hyperparameters are not tuned per arm.** A learning rate of 1e-4 may be suboptimal for S1, so the S1 result is a lower bound for that arm.
- **Absolute AUROC values are optimistic** because early stopping uses the same validation split. The comparison between arms is symmetric.
- **Other differences in Hodaya's notebook are not tested:** image-level training, her augmentations, and resizing without padding. Only fine-tuning depth is tested here.
- **`comparison_with_model_a.json` from new runs is not used:** Model A in the shared folder was overwritten (C = 0.01, 18.10.2026), so new runs would compare against that model, not the frozen Model A.

## 12. Storage

- Each run writes to a new folder `DATA/RUNS_ARTEM/05_MODEL_B_FINETUNE/run_<UTC>/` (the existing 05 mechanism). `run_metadata.json` gets these extra fields:
  - `experiment: "ablation_finetune_depth"`;
  - `trainable_layers`;
  - `n_trainable_params`;
  - `frozen_bn_eval: true`.
- Executed notebooks: `experiments/ablation_finetune_depth/runs/05_<arm>_seed<N>.ipynb` — copies of 05 that differ only in arm and seed. No radiographs in outputs.
- Analysis: `experiments/ablation_finetune_depth/analysis.ipynb`. Results: `experiments/ablation_finetune_depth/RESULTS.md`.

## 13. Deviation log

| Date | What changed | Reason |
|---|---|---|
| 19.10.2026, before the first run | The modified code lives in a template, `experiments/ablation_finetune_depth/05_ablation_template.ipynb`, not in `notebooks/05_MURA_ResNet18_FineTune.ipynb` (§4). Run notebooks are copies of this template. The template also stops on a non-T4 GPU and prints the registry row and the R0 check (§6). | `notebooks/05` holds the executed seed-23 run behind the frozen Model B; changing its code would make code and saved outputs disagree. The training logic is otherwise as specified in §3–§4. |
| 20.10.2026, before the first S1–S3 run | All remaining runs are executed in **one Colab T4 session**: the template is opened once; before each run only `seed` and `trainable_layers` in the Configuration cell are changed, then Run all, then File → Save a copy in GitHub under the §14 name. Drive is mounted once, and the images staged on the local disk by the first run are reused through the notebook's own skip logic. Order: S3 seed 23 (fresh runtime), R0b, S3 seeds 42 and 7, S2 × 3, S1 × 3. **R0b** = S4 seed 23 in the reused runtime, judged by the §6 rule against `run_20261017T095546Z` (bit-identical AUROC is expected and reported). If R0b fails, every run made in the reused runtime is repeated in a fresh runtime. | Every new runtime needs a manual Drive OAuth confirmation; one session needs it once. The code is unchanged and all random generators are reseeded in the Configuration cell. |
| 20.10.2026, before the first S1–S3 run | **Added a descriptive comparison with the frozen Model A (C = 0.1)**, `model_a@e69895841ee1` (AUROC 0.880442; specificity at sensitivity ≥ 95% 0.393365). Its validation predictions are read from Hagit's copy of `BASELINE_MODEL_A/validation_predictions.csv`, whose sha256 `03e93f6e54aa80cb15c0b8445392224b060e843c87e4d243ab59017fd4a91455` equals the value frozen in the 06a policy `run_20261017T105209Z`. For every arm: seed-averaged Δ AUROC vs Model A with the §8 bootstrap, and specificity at sensitivity ≥ 95% vs Model A (06a margin 0.02). Read only; not part of the §9 decision. | The shared `BASELINE_MODEL_A` now holds A′ (C = 0.01, overwritten 18.10.2026). Artem chose the C = 0.1 version as the Model A reference. |

## 14. Run registry

| Arm | seed | Notebook (`runs/`) | run_id | Date | Status |
|---|---|---|---|---|---|
| R0 (S4) | 23 | `05_S4_seed23_R0.ipynb` (saved 20.10.2026, commit 39cdc7d) | `run_20261019T084628Z` | 19.10.2026 | done: AUROC 0.902042, AUPRC 0.868998, best epoch 6 of 10 — identical to `run_20261017T095546Z`; Tesla T4, torch 2.11.0+cu130 → **PASS** (§6): the existing S4 runs stay the control |
| R0b (S4, reused runtime) | 23 | `05_S4_seed23_R0b.ipynb` | `run_20261020T172538Z` | 20.10.2026 | done: AUROC 0.902042, best epoch 6 of 10; `validation_predictions.csv` bit-identical to `run_20261017T095546Z` (max abs Δp = 0) → **PASS** (§6, §13) |
| S3 | 23 | `05_S3_seed23.ipynb` | `run_20261020T170941Z` | 20.10.2026 | done: AUROC 0.904279, AUPRC 0.879486, best epoch 8 of 12 |
| S3 | 42 | `05_S3_seed42.ipynb` | `run_20261020T173835Z` | 20.10.2026 | done: AUROC 0.904896, AUPRC 0.880614, best epoch 7 of 11 |
| S3 | 7 | `05_S3_seed7.ipynb` | `run_20261020T191814Z` | 20.10.2026 | done: AUROC 0.901654, AUPRC 0.870017, best epoch 2 of 6 |
| S2 | 23 | `05_S2_seed23.ipynb` | `run_20261020T192952Z` | 20.10.2026 | done: AUROC 0.901328, AUPRC 0.872449, best epoch 4 of 8 |
| S2 | 42 | `05_S2_seed42.ipynb` | `run_20261020T194601Z` | 20.10.2026 | done: AUROC 0.901760, AUPRC 0.869341, best epoch 7 of 11 |
| S2 | 7 | `05_S2_seed7.ipynb` | `run_20261020T200020Z` | 20.10.2026 | done: AUROC 0.896043, AUPRC 0.864971, best epoch 1 of 5 |
| S1 | 23 | `05_S1_seed23.ipynb` | `run_20261020T200935Z` | 20.10.2026 | done: AUROC 0.860357, AUPRC 0.820281, best epoch 15 of 15 |
| S1 | 42 | `05_S1_seed42.ipynb` | `run_20261020T202457Z` | 20.10.2026 | done: AUROC 0.857494, AUPRC 0.815333, best epoch 15 of 15 |
| S1 | 7 | `05_S1_seed7.ipynb` | `run_20261020T204155Z` | 20.10.2026 | done: AUROC 0.859326, AUPRC 0.814357, best epoch 15 of 15 |
| S3 | 101 | `05_S3_seed101.ipynb` | `run_20261020T205951Z` | 20.10.2026 | done (stage 2): AUROC 0.900879, AUPRC 0.874321, best epoch 3 of 7 |
| S4 | 101 | `05_S4_seed101.ipynb` | `run_20261020T211047Z` | 20.10.2026 | done (stage 2): AUROC 0.903742, AUPRC 0.873144, best epoch 6 of 10 |
| S3 | 202 | `05_S3_seed202.ipynb` | `run_20261020T212344Z` | 20.10.2026 | done (stage 2): AUROC 0.907812, AUPRC 0.889110, best epoch 7 of 11 |
| S4 | 202 | `05_S4_seed202.ipynb` | `run_20261020T213617Z` | 20.10.2026 | done (stage 2): AUROC 0.902024, AUPRC 0.867068, best epoch 4 of 8 |
| S2 | 101 | `05_S2_seed101.ipynb` | `run_20261020T214625Z` | 20.10.2026 | done (stage 2): AUROC 0.903363, AUPRC 0.882857, best epoch 3 of 7 |
| S2 | 202 | `05_S2_seed202.ipynb` | `run_20261020T215534Z` | 20.10.2026 | done (stage 2): AUROC 0.903689, AUPRC 0.879105, best epoch 3 of 7 |
| S1 | 101 | `05_S1_seed101.ipynb` | `run_20261020T220811Z` | 20.10.2026 | done (stage 2): AUROC 0.863740, AUPRC 0.816789, best epoch 15 of 15 |
| S1 | 202 | `05_S1_seed202.ipynb` | `run_20261020T222524Z` | 20.10.2026 | done (stage 2): AUROC 0.864629, AUPRC 0.819128, best epoch 15 of 15 |

Stage 2 (§10) was triggered because the stage-1 S3 vs S4 result was "inconclusive" (Δ +0.0009, 95% CI [−0.0091, +0.0117]). Final 5-seed result: Δ +0.0011, 95% CI [−0.0072, +0.0089] → "no practically relevant difference". S4 seeds 23, 42 and 7 are the existing control runs listed in §5. Full results: `RESULTS.md`, `analysis.ipynb`, `results/`.
