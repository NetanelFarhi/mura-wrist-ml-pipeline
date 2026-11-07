# Results: fine-tuning depth ablation for ResNet-18 (Model B)

**Protocol:** `PROTOCOL.md` (pre-registered 19.10.2026; deviations in §13; run registry in §14)
**Analysis:** `analysis.ipynb` (executed 20.10.2026); tables and figure in `results/`
**Runs:** 20.10.2026, Google Colab, Tesla T4, torch 2.11.0+cu130 (all runs, including the three pre-existing S4 runs)
**Scope:** internal validation only (691 studies, 653 patients, 269 ABNORMAL). The official MURA `valid` split was not read. All numbers are a *development comparison* (§11).

## 1. Answer

Training `layer3 + layer4 + fc` (S3, 93.9% of the parameters) gives the same study-level AUROC as training the whole network (S4): **no practically relevant difference** (§9, final 5-seed analysis). Training only `layer4 + fc` (S2) is close to S4, but the interval is too wide to call it equivalent. Training only the classifier head (S1) is clearly worse under this training recipe.

## 2. Primary comparison: S3 vs S4 (§8–§10)

| Stage | Seeds per arm | Δ AUROC (S3 − S4) | 95% CI (patient bootstrap, 2,000) | §9 decision |
|---|---|---|---|---|
| 1 | 23, 42, 7 | +0.0009 | [−0.0091, +0.0117] | inconclusive → stage 2 (§10) |
| 2 (final) | + 101, 202 | **+0.0011** | **[−0.0072, +0.0089]** | **no practically relevant difference** |

Seed-matched differences S3 − S4: +0.0022 (23), +0.0031 (42), −0.0027 (7), −0.0029 (101), +0.0058 (202).

## 3. All arms (5 seeds each; mean ± SD across seeds)

| Arm | Trainable | AUROC | AUPRC | Spec. at sens. ≥ 95% | FP / FN | Best epoch | Loss gap at best |
|---|---|---|---|---|---|---|---|
| S1 `fc` | 513 | 0.8611 ± 0.0030 | 0.8172 | 0.424 ± 0.022 | 243 / 13 | 15 (of 15) | −0.014 |
| S2 `layer4+fc` | 8,394,241 | 0.9012 ± 0.0031 | 0.8737 | 0.455 ± 0.029 | 230 / 13 | 3.6 | 0.062 |
| S3 `layer3+layer4+fc` | 10,493,953 | 0.9039 ± 0.0028 | 0.8787 | 0.443 ± 0.021 | 235 / 13 | 5.4 | 0.184 |
| S4 all (control) | 11,177,025 | 0.9028 ± 0.0012 | 0.8709 | 0.429 ± 0.059 | 241 / 13 | 5.6 | 0.121 |
| Model A (C = 0.1) | — | 0.8804 | 0.8515 | 0.393 | 256 / 13 | — | — |

Time per epoch was about 1.0 min in every arm, so freezing did not make training faster in this setup.

Secondary contrasts with S4 (descriptive, §8.1):

| Contrast | Δ AUROC | 95% CI | §9 rule applied descriptively |
|---|---|---|---|
| S2 − S4 | −0.0016 | [−0.0126, +0.0098] | inconclusive |
| S1 − S4 | −0.0417 | [−0.0643, −0.0183] | S4 better |

## 4. Comparison with the frozen Model A, C = 0.1 (§13 addition)

Model A predictions were read from a copy whose sha256 (`03e93f6e…4a91455`) equals the value frozen in the 06a policy `run_20261017T105209Z`.

| Arm | Δ AUROC vs A | 95% CI | Δ spec@95 vs A | 95% CI | Seeds with spec gain ≥ 0.02 |
|---|---|---|---|---|---|
| S1 | −0.019 | [−0.037, −0.002] | +0.031 | [−0.088, +0.115] | 3 of 5 |
| S2 | +0.021 | [+0.005, +0.038] | +0.061 | [−0.033, +0.167] | 5 of 5 |
| S3 | +0.023 | [+0.006, +0.041] | +0.050 | [−0.078, +0.170] | 5 of 5 |
| S4 | +0.022 | [+0.003, +0.042] | +0.036 | [−0.088, +0.180] | 3 of 5 |

S2, S3 and S4 rank studies better than Model A. At the 95%-sensitivity operating point the gain in specificity is positive on average but its interval includes zero in every arm. The specificity CI re-derives the threshold inside every resample.

## 5. Reproducibility check in the reused runtime (R0b, §13)

R0b (S4, seed 23, run in the shared Colab session) reproduced `run_20261017T095546Z` **bit for bit**: identical `validation_predictions.csv` (sha256 `7f4f49df…`), AUROC 0.9020419669127363, max |Δp| = 0. Running the arms one after another in one session did not change results.

## 6. Interpretation

1. **Model B does not need a change of fine-tuning depth.** Freezing the stem, `layer1` and `layer2` (S3) neither helps nor hurts AUROC within the pre-registered ±0.01 margin. Hodaya's freezing choice is therefore not the reason for any difference between her notebook and Model B; the other factors listed in §11 were not tested.
2. **S3 had the most stable operating point.** The seed SD of specificity at sensitivity ≥ 95% was 0.021 for S3 versus 0.059 for S4, and all 5 S3 seeds beat Model A by ≥ 0.02 (S4: 3 of 5). This is descriptive and was not a pre-registered endpoint.
3. **The 06a choice of B seed 7 rests on seed luck.** S4 specificity at sensitivity ≥ 95% ranges 0.356–0.507 across 5 seeds (mean 0.429). Seed 7 (0.469) is above the mean, so its margin over Model A overstates the typical gain.
4. **S1 is a lower bound, not evidence against frozen features.** All five S1 runs were still improving at the 15-epoch limit (best epoch 15). Model A, which also uses frozen ImageNet ResNet-18 features but fits a tuned logistic regression to convergence, is better than S1 by 0.019 AUROC.

## 7. Limitations

- Model selection (early stopping, aggregation choice) used the same internal validation split, so absolute AUROC values are optimistic; the comparison between arms is symmetric.
- With 691 studies the bootstrap interval is about ±0.01 even between two seeds of the same arm, so ±0.01 is close to the resolution of this split. Stage 1 was inconclusive for this reason; stage 2 resolved it.
- Hyperparameters were not tuned per arm (learning rate 1e-4 is too low for S1).
- Not tested: image-level training, other augmentations, resizing without padding (§11).
- Results do not enter 06a. Switching Model B to another arm would need a team decision, a new 06a freeze and one 06b run.

## 8. Execution notes

- 17 new training runs (S1–S3 × 5 seeds, S4 × 2 stage-2 seeds) plus R0b, all on Tesla T4 with torch 2.11.0+cu130; every executed notebook is in `runs/` (21 code cells executed, no errors, no Colab link).
- Drive had to be re-authorised once in the session (before S3 seed 7); this does not affect the code path.
- After the last run, a temporary cell in the Colab session (not saved to any notebook) wrote `RUNS_ARTEM/ABLATION_FINETUNE_DEPTH/inputs_bundle_20261020T224202Z.json` with the training logs, metadata and prediction sha256 of all runs; the analysis checks every prediction file against it.
