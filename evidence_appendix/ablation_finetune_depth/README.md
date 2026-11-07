# Evidence appendix: fine-tuning depth ablation (from branch `ARTEM`)

**Status:** evidence appendix, imported from branch `ARTEM` @ `d0c0b3c`. It is **not** the
canonical Model B and not part of the runtime. Canonical Model B: root `README.md`
(Checkpoint 1 results) and `models/README.md`.

## Question and answer

- **Question:** does the number of trainable ResNet-18 layers change study-level
  performance of Model B when everything else is held constant? Arms: S1 = `fc` only,
  S2 = `layer4 + fc`, S3 = `layer3 + layer4 + fc`, S4 = all layers (control);
  5 seeds per arm.
- **Answer:** S3 vs S4 (primary comparison): ΔAUROC +0.0011, 95% CI [−0.0072, +0.0089]
  → no practically relevant difference (pre-registered margin ±0.01). S2 is close to S4;
  S1 is clearly worse (AUROC 0.861 vs 0.903).
- **Against Model A (C = 0.1):** S2–S4 rank studies better (ΔAUROC ≈ +0.02, CI above 0).
  The specificity gain at sensitivity ≥ 95% is positive on average, but its CI includes 0
  in every arm.

## Why it counts as evidence

| Element | Where |
|---|---|
| Pre-registered protocol | `PROTOCOL.md`, first committed on branch `ARTEM` in `3b28212` (19.10.2026, 07:50 UTC), before the first run (R0, `run_20261019T084628Z`, 08:46 UTC) |
| Decision rule and two-stage design fixed in advance | `PROTOCOL.md` §9–10 |
| Bit-identical reproducibility check (R0, R0b) | `PROTOCOL.md` §6, §14; `RESULTS.md` §5 |
| Paired patient-level bootstrap (2,000 resamples), threshold re-derived in every resample | `analysis.ipynb`; `PROTOCOL.md` §8 |

## Files

| File | Content |
|---|---|
| `PROTOCOL.md` | Pre-registered protocol, deviation log (§13), run registry (§14). Copied unchanged |
| `RESULTS.md` | Results and interpretation. Copied unchanged |
| `analysis.ipynb` | Statistical analysis of the saved predictions (no training). Copied unchanged |
| `results/` | 5 result tables and 1 figure. Copied unchanged |

Training notebooks are **not** copied, so that `main` keeps a single Model B training
pipeline. They stay on branch `ARTEM` @ `d0c0b3c`:
[template](https://github.com/NetanelFarhi/mura-wrist-ml-pipeline/blob/d0c0b3c97ca5f6795591f30bafb83336cf3d6509/experiments/ablation_finetune_depth/05_ablation_template.ipynb)
and [19 executed runs](https://github.com/NetanelFarhi/mura-wrist-ml-pipeline/tree/d0c0b3c97ca5f6795591f30bafb83336cf3d6509/experiments/ablation_finetune_depth/runs).
Paths in `PROTOCOL.md` and `RESULTS.md` that are not in this folder (for example
`notebooks/05_MURA_ResNet18_FineTune.ipynb`, `seed_runs/`, `runs/`) refer to that commit.
Branch `ARTEM` must be kept: its history carries the pre-registration timestamp.

## How it relates to the canonical Model B

The ablation used a different training recipe from the canonical Model B in `main`:

| Setting | Canonical Model B (`main`) | Ablation S4 (`ARTEM`) |
|---|---|---|
| Studies per batch | 2 | 8 |
| Augmentation | rotation ≤ 7°, brightness 0.08, contrast 0.10 | rotation ≤ 7°, translation 0.03, scale 0.95–1.05, brightness 0.05 |
| Aggregation | mean, fixed in advance | mean or max, chosen on validation (ties → mean) |
| Seeds | 1 (42) | 5 |

Shared: ResNet-18 with ImageNet weights, all layers trained, AdamW (lr 1e-4, weight decay
1e-4), ReduceLROnPlateau (factor 0.5), up to 15 epochs, early stopping on validation AUROC
(patience 4), same preprocessing, weighted BCE on the study-level mean probability, same
691-study internal validation.

Use the ablation for:

- **seed-to-seed variability:** S4 AUROC is 0.9028 ± 0.0012 (mean ± SD, 5 seeds), but
  specificity at sensitivity ≥ 95% ranges 0.356–0.507. The specificity of a single run
  (canonical Model B: 58.3%) should therefore be read with caution;
- **method:** the same paired bootstrap with a re-derived threshold is used for the
  canonical comparison in `scripts/build_paired_b_minus_a.py`;
- **fine-tuning depth:** training all layers, as the canonical training script does, is
  not worse than partial fine-tuning and is better than training only the head.

Do not use it as a second Model B result.

## Model A reference

`analysis.ipynb` compares the arms with Model A (C = 0.1), read from a copy with SHA-256
`03e93f6e54aa80cb15c0b8445392224b060e843c87e4d243ab59017fd4a91455` — the same Model A
input as `scripts/build_paired_b_minus_a.py`. The per-run notebooks on `ARTEM` also print
a comparison with `BASELINE_MODEL_A/validation_predictions.csv`, which had been overwritten
by a C = 0.01 model on 18.10.2026; that comparison is not used (`PROTOCOL.md` §11).

## How to reproduce

1. **Statistics (minutes):** open `analysis.ipynb` in Colab with access to the shared
   Drive. It reads the per-run `validation_predictions.csv` files from `DATA/RUNS_ARTEM/`
   (not in Git: they contain study IDs), checks each file's SHA-256 against the run bundle
   and regenerates `results/`.
2. **One training run (~13 min, Colab T4):** open any notebook from the `ARTEM` runs folder
   (link above); they differ only in arm and seed. For example, S4 seed 23 should give
   AUROC 0.902042 with bit-identical predictions, as in checks R0 and R0b. The template
   stops on a non-T4 GPU.
3. **All runs:** 20 runs × ~13 min ≈ 4–5 hours.

## Limitations

- Internal validation only (691 studies, 653 patients); the official MURA valid split was
  not read.
- The same validation split was used for early stopping and aggregation choice, so
  absolute AUROC values are optimistic; comparisons between arms are symmetric.
- A margin of ±0.01 AUROC is close to the resolution of a 691-study split.

## Data

No radiographs and no per-study predictions are stored in this folder.
