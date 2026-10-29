# Model artifacts

This directory is intentionally empty in Git. Raw weights, per-study
predictions, and derived artifacts stay in authorised local/Drive storage.

## Canonical Checkpoint 1 Model B bundle

Notebooks 05 and 06 expect one explicit local/Drive bundle at
`DATA/MODEL_B_FINE_TUNING/`. It is the fine-tuned ResNet-18 with **mean
probability study aggregation** and the frozen internal-validation policy
`sensitivity >= 95%` at threshold `0.263567715883255`.

Required contents are listed in
[`configs/model_b_artifact_bundle.json`](../configs/model_b_artifact_bundle.json).
The canonical checkpoint is named `best_resnet18_study_level.pt` and its
SHA-256 is recorded there. Verify a local bundle before using it:

```text
python scripts/verify_model_artifact_bundle.py --artifact-dir <authorised-local-or-Drive-bundle>
```

The similarly named `resnet18_model_b_best.pt` run dated 2026-10-19 is a
separate partial-fine-tuning experiment (layer3/layer4/fc; different results
and no frozen 95%-sensitivity policy). Store it under a clearly named
experimental folder; it must never silently replace the canonical bundle.

Model selection and threshold selection use internal train/validation only.
Official MURA `valid` is not permitted for later selection; see
[`reports/TEST_SET_GOVERNANCE.md`](../reports/TEST_SET_GOVERNANCE.md).

