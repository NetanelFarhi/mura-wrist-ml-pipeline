# MURA XR_WRIST research project

Study-level **NORMAL / ABNORMAL** classification of wrist radiographs from Stanford MURA v1.1. This is a research prototype for human review, not a fracture diagnosis or clinical system.

## Main development result

On the current internal patient-level validation cohort (691 studies from 653 patients), the fine-tuned Model B mean-probability policy reached **AUROC 0.913, AUPRC 0.878, sensitivity 95.2%, and specificity 58.3%** at its frozen threshold. These are development results, not an untouched test result. Full metrics and uncertainty intervals are in [Checkpoint 1 results](docs/results/checkpoint1/README.md).

## Current workflow

- [Notebook guide](notebooks/README.md): canonical steps 01–06, diagnostic 07, local viewer 08, and deterministic safety/policy 09.
- Canonical Model B artifacts live outside Git at `DATA/MODEL_B_FINE_TUNING/`; the bundle requirements and checkpoint fingerprint are defined in [model bundle metadata](configs/model_b_artifact_bundle.json). Do not rerun fine-tuning as part of repository organization.
- Any older `models/resnet18` location is historical only; it is not the canonical Model B source.
- [Governance and result lineage](reports/TEST_SET_GOVERNANCE.md) explain the one-time historical use of official MURA `valid`.
- [Wrist Review app](apps/wrist_review/README.md) documents the local research interface.

See [docs](docs/README.md) for reviewer materials and [reports](reports/README.md) for current, historical, experimental, governance, and review documents. Raw radiographs, checkpoints, private manifests, and per-study prediction files remain outside Git.
