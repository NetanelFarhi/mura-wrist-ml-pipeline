# MURA XR_WRIST — Study-Level Abnormality Classification

A research project using **Stanford MURA v1.1 wrist radiographs** to classify an imaging study as **NORMAL / ABNORMAL** and support human review. A study can contain several X-ray views; the pipeline combines their evidence into one study-level result.

The project brings together data preparation, a frozen-feature baseline, a fine-tuned ResNet-18, reproducible evaluation, deterministic safety checks, and a local **Wrist Review** application. The scope is MURA-defined abnormality: the labels do not establish an acute-fracture diagnosis or lesion location. This is a research prototype, not a clinically validated system.

## Team

| Team member | Current responsibilities |
| --- | --- |
| **Netanel** | Model-bundle and runtime integration, study-level results, deterministic safety, local application/demo, bounded-agent workstream, and coordination of the experimental all-anatomy extension. |
| **Hagit** | Data access and governance, patient/study/view organization, preprocessing, and data-scope review. |
| **Hodaya** | Core fine-tuned model workflow, model-training evidence, model review, and the project presentation (upload pending). |
| **Artium** | Product framing, reproducibility review, and final-presentation readiness. |

These workstreams follow the [project charter](CHARTER.md); ownership does not imply sole authorship of every notebook. The proposed adaptive agent is a separate workstream from the implemented deterministic policy.

## How the project works

1. **Prepare the data:** organize patients, studies, and views; create patient-grouped internal train/validation splits from official MURA `train`.
2. **Build the baseline — Model A:** extract frozen ImageNet ResNet-18 embeddings, average them within each study, and fit Logistic Regression. Select regularization with patient-grouped cross-validation inside internal train.
3. **Evaluate the existing fine-tuned model — Model B:** use the saved ResNet-18 checkpoint and average per-view probabilities into one study probability.
4. **Apply the operating policy:** select each model's highest internal-validation threshold that reaches sensitivity ≥95%, then freeze that model-specific threshold.
5. **Support human review:** apply technical QC, declared-anatomy checks, and deterministic routing before presenting evidence in the local app. Routing can raise priority; human review remains required.

## Main result and model comparison

**Model B reached AUROC 0.913 on the current internal validation cohort.**

Both models below were evaluated on the **same 691 studies from 653 patients**, comprising 422 NORMAL and 269 ABNORMAL studies. This cohort comes from official MURA `train`; these are **development/validation results**, not an untouched final test.

| Metric | Model A — frozen ResNet-18 + Logistic Regression | Model B — fine-tuned ResNet-18, mean probability |
| --- | ---: | ---: |
| AUROC | 0.880 | **0.913** |
| AUPRC | 0.852 | **0.878** |
| Sensitivity at the ≥95% target | 95.2% | 95.2% |
| Specificity at that operating point | 39.3% | 58.3% |
| False negatives | 13 | 13 |
| False positives | 256 | 176 |
| Studies flagged at that operating point | 74.1% | 62.5% |
| Model-specific threshold, rounded | 0.063774 | 0.263568 |

Values come from the [current comparison CSV](docs/results/checkpoint1/model_comparison.csv). The paired patient-cluster bootstrap estimates a Model B minus Model A AUROC difference of **+0.032 (95% CI: +0.012 to +0.052)**. The observed specificity improvement is **not statistically supported on this cohort**; threshold selection used the same development population.

See the [full Checkpoint 1 results](docs/results/checkpoint1/README.md) for uncertainty intervals, paired comparisons, figures, training history, and the secondary max-pooling diagnostic. Mean probability pooling remains the primary Model B policy.

## Notebook map

| Notebooks | Purpose | Status |
| --- | --- | --- |
| **01–02** | Data preparation, EDA, preprocessing, and manifests | Canonical workflow |
| **03–04** | Model A baseline and evaluation | Canonical workflow |
| **05–06** | Verify the existing Model B bundle; evaluate and compare | Canonical workflow; keep 05 in `verify_existing` mode |
| **07** | Shortcut and geometry audit | Diagnostic |
| **08** | Local study viewer | Optional utility |
| [**09 — Deterministic Agent Policy**](notebooks/09_MURA_Deterministic_Agent_Policy.ipynb) | Demonstrate policy v1.1 and safety checks with synthetic inputs | Current policy demonstration; execution verification pending |
| [**09 — Deterministic Safety Tools**](notebooks/09_MURA_Deterministic_Safety_Tools.ipynb) | Existing deterministic safety smoke checks | Supporting notebook |
| [Legacy notebooks](notebooks/legacy/README.md) | Preserve earlier and contributor work, including Hagit's Agentic AI draft | Historical; outside the active runtime |
| [All-anatomy notebooks](notebooks/experimental_all_anatomies/README.md) | Explore extension beyond XR_WRIST | Experimental; outside the wrist MVP claim |

The [notebook guide](notebooks/README.md) links every canonical step and explains required artifacts and configuration. The [per-notebook reviews](notebooks/reviews/README.md) list dependencies and findings. These reviews are static: notebook execution and a complete sequential run have not been verified as part of this documentation work.

## Use the existing Model B artifacts

The canonical external bundle is **`DATA/MODEL_B_FINE_TUNING/`**. Its checkpoint is `best_resnet18_study_level.pt`, with mean probability aggregation and the frozen threshold `0.263567715883255`. The [bundle specification](configs/model_b_artifact_bundle.json) records required files and the checkpoint checksum.

Use Python **3.10 or 3.11** and install the [repository dependencies](requirements.txt) from the repository root:

```powershell
python -m pip install -r requirements.txt
```

Obtain the existing bundle from authorized storage, then verify your local copy using the [artifact setup instructions](models/README.md). Notebook 05 defaults to `verify_existing`; **do not rerun fine-tuning to set up or review this project**. Any older `models/resnet18/` location is historical and must not replace the canonical bundle.

The following files must remain available and unchanged for the current policy and bundle checks:

- [`tools/agent_policy.py`](tools/agent_policy.py) — policy version `1.1`.
- [`src/runtime/deterministic_safety.py`](src/runtime/deterministic_safety.py) — deterministic safety implementation.
- [`configs/model_b_artifact_bundle.json`](configs/model_b_artifact_bundle.json) — canonical bundle metadata.

Raw radiographs, model weights, private manifests, and per-study predictions stay outside Git.

## Wrist Review application

The [local app guide](apps/wrist_review/README.md) explains how to point `MURA_APP_MODEL_DIR` to an authorized local copy of the canonical bundle and launch the browser interface. It uses the verified Model B bundle and deterministic safety policy.

The app checks technical image quality and **declared** anatomy; the anatomy selector is not an automatic body-part detector. Missing or invalid artifacts prevent a model result. The current app does not implement the proposed adaptive Agentic AI layer. The new policy notebook demonstrates synthetic decisions without loading weights or executing TTA/Grad-CAM tools.

## Results, governance, and further reading

- [Full current results](docs/results/checkpoint1/README.md) — metrics, uncertainty, paired comparison, and diagnostic figures.
- [Test-set governance](reports/TEST_SET_GOVERNANCE.md) — official MURA `valid` was observed once for the historical Phase-1 reference and cannot serve as an untouched population for later selection.
- [Result lineage](docs/RESULT_LINEAGE.md) — current versus historical splits, operating policies, and model boundaries.
- [Documentation index](docs/README.md) — reviewer materials and supporting project documentation.
- [Reports index](reports/README.md) — current results, historical evidence, experiments, governance, and reviews.
- [Project charter](CHARTER.md) — scope, success criteria, team workstreams, and the proposed agent design.
