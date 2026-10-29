# Incoming notebooks review — 2026-10-29

This is a local integration review of the two supplied notebooks. It records
what can be reused without rewriting historical work or treating an already
observed official-valid split as a new test.

## Inputs reviewed

- `08_MURA_Official_Validation.ipynb`
- `09_MURA_Agentic_AI.ipynb`
- Shared Drive `DATA/MODEL_B_FINE_TUNING/`
- Existing canonical Checkpoint 1 artifacts and governance documents

## Model-artifact finding and remediation

At the start of this review, shared Drive `DATA/MODEL_B_FINE_TUNING/` contained the separate
2026-10-19 experiment: `resnet18_model_b_best.pt`, a partial fine-tuning run
with layer3/layer4/fc trainable. Its saved internal-validation ranking metrics
are AUROC 0.901461 and AUPRC 0.872082. It does not contain the artifact bundle
expected by the canonical Notebooks 05–06.

The canonical Checkpoint 1 Model B bundle instead contains
`best_resnet18_study_level.pt`, `selection.json`, the frozen primary policy,
training history, and mean/max/internal-validation prediction files. Its
checkpoint SHA-256 is recorded in
`configs/model_b_artifact_bundle.json`. The 2026-10-19 experiment was preserved
under `DATA/EXPERIMENT_PARTIAL_FT_2026-10-19/`; it was not deleted. The
canonical metadata, policy, history, and internal-validation prediction files
were then restored to `DATA/MODEL_B_FINE_TUNING/`.

The canonical checkpoint is 134 MB, larger than the connected Drive upload
limit. It is therefore stored as verified parts in `checkpoint_parts/`, with a
small standalone reassembly utility. Rebuild it once in the artifact directory
before a shared Drive/Colab runtime loads Model B. Its required name and
SHA-256 are `best_resnet18_study_level.pt` and
`63AEDC6BEC1534BDED7C3CA4D0DAF03AFC17BF92166CB7848506A6F3A82DE5B6`.
Notebooks 05–06 now fail closed until the complete bundle verifies.

## Notebook 08 — retain ideas, do not merge as a canonical evaluation

Useful components:

- image-only inference before labels are joined;
- study-level mean probability pooling;
- separation of inference records from label matching;
- integrity checks for study/image overlap.

It must not become the current official result notebook because it loads a
different checkpoint from `DATA/RESNET18_FINE_TUNING/resnet18_best.pt`, uses a
different preprocessing path and threshold (`0.060089...`), and evaluates the
already-observed official MURA valid split again. Its displayed results are
therefore historical exploratory output, not the canonical Checkpoint 1 Model
B result and not a new untouched test result.

The generic label-blind study-inference pattern can be reused later against
the canonical bundle, with only authorised runtime data and without using
official-valid labels for any selection.

## Notebook 09 — retain deterministic tool ideas, not an adaptive agent claim

Useful deterministic components:

- technical input-QC prototype;
- complete-study inference interface;
- view-probability disagreement summary;
- anatomy/scope blocking before classification;
- tests showing unsupported or missing anatomy does not reach the classifier.

Required changes before any integration:

1. Load only the verified canonical Model B bundle and its exact
   preprocessing/threshold; never the `RESNET18_FINE_TUNING` checkpoint used
   by the supplied notebook.
2. Do not use official-valid records or ground truth as the Agent's working
   evaluation set.
3. Keep QC, support checks, and mandatory human-review routing deterministic
   and non-overridable.
4. Do not call this an adaptive agent yet. It is a useful deterministic tool
   prototype; the bounded adaptive layer and its independent evaluation remain
   planned as defined in `agents/AGENT_DESIGN.md` and
   `evaluation/AGENT_EVAL_PLAN.md`.
5. Treat paths, metadata, OCR text, and documents as untrusted data; they
   cannot alter tool policy or guardrails.

## Integration order

1. Restore and verify the canonical Model B bundle in shared Drive.
2. Make Notebook 05 fail closed if the canonical bundle is missing, instead
   of silently training another model.
3. Make Notebook 06 verify artifact identity before comparing Model A and B.
4. Reuse the deterministic tool ideas only after Step 1, with synthetic and
   internal-development tests rather than official-valid selection.
5. Keep the original incoming notebooks outside the canonical notebook path
   until their provenance is fully reconciled.
