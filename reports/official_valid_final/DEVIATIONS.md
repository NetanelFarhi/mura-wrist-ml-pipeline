# Deviations from PREREGISTRATION.md — official MURA `valid` final run

Registration commit: `7d860849d643a96c0efdf2869923818bfe55ecec`. Official run folder: `official_20261114T203242Z`. `PREREGISTRATION.md` itself was not changed.

## 1. Run notebook changed after registration (14.11.2026)

**What the pre-registration says.** The run notebook and `PREREGISTRATION.md` "are not changed" after they enter `main`.

**What happened.** The first attempt with `MODE = "official"` stopped in Cell 2 with `KeyError: 'label'` while the notebook was parsing the manifest `wrist_official_validation.csv`. That manifest lists files only (`file_id, file_name, size, added_at, md5_hash`); it has no `label` and `patient_id` columns, which the internal-validation manifest has.

**What was and was not touched.** Only the manifest (a list of file paths) was read. The stop came before the model was loaded: no image of official `valid` was processed by the model, and no prediction, metric, `summary.json` or `official_*` folder was produced.

**The change.** Four lines were added to Cell 2 of `notebooks/10_MURA_Official_Valid_Final_Evaluation.ipynb`: when the columns are absent, the label and the patient are taken from the file path (`patientN/studyM_positive|negative`), which is how MURA defines them. The fixed notebook was committed to `main` before the official run (commit "Fix notebook 10: read label and patient from the path in the official manifest").

**What did not change.** Model, weights, study aggregation, threshold, agent policy, deterministic rules, metrics, bootstrap and the reading rule. `CODE_COMMIT` stayed the registration commit.

**Checks.**
- After the fix, the internal check passed 39 of 39, with prediction files byte-identical to the check before the fix (`internal_20261114T201413Z` vs `internal_20261114T194657Z`).
- In the official run the notebook confirmed the expected population: 237 studies, 659 images, 207 patients (97 ABNORMAL, 140 NORMAL); 38 of 38 checks passed.

**Effect on the result.** None on the numbers: the fix only makes the manifest readable. It is still a deviation, and it is reported here and in the root `README.md`.

## No other deviations

One official run was completed. Nothing was changed after its result was seen.
