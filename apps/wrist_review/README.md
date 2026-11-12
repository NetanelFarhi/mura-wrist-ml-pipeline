# Wrist Review worklist app

Wrist Review is a local, browser-based research prototype for **worklist triage**.
It receives a local folder of de-identified wrist studies, keeps every study's views
together, and produces an ordered queue for human review. It does not diagnose a
fracture and it never clears a case from clinical review.

The app calls the verified Checkpoint 1 Model B bundle and the deterministic safety
policy first. A **bounded local ambiguity investigation** then runs only when the
real views disagree across the frozen threshold. It performs one approved TTA
stability check and returns structured evidence. It cannot change the model score,
threshold, deterministic route, or mandatory human-review requirement. This is not
yet an adaptive LLM/LangGraph agent and must not be presented as one.

## Local patient-folder layout

Choose one patient folder at a time. After each selection, use **Add patient
folder** in the app to append another patient to the same worklist. Alternatively,
create one ZIP with several patient folders and select **Add ZIP worklist**. The
ZIP is read in process memory only; it is not written to disk. Both layouts below
are supported:

```text
WORKLIST/
  CASE-001/
    view-1.png
    view-2.png

WORKLIST/
  PATIENT-001/
    STUDY-001/
      view-1.png
      view-2.png
```

The immediate image folder is treated as one study. A study may contain one to
eight views. Selecting the same patient/study again does not duplicate it in
the worklist. A study with duplicate image names or more than eight views is
not sent to the model and is shown as requiring input attention.

Folder and file names are displayed locally in the interface, so use local aliases
only — never patient identifiers or other identifying information. For a ZIP,
preserve the same folder tree and use a maximum of 100 MB compressed, 50 MB
expanded, 400 files, and 15 MB per image.

## Run locally

Install repository dependencies, place the authorized canonical Model B bundle outside
Git, then run the server from the repository root:

```powershell
$env:MURA_APP_MODEL_DIR = "C:\\authorized-local-artifacts\\MODEL_B_FINE_TUNING"
python apps/wrist_review/server.py
```

Open `http://127.0.0.1:8601` in a browser. The default artifact location is
`local_artifacts/MODEL_B_FINE_TUNING/`, which is intentionally not committed. The
required files and checksum are specified in
[`configs/model_b_artifact_bundle.json`](../../configs/model_b_artifact_bundle.json).

## What happens to each study

1. The browser appends each selected patient folder to the same local worklist,
   groups its images by study folder, and sends one study at a time to the local
   server.
2. Technical QC and the declared-anatomy rule run before Model B inference.
3. For a supported study, every view receives a score; their mean is compared with the
   frozen threshold.
4. If view scores straddle the frozen threshold, the bounded ambiguity workflow runs
   a small brightness/contrast TTA check on the view closest to that threshold. It
   adds evidence only; it never modifies the original route.
5. The worklist is ordered as `RAISE_PRIORITY`, input-attention cases, then
   `STANDARD_HUMAN_REVIEW`. Within a route, higher study scores appear first.
6. Every route requires human review. Inputs stay in process memory for the current
   request and are not written by the application.

## Safety boundaries

- The anatomy selector is a declared routing input, not automatic body-part recognition.
- Unsupported, corrupt, tiny, or near-uniform inputs fail safely without a model result.
- The app verifies the Model B artifact checksum before loading inference.
- A model flag can raise queue priority only. It cannot diagnose, localize a finding, or
  replace human review.
- The bounded ambiguity workflow has a maximum of one optional TTA call and runs only
  after successful QC, supported anatomy, inference, and view-level disagreement.
- Do not use official MURA `valid` labels for app or Agent selection.

See the [app overview](../../docs/app/Wrist_Review_App.md) and the
[test-set governance record](../../reports/TEST_SET_GOVERNANCE.md) for project boundaries.
The exact implementation status is recorded in
[AGENT_IMPLEMENTATION_STATUS.md](../../agents/AGENT_IMPLEMENTATION_STATUS.md).
