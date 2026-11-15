# Wrist Review

Local browser prototype for ordering wrist studies for **human review**. It uses
the verified Model B bundle and deterministic safety checks; it is not a
diagnostic tool and never clears a case from human review.

## What it does

1. Add patient folders one at a time, or upload one ZIP worklist.
2. Run QC, declared-wrist scope check, Model B, and mean view aggregation.
3. Order cases for review. An abnormal mean, view disagreement, or a borderline
   mean authorizes local TTA and Grad-CAM evidence; those tools cannot change the
   score, threshold, route, or need for human review.
4. Optionally request a deidentified CrewAI research explanation. A PDF download
   appears only when that explanation successfully completes.

The CrewAI bridge is off by default. It receives no images, paths, file names,
labels, or identifiers, and it cannot run inference or alter the deterministic
route. A consistent normal case uses no optional tools.

## Run locally

Place the authorized Model B bundle outside Git, set its location, then run:

```powershell
$env:MURA_APP_MODEL_DIR = "C:\authorized-local-artifacts\MODEL_B_FINE_TUNING"
python apps/wrist_review/server.py
```

Open `http://127.0.0.1:8601`. Required artifact names and checksum are in
[`configs/model_b_artifact_bundle.json`](../../configs/model_b_artifact_bundle.json).

### Optional CrewAI explanation

From this folder:

```powershell
.\install_agent_runtime.ps1
.\start_with_crewai.ps1 -ModelDir "C:\authorized-local-artifacts\MODEL_B_FINE_TUNING"
```

The launcher asks for a Google AI Studio key only in the local terminal and does
not save it. Do not place keys in Git, source files, or notebooks.

## Input layout

Use local aliases only. A study may contain 1–8 views.

```text
WORKLIST/
  CASE-001/
    view-1.png
    view-2.png
```

For a ZIP, each top-level folder can be a patient/case and contain one or more
study folders. Uploads are processed in memory only.

## Demo

[`demo/`](demo/) contains the recorded app walkthrough and one deidentified
example PDF report. The technical capture is separate from the final-presentation
backup described in [`presentation/README.md`](../../presentation/README.md).

More project context: [app overview](../../docs/app/Wrist_Review_App.md) and
[implementation status](../../agents/AGENT_IMPLEMENTATION_STATUS.md).
