# Wrist Review worklist app

Wrist Review is a local, browser-based research prototype for **worklist triage**.
It receives a local folder of de-identified wrist studies, keeps every study's views
together, and produces an ordered queue for human review. It does not diagnose a
fracture and it never clears a case from clinical review.

The app calls the verified Checkpoint 1 Model B bundle and deterministic safety
policy first. A **bounded local evidence investigation** is authorized only by the
explicit policy in [`app_optional_tool_policy.json`](../../configs/agentic_ai/app_optional_tool_policy.json):
an abnormal mean study prediction, view scores that disagree across the frozen
threshold, or a mean within the displayed borderline band around that threshold.
When authorized, it runs one local TTA stability check and one in-memory Grad-CAM
summary. Neither tool can change the model score, threshold, deterministic route, or
mandatory human-review requirement.

The initial borderline band is **±5 percentage points** around the frozen threshold.
It is a transparent workflow setting, not a learned result: if the team changes it,
they must edit the JSON policy, review that decision, and re-run the app checks.

An **optional CrewAI research explanation** can read a de-identified summary for
every valid supported study. It is off by default and requires an operator to
configure it. A consistent normal study therefore gets a no-tools explanation;
an authorized study may receive only the precomputed TTA and Grad-CAM **summaries**.
The CrewAI layer receives no radiograph
pixels, local paths, file names, labels, or patient identifiers; it may not run
inference or modify Model B, the threshold, queue route, or review requirement.

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

### Optional CrewAI research explanation

The default application is fully local. Do **not** enable this option until the
team has approved the provider, credentials, and research-data boundary. To use
the bounded CrewAI explanation, install the
separate Agent dependencies and configure credentials only in your own local
shell (never in a notebook, source file, or Git):

```powershell
pip install -r requirements_agent.txt
$env:MURA_CREWAI_ENABLED = "1"
$env:MURA_CREWAI_LLM_MODEL = "gemini/gemini-3.1-flash-lite"
$env:GOOGLE_API_KEY = "your-local-provider-key"
python apps/wrist_review/server.py
```

The server creates a fresh single-agent CrewAI run for each valid supported study.
It permits at most three LLM-call reservations and three agent iterations per study.
The agent receives no optional tool for a consistent normal study. When the fixed
policy authorizes evidence, it can retrieve only precomputed TTA and Grad-CAM
summaries; Grad-CAM maps themselves are never sent or stored. If
configuration, a provider call, the response schema, or the evidence contract fails,
the deterministic local route remains in force and the research explanation is marked
unavailable.

## What happens to each study

1. The browser appends each selected patient folder to the same local worklist,
   groups its images by study folder, and sends one study at a time to the local
   server.
2. Technical QC and the declared-anatomy rule run before Model B inference.
3. For a supported study, every view receives a score; their mean is compared with the
   frozen threshold.
4. If the explicit optional-evidence policy is triggered by an abnormal mean,
   views that straddle the frozen threshold, or a configured borderline mean, the
   bounded workflow runs a small brightness/contrast TTA check on the view closest
   to that threshold and an in-memory Grad-CAM summary for each view. They add
   evidence only; neither modifies the original route or claims lesion localization.
5. If enabled explicitly, a fresh CrewAI instance produces a validated,
   de-identified research explanation for every valid supported study. It receives
   no optional tool for a consistent normal study and may retrieve the precomputed
   TTA and Grad-CAM summaries only when the fixed policy authorizes both. It is not a new classifier
   and does not receive images or change the route.
6. The worklist is ordered as `RAISE_PRIORITY`, input-attention cases, then
   `STANDARD_HUMAN_REVIEW`. Within a route, higher study scores appear first.
7. Every route requires human review. Inputs stay in process memory for the current
   request and are not written by the application.

## Safety boundaries

- The anatomy selector is a declared routing input, not automatic body-part recognition.
- Unsupported, corrupt, tiny, or near-uniform inputs fail safely without a model result.
- The app verifies the Model B artifact checksum before loading inference.
- A model flag can raise queue priority only. It cannot diagnose, localize a finding, or
  replace human review.
- The bounded local workflow has a maximum of two fixed tools: one TTA check and
  one in-memory Grad-CAM summary. They run only after successful QC, supported
  anatomy, inference, and an explicit policy trigger (abnormal mean, view-level
  disagreement, or configured borderline mean).
- The optional CrewAI bridge uses a fresh agent/task/crew per valid supported study.
  It has no optional tool for a consistent normal study and can retrieve only
  precomputed TTA and Grad-CAM summaries for an authorized study; it has no model
  or route authority.
- A CrewAI configuration or response failure cannot cause a priority reduction or a
  clearance; the app retains the deterministic route and human review.
- Do not use official MURA `valid` labels for app or Agent selection.

See the [app overview](../../docs/app/Wrist_Review_App.md) and the
[test-set governance record](../../reports/TEST_SET_GOVERNANCE.md) for project boundaries.
The exact implementation status is recorded in
[AGENT_IMPLEMENTATION_STATUS.md](../../agents/AGENT_IMPLEMENTATION_STATUS.md).
