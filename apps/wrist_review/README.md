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

When that explanation completes successfully, the selected-study panel offers
**Download PDF report**. The PDF is generated in memory for that request only;
it contains the model score and fixed route, the LLM explanation, the evidence
provided to it, and its stated limitations. It deliberately contains **no
radiographs, image paths, file names, labels, or patient identifiers**. If
CrewAI is disabled, unavailable, or returns an invalid response, there is no
PDF button because the app must not present a generated LLM report that does
not exist.

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

## Reading the worklist

After review, each study has a short, visible workflow state in both the queue
and the selected-study panel:

- **Routine case**: a consistent score with no optional evidence trigger.
- **Ambiguous views**: view scores fall on different sides of the frozen
  threshold; TTA and Grad-CAM summaries are authorized as evidence only.
- **Borderline score**: the mean score falls in the configured borderline band;
  the same bounded evidence tools are authorized.
- **Priority signal**: the mean score crossed the frozen threshold.
- **Unsupported anatomy** or **Input attention**: the safety layer stopped
  before inference; no optional evidence tool runs.

These are workflow descriptions, not diagnoses. An optional tool label shows
what was actually used for that study; it never means that the tool changed the
score, frozen threshold, route, or human-review requirement.

### Agent & evidence report in the app

The results page includes an **Agent & evidence report** for every study. It
makes the separation between the two layers explicit:

- The **deterministic safety/model layer** runs first for every supported input
  and owns the final worklist route.
- The **bounded local evidence workflow** runs TTA and Grad-CAM only after one
  of the three fixed triggers above. For a consistent normal case, it records
  `No extra tools needed`; it does not run either optional tool.
- The separate **CrewAI research explanation** is opt-in and defaults to off.
  Its status is reported separately; it is never required for a deterministic
  route and cannot modify one.

Each row shows the actual trigger, local tools used, and CrewAI status. This is
an execution record for the prototype, not a clinical recommendation.

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

The simplest local setup is deliberately separate from Git and does not save a
provider key. From `apps/wrist_review`, run:

```powershell
.\install_agent_runtime.ps1
.\start_with_crewai.ps1 -ModelDir "C:\authorized-local-artifacts\MODEL_B_FINE_TUNING"
```

The launcher asks for the Google AI Studio key in the local PowerShell window,
keeps it only for that running process, enables the CrewAI bridge, and starts
the app. It checks that the selected folder contains the canonical Model B
weights before starting. Do not paste a key into a notebook, source file, Git,
or chat. CrewAI telemetry is disabled for this launcher. CrewAI may create its
own local runtime state under the operator's application-data folder; this is
outside Git and never contains the uploaded radiographs.

Advanced users can configure the equivalent environment variables manually:

```powershell
$env:MURA_CREWAI_ENABLED = "1"
$env:MURA_CREWAI_LLM_MODEL = "gemini/gemini-3.1-flash-lite"
$env:GOOGLE_API_KEY = "your-local-provider-key"
$env:CREWAI_DISABLE_TELEMETRY = "true"
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

The PDF export uses `reportlab`, included in `requirements_agent.txt`. It is
available only after a completed, schema-validated explanation; generating the
PDF makes no additional LLM call and does not write a copy of the report to disk.

### Recorded demo and example report

The [`demo/`](demo/) folder contains two companion artifacts for the current
application version:

- [`Wrist Review demo (1.5x, with report ending)`](demo/Wrist_Review_Demo_1_5x_with_PDF.mp4)
  shows the local multi-study worklist, deterministic priority order, optional
  evidence states, and the generated report as its final six seconds. Playback
  is intentionally 1.5x and the report page is held separately for reading.
  It is a technical app-version capture, not the reviewed final-presentation
  fallback; see [`presentation/README.md`](../../presentation/README.md).
- [`Example generated research report`](demo/MURA_WRIST_NO_TOOLS_001_research_explanation.pdf)
  is the de-identified PDF exported by the app after a successful no-tools
  CrewAI explanation. It is an execution example, not an evaluation result or
  a clinical recommendation.

The demo uses local study aliases only. It does not include original MURA paths,
ground-truth labels, patient identifiers, or any radiograph pixels in the PDF.

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
6. A completed explanation can be exported as a de-identified, in-memory PDF.
   A disabled, unavailable, or invalid explanation cannot produce a PDF.
7. The worklist is ordered as `RAISE_PRIORITY`, input-attention cases, then
   `STANDARD_HUMAN_REVIEW`. Within a route, higher study scores appear first.
8. Every route requires human review. Inputs stay in process memory for the current
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
