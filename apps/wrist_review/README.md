# Wrist Review local app

This folder contains the source of the local-only, browser-based Wrist Review
research prototype. It calls only the verified, frozen Checkpoint 1 Model B
bundle and the deterministic safety policy. It is not a clinical deployment
and it does not implement the planned adaptive Agentic AI layer.

## Run locally

Install the repository dependencies, place the authorized canonical Model B
bundle outside Git, then run the server from the repository root:

```powershell
$env:MURA_APP_MODEL_DIR = "C:\authorized-local-artifacts\MODEL_B_FINE_TUNING"
python apps/wrist_review/server.py
```

Open `http://127.0.0.1:8601` in a browser.

The default artifact location is
`local_artifacts/MODEL_B_FINE_TUNING/`, which is intentionally not committed.
The expected files and checksum are specified in
[`configs/model_b_artifact_bundle.json`](../../configs/model_b_artifact_bundle.json).

## Data and safety boundaries

- Image bytes stay in process memory for the local request only.
- The UI supports 1–8 views and never turns padding into a real view.
- QC and declared-anatomy checks execute before model inference.
- Unsupported/corrupt/missing inputs route safely without model inference.
- A model route may raise priority, never lower or replace human review.
- Do not use official MURA `valid` labels for any app or agent selection.

See the [app overview](../../docs/app/Wrist_Review_App.md) for the screen flow
and the [test-set governance](../../reports/TEST_SET_GOVERNANCE.md) for the
historical-evaluation limitation.
