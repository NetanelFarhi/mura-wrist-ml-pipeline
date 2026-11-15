# Runnable applications

This folder contains the runnable product interfaces of the project. It is kept
separate from the analytical and reusable layers required by the project
structure:

- [`wrist_review/`](wrist_review/) is the local browser application used to
  create a human-review worklist from supported wrist studies. It owns the UI,
  local server, launch instructions, and app-specific static assets.
- [`../src/`](../src/) contains reusable runtime functions used by the app and
  other project workflows.
- [`../agents/`](../agents/) contains the agent specification, prompts,
  guardrails, and implementation records.
- [`../evaluation/`](../evaluation/) contains the evaluation plans and test
  materials.
- [`../presentation/`](../presentation/) is reserved for the final slide deck
  and a reviewed, presentation-safe demo backup.

The application is intentionally not placed in `notebooks/`: notebooks document
the research workflow, whereas `apps/wrist_review/` is the runnable product
surface.
