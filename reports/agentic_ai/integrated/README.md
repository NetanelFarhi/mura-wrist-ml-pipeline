# Integrated agent-report examples

This directory keeps only deidentified, aggregate JSON examples of the
deterministic Model B and agent-evidence workflow. They are included to make the
report schema and safety boundary reviewable without publishing source-study
paths, image-level predictions, Grad-CAM locations, raw images, labels, or
person-like identifiers.

The three `MURA_WRIST_AGENT_EXAMPLE_*` files describe the same representative
research example at different levels of detail:

- `complete_research_report`: mandatory checks, frozen classification, optional-evidence summary, and limitations;
- `final_integrated_research_report`: compact handoff across the deterministic and optional-evidence layers;
- `frozen_result`: minimal deterministic result before a research explanation.

These are research-prototype artifacts, not clinical reports or independent
evaluation evidence. Detailed per-study runtime outputs remain local and are
deliberately excluded by the repository data-governance check.
