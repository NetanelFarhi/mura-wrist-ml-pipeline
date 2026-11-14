# Presentation and demo preparation

No final slide deck or demo recording is committed yet. This folder exists so
the final presentation and its non-sensitive demo fallback have one clear,
reviewable location.

## Recommended ten-slide outline

1. Project title, one-line value proposition, and team names.
2. User, workflow problem, and honest MURA-abnormality scope.
3. The working deterministic system: study -> model -> safety route -> human.
4. Architecture, including what the future bounded agent may and may not do.
5. Data, patient/study/view structure, and patient-safe evaluation protocol.
6. Model performance: distinguish the Model A/B internal-validation comparison
   in `docs/results/checkpoint1/` from the frozen Model B internal-versus-official
   comparison in `reports/official_valid_final/report.md`. Model B sensitivity:
   95.2% internal vs 89.7% official (95% CI 82.8–95.3%); AUROC: 0.913 vs 0.903.
   The official set had prior exposure. Possible case-mix and sampling effects
   are hypotheses, not established explanations.
7. Product/interface walkthrough using only sanitized local UI screens.
8. Limitations and what did not improve: official sensitivity below the 95% target in the point estimate,
   attention/partial-fine-tuning results, calibration and shortcut caveats.
9. Next step, agent-evaluation plan, and transparent future cost assumptions.
10. Closing conclusion and required human-review safety boundary.

## Required demo fallback

Before the final presentation, add a short recording that demonstrates only:

1. selecting non-sensitive placeholder images;
2. deterministic QC/anatomy routing; and
3. the invariant that every route retains human review.

Do not record MURA radiographs, identifiers, credentials, local Drive paths,
or a fabricated patient/model outcome. The app should show its intentional
`model_bundle_missing` state if the verified local artifact bundle is absent.
