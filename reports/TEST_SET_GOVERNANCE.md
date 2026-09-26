# Test-set governance

## Historical Phase-1 use of official MURA `valid`

Official MURA `valid` was used **once** as the frozen Phase-1 reference evaluation. Before it was opened, the project used official MURA `train` to create patient-level internal training and validation populations; selected Model A's regularization, Model B's checkpoint/aggregation, and both operating thresholds on internal validation only. This followed the intended threshold-selection logic.

After official-valid was opened, the project observed the Phase-1 study-level predictions, metrics/bootstraps, model comparison, and descriptive error analyses. Those results are preserved as historical evidence in the repository.

## Rule for all later work

Because the official-valid result has now been observed, it is not an untouched evaluation population for later model, threshold, calibration, architecture, aggregation, or agent selection. Phase 2 and future development may cite it only as a historical frozen reference; they must not inspect its labels to choose anything. This applies even when a new candidate is compared with an old Phase-1 result.

## Open methodological decision for advanced claims

If a genuinely untouched advanced-model evaluation population is required, the team and lecturer must choose and pre-register one of these scientifically defensible options:

1. patient-grouped development cross-validation with an explicit transparent reuse limitation;
2. compatible external validation data after confirming task/anatomy/label compatibility; or
3. a lecturer-approved final evaluation protocol.

Do not silently create a “new test” from data already used for development and call it untouched.
