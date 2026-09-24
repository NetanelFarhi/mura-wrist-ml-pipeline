# Model artifacts

This directory is intentionally empty in source control. Training scripts write only local artifacts here:

- `frozen_logistic.joblib` and its metadata for Model A.
- `resnet18_best.pt` and its validation-selection metadata for Model B.

Model selection and threshold selection must use training and validation data only. Do not overwrite a final-test result with a new tuning run.

