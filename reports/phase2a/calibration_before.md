# Calibration audit (validation only)

Uncalibrated: Brier=0.109157; NLL=0.373411; ECE (10 bins)=0.048050; AUROC=0.910715; AUPRC=0.884109.

Cross-fitted temperature scaling (5 folds; temperatures: 1.4764, 1.3012, 1.4651, 1.2984, 1.4169): Brier=0.109844; NLL=0.361675; ECE=0.039236; AUROC=0.909275; AUPRC=0.882161.

The scaled metrics are cross-fitted: each study was calibrated by a temperature fitted without that study. Temperature scaling is monotonic, so any unchanged ranking metric is expected and is not presented as a ranking improvement.
