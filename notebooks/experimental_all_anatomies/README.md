# Experimental: all-anatomy MURA extension

These notebooks evaluate a development-only extension from the project's
XR_WRIST MVP to all seven MURA anatomy categories. They are separate from the
canonical wrist pipeline and must not be described as final test results.

The sequence is:

1. `10_MURA_All_Anatomies_Data_Preparation.ipynb`
2. `11_MURA_All_Anatomies_Preprocessing.ipynb`
3. `12_MURA_All_Anatomies_Baseline_ResNet18_LogReg.ipynb`
4. `13_MURA_All_Anatomies_Development_Evaluation.ipynb`
5. `14_MURA_All_Anatomies_FineTuned_ResNet18.ipynb`
6. `15_MURA_All_Anatomies_Model_Comparison.ipynb`
7. `16_MURA_All_Anatomies_Shortcut_Audit.ipynb`

Raw MURA images, model weights, manifests containing local paths, and
per-study predictions are intentionally excluded from Git. Run the scripts
and notebooks only with authorized local data and model artifacts.
