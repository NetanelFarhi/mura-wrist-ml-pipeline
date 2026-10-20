# Legacy notebooks

This folder preserves earlier notebooks that informed the project but are not
the current reproducible workflow. They are retained for traceability only.

- `00_data_access_audit.ipynb` and `01_eda.ipynb` are early local-data
  scaffolds.
- `02_preprocessing.ipynb`, `03_MURA_Baseline_ResNet18_LogReg.ipynb`, and
  `04_MURA_Model_Evaluation.ipynb` are earlier versions of the baseline path.
- The two `Hodaya*` notebooks preserve a pilot fine-tuning run and an earlier
  evaluation copy. They are not the canonical Model B protocol.

Use the numbered notebooks in the parent folder for current work. Historical
notebooks must not be used to make new model-, threshold-, or test-set
selection decisions without first reconciling their split and evaluation
protocol with the current workflow.

Their execution outputs have been cleared before publication so that raw
radiographs, identifiers, Drive paths, widgets, and stale result tables are
not redistributed through Git. The original historical result summaries are
preserved separately under `reports/`.
