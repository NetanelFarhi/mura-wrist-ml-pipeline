# Review: `09_MURA_Deterministic_Agent_Policy.ipynb`

[Open notebook source](../09_MURA_Deterministic_Agent_Policy.ipynb)

**Classification:** Active deterministic policy demonstration  
**Purpose:** Exercise policy v1.1 and deterministic safety with synthetic records, using the threshold metadata from the canonical Model B bundle.

## Inputs and execution

Repository files: `tools/agent_policy.py`, `src/runtime/deterministic_safety.py`, and `configs/model_b_artifact_bundle.json`. Runtime dependencies: Python, NumPy and Pillow. No checkpoint or MURA data is read.

The notebook is new and has not been executed. Its assertions cover a valid priority record, a valid routine record, a malformed record that must safety-block, synthetic image QC, and anatomy scope. No fine-tuning, LLM call, TTA or Grad-CAM tool execution is included.

## Findings

This does not change the protected Python files or the existing `09_MURA_Deterministic_Safety_Tools.ipynb`. The canonical checkpoint metadata supplies the frozen threshold; weights remain at `DATA/MODEL_B_FINE_TUNING/` and are not loaded. Run a safe smoke check before relying on the notebook's outputs.
