"""Contract tests for the Checkpoint-1 agent stub (no MURA data needed)."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.pipeline.agent_stub import (
    CONTRACT_VERSION,
    STUB_LIMITATION,
    AgentReport,
    ModelOutput,
    agent_stub,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_07 = REPO_ROOT / "notebooks" / "07_MURA_Agent_Stub_Pipeline.ipynb"


def make_output(probability=0.4, threshold=0.3, views=None):
    return ModelOutput(
        study_id="patient11185/study1_positive",
        model_version="model_b@0123456789ab",
        study_probability=probability,
        threshold=threshold,
        view_probabilities=views,
    )


class AgentStubContractTest(unittest.TestCase):
    def test_above_threshold_is_flagged(self):
        report = agent_stub(make_output(probability=0.40, threshold=0.30))
        self.assertEqual(report.model_flag, "ABOVE_THRESHOLD")

    def test_below_threshold_is_not_flagged(self):
        report = agent_stub(make_output(probability=0.10, threshold=0.30))
        self.assertEqual(report.model_flag, "BELOW_THRESHOLD")

    def test_equal_to_threshold_is_flagged_like_the_evaluation_notebooks(self):
        report = agent_stub(make_output(probability=0.30, threshold=0.30))
        self.assertEqual(report.model_flag, "ABOVE_THRESHOLD")

    def test_stub_status_limitations_and_unchanged_evidence(self):
        x = make_output(views=[0.2, 0.7])
        report = agent_stub(x, trace_id="t-1")
        self.assertIsInstance(report, AgentReport)
        self.assertEqual(report.agent_status, "STUB")
        self.assertEqual(report.limitations, (STUB_LIMITATION,))
        self.assertIs(report.evidence, x)
        self.assertEqual(report.case_id, x.study_id)
        self.assertEqual(report.trace_id, "t-1")
        self.assertEqual(report.contract_version, CONTRACT_VERSION)

    def test_trace_id_is_generated_when_missing(self):
        first, second = agent_stub(make_output()), agent_stub(make_output())
        self.assertTrue(first.trace_id)
        self.assertNotEqual(first.trace_id, second.trace_id)

    def test_report_is_json_serialisable(self):
        record = json.loads(agent_stub(make_output(views=[0.2, 0.7]), trace_id="t").to_json())
        self.assertEqual(record["evidence"]["view_probabilities"], [0.2, 0.7])
        self.assertEqual(record["limitations"], [STUB_LIMITATION])
        self.assertEqual(record["agent_status"], "STUB")

    def test_model_a_output_without_views_is_valid(self):
        record = agent_stub(make_output(views=None)).to_dict()
        self.assertIsNone(record["evidence"]["view_probabilities"])

    def test_invalid_inputs_are_rejected(self):
        for kwargs in (
            {"probability": 1.2},
            {"probability": -0.1},
            {"probability": float("nan")},
            {"threshold": 1.5},
            {"views": []},
            {"views": [0.2, 2.0]},
        ):
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}), self.assertRaises(ValueError):
                make_output(**kwargs)
        with self.assertRaises(ValueError):
            ModelOutput(study_id=" ", model_version="m", study_probability=0.1, threshold=0.2)
        with self.assertRaises(TypeError):
            agent_stub({"study_id": "x"})

    def test_model_output_is_immutable(self):
        x = make_output(views=[0.1])
        with self.assertRaises(Exception):
            x.study_probability = 0.9  # type: ignore[misc]
        self.assertIsInstance(x.view_probabilities, tuple)


class NotebookCopyTest(unittest.TestCase):
    """Notebook 07 runs in Colab without the repo, so it carries a copy of the
    module. This test keeps that copy identical to src/pipeline/agent_stub.py."""

    def test_notebook_07_copy_matches_module(self):
        if not NOTEBOOK_07.exists():
            self.skipTest("notebook 07 not present")
        notebook = json.loads(NOTEBOOK_07.read_text(encoding="utf-8"))
        magic = "%%writefile agent_stub.py\n"
        copies = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code" and "".join(cell["source"]).startswith(magic)
        ]
        self.assertEqual(len(copies), 1, "expected exactly one %%writefile agent_stub.py cell")
        module_text = (REPO_ROOT / "src" / "pipeline" / "agent_stub.py").read_text(encoding="utf-8")
        self.assertEqual(copies[0][len(magic):].strip(), module_text.strip())


if __name__ == "__main__":
    unittest.main()
