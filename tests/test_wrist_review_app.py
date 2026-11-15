"""Safety checks for the local Wrist Review app configuration.

These tests do not load a checkpoint or any radiograph.  They verify that the
local app resolves its repository paths predictably and refuses inference when
the required, non-versioned Model B bundle is absent.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import zipfile
from pathlib import Path

from PIL import Image

from src.runtime.deterministic_safety import StudyImage


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = REPOSITORY_ROOT / "apps" / "wrist_review" / "core.py"
SERVER_PATH = REPOSITORY_ROOT / "apps" / "wrist_review" / "server.py"
RESEARCH_PDF_PATH = REPOSITORY_ROOT / "apps" / "wrist_review" / "research_report_pdf.py"


def load_app_core():
    """Load the standalone app adapter without requiring an apps package."""

    module_name = "wrist_review_app_core_for_test"
    specification = importlib.util.spec_from_file_location(module_name, CORE_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def load_app_server():
    """Load the local HTTP shell so archive handling can be unit-tested."""

    module_name = "wrist_review_app_server_for_test"
    app_directory = str(SERVER_PATH.parent)
    sys.path.insert(0, app_directory)
    try:
        specification = importlib.util.spec_from_file_location(module_name, SERVER_PATH)
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        specification.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(app_directory)


def load_research_pdf_exporter():
    """Load the PDF presentation layer without starting the app server."""

    module_name = "wrist_review_research_pdf_for_test"
    specification = importlib.util.spec_from_file_location(module_name, RESEARCH_PDF_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def test_local_app_default_paths_resolve_inside_repository(monkeypatch):
    monkeypatch.delenv("MURA_APP_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("MURA_APP_MODEL_DIR", raising=False)

    app_core = load_app_core()
    configuration = app_core.resolve_configuration()

    assert configuration.project_root == REPOSITORY_ROOT.resolve()
    assert configuration.model_dir == (
        REPOSITORY_ROOT / "local_artifacts" / "MODEL_B_FINE_TUNING"
    ).resolve()


def test_local_app_fails_closed_without_model_bundle(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    missing_bundle = tmp_path / "missing_bundle"
    monkeypatch.setenv("MURA_APP_PROJECT_ROOT", str(project_root))
    monkeypatch.setenv("MURA_APP_MODEL_DIR", str(missing_bundle))

    app_core = load_app_core()
    status = app_core.artifact_status(app_core.resolve_configuration())

    assert status["ready"] is False
    assert status["issues"] == ["model_bundle_missing"]


def test_local_app_rejects_more_than_eight_views_before_inference():
    app_core = load_app_core()

    class RuntimeThatMustNotRun:
        def predict_view_probabilities(self, _images):  # pragma: no cover - must not be called
            raise AssertionError("The model must not run for an over-limit study.")

    result = app_core.run_guarded_review(
        RuntimeThatMustNotRun(),
        case_alias="LOCAL-CASE",
        declared_anatomy="XR_WRIST",
        images=[object()] * (app_core.MAX_STUDY_VIEWS + 1),
    )

    assert result["deterministic_outcome"] == "NEEDS_HUMAN_REVIEW"
    assert result["classification_performed"] is False
    assert result["reason"] == "too_many_views_maximum_8"


def test_worklist_metadata_prioritizes_without_removing_human_review():
    app_core = load_app_core()

    priority = app_core.worklist_metadata(
        {
            "deterministic_outcome": "RAISE_PRIORITY",
            "view_evidence": {"mean_probability": 0.8, "threshold_disagreement": False},
        }
    )
    ambiguous = app_core.worklist_metadata(
        {
            "deterministic_outcome": "STANDARD_HUMAN_REVIEW",
            "view_evidence": {"mean_probability": 0.2, "threshold_disagreement": True},
        }
    )

    assert priority["bucket"] == "priority"
    assert priority["rank"] == 0
    assert priority["human_review_required"] is True
    assert priority["agent_status"] == "not_run"
    assert ambiguous["bucket"] == "routine"
    assert ambiguous["ambiguous_view_evidence"] is True


def test_browser_worklist_groups_folder_studies_and_does_not_truncate_views():
    script = (REPOSITORY_ROOT / "apps" / "wrist_review" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    page = (REPOSITORY_ROOT / "apps" / "wrist_review" / "static" / "index.html").read_text(
        encoding="utf-8"
    )

    assert "const MAX_STUDY_VIEWS = 8;" in script
    assert "function buildWorklist" in script
    assert "const existingKeys" in script
    assert "patientFolders" in script
    assert "too_many_views_maximum_${MAX_STUDY_VIEWS}" in script
    assert "files.slice(0, 8)" not in script
    assert "webkitdirectory" in page
    assert "Add patient folders" in page
    assert "Add patient folder" in page
    assert "zipWorklistPicker" in page
    assert 'fetch("/api/worklist-zip"' in script
    assert "function agentEvidencePanel" in script
    assert 'fetch("/api/research-report.pdf"' in script
    assert "Evidence only — it cannot change the model score" in script
    assert "Bounded optional evidence" in page


def test_zip_worklist_is_read_in_memory_and_preserves_patient_study_paths():
    server = load_app_server()
    image_bytes = valid_synthetic_view("synthetic.png").image_bytes
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("CASE-001/STUDY-01/view-a.png", image_bytes)
        archive.writestr("CASE-002/STUDY-03/view-b.png", image_bytes)
        archive.writestr("README.txt", "not an image")

    result = server.extract_worklist_archive(archive_bytes.getvalue())

    assert result["storage"] == "in_memory_only"
    assert result["image_count"] == 2
    assert [entry["relative_path"] for entry in result["files"]] == [
        "CASE-001/STUDY-01/view-a.png",
        "CASE-002/STUDY-03/view-b.png",
    ]
    assert all(entry["data_url"].startswith("data:image/png;base64,") for entry in result["files"])


def test_zip_worklist_rejects_unsafe_archive_path():
    server = load_app_server()
    image_bytes = valid_synthetic_view("synthetic.png").image_bytes
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("../outside.png", image_bytes)

    try:
        server.extract_worklist_archive(archive_bytes.getvalue())
    except ValueError as error:
        assert "unsafe file path" in str(error)
    else:  # pragma: no cover - safety contract
        raise AssertionError("Unsafe ZIP paths must be rejected.")


def valid_synthetic_view(image_id: str) -> StudyImage:
    """Create a non-sensitive valid image payload for wiring tests."""

    image = Image.linear_gradient("L").resize((256, 256))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return StudyImage(image_id=image_id, image_bytes=buffer.getvalue())


def test_app_connects_bounded_agent_for_priority_review(monkeypatch):
    monkeypatch.delenv("MURA_CREWAI_ENABLED", raising=False)
    app_core = load_app_core()
    monkeypatch.setattr(
        app_core,
        "_gradcam_explanation_check",
        lambda _runtime, *, images, base_probabilities: {
            "status": "SUCCESS",
            "method": "synthetic_gradcam",
            "views_processed": len(images),
            "view_summaries": [{"view_index": index} for index, _ in enumerate(base_probabilities)],
            "maps_persisted": False,
            "maps_shared_with_crewai": False,
        },
    )

    class RuntimeWithTta:
        threshold = 0.263567715883255
        model_name = "Synthetic test runtime"
        checkpoint_sha256 = "a" * 64

        def __init__(self):
            self.calls: list[int] = []

        def predict_view_probabilities(self, images):
            self.calls.append(len(images))
            return [0.20, 0.30] if len(self.calls) == 1 else [0.23, 0.24, 0.25, 0.26]

    runtime = RuntimeWithTta()
    result = app_core.run_guarded_review(
        runtime,
        case_alias="SYNTHETIC-CASE",
        declared_anatomy="XR_WRIST",
        images=[valid_synthetic_view("view_1.png"), valid_synthetic_view("view_2.png")],
    )

    assert runtime.calls == [2, 4]
    assert result["deterministic_outcome"] == "RAISE_PRIORITY"
    assert result["agent_report"]["agent_status"] == "completed_bounded_investigation"
    assert result["agent_report"]["final_outcome"] == "RAISE_PRIORITY"
    assert result["optional_tool_policy"]["status"] == "AUTHORIZED"
    assert result["agent_report"]["allowed_tools_called"][-1]["tool"] == "gradcam_explainer"
    assert result["research_agent_report"]["agent_status"] == "not_configured"
    assert result["worklist"]["agent_status"] == "completed_bounded_investigation"


def test_app_collects_tta_for_abnormal_priority_without_view_disagreement(monkeypatch):
    monkeypatch.delenv("MURA_CREWAI_ENABLED", raising=False)
    app_core = load_app_core()
    monkeypatch.setattr(
        app_core,
        "_gradcam_explanation_check",
        lambda _runtime, *, images, base_probabilities: {
            "status": "SUCCESS",
            "method": "synthetic_gradcam",
            "views_processed": len(images),
            "view_summaries": [{"view_index": index} for index, _ in enumerate(base_probabilities)],
            "maps_persisted": False,
            "maps_shared_with_crewai": False,
        },
    )

    class RuntimeWithAbnormalMean:
        threshold = 0.263567715883255
        model_name = "Synthetic test runtime"
        checkpoint_sha256 = "c" * 64

        def __init__(self):
            self.calls: list[int] = []

        def predict_view_probabilities(self, images):
            self.calls.append(len(images))
            return [0.30 for _ in images]

    runtime = RuntimeWithAbnormalMean()
    result = app_core.run_guarded_review(
        runtime,
        case_alias="SYNTHETIC-CASE",
        declared_anatomy="XR_WRIST",
        images=[valid_synthetic_view("view_1.png")],
    )

    assert runtime.calls == [1, 4]
    assert result["deterministic_outcome"] == "RAISE_PRIORITY"
    assert result["agent_report"]["agent_status"] == "completed_bounded_investigation"
    assert result["agent_report"]["triggers_fired"] == [
        "abnormal_mean_prediction",
        "borderline_mean_probability",
    ]


def test_app_does_not_execute_tta_for_non_ambiguous_study(monkeypatch):
    monkeypatch.delenv("MURA_CREWAI_ENABLED", raising=False)
    app_core = load_app_core()

    class RuntimeWithoutTta:
        threshold = 0.263567715883255
        model_name = "Synthetic test runtime"
        checkpoint_sha256 = "b" * 64

        def __init__(self):
            self.calls = 0

        def predict_view_probabilities(self, images):
            self.calls += 1
            return [0.10 for _ in images]

    runtime = RuntimeWithoutTta()
    result = app_core.run_guarded_review(
        runtime,
        case_alias="SYNTHETIC-CASE",
        declared_anatomy="XR_WRIST",
        images=[valid_synthetic_view("view_1.png")],
    )

    assert runtime.calls == 1
    assert result["agent_report"]["agent_status"] == "not_called_no_optional_tools"
    assert result["research_agent_report"]["agent_status"] == "not_configured"


def completed_research_review() -> dict:
    """Safe, synthetic output equivalent to a completed LLM explanation."""

    return {
        "case_id": "LOCAL-DEMO-001",
        "deterministic_outcome": "RAISE_PRIORITY",
        "model_predicted_abnormal": True,
        "view_evidence": {
            "mean_probability": 0.41,
            "threshold": 0.263567715883255,
            "threshold_disagreement": True,
        },
        "agent_report": {
            "allowed_tools_called": [
                {"tool": "uncertainty_tta", "status": "OK"},
                {"tool": "gradcam_explainer", "status": "OK"},
            ]
        },
        "research_agent_report": {
            "agent_status": "completed_research_explanation",
            "research_use_only": True,
            "human_review_required": True,
            "summary": "The fixed study score crossed the review threshold. Local evidence was summarized for human review only.",
            "evidence_used": ["Frozen study score and threshold", "Authorized local TTA summary"],
            "limitations": ["Research prototype only.", "Human review remains required."],
        },
    }


def test_completed_llm_explanation_exports_a_memory_only_pdf():
    from pypdf import PdfReader

    exporter = load_research_pdf_exporter()
    pdf_bytes = exporter.build_research_report_pdf(completed_research_review())

    assert pdf_bytes.startswith(b"%PDF")
    assert exporter.report_filename(completed_research_review()) == "LOCAL-DEMO-001_research_explanation.pdf"
    extracted = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(pdf_bytes)).pages)
    assert "LLM research explanation" in extracted
    assert "Priority human review" in extracted
    assert "no diagnostic recommendation" in extracted.lower()
    assert "image1.png" not in extracted


def test_research_pdf_refuses_an_unavailable_llm_report():
    exporter = load_research_pdf_exporter()
    review = completed_research_review()
    review["research_agent_report"]["agent_status"] = "not_configured"

    try:
        exporter.build_research_report_pdf(review)
    except ValueError as error:
        assert "completed CrewAI research explanation" in str(error)
    else:  # pragma: no cover - PDF safety contract
        raise AssertionError("An unavailable LLM explanation must not create a PDF.")
