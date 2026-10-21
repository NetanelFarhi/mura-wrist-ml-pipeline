"""Small regression checks for the Git data-governance guard."""

from __future__ import annotations

from scripts.check_repo_data_governance import unsafe_paths, unsafe_public_content


def test_safe_diagram_name_is_not_misclassified_as_raw_data() -> None:
    assert unsafe_paths(["docs/assets/MURA_System_Architecture.png"]) == []


def test_raw_mura_image_path_is_rejected() -> None:
    violations = unsafe_paths(["data/raw/MURA-v1.1/train/XR_WRIST/image.png"])
    assert any("raw-medical-image-like" in item for item in violations)


def test_public_result_rejects_identifier_like_content(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "unsafe.csv").write_text("study_id\npatient1234/study1\n", encoding="utf-8")

    violations = unsafe_public_content(["reports/unsafe.csv"])
    assert any("identifier-like" in item for item in violations)
