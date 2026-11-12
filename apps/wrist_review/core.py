"""Local inference adapter for the Wrist Review Workspace.

This module deliberately connects only to the verified Checkpoint 1 Model B
bundle.  It is not a clinical deployment layer and it does not access the
official MURA validation/test population.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import sys
import time
import warnings
from typing import Any, Sequence

import torch
from PIL import Image, ImageEnhance


MAX_STUDY_VIEWS = 8


def worklist_metadata(result: dict[str, Any]) -> dict[str, Any]:
    """Describe one completed study for the local human-review worklist.

    The worklist is an ordering aid only.  It never converts a study into a
    diagnosis or removes it from human review.  The bounded ambiguity
    investigation is reported explicitly; it cannot change the model score,
    threshold, or human-review requirement.
    """

    outcome = str(result.get("deterministic_outcome", "NEEDS_HUMAN_REVIEW"))
    evidence = result.get("view_evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    raw_score = evidence.get("mean_probability")
    score = float(raw_score) if isinstance(raw_score, (int, float)) else None
    disagreement = bool(evidence.get("threshold_disagreement", False))
    agent_report = result.get("agent_report")
    agent_report = agent_report if isinstance(agent_report, dict) else {}

    if outcome == "RAISE_PRIORITY":
        bucket, rank = "priority", 0
    elif outcome in {"UNSUPPORTED", "NEEDS_HUMAN_REVIEW"}:
        bucket, rank = "input_attention", 1
    else:
        bucket, rank = "routine", 2

    return {
        "bucket": bucket,
        "rank": rank,
        "study_score": score,
        "ambiguous_view_evidence": disagreement,
        "human_review_required": True,
        "agent_status": str(agent_report.get("agent_status", "not_run")),
        "agent_note": str(
            agent_report.get(
                "reason_for_escalation_or_rejection",
                "Bounded agent report was unavailable; deterministic routing was retained.",
            )
        ),
    }


@dataclass(frozen=True)
class AppConfiguration:
    """Locations for the local-only app and its verified artifact bundle."""

    project_root: Path
    model_dir: Path

    @property
    def checkpoint_path(self) -> Path:
        return self.model_dir / "best_resnet18_study_level.pt"

    @property
    def bundle_spec_path(self) -> Path:
        return self.project_root / "configs" / "model_b_artifact_bundle.json"

    @property
    def policy_path(self) -> Path:
        return self.model_dir / "frozen_primary_operating_policy.json"


@dataclass
class ModelRuntime:
    """Loaded Model B plus the exact deterministic evaluation transform."""

    model: torch.nn.Module
    evaluation_transform: Any
    device: torch.device
    threshold: float
    checkpoint_sha256: str
    model_name: str

    @torch.inference_mode()
    def predict_view_probabilities(self, images: Sequence[Any]) -> list[float]:
        """Predict one probability per real uploaded image, in input order."""

        if not images:
            return []
        tensors: list[torch.Tensor] = []
        for item in images:
            image_bytes = getattr(item, "image_bytes", None)
            if not isinstance(image_bytes, bytes):
                raise TypeError("Each study view must contain image bytes.")
            try:
                with Image.open(io.BytesIO(image_bytes)) as source:
                    image = source.copy()
            except (OSError, ValueError) as exc:
                raise ValueError("A view could not be decoded for model inference.") from exc
            tensor = self.evaluation_transform(image)
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError("The configured evaluation transform did not return a torch tensor.")
            tensors.append(tensor)

        batch = torch.stack(tensors, dim=0).to(self.device)
        probabilities = torch.sigmoid(self.model(batch)).detach().cpu().tolist()
        return [float(value) for value in probabilities]


def resolve_configuration() -> AppConfiguration:
    """Resolve local defaults, while allowing a portable launcher to override them."""

    app_root = Path(__file__).resolve().parent
    project_root = Path(os.environ.get("MURA_APP_PROJECT_ROOT", app_root.parents[1]))
    model_dir = Path(
        os.environ.get(
            "MURA_APP_MODEL_DIR",
            project_root / "local_artifacts" / "MODEL_B_FINE_TUNING",
        )
    )
    return AppConfiguration(project_root=project_root.expanduser().resolve(), model_dir=model_dir.expanduser().resolve())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required app artifact is missing: {path.name}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not read JSON artifact {path.name}.") from exc


def file_sha256(path: Path) -> str:
    """Return an uppercase SHA-256 checksum without loading the whole file at once."""

    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def artifact_status(config: AppConfiguration) -> dict[str, Any]:
    """Verify the exact bundle identity before a user can request inference."""

    status: dict[str, Any] = {
        "ready": False,
        "project_root": str(config.project_root),
        "model_dir": str(config.model_dir),
        "issues": [],
    }
    issues: list[str] = status["issues"]
    if not config.project_root.is_dir():
        issues.append("project_root_missing")
        return status
    if not config.model_dir.is_dir():
        issues.append("model_bundle_missing")
        return status
    if not config.bundle_spec_path.is_file():
        issues.append("bundle_spec_missing")
        return status
    if not config.policy_path.is_file():
        issues.append("frozen_policy_missing")
        return status
    if not config.checkpoint_path.is_file():
        issues.append("checkpoint_missing")
        return status

    bundle = _read_json(config.bundle_spec_path)
    policy = _read_json(config.policy_path)
    expected = str(bundle.get("required_files", {}).get("best_resnet18_study_level.pt", {}).get("sha256", "")).upper()
    actual = file_sha256(config.checkpoint_path)
    if not expected or actual != expected:
        issues.append("checkpoint_hash_mismatch")
        status.update({"expected_checkpoint_sha256": expected, "actual_checkpoint_sha256": actual})
        return status
    if policy.get("source_split") != "internal_validation" or policy.get("test_used_for_selection") is not False:
        issues.append("policy_provenance_invalid")
        return status
    if policy.get("study_aggregation") != "mean_probability":
        issues.append("unexpected_study_aggregation")
        return status

    status.update(
        {
            "ready": True,
            "model": policy.get("model", "Model B"),
            "threshold": float(policy["threshold"]),
            "target_sensitivity": float(policy["target_sensitivity"]),
            "validation_sensitivity": float(policy["sensitivity"]),
            "validation_specificity": float(policy["specificity"]),
            "checkpoint_sha256": actual,
            "study_aggregation": policy["study_aggregation"],
            "selection_split": policy["source_split"],
        }
    )
    return status


def load_verified_runtime(config: AppConfiguration) -> ModelRuntime:
    """Load only the canonical Model B checkpoint after verifying its checksum."""

    status = artifact_status(config)
    if not status["ready"]:
        raise RuntimeError("Verified Model B artifacts are unavailable: " + ", ".join(status["issues"]))

    if str(config.project_root) not in sys.path:
        sys.path.insert(0, str(config.project_root))
    from src.data.transforms import build_eval_transform
    from src.models.resnet18_classifier import build_resnet18_classifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("The verified checkpoint does not contain model_state_dict.")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Building a randomly initialized ResNet-18.*",
            category=RuntimeWarning,
        )
        model = build_resnet18_classifier(weights="none")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return ModelRuntime(
        model=model,
        evaluation_transform=build_eval_transform(image_size=224),
        device=device,
        threshold=float(status["threshold"]),
        checkpoint_sha256=str(status["checkpoint_sha256"]),
        model_name=str(status["model"]),
    )


def run_guarded_review(
    runtime: ModelRuntime,
    *,
    case_alias: str,
    declared_anatomy: str | None,
    images: Sequence[Any],
) -> dict[str, Any]:
    """Run QC first, then exact Model B inference through the immutable policy."""

    if len(images) > MAX_STUDY_VIEWS:
        return {
            "case_id": str(case_alias),
            "deterministic_outcome": "NEEDS_HUMAN_REVIEW",
            "classification_performed": False,
            "reason": f"too_many_views_maximum_{MAX_STUDY_VIEWS}",
            "human_review_required": True,
        }

    from src.runtime.deterministic_safety import guarded_study_route

    captured_probabilities: list[float] = []

    def classifier_with_evidence(study_images: Sequence[Any]) -> list[float]:
        values = runtime.predict_view_probabilities(study_images)
        captured_probabilities[:] = values
        return values

    started = time.perf_counter()
    result = guarded_study_route(
        case_id=case_alias or "LOCAL_CASE",
        declared_anatomy=declared_anatomy,
        images=images,
        classifier=classifier_with_evidence,
        threshold=runtime.threshold,
    )
    result["runtime_seconds"] = round(time.perf_counter() - started, 3)
    result["model_metadata"] = {
        "model": runtime.model_name,
        "checkpoint_sha256_prefix": runtime.checkpoint_sha256[:12],
        "study_aggregation": "mean_probability",
        "frozen_threshold": runtime.threshold,
        "input_storage": "in-memory local session only",
    }
    if captured_probabilities:
        result["per_view_probabilities"] = [round(value, 6) for value in captured_probabilities]
    result["agent_report"] = _run_bounded_ambiguity_investigation(
        runtime=runtime,
        images=images,
        deterministic_result=result,
        base_probabilities=captured_probabilities,
    )
    result["worklist"] = worklist_metadata(result)
    return result


def _encode_png(image: Image.Image) -> bytes:
    """Encode a transient augmentation without writing image pixels to disk."""

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _tta_uncertainty_check(
    runtime: ModelRuntime,
    *,
    image: Any,
    original_probability: float,
) -> dict[str, Any]:
    """Measure small brightness/contrast perturbations for one ambiguous view.

    This is evidence only.  It does not replace the original score, threshold,
    deterministic route, or required human review.
    """

    from src.runtime.deterministic_safety import StudyImage

    image_bytes = getattr(image, "image_bytes", None)
    image_id = str(getattr(image, "image_id", "ambiguous_view"))
    if not isinstance(image_bytes, bytes):
        raise TypeError("Ambiguity check requires image bytes.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            source_image = source.convert("RGB").copy()
    except (OSError, ValueError) as exc:
        raise ValueError("Ambiguity check could not decode the selected view.") from exc

    augmentations = {
        "brightness_minus_5pct": ImageEnhance.Brightness(source_image).enhance(0.95),
        "brightness_plus_5pct": ImageEnhance.Brightness(source_image).enhance(1.05),
        "contrast_minus_5pct": ImageEnhance.Contrast(source_image).enhance(0.95),
        "contrast_plus_5pct": ImageEnhance.Contrast(source_image).enhance(1.05),
    }
    augmented_views = [
        StudyImage(image_id=f"{image_id}__{name}", image_bytes=_encode_png(variant))
        for name, variant in augmentations.items()
    ]
    variant_probabilities = runtime.predict_view_probabilities(augmented_views)
    all_probabilities = [float(original_probability), *[float(value) for value in variant_probabilities]]
    threshold = float(runtime.threshold)
    variant_scores = {
        name: round(float(score), 6)
        for name, score in zip(augmentations, variant_probabilities, strict=True)
    }
    threshold_disagreement = len({value >= threshold for value in all_probabilities}) > 1
    return {
        "status": "SUCCESS",
        "selected_view_id": image_id,
        "original_probability": round(float(original_probability), 6),
        "augmentation_probabilities": variant_scores,
        "tta_mean_probability": round(sum(all_probabilities) / len(all_probabilities), 6),
        "tta_std": round(float(torch.tensor(all_probabilities).std(unbiased=False).item()), 6),
        "tta_range": round(max(all_probabilities) - min(all_probabilities), 6),
        "threshold_disagreement": bool(threshold_disagreement),
        "threshold": threshold,
        "augmentation_count": len(variant_probabilities),
        "scope": "local_evidence_only",
    }


def _run_bounded_ambiguity_investigation(
    *,
    runtime: ModelRuntime,
    images: Sequence[Any],
    deterministic_result: dict[str, Any],
    base_probabilities: Sequence[float],
) -> dict[str, Any]:
    """Run the approved local agent path, only if its ambiguity trigger fires."""

    from src.runtime.bounded_agent import run_bounded_agent

    if len(images) != len(base_probabilities):
        return run_bounded_agent(deterministic_result, tta_runner=None)

    evidence = deterministic_result.get("view_evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    if not bool(evidence.get("threshold_disagreement", False)):
        return run_bounded_agent(deterministic_result, tta_runner=None)

    closest_index = min(
        range(len(base_probabilities)),
        key=lambda index: abs(float(base_probabilities[index]) - runtime.threshold),
    )

    return run_bounded_agent(
        deterministic_result,
        tta_runner=lambda: _tta_uncertainty_check(
            runtime,
            image=images[closest_index],
            original_probability=float(base_probabilities[closest_index]),
        ),
    )


def deidentified_evidence_record(result: dict[str, Any]) -> str:
    """Create a safe-to-copy record without image bytes, paths, or patient identifiers."""

    allowed = {
        "case_id",
        "deterministic_outcome",
        "classification_performed",
        "reason",
        "anatomy",
        "qc_results",
        "view_evidence",
        "per_view_probabilities",
        "model_predicted_abnormal",
        "human_review_required",
        "runtime_seconds",
        "model_metadata",
        "agent_report",
        "worklist",
    }
    safe = {key: value for key, value in result.items() if key in allowed}
    return json.dumps(safe, indent=2, ensure_ascii=False)
