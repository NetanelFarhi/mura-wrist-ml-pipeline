"""Typed deterministic safeguards for a future MURA wrist runtime.

This module is deliberately not an adaptive agent. It implements the hard
rules that must run first: image QC, anatomy/scope checks, complete-study
validation, immutable threshold application, and evidence summarisation.
Every supported case stays in the human-review workflow; the model can only
raise priority, never clear a case from standard review.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
from typing import Callable, Mapping, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError


SUPPORTED_ANATOMY = "XR_WRIST"
KNOWN_MURA_ANATOMIES = frozenset(
    {
        "XR_WRIST",
        "XR_HAND",
        "XR_FINGER",
        "XR_FOREARM",
        "XR_ELBOW",
        "XR_HUMERUS",
        "XR_SHOULDER",
    }
)


@dataclass(frozen=True)
class StudyImage:
    """A de-identified image payload belonging to one user-supplied study."""

    image_id: str
    image_bytes: bytes


Classifier = Callable[[Sequence[StudyImage]], Sequence[float]]


def technical_qc(image_bytes: bytes, image_id: str = "UNKNOWN") -> dict[str, object]:
    """Run non-clinical technical QC on one encoded image.

    The thresholds reject clearly unusable files (empty, corrupt, too small,
    or near-uniform). They are engineering guardrails, not clinical quality
    labels and not a replacement for human review.
    """

    result: dict[str, object] = {
        "image_id": image_id,
        "status": "FAIL",
        "width": None,
        "height": None,
        "mean_brightness": None,
        "contrast_std": None,
        "issues": [],
    }
    issues: list[str] = result["issues"]  # type: ignore[assignment]
    if not isinstance(image_bytes, bytes) or not image_bytes:
        issues.append("empty_or_invalid_input")
        return result

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            pixels = np.asarray(image.convert("L"), dtype=np.float32)
    except (UnidentifiedImageError, OSError, ValueError):
        issues.append("unreadable_image")
        return result

    if pixels.ndim != 2 or not pixels.size:
        issues.append("invalid_image_array")
        return result

    height, width = pixels.shape
    result["width"] = int(width)
    result["height"] = int(height)
    mean_brightness = float(pixels.mean())
    contrast_std = float(pixels.std())
    result["mean_brightness"] = mean_brightness
    result["contrast_std"] = contrast_std

    if width < 128 or height < 128:
        issues.append("image_too_small")
    if mean_brightness < 5:
        issues.append("extremely_dark")
    if mean_brightness > 250:
        issues.append("extremely_bright")
    if contrast_std < 5:
        issues.append("near_uniform_image")

    blocking = {"image_too_small", "near_uniform_image"}
    result["status"] = "FAIL" if blocking.intersection(issues) else ("WARNING" if issues else "PASS")
    return result


def validate_declared_anatomy(anatomy: str | None) -> dict[str, object]:
    """Validate a declared body part without treating metadata as instructions."""

    if not isinstance(anatomy, str) or not anatomy.strip():
        return {"status": "UNCERTAIN_ANATOMY", "anatomy": None, "issues": ["missing_anatomy"]}
    normalized = anatomy.strip().upper()
    if normalized == SUPPORTED_ANATOMY:
        return {"status": "SUPPORTED_WRIST", "anatomy": normalized, "issues": []}
    if normalized in KNOWN_MURA_ANATOMIES:
        return {"status": "UNSUPPORTED", "anatomy": normalized, "issues": ["unsupported_anatomy"]}
    return {"status": "UNCERTAIN_ANATOMY", "anatomy": normalized, "issues": ["unknown_anatomy"]}


def view_evidence(probabilities: Sequence[float], threshold: float) -> dict[str, object]:
    """Summarise per-view probabilities while preserving an immutable threshold."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        return {"status": "INVALID_INPUT", "reason": "no_view_probabilities"}
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        return {"status": "INVALID_INPUT", "reason": "invalid_probability"}
    classes = (values >= threshold).astype(int)
    return {
        "status": "SUCCESS",
        "view_count": int(len(values)),
        "mean_probability": float(values.mean()),
        "min_probability": float(values.min()),
        "max_probability": float(values.max()),
        "probability_range": float(np.ptp(values)),
        "probability_std": float(values.std()),
        "threshold_disagreement": bool(len(np.unique(classes)) > 1),
        "threshold": float(threshold),
    }


def guarded_study_route(
    *,
    case_id: str,
    declared_anatomy: str | None,
    images: Sequence[StudyImage],
    classifier: Classifier,
    threshold: float,
) -> dict[str, object]:
    """Apply mandatory safeguards, then return a non-diagnostic workflow route.

    ``classifier`` is injected so this policy can be unit-tested without a
    checkpoint. Its returned values must correspond one-to-one with real views.
    The function never changes the threshold or returns a clearance decision.
    """

    safe_case_id = str(case_id)
    if not images:
        return {
            "case_id": safe_case_id,
            "deterministic_outcome": "NEEDS_HUMAN_REVIEW",
            "classification_performed": False,
            "reason": "empty_study",
        }
    image_ids = [image.image_id for image in images]
    if len(image_ids) != len(set(image_ids)):
        return {
            "case_id": safe_case_id,
            "deterministic_outcome": "NEEDS_HUMAN_REVIEW",
            "classification_performed": False,
            "reason": "duplicate_image_ids",
        }

    anatomy = validate_declared_anatomy(declared_anatomy)
    if anatomy["status"] != "SUPPORTED_WRIST":
        outcome = "UNSUPPORTED" if anatomy["status"] == "UNSUPPORTED" else "NEEDS_HUMAN_REVIEW"
        return {
            "case_id": safe_case_id,
            "deterministic_outcome": outcome,
            "classification_performed": False,
            "reason": anatomy["status"],
            "anatomy": anatomy,
        }

    qc_results = [technical_qc(image.image_bytes, image.image_id) for image in images]
    if any(result["status"] == "FAIL" for result in qc_results):
        return {
            "case_id": safe_case_id,
            "deterministic_outcome": "NEEDS_HUMAN_REVIEW",
            "classification_performed": False,
            "reason": "technical_qc_failure",
            "anatomy": anatomy,
            "qc_results": qc_results,
        }

    probabilities = list(classifier(images))
    if len(probabilities) != len(images):
        return {
            "case_id": safe_case_id,
            "deterministic_outcome": "NEEDS_HUMAN_REVIEW",
            "classification_performed": False,
            "reason": "classifier_view_count_mismatch",
            "anatomy": anatomy,
            "qc_results": qc_results,
        }
    evidence = view_evidence(probabilities, threshold)
    if evidence["status"] != "SUCCESS":
        return {
            "case_id": safe_case_id,
            "deterministic_outcome": "NEEDS_HUMAN_REVIEW",
            "classification_performed": False,
            "reason": evidence["reason"],
            "anatomy": anatomy,
            "qc_results": qc_results,
        }

    model_flag = int(float(evidence["mean_probability"]) >= threshold)
    disagreement = bool(evidence["threshold_disagreement"])
    outcome = "RAISE_PRIORITY" if model_flag or disagreement else "STANDARD_HUMAN_REVIEW"
    return {
        "case_id": safe_case_id,
        "deterministic_outcome": outcome,
        "classification_performed": True,
        "reason": "model_flag_or_view_disagreement" if outcome == "RAISE_PRIORITY" else "standard_human_review",
        "anatomy": anatomy,
        "qc_results": qc_results,
        "view_evidence": evidence,
        "model_predicted_abnormal": bool(model_flag),
        "human_review_required": True,
    }
