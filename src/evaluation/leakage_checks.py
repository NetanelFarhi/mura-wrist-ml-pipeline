"""Dataset-split and evaluation-provenance checks for Phase 1.

The MURA-X prediction unit is a study.  These checks deliberately operate on
the image-level manifest, because that is the earliest point at which a study
or patient can accidentally cross a split boundary.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


class LeakageError(ValueError):
    """Raised when a split or provenance record permits evaluation leakage."""


def _normalise_split(value: object) -> str:
    """Return a stable comparison form for a split name."""

    normalised = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    # The data pipeline uses the concise on-disk name ``val``.  Evaluation
    # reports use the more explicit ``validation`` spelling.  Treating these
    # as aliases prevents a safe manifest from being rejected solely because
    # of its documented split vocabulary.
    aliases = {
        "training": "train",
        "val": "validation",
        "valid": "validation",
    }
    return aliases.get(normalised, normalised)


def _split_set(value: object) -> set[str]:
    """Coerce a scalar or collection of split names to a normalized set."""

    if value is None:
        return set()
    if isinstance(value, str):
        return {_normalise_split(value)}
    if isinstance(value, Mapping):
        # A common provenance shape is {"splits": ["train"]}.
        if "splits" in value:
            return _split_set(value["splits"])
        raise TypeError("A provenance role mapping must contain a 'splits' key.")
    if isinstance(value, Iterable):
        return {_normalise_split(item) for item in value}
    return {_normalise_split(value)}


def _sample(values: Iterable[object], limit: int = 10) -> list[str]:
    """Return a compact, deterministic sample for a human-readable report."""

    return [str(value) for value in list(values)[:limit]]


@dataclass(frozen=True)
class IntegrityReport:
    """Outcome of an image-manifest split integrity audit."""

    valid: bool
    n_rows: int
    split_counts: dict[str, int]
    violations: dict[str, list[str]]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def raise_if_invalid(self) -> None:
        if not self.valid:
            summary = "; ".join(
                f"{kind}: {', '.join(examples)}"
                for kind, examples in self.violations.items()
            )
            raise LeakageError(f"Split integrity check failed. {summary}")


@dataclass(frozen=True)
class ProvenanceReport:
    """Outcome of checks over recorded model/preprocessing/threshold sources."""

    valid: bool
    role_splits: dict[str, list[str]]
    violations: dict[str, list[str]]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def raise_if_invalid(self) -> None:
        if not self.valid:
            summary = "; ".join(
                f"{kind}: {', '.join(examples)}"
                for kind, examples in self.violations.items()
            )
            raise LeakageError(f"Evaluation provenance check failed. {summary}")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Manifest is missing required column(s): {', '.join(missing)}")


def audit_split_integrity(
    manifest: pd.DataFrame,
    *,
    split_column: str = "final_split",
    patient_column: str = "patient_id",
    study_column: str = "study_id",
    label_column: str = "label",
    image_column: str | None = "image_path",
    allowed_splits: Iterable[str] | None = ("train", "validation", "test"),
) -> IntegrityReport:
    """Audit a manifest without mutating it.

    The report checks the hierarchy required by Phase 1: a study belongs to
    one patient, every image in a study stays in one split, studies/patients do
    not cross split boundaries, and labels do not disagree within a study.
    """

    if not isinstance(manifest, pd.DataFrame):
        raise TypeError("manifest must be a pandas DataFrame")
    required = [patient_column, study_column, split_column, label_column]
    if image_column is not None:
        required.append(image_column)
    _require_columns(manifest, required)

    frame = manifest.copy()
    split_values = frame[split_column].map(_normalise_split)
    violations: dict[str, list[str]] = {}

    null_columns = [
        column
        for column in (patient_column, study_column, split_column, label_column)
        if frame[column].isna().any()
    ]
    if null_columns:
        violations["missing_identity_or_label"] = null_columns

    if allowed_splits is not None:
        allowed = {_normalise_split(value) for value in allowed_splits}
        unknown = sorted(set(split_values.dropna()) - allowed)
        if unknown:
            violations["unknown_split"] = unknown

    # A study has exactly one owning patient in the manifest hierarchy.
    study_patient_counts = frame.groupby(study_column, dropna=False)[patient_column].nunique(
        dropna=False
    )
    multi_patient_studies = study_patient_counts[study_patient_counts > 1].index.tolist()
    if multi_patient_studies:
        violations["study_assigned_to_multiple_patients"] = _sample(multi_patient_studies)

    patient_split_counts = frame.assign(_split=split_values).groupby(
        patient_column, dropna=False
    )["_split"].nunique(dropna=False)
    overlapping_patients = patient_split_counts[patient_split_counts > 1].index.tolist()
    if overlapping_patients:
        violations["patient_overlap_across_splits"] = _sample(overlapping_patients)

    study_split_counts = frame.assign(_split=split_values).groupby(
        study_column, dropna=False
    )["_split"].nunique(dropna=False)
    overlapping_studies = study_split_counts[study_split_counts > 1].index.tolist()
    if overlapping_studies:
        # This explicitly verifies that all image rows in a study stay together.
        sample = _sample(overlapping_studies)
        violations["study_overlap_across_splits"] = sample
        violations["study_images_not_grouped"] = sample

    study_label_counts = frame.groupby(study_column, dropna=False)[label_column].nunique(
        dropna=False
    )
    inconsistent_labels = study_label_counts[study_label_counts > 1].index.tolist()
    if inconsistent_labels:
        violations["inconsistent_labels_within_study"] = _sample(inconsistent_labels)

    if image_column is not None:
        # A path reused under two study identifiers is incompatible with the
        # patient -> study -> image hierarchy and can create indirect leakage.
        image_study_counts = frame.groupby(image_column, dropna=False)[study_column].nunique(
            dropna=False
        )
        reused_paths = image_study_counts[image_study_counts > 1].index.tolist()
        if reused_paths:
            violations["image_path_in_multiple_studies"] = _sample(reused_paths)

    train_aliases = {"train", "training"}
    validation_aliases = {"validation", "val", "valid"}
    entity_splits = frame.assign(_split=split_values).groupby(patient_column, dropna=False)[
        "_split"
    ].agg(set)
    train_validation_patients = entity_splits[
        entity_splits.map(
            lambda values: bool(values & train_aliases) and bool(values & validation_aliases)
        )
    ].index.tolist()
    if train_validation_patients:
        violations["validation_mixed_into_training"] = _sample(train_validation_patients)

    split_counts = {
        str(split): int(count)
        for split, count in split_values.value_counts(dropna=False).sort_index().items()
    }
    return IntegrityReport(
        valid=not violations,
        n_rows=int(len(frame)),
        split_counts=split_counts,
        violations=violations,
    )


def validate_split_integrity(
    manifest: pd.DataFrame, **kwargs: Any
) -> IntegrityReport:
    """Run :func:`audit_split_integrity` and raise on any violation."""

    report = audit_split_integrity(manifest, **kwargs)
    report.raise_if_invalid()
    return report


def assert_no_patient_overlap(
    manifest: pd.DataFrame,
    *,
    split_column: str = "final_split",
    patient_column: str = "patient_id",
) -> None:
    """Raise if any patient appears in more than one split."""

    _require_columns(manifest, [split_column, patient_column])
    counts = manifest.assign(_split=manifest[split_column].map(_normalise_split)).groupby(
        patient_column, dropna=False
    )["_split"].nunique(dropna=False)
    offenders = counts[counts > 1].index.tolist()
    if offenders:
        raise LeakageError(
            "Patient overlap across splits: " + ", ".join(_sample(offenders))
        )


def assert_no_study_overlap(
    manifest: pd.DataFrame,
    *,
    split_column: str = "final_split",
    study_column: str = "study_id",
) -> None:
    """Raise if one study's images appear in more than one split."""

    _require_columns(manifest, [split_column, study_column])
    counts = manifest.assign(_split=manifest[split_column].map(_normalise_split)).groupby(
        study_column, dropna=False
    )["_split"].nunique(dropna=False)
    offenders = counts[counts > 1].index.tolist()
    if offenders:
        raise LeakageError(
            "Study overlap across splits: " + ", ".join(_sample(offenders))
        )


def assert_study_label_consistency(
    manifest: pd.DataFrame,
    *,
    study_column: str = "study_id",
    label_column: str = "label",
) -> None:
    """Raise if image rows within a study disagree about the study label."""

    _require_columns(manifest, [study_column, label_column])
    counts = manifest.groupby(study_column, dropna=False)[label_column].nunique(dropna=False)
    offenders = counts[counts > 1].index.tolist()
    if offenders:
        raise LeakageError(
            "Inconsistent labels within study: " + ", ".join(_sample(offenders))
        )


_FIT_ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "model_fit": ("model_fit_splits", "fit_splits", "training_splits", "model_training_splits"),
    "preprocessing_fit": (
        "preprocessing_fit_splits",
        "preprocessor_fit_splits",
        "feature_fit_splits",
        "normalization_fit_splits",
    ),
}

_VALIDATION_SELECTION_ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "hyperparameter_selection": (
        "hyperparameter_selection_splits",
        "tuning_splits",
        "model_selection_splits",
    ),
    "aggregation_selection": (
        "aggregation_selection_splits",
        "aggregation_tuning_splits",
    ),
}

_THRESHOLD_ROLE_ALIASES = (
    "threshold_selection_splits",
    "operating_threshold_splits",
    "threshold_tuning_splits",
)


def _extract_role_splits(
    provenance: Mapping[str, object], aliases: Mapping[str, tuple[str, ...]]
) -> dict[str, set[str]]:
    role_splits: dict[str, set[str]] = {}
    for role, keys in aliases.items():
        values: set[str] = set()
        seen = False
        for key in keys:
            if key in provenance:
                seen = True
                values.update(_split_set(provenance[key]))
        if seen:
            role_splits[role] = values
    return role_splits


def audit_fitting_provenance(
    provenance: Mapping[str, object],
    *,
    train_split: str = "train",
    validation_split: str = "validation",
    test_split: str = "test",
) -> ProvenanceReport:
    """Check that fitting and validation decisions cannot use final-test data.

    Fitting/preprocessing is restricted to the train split.  Hyperparameter and
    aggregation selection are restricted to validation.  The final test split
    may be evaluated, but never appears in these roles.
    """

    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping of roles to split names")

    train = _normalise_split(train_split)
    validation = _normalise_split(validation_split)
    test = _normalise_split(test_split)
    role_splits = _extract_role_splits(provenance, _FIT_ROLE_ALIASES)
    role_splits.update(
        _extract_role_splits(provenance, _VALIDATION_SELECTION_ROLE_ALIASES)
    )
    violations: dict[str, list[str]] = {}

    for role in ("model_fit", "preprocessing_fit"):
        splits = role_splits.get(role)
        if splits is not None and splits != {train}:
            violations[f"{role}_must_use_train_only"] = sorted(splits)

    for role in ("hyperparameter_selection", "aggregation_selection"):
        splits = role_splits.get(role)
        if splits is not None and splits != {validation}:
            violations[f"{role}_must_use_validation_only"] = sorted(splits)

    # Produce a direct, easily searchable test-exclusion finding even if a
    # caller uses a role alias that has a more specific validation error above.
    test_roles = [role for role, splits in role_splits.items() if test in splits]
    if test_roles:
        violations["test_data_used_for_fitting_or_selection"] = sorted(test_roles)

    serialized = {role: sorted(splits) for role, splits in role_splits.items()}
    return ProvenanceReport(
        valid=not violations,
        role_splits=serialized,
        violations=violations,
    )


def validate_fitting_provenance(
    provenance: Mapping[str, object], **kwargs: Any
) -> ProvenanceReport:
    """Run :func:`audit_fitting_provenance` and raise on a violation."""

    report = audit_fitting_provenance(provenance, **kwargs)
    report.raise_if_invalid()
    return report


def audit_threshold_provenance(
    provenance: Mapping[str, object],
    *,
    validation_split: str = "validation",
    test_split: str = "test",
    require_record: bool = True,
) -> ProvenanceReport:
    """Verify that an operating threshold was selected from validation only."""

    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping of roles to split names")

    validation = _normalise_split(validation_split)
    test = _normalise_split(test_split)
    seen = False
    splits: set[str] = set()
    for key in _THRESHOLD_ROLE_ALIASES:
        if key in provenance:
            seen = True
            splits.update(_split_set(provenance[key]))

    violations: dict[str, list[str]] = {}
    if require_record and not seen:
        violations["threshold_selection_not_recorded"] = [
            "Provide threshold_selection_splits=['validation']."
        ]
    elif splits != {validation}:
        violations["threshold_selection_must_use_validation_only"] = sorted(splits)
    if test in splits:
        violations["test_data_used_for_threshold_selection"] = sorted(splits)

    return ProvenanceReport(
        valid=not violations,
        role_splits={"threshold_selection": sorted(splits)} if seen else {},
        violations=violations,
    )


def validate_threshold_provenance(
    provenance: Mapping[str, object], **kwargs: Any
) -> ProvenanceReport:
    """Run :func:`audit_threshold_provenance` and raise on a violation."""

    report = audit_threshold_provenance(provenance, **kwargs)
    report.raise_if_invalid()
    return report


def validate_evaluation_provenance(
    provenance: Mapping[str, object], **kwargs: Any
) -> dict[str, dict[str, Any]]:
    """Validate fitting and threshold provenance together.

    A JSON-serializable pair of reports is returned so callers can persist the
    evidence used to establish that the final test set was sealed.
    """

    fitting_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in {"train_split", "validation_split", "test_split"}
    }
    threshold_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in {"validation_split", "test_split", "require_record"}
    }
    fitting = validate_fitting_provenance(provenance, **fitting_kwargs)
    threshold = validate_threshold_provenance(provenance, **threshold_kwargs)
    return {
        "fitting": fitting.as_dict(),
        "threshold": threshold.as_dict(),
    }


# Readable aliases for tests and downstream scripts.
check_split_integrity = audit_split_integrity
check_fitting_provenance = audit_fitting_provenance
check_threshold_provenance = audit_threshold_provenance
