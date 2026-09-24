"""Validation-tuned Logistic Regression over frozen, study-pooled embeddings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.training.metrics import safe_auprc, safe_auroc


SelectionMetric = Literal["auroc", "auprc"]


@dataclass(frozen=True)
class LogisticBaselineSelection:
    """The validation-only hyperparameter selection record for Model A."""

    selected_c: float
    selection_metric: SelectionMetric
    validation_scores: dict[str, float]
    candidate_c_values: tuple[float, ...]
    class_weight: str | None
    random_seed: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_features_and_labels(
    embeddings: np.ndarray | Sequence[Sequence[float]],
    labels: np.ndarray | Sequence[int],
    *,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(embeddings, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(f"{split_name} embeddings must be a non-empty [studies, features] matrix.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"{split_name} embeddings has {x.shape[0]} rows but labels has {y.shape[0]} values."
        )
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{split_name} embeddings contain NaN or infinity.")
    labels_seen = set(y.tolist())
    if not labels_seen.issubset({0, 1}):
        raise ValueError(f"{split_name} labels must be binary 0/1; found {sorted(labels_seen)}.")
    return x, y


class FrozenEmbeddingLogisticBaseline:
    """Small, deterministic C search that never accepts test data in ``fit``.

    Each candidate pipeline is fitted on training studies only.  The validation
    set is only scored to choose C.  The selected fitted estimator remains the
    train-only version by default, which makes it impossible for the usual
    script path to leak validation samples into model fitting.
    """

    def __init__(
        self,
        *,
        c_values: Sequence[float] = (0.1, 1.0, 10.0),
        selection_metric: SelectionMetric = "auroc",
        class_weight: Literal["balanced"] | None = None,
        random_seed: int = 42,
        max_iter: int = 2000,
    ) -> None:
        parsed_c_values = tuple(float(value) for value in c_values)
        if not 1 <= len(parsed_c_values) <= 3:
            raise ValueError(
                "Phase 1 permits a deliberately small validation search: provide one to three C values."
            )
        if len(set(parsed_c_values)) != len(parsed_c_values) or any(value <= 0 for value in parsed_c_values):
            raise ValueError("C values must be unique positive numbers.")
        if selection_metric not in ("auroc", "auprc"):
            raise ValueError("selection_metric must be 'auroc' or 'auprc'.")
        if class_weight not in (None, "balanced"):
            raise ValueError("class_weight may be None or 'balanced'.")
        self.c_values = parsed_c_values
        self.selection_metric = selection_metric
        self.class_weight = class_weight
        self.random_seed = int(random_seed)
        self.max_iter = int(max_iter)
        self.estimator_: Pipeline | None = None
        self.selection_: LogisticBaselineSelection | None = None
        self.fitted_on_: str | None = None

    def _make_estimator(self, c_value: float) -> Pipeline:
        # liblinear is deterministic under random_state and behaves well for
        # the small/medium study-level feature table expected in Phase 1.
        classifier = LogisticRegression(
            C=float(c_value),
            class_weight=self.class_weight,
            solver="liblinear",
            max_iter=self.max_iter,
            random_state=self.random_seed,
        )
        return Pipeline((("scale", StandardScaler()), ("logistic", classifier)))

    def fit(
        self,
        train_embeddings: np.ndarray | Sequence[Sequence[float]],
        train_labels: np.ndarray | Sequence[int],
        validation_embeddings: np.ndarray | Sequence[Sequence[float]],
        validation_labels: np.ndarray | Sequence[int],
    ) -> "FrozenEmbeddingLogisticBaseline":
        """Fit candidates on train and select C by validation score only."""

        x_train, y_train = _validate_features_and_labels(
            train_embeddings, train_labels, split_name="train"
        )
        x_validation, y_validation = _validate_features_and_labels(
            validation_embeddings, validation_labels, split_name="validation"
        )
        if x_train.shape[1] != x_validation.shape[1]:
            raise ValueError("Train and validation embeddings have different feature dimensions.")
        if len(np.unique(y_train)) < 2:
            raise ValueError("Logistic Regression training requires both NORMAL and ABNORMAL studies.")
        if len(np.unique(y_validation)) < 2:
            raise ValueError("Validation C selection requires both NORMAL and ABNORMAL studies.")

        best_estimator: Pipeline | None = None
        best_c: float | None = None
        best_score = float("-inf")
        validation_scores: dict[str, float] = {}
        epsilon = 1e-12
        for c_value in sorted(self.c_values):
            candidate = self._make_estimator(c_value)
            candidate.fit(x_train, y_train)
            probabilities = candidate.predict_proba(x_validation)[:, 1]
            score = (
                safe_auroc(y_validation, probabilities)
                if self.selection_metric == "auroc"
                else safe_auprc(y_validation, probabilities)
            )
            if score is None:  # covered by the validation class check, kept defensive
                raise ValueError(f"Cannot compute validation {self.selection_metric} for a one-class split.")
            validation_scores[str(c_value)] = float(score)
            if score > best_score + epsilon or (
                abs(score - best_score) <= epsilon and (best_c is None or c_value < best_c)
            ):
                best_estimator, best_c, best_score = candidate, c_value, float(score)

        if best_estimator is None or best_c is None:
            raise RuntimeError("No Logistic Regression candidate was fitted.")
        self.estimator_ = best_estimator
        self.selection_ = LogisticBaselineSelection(
            selected_c=float(best_c),
            selection_metric=self.selection_metric,
            validation_scores=validation_scores,
            candidate_c_values=tuple(sorted(self.c_values)),
            class_weight=self.class_weight,
            random_seed=self.random_seed,
        )
        self.fitted_on_ = "train_only"
        return self

    def refit_on_train_and_validation_after_freeze(
        self,
        train_embeddings: np.ndarray | Sequence[Sequence[float]],
        train_labels: np.ndarray | Sequence[int],
        validation_embeddings: np.ndarray | Sequence[Sequence[float]],
        validation_labels: np.ndarray | Sequence[int],
    ) -> "FrozenEmbeddingLogisticBaseline":
        """Optionally refit with the already-selected C after all choices freeze.

        This method deliberately has no test-data parameters.  Call it only
        after recording the validation-selected C and threshold; the standard
        Phase 1 scripts retain the train-only estimator by default.
        """

        if self.selection_ is None:
            raise RuntimeError("Call fit(train, validation) before refitting the selected C.")
        x_train, y_train = _validate_features_and_labels(
            train_embeddings, train_labels, split_name="train"
        )
        x_validation, y_validation = _validate_features_and_labels(
            validation_embeddings, validation_labels, split_name="validation"
        )
        if x_train.shape[1] != x_validation.shape[1]:
            raise ValueError("Train and validation embeddings have different feature dimensions.")
        x_combined = np.concatenate((x_train, x_validation), axis=0)
        y_combined = np.concatenate((y_train, y_validation), axis=0)
        self.estimator_ = self._make_estimator(self.selection_.selected_c)
        self.estimator_.fit(x_combined, y_combined)
        self.fitted_on_ = "train_plus_validation_after_selection_freeze"
        return self

    def predict_proba(self, embeddings: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        """Return abnormal-study probability for each mean-pooled embedding."""

        if self.estimator_ is None:
            raise RuntimeError("The baseline is not fitted. Call fit with train and validation data first.")
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("embeddings must be a non-empty [studies, features] matrix.")
        return np.asarray(self.estimator_.predict_proba(x)[:, 1], dtype=np.float64)

    def save(self, output_path: str | Path) -> Path:
        """Save the fitted pipeline and its validation-selection provenance."""

        if self.estimator_ is None or self.selection_ is None:
            raise RuntimeError("Cannot save an unfitted baseline.")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, input_path: str | Path) -> "FrozenEmbeddingLogisticBaseline":
        """Load a baseline persisted by :meth:`save`."""

        loaded = joblib.load(Path(input_path))
        if not isinstance(loaded, cls):
            raise TypeError(f"Artifact does not contain {cls.__name__}: {input_path}")
        return loaded

    def selection_metadata(self) -> dict[str, object]:
        if self.selection_ is None:
            raise RuntimeError("The baseline is not fitted.")
        return {**self.selection_.to_dict(), "fitted_on": self.fitted_on_}


__all__ = [
    "FrozenEmbeddingLogisticBaseline",
    "LogisticBaselineSelection",
    "SelectionMetric",
]
