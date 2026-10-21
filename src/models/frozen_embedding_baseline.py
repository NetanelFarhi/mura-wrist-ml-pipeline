"""Patient-grouped-CV Logistic Regression over frozen study embeddings.

The baseline deliberately separates two development decisions:

* choose Logistic Regression ``C`` with patient-grouped CV *inside* internal
  training data; then
* fit the selected estimator on all internal-training studies.

Internal validation is not accepted by :meth:`fit`, which makes it impossible
for this class to use validation labels while choosing ``C``. Validation is
reserved for the per-model operating threshold and reporting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.training.metrics import safe_auprc, safe_auroc


SelectionMetric = Literal["auroc", "auprc"]


@dataclass(frozen=True)
class LogisticBaselineSelection:
    """Auditable train-only hyperparameter-selection record for Model A."""

    selected_c: float
    selection_metric: SelectionMetric
    candidate_results: tuple[dict[str, float], ...]
    candidate_c_values: tuple[float, ...]
    group_cv_folds: int
    group_field: str
    class_weight: str | None
    random_seed: int
    tie_break: str = "higher_metric_then_lower_C"

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


def make_patient_grouped_folds(
    labels: np.ndarray | Sequence[int],
    patient_ids: np.ndarray | Sequence[str],
    *,
    n_splits: int,
    random_seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return validated, patient-disjoint internal-training folds.

    Stratification is used only to make class mix more stable. The non-
    negotiable safety condition is grouping by ``patient_id``.
    """

    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    groups = np.asarray(patient_ids, dtype=str).reshape(-1)
    if len(y) != len(groups):
        raise ValueError("patient_ids must have one value per training study.")
    if n_splits < 2:
        raise ValueError("group_cv_folds must be at least 2.")
    if len(np.unique(y)) < 2:
        raise ValueError("Grouped CV C selection requires both NORMAL and ABNORMAL training studies.")
    if np.any(groups == ""):
        raise ValueError("Every training study needs a non-empty patient_id for grouped CV.")
    if len(np.unique(groups)) < n_splits:
        raise ValueError("Not enough unique patients for the requested grouped CV folds.")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(random_seed),
    )
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_index, holdout_index in splitter.split(np.zeros(len(y)), y, groups):
        train_index = np.asarray(train_index, dtype=np.int64)
        holdout_index = np.asarray(holdout_index, dtype=np.int64)
        if len(train_index) == 0 or len(holdout_index) == 0:
            raise RuntimeError("Grouped CV produced an empty fold.")
        if set(groups[train_index]).intersection(groups[holdout_index]):
            raise RuntimeError("Patient leakage detected inside grouped CV.")
        folds.append((train_index, holdout_index))

    if len(folds) != n_splits:
        raise RuntimeError("Grouped CV did not produce the requested number of folds.")
    return folds


class FrozenEmbeddingLogisticBaseline:
    """Small, deterministic Model A baseline with train-only C selection."""

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
                "Phase 1 permits a deliberately small grouped-CV search: provide one to three C values."
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
        train_patient_ids: np.ndarray | Sequence[str],
        *,
        group_cv_folds: int = 5,
    ) -> "FrozenEmbeddingLogisticBaseline":
        """Choose ``C`` with grouped CV inside internal training only.

        There are intentionally no validation arguments. After this method
        returns, callers may score a separate internal validation set once to
        select the pre-registered study-level operating threshold.
        """

        x_train, y_train = _validate_features_and_labels(
            train_embeddings, train_labels, split_name="train"
        )
        groups = np.asarray(train_patient_ids, dtype=str).reshape(-1)
        folds = make_patient_grouped_folds(
            y_train,
            groups,
            n_splits=int(group_cv_folds),
            random_seed=self.random_seed,
        )

        best_c: float | None = None
        best_score = float("-inf")
        candidate_results: list[dict[str, float]] = []
        epsilon = 1e-12
        for c_value in sorted(self.c_values):
            oof_probability = np.full(len(y_train), np.nan, dtype=np.float64)
            for train_index, holdout_index in folds:
                candidate = self._make_estimator(c_value)
                candidate.fit(x_train[train_index], y_train[train_index])
                oof_probability[holdout_index] = candidate.predict_proba(
                    x_train[holdout_index]
                )[:, 1]
            if not np.all(np.isfinite(oof_probability)):
                raise RuntimeError("Grouped CV did not create one OOF probability per train study.")
            auroc = safe_auroc(y_train, oof_probability)
            auprc = safe_auprc(y_train, oof_probability)
            if auroc is None or auprc is None:
                raise ValueError("Cannot compute grouped-CV metrics for a one-class training population.")
            score = auroc if self.selection_metric == "auroc" else auprc
            candidate_results.append(
                {
                    "C": float(c_value),
                    "train_group_cv_auroc": float(auroc),
                    "train_group_cv_auprc": float(auprc),
                }
            )
            if score > best_score + epsilon or (
                abs(score - best_score) <= epsilon and (best_c is None or c_value < best_c)
            ):
                best_c, best_score = float(c_value), float(score)

        if best_c is None:
            raise RuntimeError("No Logistic Regression candidate was fitted.")
        final_estimator = self._make_estimator(best_c)
        final_estimator.fit(x_train, y_train)
        self.estimator_ = final_estimator
        self.selection_ = LogisticBaselineSelection(
            selected_c=best_c,
            selection_metric=self.selection_metric,
            candidate_results=tuple(candidate_results),
            candidate_c_values=tuple(sorted(self.c_values)),
            group_cv_folds=int(group_cv_folds),
            group_field="patient_id",
            class_weight=self.class_weight,
            random_seed=self.random_seed,
        )
        self.fitted_on_ = "all_internal_train_after_patient_grouped_cv"
        return self

    def predict_proba(self, embeddings: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        """Return abnormal-study probabilities for mean-pooled study embeddings."""

        if self.estimator_ is None:
            raise RuntimeError("The baseline is not fitted. Call fit on internal training data first.")
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("embeddings must be a non-empty [studies, features] matrix.")
        return np.asarray(self.estimator_.predict_proba(x)[:, 1], dtype=np.float64)

    def save(self, output_path: str | Path) -> Path:
        """Save the fitted pipeline and its train-only-C-selection provenance."""

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
        return {
            **self.selection_.to_dict(),
            "fitted_on": self.fitted_on_,
            "internal_validation_used_for_c_selection": False,
        }


__all__ = [
    "FrozenEmbeddingLogisticBaseline",
    "LogisticBaselineSelection",
    "SelectionMetric",
    "make_patient_grouped_folds",
]
