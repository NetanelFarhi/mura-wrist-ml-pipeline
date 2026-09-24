"""Loss construction and explicit class-imbalance decisions for Phase 1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


ClassWeightMode = Literal["none", "auto", "balanced"]


@dataclass(frozen=True)
class ClassWeightingDecision:
    """Recorded rationale for whether training uses a positive-class weight."""

    mode: ClassWeightMode
    use_positive_weight: bool
    positive_weight: float
    n_normal: int
    n_abnormal: int
    imbalance_ratio: float
    imbalance_ratio_threshold: float
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def decide_class_weighting(
    labels: Sequence[int] | np.ndarray,
    *,
    mode: ClassWeightMode = "auto",
    imbalance_ratio_threshold: float = 1.5,
) -> ClassWeightingDecision:
    """Decide class weighting from *training studies only*.

    ``auto`` activates only when the majority/minority study ratio reaches the
    documented threshold.  ``balanced`` is an explicit user choice; it is
    still recorded in the training artifact so it cannot be mistaken for an
    unexamined default.  Weighting never examines validation or test labels.
    """

    if mode not in ("none", "auto", "balanced"):
        raise ValueError("mode must be one of 'none', 'auto', or 'balanced'.")
    if imbalance_ratio_threshold < 1.0:
        raise ValueError("imbalance_ratio_threshold must be at least 1.0.")
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(y) == 0:
        raise ValueError("Cannot decide class weighting from an empty training label set.")
    if not set(y.tolist()).issubset({0, 1}):
        raise ValueError("Class weighting expects binary labels encoded as 0 (NORMAL) and 1 (ABNORMAL).")

    n_normal = int(np.sum(y == 0))
    n_abnormal = int(np.sum(y == 1))
    if n_normal == 0 or n_abnormal == 0:
        raise ValueError("Training requires both NORMAL and ABNORMAL studies.")
    imbalance_ratio = max(n_normal, n_abnormal) / min(n_normal, n_abnormal)
    positive_weight = n_normal / n_abnormal
    if mode == "none":
        use_weight = False
        rationale = "Disabled explicitly; unweighted study-level BCE is used."
    elif mode == "balanced":
        use_weight = True
        rationale = (
            "Enabled explicitly. Positive-class weight equals normal/abnormal "
            "study count, computed from training labels only."
        )
    elif imbalance_ratio >= imbalance_ratio_threshold:
        use_weight = True
        rationale = (
            "Enabled automatically because the training-study majority/minority ratio "
            f"({imbalance_ratio:.3f}) meets the configured threshold "
            f"({imbalance_ratio_threshold:.3f})."
        )
    else:
        use_weight = False
        rationale = (
            "Not used: the training-study majority/minority ratio "
            f"({imbalance_ratio:.3f}) is below the configured threshold "
            f"({imbalance_ratio_threshold:.3f})."
        )
    return ClassWeightingDecision(
        mode=mode,
        use_positive_weight=use_weight,
        positive_weight=float(positive_weight),
        n_normal=n_normal,
        n_abnormal=n_abnormal,
        imbalance_ratio=float(imbalance_ratio),
        imbalance_ratio_threshold=float(imbalance_ratio_threshold),
        rationale=rationale,
    )


class StudyProbabilityBCELoss(nn.Module):
    """Differentiable binary cross-entropy on one aggregated probability/study.

    The trainer obtains a view probability for every image and mean- or
    max-aggregates them within each study before calling this loss.  This makes
    the optimization target match the stated study-level prediction unit.
    """

    def __init__(self, decision: ClassWeightingDecision | None = None) -> None:
        super().__init__()
        positive_weight = (
            decision.positive_weight
            if decision is not None and decision.use_positive_weight
            else 1.0
        )
        self.register_buffer("positive_weight", torch.tensor(float(positive_weight)))
        self.use_positive_weight = bool(decision is not None and decision.use_positive_weight)

    def forward(self, probabilities: Tensor, targets: Tensor) -> Tensor:
        probabilities = probabilities.reshape(-1).clamp(min=1e-7, max=1.0 - 1e-7)
        targets = targets.reshape(-1).to(dtype=probabilities.dtype)
        if probabilities.shape != targets.shape:
            raise ValueError(
                "Study probabilities and targets must have the same shape; "
                f"got {tuple(probabilities.shape)} and {tuple(targets.shape)}."
            )
        per_study = F.binary_cross_entropy(probabilities, targets, reduction="none")
        if not self.use_positive_weight:
            return per_study.mean()
        weights = torch.where(
            targets > 0.5,
            self.positive_weight.to(dtype=probabilities.dtype),
            torch.ones_like(targets),
        )
        return torch.sum(per_study * weights) / torch.sum(weights)


def make_study_probability_loss(
    decision: ClassWeightingDecision | None = None,
    *,
    device: torch.device | str | None = None,
) -> StudyProbabilityBCELoss:
    """Create the study-level loss on the selected compute device."""

    criterion = StudyProbabilityBCELoss(decision)
    if device is not None:
        criterion = criterion.to(torch.device(device))
    return criterion


def make_bce_with_logits_loss(
    decision: ClassWeightingDecision | None = None,
    *,
    device: torch.device | str | None = None,
) -> nn.BCEWithLogitsLoss:
    """Build a view-level BCE loss for explicitly labelled-view diagnostics.

    Phase 1 training uses :class:`StudyProbabilityBCELoss`; this helper is
    provided for controlled diagnostics without duplicating weighting policy.
    """

    positive_weight = (
        decision.positive_weight
        if decision is not None and decision.use_positive_weight
        else 1.0
    )
    kwargs: dict[str, Tensor] = {}
    if positive_weight != 1.0:
        kwargs["pos_weight"] = torch.tensor([positive_weight], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(**kwargs)
    if device is not None:
        criterion = criterion.to(torch.device(device))
    return criterion


__all__ = [
    "ClassWeightMode",
    "ClassWeightingDecision",
    "StudyProbabilityBCELoss",
    "decide_class_weighting",
    "make_bce_with_logits_loss",
    "make_study_probability_loss",
]
