
"""
MURA Agent — Deterministic Research Tool-Selection Policy

Validates supplied Agent workflow records and selects
optional explanation tools.

No inference, image processing, or explanation execution.
No modification of model outputs or input dictionaries.
"""

import math

POLICY_VERSION = "1.1"

OPTIONAL_TOOLS = [
    "tta_uncertainty",
    "gradcam_explainer",
]


def _valid_probability(value):
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def _valid_nonnegative(value):
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and value >= 0.0
    )


def _close(a, b, tolerance=1e-10):
    return math.isclose(
        a,
        b,
        rel_tol=0.0,
        abs_tol=tolerance,
    )


def validate_agent_result(agent_result, selected_threshold):
    """
    Validate mandatory safety, frozen classification,
    view consistency, and deterministic Agent action.

    Returns a JSON-serializable validation record.
    """

    issues = []

    def invalid(message):
        return {
            "status": "INVALID",
            "issues": [message],
            "research_use_only": True,
        }

    if (
        not _valid_probability(selected_threshold)
        or not 0.0 < selected_threshold < 1.0
    ):
        return invalid("Invalid frozen selected threshold.")

    if not isinstance(agent_result, dict):
        return invalid("Agent result must be a dictionary.")

    required = [
        "study_id",
        "workflow_status",
        "qc_status",
        "anatomy_status",
        "classification_status",
        "classification_performed",
        "classification_valid",
        "classification_modified",
        "predicted_class",
        "study_probability",
        "classification_threshold",
        "agent_action",
        "view_consistency",
    ]

    missing = [
        key for key in required
        if key not in agent_result
    ]

    if missing:
        return invalid(f"Missing Agent fields: {missing}")

    # --------------------------------------------------------
    # Mandatory workflow safety
    # --------------------------------------------------------

    if (
        not isinstance(agent_result["study_id"], str)
        or not agent_result["study_id"].strip()
    ):
        issues.append("Invalid study identifier.")

    expected_statuses = {
        "workflow_status": "SUCCESS",
        "qc_status": "PASS",
        "anatomy_status": "SUPPORTED_WRIST",
        "classification_status": "SUCCESS",
    }

    for field, expected in expected_statuses.items():
        if agent_result[field] != expected:
            issues.append(f"{field} is not {expected}.")

    if agent_result["classification_performed"] is not True:
        issues.append("Classification was not performed.")

    if agent_result["classification_valid"] is not True:
        issues.append("Classification is not valid.")

    if agent_result["classification_modified"] is not False:
        issues.append("Classification modification detected.")

    # --------------------------------------------------------
    # Frozen study classification
    # --------------------------------------------------------

    predicted_class = agent_result["predicted_class"]
    probability = agent_result["study_probability"]
    classification_threshold = agent_result[
        "classification_threshold"
    ]

    valid_class = (
        type(predicted_class) is int
        and predicted_class in (0, 1)
    )

    if not valid_class:
        issues.append("Invalid predicted class.")

    if not _valid_probability(probability):
        issues.append("Invalid study probability.")

    if (
        not _valid_probability(classification_threshold)
        or not _close(
            classification_threshold,
            selected_threshold,
        )
    ):
        issues.append(
            "Classification threshold differs from frozen threshold."
        )

    if valid_class and _valid_probability(probability):
        expected_class = int(
            probability >= selected_threshold
        )

        if predicted_class != expected_class:
            issues.append(
                "Predicted class contradicts frozen threshold."
            )

    # --------------------------------------------------------
    # View-consistency structure
    # --------------------------------------------------------

    view = agent_result["view_consistency"]

    if not isinstance(view, dict):
        issues.append("Missing or invalid view-consistency result.")

        return {
            "status": "INVALID",
            "issues": issues,
            "research_use_only": True,
        }

    view_required = [
        "status",
        "image_count",
        "min_probability",
        "max_probability",
        "mean_probability",
        "probability_range",
        "probability_std",
        "threshold_disagreement",
        "image_classes",
        "threshold",
    ]

    missing_view = [
        key for key in view_required
        if key not in view
    ]

    if missing_view:
        issues.append(
            f"Missing view-consistency fields: {missing_view}"
        )

        return {
            "status": "INVALID",
            "issues": issues,
            "research_use_only": True,
        }

    if view["status"] != "SUCCESS":
        issues.append("View-consistency status is not SUCCESS.")

    view_threshold = view["threshold"]

    if (
        not _valid_probability(view_threshold)
        or not _close(
            view_threshold,
            selected_threshold,
        )
    ):
        issues.append(
            "View threshold differs from frozen threshold."
        )

    # --------------------------------------------------------
    # Image classes and disagreement
    # --------------------------------------------------------

    image_count = view["image_count"]
    image_classes = view["image_classes"]

    valid_count = (
        type(image_count) is int
        and image_count >= 1
    )

    valid_classes = (
        isinstance(image_classes, list)
        and all(
            type(value) is int and value in (0, 1)
            for value in image_classes
        )
    )

    if not valid_count:
        issues.append("Invalid image count.")

    if not valid_classes:
        issues.append("Invalid image classes.")

    complete_classes = (
        valid_count
        and valid_classes
        and len(image_classes) == image_count
    )

    if (
        valid_count
        and valid_classes
        and not complete_classes
    ):
        issues.append(
            "Image-class count differs from image count."
        )

    image_predictions = agent_result.get("image_predictions")

    if isinstance(image_predictions, list) and valid_count:
        if len(image_predictions) != image_count:
            issues.append(
                "View image count differs from Agent image count."
            )

    disagreement = view["threshold_disagreement"]

    if type(disagreement) is not bool:
        issues.append("Invalid threshold-disagreement flag.")

    elif complete_classes:
        expected_disagreement = (
            len(set(image_classes)) > 1
        )

        if disagreement != expected_disagreement:
            issues.append(
                "Threshold disagreement contradicts image classes."
            )

    # --------------------------------------------------------
    # View probability statistics
    # --------------------------------------------------------

    minimum = view["min_probability"]
    maximum = view["max_probability"]
    mean = view["mean_probability"]
    probability_range = view["probability_range"]
    probability_std = view["probability_std"]

    valid_min = _valid_probability(minimum)
    valid_max = _valid_probability(maximum)
    valid_mean = _valid_probability(mean)

    if not valid_min:
        issues.append("Invalid minimum image probability.")

    if not valid_max:
        issues.append("Invalid maximum image probability.")

    if not valid_mean:
        issues.append("Invalid mean image probability.")

    if valid_min and valid_max:
        if minimum > maximum:
            issues.append("Minimum exceeds maximum.")

    if valid_min and valid_max and valid_mean:
        if minimum <= maximum:
            if not (
                minimum - 1e-6
                <= mean
                <= maximum + 1e-6
            ):
                issues.append(
                    "Mean probability lies outside reported bounds."
                )

    if valid_mean and _valid_probability(probability):
        if not _close(mean, probability, tolerance=1e-5):
            issues.append(
                "Study probability differs from view mean."
            )

    if not _valid_probability(probability_range):
        issues.append("Invalid probability range.")

    elif valid_min and valid_max:
        if not _close(
            probability_range,
            maximum - minimum,
            tolerance=1e-5,
        ):
            issues.append(
                "Probability range contradicts reported bounds."
            )

    if not _valid_nonnegative(probability_std):
        issues.append("Invalid probability standard deviation.")

    if (
        valid_count
        and image_count == 1
        and _valid_nonnegative(probability_std)
        and probability_std > 1e-6
    ):
        issues.append(
            "Single-image study has nonzero probability standard deviation."
        )

    # --------------------------------------------------------
    # Additional class/bound checks
    # --------------------------------------------------------

    if complete_classes and valid_min and valid_max:

        if image_count == 1 and not _close(
            minimum,
            maximum,
            tolerance=1e-6,
        ):
            issues.append(
                "Single-image study has inconsistent bounds."
            )

        if all(value == 1 for value in image_classes):
            if minimum < selected_threshold - 1e-6:
                issues.append(
                    "All-positive image classes contradict minimum."
                )

        if all(value == 0 for value in image_classes):
            if maximum >= selected_threshold + 1e-6:
                issues.append(
                    "All-negative image classes contradict maximum."
                )

        if (
            type(disagreement) is bool
            and disagreement
        ):
            if not (
                minimum < selected_threshold
                and maximum >= selected_threshold
            ):
                issues.append(
                    "Disagreement contradicts probability bounds."
                )

    # --------------------------------------------------------
    # Deterministic Agent decision
    # --------------------------------------------------------

    if valid_class and type(disagreement) is bool:

        expected_action = (
            "priority_human_review"
            if predicted_class == 1 or disagreement
            else "routine_workflow"
        )

        if agent_result["agent_action"] != expected_action:
            issues.append(
                "Agent action contradicts decision policy."
            )

    return {
        "status": "VALID" if not issues else "INVALID",
        "issues": issues,
        "research_use_only": True,
    }


def select_agent_tools(agent_result, selected_threshold):
    """
    Select optional research explanation tools only after
    complete deterministic validation.

    No tools are executed.
    """

    validation = validate_agent_result(
        agent_result,
        selected_threshold,
    )

    study_id = (
        agent_result.get("study_id")
        if isinstance(agent_result, dict)
        else None
    )

    action = (
        agent_result.get("agent_action")
        if isinstance(agent_result, dict)
        else None
    )

    if validation["status"] != "VALID":

        return {
            "study_id": study_id,
            "policy_version": POLICY_VERSION,
            "policy_status": "SAFETY_BLOCK",
            "mandatory_safety_satisfied": False,
            "original_agent_action": action,
            "selected_tools": [],
            "tool_decisions": {
                tool: {
                    "requested": False,
                    "reason": "Safety or consistency validation failed.",
                }
                for tool in OPTIONAL_TOOLS
            },
            "requires_human_review": True,
            "research_use_only": True,
            "reason": "Safety or consistency validation failed.",
            "validation_issues": validation["issues"],
        }

    if action == "priority_human_review":

        selected_tools = list(OPTIONAL_TOOLS)

        reason = (
            "Valid priority-review study: request "
            "supplementary TTA and Grad-CAM explanations."
        )

    elif action == "routine_workflow":

        selected_tools = []

        reason = (
            "Valid routine research workflow: "
            "optional explanations not requested."
        )

    else:
        raise AssertionError(
            "Validated Agent action is unexpectedly unsupported."
        )

    return {
        "study_id": study_id,
        "policy_version": POLICY_VERSION,
        "policy_status": "SUCCESS",
        "mandatory_safety_satisfied": True,
        "original_agent_action": action,
        "selected_tools": selected_tools,
        "tool_decisions": {
            tool: {
                "requested": tool in selected_tools,
                "reason": reason,
            }
            for tool in OPTIONAL_TOOLS
        },
        "requires_human_review": True,
        "research_use_only": True,
        "reason": reason,
        "validation_issues": [],
    }
