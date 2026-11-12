"""MURA Agent structured research report generator."""

import copy
import math

def generate_agent_report(agent_result, image_count):
    """
    Generate a read-only structured research report
    from an existing MURA Agent Orchestrator result.

    No model inference, QC execution, or Agent
    decision recalculation is performed.
    """

    if not isinstance(agent_result, dict):
        raise TypeError("agent_result must be a dictionary")

    if type(image_count) is not int or image_count < 1:
        raise ValueError("image_count must be a positive integer")

    workflow_status = agent_result.get("workflow_status")
    classification_status = agent_result.get(
        "classification_status"
    )
    classification_performed = agent_result.get(
        "classification_performed"
    )
    classification_valid = agent_result.get(
        "classification_valid"
    )

    successful_classification = (
        workflow_status == "COMPLETED"
        and classification_status == "COMPLETED"
        and classification_performed is True
        and classification_valid is True
    )

    probability = agent_result.get("study_probability")
    predicted_class = agent_result.get("predicted_class")
    threshold = agent_result.get("classification_threshold")

    # --------------------------------------------------
    # 1. Validate classification information
    # --------------------------------------------------

    if successful_classification:

        if type(predicted_class) is not int:
            raise ValueError("Predicted class must be an integer")

        if predicted_class not in (0, 1):
            raise ValueError("Predicted class must be 0 or 1")

        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(float(probability))
            or not 0.0 <= float(probability) <= 1.0
        ):
            raise ValueError("Invalid study probability")

        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or not 0.0 <= float(threshold) <= 1.0
        ):
            raise ValueError("Invalid classification threshold")

        if predicted_class != int(probability >= threshold):
            raise ValueError(
                "Predicted class is inconsistent with threshold"
            )

        classification = (
            "Abnormality predicted"
            if predicted_class == 1
            else "No abnormality predicted"
        )

        disagreement = agent_result.get(
            "image_threshold_disagreement"
        )

        if disagreement is not None and type(disagreement) is not bool:
            raise ValueError(
                "Invalid image threshold disagreement"
            )

    else:

        # Do not display residual predictions as valid
        # when classification is blocked or incomplete.
        probability = None
        predicted_class = None
        threshold = None
        disagreement = None
        classification = "Not available"

    # --------------------------------------------------
    # 2. Extract Agent decision without recalculation
    # --------------------------------------------------

    original_action = agent_result.get("agent_action")

    if not isinstance(original_action, dict):
        raise ValueError("agent_action must be a dictionary")

    action = original_action.get("action")
    action_reason = original_action.get("reason")

    allowed_actions = {
        "routine_workflow",
        "priority_human_review",
        "human_escalation",
        "unsupported_input"
    }

    if action not in allowed_actions:
        raise ValueError("Invalid Agent workflow action")

    # --------------------------------------------------
    # 3. Preserve available QC and anatomical metadata
    # --------------------------------------------------

    anatomy_result = agent_result.get("anatomy_result")

    if not isinstance(anatomy_result, dict):
        anatomy_result = {}

    qc_results = agent_result.get("qc_results")

    if not isinstance(qc_results, list):
        qc_results = []

    issues = agent_result.get("issues")

    if not isinstance(issues, list):
        issues = []

    # --------------------------------------------------
    # 4. Construct structured research report
    # --------------------------------------------------

    report = {
        "report_type": "MURA Agent Research Report",
        "research_use_only": True,

        "study_information": {
            "study_id": agent_result.get("study_id"),
            "image_count": image_count
        },

        "workflow": {
            "status": workflow_status,
            "reason": agent_result.get("workflow_reason"),
            "issues": copy.deepcopy(issues)
        },

        "technical_qc": {
            "status": agent_result.get("qc_status"),
            "results": copy.deepcopy(qc_results)
        },

        "anatomical_scope": {
            "status": agent_result.get("anatomy_status"),
            "metadata_source": anatomy_result.get(
                "metadata_source"
            ),
            "anatomy_image_verified": anatomy_result.get(
                "anatomy_image_verified"
            ),
            "issues": copy.deepcopy(
                anatomy_result.get("issues", [])
            )
        },

        "classification": {
            "status": classification_status,
            "performed": classification_performed,
            "valid": classification_valid,
            "predicted_class": predicted_class,
            "interpretation": classification,
            "study_probability": probability,
            "threshold": threshold,
            "image_threshold_disagreement": disagreement
        },

        "agent_decision": {
            "action": action,
            "reason": action_reason,
            "classification_modified": agent_result.get(
                "classification_modified"
            )
        },

        "limitations": [
            "Research prototype; not validated for clinical use.",
            "Model predictions are not radiologist diagnoses.",
            "Anatomical scope validation relies on supplied metadata.",
            "Technical QC does not establish diagnostic adequacy.",
            "Image-level threshold agreement does not establish diagnostic correctness.",
            "The prespecified 95% sensitivity target was not met on official held-out validation.",
            "Agent workflow actions are experimental and are not clinically validated triage recommendations."
        ]
    }

    return report
