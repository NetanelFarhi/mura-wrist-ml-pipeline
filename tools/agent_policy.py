"""
MURA Agent — Research Policy Candidate

Validated policy functions from Notebook 09.

Research use only.
No model inference or explanation execution.
"""
import math
import numpy as np
POLICY_VERSION = '1.1.0'

def view_consistency(image_predictions, selected_threshold):
    """
    Analyze variability between image-level predictions.

    Uses the frozen classification threshold from Notebook 08.
    Does not modify study-level classification.
    """
    result = {'status': 'INVALID_INPUT', 'reason': None}
    threshold = float(selected_threshold)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        result['reason'] = 'Threshold must be between 0 and 1'
        return result
    if not isinstance(image_predictions, (list, tuple)):
        result['reason'] = 'Expected a list of image predictions'
        return result
    if not image_predictions:
        return {'status': 'INSUFFICIENT_DATA', 'reason': 'No image predictions provided'}
    if any((not isinstance(item, dict) or 'probability' not in item or (not isinstance(item['probability'], (int, float, np.number))) or isinstance(item['probability'], (bool, np.bool_)) for item in image_predictions)):
        result['reason'] = 'Invalid image prediction records'
        return result
    probabilities = np.asarray([item['probability'] for item in image_predictions], dtype=np.float64)
    if not np.all(np.isfinite(probabilities)):
        result['reason'] = 'Non-finite probabilities detected'
        return result
    if np.any((probabilities < 0) | (probabilities > 1)):
        result['reason'] = 'Probabilities must be between 0 and 1'
        return result
    image_classes = (probabilities >= threshold).astype(int)
    return {'status': 'SUCCESS', 'image_count': len(probabilities), 'min_probability': float(np.min(probabilities)), 'max_probability': float(np.max(probabilities)), 'mean_probability': float(np.mean(probabilities)), 'probability_range': float(np.ptp(probabilities)), 'probability_std': float(np.std(probabilities, ddof=0)), 'threshold_disagreement': bool(np.any(image_classes == 0) and np.any(image_classes == 1)), 'image_classes': image_classes.tolist(), 'threshold': float(threshold)}

def select_explanation_tools(agent_result, selected_threshold, probability_tolerance):
    """
    Deterministic, fail-closed explanation-tool selection.

    Uses validated Agent outputs only.
    Does not modify classification or Agent decisions.
    Does not execute TTA or Grad-CAM.
    """
    failures = []

    def require(condition, message):
        if not condition:
            failures.append(message)
    if not isinstance(agent_result, dict):
        agent_result = {}
        failures.append('Agent result is not a dictionary.')
    study_id = agent_result.get('study_id')
    action_record = agent_result.get('agent_action')
    action = action_record.get('action') if isinstance(action_record, dict) else None
    require(isinstance(study_id, str) and bool(study_id.strip()), 'Invalid study identifier.')
    anatomy = agent_result.get('anatomy_result')
    require(agent_result.get('anatomy_status') == 'SUPPORTED_WRIST', 'Anatomical scope not supported.')
    require(isinstance(anatomy, dict) and anatomy.get('status') == 'SUPPORTED_WRIST' and (anatomy.get('anatomical_region') == 'XR_WRIST') and (anatomy.get('issues') == []), 'Invalid anatomical scope record.')
    require(agent_result.get('workflow_status') == 'COMPLETED', 'Workflow not completed.')
    qc_results = agent_result.get('qc_results')
    require(agent_result.get('qc_status') == 'PASS', 'Technical QC did not pass.')
    require(isinstance(qc_results, list) and len(qc_results) > 0 and all((isinstance(item, dict) and isinstance(item.get('image_id'), str) and bool(item['image_id']) and (item.get('status') == 'PASS') and (item.get('blocking_failure') is False) and (item.get('issues') == []) for item in qc_results)), 'Invalid image-level QC records.')
    require(agent_result.get('classification_status') == 'COMPLETED', 'Classification not completed.')
    require(agent_result.get('classification_performed') is True, 'Classification not performed.')
    require(agent_result.get('classification_valid') is True, 'Classification invalid.')
    require(agent_result.get('classification_modified') is False, 'Classification modified or not verified.')
    classification = agent_result.get('classification_result')
    require(isinstance(classification, dict) and classification.get('status') == 'COMPLETED', 'Invalid classification result record.')

    def valid_probability(value):
        return type(value) in (int, float) and math.isfinite(value) and (0 <= value <= 1)
    probability = agent_result.get('study_probability')
    predicted_class = agent_result.get('predicted_class')
    threshold = agent_result.get('classification_threshold')
    require(valid_probability(probability), 'Invalid study probability.')
    require(type(predicted_class) is int and predicted_class in (0, 1), 'Invalid predicted class.')
    require(valid_probability(threshold), 'Invalid classification threshold.')
    canonical_threshold = float(selected_threshold)
    require(valid_probability(threshold) and math.isclose(threshold, canonical_threshold, rel_tol=0, abs_tol=1e-12), 'Frozen threshold mismatch.')
    if valid_probability(probability) and valid_probability(threshold):
        if type(predicted_class) is int and predicted_class in (0, 1):
            require(predicted_class == int(probability >= threshold), 'Predicted class inconsistent with threshold.')
    image_predictions = agent_result.get('image_predictions')
    valid_images = isinstance(image_predictions, list) and len(image_predictions) > 0 and all((isinstance(item, dict) and isinstance(item.get('image_id'), str) and bool(item['image_id']) and valid_probability(item.get('probability')) for item in image_predictions))
    require(valid_images, 'Invalid image predictions.')
    disagreement = None
    if valid_images and valid_probability(threshold):
        image_ids = [item['image_id'] for item in image_predictions]
        qc_ids = [item['image_id'] for item in qc_results] if isinstance(qc_results, list) and all((isinstance(item, dict) and isinstance(item.get('image_id'), str) for item in qc_results)) else []
        require(len(image_ids) == len(set(image_ids)) and len(qc_ids) == len(set(qc_ids)) and (set(image_ids) == set(qc_ids)), 'Image prediction and QC coverage mismatch.')
        image_probabilities = [float(item['probability']) for item in image_predictions]
        calculated_mean = sum(image_probabilities) / len(image_probabilities)
        if valid_probability(probability):
            require(math.isclose(probability, calculated_mean, rel_tol=0, abs_tol=float(probability_tolerance)), 'Study probability inconsistent with image aggregation.')
        image_classes = [p >= threshold for p in image_probabilities]
        disagreement = any(image_classes) and (not all(image_classes))
        require(agent_result.get('image_threshold_disagreement') is disagreement, 'Image threshold disagreement mismatch.')
    require(action in ('priority_human_review', 'routine_workflow'), 'Invalid or unsupported Agent action.')
    if type(predicted_class) is int and predicted_class in (0, 1) and (disagreement is not None):
        expected_action = 'priority_human_review' if predicted_class == 1 or disagreement else 'routine_workflow'
        require(action == expected_action, 'Agent action inconsistent with original decision logic.')
    if isinstance(action_record, dict):
        require(action_record.get('classification_modified') is False, 'Agent action indicates modified classification.')
        require(action_record.get('predicted_class') == predicted_class, 'Agent action class mismatch.')
        require(action_record.get('study_probability') == probability, 'Agent action probability mismatch.')
        require(action_record.get('threshold') == threshold, 'Agent action threshold mismatch.')
        require(disagreement is not None and action_record.get('threshold_disagreement') is disagreement, 'Agent action disagreement mismatch.')
    safety_passed = len(failures) == 0
    request_tools = safety_passed and action == 'priority_human_review'
    if not safety_passed:
        reason = 'Mandatory validation failed: ' + ' '.join(failures)
    elif request_tools:
        reason = 'Validated priority-review study.'
    else:
        reason = 'Validated routine workflow.'
    return {'study_id': study_id if isinstance(study_id, str) else None, 'policy_version': POLICY_VERSION, 'mandatory_safety_passed': safety_passed, 'original_agent_action': action, 'tta_decision': 'REQUEST' if request_tools else 'SKIP', 'gradcam_decision': 'REQUEST' if request_tools else 'SKIP', 'tta_reason': reason, 'gradcam_reason': reason, 'validation_failures': failures, 'tools_executed': False, 'research_use_only': True}

def validate_view_agent_consistency(agent_result, view_result, selected_threshold, probability_tolerance):
    """
    Validate frozen image predictions, view-consistency metrics,
    study classification, and original Agent action.

    Does not modify inputs or execute explanation tools.
    """
    issues = []

    def check(condition, message):
        if not condition:
            issues.append(message)

    def valid_probability(value):
        return type(value) in (int, float) and math.isfinite(value) and (0 <= value <= 1)
    if not isinstance(agent_result, dict):
        agent_result = {}
        issues.append('Invalid Agent result.')
    if not isinstance(view_result, dict):
        view_result = {}
        issues.append('Invalid view-consistency result.')
    check(view_result.get('status') == 'SUCCESS', 'View-consistency status is not SUCCESS.')
    predictions = agent_result.get('image_predictions')
    valid_predictions = isinstance(predictions, list) and len(predictions) > 0 and all((isinstance(p, dict) and valid_probability(p.get('probability')) for p in predictions))
    check(valid_predictions, 'Invalid image predictions.')
    threshold = agent_result.get('classification_threshold')
    frozen_threshold = float(selected_threshold)
    check(valid_probability(threshold) and math.isclose(threshold, frozen_threshold, rel_tol=0, abs_tol=1e-12), 'Agent classification threshold mismatch.')
    view_threshold = view_result.get('threshold')
    check(valid_probability(view_threshold) and math.isclose(view_threshold, frozen_threshold, rel_tol=0, abs_tol=1e-12), 'View-consistency threshold mismatch.')
    probability = agent_result.get('study_probability')
    predicted_class = agent_result.get('predicted_class')
    check(valid_probability(probability), 'Invalid study probability.')
    check(type(predicted_class) is int and predicted_class in (0, 1), 'Invalid predicted class.')
    action_record = agent_result.get('agent_action')
    action = action_record.get('action') if isinstance(action_record, dict) else None
    if valid_predictions:
        probabilities = [p['probability'] for p in predictions]
        expected_view = view_consistency(predictions, selected_threshold=selected_threshold)
        required_fields = ['status', 'image_count', 'min_probability', 'max_probability', 'mean_probability', 'probability_range', 'probability_std', 'threshold_disagreement', 'image_classes', 'threshold']
        for field in required_fields:
            check(field in view_result, f'Missing view-consistency field: {field}.')
        numeric_fields = ['min_probability', 'max_probability', 'mean_probability', 'probability_range', 'probability_std']
        for field in numeric_fields:
            actual = view_result.get(field)
            expected = expected_view.get(field)
            check(valid_probability(actual) and math.isclose(actual, expected, rel_tol=0, abs_tol=float(probability_tolerance)), f'View-consistency metric mismatch: {field}.')
        image_count = view_result.get('image_count')
        check(type(image_count) is int and image_count == len(predictions), 'View-consistency image count mismatch.')
        image_classes = view_result.get('image_classes')
        expected_classes = expected_view['image_classes']
        check(isinstance(image_classes, list) and len(image_classes) == len(predictions) and all((type(c) is int and c in (0, 1) for c in image_classes)) and (image_classes == expected_classes), 'View-consistency image classes mismatch.')
        disagreement = view_result.get('threshold_disagreement')
        check(type(disagreement) is bool and disagreement is expected_view['threshold_disagreement'], 'View-consistency disagreement mismatch.')
        check(agent_result.get('image_threshold_disagreement') is expected_view['threshold_disagreement'], 'Agent disagreement flag mismatch.')
        if isinstance(action_record, dict):
            check(action_record.get('threshold_disagreement') is expected_view['threshold_disagreement'], 'Agent action disagreement mismatch.')
        if valid_probability(probability):
            check(math.isclose(probability, expected_view['mean_probability'], rel_tol=0, abs_tol=float(probability_tolerance)), 'Study probability aggregation mismatch.')
        if valid_probability(probability) and valid_probability(threshold) and (type(predicted_class) is int) and (predicted_class in (0, 1)):
            check(predicted_class == int(probability >= threshold), 'Study classification inconsistent with threshold.')
            expected_action = 'priority_human_review' if predicted_class == 1 or expected_view['threshold_disagreement'] else 'routine_workflow'
            check(action == expected_action, 'Agent action inconsistent with decision rules.')
    policy_result = select_explanation_tools(agent_result, selected_threshold=selected_threshold, probability_tolerance=probability_tolerance)
    check(policy_result['mandatory_safety_passed'] is True, 'Step 49 mandatory safety validation failed.')
    valid = len(issues) == 0
    return {'study_id': agent_result.get('study_id'), 'validation_status': 'VALID' if valid else 'INVALID', 'validation_passed': valid, 'issues': issues, 'tta_decision': policy_result['tta_decision'] if valid else 'SKIP', 'gradcam_decision': policy_result['gradcam_decision'] if valid else 'SKIP', 'tools_executed': False, 'research_use_only': True}


def evaluate_agent_policy(
    agent_result,
    view_result,
    selected_threshold,
    probability_tolerance,
):
    """
    Evaluate deterministic research policy.

    The supplied Agent action and frozen classification
    are never modified.

    Optional explanation tools are never executed.
    """

    validation = validate_view_agent_consistency(
        agent_result,
        view_result,
        selected_threshold=selected_threshold,
        probability_tolerance=probability_tolerance,
    )

    valid = validation["validation_passed"] is True

    study_id = (
        agent_result.get("study_id")
        if isinstance(agent_result, dict)
        else None
    )

    action_record = (
        agent_result.get("agent_action")
        if isinstance(agent_result, dict)
        else None
    )

    original_action = (
        action_record.get("action")
        if isinstance(action_record, dict)
        else None
    )

    if valid:
        selection = select_explanation_tools(
            agent_result,
            selected_threshold=selected_threshold,
            probability_tolerance=probability_tolerance,
        )

        # Independent safety gate: both validators must pass.
        valid = selection["mandatory_safety_passed"] is True

    else:
        selection = None

    if valid:
        tta_decision = selection["tta_decision"]
        gradcam_decision = selection["gradcam_decision"]

        tta_reason = selection["tta_reason"]
        gradcam_reason = selection["gradcam_reason"]

        issues = []
        policy_status = "VALID"

    else:
        tta_decision = "SKIP"
        gradcam_decision = "SKIP"

        tta_reason = "Safety or consistency validation failed."
        gradcam_reason = "Safety or consistency validation failed."

        issues = list(validation["issues"])

        if selection is not None:
            issues.extend(selection.get("validation_failures", []))

        if not issues:
            issues.append("Mandatory policy validation failed.")

        policy_status = "SAFETY_BLOCK"

    selected_tools = []

    if tta_decision == "REQUEST":
        selected_tools.append("tta_uncertainty")

    if gradcam_decision == "REQUEST":
        selected_tools.append("gradcam_explainer")

    return {
        "study_id": study_id,
        "policy_version": POLICY_VERSION,
        "policy_status": policy_status,
        "mandatory_safety_passed": valid,
        "original_agent_action": original_action,
        "selected_tools": selected_tools,
        "tta_decision": tta_decision,
        "gradcam_decision": gradcam_decision,
        "tta_reason": tta_reason,
        "gradcam_reason": gradcam_reason,
        "validation_issues": issues,
        "requires_human_review": (
            not valid or original_action == "priority_human_review"
        ),
        "tools_executed": False,
        "research_use_only": True,
    }
