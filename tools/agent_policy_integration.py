"""
MURA Agent Policy Integration
Research use only.

Applies the validated policy to an existing Agent result.
No model inference or explanation tool execution.
"""

import copy


def apply_agent_policy(
    agent_result,
    policy_module,
    selected_threshold,
    probability_tolerance,
):
    """
    Apply validated policy to an existing Agent result.

    Missing image_predictions are passed to the
    validated policy as None, allowing fail-closed
    handling without a KeyError.

    Does not modify the Agent result or execute tools.
    """

    original_result = copy.deepcopy(agent_result)

    # Safe access: missing predictions become None.
    predictions = agent_result.get("image_predictions")

    view_result = policy_module.view_consistency(
        predictions,
        selected_threshold=selected_threshold
    )

    policy_result = policy_module.evaluate_agent_policy(
        agent_result,
        view_result,
        selected_threshold=selected_threshold,
        probability_tolerance=probability_tolerance
    )

    assert agent_result == original_result, (
        "Agent result was modified."
    )

    return {
        "agent_result": copy.deepcopy(agent_result),
        "view_consistency": view_result,
        "policy_result": policy_result,
    }
