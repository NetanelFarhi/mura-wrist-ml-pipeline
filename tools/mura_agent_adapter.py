"""
MURA Agent Orchestrator Integration Adapter.

Research use only.

Runs the original Agent orchestrator and then applies
the validated policy through the integration module.

The adapter does not execute TTA or Grad-CAM.

WARNING:
Calling this adapter with the real orchestrator
may execute model inference.
"""

import copy


def run_mura_agent_with_policy(
    study_id,
    images,
    image_paths,
    orchestrator,
    policy_module,
    integration_module,
    selected_threshold,
    probability_tolerance,
):
    """
    Run the original MURA Agent orchestrator and then
    evaluate the validated policy.

    The policy does not modify the original Agent decision.
    Explanation tools are not executed by this adapter.

    IMPORTANT:
    Calling this function executes the orchestrator,
    which may perform model inference.
    """

    # 1. Execute the original Agent workflow.
    agent_result = orchestrator(
        study_id=study_id,
        images=images,
        image_paths=image_paths,
    )

    # 2. Preserve the original Agent result.
    agent_snapshot = copy.deepcopy(agent_result)

    # 3. Apply the validated policy.
    integration_result = integration_module.apply_agent_policy(
        agent_result,
        policy_module=policy_module,
        selected_threshold=selected_threshold,
        probability_tolerance=probability_tolerance,
    )

    # 4. Verify original Agent output was preserved.
    assert agent_result == agent_snapshot, (
        "Original Agent result was modified."
    )

    return integration_result
