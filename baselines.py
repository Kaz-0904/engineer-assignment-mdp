from __future__ import annotations

from typing import Dict, List

from mdp_env import EngineerAssignmentMDP


Policy = List[int]


def _policy_template(mdp: EngineerAssignmentMDP) -> Policy:
    return [mdp.action_to_idx[mdp.get_legal_actions(state)[0]] for state in mdp.states]


def baseline_1_high_only(mdp: EngineerAssignmentMDP) -> Policy:
    policy = _policy_template(mdp)
    for s_idx, (z, _k) in enumerate(mdp.states):
        if z == "B":
            policy[s_idx] = mdp.action_to_idx["a_H"]
    return policy


def baseline_2_low_only(mdp: EngineerAssignmentMDP) -> Policy:
    policy = _policy_template(mdp)
    for s_idx, (z, _k) in enumerate(mdp.states):
        if z == "B":
            policy[s_idx] = mdp.action_to_idx["a_L"]
    return policy


def baseline_3_train_then_high(mdp: EngineerAssignmentMDP) -> Policy:
    policy = _policy_template(mdp)
    for s_idx, (z, k) in enumerate(mdp.states):
        if z == "B":
            policy[s_idx] = mdp.action_to_idx["a_T"] if k <= 1 else mdp.action_to_idx["a_H"]
    return policy


def baseline_4_myopic_utilization(mdp: EngineerAssignmentMDP) -> Policy:
    policy = _policy_template(mdp)
    for s_idx, (z, k) in enumerate(mdp.states):
        if z == "B":
            policy[s_idx] = mdp.action_to_idx["a_H"] if mdp.params.p_H[k] >= mdp.params.p_L[k] else mdp.action_to_idx["a_L"]
    return policy


def build_baseline_policies(mdp: EngineerAssignmentMDP) -> Dict[str, Policy]:
    return {
        "baseline_1_high_only": baseline_1_high_only(mdp),
        "baseline_2_low_only": baseline_2_low_only(mdp),
        "baseline_3_train_then_high": baseline_3_train_then_high(mdp),
        "baseline_4_myopic_utilization": baseline_4_myopic_utilization(mdp),
    }
