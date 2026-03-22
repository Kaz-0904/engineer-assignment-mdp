from __future__ import annotations

from typing import Dict, List

from mdp_env import EngineerAssignmentMDP


Policy = List[int]


def _argmax(values: List[float]) -> int:
    best_idx = 0
    best_value = values[0]
    for idx, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_idx = idx
            best_value = value
    return best_idx


def greedy_policy_from_values(mdp: EngineerAssignmentMDP, values: List[float]) -> Policy:
    policy = [0 for _ in range(mdp.num_states)]
    for s_idx, state in enumerate(mdp.states):
        legal_indices = [mdp.action_to_idx[a] for a in mdp.get_legal_actions(state)]
        q_values = [mdp.expected_return(s_idx, a_idx, values) for a_idx in legal_indices]
        policy[s_idx] = legal_indices[_argmax(q_values)]
    return policy


def evaluate_policy(mdp: EngineerAssignmentMDP, policy: Policy, theta: float = 1e-10, max_iterations: int = 10000) -> List[float]:
    values = [0.0 for _ in range(mdp.num_states)]
    for _ in range(max_iterations):
        delta = 0.0
        new_values = values.copy()
        for s_idx in range(mdp.num_states):
            new_values[s_idx] = mdp.expected_return(s_idx, policy[s_idx], values)
            delta = max(delta, abs(new_values[s_idx] - values[s_idx]))
        values = new_values
        if delta < theta:
            break
    return values


def value_iteration(mdp: EngineerAssignmentMDP, theta: float = 1e-10, max_iterations: int = 10000) -> Dict[str, object]:
    values = [0.0 for _ in range(mdp.num_states)]
    history: List[float] = []
    iteration = 0
    for iteration in range(max_iterations):
        delta = 0.0
        new_values = values.copy()
        for s_idx, state in enumerate(mdp.states):
            legal_indices = [mdp.action_to_idx[a] for a in mdp.get_legal_actions(state)]
            q_values = [mdp.expected_return(s_idx, a_idx, values) for a_idx in legal_indices]
            new_values[s_idx] = max(q_values)
            delta = max(delta, abs(new_values[s_idx] - values[s_idx]))
        values = new_values
        history.append(delta)
        if delta < theta:
            break
    return {"values": values, "policy": greedy_policy_from_values(mdp, values), "iterations": iteration + 1, "delta_history": history}


def policy_iteration(mdp: EngineerAssignmentMDP, theta: float = 1e-10, max_iterations: int = 1000) -> Dict[str, object]:
    policy = [mdp.action_to_idx[mdp.get_legal_actions(state)[0]] for state in mdp.states]
    stable = False
    iteration = 0
    while not stable and iteration < max_iterations:
        iteration += 1
        values = evaluate_policy(mdp, policy, theta=theta)
        stable = True
        for s_idx, state in enumerate(mdp.states):
            legal_indices = [mdp.action_to_idx[a] for a in mdp.get_legal_actions(state)]
            q_values = [mdp.expected_return(s_idx, a_idx, values) for a_idx in legal_indices]
            best_action = legal_indices[_argmax(q_values)]
            if best_action != policy[s_idx]:
                stable = False
            policy[s_idx] = best_action
    values = evaluate_policy(mdp, policy, theta=theta)
    return {"values": values, "policy": policy, "iterations": iteration, "stable": stable}


def policy_to_action_names(mdp: EngineerAssignmentMDP, policy: Policy) -> Dict[str, str]:
    return {str(state): mdp.actions[policy[s_idx]] for s_idx, state in enumerate(mdp.states)}
