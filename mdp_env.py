from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Dict, List, Sequence, Tuple


State = Tuple[str, int]
Action = str


@dataclass(frozen=True)
class MDPParameters:
    """TeXノートおよび依頼文の仮パラメータをまとめた設定。"""

    K: int = 3
    gamma: float = 0.95
    R_H: float = 10.0
    R_L: float = 6.0
    C_w: float = 4.0
    C_edu: float = 1.0
    d_H: float = 0.75
    d_L: float = 0.85
    p_H: Tuple[float, ...] = (0.10, 0.30, 0.55, 0.80)
    p_L: Tuple[float, ...] = (0.40, 0.60, 0.75, 0.90)
    q_T: Tuple[float, ...] = (0.70, 0.55, 0.40, 0.00)

    def validate(self) -> None:
        expected_len = self.K + 1
        for name, values in (("p_H", self.p_H), ("p_L", self.p_L), ("q_T", self.q_T)):
            if len(values) != expected_len:
                raise ValueError(f"{name} の長さは K+1={expected_len} である必要があります。")
            for value in values:
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{name} の要素は [0, 1] に入る必要があります。")
        for name, value in (("gamma", self.gamma), ("d_H", self.d_H), ("d_L", self.d_L)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} は [0, 1] の範囲で指定してください。")


class EngineerAssignmentMDP:
    Z_VALUES: Tuple[str, ...] = ("B", "H", "L")
    ACTIONS: Tuple[Action, ...] = ("a_H", "a_L", "a_T", "a_N")

    def __init__(self, params: MDPParameters | None = None):
        self.params = params or MDPParameters()
        self.params.validate()
        self.states: List[State] = [(z, k) for z in self.Z_VALUES for k in range(self.params.K + 1)]
        self.state_to_idx: Dict[State, int] = {state: idx for idx, state in enumerate(self.states)}
        self.actions: List[Action] = list(self.ACTIONS)
        self.action_to_idx: Dict[Action, int] = {action: idx for idx, action in enumerate(self.actions)}
        self.legal_actions_map: Dict[State, List[Action]] = {
            state: self._legal_actions_for_state(state) for state in self.states
        }
        self.transition_probabilities = [
            [[0.0 for _ in self.states] for _ in self.actions] for _ in self.states
        ]
        self.rewards = [
            [[0.0 for _ in self.states] for _ in self.actions] for _ in self.states
        ]
        self._build_transition_and_reward_tables()

    @property
    def num_states(self) -> int:
        return len(self.states)

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    def _legal_actions_for_state(self, state: State) -> List[Action]:
        return ["a_H", "a_L", "a_T", "a_N"] if state[0] == "B" else ["a_N"]

    def reward_from_next_state(self, action: Action, next_state: State) -> float:
        z_prime = next_state[0]
        reward = self.params.R_H if z_prime == "H" else self.params.R_L if z_prime == "L" else -self.params.C_w
        if action == "a_T":
            reward -= self.params.C_edu
        return reward

    def _set_transition(self, state: State, action: Action, next_state: State, probability: float) -> None:
        s_idx = self.state_to_idx[state]
        a_idx = self.action_to_idx[action]
        sp_idx = self.state_to_idx[next_state]
        self.transition_probabilities[s_idx][a_idx][sp_idx] += probability
        self.rewards[s_idx][a_idx][sp_idx] = self.reward_from_next_state(action, next_state)

    def _build_transition_and_reward_tables(self) -> None:
        p = self.params
        for state in self.states:
            z, k = state
            if z == "B":
                self._set_transition(state, "a_H", ("H", k), p.p_H[k])
                self._set_transition(state, "a_H", ("B", k), 1.0 - p.p_H[k])
                self._set_transition(state, "a_L", ("L", k), p.p_L[k])
                self._set_transition(state, "a_L", ("B", k), 1.0 - p.p_L[k])
                if k < p.K:
                    self._set_transition(state, "a_T", ("B", k + 1), p.q_T[k])
                    self._set_transition(state, "a_T", ("B", k), 1.0 - p.q_T[k])
                else:
                    self._set_transition(state, "a_T", ("B", k), 1.0)
                self._set_transition(state, "a_N", ("B", k), 1.0)
            elif z == "H":
                self._set_transition(state, "a_N", ("H", k), p.d_H)
                self._set_transition(state, "a_N", ("B", max(k - 1, 0)), 1.0 - p.d_H)
            else:
                self._set_transition(state, "a_N", ("L", k), p.d_L)
                self._set_transition(state, "a_N", ("B", max(k - 1, 0)), 1.0 - p.d_L)

    def get_legal_actions(self, state: State) -> List[Action]:
        return list(self.legal_actions_map[state])

    def validate_transition_probabilities(self, atol: float = 1e-10) -> List[Tuple[State, Action, float]]:
        violations: List[Tuple[State, Action, float]] = []
        for state in self.states:
            s_idx = self.state_to_idx[state]
            for action in self.get_legal_actions(state):
                a_idx = self.action_to_idx[action]
                total = sum(self.transition_probabilities[s_idx][a_idx])
                if abs(total - 1.0) > atol:
                    violations.append((state, action, total))
        return violations

    def expected_return(self, state_idx: int, action_idx: int, values: Sequence[float]) -> float:
        total = 0.0
        for sp_idx, prob in enumerate(self.transition_probabilities[state_idx][action_idx]):
            if prob == 0.0:
                continue
            reward = self.rewards[state_idx][action_idx][sp_idx]
            total += prob * (reward + self.params.gamma * values[sp_idx])
        return total

    def sample_next_state(self, rng: Random, state: State, action: Action) -> Tuple[State, float]:
        s_idx = self.state_to_idx[state]
        a_idx = self.action_to_idx[action]
        draw = rng.random()
        cumulative = 0.0
        for sp_idx, prob in enumerate(self.transition_probabilities[s_idx][a_idx]):
            cumulative += prob
            if draw <= cumulative + 1e-12:
                return self.states[sp_idx], self.rewards[s_idx][a_idx][sp_idx]
        last_idx = self.num_states - 1
        return self.states[last_idx], self.rewards[s_idx][a_idx][last_idx]

    def describe_problem(self) -> Dict[str, object]:
        return {
            "problem_setting": "待機中の技術者に対して、高単価案件・低単価案件・研修・現状維持のどれを選ぶと長期報酬が最大化されるかを考える有限状態MDP",
            "state": "s=(z,k), z∈{B,H,L}, k∈{0,...,K}",
            "action": "A={a_H,a_L,a_T,a_N}、ただし H/L では a_N のみ合法",
            "transition": "Bでは応募成功・研修成功に応じて遷移し、H/Lでは継続またはベンチ復帰時に適合度が1低下",
            "reward": "遷移後状態 z' が H なら +R_H、L なら +R_L、B なら -C_w。a_T では追加で -C_edu",
            "objective": "期待割引累積報酬 E[Σ γ^t r_t] の最大化",
            "baseline_policies": ["baseline_1: 常に高単価案件狙い", "baseline_2: 常に低単価案件狙い", "baseline_3: 低適合度では研修、高適合度では高単価", "baseline_4: その場の稼働確率最大化"],
            "evaluation_metrics": ["平均累積報酬", "平均稼働率", "H滞在割合", "L滞在割合", "B滞在割合", "平均適合度", "ベンチからの平均脱出速度"],
        }
