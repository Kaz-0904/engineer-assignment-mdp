from __future__ import annotations

from random import Random
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

from baselines import build_baseline_policies
from mdp_env import EngineerAssignmentMDP, State
from solvers import policy_iteration, value_iteration


def simulate_policy(
    mdp: EngineerAssignmentMDP,
    policy: List[int],
    num_episodes: int = 1000,
    num_steps: int = 100,
    initial_state: Optional[State] = None,
    seed: int = 42,
) -> Dict[str, object]:
    rng = Random(seed)
    initial_state = initial_state or ("B", 0)
    discounted_returns: List[float] = []
    utilization_rates: List[float] = []
    high_ratios: List[float] = []
    low_ratios: List[float] = []
    bench_ratios: List[float] = []
    avg_ks: List[float] = []
    bench_exit_speeds: List[float] = []

    for _episode in range(num_episodes):
        state = initial_state
        discounted_return = 0.0
        state_counts = {"B": 0, "H": 0, "L": 0}
        k_history: List[int] = []
        bench_run_length = 0
        bench_escape_times: List[int] = []

        for t in range(num_steps):
            z, k = state
            state_counts[z] += 1
            k_history.append(k)
            if z == "B":
                bench_run_length += 1
            s_idx = mdp.state_to_idx[state]
            action = mdp.actions[policy[s_idx]]
            next_state, reward = mdp.sample_next_state(rng, state, action)
            discounted_return += (mdp.params.gamma ** t) * reward
            if state[0] == "B" and next_state[0] in {"H", "L"}:
                bench_escape_times.append(bench_run_length)
                bench_run_length = 0
            elif state[0] != "B":
                bench_run_length = 0
            state = next_state

        discounted_returns.append(discounted_return)
        utilization_rates.append((state_counts["H"] + state_counts["L"]) / num_steps)
        high_ratios.append(state_counts["H"] / num_steps)
        low_ratios.append(state_counts["L"] / num_steps)
        bench_ratios.append(state_counts["B"] / num_steps)
        avg_ks.append(mean(k_history))
        bench_exit_speeds.append(mean(bench_escape_times) if bench_escape_times else float(num_steps))

    return {
        "mean_discounted_return": mean(discounted_returns),
        "std_discounted_return": pstdev(discounted_returns),
        "mean_utilization_rate": mean(utilization_rates),
        "mean_high_ratio": mean(high_ratios),
        "mean_low_ratio": mean(low_ratios),
        "mean_bench_ratio": mean(bench_ratios),
        "mean_k": mean(avg_ks),
        "mean_bench_exit_time": mean(bench_exit_speeds),
        "episode_returns": discounted_returns,
    }


def compare_policies(mdp: EngineerAssignmentMDP, num_episodes: int = 1000, num_steps: int = 100, seed: int = 42) -> Dict[str, Dict[str, object]]:
    vi_result = value_iteration(mdp)
    pi_result = policy_iteration(mdp)
    policies = {"value_iteration_optimal": vi_result["policy"], "policy_iteration_optimal": pi_result["policy"], **build_baseline_policies(mdp)}
    return {
        name: simulate_policy(mdp, policy, num_episodes=num_episodes, num_steps=num_steps, seed=seed + i)
        for i, (name, policy) in enumerate(policies.items())
    }


def results_to_table(results: Dict[str, Dict[str, object]]) -> List[Dict[str, float]]:
    rows = []
    for name, metrics in results.items():
        rows.append({
            "policy": name,
            "mean_discounted_return": metrics["mean_discounted_return"],
            "mean_utilization_rate": metrics["mean_utilization_rate"],
            "mean_high_ratio": metrics["mean_high_ratio"],
            "mean_low_ratio": metrics["mean_low_ratio"],
            "mean_bench_ratio": metrics["mean_bench_ratio"],
            "mean_k": metrics["mean_k"],
            "mean_bench_exit_time": metrics["mean_bench_exit_time"],
        })
    return sorted(rows, key=lambda row: row["mean_discounted_return"], reverse=True)


def plot_policy_comparison(results: Dict[str, Dict[str, object]], save_path: Optional[str] = None) -> Tuple[Any, Any]:
    import matplotlib.pyplot as plt

    metrics = [
        ("mean_discounted_return", "平均累積報酬"),
        ("mean_utilization_rate", "平均稼働率"),
        ("mean_high_ratio", "H滞在割合"),
        ("mean_low_ratio", "L滞在割合"),
        ("mean_bench_ratio", "B滞在割合"),
        ("mean_k", "平均適合度"),
        ("mean_bench_exit_time", "平均ベンチ脱出時間"),
    ]
    policy_names = list(results.keys())
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    flat_axes = axes.flatten()
    for ax, (metric_key, title) in zip(flat_axes, metrics):
        values = [results[name][metric_key] for name in policy_names]
        ax.bar(policy_names, values, color="steelblue")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
    flat_axes[-1].axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, axes


if __name__ == "__main__":
    mdp = EngineerAssignmentMDP()
    print('transition violations:', mdp.validate_transition_probabilities())
    results = compare_policies(mdp)
    for row in results_to_table(results):
        print(row)
    try:
        plot_policy_comparison(results, save_path='policy_comparison.png')
        print('saved plot to policy_comparison.png')
    except ModuleNotFoundError as exc:
        print(f'plot skipped because matplotlib is unavailable: {exc}')
