"""
DQN Evaluation & Benchmarking (Integration Placeholder for Claude)
=================================================================
Module: backend.evaluate

INSTRUCTIONS FOR CLAUDE:
------------------------
Evaluate the trained DQN model checkpoint (models/dqn_parking.pth) against
various parking scenarios and benchmark against the NetworkX optimal baseline.

Metrics Measured:
- Navigation Success Rate (%)
- Average Steps to Park
- Average Cumulative Reward
- Invalid Action / Collision Rate

Run Command:
    python backend/evaluate.py
"""

import os
import sys
from typing import Dict, Any, List
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.parking_environment import ParkingEnvironment, find_parking, load_model
from backend.dqn_agent import DQNAgent


def evaluate_agent(
    model_path: str = "models/dqn_parking.pth",
    num_episodes: int = 50,
    occupancy_rates: List[float] = [0.25, 0.50, 0.75],
) -> Dict[str, Any]:
    """
    Evaluates policy across varying congestion levels.
    """
    print("=" * 65)
    print("  Intelligent Parking Navigation - Policy Evaluation Benchmark")
    print("=" * 65)

    env = ParkingEnvironment(max_steps=50)

    # Check if trained weights exist
    has_model = os.path.exists(model_path)
    if has_model:
        agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)
        agent.load(model_path)
        print(f"Loaded trained DQN policy from: {model_path}")
    else:
        print(f"Notice: '{model_path}' not found.")
        print("Evaluating NetworkX Shortest Path baseline as benchmark reference.")
        agent = None

    results = {}

    for occ_rate in occupancy_rates:
        successes = 0
        total_steps = []
        total_rewards = []
        invalid_moves = 0

        for ep in range(num_episodes):
            state, info = env.reset(seed=ep * 10, random_occupancy_rate=occ_rate)
            ep_reward = 0.0
            ep_steps = 0
            done = False

            while not done and ep_steps < env.max_steps:
                if agent is not None:
                    valid_actions = env.get_valid_actions()
                    action = agent.select_action(state, epsilon=0.0, valid_actions=valid_actions)
                else:
                    # Baseline Dijkstra path action
                    best_spot, _ = env._find_nearest_available_spot(env.current_node)
                    if best_spot is None:
                        break
                    import networkx as nx
                    path = nx.shortest_path(env.graph, env.current_node, best_spot, weight="weight")
                    if len(path) < 2:
                        break
                    action = env.graph[path[0]][path[1]]["action"]

                next_state, reward, term, trunc, step_info = env.step(action)
                done = term or trunc
                ep_reward += reward
                ep_steps += 1
                state = next_state

                if "Invalid move" in step_info.get("message", "") or "Blocked" in step_info.get("message", ""):
                    invalid_moves += 1

                if step_info.get("success"):
                    successes += 1
                    break

            total_steps.append(ep_steps)
            total_rewards.append(ep_reward)

        succ_pct = (successes / num_episodes) * 100
        avg_step = float(np.mean(total_steps))
        avg_rew = float(np.mean(total_rewards))

        results[f"occupancy_{int(occ_rate * 100)}%"] = {
            "success_rate_pct": round(succ_pct, 2),
            "avg_steps": round(avg_step, 2),
            "avg_reward": round(avg_rew, 2),
            "invalid_moves_total": invalid_moves,
        }

        print(f"\n[Congestion: {int(occ_rate * 100)}% Occupied - {num_episodes} Trials]")
        print(f"  * Success Rate  : {succ_pct:5.1f}%")
        print(f"  * Avg Steps     : {avg_step:5.2f}")
        print(f"  * Avg Reward    : {avg_rew:5.2f}")
        print(f"  * Invalid Moves : {invalid_moves}")

    print("\n" + "=" * 65)
    print("Benchmark complete.")
    return results


if __name__ == "__main__":
    evaluate_agent()
