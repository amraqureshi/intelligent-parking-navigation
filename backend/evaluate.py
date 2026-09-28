"""
DQN vs. NetworkX Baseline Evaluation & Benchmarking
===================================================
Module: backend.evaluate

Compares the trained Deep Q-Network (DQN) policy against the optimal NetworkX
Dijkstra shortest-path baseline across identical, reproducibly seeded parking
scenarios and varied congestion levels (25%, 50%, 75% spot occupancy).

Outputs Generated:
- results/comparison_results.json
- results/comparison_results.csv
- results/performance_vs_occupancy.png

Run Command:
    python backend/evaluate.py
    python backend/evaluate.py --baseline-only
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.parking_environment import ParkingEnvironment
from backend.dqn_agent import DQNAgent


def run_baseline_episode(env: ParkingEnvironment, seed: int, occ_rate: float) -> Dict[str, Any]:
    """
    Executes a single evaluation episode using the NetworkX Dijkstra baseline.
    At each step, calculates shortest path to the closest reachable unoccupied spot.
    """
    state, info = env.reset(seed=seed, random_occupancy_rate=occ_rate)
    ep_reward = 0.0
    ep_steps = 0
    invalid_moves = 0
    done = False
    success = False

    while not done and ep_steps < env.max_steps:
        best_spot, min_dist = env._find_nearest_available_spot(env.current_node)
        if best_spot is None or min_dist == float("inf"):
            # No reachable available spot
            break

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
            success = True
            break

    return {
        "success": success,
        "steps": ep_steps,
        "reward": ep_reward,
        "invalid_moves": invalid_moves,
    }


def run_dqn_episode(env: ParkingEnvironment, agent: DQNAgent, seed: int, occ_rate: float) -> Dict[str, Any]:
    """
    Executes a single evaluation episode using greedy DQN policy inference (epsilon=0.0).
    """
    state, info = env.reset(seed=seed, random_occupancy_rate=occ_rate)
    ep_reward = 0.0
    ep_steps = 0
    invalid_moves = 0
    done = False
    success = False

    while not done and ep_steps < env.max_steps:
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, epsilon=0.0, valid_actions=valid_actions)

        next_state, reward, term, trunc, step_info = env.step(action)
        done = term or trunc
        ep_reward += reward
        ep_steps += 1
        state = next_state

        if "Invalid move" in step_info.get("message", "") or "Blocked" in step_info.get("message", ""):
            invalid_moves += 1

        if step_info.get("success"):
            success = True
            break

    return {
        "success": success,
        "steps": ep_steps,
        "reward": ep_reward,
        "invalid_moves": invalid_moves,
    }


def evaluate_agent(
    model_path: str = "models/dqn_parking.pth",
    num_episodes: int = 50,
    occupancy_rates: Optional[List[float]] = None,
    allow_baseline_only: bool = False,
    output_dir: str = "results",
) -> Dict[str, Any]:
    """
    Runs systematic evaluation comparing NetworkX Baseline and DQN Agent
    across identical seeded scenarios and variable occupancy rates.
    """
    if occupancy_rates is None:
        occupancy_rates = [0.25, 0.50, 0.75]

    print("=" * 72)
    print("  Intelligent Parking Navigation - Policy Evaluation Benchmark")
    print("=" * 72)
    print(f"Model Checkpoint : {model_path}")
    print(f"Episodes / Level : {num_episodes}")
    print(f"Occupancy Rates  : {[f'{int(r*100)}%' for r in occupancy_rates]}")
    print("=" * 72)

    has_model = os.path.exists(model_path)
    env = ParkingEnvironment(max_steps=50)

    if not has_model:
        if not allow_baseline_only:
            raise FileNotFoundError(
                f"DQN checkpoint '{model_path}' not found! "
                f"Train the model first using 'python backend/train_model.py', "
                f"or pass allow_baseline_only=True / --baseline-only to run only the baseline benchmark."
            )
        print(f"\n[NOTICE] Checkpoint '{model_path}' not found.")
        print("[NOTICE] Evaluating NetworkX Shortest Path baseline only (no fabricated DQN results).\n")
        agent = None
    else:
        agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)
        agent.load(model_path)
        print(f"\n[INFO] Loaded trained DQN agent from: {model_path}\n")

    results: Dict[str, Any] = {
        "num_episodes_per_level": num_episodes,
        "occupancy_rates": occupancy_rates,
        "metrics_by_occupancy": {},
        "summary": {},
    }

    csv_rows = []
    # Header: Method, Occupancy_Rate, Success_Rate_Pct, Avg_Steps, Avg_Reward, Invalid_Moves_Total
    csv_rows.append("Method,Occupancy_Rate,Success_Rate_Pct,Avg_Steps,Avg_Reward,Invalid_Moves_Total")

    for occ_rate in occupancy_rates:
        occ_key = f"occupancy_{int(occ_rate * 100)}%"
        print(f"\nEvaluating Congestion Level: {int(occ_rate * 100)}% Occupied ({num_episodes} identical trials)")

        # Baseline evaluation
        b_succ, b_steps, b_rews, b_invalids = 0, [], [], 0
        for ep in range(num_episodes):
            seed = int(occ_rate * 1000) + ep * 13
            b_res = run_baseline_episode(env, seed=seed, occ_rate=occ_rate)
            if b_res["success"]:
                b_succ += 1
            b_steps.append(b_res["steps"])
            b_rews.append(b_res["reward"])
            b_invalids += b_res["invalid_moves"]

        b_succ_pct = round((b_succ / num_episodes) * 100.0, 2)
        b_avg_step = round(float(np.mean(b_steps)), 2)
        b_avg_rew = round(float(np.mean(b_rews)), 2)

        occ_result: Dict[str, Any] = {
            "baseline": {
                "success_rate_pct": b_succ_pct,
                "avg_steps": b_avg_step,
                "avg_reward": b_avg_rew,
                "invalid_moves_total": b_invalids,
            }
        }
        csv_rows.append(f"Baseline,{int(occ_rate * 100)}%,{b_succ_pct},{b_avg_step},{b_avg_rew},{b_invalids}")

        print(f"  [Baseline] Success: {b_succ_pct:5.1f}% | Avg Steps: {b_avg_step:5.2f} | Avg Reward: {b_avg_rew:6.2f} | Invalids: {b_invalids}")

        # DQN evaluation if available
        if agent is not None:
            d_succ, d_steps, d_rews, d_invalids = 0, [], [], 0
            for ep in range(num_episodes):
                seed = int(occ_rate * 1000) + ep * 13
                d_res = run_dqn_episode(env, agent=agent, seed=seed, occ_rate=occ_rate)
                if d_res["success"]:
                    d_succ += 1
                d_steps.append(d_res["steps"])
                d_rews.append(d_res["reward"])
                d_invalids += d_res["invalid_moves"]

            d_succ_pct = round((d_succ / num_episodes) * 100.0, 2)
            d_avg_step = round(float(np.mean(d_steps)), 2)
            d_avg_rew = round(float(np.mean(d_rews)), 2)

            occ_result["dqn"] = {
                "success_rate_pct": d_succ_pct,
                "avg_steps": d_avg_step,
                "avg_reward": d_avg_rew,
                "invalid_moves_total": d_invalids,
            }
            csv_rows.append(f"DQN,{int(occ_rate * 100)}%,{d_succ_pct},{d_avg_step},{d_avg_rew},{d_invalids}")

            print(f"  [DQN Agent] Success: {d_succ_pct:5.1f}% | Avg Steps: {d_avg_step:5.2f} | Avg Reward: {d_avg_rew:6.2f} | Invalids: {d_invalids}")

        results["metrics_by_occupancy"][occ_key] = occ_result

    # Save output artifacts
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "comparison_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\n[Artifact] JSON results saved to: {json_path}")

    csv_path = os.path.join(output_dir, "comparison_results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_rows) + "\n")
    print(f"[Artifact] CSV results saved to : {csv_path}")

    # Generate visual comparison plot
    plot_path = os.path.join(output_dir, "performance_vs_occupancy.png")
    plot_comparison(results, plot_path)
    print(f"[Artifact] Plot saved to        : {plot_path}")

    print("\n" + "=" * 72)
    print("  Benchmark Evaluation Completed Successfully")
    print("=" * 72)
    return results


def plot_comparison(results: Dict[str, Any], save_path: str):
    """
    Renders high-quality multi-panel visualization comparing Baseline and DQN metrics.
    """
    occupancies = results["occupancy_rates"]
    labels = [f"{int(r * 100)}%" for r in occupancies]
    x = np.arange(len(labels))
    width = 0.35

    baseline_succ = [results["metrics_by_occupancy"][f"occupancy_{int(r*100)}%"]["baseline"]["success_rate_pct"] for r in occupancies]
    baseline_steps = [results["metrics_by_occupancy"][f"occupancy_{int(r*100)}%"]["baseline"]["avg_steps"] for r in occupancies]
    baseline_rews = [results["metrics_by_occupancy"][f"occupancy_{int(r*100)}%"]["baseline"]["avg_reward"] for r in occupancies]

    has_dqn = "dqn" in next(iter(results["metrics_by_occupancy"].values()))
    if has_dqn:
        dqn_succ = [results["metrics_by_occupancy"][f"occupancy_{int(r*100)}%"]["dqn"]["success_rate_pct"] for r in occupancies]
        dqn_steps = [results["metrics_by_occupancy"][f"occupancy_{int(r*100)}%"]["dqn"]["avg_steps"] for r in occupancies]
        dqn_rews = [results["metrics_by_occupancy"][f"occupancy_{int(r*100)}%"]["dqn"]["avg_reward"] for r in occupancies]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Subplot 1: Success Rate (%)
    ax1 = axes[0]
    if has_dqn:
        ax1.bar(x - width/2, baseline_succ, width, label="Baseline (Dijkstra)", color="#3b82f6", alpha=0.9)
        ax1.bar(x + width/2, dqn_succ, width, label="DQN Agent", color="#10b981", alpha=0.9)
    else:
        ax1.bar(x, baseline_succ, width, label="Baseline (Dijkstra)", color="#3b82f6", alpha=0.9)
    ax1.set_title("Success Rate vs Congestion", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Parking Lot Occupancy", fontsize=10)
    ax1.set_ylabel("Success Rate (%)", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 105)
    ax1.legend(loc="lower left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Average Steps
    ax2 = axes[1]
    if has_dqn:
        ax2.bar(x - width/2, baseline_steps, width, label="Baseline (Dijkstra)", color="#3b82f6", alpha=0.9)
        ax2.bar(x + width/2, dqn_steps, width, label="DQN Agent", color="#10b981", alpha=0.9)
    else:
        ax2.bar(x, baseline_steps, width, label="Baseline (Dijkstra)", color="#3b82f6", alpha=0.9)
    ax2.set_title("Average Navigation Steps", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Parking Lot Occupancy", fontsize=10)
    ax2.set_ylabel("Average Steps to Park", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Subplot 3: Average Reward
    ax3 = axes[2]
    if has_dqn:
        ax3.bar(x - width/2, baseline_rews, width, label="Baseline (Dijkstra)", color="#3b82f6", alpha=0.9)
        ax3.bar(x + width/2, dqn_rews, width, label="DQN Agent", color="#10b981", alpha=0.9)
    else:
        ax3.bar(x, baseline_rews, width, label="Baseline (Dijkstra)", color="#3b82f6", alpha=0.9)
    ax3.set_title("Average Cumulative Reward", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Parking Lot Occupancy", fontsize=10)
    ax3.set_ylabel("Cumulative Reward", fontsize=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.legend(loc="lower left")
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DQN and Baseline on Parking Environment")
    parser.add_argument("--model-path", type=str, default="models/dqn_parking.pth", help="Path to trained DQN checkpoint")
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes per occupancy rate")
    parser.add_argument("--baseline-only", action="store_true", help="Evaluate baseline only without checking for DQN checkpoint")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save evaluation results")

    args = parser.parse_args()
    evaluate_agent(
        model_path=args.model_path,
        num_episodes=args.episodes,
        allow_baseline_only=args.baseline_only,
        output_dir=args.output_dir,
    )
