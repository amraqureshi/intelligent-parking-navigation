"""
DQN Training Pipeline
=====================
Module: backend.train_model

Trains the Deep Q-Network agent on the ParkingEnvironment over 400 episodes.
Tracks reward progression, rolling success rates, Bellman loss, and saves:
- Best model checkpoint: models/dqn_parking.pth
- Training curve visualization: results/training_curve.png

Run Command:
    python backend/train_model.py
"""

import os
import sys
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.parking_environment import ParkingEnvironment
from backend.dqn_agent import DQNAgent
from backend.replay_buffer import ReplayBuffer


def train_dqn(
    num_episodes: int = 400,
    max_steps_per_episode: int = 50,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 5e-4,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.992,
    target_update_freq: int = 15,
    seed: int = 42,
    save_path: str = "models/dqn_parking.pth",
    results_path: str = "results/training_curve.png",
):
    """
    Main training routine for DQN parking navigation.
    """
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("=" * 65)
    print("  Intelligent Parking Navigation - DQN Training Pipeline")
    print("=" * 65)
    print(f"Target Checkpoint : {save_path}")
    print(f"Episodes          : {num_episodes}")
    print(f"Batch Size        : {batch_size}")
    print(f"Learning Rate     : {lr}")
    print(f"Discount Factor   : {gamma}")
    print(f"Epsilon Decay     : {epsilon_decay} -> min {epsilon_end}")
    print("=" * 65)

    env = ParkingEnvironment(max_steps=max_steps_per_episode)
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim, lr=lr, gamma=gamma)
    replay_buffer = ReplayBuffer(capacity=25000)

    epsilon = epsilon_start
    episode_rewards = []
    success_history = []
    loss_history = []

    # [1/3] Buffer Warm-up
    print("\n[1/3] Pre-filling Replay Buffer with random exploration...")
    prefill_steps = 600
    p_step = 0
    state, _ = env.reset(seed=seed, random_occupancy_rate=0.5)
    while p_step < prefill_steps:
        valid = env.get_valid_actions()
        action = np.random.choice(valid) if valid else np.random.randint(env.action_dim)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push(state, action, reward, next_state, done)
        p_step += 1
        state = env.reset(random_occupancy_rate=float(np.random.uniform(0.2, 0.8)))[0] if done else next_state

    print(f"Replay buffer pre-filled with {len(replay_buffer)} transitions.")
    print("\n[2/3] Beginning Training Loop (400 episodes)...")

    best_score = -float("inf")
    best_weights = None

    for episode in range(1, num_episodes + 1):
        # Vary occupancy rate (20% to 80%) for generalized navigation
        occ_rate = float(np.random.uniform(0.2, 0.8))
        state, _ = env.reset(random_occupancy_rate=occ_rate)
        ep_reward = 0.0
        ep_loss = []
        ep_success = False

        for step in range(max_steps_per_episode):
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, epsilon=epsilon, valid_actions=valid_actions)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)
            loss = agent.train_step(replay_buffer, batch_size=batch_size)
            if loss is not None:
                ep_loss.append(loss)

            ep_reward += reward
            state = next_state

            if info.get("success"):
                ep_success = True

            if done:
                break

        # Decay exploration rate
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Update target network periodically
        if episode % target_update_freq == 0:
            agent.update_target_network()

        episode_rewards.append(ep_reward)
        success_history.append(1.0 if ep_success else 0.0)
        avg_loss = float(np.mean(ep_loss)) if ep_loss else 0.0
        loss_history.append(avg_loss)

        if episode % 25 == 0 or episode == num_episodes:
            window = min(25, episode)
            recent_success = float(np.mean(success_history[-window:])) * 100.0
            recent_reward = float(np.mean(episode_rewards[-window:]))
            print(
                f"Episode {episode:03d}/{num_episodes} | "
                f"Avg Reward (last {window:02d}): {recent_reward:6.2f} | "
                f"Success Rate: {recent_success:5.1f}% | "
                f"Epsilon: {epsilon:5.3f} | "
                f"Avg Loss: {avg_loss:6.4f}"
            )

            # Combined score: prioritize success rate, broken by average reward
            score = recent_success * 10.0 + recent_reward
            if score >= best_score:
                best_score = score
                best_weights = copy.deepcopy(agent.policy_net.state_dict())
                agent.save(save_path)

    print("\n[3/3] Training Complete! Saving final checkpoint and plots...")
    if best_weights is not None:
        agent.policy_net.load_state_dict(best_weights)
    agent.save(save_path)
    print(f"Optimal model weights checkpoint saved to: {save_path}")

    # Generate and save training progression plot
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Reward Progression
    ax1.plot(episode_rewards, label="Episode Reward", alpha=0.4, color="#38bdf8")
    rolling_window = 20
    if len(episode_rewards) >= rolling_window:
        rolling_reward = np.convolve(episode_rewards, np.ones(rolling_window)/rolling_window, mode="valid")
        ax1.plot(range(rolling_window - 1, len(episode_rewards)), rolling_reward,
                 label=f"{rolling_window}-Ep Moving Avg", color="#0284c7", linewidth=2.2)
    ax1.set_title("Reward Progression During Training", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Episode", fontsize=10)
    ax1.set_ylabel("Total Cumulative Reward", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # Success Rate Progression
    if len(success_history) >= rolling_window:
        rolling_succ = np.convolve(success_history, np.ones(rolling_window)/rolling_window, mode="valid") * 100.0
        ax2.plot(range(rolling_window - 1, len(success_history)), rolling_succ,
                 label=f"{rolling_window}-Ep Success %", color="#10b981", linewidth=2.2)
    ax2.set_title("Parking Navigation Success Rate (%)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Episode", fontsize=10)
    ax2.set_ylabel("Success Rate (%)", fontsize=10)
    ax2.set_ylim(-5, 105)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(results_path, dpi=160)
    plt.close()
    print(f"Training visualization saved to: {results_path}")


if __name__ == "__main__":
    train_dqn()
