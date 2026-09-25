"""
DQN Training Pipeline (Integration Placeholder for Claude)
=========================================================
Module: backend.train_model

INSTRUCTIONS FOR CLAUDE:
------------------------
This script runs the DQN training loop on the ParkingEnvironment.

Key Training Specifications:
- Environment: ParkingEnvironment(max_steps=50)
- Episodes: 300 to 600 episodes (fast training within minutes on CPU/GPU)
- Replay Pre-fill: Warm up buffer with 500-1000 transitions before training
- Evaluation: Evaluate every 50 episodes on fixed scenarios
- Checkpoint Output: models/dqn_parking.pth
- Reward Curve Output: results/training_curve.png

Run Command:
    python backend/train_model.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

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
    lr: float = 1e-3,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.992,
    target_update_freq: int = 20,
    save_path: str = "models/dqn_parking.pth",
    results_path: str = "results/training_curve.png",
):
    """
    Main training routine for DQN parking navigation.
    """
    print("=" * 60)
    print("  Intelligent Parking Navigation - DQN Training Pipeline")
    print("=" * 60)
    print(f"Target Checkpoint : {save_path}")
    print(f"Episodes          : {num_episodes}")
    print(f"Batch Size        : {batch_size}")
    print(f"Learning Rate     : {lr}")
    print("=" * 60)

    env = ParkingEnvironment(max_steps=max_steps_per_episode)
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim, lr=lr, gamma=gamma)
    replay_buffer = ReplayBuffer(capacity=25000)

    epsilon = epsilon_start
    episode_rewards = []
    success_history = []
    loss_history = []

    # -----------------------------------------------------------------------
    # TODO (Claude): Finalize and tune the training loop below as needed.
    # Standard DQN loop structure provided for immediate execution:
    # -----------------------------------------------------------------------
    print("\n[1/3] Pre-filling Replay Buffer with random exploration...")
    prefill_steps = 500
    p_step = 0
    state, _ = env.reset(random_occupancy_rate=0.5)
    while p_step < prefill_steps:
        # Use valid actions to ensure high-quality transitions
        valid = env.get_valid_actions()
        action = np.random.choice(valid) if valid else np.random.randint(env.action_dim)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push(state, action, reward, next_state, done)
        p_step += 1
        state = env.reset(random_occupancy_rate=0.5)[0] if done else next_state

    print(f"Replay buffer pre-filled with {len(replay_buffer)} transitions.")
    print("\n[2/3] Beginning Training Loop...")

    best_success_rate = 0.0

    for episode in range(1, num_episodes + 1):
        # Vary occupancy rate for generalized navigation
        occ_rate = float(np.random.uniform(0.3, 0.7))
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

        # Decay exploration
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Update target network
        if episode % target_update_freq == 0:
            agent.update_target_network()

        episode_rewards.append(ep_reward)
        success_history.append(1.0 if ep_success else 0.0)
        avg_loss = np.mean(ep_loss) if ep_loss else 0.0
        loss_history.append(avg_loss)

        if episode % 25 == 0 or episode == num_episodes:
            recent_success = np.mean(success_history[-25:]) * 100
            recent_reward = np.mean(episode_rewards[-25:])
            print(
                f"Episode {episode:03d}/{num_episodes} | "
                f"Avg Reward (last 25): {recent_reward:6.2f} | "
                f"Success Rate: {recent_success:5.1f}% | "
                f"Epsilon: {epsilon:4.3f} | "
                f"Avg Loss: {avg_loss:.4f}"
            )

            if recent_success >= best_success_rate:
                best_success_rate = recent_success
                agent.save(save_path)

    print("\n[3/3] Training Complete! Saving plots and final weights...")
    agent.save(save_path)

    # Plot results
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, label="Episode Reward", alpha=0.6, color="blue")
    rolling_reward = np.convolve(episode_rewards, np.ones(20)/20, mode="valid")
    plt.plot(rolling_reward, label="20-Ep Moving Avg", color="darkblue", linewidth=2)
    plt.title("Reward Progression")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    rolling_succ = np.convolve(success_history, np.ones(20)/20, mode="valid") * 100
    plt.plot(rolling_succ, label="Success Rate (%)", color="green", linewidth=2)
    plt.title("Parking Success Rate")
    plt.xlabel("Episode")
    plt.ylabel("Success %")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(results_path, dpi=150)
    plt.close()
    print(f"Training visualization saved to: {results_path}")


if __name__ == "__main__":
    train_dqn()
