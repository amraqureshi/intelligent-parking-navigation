"""
Deep Q-Network Agent (Integration Placeholder for Claude)
========================================================
Module: backend.dqn_agent

INSTRUCTIONS FOR CLAUDE:
------------------------
Implement the DQN Agent logic in this module.

Expected Specifications:
- Networks:
    - `policy_net`: Active Q-network updated via gradient descent.
    - `target_net`: Stabilizing Q-network updated periodically (e.g., every 500 steps).
- Hyperparameters (Recommended defaults for a college mini-project):
    - `lr`: 1e-3 (Adam optimizer)
    - `gamma` (discount factor): 0.98 - 0.99
    - `epsilon_start`: 1.0, `epsilon_end`: 0.05, `epsilon_decay`: 0.995
    - `batch_size`: 64
- Key Methods:
    - `select_action(state, epsilon=0.0, valid_actions=None) -> int`
    - `train_step(replay_buffer, batch_size) -> float (loss)`
    - `update_target_network()`
    - `save(filepath)`
    - `load(filepath)`
"""

from typing import Optional, List
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from backend.dqn_model import DQNNetwork
from backend.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    DQN Agent managing action selection, Bellman updates, and network checkpoints.
    """

    def __init__(
        self,
        state_dim: int = 47,
        action_dim: int = 4,
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # -------------------------------------------------------------------
        # Networks & Optimizer
        # -------------------------------------------------------------------
        self.policy_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for stable learning

    def select_action(
        self,
        state: np.ndarray,
        epsilon: float = 0.0,
        valid_actions: Optional[List[int]] = None,
    ) -> int:
        """
        Selects an action using epsilon-greedy exploration.
        Optional valid_actions list restricts choices to physically valid moves.

        Args:
            state: Length 47 state vector.
            epsilon: Probability of taking a random action.
            valid_actions: Optional list of valid action ints [0, 1, 2, 3].

        Returns:
            int: Selected action in {0, 1, 2, 3}.
        """
        # -------------------------------------------------------------------
        # TODO (Claude): Finalize action selection strategy.
        # Baseline epsilon-greedy implementation provided below.
        # -------------------------------------------------------------------
        legal_actions = valid_actions if valid_actions else list(range(self.action_dim))

        if random.random() < epsilon:
            return random.choice(legal_actions)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_t).squeeze(0).cpu().numpy()

        if valid_actions is not None:
            # Mask invalid actions with a large negative value
            mask = np.full(self.action_dim, -1e9)
            for act in legal_actions:
                mask[act] = q_values[act]
            return int(np.argmax(mask))

        return int(np.argmax(q_values))

    def train_step(self, replay_buffer: ReplayBuffer, batch_size: int = 64) -> Optional[float]:
        """
        Performs a single Bellman Q-learning optimization step.

        Args:
            replay_buffer: Experience buffer containing transitions.
            batch_size: Number of transitions to sample.

        Returns:
            float: Training loss value, or None if buffer has insufficient data.
        """
        # -------------------------------------------------------------------
        # TODO (Claude): Finalize the Bellman loss and gradient update step.
        # Reference template provided below:
        # -------------------------------------------------------------------
        if len(replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = replay_buffer.sample(
            batch_size, device=self.device
        )

        self.policy_net.train()

        # Compute Q(s, a)
        curr_q = self.policy_net(states).gather(1, actions)

        # Double DQN target computation: policy net selects action, target net evaluates value
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        loss = self.loss_fn(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for numerical stability
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Polyak soft update for smooth target network tracking
        self.soft_update_target(tau=0.005)

        return float(loss.item())

    def soft_update_target(self, tau: float = 0.005):
        """Soft update model parameters: theta_target = tau*theta_local + (1 - tau)*theta_target."""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def update_target_network(self):
        """Copies policy network parameters into target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath: str):
        """Saves policy network checkpoint to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.policy_net.state_dict(), filepath)
        print(f"[DQNAgent] Model checkpoint saved to: {filepath}")

    def load(self, filepath: str):
        """Loads policy and target network checkpoint from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        state_dict = torch.load(filepath, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        self.policy_net.eval()
        print(f"[DQNAgent] Model loaded successfully from: {filepath}")

