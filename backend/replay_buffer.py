"""
Experience Replay Buffer (Integration Placeholder for Claude)
=============================================================
Module: backend.replay_buffer

INSTRUCTIONS FOR CLAUDE:
------------------------
Implement the standard or prioritized experience replay buffer in this module.

Expected Specifications:
- Store transitions: (state, action, reward, next_state, done)
    - state: np.ndarray or torch.Tensor of shape (47,)
    - action: int (0 to 3)
    - reward: float
    - next_state: np.ndarray or torch.Tensor of shape (47,)
    - done: bool (terminal or truncated)
- Sampling:
    - Return mini-batches converted to PyTorch tensors on the appropriate device (CPU/CUDA).
- Capacity:
    - Typically 10,000 to 50,000 transitions for this 29-node graph environment.
"""

from collections import deque
import random
from typing import Tuple, List, Any
import numpy as np
import torch


class ReplayBuffer:
    """
    Standard FIFO Replay Buffer for Deep Q-Learning transitions.
    """

    def __init__(self, capacity: int = 20000):
        """
        Args:
            capacity: Maximum number of experiences to retain.
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Appends a transition tuple to the buffer.
        """
        self.buffer.append((
            np.array(state, dtype=np.float32, copy=False),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32, copy=False),
            bool(done),
        ))

    def sample(
        self,
        batch_size: int,
        device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples a random mini-batch of experiences and converts them to PyTorch tensors.

        Args:
            batch_size: Number of transitions to sample.
            device: Target torch device ('cpu' or 'cuda').

        Returns:
            Tuple of (states, actions, rewards, next_states, dones):
                - states: FloatTensor (batch_size, 47)
                - actions: LongTensor (batch_size, 1)
                - rewards: FloatTensor (batch_size, 1)
                - next_states: FloatTensor (batch_size, 47)
                - dones: FloatTensor (batch_size, 1)
        """
        # -------------------------------------------------------------------
        # TODO (Claude): Finalize or customize sampling logic below if needed.
        # -------------------------------------------------------------------
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(device)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self) -> int:
        """Returns the current number of experiences stored."""
        return len(self.buffer)
