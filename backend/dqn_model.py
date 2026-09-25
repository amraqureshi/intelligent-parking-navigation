"""
DQN Neural Network Architecture (Integration Placeholder for Claude)
===================================================================
Module: backend.dqn_model

INSTRUCTIONS FOR CLAUDE:
------------------------
Implement the Deep Q-Network (DQN) PyTorch architecture in this module.

Expected Specifications:
- Framework: PyTorch (torch.nn)
- Input State Dimension: 47 (np.float32)
- Output Action Dimension: 4 (Discrete: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
- Recommended Architecture:
    - Layer 1: nn.Linear(state_dim, 128) -> nn.ReLU()
    - Layer 2: nn.Linear(128, 128) -> nn.ReLU()
    - Layer 3: nn.Linear(128, 64)  -> nn.ReLU()
    - Output:  nn.Linear(64, action_dim)  (Raw Q-values for each action)
    - Optional: Dueling DQN heads (Value stream + Advantage stream) for extra stability.

Interfaces Required by the Project:
- `DQNNetwork(nn.Module)`:
    - `__init__(self, state_dim: int = 47, action_dim: int = 4)`
    - `forward(self, x: torch.Tensor) -> torch.Tensor`
"""

import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for parking lot navigation.

    Maps state vector of size 47 to Q-values for 4 discrete actions:
    [Q(s, UP), Q(s, DOWN), Q(s, LEFT), Q(s, RIGHT)]
    """

    def __init__(self, state_dim: int = 47, action_dim: int = 4, hidden_dims: tuple = (128, 128, 64)):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # -------------------------------------------------------------------
        # TODO (Claude): Finalize network architecture below if needed.
        # Default standard MLP baseline provided for immediate compatibility.
        # -------------------------------------------------------------------
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute Q-values for each action.

        Args:
            x (torch.Tensor): Shape (batch_size, state_dim) or (state_dim,)

        Returns:
            torch.Tensor: Q-values of shape (batch_size, action_dim)
        """
        return self.net(x)
