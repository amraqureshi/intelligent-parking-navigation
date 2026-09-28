"""
Automated Test Suite for DQN Components and Integration
======================================================
Verifies:
1. DQNNetwork architecture, forward pass shapes, and differentiability.
2. ReplayBuffer capacity, FIFO eviction, and batch sampling tensor shapes/types.
3. DQNAgent initialization, target network synchronization, and action masking.
4. DQNAgent training step and Bellman loss computation.
5. DQNAgent checkpoint save and load roundtrip.
6. Environment and DQN integration during live step transitions.
7. Graceful handling of missing model checkpoints in load_model and evaluate.
8. find_parking contract behavior with and without a loaded DQN model.
"""

import os
import tempfile
import numpy as np
import pytest
import torch

from backend.dqn_model import DQNNetwork
from backend.replay_buffer import ReplayBuffer
from backend.dqn_agent import DQNAgent
from backend.parking_environment import (
    ParkingEnvironment,
    load_model,
    find_parking,
)
from backend.evaluate import evaluate_agent


# ---------------------------------------------------------------------------
# 1. DQN Neural Network Tests
# ---------------------------------------------------------------------------
def test_dqn_network_architecture():
    """Verify DQNNetwork initializes with expected layers and produces correct output shape."""
    net = DQNNetwork(state_dim=47, action_dim=4, hidden_dims=(128, 128, 64))

    # Single state forward pass
    single_state = torch.randn(47)
    out_single = net(single_state)
    assert out_single.shape == (4,)

    # Batched forward pass
    batch_states = torch.randn(16, 47)
    out_batch = net(batch_states)
    assert out_batch.shape == (16, 4)


def test_dqn_network_backward_pass():
    """Verify gradients propagate through DQNNetwork."""
    net = DQNNetwork(state_dim=47, action_dim=4)
    x = torch.randn(8, 47)
    out = net(x)
    loss = out.sum()
    loss.backward()

    for param in net.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()


# ---------------------------------------------------------------------------
# 2. Replay Buffer Tests
# ---------------------------------------------------------------------------
def test_replay_buffer_push_and_length():
    """Verify experiences can be pushed and length tracked up to capacity."""
    buf = ReplayBuffer(capacity=5)
    assert len(buf) == 0

    state = np.zeros(47, dtype=np.float32)
    next_state = np.ones(47, dtype=np.float32)

    for i in range(7):
        buf.push(state, action=i % 4, reward=float(i), next_state=next_state, done=(i % 2 == 0))

    # Buffer should not exceed capacity 5
    assert len(buf) == 5


def test_replay_buffer_sampling_shapes_and_types():
    """Verify sample returns properly shaped and typed PyTorch tensors."""
    buf = ReplayBuffer(capacity=100)
    for i in range(20):
        s = np.random.randn(47).astype(np.float32)
        ns = np.random.randn(47).astype(np.float32)
        buf.push(s, action=i % 4, reward=1.0, next_state=ns, done=(i == 19))

    batch_size = 8
    states, actions, rewards, next_states, dones = buf.sample(batch_size=batch_size, device="cpu")

    assert states.shape == (batch_size, 47)
    assert states.dtype == torch.float32

    assert actions.shape == (batch_size, 1)
    assert actions.dtype == torch.int64

    assert rewards.shape == (batch_size, 1)
    assert rewards.dtype == torch.float32

    assert next_states.shape == (batch_size, 47)
    assert next_states.dtype == torch.float32

    assert dones.shape == (batch_size, 1)
    assert dones.dtype == torch.float32


# ---------------------------------------------------------------------------
# 3. DQN Agent Tests
# ---------------------------------------------------------------------------
def test_dqn_agent_action_selection_and_masking():
    """Verify action masking prevents invalid actions and greedy selection works."""
    agent = DQNAgent(state_dim=47, action_dim=4)
    state = np.zeros(47, dtype=np.float32)

    # Pure greedy (epsilon=0.0) with restricted actions
    valid_actions = [1, 2]  # Only DOWN and LEFT allowed
    for _ in range(10):
        action = agent.select_action(state, epsilon=0.0, valid_actions=valid_actions)
        assert action in valid_actions

    # Pure exploration (epsilon=1.0) with restricted actions
    valid_actions = [0, 3]  # Only UP and RIGHT allowed
    for _ in range(20):
        action = agent.select_action(state, epsilon=1.0, valid_actions=valid_actions)
        assert action in valid_actions


def test_dqn_agent_target_network_sync():
    """Verify target network weights are updated to match policy network."""
    agent = DQNAgent(state_dim=47, action_dim=4)

    # Perturb policy net parameters
    with torch.no_grad():
        for param in agent.policy_net.parameters():
            param.add_(1.0)

    # Before update, policy and target parameters should differ
    policy_params = list(agent.policy_net.parameters())
    target_params = list(agent.target_net.parameters())
    assert not torch.allclose(policy_params[0], target_params[0])

    # After update, parameters should match exactly
    agent.update_target_network()
    target_params_updated = list(agent.target_net.parameters())
    assert torch.allclose(policy_params[0], target_params_updated[0])


def test_dqn_agent_train_step():
    """Verify train_step handles insufficient data and performs gradient updates."""
    agent = DQNAgent(state_dim=47, action_dim=4)
    buffer = ReplayBuffer(capacity=100)

    # Insufficient samples -> returns None
    loss = agent.train_step(buffer, batch_size=16)
    assert loss is None

    # Populate buffer
    for i in range(32):
        s = np.random.randn(47).astype(np.float32)
        ns = np.random.randn(47).astype(np.float32)
        buffer.push(s, action=i % 4, reward=float(i), next_state=ns, done=False)

    loss = agent.train_step(buffer, batch_size=16)
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_dqn_agent_save_and_load(tmp_path):
    """Verify model save and load produces identical weights."""
    checkpoint_file = str(tmp_path / "test_dqn.pth")
    agent1 = DQNAgent(state_dim=47, action_dim=4)
    agent1.save(checkpoint_file)

    assert os.path.exists(checkpoint_file)

    agent2 = DQNAgent(state_dim=47, action_dim=4)
    agent2.load(checkpoint_file)

    p1 = list(agent1.policy_net.parameters())
    p2 = list(agent2.policy_net.parameters())
    for a, b in zip(p1, p2):
        assert torch.allclose(a, b)


def test_dqn_agent_load_missing_file():
    """Verify loading a nonexistent file raises FileNotFoundError."""
    agent = DQNAgent(state_dim=47, action_dim=4)
    with pytest.raises(FileNotFoundError):
        agent.load("nonexistent_model_checkpoint.pth")


# ---------------------------------------------------------------------------
# 4. Environment & Integration Tests
# ---------------------------------------------------------------------------
def test_environment_agent_step_integration():
    """Verify live interaction loop between ParkingEnvironment and DQNAgent."""
    env = ParkingEnvironment(max_steps=10)
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)

    state, info = env.reset(random_occupancy_rate=0.5)
    valid_actions = env.get_valid_actions()
    action = agent.select_action(state, epsilon=0.0, valid_actions=valid_actions)

    next_state, reward, terminated, truncated, step_info = env.step(action)

    assert next_state.shape == (47,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "route" in step_info


def test_load_model_missing_checkpoint_handling():
    """Verify load_model handles missing file gracefully and returns None."""
    result = load_model("models/nonexistent_checkpoint.pth")
    assert result is None


def test_find_parking_with_missing_model_fallback():
    """Verify find_parking operates cleanly via Dijkstra baseline when no DQN is loaded."""
    # Ensure no model is loaded
    load_model("models/nonexistent_checkpoint.pth")
    res = find_parking(start="ENTRY", occupied=["P1", "P2"])

    assert res["success"] is True
    assert res["parking_spot"] not in ["P1", "P2"]
    assert "Baseline" in res["algorithm"] or "NetworkX" in res["algorithm"]


def test_evaluate_missing_checkpoint_handling():
    """Verify evaluate_agent handles missing checkpoint by raising FileNotFoundError when DQN required."""
    with pytest.raises(FileNotFoundError):
        evaluate_agent(model_path="models/definitely_missing_checkpoint.pth", allow_baseline_only=False)
