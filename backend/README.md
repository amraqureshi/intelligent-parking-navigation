# Backend Documentation

Please refer to the root [README.md](../README.md) for full project documentation, architectural diagrams, state space definitions, action mappings, and DQN specifications.

### Key Backend Modules
- `parking_environment.py`: NetworkX graph and Gym-compatible parking simulation environment
- `dqn_model.py`: DQN PyTorch neural network architecture
- `replay_buffer.py`: High-throughput experience replay buffer
- `dqn_agent.py`: DQNAgent policy & target network logic with action masking
- `train_model.py`: 400-episode DQN training pipeline
- `evaluate.py`: Policy evaluation benchmark comparing DQN against Dijkstra baseline
- `api.py`: FastAPI REST API server connecting the environment & DQN policy to interactive frontends
