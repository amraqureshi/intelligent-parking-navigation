# Intelligent Parking Navigation Using Deep Q-Network (DQN)

A top-down intelligent parking simulation environment and PyTorch-compatible Deep Q-Network (DQN) backend foundation for autonomous parking navigation in a constrained graph layout.

---

## 1. Project Overview & Architecture

This repository implements:
1. **NetworkX Parking Graph**: Deterministic topological layout featuring driving aisles, intersections, entry/exit gates, and 12 distinct parking spots ($P_1$ to $P_{12}$).
2. **Gym-Style Simulation Environment (`ParkingEnvironment`)**:
   - Discrete 4-directional action space (UP, DOWN, LEFT, RIGHT).
   - Fixed-size 47-dimensional normalized state vector suitable for PyTorch neural networks.
   - Comprehensive reward shaping with collision avoidance, step penalties, and terminal parking rewards.
   - Dynamic spot occupancy and action masking.
3. **Integration Contract**: Exposed endpoints (`find_parking`, `get_parking_layout`, `get_parking_status`, `load_model`) ready for Streamlit frontend visualizers and DQN inference.
4. **Integration Placeholders for Claude**: Clean, typed interfaces for `dqn_model.py`, `replay_buffer.py`, `dqn_agent.py`, `train_model.py`, and `evaluate.py`.

---

## 2. Directory Structure

```
parkinglot/
├── backend/
│   ├── __init__.py               # Exports environment and contract functions
│   ├── parking_environment.py    # Gym-style simulation & NetworkX graph layout
│   ├── dqn_model.py              # PyTorch DQN architecture (Claude integration)
│   ├── replay_buffer.py          # Experience replay buffer (Claude integration)
│   ├── dqn_agent.py              # DQNAgent action & update logic (Claude integration)
│   ├── train_model.py            # Training pipeline & reward curves (Claude integration)
│   ├── evaluate.py               # Policy evaluation & benchmark (Claude integration)
│   ├── requirements.txt          # Backend dependencies
│   └── README.md                 # Backend documentation
├── models/                       # Checkpoints directory (dqn_parking.pth)
├── results/                      # Evaluation logs and plots (training_curve.png)
├── test_environment.py           # Automated test suite verifying the environment
├── main.py                       # Integration demo showcasing API contract
├── requirements.txt              # Root dependencies
└── README.md                     # Main documentation
```

---

## 3. Parking Facility Layout

The facility is modeled as a directed graph (`nx.DiGraph`) on a discrete coordinate grid:

```
 y
 5 [J4]===================[J5]===================[J6]   <-- Upper Thoroughfare
     |                      |                      |
 4 [P5]-N3-[P6]         [P11]-N6-[P12]             N9
     |                      |                      |
 3 [P3]-N2-[P4]          [P9]-N5-[P10]             N8
     |                      |                      |
 2 [P1]-N1-[P2]          [P7]-N4-[P8]              N7
     |                      |                      |
 1 ENTRY ===== [J1] ===== [J2] ===== [J3] ===== EXIT   <-- Lower Thoroughfare
   (0,1)       (2,1)     (5,1)     (8,1)       (10,1)
   ----------------------------------------------------> x
```

### Node Identifiers (String IDs)
- **Entrance & Exit**: `"ENTRY"` at $(0, 1)$, `"EXIT"` at $(10, 1)$
- **Intersections / Junctions**:
  - Lower thoroughfare: `"J1"` $(2, 1)$, `"J2"` $(5, 1)$, `"J3"` $(8, 1)$
  - Upper thoroughfare: `"J4"` $(2, 5)$, `"J5"` $(5, 5)$, `"J6"` $(8, 5)$
- **Driving Aisle Nodes**:
  - Aisle 1: `"N1"` $(2, 2)$, `"N2"` $(2, 3)$, `"N3"` $(2, 4)$
  - Aisle 2: `"N4"` $(5, 2)$, `"N5"` $(5, 3)$, `"N6"` $(5, 4)$
  - Aisle 3 (Bypass): `"N7"` $(8, 2)$, `"N8"` $(8, 3)$, `"N9"` $(8, 4)$
- **Parking Spot IDs**: 12 spots (`"P1"` to `"P12"`)
  - Aisle 1 spots: `P1`, `P2`, `P3`, `P4`, `P5`, `P6`
  - Aisle 2 spots: `P7`, `P8`, `P9`, `P10`, `P11`, `P12`

---

## 4. Action Space

Discrete action space of size 4:

| Action ID | Name | Coordinate Delta | Graph Mapping |
| :---: | :---: | :---: | :--- |
| `0` | **UP** | $\Delta y = +1$ | Traverses outgoing edge pointing North |
| `1` | **DOWN** | $\Delta y = -1$ | Traverses outgoing edge pointing South |
| `2` | **LEFT** | $\Delta x = -1$ | Traverses outgoing edge pointing West (e.g. into left spots) |
| `3` | **RIGHT** | $\Delta x = +1$ | Traverses outgoing edge pointing East (e.g. into right spots) |

- **Invalid Moves**: If no edge exists in the chosen direction, the agent remains at the current node and receives a $-2.0$ penalty.
- **Occupied Spot Handling**: If the target node is an occupied spot, movement is blocked, vehicle remains at the current node, and receives a $-5.0$ collision penalty.
- **Action Masking**: Call `env.get_valid_actions()` or `env.get_valid_action_mask()` to prevent invalid moves during inference.

---

## 5. State Representation (Fixed-Size: 47 Floats)

Each observation is a fixed-size 1D NumPy array (`dtype=np.float32`, shape: `(47,)`):

| Slice | Features | Description | Value Range |
| :--- | :---: | :--- | :---: |
| `[0:29]` | 29 | **One-Hot Current Node**: Indicator for the current node across all 29 graph nodes in sorted alphabetical order. | $\{0.0, 1.0\}$ |
| `[29:31]` | 2 | **Normalized Coordinates**: `(x / 10.0, y / 5.0)` of current node. | $[0.0, 1.0]$ |
| `[31:43]` | 12 | **Spot Occupancy Vector**: Binary flag for each spot `P1` through `P12` (`1.0` if occupied, `0.0` if free). | $\{0.0, 1.0\}$ |
| `[43:45]` | 2 | **Relative Vector to Nearest Available Spot**: `(dx / 10.0, dy / 5.0)` from vehicle to closest reachable available spot. | $[-1.0, 1.0]$ |
| `[45]` | 1 | **Normalized Graph Distance**: Shortest path distance to nearest available spot normalized by `15.0` (`1.0` if unreachable). | $[0.0, 1.0]$ |
| `[46]` | 1 | **Step Budget Elapsed**: `current_step / max_steps`. | $[0.0, 1.0]$ |

---

## 6. Reward Scheme

- **Terminal Success** (reaches an available parking spot): `+50.0`
- **Step Penalty** (encourages minimum-step navigation): `-0.2`
- **Distance-Progress Shaping** (potential shaping to closest spot): $+0.5 \times (d_{\text{prev}} - d_{\text{curr}})$
- **Invalid Move Penalty** (wall/no edge): `-2.0`
- **Occupied Spot Collision Penalty**: `-5.0`
- **Exit without Parking**: `-10.0`
- **Step Limit Timeout** ($> 50$ steps): `-10.0`

---

## 7. Exposed Integration Contract

The backend exposes these core functions in `backend/__init__.py`:

### `get_parking_layout()`
Returns full static graph topology with coordinates and types for Streamlit rendering.

### `get_parking_status(occupied=None)`
Returns current counts, occupancy percentage, and a dictionary status map for all spots:
```python
{
    "total_spots": 12,
    "occupied_count": 6,
    "available_count": 6,
    "occupancy_rate": 0.5,
    "occupied_spots": ["P1", "P2", "P4", "P6", "P7", "P10"],
    "available_spots": ["P3", "P5", "P8", "P9", "P11", "P12"],
    "status_map": {"P1": "occupied", "P3": "available", ...}
}
```

### `find_parking(start="ENTRY", occupied=None)`
Finds parking and returns the exact dictionary format:
```json
{
    "success": true,
    "parking_spot": "P3",
    "route": ["ENTRY", "J1", "N1", "N2", "P3"],
    "steps": 4,
    "total_reward": 53.6,
    "message": "Parking spot found"
}
```

### `load_model(model_path=None)`
Loads the trained PyTorch DQN weights from `models/dqn_parking.pth` for live inference.

---

## 8. Exact Interface Claude Must Implement

Claude is responsible for completing the DQN implementation. The environment and placeholder files are structured so Claude only needs to implement standard RL components:

1. **`backend/dqn_model.py`**:
   - `DQNNetwork(nn.Module)`:
     - `__init__(self, state_dim=47, action_dim=4)`
     - `forward(self, x: torch.Tensor) -> torch.Tensor` returning Q-values of shape `(batch_size, 4)`.

2. **`backend/replay_buffer.py`**:
   - `ReplayBuffer(capacity=20000)`:
     - `push(state, action, reward, next_state, done)`
     - `sample(batch_size, device) -> (states, actions, rewards, next_states, dones)`
     - `__len__() -> int`

3. **`backend/dqn_agent.py`**:
   - `DQNAgent(state_dim=47, action_dim=4, lr=1e-3, gamma=0.99)`:
     - `select_action(state, epsilon, valid_actions) -> int`
     - `train_step(replay_buffer, batch_size=64) -> float (loss)`
     - `update_target_network()`
     - `save(filepath: str)`
     - `load(filepath: str)`

4. **`backend/train_model.py`**:
   - Runs training loop over ~400 episodes.
   - Saves final weights to `models/dqn_parking.pth`.
   - Saves reward and success curves to `results/training_curve.png`.

5. **`backend/evaluate.py`**:
   - Runs evaluation episodes and outputs success rate, average steps, and collision stats across 25%, 50%, and 75% congestion levels.

---

## 9. Installation & Verification

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Environment Test Suite
```bash
python test_environment.py
```
*(Runs comprehensive tests for resets, action transitions, occupied spot avoidance, terminal success/failure, and fixed state shape).*

### Run Integration Contract Demo
```bash
python main.py
```
*(Demonstrates `get_parking_layout()`, `get_parking_status()`, and `find_parking()` with route generation).*
