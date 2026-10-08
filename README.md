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
│   ├── dqn_model.py              # PyTorch DQN architecture
│   ├── replay_buffer.py          # Experience replay buffer
│   ├── dqn_agent.py              # DQNAgent policy & target network logic
│   ├── train_model.py            # Training pipeline & reward curves
│   ├── evaluate.py               # Policy evaluation & benchmark
│   ├── api.py                    # FastAPI REST API backend service
│   ├── requirements.txt          # Backend dependencies
│   └── README.md                 # Backend documentation
├── frontend/                     # Interactive React 19 + TypeScript + Vite Web App
│   ├── src/
│   │   ├── api/client.ts         # Typed API client connecting to FastAPI
│   │   ├── components/           # Canvas, controls, telemetry, log, modals
│   │   ├── utils/coordinates.ts  # Canvas grid coordinate mapper
│   │   ├── types/simulation.ts   # TypeScript interfaces & contracts
│   │   ├── App.tsx               # Primary dashboard layout & animation loops
│   │   └── main.tsx              # React DOM entry point
│   ├── package.json
│   └── README.md                 # Frontend documentation
├── models/                       # Checkpoints directory (dqn_parking.pth)
├── results/                      # Evaluation logs and plots (training_curve.png, layout)
├── app.py                        # Interactive Streamlit Python dashboard
├── main.py                       # Integration demo showcasing API contract
├── test_environment.py           # Automated test suite verifying environment (11 tests)
├── test_dqn.py                   # Automated test suite verifying DQN & agent (13 tests)
├── test_api.py                   # Automated test suite verifying FastAPI REST API (12 tests)
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

## 8. Deep Q-Network (DQN) Implementation

The reinforcement learning pipeline is fully implemented across the following components:

1. **`backend/dqn_model.py` (`DQNNetwork`)**:
   - 3-layer Multi-Layer Perceptron (MLP) with dimensions: `47 -> 128 -> 128 -> 64 -> 4`.
   - ReLU non-linear activations.
   - Outputs raw Q-values for the 4 discrete actions (UP, DOWN, LEFT, RIGHT).

2. **`backend/replay_buffer.py` (`ReplayBuffer`)**:
   - High-throughput FIFO circular experience replay buffer (capacity: 25,000 transitions).
   - Uniform mini-batch sampling producing PyTorch Float/Long tensors on CPU/CUDA.

3. **`backend/dqn_agent.py` (`DQNAgent`)**:
   - Double DQN (DDQN) target estimation: decouple action selection (policy network) from value evaluation (target network) to prevent overestimation bias.
   - Smooth L1 (Huber) loss function with gradient clipping (`max_norm=1.0`).
   - Polyak soft target network tracking (`tau=0.005`) for continuous stability.
   - Action masking in `select_action` to prevent invalid physical graph traversals.
   - Checkpoint serialization (`save`/`load`) with `weights_only=True` safe loading.

4. **`backend/train_model.py` (`train_dqn`)**:
   - 400-episode training pipeline with experience replay warm-up (600 steps).
   - Randomized occupancy curriculum ($20\%$ to $80\%$) across episodes.
   - Tracks rolling 20-episode rewards and success rates.
   - Saves optimal checkpoint to `models/dqn_parking.pth` and reward/success curve to `results/training_curve.png`.

5. **`backend/evaluate.py` (`evaluate_agent`)**:
   - Rigorous side-by-side benchmark comparing NetworkX Dijkstra baseline against the trained DQN policy over identical, reproducibly seeded test episodes.
   - Evaluates 25%, 50%, and 75% lot occupancy levels (50 trials each, 150 trials total).
   - Explicit missing checkpoint handling (raises clear exception when `--baseline-only` is not specified).
   - Generates `results/comparison_results.json`, `results/comparison_results.csv`, and `results/performance_vs_occupancy.png`.

---

## 9. Evaluation & Benchmark Results

Benchmarked across 50 seeded trials per congestion level (150 identical comparisons):

| Occupancy Rate | Policy | Success Rate (%) | Avg Steps to Park | Avg Cumulative Reward | Invalid Moves |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **25%** | **Baseline (Dijkstra)** | 100.0% | 3.02 | 50.91 | 0 |
| **25%** | **DQN Agent** | **100.0%** | **3.02** | **50.91** | **0** |
| **50%** | **Baseline (Dijkstra)** | 100.0% | 3.18 | 50.95 | 0 |
| **50%** | **DQN Agent** | **100.0%** | **3.18** | **50.95** | **0** |
| **75%** | **Baseline (Dijkstra)** | 100.0% | 3.80 | 51.14 | 0 |
| **75%** | **DQN Agent** | **92.0%** | **7.54** | **45.42** | **0** |

- At low to moderate congestion (25% and 50%), the DQN agent achieves **100% optimal navigation**, matching the theoretical Dijkstra shortest path step-for-step with zero invalid moves.
- Under high congestion (75%), DQN achieves **92.0% success rate** with 0 invalid moves.

---

## 10. Installation & Usage Instructions

### 1. Install Dependencies
```bash
# Python dependencies
pip install -r requirements.txt

# Frontend dependencies
cd frontend && npm install && cd ..
```

### 2. Run Test Suites
Run the full 36-test automated test suite across all subsystems:
```bash
pytest test_environment.py test_dqn.py test_api.py -v
```
- `test_environment.py` (11 tests): Graph layout, state shape, step transitions, collision prevention, edge cases.
- `test_dqn.py` (13 tests): Network forward/backward pass, replay buffer, DDQN agent updates, action masking, save/load, checkpoint error handling.
- `test_api.py` (12 tests): FastAPI health diagnostics, layout topology, simulation reset, interactive spot toggling, scenario presets, step-by-step stepping, DQN search, and full occupancy handling.

### 3. Launch Interactive Interfaces

#### Option A: Interactive React 19 Web App (Full-Stack)
Start the FastAPI REST backend server:
```bash
uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
```
In a second terminal, launch the Vite development server:
```bash
cd frontend
npm run dev
```
Access the interactive digital twin simulation at `http://localhost:5173`.

#### Option B: Streamlit Python Dashboard
Launch the pure Python dashboard directly without Node.js:
```bash
streamlit run app.py
```
Access the dashboard at `http://localhost:8501`.

### 4. Train DQN Model
Train the agent for 400 episodes and generate training curves:
```bash
python backend/train_model.py
```
Outputs:
- Checkpoint: `models/dqn_parking.pth`
- Visual Curve: `results/training_curve.png`

### 5. Run Policy Evaluation Benchmark
Run side-by-side benchmark of Baseline vs DQN:
```bash
python backend/evaluate.py
```
To evaluate only the NetworkX baseline without a DQN model:
```bash
python backend/evaluate.py --baseline-only
```
Outputs:
- JSON Summary: `results/comparison_results.json`
- CSV Table: `results/comparison_results.csv`
- Comparative Plot: `results/performance_vs_occupancy.png`

### 6. Run Main Integration Demo
Demonstrates the API contract, loading the trained model, and exporting layout visualization:
```bash
python main.py
```
Output:
- Facility Layout Diagram: `results/parking_layout.png`

