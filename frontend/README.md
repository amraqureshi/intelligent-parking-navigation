# Intelligent Parking Navigation - Frontend Web Application

An interactive, high-performance visual dashboard built with **React 19**, **TypeScript**, and **Vite** to simulate autonomous parking navigation powered by a PyTorch Double Deep Q-Network (DDQN) agent and NetworkX topological graph.

---

## Features

- **Interactive Canvas & Digital Twin**: Real-time 2D top-down visualization of the 10x5 parking facility, featuring 12 parking bays ($P_1$ to $P_{12}$), driving aisles, directional lanes, and entry/exit gates.
- **Dynamic Occupancy Management**: Interactive one-click spot toggling between available (green) and occupied (red).
- **Preset Scenarios**: Quick presets for congestion levels:
  - `Light` (25% occupied)
  - `Medium` (50% occupied)
  - `Heavy` (75% occupied)
  - `Aisle 1 Full` ($P_1-P_6$)
  - `Aisle 2 Full` ($P_7-P_{12}$)
  - `Free Lot` (0%) & `Random`
- **Real-Time DQN Navigation**:
  - **Autonomous Search**: Executes greedy policy inference through the trained DDQN model, highlighting the optimal path and driving smoothly node-by-node.
  - **Manual Single-Step**: Step through policy transitions one action at a time to inspect state changes and intermediate rewards.
- **Agent Telemetry & Benchmark Metrics**:
  - Live inspection of vehicle coordinates, step counter, cumulative reward, and distance to nearest bay.
  - Deep architectural breakdown of the 47-dimensional observation space and neural network specs.
  - Side-by-side benchmark comparison against the theoretical Dijkstra shortest-path baseline.
- **Activity Feed**: Comprehensive event logging showing every graph traversal, action taken, and reward calculation.

---

## Architecture & Technology Stack

- **Framework**: React 19 (`react`, `react-dom`)
- **Language**: TypeScript with strict typing
- **Build Tool**: Vite 6
- **Icons**: Lucide React (`lucide-react`)
- **Backend API**: FastAPI (`/api/*`) via Vite development proxy (`http://127.0.0.1:8000`)

---

## Getting Started

### 1. Ensure Backend is Running
Before starting the frontend, ensure the FastAPI server is running from the repository root:
```bash
uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Install Frontend Dependencies
```bash
cd frontend
npm install
```

### 3. Start Development Server
```bash
npm run dev
```
Open your browser at [http://localhost:5173](http://localhost:5173).

### 4. Build for Production
```bash
npm run build
```
Generates production assets in `frontend/dist`.
