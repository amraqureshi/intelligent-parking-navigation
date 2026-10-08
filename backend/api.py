"""
Intelligent Parking Navigation - FastAPI Backend Server
======================================================
Module: backend.api

Provides RESTful endpoints connecting the interactive frontend simulation
to the underlying NetworkX environment and trained PyTorch DQN agent.

Run Command:
    uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.parking_environment import (
    ParkingEnvironment,
    create_parking_graph,
    get_parking_layout,
    get_parking_status,
    find_parking,
    load_model,
    ACTION_NAMES,
)
from backend.dqn_agent import DQNAgent

# ---------------------------------------------------------------------------
# FastAPI App & CORS Setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ParkNav AI Backend",
    description="REST API for Deep Q-Network Intelligent Parking Navigation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Simulation State Manager
# ---------------------------------------------------------------------------
class SimulationSession:
    def __init__(self, checkpoint_path: str = "models/dqn_parking.pth"):
        self.checkpoint_path = checkpoint_path
        self.env = ParkingEnvironment(max_steps=50)
        self.agent: Optional[DQNAgent] = None
        self.model_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Simulation session dynamic variables
        self.is_searching = False
        self.is_parked = False
        self.parked_spot: Optional[str] = None
        self.target_spot: Optional[str] = None
        self.planned_route: List[str] = []
        self.last_action: Optional[str] = None
        self.last_message: str = "Simulation initialized."

        # Load trained DQN weights if present
        self.init_agent()
        # Initial reset
        self.reset(start_node="ENTRY", random_occupancy_rate=0.5)

    def init_agent(self):
        """Attempts to load the trained DQN policy."""
        if os.path.exists(self.checkpoint_path):
            try:
                self.agent = DQNAgent(
                    state_dim=self.env.state_dim,
                    action_dim=self.env.action_dim,
                    device=self.device,
                )
                self.agent.load(self.checkpoint_path)
                load_model(self.checkpoint_path)
                self.model_loaded = True
                print(f"[SimulationSession] DQN policy loaded from '{self.checkpoint_path}'.")
            except Exception as e:
                print(f"[SimulationSession] Warning loading model: {e}")
                self.agent = None
                self.model_loaded = False
        else:
            print(f"[SimulationSession] Checkpoint '{self.checkpoint_path}' not found.")
            self.agent = None
            self.model_loaded = False

    def reset(
        self,
        start_node: str = "ENTRY",
        occupied_spots: Optional[List[str]] = None,
        random_occupancy_rate: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        """Resets the simulation environment."""
        self.env.reset(
            start_node=start_node,
            occupied_spots=occupied_spots,
            random_occupancy_rate=random_occupancy_rate,
            seed=seed,
        )
        self.is_searching = False
        self.is_parked = False
        self.parked_spot = None
        self.target_spot = None
        self.planned_route = []
        self.last_action = None
        self.last_message = f"Environment reset at node '{start_node}'."

    def get_serialized_state(self) -> Dict[str, Any]:
        """Returns comprehensive snapshot of the simulation state."""
        status = get_parking_status(list(self.env.occupied_spots))
        curr_pos = self.env.graph.nodes[self.env.current_node]["pos"]

        valid_acts = self.env.get_valid_actions()
        valid_actions_list = [
            {"action": a, "name": ACTION_NAMES[a]} for a in valid_acts
        ]

        nearest_spot, shortest_dist = self.env._find_nearest_available_spot(self.env.current_node)

        return {
            "current_node": self.env.current_node,
            "current_pos": {"x": curr_pos[0], "y": curr_pos[1]},
            "current_step": self.env.current_step,
            "max_steps": self.env.max_steps,
            "total_reward": round(self.env.total_reward, 2),
            "occupied_spots": sorted(list(self.env.occupied_spots)),
            "available_spots": sorted(self.env.get_available_spots()),
            "total_spots": status["total_spots"],
            "occupied_count": status["occupied_count"],
            "available_count": status["available_count"],
            "occupancy_rate": status["occupancy_rate"],
            "status_map": status["status_map"],
            "route_history": list(self.env.visited_nodes),
            "planned_route": self.planned_route,
            "is_searching": self.is_searching,
            "is_parked": self.is_parked,
            "parked_spot": self.parked_spot,
            "target_spot": self.target_spot,
            "nearest_available_spot": nearest_spot,
            "nearest_spot_distance": shortest_dist if shortest_dist != float("inf") else None,
            "valid_actions": valid_actions_list,
            "last_action": self.last_action,
            "last_message": self.last_message,
            "model_loaded": self.model_loaded,
        }


# Singleton session instance
session = SimulationSession()


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------
class ResetRequest(BaseModel):
    start_node: str = Field(default="ENTRY", description="Initial node for vehicle")
    occupied_spots: Optional[List[str]] = Field(default=None, description="Explicit occupied spot IDs")
    random_occupancy_rate: Optional[float] = Field(default=None, description="Occupancy ratio (0.0 to 1.0)")
    seed: Optional[int] = Field(default=None, description="RNG seed")


class ToggleSpotRequest(BaseModel):
    spot_id: str = Field(..., description="Target parking spot (P1 to P12)")
    occupied: Optional[bool] = Field(default=None, description="Target occupancy status; toggles if None")


class ScenarioRequest(BaseModel):
    preset: str = Field(
        ...,
        description="Scenario preset: 'low' (25%), 'medium' (50%), 'high' (75%), 'aisle1_full', 'aisle2_full', 'empty', or 'random'",
    )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health_check():
    """Health status and model checkpoint diagnostics."""
    return {
        "status": "online",
        "service": "intelligent-parking-navigation",
        "model_loaded": session.model_loaded,
        "checkpoint_exists": os.path.exists(session.checkpoint_path),
        "checkpoint_path": session.checkpoint_path,
        "device": session.device,
        "num_nodes": session.env.num_nodes,
        "num_spots": session.env.num_parking_spots,
    }


@app.get("/api/layout")
def get_layout():
    """Returns static topological layout with nodes, edges, spots, and bounds."""
    layout_data = get_parking_layout()
    G = session.env.graph

    # Enrich nodes with pixel coordinate normalization hints
    nodes_list = []
    for node_id, d in G.nodes(data=True):
        nodes_list.append({
            "id": node_id,
            "x": d["pos"][0],
            "y": d["pos"][1],
            "type": d.get("type", "lane"),
            "label": d.get("label", node_id),
        })

    # Enrich edges
    edges_list = []
    for u, v, d in G.edges(data=True):
        edges_list.append({
            "source": u,
            "target": v,
            "action": d.get("action"),
            "action_name": d.get("action_name", ACTION_NAMES.get(d.get("action"), "")),
            "weight": d.get("weight", 1.0),
        })

    return {
        "nodes": nodes_list,
        "edges": edges_list,
        "parking_spots": layout_data["parking_spots"],
        "grid_bounds": layout_data["grid_bounds"],
    }


@app.get("/api/state")
def get_state():
    """Returns the current simulation state."""
    return session.get_serialized_state()


@app.post("/api/reset")
def reset_simulation(req: ResetRequest = ResetRequest()):
    """Resets the simulation to the specified parameters."""
    if req.start_node not in session.env.graph.nodes:
        raise HTTPException(status_code=400, detail=f"Invalid start node: '{req.start_node}'")

    session.reset(
        start_node=req.start_node,
        occupied_spots=req.occupied_spots,
        random_occupancy_rate=req.random_occupancy_rate,
        seed=req.seed,
    )
    return session.get_serialized_state()


@app.post("/api/toggle-spot")
def toggle_spot(req: ToggleSpotRequest):
    """Interactively toggles a parking spot between free and occupied."""
    spot = req.spot_id.upper()
    if spot not in session.env.parking_spots:
        raise HTTPException(status_code=400, detail=f"Invalid parking spot: '{spot}'")

    current_occupied = set(session.env.occupied_spots)
    if req.occupied is not None:
        if req.occupied:
            current_occupied.add(spot)
        else:
            current_occupied.discard(spot)
    else:
        if spot in current_occupied:
            current_occupied.remove(spot)
        else:
            current_occupied.add(spot)

    session.env.set_occupied_spots(list(current_occupied))
    session.last_message = f"Spot {spot} toggled to {'occupied' if spot in current_occupied else 'available'}."
    return session.get_serialized_state()


@app.post("/api/scenario")
def set_scenario(req: ScenarioRequest):
    """Configures supported parking scenario presets."""
    preset = req.preset.lower().strip()
    all_spots = sorted(session.env.parking_spots)

    if preset in ["low", "25", "25%", "25% occupied (light)", "light"]:
        # 3 spots occupied (P1, P7, P10)
        chosen = ["P1", "P7", "P10"]
    elif preset in ["medium", "50", "50%", "50% occupied (medium)"]:
        # 6 spots occupied (P1, P2, P4, P7, P8, P11)
        chosen = ["P1", "P2", "P4", "P7", "P8", "P11"]
    elif preset in ["high", "75", "75%", "75% occupied (heavy)", "heavy"]:
        # 9 spots occupied (P1, P2, P3, P4, P5, P7, P8, P9, P11) -> Leaves P6, P10, P12
        chosen = ["P1", "P2", "P3", "P4", "P5", "P7", "P8", "P9", "P11"]
    elif preset in ["aisle1_full", "aisle 1 full", "aisle 1 full (p01-p06)", "aisle1"]:
        # All of Aisle 1 (P1-P6) occupied -> agent must navigate to Aisle 2
        chosen = ["P1", "P2", "P3", "P4", "P5", "P6"]
    elif preset in ["aisle2_full", "aisle 2 full", "aisle 2 full (p07-p12)", "aisle2"]:
        # All of Aisle 2 (P7-P12) occupied -> agent must navigate to Aisle 1
        chosen = ["P7", "P8", "P9", "P10", "P11", "P12"]
    elif preset in ["empty", "free", "100% free lot", "0%"]:
        chosen = []
    elif preset in ["full", "100%", "100% occupied", "all"]:
        chosen = list(all_spots)
    elif preset in ["random", "random sampling"]:
        k = int(np.random.randint(2, 10))
        chosen = np.random.choice(all_spots, size=k, replace=False).tolist()
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown preset: '{preset}'. Available: low, medium, high, aisle1_full, aisle2_full, empty, random, full",
        )

    session.reset(start_node="ENTRY", occupied_spots=chosen)
    session.last_message = f"Applied scenario preset '{preset}' ({len(chosen)}/12 occupied)."
    return session.get_serialized_state()


@app.post("/api/search")
def start_parking_search():
    """
    Executes a complete intelligent parking search using the trained DQN policy.
    Simulates step-by-step navigation from the vehicle's current position to a free spot.
    """
    if not session.env.get_available_spots():
        session.is_searching = False
        session.planned_route = [session.env.current_node]
        session.target_spot = None
        session.last_message = "No parking spots are currently available!"
        return {
            "success": False,
            "parking_spot": None,
            "route": [session.env.current_node],
            "actions": [],
            "steps": 0,
            "reward": 0.0,
            "total_reward": 0.0,
            "transitions": [],
            "algorithm": "Deep Q-Network (PyTorch)",
            "message": "No parking spots are currently available!",
        }

    # Check if DQN model is loaded
    if not session.model_loaded or session.agent is None:
        raise HTTPException(
            status_code=503,
            detail="Trained DQN model checkpoint ('models/dqn_parking.pth') is not loaded. Train the model using 'python backend/train_model.py' first.",
        )

    # If vehicle was already parked or is at a terminal spot/exit, start cleanly from ENTRY
    start_node = session.env.current_node
    if session.is_parked or start_node.startswith("P") or start_node == "EXIT":
        session.reset(start_node="ENTRY", occupied_spots=list(session.env.occupied_spots))
        start_node = "ENTRY"

    # Clone environment to trace the exact DQN trajectory without modifying current state yet
    sim_env = ParkingEnvironment(max_steps=50)
    sim_env.reset(
        start_node=start_node,
        occupied_spots=list(session.env.occupied_spots),
    )

    state = sim_env.get_state()
    done = False
    step_records = []
    total_reward = 0.0
    parked_spot = None
    route = [sim_env.current_node]

    while not done and sim_env.current_step < sim_env.max_steps:
        valid_actions = sim_env.get_valid_actions()
        if not valid_actions:
            break

        # Greedy DQN action selection with action masking
        action = session.agent.select_action(state, epsilon=0.0, valid_actions=valid_actions)
        prev_node = sim_env.current_node
        next_state, reward, terminated, truncated, step_info = sim_env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = next_state
        route.append(sim_env.current_node)

        step_records.append({
            "step": sim_env.current_step,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "from_node": prev_node,
            "to_node": sim_env.current_node,
            "reward": round(reward, 2),
            "success": step_info.get("success", False),
            "message": step_info.get("message", ""),
        })

        if step_info.get("success"):
            parked_spot = step_info.get("parking_spot")
            break

    success = bool(parked_spot is not None)
    session.is_searching = True
    session.planned_route = route
    session.target_spot = parked_spot
    session.last_message = (
        f"DQN found route to '{parked_spot}' in {len(route)-1} steps."
        if success
        else "DQN could not reach a spot within the step budget."
    )

    return {
        "success": success,
        "parking_spot": parked_spot,
        "route": route,
        "actions": [r["action_name"] for r in step_records],
        "steps": len(route) - 1,
        "reward": round(total_reward, 2),
        "total_reward": round(total_reward, 2),
        "transitions": step_records,
        "algorithm": "Deep Q-Network (PyTorch)",
        "message": session.last_message,
    }


@app.post("/api/step")
def step_simulation():
    """
    Advances the simulation by a single step selected by the DQN agent.
    """
    if session.is_parked:
        return {
            "done": True,
            "terminal": True,
            "finished": True,
            "is_parked": True,
            "message": f"Vehicle is already parked at '{session.parked_spot}'.",
            "state": session.get_serialized_state(),
        }

    if not session.model_loaded or session.agent is None:
        raise HTTPException(
            status_code=503,
            detail="Trained DQN model checkpoint is not loaded.",
        )

    valid_actions = session.env.get_valid_actions()
    if not valid_actions:
        return {
            "done": True,
            "terminal": True,
            "finished": True,
            "is_parked": session.is_parked,
            "message": "No valid actions available from current node.",
            "state": session.get_serialized_state(),
        }

    state = session.env.get_state()
    action = session.agent.select_action(state, epsilon=0.0, valid_actions=valid_actions)
    prev_node = session.env.current_node

    next_state, reward, terminated, truncated, step_info = session.env.step(action)
    done = terminated or truncated

    session.last_action = ACTION_NAMES[action]
    session.last_message = step_info.get("message", "")

    if step_info.get("success"):
        session.is_parked = True
        session.is_searching = False
        session.parked_spot = step_info.get("parking_spot")
        # Mark space as occupied upon parking
        if session.parked_spot:
            curr_occ = set(session.env.occupied_spots)
            curr_occ.add(session.parked_spot)
            session.env.set_occupied_spots(list(curr_occ))

    return {
        "action": action,
        "action_name": ACTION_NAMES[action],
        "selected_action": ACTION_NAMES[action],
        "prev_node": prev_node,
        "current_node": session.env.current_node,
        "reward": round(reward, 2),
        "total_reward": round(session.env.total_reward, 2),
        "cumulative_reward": round(session.env.total_reward, 2),
        "route": list(session.env.visited_nodes),
        "is_parked": session.is_parked,
        "done": done,
        "terminal": done,
        "finished": done,
        "success": step_info.get("success", False),
        "parking_spot": session.parked_spot,
        "message": session.last_message,
        "state": session.get_serialized_state(),
    }


@app.get("/api/metrics")
def get_metrics():
    """Returns evaluation benchmarks and model architecture specifications."""
    benchmark_file = "results/comparison_results.json"
    benchmark_data = None
    if os.path.exists(benchmark_file):
        try:
            with open(benchmark_file, "r", encoding="utf-8") as f:
                benchmark_data = json.load(f)
        except Exception:
            pass

    return {
        "model_architecture": {
            "type": "Double Deep Q-Network (DDQN)",
            "framework": "PyTorch",
            "state_dimension": session.env.state_dim,
            "action_dimension": session.env.action_dim,
            "layers": [
                {"name": "Input", "units": 47},
                {"name": "Hidden Layer 1", "units": 128, "activation": "ReLU"},
                {"name": "Hidden Layer 2", "units": 128, "activation": "ReLU"},
                {"name": "Hidden Layer 3", "units": 64, "activation": "ReLU"},
                {"name": "Output Layer (Q-Values)", "units": 4, "activation": "Linear"},
            ],
            "training_episodes": 400,
            "loss_function": "Smooth L1 (Huber) Loss",
            "target_update": "Polyak Soft Update (tau=0.005)",
        },
        "benchmarks": benchmark_data,
        "facility_specs": {
            "total_nodes": session.env.num_nodes,
            "total_parking_spots": session.env.num_parking_spots,
            "driving_lanes": ["Lower Thoroughfare", "Aisle 1", "Aisle 2", "Aisle 3 (Bypass)", "Upper Thoroughfare"],
            "grid_dimensions": "10 x 5",
        },
    }
