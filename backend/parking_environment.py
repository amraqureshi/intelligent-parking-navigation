"""
Intelligent Parking Navigation - Parking Simulation Environment
===============================================================
A top-down Gym-style graph navigation environment representing a realistic
parking facility. Compatible with PyTorch DQN agents and Streamlit frontends.

Action Space:
    Discrete(4):
        0 -> UP    (North, dy > 0)
        1 -> DOWN  (South, dy < 0)
        2 -> LEFT  (West,  dx < 0)
        3 -> RIGHT (East,  dx > 0)

State Space:
    Fixed-size 1D NumPy array (shape: (47,), dtype: np.float32)
    Detailed breakdown documented in ParkingEnvironment.get_state()
"""

from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
import networkx as nx

# ---------------------------------------------------------------------------
# Global Constants & Action Space
# ---------------------------------------------------------------------------
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

ACTION_NAMES: Dict[int, str] = {
    ACTION_UP: "UP",
    ACTION_DOWN: "DOWN",
    ACTION_LEFT: "LEFT",
    ACTION_RIGHT: "RIGHT",
}

ACTION_VECTORS: Dict[int, Tuple[int, int]] = {
    ACTION_UP: (0, 1),
    ACTION_DOWN: (0, -1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}

# Reward Scheme
REWARD_SUCCESS = 50.0           # Reached available parking spot
REWARD_STEP = -0.2              # Cost per step to encourage optimal path
REWARD_INVALID_MOVE = -2.0      # Attempted move against walls/no edge
REWARD_OCCUPIED_COLLISION = -5.0 # Attempted move into an occupied spot
REWARD_EXIT_UNPARKED = -10.0    # Left through EXIT without parking
REWARD_TIMEOUT = -10.0          # Max steps reached without finding spot
SHAPING_WEIGHT = 0.5            # Potential-based reward shaping coefficient


# ---------------------------------------------------------------------------
# Layout & Graph Construction
# ---------------------------------------------------------------------------
def create_parking_graph() -> nx.DiGraph:
    """
    Constructs a top-down, realistic parking facility graph using NetworkX.

    Layout Grid:
        - Driving Lanes:
            - Bottom Main Lane (y=1): ENTRY (0,1) -> J1 (2,1) <-> J2 (5,1) <-> J3 (8,1) -> EXIT (10,1)
            - Aisle 1 (x=2): J1 (2,1) <-> N1 (2,2) <-> N2 (2,3) <-> N3 (2,4) <-> J4 (2,5)
            - Aisle 2 (x=5): J2 (5,1) <-> N4 (5,2) <-> N5 (5,3) <-> N6 (5,4) <-> J5 (5,5)
            - Aisle 3 (x=8, Circulation/Bypass): J3 (8,1) <-> N7 (8,2) <-> N8 (8,3) <-> N9 (8,4) <-> J6 (8,5)
            - Top Main Lane (y=5): J4 (2,5) <-> J5 (5,5) <-> J6 (8,5)
        - Parking Spots (12 spots):
            - Off Aisle 1:
                - From N1 (2,2): P1 (1,2) [LEFT], P2 (3,2) [RIGHT]
                - From N2 (2,3): P3 (1,3) [LEFT], P4 (3,3) [RIGHT]
                - From N3 (2,4): P5 (1,4) [LEFT], P6 (3,4) [RIGHT]
            - Off Aisle 2:
                - From N4 (5,2): P7 (4,2) [LEFT], P8 (6,2) [RIGHT]
                - From N5 (5,3): P9 (4,3) [LEFT], P10 (6,3) [RIGHT]
                - From N6 (5,4): P11 (4,4) [LEFT], P12 (6,4) [RIGHT]

    Returns:
        nx.DiGraph: Directed graph with node positions, types, and edge action attributes.
    """
    G = nx.DiGraph()

    # Define Node positions and attributes
    # Driving Nodes: ENTRY, EXIT, Junctions (J1-J6), Lane nodes (N1-N9)
    driving_nodes = {
        # Lower main thoroughfare (Eastbound / Bidirectional)
        "ENTRY": {"pos": (0, 1), "type": "entry", "label": "Entry Gate"},
        "J1": {"pos": (2, 1), "type": "intersection", "label": "Aisle 1 Lower Junction"},
        "J2": {"pos": (5, 1), "type": "intersection", "label": "Aisle 2 Lower Junction"},
        "J3": {"pos": (8, 1), "type": "intersection", "label": "Aisle 3 Lower Junction"},
        "EXIT": {"pos": (10, 1), "type": "exit", "label": "Exit Gate"},

        # Aisle 1 (Bidirectional North/South)
        "N1": {"pos": (2, 2), "type": "lane", "label": "Aisle 1 - Bay 1"},
        "N2": {"pos": (2, 3), "type": "lane", "label": "Aisle 1 - Bay 2"},
        "N3": {"pos": (2, 4), "type": "lane", "label": "Aisle 1 - Bay 3"},
        "J4": {"pos": (2, 5), "type": "intersection", "label": "Aisle 1 Upper Junction"},

        # Aisle 2 (Bidirectional North/South)
        "N4": {"pos": (5, 2), "type": "lane", "label": "Aisle 2 - Bay 1"},
        "N5": {"pos": (5, 3), "type": "lane", "label": "Aisle 2 - Bay 2"},
        "N6": {"pos": (5, 4), "type": "lane", "label": "Aisle 2 - Bay 3"},
        "J5": {"pos": (5, 5), "type": "intersection", "label": "Aisle 2 Upper Junction"},

        # Aisle 3 (Circulation / Bypass)
        "N7": {"pos": (8, 2), "type": "lane", "label": "Aisle 3 - Bay 1"},
        "N8": {"pos": (8, 3), "type": "lane", "label": "Aisle 3 - Bay 2"},
        "N9": {"pos": (8, 4), "type": "lane", "label": "Aisle 3 - Bay 3"},
        "J6": {"pos": (8, 5), "type": "intersection", "label": "Aisle 3 Upper Junction"},
    }

    # Parking Spots (12 spots)
    parking_nodes = {
        "P1": {"pos": (1, 2), "type": "parking", "label": "Parking Spot P1"},
        "P2": {"pos": (3, 2), "type": "parking", "label": "Parking Spot P2"},
        "P3": {"pos": (1, 3), "type": "parking", "label": "Parking Spot P3"},
        "P4": {"pos": (3, 3), "type": "parking", "label": "Parking Spot P4"},
        "P5": {"pos": (1, 4), "type": "parking", "label": "Parking Spot P5"},
        "P6": {"pos": (3, 4), "type": "parking", "label": "Parking Spot P6"},
        "P7": {"pos": (4, 2), "type": "parking", "label": "Parking Spot P7"},
        "P8": {"pos": (6, 2), "type": "parking", "label": "Parking Spot P8"},
        "P9": {"pos": (4, 3), "type": "parking", "label": "Parking Spot P9"},
        "P10": {"pos": (6, 3), "type": "parking", "label": "Parking Spot P10"},
        "P11": {"pos": (4, 4), "type": "parking", "label": "Parking Spot P11"},
        "P12": {"pos": (6, 4), "type": "parking", "label": "Parking Spot P12"},
    }

    for node_id, attrs in {**driving_nodes, **parking_nodes}.items():
        G.add_node(node_id, **attrs)

    def _determine_action(u_pos: Tuple[int, int], v_pos: Tuple[int, int]) -> int:
        dx = v_pos[0] - u_pos[0]
        dy = v_pos[1] - u_pos[1]
        if dx > 0 and dy == 0:
            return ACTION_RIGHT
        elif dx < 0 and dy == 0:
            return ACTION_LEFT
        elif dy > 0 and dx == 0:
            return ACTION_UP
        elif dy < 0 and dx == 0:
            return ACTION_DOWN
        else:
            raise ValueError(f"Diagonal or non-cardinal move from {u_pos} to {v_pos}")

    def add_edge_pair(u: str, v: str, bidirectional: bool = True):
        u_pos = G.nodes[u]["pos"]
        v_pos = G.nodes[v]["pos"]
        act_uv = _determine_action(u_pos, v_pos)
        G.add_edge(u, v, action=act_uv, action_name=ACTION_NAMES[act_uv], weight=1.0)
        if bidirectional:
            act_vu = _determine_action(v_pos, u_pos)
            G.add_edge(v, u, action=act_vu, action_name=ACTION_NAMES[act_vu], weight=1.0)

    # Lower Main Lane: ENTRY -> J1, J1 <-> J2, J2 <-> J3, J3 -> EXIT
    add_edge_pair("ENTRY", "J1", bidirectional=False) # One-way entry
    add_edge_pair("J1", "J2", bidirectional=True)
    add_edge_pair("J2", "J3", bidirectional=True)
    add_edge_pair("J3", "EXIT", bidirectional=False)  # One-way exit

    # Top Main Lane: J4 <-> J5 <-> J6
    add_edge_pair("J4", "J5", bidirectional=True)
    add_edge_pair("J5", "J6", bidirectional=True)

    # Vertical Aisles (Bidirectional driving)
    add_edge_pair("J1", "N1", bidirectional=True)
    add_edge_pair("N1", "N2", bidirectional=True)
    add_edge_pair("N2", "N3", bidirectional=True)
    add_edge_pair("N3", "J4", bidirectional=True)

    add_edge_pair("J2", "N4", bidirectional=True)
    add_edge_pair("N4", "N5", bidirectional=True)
    add_edge_pair("N5", "N6", bidirectional=True)
    add_edge_pair("N6", "J5", bidirectional=True)

    add_edge_pair("J3", "N7", bidirectional=True)
    add_edge_pair("N7", "N8", bidirectional=True)
    add_edge_pair("N8", "N9", bidirectional=True)
    add_edge_pair("N9", "J6", bidirectional=True)

    # Parking Spot Entrances (Pull-in from driving lane)
    # Aisle 1
    add_edge_pair("N1", "P1", bidirectional=False)
    add_edge_pair("N1", "P2", bidirectional=False)
    add_edge_pair("N2", "P3", bidirectional=False)
    add_edge_pair("N2", "P4", bidirectional=False)
    add_edge_pair("N3", "P5", bidirectional=False)
    add_edge_pair("N3", "P6", bidirectional=False)

    # Aisle 2
    add_edge_pair("N4", "P7", bidirectional=False)
    add_edge_pair("N4", "P8", bidirectional=False)
    add_edge_pair("N5", "P9", bidirectional=False)
    add_edge_pair("N5", "P10", bidirectional=False)
    add_edge_pair("N6", "P11", bidirectional=False)
    add_edge_pair("N6", "P12", bidirectional=False)

    return G


# ---------------------------------------------------------------------------
# Parking Environment
# ---------------------------------------------------------------------------
class ParkingEnvironment:
    """
    Gym-style environment for parking spot navigation on a NetworkX graph.

    Action Space:
        0: UP, 1: DOWN, 2: LEFT, 3: RIGHT

    State Representation:
        Length 47 vector (np.float32):
        - [0:29]  One-hot encoding of current node (all 29 nodes in sorted order)
        - [29:31] Normalized (x, y) coordinates of current node in [0, 1]
        - [31:43] Occupancy status of spots P1-P12 (1.0 = occupied, 0.0 = free)
        - [43:45] Normalized relative vector (dx, dy) to nearest available parking spot
        - [45]    Normalized shortest-path distance to nearest available parking spot
        - [46]    Normalized step fraction elapsed (current_step / max_steps)
    """

    def __init__(self, max_steps: int = 50, default_occupied: Optional[List[str]] = None):
        self.graph = create_parking_graph()
        self.max_steps = max_steps

        # Canonical sorted node lists for consistent state vector ordering
        self.all_nodes: List[str] = sorted(list(self.graph.nodes()))
        self.node_to_idx: Dict[str, int] = {node: idx for idx, node in enumerate(self.all_nodes)}
        self.num_nodes: int = len(self.all_nodes) # 29

        self.parking_spots: List[str] = sorted([
            n for n, d in self.graph.nodes(data=True) if d.get("type") == "parking"
        ]) # 12 spots: P1 through P12
        self.num_parking_spots: int = len(self.parking_spots)

        # Coordinate bounds for normalization
        xs = [d["pos"][0] for _, d in self.graph.nodes(data=True)]
        ys = [d["pos"][1] for _, d in self.graph.nodes(data=True)]
        self.min_x, self.max_x = min(xs), max(xs) # 0 to 10
        self.min_y, self.max_y = min(ys), max(ys) # 1 to 5
        self.max_dist: float = 15.0 # Normalizer for graph distance

        # Environment Dimensions
        self.action_dim: int = 4
        self.state_dim: int = self.num_nodes + 2 + self.num_parking_spots + 2 + 1 + 1 # 47

        # Dynamic state variables
        self.current_node: str = "ENTRY"
        self.occupied_spots: Set[str] = set()
        self.current_step: int = 0
        self.total_reward: float = 0.0
        self.visited_nodes: List[str] = []

        if default_occupied is not None:
            self.set_occupied_spots(default_occupied)

    # -----------------------------------------------------------------------
    # Setup & Reset
    # -----------------------------------------------------------------------
    def set_occupied_spots(self, occupied: List[str]):
        """Sets the occupied parking spots."""
        valid_occupied = {s for s in occupied if s in self.parking_spots}
        self.occupied_spots = valid_occupied

    def get_available_spots(self) -> List[str]:
        """Returns list of currently available (unoccupied) parking spots."""
        return [s for s in self.parking_spots if s not in self.occupied_spots]

    def reset(
        self,
        seed: Optional[int] = None,
        start_node: Optional[str] = None,
        occupied_spots: Optional[List[str]] = None,
        random_occupancy_rate: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to an initial state.

        Args:
            seed: Optional random seed for reproducible occupancy generation.
            start_node: Vehicle start node ID (defaults to 'ENTRY').
            occupied_spots: Explicit list of occupied spot IDs (e.g. ['P1', 'P3']).
            random_occupancy_rate: Fraction of spots occupied (0.0 to 1.0) if randomly sampled.

        Returns:
            Tuple of (initial_state_vector, info_dict)
        """
        rng = np.random.default_rng(seed)

        if start_node is not None:
            if start_node not in self.graph.nodes:
                raise ValueError(f"Invalid start node: {start_node}")
            self.current_node = start_node
        else:
            self.current_node = "ENTRY"

        if occupied_spots is not None:
            self.set_occupied_spots(occupied_spots)
        elif random_occupancy_rate is not None:
            k = int(round(self.num_parking_spots * np.clip(random_occupancy_rate, 0.0, 1.0)))
            chosen = rng.choice(self.parking_spots, size=k, replace=False).tolist()
            self.set_occupied_spots(chosen)
        else:
            # Default: ~50% random occupancy if not previously specified
            if not self.occupied_spots:
                chosen = rng.choice(self.parking_spots, size=6, replace=False).tolist()
                self.set_occupied_spots(chosen)

        self.current_step = 0
        self.total_reward = 0.0
        self.visited_nodes = [self.current_node]

        info = {
            "current_node": self.current_node,
            "occupied_spots": list(self.occupied_spots),
            "available_spots": self.get_available_spots(),
            "step": self.current_step,
        }
        return self.get_state(), info

    # -----------------------------------------------------------------------
    # Action Masking & Validation
    # -----------------------------------------------------------------------
    def get_valid_actions(self) -> List[int]:
        """
        Returns a list of integer actions that are physically valid from the current node.
        An action is valid if:
        1. An outgoing graph edge exists in that cardinal direction.
        2. The destination node is NOT an occupied parking spot.
        """
        valid_actions = []
        for _, neighbor, data in self.graph.out_edges(self.current_node, data=True):
            act = data.get("action")
            if act is not None:
                # Disallow entering an occupied parking spot
                if neighbor in self.occupied_spots:
                    continue
                valid_actions.append(act)
        return sorted(list(set(valid_actions)))

    def get_valid_action_mask(self) -> np.ndarray:
        """
        Returns a boolean array of length 4 for PyTorch action-masked DQN policies.
        mask[a] is True if action a is valid, False otherwise.
        """
        mask = np.zeros(self.action_dim, dtype=bool)
        for act in self.get_valid_actions():
            mask[act] = True
        return mask

    # -----------------------------------------------------------------------
    # Step Transition
    # -----------------------------------------------------------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Applies a discrete action to the vehicle.

        Args:
            action: 0 (UP), 1 (DOWN), 2 (LEFT), 3 (RIGHT)

        Returns:
            Tuple of:
                next_state (np.ndarray): Fixed-size shape (47,) state vector
                reward (float): Reward received for this transition
                terminated (bool): True if reached available spot or exit
                truncated (bool): True if max step limit reached
                info (dict): Diagnostic info and route history
        """
        if action not in ACTION_NAMES:
            raise ValueError(f"Invalid action {action}. Expected 0, 1, 2, or 3.")

        self.current_step += 1
        prev_node = self.current_node
        target_node: Optional[str] = None

        # Check for outgoing edge corresponding to action
        for _, neighbor, data in self.graph.out_edges(self.current_node, data=True):
            if data.get("action") == action:
                target_node = neighbor
                break

        terminated = False
        truncated = False
        reward = 0.0
        message = ""
        success = False
        parked_spot: Optional[str] = None

        # Case 1: No edge in that direction (wall / invalid movement)
        if target_node is None:
            reward = REWARD_INVALID_MOVE
            message = f"Invalid move: No traversable edge {ACTION_NAMES[action]} from {self.current_node}."

        # Case 2: Attempting to move into an occupied parking spot
        elif target_node in self.occupied_spots:
            reward = REWARD_OCCUPIED_COLLISION
            message = f"Blocked: Parking spot {target_node} is occupied."

        # Case 3: Valid traversal
        else:
            # Distance progress shaping before updating position
            dist_prev = self._distance_to_nearest_available(prev_node)
            dist_curr = self._distance_to_nearest_available(target_node)
            shaping_reward = (dist_prev - dist_curr) * SHAPING_WEIGHT

            # Apply step cost and move vehicle
            self.current_node = target_node
            self.visited_nodes.append(self.current_node)
            reward = REWARD_STEP + shaping_reward

            # Check if reached an available parking spot (Terminal Success)
            if self.current_node in self.parking_spots and self.current_node not in self.occupied_spots:
                terminated = True
                success = True
                parked_spot = self.current_node
                reward += REWARD_SUCCESS
                message = f"Success: Parked at available spot {self.current_node}!"

            # Check if reached EXIT without parking
            elif self.current_node == "EXIT":
                terminated = True
                success = False
                reward += REWARD_EXIT_UNPARKED
                message = "Terminated: Reached EXIT without parking."

        # Check step truncation
        if not terminated and self.current_step >= self.max_steps:
            truncated = True
            reward += REWARD_TIMEOUT
            message = "Truncated: Maximum step limit reached."

        self.total_reward += reward

        info = {
            "success": success,
            "parking_spot": parked_spot,
            "route": list(self.visited_nodes),
            "steps": self.current_step,
            "current_node": self.current_node,
            "prev_node": prev_node,
            "action_taken": ACTION_NAMES[action],
            "total_reward": round(self.total_reward, 3),
            "message": message,
            "done": terminated or truncated,
        }

        return self.get_state(), reward, terminated, truncated, info

    # -----------------------------------------------------------------------
    # State Vector Representation
    # -----------------------------------------------------------------------
    def get_state(self) -> np.ndarray:
        """
        Constructs the fixed-size 1D NumPy array (shape: (47,), dtype=np.float32).

        Encoding:
            [0:29]  One-Hot Current Node: Indicator for which of the 29 nodes vehicle is on.
            [29:31] Normalized Position: (pos_x / max_x, pos_y / max_y).
            [31:43] Spot Occupancy: Binary flags for P1-P12 (1.0 = occupied, 0.0 = free).
            [43:45] Relative Nearest Spot: (dx / max_x, dy / max_y) to closest available spot.
            [45]    Distance to Nearest Spot: Shortest graph path normalized by max_dist.
            [46]    Step Budget: current_step / max_steps.
        """
        state = np.zeros(self.state_dim, dtype=np.float32)

        # 1. One-hot node encoding (29 features)
        curr_idx = self.node_to_idx[self.current_node]
        state[curr_idx] = 1.0

        offset = self.num_nodes # 29

        # 2. Normalized current coordinates (2 features)
        curr_pos = self.graph.nodes[self.current_node]["pos"]
        state[offset] = curr_pos[0] / max(1.0, float(self.max_x))
        state[offset + 1] = curr_pos[1] / max(1.0, float(self.max_y))
        offset += 2

        # 3. Parking spot occupancy flags for P1..P12 (12 features)
        for i, spot in enumerate(self.parking_spots):
            state[offset + i] = 1.0 if spot in self.occupied_spots else 0.0
        offset += self.num_parking_spots

        # 4. Nearest available parking spot metrics (3 features)
        nearest_spot, shortest_dist = self._find_nearest_available_spot(self.current_node)
        if nearest_spot is not None:
            near_pos = self.graph.nodes[nearest_spot]["pos"]
            dx = (near_pos[0] - curr_pos[0]) / max(1.0, float(self.max_x))
            dy = (near_pos[1] - curr_pos[1]) / max(1.0, float(self.max_y))
            norm_dist = min(1.0, shortest_dist / self.max_dist)
        else:
            dx, dy, norm_dist = 0.0, 0.0, 1.0

        state[offset] = dx
        state[offset + 1] = dy
        state[offset + 2] = norm_dist
        offset += 3

        # 5. Step budget progress (1 feature)
        state[offset] = float(self.current_step) / float(self.max_steps)

        return state

    # -----------------------------------------------------------------------
    # Distance & Graph Helpers
    # -----------------------------------------------------------------------
    def _find_nearest_available_spot(self, from_node: str) -> Tuple[Optional[str], float]:
        """Finds the nearest reachable unoccupied parking spot from a node."""
        avail = self.get_available_spots()
        if not avail:
            return None, float("inf")

        best_spot = None
        min_dist = float("inf")

        for spot in avail:
            if nx.has_path(self.graph, from_node, spot):
                dist = nx.shortest_path_length(self.graph, from_node, spot, weight="weight")
                if dist < min_dist:
                    min_dist = dist
                    best_spot = spot

        return best_spot, min_dist

    def _distance_to_nearest_available(self, node: str) -> float:
        """Returns shortest path distance from node to nearest available spot."""
        _, dist = self._find_nearest_available_spot(node)
        return dist if dist != float("inf") else 20.0

    # -----------------------------------------------------------------------
    # Rendering & Layout Data
    # -----------------------------------------------------------------------
    def render(self, mode: str = "ansi") -> str:
        """Returns string representation of current environment state."""
        avail = self.get_available_spots()
        out = (
            f"[ParkingEnvironment Step {self.current_step}/{self.max_steps}]\n"
            f"  Vehicle Node: {self.current_node} {self.graph.nodes[self.current_node]['pos']}\n"
            f"  Occupied Spots ({len(self.occupied_spots)}): {sorted(list(self.occupied_spots))}\n"
            f"  Available Spots ({len(avail)}): {avail}\n"
            f"  Valid Actions: {[ACTION_NAMES[a] for a in self.get_valid_actions()]}\n"
            f"  Total Reward: {self.total_reward:.2f}\n"
        )
        if mode == "human":
            print(out)
        return out


# ---------------------------------------------------------------------------
# Global Integration Contract Functions
# ---------------------------------------------------------------------------

_LOADED_MODEL = None  # Global reference holder for Claude's loaded PyTorch model


def get_parking_layout() -> Dict[str, Any]:
    """
    Returns complete static top-down parking facility layout data.
    Designed for zero-effort Streamlit / UI rendering.

    Returns:
        dict: Containing nodes list, edges list, parking spots list, and grid bounds.
    """
    G = create_parking_graph()
    nodes_data = []
    for node, attrs in G.nodes(data=True):
        nodes_data.append({
            "id": node,
            "x": attrs["pos"][0],
            "y": attrs["pos"][1],
            "type": attrs["type"],
            "label": attrs.get("label", node),
        })

    edges_data = []
    for u, v, attrs in G.edges(data=True):
        edges_data.append({
            "from": u,
            "to": v,
            "action": attrs.get("action"),
            "action_name": attrs.get("action_name"),
        })

    parking_spots = sorted([n for n, d in G.nodes(data=True) if d.get("type") == "parking"])

    return {
        "nodes": nodes_data,
        "edges": edges_data,
        "parking_spots": parking_spots,
        "entry_node": "ENTRY",
        "exit_node": "EXIT",
        "grid_bounds": {
            "min_x": 0,
            "max_x": 10,
            "min_y": 1,
            "max_y": 5,
        },
    }


def get_parking_status(occupied: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Returns current parking status summary and per-spot availability map.

    Args:
        occupied: Optional list of occupied spot IDs. If None, default example occupied spots are used.

    Returns:
        dict: Detailed spot status breakdown.
    """
    env = ParkingEnvironment()
    if occupied is not None:
        env.set_occupied_spots(occupied)
    else:
        # Standard default demonstration state
        env.set_occupied_spots(["P1", "P2", "P4", "P6", "P7", "P10"])

    avail = env.get_available_spots()
    total = env.num_parking_spots

    status_map = {
        spot: ("occupied" if spot in env.occupied_spots else "available")
        for spot in env.parking_spots
    }

    return {
        "total_spots": total,
        "occupied_count": len(env.occupied_spots),
        "available_count": len(avail),
        "occupancy_rate": round(len(env.occupied_spots) / total, 3),
        "occupied_spots": sorted(list(env.occupied_spots)),
        "available_spots": sorted(avail),
        "status_map": status_map,
    }


def load_model(model_path: Optional[str] = None) -> Any:
    """
    Loads trained PyTorch DQN weights from file into global cache.
    Integration placeholder for Claude's DQNAgent / DQNNetwork.

    Args:
        model_path: Path to .pth / .pt model checkpoint file.

    Returns:
        Loaded model/agent object, or None if weights not yet trained.
    """
    global _LOADED_MODEL
    import os

    default_path = os.path.join(os.path.dirname(__file__), "..", "models", "dqn_parking.pth")
    path_to_use = model_path or default_path

    if not os.path.exists(path_to_use):
        print(f"[load_model] Notice: Model checkpoint not found at '{path_to_use}'.")
        print("[load_model] Place your trained model file at models/dqn_parking.pth.")
        _LOADED_MODEL = None
        return None

    try:
        import torch
        # Attempt loading using Claude's dqn_model if present
        from backend.dqn_model import DQNNetwork
        model = DQNNetwork(state_dim=47, action_dim=4)
        model.load_state_dict(torch.load(path_to_use, map_location="cpu"))
        model.eval()
        _LOADED_MODEL = model
        print(f"[load_model] Successfully loaded DQN model from '{path_to_use}'.")
        return _LOADED_MODEL
    except Exception as e:
        print(f"[load_model] Note: Could not instantiate DQN model ({e}). Claude will finalize.")
        return None


def find_parking(start: str = "ENTRY", occupied: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Executes parking navigation from `start` node given `occupied` spots.

    Returns dictionary matching exact required contract:
    {
        "success": True,
        "parking_spot": "P3",
        "route": ["ENTRY", "N1", "N2", "P3"],
        "steps": 3,
        "total_reward": 10.0,
        "message": "Parking spot found"
    }

    Notes on Execution:
    - If a trained DQN model is loaded via `load_model()`, navigation is driven by the DQN agent.
    - If no DQN model is loaded, the environment evaluates graph reachability and runs a NetworkX
      shortest path baseline to provide route and reward diagnostics, with clear algorithm attribution.
    """
    global _LOADED_MODEL
    env = ParkingEnvironment()
    state, info = env.reset(start_node=start, occupied_spots=occupied)

    available_spots = env.get_available_spots()
    if not available_spots:
        return {
            "success": False,
            "parking_spot": None,
            "route": [start],
            "steps": 0,
            "total_reward": 0.0,
            "message": "No parking spots are currently available.",
            "algorithm": "Reachability Check",
        }

    # Verify if any available spot is actually reachable from start
    best_target, min_dist = env._find_nearest_available_spot(start)
    if best_target is None or min_dist == float("inf"):
        return {
            "success": False,
            "parking_spot": None,
            "route": [start],
            "steps": 0,
            "total_reward": 0.0,
            "message": f"No available parking spot is reachable from '{start}'.",
            "algorithm": "Reachability Check",
        }

    # -----------------------------------------------------------------------
    # DQN INFERENCE (Integrated when Claude implements DQNAgent)
    # -----------------------------------------------------------------------
    if _LOADED_MODEL is not None:
        import torch
        route = [env.current_node]
        done = False
        step_count = 0
        total_reward = 0.0
        parked_spot = None

        with torch.no_grad():
            while not done and step_count < env.max_steps:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = _LOADED_MODEL(state_tensor).squeeze(0).numpy()

                # Action masking for valid legal moves
                valid_mask = env.get_valid_action_mask()
                q_values[~valid_mask] = -1e9
                action = int(np.argmax(q_values))

                next_state, reward, terminated, truncated, step_info = env.step(action)
                total_reward += reward
                step_count += 1
                state = next_state
                route.append(env.current_node)
                done = terminated or truncated

                if step_info.get("success"):
                    parked_spot = step_info.get("parking_spot")

        return {
            "success": bool(parked_spot is not None),
            "parking_spot": parked_spot,
            "route": route,
            "steps": step_count,
            "total_reward": round(total_reward, 2),
            "message": "Parking spot found" if parked_spot else "DQN navigation failed to reach spot within limit.",
            "algorithm": "Deep Q-Network (PyTorch)",
        }

    # -----------------------------------------------------------------------
    # BASELINE REACHABILITY & COMPARISON (NetworkX Dijkstra Baseline)
    # Clearly documented fallback so backend is runnable before Claude trains DQN.
    # -----------------------------------------------------------------------
    # Compute shortest path from start to nearest reachable available spot
    path = nx.shortest_path(env.graph, source=start, target=best_target, weight="weight")

    # Step through environment using actions dictated by the path edges
    sim_env = ParkingEnvironment()
    sim_env.reset(start_node=start, occupied_spots=occupied)

    route = [start]
    accumulated_reward = 0.0
    final_success = False

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        action = env.graph[u][v]["action"]
        _, rew, term, trunc, step_info = sim_env.step(action)
        accumulated_reward += rew
        route.append(v)
        if step_info.get("success"):
            final_success = True
            break
        if term or trunc:
            break

    return {
        "success": final_success,
        "parking_spot": best_target if final_success else None,
        "route": route,
        "steps": len(route) - 1,
        "total_reward": round(accumulated_reward, 2),
        "message": f"Parking spot found: {best_target}" if final_success else "Baseline navigation could not complete.",
        "algorithm": "NetworkX Shortest Path Baseline (DQN Placeholder Ready)",
    }
