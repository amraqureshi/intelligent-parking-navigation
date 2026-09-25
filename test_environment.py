"""
Automated Test Suite for Parking Simulation Environment
=======================================================
Verifies:
1. Environment reset and initial configuration
2. Step transitions and directional movement
3. Invalid-move prevention (walls / boundaries)
4. Occupied-spot prevention (cannot enter or park in occupied spots)
5. Terminal success on entering an available parking spot
6. Terminal exit without parking
7. State vector shape consistency and normalization
8. Integration contract functions (get_parking_layout, get_parking_status, find_parking)
"""

import numpy as np
import networkx as nx
import pytest

from backend.parking_environment import (
    ParkingEnvironment,
    create_parking_graph,
    get_parking_layout,
    get_parking_status,
    find_parking,
    load_model,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    REWARD_SUCCESS,
    REWARD_INVALID_MOVE,
    REWARD_OCCUPIED_COLLISION,
)


def test_reset():
    """Verify reset properly sets start node, step count, and returns valid state."""
    env = ParkingEnvironment()
    state, info = env.reset(start_node="ENTRY", occupied_spots=["P1", "P2"])

    assert env.current_node == "ENTRY"
    assert env.current_step == 0
    assert env.total_reward == 0.0
    assert isinstance(state, np.ndarray)
    assert state.shape == (47,)
    assert state.dtype == np.float32
    assert "current_node" in info
    assert "occupied_spots" in info
    assert set(env.occupied_spots) == {"P1", "P2"}


def test_step_valid_movement():
    """Verify taking a valid directional action moves the vehicle along graph edge."""
    env = ParkingEnvironment()
    env.reset(start_node="ENTRY")

    # From ENTRY (0, 1), RIGHT leads to J1 (2, 1)
    next_state, reward, terminated, truncated, info = env.step(ACTION_RIGHT)

    assert env.current_node == "J1"
    assert env.current_step == 1
    assert not terminated
    assert not truncated
    assert info["route"] == ["ENTRY", "J1"]
    assert next_state.shape == (47,)


def test_invalid_movement_prevention():
    """Verify moving where no edge exists keeps the vehicle in place and penalizes."""
    env = ParkingEnvironment()
    env.reset(start_node="ENTRY")

    # From ENTRY (0, 1), there is no UP, DOWN, or LEFT edge
    next_state, reward, terminated, truncated, info = env.step(ACTION_UP)

    assert env.current_node == "ENTRY"  # Did not move
    assert env.current_step == 1
    assert reward == REWARD_INVALID_MOVE
    assert not terminated
    assert "Invalid move" in info["message"]


def test_occupied_spot_cannot_be_entered():
    """Verify that occupied parking spots cannot be entered or selected."""
    env = ParkingEnvironment()
    # P1 is off N1 via LEFT
    env.reset(start_node="N1", occupied_spots=["P1"])

    # 1. Action masking must exclude LEFT
    valid_actions = env.get_valid_actions()
    assert ACTION_LEFT not in valid_actions

    # 2. Attempting to force action LEFT must be blocked
    next_state, reward, terminated, truncated, info = env.step(ACTION_LEFT)
    assert env.current_node == "N1"  # Vehicle remains at N1
    assert reward == REWARD_OCCUPIED_COLLISION
    assert not terminated
    assert info["success"] is False
    assert "occupied" in info["message"].lower()


def test_terminal_success_on_available_spot():
    """Verify that moving into an unoccupied spot triggers terminal success."""
    env = ParkingEnvironment()
    # P2 is off N1 via RIGHT, ensure P2 is free
    env.reset(start_node="N1", occupied_spots=["P1"])

    next_state, reward, terminated, truncated, info = env.step(ACTION_RIGHT)
    assert env.current_node == "P2"
    assert terminated is True
    assert truncated is False
    assert info["success"] is True
    assert info["parking_spot"] == "P2"
    assert reward >= REWARD_SUCCESS


def test_terminal_exit_without_parking():
    """Verify that driving to EXIT terminates with negative reward."""
    env = ParkingEnvironment()
    env.reset(start_node="J3")

    # From J3 (8, 1), RIGHT leads to EXIT (10, 1)
    next_state, reward, terminated, truncated, info = env.step(ACTION_RIGHT)
    assert env.current_node == "EXIT"
    assert terminated is True
    assert info["success"] is False
    assert "EXIT" in info["message"]


def test_state_vector_shape_and_consistency():
    """Verify that state vector shape is strictly (47,) and has no NaNs across steps."""
    env = ParkingEnvironment(max_steps=30)
    for trial in range(5):
        state, _ = env.reset(random_occupancy_rate=0.5)
        assert state.shape == (47,)
        assert not np.isnan(state).any()
        assert not np.isinf(state).any()

        for _ in range(15):
            valid = env.get_valid_actions()
            act = valid[0] if valid else 0
            state, _, term, trunc, _ = env.step(act)
            assert state.shape == (47,)
            assert not np.isnan(state).any()
            if term or trunc:
                break


def test_integration_contract_layout():
    """Verify get_parking_layout returns complete topology for frontend."""
    layout = get_parking_layout()
    assert "nodes" in layout
    assert "edges" in layout
    assert "parking_spots" in layout
    assert "grid_bounds" in layout
    assert len(layout["parking_spots"]) == 12
    assert len(layout["nodes"]) == 29


def test_integration_contract_status():
    """Verify get_parking_status returns accurate breakdown."""
    status = get_parking_status(occupied=["P1", "P2", "P3"])
    assert status["total_spots"] == 12
    assert status["occupied_count"] == 3
    assert status["available_count"] == 9
    assert status["status_map"]["P1"] == "occupied"
    assert status["status_map"]["P4"] == "available"


def test_find_parking_route_validity():
    """Verify find_parking returns valid connected graph route and expected dict schema."""
    result = find_parking(start="ENTRY", occupied=["P1", "P2", "P4", "P6"])

    assert isinstance(result, dict)
    for key in ["success", "parking_spot", "route", "steps", "total_reward", "message"]:
        assert key in result

    assert result["success"] is True
    assert result["parking_spot"] not in ["P1", "P2", "P4", "P6"]
    assert result["steps"] == len(result["route"]) - 1

    # Verify route edges actually exist in the graph
    G = create_parking_graph()
    route = result["route"]
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        assert G.has_edge(u, v), f"Edge ({u}, {v}) in route does not exist in graph!"


def test_find_parking_all_occupied_failure():
    """Verify find_parking gracefully handles 100% full parking lot."""
    all_spots = [f"P{i}" for i in range(1, 13)]
    result = find_parking(start="ENTRY", occupied=all_spots)

    assert result["success"] is False
    assert result["parking_spot"] is None
    assert result["steps"] == 0
    assert "No parking spots" in result["message"]


if __name__ == "__main__":
    print("Running automated environment test suite...")
    test_reset()
    print("  [PASS] test_reset")
    test_step_valid_movement()
    print("  [PASS] test_step_valid_movement")
    test_invalid_movement_prevention()
    print("  [PASS] test_invalid_movement_prevention")
    test_occupied_spot_cannot_be_entered()
    print("  [PASS] test_occupied_spot_cannot_be_entered")
    test_terminal_success_on_available_spot()
    print("  [PASS] test_terminal_success_on_available_spot")
    test_terminal_exit_without_parking()
    print("  [PASS] test_terminal_exit_without_parking")
    test_state_vector_shape_and_consistency()
    print("  [PASS] test_state_vector_shape_and_consistency")
    test_integration_contract_layout()
    print("  [PASS] test_integration_contract_layout")
    test_integration_contract_status()
    print("  [PASS] test_integration_contract_status")
    test_find_parking_route_validity()
    print("  [PASS] test_find_parking_route_validity")
    test_find_parking_all_occupied_failure()
    print("  [PASS] test_find_parking_all_occupied_failure")
    print("\nALL 10 TESTS PASSED SUCCESSFULLY!")
