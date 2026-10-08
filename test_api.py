"""
Automated Test Suite for FastAPI Backend Endpoints
=================================================
Verifies:
1. /api/health returns online status and checkpoint presence.
2. /api/layout returns complete graph nodes, edges, spots, and bounds.
3. /api/state returns valid simulation snapshot.
4. /api/reset resets environment cleanly.
5. /api/toggle-spot correctly modifies spot occupancy.
6. /api/scenario correctly applies presets (low, medium, high, aisle1_full).
7. /api/search computes a valid DQN route to an available spot.
8. /api/step advances simulation state cleanly.
9. /api/metrics returns architecture specifications and benchmark data.
"""

import pytest
from fastapi.testclient import TestClient

from backend.api import app, session


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["num_spots"] == 12
    assert data["num_nodes"] == 29
    assert data["checkpoint_exists"] is True
    assert data["model_loaded"] is True


def test_api_layout(client):
    res = client.get("/api/layout")
    assert res.status_code == 200
    data = res.json()
    assert len(data["nodes"]) == 29
    assert len(data["parking_spots"]) == 12
    assert "edges" in data
    assert len(data["edges"]) > 40
    assert "grid_bounds" in data


def test_api_state(client):
    res = client.get("/api/state")
    assert res.status_code == 200
    data = res.json()
    assert "current_node" in data
    assert "current_pos" in data
    assert "status_map" in data
    assert "valid_actions" in data
    assert data["total_spots"] == 12


def test_api_reset(client):
    res = client.post("/api/reset", json={"start_node": "ENTRY", "occupied_spots": ["P1", "P2"]})
    assert res.status_code == 200
    data = res.json()
    assert data["current_node"] == "ENTRY"
    assert "P1" in data["occupied_spots"]
    assert "P2" in data["occupied_spots"]
    assert "P3" in data["available_spots"]


def test_api_toggle_spot(client):
    # Free P1 first via reset
    client.post("/api/reset", json={"start_node": "ENTRY", "occupied_spots": []})
    
    # Toggle P1 -> occupied
    res1 = client.post("/api/toggle-spot", json={"spot_id": "P1", "occupied": True})
    assert res1.status_code == 200
    assert "P1" in res1.json()["occupied_spots"]

    # Toggle P1 -> available
    res2 = client.post("/api/toggle-spot", json={"spot_id": "P1", "occupied": False})
    assert res2.status_code == 200
    assert "P1" not in res2.json()["occupied_spots"]


def test_api_scenario(client):
    # Set high congestion (75%)
    res = client.post("/api/scenario", json={"preset": "high"})
    assert res.status_code == 200
    data = res.json()
    assert data["occupied_count"] == 9
    assert data["available_count"] == 3


def test_api_search_dqn(client):
    # Reset with standard scenario
    client.post("/api/scenario", json={"preset": "medium"})

    res = client.post("/api/search")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["parking_spot"] is not None
    assert len(data["route"]) >= 2
    assert data["route"][0] == "ENTRY"
    assert data["parking_spot"] in ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12"]
    assert "Deep Q-Network" in data["algorithm"]


def test_api_step(client):
    client.post("/api/reset", json={"start_node": "ENTRY", "occupied_spots": ["P1", "P2"]})

    res = client.post("/api/step")
    assert res.status_code == 200
    data = res.json()
    assert "action_name" in data
    assert "current_node" in data
    assert "reward" in data
    assert "state" in data


def test_api_metrics(client):
    res = client.get("/api/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "model_architecture" in data
    assert data["model_architecture"]["type"] == "Double Deep Q-Network (DDQN)"
    assert data["model_architecture"]["training_episodes"] == 400
    assert "benchmarks" in data


def test_api_search_full_occupancy(client):
    """Verify that full occupancy (all 12 spots occupied) is handled gracefully."""
    all_spots = [f"P{i}" for i in range(1, 13)]
    client.post("/api/reset", json={"start_node": "ENTRY", "occupied_spots": all_spots})

    res = client.post("/api/search")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is False
    assert data["parking_spot"] is None
    assert "No parking spots" in data["message"]


def test_api_search_action_sequence_and_valid_transitions(client):
    """Verify search response contains actions, reward, and valid graph transitions."""
    client.post("/api/scenario", json={"preset": "low"})
    res = client.post("/api/search")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "actions" in data
    assert len(data["actions"]) == len(data["route"]) - 1
    assert "reward" in data
    assert "total_reward" in data
    assert data["reward"] == data["total_reward"]


def test_api_scenario_presets_and_aliases(client):
    """Verify various preset naming variants work correctly."""
    for preset_name, exp_occ in [
        ("25% Occupied (Light)", 3),
        ("50% Occupied (Medium)", 6),
        ("75% Occupied (Heavy)", 9),
        ("Aisle 1 Full (P01-P06)", 6),
        ("Aisle 2 Full (P07-P12)", 6),
        ("100% Free Lot", 0),
        ("full", 12),
    ]:
        res = client.post("/api/scenario", json={"preset": preset_name})
        assert res.status_code == 200
        assert res.json()["occupied_count"] == exp_occ

