"""
Backend Package for Intelligent Parking Navigation
==================================================
Exposes the core environment, layout utilities, and integration endpoints.
"""

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
    ACTION_NAMES,
    ACTION_VECTORS,
    REWARD_SUCCESS,
    REWARD_STEP,
    REWARD_INVALID_MOVE,
    REWARD_OCCUPIED_COLLISION,
    REWARD_EXIT_UNPARKED,
    REWARD_TIMEOUT,
)

__all__ = [
    "ParkingEnvironment",
    "create_parking_graph",
    "get_parking_layout",
    "get_parking_status",
    "find_parking",
    "load_model",
    "ACTION_UP",
    "ACTION_DOWN",
    "ACTION_LEFT",
    "ACTION_RIGHT",
    "ACTION_NAMES",
    "ACTION_VECTORS",
    "REWARD_SUCCESS",
    "REWARD_STEP",
    "REWARD_INVALID_MOVE",
    "REWARD_OCCUPIED_COLLISION",
    "REWARD_EXIT_UNPARKED",
    "REWARD_TIMEOUT",
]
