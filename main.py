"""
Intelligent Parking Navigation - Main Integration Demo
======================================================
Demonstrates the exposed backend API contract:
1. get_parking_layout()
2. get_parking_status()
3. find_parking(start, occupied)
4. Saves a top-down matplotlib visualization to results/parking_layout.png
"""

import os
import json
import matplotlib.pyplot as plt
import networkx as nx

from backend.parking_environment import (
    ParkingEnvironment,
    get_parking_layout,
    get_parking_status,
    find_parking,
    load_model,
    create_parking_graph,
)


def export_layout_visualization(save_path: str = "results/parking_layout.png"):
    """
    Renders a top-down visual schematic of the parking facility with lanes,
    parking bays, entry/exit gates, and labels.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    G = create_parking_graph()
    pos = {node: G.nodes[node]["pos"] for node in G.nodes()}

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor("#1e293b")  # Dark modern background
    fig.patch.set_facecolor("#0f172a")

    # Draw driving edges
    driving_edges = [
        (u, v) for u, v in G.edges()
        if G.nodes[u]["type"] != "parking" and G.nodes[v]["type"] != "parking"
    ]
    nx.draw_networkx_edges(
        G, pos, edgelist=driving_edges, ax=ax,
        edge_color="#64748b", width=2.5, arrows=True, arrowsize=14, node_size=700
    )

    # Draw parking entrance edges
    parking_edges = [
        (u, v) for u, v in G.edges() if G.nodes[v]["type"] == "parking"
    ]
    nx.draw_networkx_edges(
        G, pos, edgelist=parking_edges, ax=ax,
        edge_color="#38bdf8", width=2.0, style="dashed", arrows=True, arrowsize=12, node_size=700
    )

    # Node color groupings
    entry_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "entry"]
    exit_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "exit"]
    junction_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "intersection"]
    lane_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "lane"]
    parking_spots = [n for n, d in G.nodes(data=True) if d["type"] == "parking"]

    nx.draw_networkx_nodes(G, pos, nodelist=entry_nodes, node_color="#22c55e", node_shape="s", node_size=900, ax=ax, label="Entry Gate")
    nx.draw_networkx_nodes(G, pos, nodelist=exit_nodes, node_color="#ef4444", node_shape="s", node_size=900, ax=ax, label="Exit Gate")
    nx.draw_networkx_nodes(G, pos, nodelist=junction_nodes, node_color="#f59e0b", node_shape="o", node_size=700, ax=ax, label="Junction")
    nx.draw_networkx_nodes(G, pos, nodelist=lane_nodes, node_color="#94a3b8", node_shape="o", node_size=600, ax=ax, label="Lane Node")
    nx.draw_networkx_nodes(G, pos, nodelist=parking_spots, node_color="#3b82f6", node_shape="h", node_size=800, ax=ax, label="Parking Spot (P1-P12)")

    # Labels
    labels = {n: n for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_color="#ffffff", font_weight="bold", ax=ax)

    ax.set_title("Intelligent Parking Lot Topological Layout (NetworkX)", color="#f8fafc", fontsize=15, fontweight="bold", pad=15)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, colors="#94a3b8")
    ax.grid(True, linestyle="--", alpha=0.2, color="#94a3b8")
    ax.set_xlabel("X Coordinate (Grid Lanes)", color="#94a3b8", fontsize=11)
    ax.set_ylabel("Y Coordinate (Grid Aisles)", color="#94a3b8", fontsize=11)

    legend = ax.legend(loc="upper right", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[Visualization] Schematic saved to: {save_path}")


def main():
    print("=" * 70)
    print("  Intelligent Parking Navigation Using Deep Q-Network (DQN)")
    print("  Backend Foundation & Environment Demo")
    print("=" * 70)

    # 1. Inspect Facility Layout
    layout = get_parking_layout()
    print(f"\n[1] Facility Layout Loaded:")
    print(f"    - Total Nodes   : {len(layout['nodes'])}")
    print(f"    - Total Edges   : {len(layout['edges'])}")
    print(f"    - Parking Spots : {layout['parking_spots']}")
    print(f"    - Grid Bounds   : {layout['grid_bounds']}")

    # 2. Check Parking Status
    occupied_example = ["P1", "P2", "P3", "P7", "P8"]
    status = get_parking_status(occupied=occupied_example)
    print(f"\n[2] Facility Parking Status:")
    print(f"    - Total Spots   : {status['total_spots']}")
    print(f"    - Occupied ({status['occupied_count']}): {status['occupied_spots']}")
    print(f"    - Available ({status['available_count']}): {status['available_spots']}")
    print(f"    - Occupancy Rate: {status['occupancy_rate'] * 100:.1f}%")

    # 3. Test find_parking() Navigation Contract
    print("\n[3] Testing find_parking() Contract:")
    nav_result = find_parking(start="ENTRY", occupied=occupied_example)
    print("    Result Payload:")
    print(json.dumps(nav_result, indent=4))

    # 4. Test Full Occupancy Handling
    all_occupied = [f"P{i}" for i in range(1, 13)]
    full_result = find_parking(start="ENTRY", occupied=all_occupied)
    print(f"\n[4] Full Lot Reachability Test:")
    print(f"    - Success: {full_result['success']}")
    print(f"    - Message: {full_result['message']}")

    # 5. Export Schematic Diagram
    print("\n[5] Generating Top-Down Visual Schematic...")
    export_layout_visualization()

    print("\n" + "=" * 70)
    print("  Backend verification complete. Ready for Claude's DQN training!")
    print("=" * 70)


if __name__ == "__main__":
    main()
