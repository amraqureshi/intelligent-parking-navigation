"""
Intelligent Parking Navigation - Streamlit Dashboard
===================================================
Interactive pure-Python web dashboard for evaluating and visualizing
the Deep Q-Network (DQN) autonomous parking navigation agent.

Run Command:
    streamlit run app.py
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import streamlit as st

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.parking_environment import (
    ParkingEnvironment,
    create_parking_graph,
    get_parking_layout,
    get_parking_status,
    find_parking,
    load_model,
    ACTION_NAMES,
)

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Intelligent Parking Navigation (DQN)",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #0b101b; }
    .stMetric {
        background-color: #131b2e;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-success { background-color: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid #22c55e; }
    .badge-danger { background-color: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid #ef4444; }
    .badge-info { background-color: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid #0284c7; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper: Matplotlib Graph Renderer
# ---------------------------------------------------------------------------
def render_parking_lot(G, occupied_spots, path=None, destination=None):
    """Generates an enhanced top-down visual schematic with active vehicle path."""
    pos = {node: G.nodes[node]["pos"] for node in G.nodes()}

    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.patch.set_facecolor("#0b101b")
    ax.set_facecolor("#111827")

    # Draw all regular driving edges
    driving_edges = [
        (u, v) for u, v in G.edges()
        if G.nodes[u]["type"] != "parking" and G.nodes[v]["type"] != "parking"
    ]
    nx.draw_networkx_edges(
        G, pos, edgelist=driving_edges, ax=ax,
        edge_color="#374151", width=2.5, arrows=True, arrowsize=14, node_size=650,
    )

    # Draw bay entrance edges
    parking_edges = [
        (u, v) for u, v in G.edges() if G.nodes[v]["type"] == "parking"
    ]
    nx.draw_networkx_edges(
        G, pos, edgelist=parking_edges, ax=ax,
        edge_color="#4b5563", width=1.8, style="dashed", arrows=True, arrowsize=12, node_size=650,
    )

    # If route exists, highlight path
    if path and len(path) > 1:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(
            G, pos, edgelist=path_edges, ax=ax,
            edge_color="#38bdf8", width=4.5, arrows=True, arrowsize=20, node_size=750,
        )

    # Draw categorized nodes
    all_spots = [n for n, d in G.nodes(data=True) if d["type"] == "parking"]
    occupied_set = set(occupied_spots)
    free_spots = [s for s in all_spots if s not in occupied_set]
    occ_spots = [s for s in all_spots if s in occupied_set]

    entry_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "entry"]
    exit_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "exit"]
    junction_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "intersection"]
    lane_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "lane"]

    # Entry and Exit
    nx.draw_networkx_nodes(G, pos, nodelist=entry_nodes, node_color="#22c55e", node_shape="s", node_size=900, ax=ax, label="Entry Gate")
    nx.draw_networkx_nodes(G, pos, nodelist=exit_nodes, node_color="#ef4444", node_shape="s", node_size=900, ax=ax, label="Exit Gate")
    # Lanes & Intersections
    nx.draw_networkx_nodes(G, pos, nodelist=junction_nodes, node_color="#f59e0b", node_shape="o", node_size=700, ax=ax, label="Intersection")
    nx.draw_networkx_nodes(G, pos, nodelist=lane_nodes, node_color="#64748b", node_shape="o", node_size=550, ax=ax, label="Driving Lane")

    # Available spots (Green) vs Occupied spots (Red)
    if free_spots:
        nx.draw_networkx_nodes(G, pos, nodelist=free_spots, node_color="#10b981", node_shape="h", node_size=850, ax=ax, label="Available Bay")
    if occ_spots:
        nx.draw_networkx_nodes(G, pos, nodelist=occ_spots, node_color="#ef4444", node_shape="h", node_size=850, ax=ax, label="Occupied Bay")

    # Highlight vehicle's targeted destination spot
    if destination:
        nx.draw_networkx_nodes(G, pos, nodelist=[destination], node_color="#06b6d4", node_shape="h", node_size=1100, ax=ax, label="Target Destination")

    # Draw Labels
    labels = {n: n for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8.5, font_color="#ffffff", font_weight="bold", ax=ax)

    # Road center line annotations (aesthetic dashed lines)
    ax.axhline(1, color="#eab308", linestyle="--", linewidth=1.2, alpha=0.35)
    ax.axhline(5, color="#eab308", linestyle="--", linewidth=1.2, alpha=0.35)

    ax.set_title("Intelligent Parking Lot Digital Twin (Topological Schematic)", color="#f8fafc", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("X Coordinate (Lanes)", color="#94a3b8", fontsize=10)
    ax.set_ylabel("Y Coordinate (Aisles)", color="#94a3b8", fontsize=10)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, colors="#64748b")
    ax.grid(True, linestyle=":", alpha=0.2, color="#94a3b8")
    ax.set_xlim(-0.8, 10.8)
    ax.set_ylim(0.4, 5.6)

    ax.legend(loc="upper right", facecolor="#1f2937", edgecolor="#374151", labelcolor="#f3f4f6", fontsize=8.5)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "checkpoint_loaded" not in st.session_state:
    ckpt_path = "models/dqn_parking.pth"
    if os.path.exists(ckpt_path):
        load_model(ckpt_path)
        st.session_state.checkpoint_loaded = True
    else:
        st.session_state.checkpoint_loaded = False

if "occupied_spots" not in st.session_state:
    st.session_state.occupied_spots = ["P1", "P2", "P4", "P7", "P8", "P11"]

if "search_result" not in st.session_state:
    st.session_state.search_result = None


# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------
st.sidebar.title("🚘 ParkNav Controls")

# Model Diagnostic
if st.session_state.checkpoint_loaded:
    st.sidebar.markdown('<span class="status-badge badge-success">✓ PyTorch DDQN Loaded</span>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<span class="status-badge badge-danger">⚠ Missing Checkpoint</span>', unsafe_allow_html=True)

st.sidebar.markdown("---")

# Scenario Preset Selector
scenario = st.sidebar.selectbox(
    "Scenario Congestion Preset",
    [
        "Custom Selection",
        "Light (25% Occupied)",
        "Medium (50% Occupied)",
        "Heavy (75% Occupied)",
        "Aisle 1 Full (P1-P6)",
        "Aisle 2 Full (P7-P12)",
        "Empty Lot (0% Occupied)",
        "Full Lot (100% Occupied)",
    ],
    index=2,
)

if scenario == "Light (25% Occupied)":
    st.session_state.occupied_spots = ["P1", "P7", "P10"]
elif scenario == "Medium (50% Occupied)":
    st.session_state.occupied_spots = ["P1", "P2", "P4", "P7", "P8", "P11"]
elif scenario == "Heavy (75% Occupied)":
    st.session_state.occupied_spots = ["P1", "P2", "P3", "P4", "P5", "P7", "P8", "P9", "P11"]
elif scenario == "Aisle 1 Full (P1-P6)":
    st.session_state.occupied_spots = ["P1", "P2", "P3", "P4", "P5", "P6"]
elif scenario == "Aisle 2 Full (P7-P12)":
    st.session_state.occupied_spots = ["P7", "P8", "P9", "P10", "P11", "P12"]
elif scenario == "Empty Lot (0% Occupied)":
    st.session_state.occupied_spots = []
elif scenario == "Full Lot (100% Occupied)":
    st.session_state.occupied_spots = [f"P{i}" for i in range(1, 13)]

# Manual spot occupancy multi-select
all_available_spots = [f"P{i}" for i in range(1, 13)]
selected_occ = st.sidebar.multiselect(
    "Occupied Parking Bays",
    options=all_available_spots,
    default=st.session_state.occupied_spots,
)
st.session_state.occupied_spots = selected_occ

start_node = st.sidebar.selectbox("Vehicle Entry Point", ["ENTRY", "J1", "J2", "J3"], index=0)

col_nav1, col_nav2 = st.sidebar.columns(2)
find_clicked = col_nav1.button("🔍 Find Spot", use_container_width=True, type="primary")
reset_clicked = col_nav2.button("↺ Reset", use_container_width=True)

if reset_clicked:
    st.session_state.search_result = None
    st.session_state.occupied_spots = ["P1", "P2", "P4", "P7", "P8", "P11"]
    st.rerun()

if find_clicked:
    res = find_parking(start=start_node, occupied=st.session_state.occupied_spots)
    st.session_state.search_result = res


# ---------------------------------------------------------------------------
# Main Dashboard View
# ---------------------------------------------------------------------------
st.title("Autonomous Parking Navigation System")
st.markdown("Top-down discrete simulation using a trained **Deep Q-Network (Double DQN)** agent on NetworkX topology.")

# Top Metrics Row
status_info = get_parking_status(st.session_state.occupied_spots)
m1, m2, m3, m4 = st.columns(4)

m1.metric("Total Bays", status_info["total_spots"])
m2.metric("Available", status_info["available_count"])
m3.metric("Occupied", status_info["occupied_count"])
m4.metric("Occupancy Rate", f"{status_info['occupancy_rate'] * 100:.1f}%")

st.markdown("---")

tab_sim, tab_arch, tab_bench = st.tabs(["🗺️ Live Simulation", "🧠 DQN Architecture", "📊 Benchmark & Evaluation"])

G = create_parking_graph()

with tab_sim:
    col_plot, col_info = st.columns([7, 3])

    with col_plot:
        active_route = None
        target_bay = None
        if st.session_state.search_result and st.session_state.search_result.get("success"):
            active_route = st.session_state.search_result.get("route")
            target_bay = st.session_state.search_result.get("parking_spot")

        fig = render_parking_lot(
            G=G,
            occupied_spots=st.session_state.occupied_spots,
            path=active_route,
            destination=target_bay,
        )
        st.pyplot(fig)

    with col_info:
        st.subheader("Navigation Telemetry")
        if st.session_state.search_result:
            r = st.session_state.search_result
            if r.get("success"):
                st.markdown(f'<div class="status-badge badge-success">Target Reached: {r.get("parking_spot")}</div>', unsafe_allow_html=True)
                st.write("")
                st.write(f"**Algorithm:** {r.get('algorithm', 'Deep Q-Network')}")
                st.write(f"**Steps Taken:** {r.get('steps')} steps")
                st.write(f"**Total Reward:** `{r.get('total_reward', 0.0):.2f}`")
                st.write(f"**Route Sequence:**")
                st.code(" → ".join(r.get("route", [])))
            else:
                st.markdown(f'<div class="status-badge badge-danger">Search Failed</div>', unsafe_allow_html=True)
                st.write("")
                st.write(f"**Message:** {r.get('message', 'No route available')}")
        else:
            st.info("Click **🔍 Find Spot** in the sidebar to simulate autonomous navigation.")

        st.markdown("#### Facility Legend")
        st.markdown("- 🟩 **Available Bay**: Free stall ($P_1-P_{12}$)")
        st.markdown("- 🟥 **Occupied Bay**: Blocked stall")
        st.markdown("- 🟦 **Cyan Path**: Optimal trajectory found by DQN")
        st.markdown("- 🟨 **Junction**: Decision intersections")

with tab_arch:
    st.subheader("Deep Q-Network (Double DQN) Specification")
    c_arch1, c_arch2 = st.columns(2)

    with c_arch1:
        st.markdown(
            """
            ### Neural Network Architecture
            - **Input Dimension:** 47 features (One-hot node, coordinates, spot occupancy, relative vector, step budget)
            - **Hidden Layers:**
              - FC Layer 1: 128 units (ReLU)
              - FC Layer 2: 128 units (ReLU)
              - FC Layer 3: 64 units (ReLU)
            - **Output Dimension:** 4 Q-values (UP, DOWN, LEFT, RIGHT)
            - **Target Updates:** Polyak Soft Updates ($\\tau = 0.005$)
            - **Loss Function:** Smooth L1 (Huber) Loss
            """
        )

    with c_arch2:
        st.markdown(
            """
            ### Reward Formulation
            - **+50.0**: Successfully parked in an available bay
            - **-0.2**: Step time penalty (encourages shortest route)
            - **+0.5 $\\times \\Delta d$**: Potential distance progress shaping
            - **-2.0**: Invalid move (wall / no edge)
            - **-5.0**: Collision with occupied spot
            - **-10.0**: Exiting facility unparked / Step timeout (>50 steps)
            """
        )

    if os.path.exists("results/training_curve.png"):
        st.markdown("### Training Reward & Success Curves (400 Episodes)")
        st.image("results/training_curve.png", caption="DQN Learning Progression: Rolling Reward and Convergence Rate")

with tab_bench:
    st.subheader("DQN vs Dijkstra Shortest Path Benchmark")
    if os.path.exists("results/comparison_results.json"):
        with open("results/comparison_results.json", "r") as f:
            benchmarks = json.load(f)
        st.json(benchmarks)

    if os.path.exists("results/performance_vs_occupancy.png"):
        st.image("results/performance_vs_occupancy.png", caption="Benchmark Evaluation across 25%, 50%, and 75% Congestion")
