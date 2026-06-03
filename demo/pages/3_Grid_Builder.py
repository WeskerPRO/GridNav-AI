import streamlit as st                                  # the web UI
import sys                                               # fix import path
import os                                                # path helpers
import time                                              # animation frame pauses
import io                                                # read uploaded models
import random                                            # random-obstacle button
import numpy as np                                       # (used indirectly)
import torch                                             # load model weights

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make 'core' importable
from core.grid_utils import (render_grid_image, find_shortest_path_bfs,         # viz + BFS
                              grid_from_builder, image_to_bytes, ACTIONS_4)      # builder -> grid
from core.rl_model import (GridEnvironmentRL, run_rl_inference,                  # RL pieces
                            DQN_LSTM, INPUT_DIM, NUM_ACTIONS)
from core.supervised_model import (run_supervised_inference,                    # supervised pieces
                                    PathPredictionResNet)

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(                                      # tab + layout
    page_title="GridNav — Grid Builder",
    page_icon="🏗️",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&family=Orbitron:wght@400;700&display=swap');
    .stApp { background-color: #0D0D1A; color: #E0E0FF; }
    section[data-testid="stSidebar"] { background-color: #0F0F20; border-right: 1px solid #1A1A3A; }
    h1, h2, h3 { font-family: 'Orbitron', monospace !important; }
    h1 { color: #E94560 !important; }
    h2 { color: #00B4D8 !important; }
    h3 { color: #8888AA !important; }
    p, li { font-family: 'JetBrains Mono', monospace; color: #AAAACC; font-size:0.85rem; }
    .stButton > button { background: linear-gradient(135deg, #E94560, #533483); color:white; border:none; border-radius:4px; font-family:'JetBrains Mono',monospace; font-weight:700; padding: 0.3rem 1rem; }
    [data-testid="metric-container"] { background:#1A1A2E; border:1px solid #1A1A3A; border-radius:8px; padding:12px; }
    [data-testid="metric-container"] label { color:#8888AA !important; font-family:'JetBrains Mono',monospace !important; font-size:0.75rem !important; }
    [data-testid="metric-container"] [data-testid="metric-value"] { color:#00B4D8 !important; font-family:'JetBrains Mono',monospace !important; }
    div[data-testid="stHorizontalBlock"] button { min-width: 28px !important; padding: 4px !important; font-size: 0.7rem !important; }
</style>
""", unsafe_allow_html=True)                             # inject dark theme + tiny cell buttons

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.markdown("## 🏗️ GRID BUILDER")                # sidebar header
st.sidebar.markdown("---")

rows = st.sidebar.slider("Rows", 5, 20, 10)             # grid height
cols = st.sidebar.slider("Cols", 5, 20, 10)             # grid width

st.sidebar.markdown("### 🖊️ DRAWING MODE")
draw_mode = st.sidebar.radio(                           # what a cell-click places
    "Click cells to place:",
    ["🧱 Obstacle", "🤖 Robot Start", "🎯 Target", "⬜ Erase"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 MODELS")
rl_file  = st.sidebar.file_uploader("RL Model (.pth)",         type=["pth"], key="gb_rl")   # RL upload
sup_file = st.sidebar.file_uploader("Supervised Model (.pth)", type=["pth"], key="gb_sup")  # supervised upload
anim_speed = st.sidebar.slider("Animation speed (ms)", 50, 500, 200, 50)                    # frame pause

# =============================================================================
# SESSION STATE
# =============================================================================

def init_grid_state(rows, cols):
    return {(r, c): 'free' for r in range(rows) for c in range(cols)}  # all cells start free

if "gb_cells"     not in st.session_state: st.session_state.gb_cells     = init_grid_state(rows, cols)  # the drawing
if "gb_robot"     not in st.session_state: st.session_state.gb_robot     = None  # robot cell
if "gb_target"    not in st.session_state: st.session_state.gb_target    = None  # target cell
if "gb_rows"      not in st.session_state: st.session_state.gb_rows      = rows  # remembered size
if "gb_cols"      not in st.session_state: st.session_state.gb_cols      = cols
if "gb_rl_model"  not in st.session_state: st.session_state.gb_rl_model  = None  # loaded RL model
if "gb_sup_model" not in st.session_state: st.session_state.gb_sup_model = None  # loaded supervised model

# reset grid if size changed
if st.session_state.gb_rows != rows or st.session_state.gb_cols != cols:  # slider moved?
    st.session_state.gb_cells  = init_grid_state(rows, cols)  # wipe to fit new size
    st.session_state.gb_robot  = None
    st.session_state.gb_target = None
    st.session_state.gb_rows   = rows
    st.session_state.gb_cols   = cols

# ── load models ───────────────────────────────────────────────────────────────
if rl_file is not None and st.session_state.gb_rl_model is None:  # upload + not yet loaded
    try:
        buf   = io.BytesIO(rl_file.read())               # bytes -> buffer
        model = DQN_LSTM(INPUT_DIM, 128, NUM_ACTIONS)    # build matching net
        model.load_state_dict(torch.load(buf, weights_only=True, map_location='cpu'))  # load weights
        model.eval()
        st.session_state.gb_rl_model = model
        st.sidebar.success("✅ RL model loaded")
    except Exception as e:
        st.sidebar.error(f"RL model error: {e}")

if sup_file is not None and st.session_state.gb_sup_model is None:  # upload + not yet loaded
    try:
        buf   = io.BytesIO(sup_file.read())
        model = PathPredictionResNet(rows, cols)         # size-agnostic ResNet
        model.load_state_dict(torch.load(buf, weights_only=True, map_location='cpu'))
        model.eval()
        st.session_state.gb_sup_model = model
        st.sidebar.success("✅ Supervised model loaded")
    except Exception as e:
        st.sidebar.error(f"Supervised model error: {e}")

# =============================================================================
# MAIN
# =============================================================================

st.markdown("# 🏗️ GRID BUILDER")                         # page title
st.markdown("Draw your own maze. Place obstacles, robot start, and target. Then benchmark both models.")
st.markdown("---")

left_col, right_col = st.columns([2, 1])                # grid (wide) | controls

with left_col:
    st.markdown("## 🗺️ DRAW GRID")
    st.markdown(f"*Selected mode: **{draw_mode}***")     # show active brush

    for r in range(rows):                               # build a button per cell
        btn_cols = st.columns(cols)                     # one column per grid col
        for c in range(cols):
            cell = st.session_state.gb_cells.get((r, c), 'free')  # current cell state

            if (r, c) == st.session_state.gb_robot:     # pick the emoji to show
                label = "🤖"
            elif (r, c) == st.session_state.gb_target:
                label = "🎯"
            elif cell == 'obstacle':
                label = "🧱"
            else:
                label = "  "

            if btn_cols[c].button(label, key=f"cell_{r}_{c}"):  # cell clicked -> apply brush
                if "Obstacle" in draw_mode:
                    st.session_state.gb_cells[(r, c)] = 'obstacle'
                elif "Robot" in draw_mode:
                    st.session_state.gb_robot = (r, c)
                    st.session_state.gb_cells[(r, c)] = 'free'   # robot sits on a free cell
                elif "Target" in draw_mode:
                    st.session_state.gb_target = (r, c)
                    st.session_state.gb_cells[(r, c)] = 'free'
                elif "Erase" in draw_mode:
                    st.session_state.gb_cells[(r, c)] = 'free'
                    if st.session_state.gb_robot  == (r, c): st.session_state.gb_robot  = None  # also clear robot
                    if st.session_state.gb_target == (r, c): st.session_state.gb_target = None  # also clear target
                st.rerun()                              # redraw the grid

with right_col:
    st.markdown("## 🎮 CONTROLS")

    if st.button("🗑️  Clear Grid"):                      # wipe everything
        st.session_state.gb_cells  = init_grid_state(rows, cols)
        st.session_state.gb_robot  = None
        st.session_state.gb_target = None
        st.rerun()

    if st.button("🎲  Random Obstacles"):                # sprinkle random walls
        for r in range(rows):
            for c in range(cols):
                if (r, c) != st.session_state.gb_robot and \
                   (r, c) != st.session_state.gb_target:        # keep robot/target clear
                    st.session_state.gb_cells[(r, c)] = \
                        'obstacle' if random.random() < 0.20 else 'free'  # 20% obstacles
        st.rerun()

    st.markdown("---")
    st.markdown("### 📋 STATUS")

    robot_status   = f"🤖 {st.session_state.gb_robot}"  if st.session_state.gb_robot  else "🤖 Not placed"   # robot text
    target_status  = f"🎯 {st.session_state.gb_target}" if st.session_state.gb_target else "🎯 Not placed"   # target text
    obstacle_count = sum(1 for v in st.session_state.gb_cells.values() if v == 'obstacle')                  # wall count

    st.markdown(f"`{robot_status}`")
    st.markdown(f"`{target_status}`")
    st.markdown(f"`🧱 Obstacles: {obstacle_count}`")

    # ── real-time path validation ─────────────────────────────────────────
    if st.session_state.gb_robot and st.session_state.gb_target:  # both placed?
        _, numeric_check = grid_from_builder(            # convert drawing -> numeric grid
            st.session_state.gb_cells, rows, cols,
            st.session_state.gb_robot, st.session_state.gb_target
        )
        check_path = find_shortest_path_bfs(             # is there a path right now?
            numeric_check,
            st.session_state.gb_robot,
            st.session_state.gb_target,
            ACTIONS_4
        )
        if check_path:
            st.success(f"✅ Path exists ({len(check_path)-1} steps)")
        else:
            st.error("❌ No path — robot is trapped")

    st.markdown("---")
    run_ready = (st.session_state.gb_robot  is not None and   # need both endpoints
                 st.session_state.gb_target is not None)

    if not run_ready:
        st.warning("Place both 🤖 Robot and 🎯 Target to run.")

    run_btn = st.button("▶  SOLVE IT", disabled=not run_ready)  # enabled only when ready

# =============================================================================
# RUN INFERENCE
# =============================================================================

if run_btn and run_ready:                               # solve pressed
    _, numeric = grid_from_builder(                      # final numeric grid
        st.session_state.gb_cells, rows, cols,
        st.session_state.gb_robot, st.session_state.gb_target
    )

    cell_sz = max(20, 480 // max(rows, cols))           # px per cell

    # ── validate path before running any model ────────────────────────────
    bfs_path = find_shortest_path_bfs(                   # solvable at all?
        numeric,
        st.session_state.gb_robot,
        st.session_state.gb_target,
        ACTIONS_4
    )

    if bfs_path is None:                                # unsolvable -> stop
        st.error(
            "❌ No path exists between robot and target. "
            "BFS cannot solve this grid — RL and Supervised won't either. "
            "Remove some obstacles to create a valid path."
        )
        st.stop()

    st.success(f"✅ Valid grid — BFS optimal: {len(bfs_path)-1} steps")  # optimal length

    st.markdown("---")
    st.markdown("## 🏁 RESULTS")

    tabs = st.tabs(["🤖 RL Agent", "📚 Supervised", "🔵 BFS Optimal"])  # one tab per solver

    # ── RL tab ────────────────────────────────────────────────────────────
    with tabs[0]:
        if st.session_state.gb_rl_model is not None:    # have an RL model?
            env = GridEnvironmentRL(                     # build env on this grid
                numeric,
                st.session_state.gb_robot,
                st.session_state.gb_target
            )
            rl_path, rl_reward, rl_success = run_rl_inference(  # greedy rollout
                st.session_state.gb_rl_model, env, rows, cols
            )

            rl_display = st.empty()                      # animation slot
            for i in range(len(rl_path)):                # play the path
                img = render_grid_image(
                    numeric, rl_path[i],
                    st.session_state.gb_target,
                    path_taken=rl_path[:i+1],
                    cell_size=cell_sz
                )
                rl_display.image(image_to_bytes(img), width=500)
                time.sleep(anim_speed / 1000)            # pause between frames

            c1, c2, c3 = st.columns(3)                   # RL metrics
            c1.metric("Steps",   len(rl_path) - 1)
            c2.metric("Reward",  f"{rl_reward:.3f}")
            c3.metric("Success", "✅" if rl_success else "❌")
        else:
            st.warning("Upload RL model (.pth) in sidebar to run RL inference.")

    # ── Supervised tab ────────────────────────────────────────────────────
    with tabs[1]:
        if st.session_state.gb_sup_model is not None:   # have a supervised model?
            sup_path, sup_success = run_supervised_inference(  # rollout
                st.session_state.gb_sup_model,
                numeric,
                st.session_state.gb_robot,
                st.session_state.gb_target,
                rows, cols
            )

            sup_display = st.empty()                     # animation slot
            for i in range(len(sup_path)):               # play the path
                img = render_grid_image(
                    numeric, sup_path[i],
                    st.session_state.gb_target,
                    path_taken=sup_path[:i+1],
                    cell_size=cell_sz
                )
                sup_display.image(image_to_bytes(img), width=500)
                time.sleep(anim_speed / 1000)

            c1, c2 = st.columns(2)                       # supervised metrics
            c1.metric("Steps",   len(sup_path) - 1)
            c2.metric("Success", "✅" if sup_success else "❌")
        else:
            st.warning("Upload Supervised model (.pth) in sidebar.")

    # ── BFS tab ───────────────────────────────────────────────────────────
    with tabs[2]:
        bfs_display = st.empty()                         # animation slot
        for i in range(len(bfs_path)):                   # play the optimal path
            img = render_grid_image(
                numeric, bfs_path[i],
                st.session_state.gb_target,
                path_taken=bfs_path[:i+1],
                cell_size=cell_sz
            )
            bfs_display.image(image_to_bytes(img), width=500)
            time.sleep(anim_speed / 1000)

        st.metric("BFS Optimal Steps", len(bfs_path) - 1)  # the gold standard
        st.success("BFS always finds the shortest path.")
