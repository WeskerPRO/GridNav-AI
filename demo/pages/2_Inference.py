import streamlit as st
import sys
import os
import time
import io
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.grid_utils import (generate_random_grid, render_grid_image,
                              find_shortest_path_bfs, image_to_bytes, ACTIONS_4)
from core.rl_model import (GridEnvironmentRL, run_rl_inference,
                            DQN_LSTM, INPUT_DIM, NUM_ACTIONS)
from core.supervised_model import (run_supervised_inference,
                                    PathPredictionResNet)

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="GridNav — Inference",
    page_icon="🤖",
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
    .stButton > button { background: linear-gradient(135deg, #E94560, #533483); color:white; border:none; border-radius:4px; font-family:'JetBrains Mono',monospace; font-weight:700; }
    [data-testid="metric-container"] { background:#1A1A2E; border:1px solid #1A1A3A; border-radius:8px; padding:12px; }
    [data-testid="metric-container"] label { color:#8888AA !important; font-family:'JetBrains Mono',monospace !important; font-size:0.75rem !important; }
    [data-testid="metric-container"] [data-testid="metric-value"] { color:#00B4D8 !important; font-family:'JetBrains Mono',monospace !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.markdown("## 🤖 INFERENCE CONFIG")
st.sidebar.markdown("---")

rows       = st.sidebar.slider("Grid rows",           8,  25, 15)
cols       = st.sidebar.slider("Grid cols",           8,  25, 15)
density    = st.sidebar.slider("Obstacle density", 0.10, 0.40, 0.20, 0.05)
anim_speed = st.sidebar.slider("Animation delay (ms)", 50, 500, 150, 50)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 LOAD MODELS")

rl_file  = st.sidebar.file_uploader("RL Model (.pth)",
                                     type=["pth"], key="rl_upload")
sup_file = st.sidebar.file_uploader("Supervised Model (.pth)",
                                     type=["pth"], key="sup_upload")

st.sidebar.markdown("---")
st.sidebar.info(
    "RL model must be tested on the **same grid size** it was trained on. "
    "Match rows/cols to your training config."
)

# =============================================================================
# SESSION STATE
# =============================================================================

if "inf_numeric" not in st.session_state: st.session_state.inf_numeric = None
if "inf_robot"   not in st.session_state: st.session_state.inf_robot   = None
if "inf_target"  not in st.session_state: st.session_state.inf_target  = None
if "rl_model"    not in st.session_state: st.session_state.rl_model    = None
if "sup_model"   not in st.session_state: st.session_state.sup_model   = None

# =============================================================================
# LOAD MODELS
# =============================================================================

if rl_file is not None:
    try:
        buf   = io.BytesIO(rl_file.read())
        model = DQN_LSTM(INPUT_DIM, 128, NUM_ACTIONS)
        model.load_state_dict(torch.load(buf, weights_only=True, map_location='cpu'))
        model.eval()
        st.session_state.rl_model = model
        st.sidebar.success("✅ RL model loaded")
    except Exception as e:
        st.sidebar.error(f"Error loading RL model: {e}")

if sup_file is not None:
    try:
        buf   = io.BytesIO(sup_file.read())
        model = PathPredictionResNet(rows, cols)
        model.load_state_dict(torch.load(buf, weights_only=True, map_location='cpu'))
        model.eval()
        st.session_state.sup_model = model
        st.sidebar.success("✅ Supervised model loaded")
    except Exception as e:
        st.sidebar.error(f"Error loading supervised model: {e}")

# =============================================================================
# MAIN
# =============================================================================

st.markdown("# 🤖 INFERENCE")
st.markdown("Load trained models and watch them navigate. Compare RL vs Supervised side by side.")
st.markdown("---")

col_gen, col_run = st.columns([1, 1])
gen_btn = col_gen.button("🎲  Generate New Grid")
run_btn = col_run.button("▶  Run Inference")

# ── generate grid ─────────────────────────────────────────────────────────────
if gen_btn:
    result = generate_random_grid(rows, cols, density)
    _, numeric, _, robot_start, target = result
    if numeric is not None:
        st.session_state.inf_numeric = numeric
        st.session_state.inf_robot   = robot_start
        st.session_state.inf_target  = target
    else:
        st.error("Could not generate grid. Try lower obstacle density.")

# ── show grid if exists ───────────────────────────────────────────────────────
if st.session_state.inf_numeric is not None:
    numeric = st.session_state.inf_numeric
    cell_sz = max(16, 400 // max(rows, cols))

    if run_btn:
        # ── validate path exists ──────────────────────────────────────────
        bfs_path = find_shortest_path_bfs(
            numeric,
            st.session_state.inf_robot,
            st.session_state.inf_target,
            ACTIONS_4
        )
        if bfs_path is None:
            st.error("❌ No valid path exists on this grid. Generate a new one.")
            st.stop()

        bfs_steps = len(bfs_path) - 1

        st.markdown("---")
        st.markdown("## COMPARISON")

        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown("### 🧠 GridNav-RL (Reinforcement)")
            rl_grid_display = st.empty()

        with right_col:
            st.markdown("### 📚 GridNav-AI (Supervised)")
            sup_grid_display = st.empty()

        # ── run RL inference ──────────────────────────────────────────────
        rl_path    = [st.session_state.inf_robot]
        rl_success = False
        rl_reward  = 0

        if st.session_state.rl_model is not None:
            env = GridEnvironmentRL(
                numeric,
                st.session_state.inf_robot,
                st.session_state.inf_target
            )
            rl_path, rl_reward, rl_success = run_rl_inference(
                st.session_state.rl_model, env, rows, cols
            )
        else:
            left_col.warning("No RL model loaded. Upload a .pth file in sidebar.")

        # ── run Supervised inference ──────────────────────────────────────
        sup_path    = [st.session_state.inf_robot]
        sup_success = False

        if st.session_state.sup_model is not None:
            sup_path, sup_success = run_supervised_inference(
                st.session_state.sup_model,
                numeric,
                st.session_state.inf_robot,
                st.session_state.inf_target,
                rows, cols
            )
        else:
            right_col.warning("No supervised model loaded. Upload a .pth file in sidebar.")

        # ── animate both simultaneously ───────────────────────────────────
        max_steps = max(len(rl_path), len(sup_path))

        for i in range(max_steps):
            rl_pos  = rl_path[min(i, len(rl_path) - 1)]
            sup_pos = sup_path[min(i, len(sup_path) - 1)]

            rl_img = render_grid_image(
                numeric, rl_pos,
                st.session_state.inf_target,
                path_taken=rl_path[:i + 1],
                cell_size=cell_sz
            )
            sup_img = render_grid_image(
                numeric, sup_pos,
                st.session_state.inf_target,
                path_taken=sup_path[:i + 1],
                cell_size=cell_sz
            )

            rl_grid_display.image(image_to_bytes(rl_img),   width='stretch')
            sup_grid_display.image(image_to_bytes(sup_img), width='stretch')

            time.sleep(anim_speed / 1000)

        # ── metrics ───────────────────────────────────────────────────────
        rl_steps  = len(rl_path) - 1
        sup_steps = len(sup_path) - 1
        rl_eff    = (bfs_steps / rl_steps  * 100) if rl_steps  > 0 else 0
        sup_eff   = (bfs_steps / sup_steps * 100) if sup_steps > 0 else 0

        with left_col:
            c1, c2, c3 = st.columns(3)
            c1.metric("Steps",      str(rl_steps))
            c2.metric("BFS Opt",    str(bfs_steps))
            c3.metric("Efficiency", f"{rl_eff:.0f}%")
            st.metric("Success", "✅ Yes" if rl_success else "❌ No")
            st.metric("Reward",  f"{rl_reward:.3f}")

        with right_col:
            c1, c2, c3 = st.columns(3)
            c1.metric("Steps",      str(sup_steps))
            c2.metric("BFS Opt",    str(bfs_steps))
            c3.metric("Efficiency", f"{sup_eff:.0f}%")
            st.metric("Success", "✅ Yes" if sup_success else "❌ No")

        # ── summary ───────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 SUMMARY")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("BFS Optimal", str(bfs_steps))
        sc2.metric("RL Steps",    str(rl_steps))
        sc3.metric("Sup Steps",   str(sup_steps))

    else:
        # show grid without running
        img = render_grid_image(
            numeric,
            st.session_state.inf_robot,
            st.session_state.inf_target,
            cell_size=cell_sz
        )
        st.image(image_to_bytes(img), width='stretch',
                 caption="Generated grid — press Run Inference to start")

else:
    st.info("Press **Generate New Grid** to create a grid, then **Run Inference** to compare models.")
    st.markdown("""
    **How to use:**
    1. Upload RL model (.pth) and/or Supervised model (.pth) in sidebar
    2. Set grid size to match what the model was trained on
    3. Generate a grid
    4. Click Run Inference to watch both models navigate simultaneously
    """)