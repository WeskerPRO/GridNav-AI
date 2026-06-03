import streamlit as st                                  # the web UI
import sys                                               # fix import path
import os                                                # path helpers
import time                                              # sleep between animation frames
import io                                                # read uploaded model bytes
import numpy as np                                       # (used indirectly)
import torch                                             # load model weights

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make 'core' importable
from core.grid_utils import (generate_random_grid, render_grid_image,           # grid + viz
                              find_shortest_path_bfs, image_to_bytes, ACTIONS_4)  # BFS + 4-moves
from core.rl_model import (GridEnvironmentRL, run_rl_inference,                  # RL pieces
                            DQN_LSTM, INPUT_DIM, NUM_ACTIONS)
from core.supervised_model import (run_supervised_inference,                    # supervised pieces
                                    PathPredictionResNet)

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(                                      # tab + layout
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
""", unsafe_allow_html=True)                             # inject dark theme

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.markdown("## 🤖 INFERENCE CONFIG")            # sidebar header
st.sidebar.markdown("---")

rows       = st.sidebar.slider("Grid rows",           8,  25, 15)      # grid height
cols       = st.sidebar.slider("Grid cols",           8,  25, 15)      # grid width
density    = st.sidebar.slider("Obstacle density", 0.10, 0.40, 0.20, 0.05)  # how cluttered
anim_speed = st.sidebar.slider("Animation delay (ms)", 50, 500, 150, 50)    # frame pause

st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 LOAD MODELS")

rl_file  = st.sidebar.file_uploader("RL Model (.pth)",          # upload an RL checkpoint
                                     type=["pth"], key="rl_upload")
sup_file = st.sidebar.file_uploader("Supervised Model (.pth)",  # upload a supervised checkpoint
                                     type=["pth"], key="sup_upload")

st.sidebar.markdown("---")
st.sidebar.info(                                         # reminder about size matching
    "RL model must be tested on the **same grid size** it was trained on. "
    "Match rows/cols to your training config."
)

# =============================================================================
# SESSION STATE
# =============================================================================

if "inf_numeric" not in st.session_state: st.session_state.inf_numeric = None  # current grid
if "inf_robot"   not in st.session_state: st.session_state.inf_robot   = None  # robot start
if "inf_target"  not in st.session_state: st.session_state.inf_target  = None  # target
if "rl_model"    not in st.session_state: st.session_state.rl_model    = None  # loaded RL model
if "sup_model"   not in st.session_state: st.session_state.sup_model   = None  # loaded supervised model

# =============================================================================
# LOAD MODELS
# =============================================================================

if rl_file is not None:                                  # an RL file was uploaded
    try:
        buf   = io.BytesIO(rl_file.read())               # bytes -> buffer
        model = DQN_LSTM(INPUT_DIM, 128, NUM_ACTIONS)    # build matching net
        model.load_state_dict(torch.load(buf, weights_only=True, map_location='cpu'))  # load weights
        model.eval()
        st.session_state.rl_model = model                # remember it
        st.sidebar.success("✅ RL model loaded")
    except Exception as e:                               # dim mismatch / bad file
        st.sidebar.error(f"Error loading RL model: {e}")

if sup_file is not None:                                 # a supervised file was uploaded
    try:
        buf   = io.BytesIO(sup_file.read())
        model = PathPredictionResNet(rows, cols)         # size-agnostic ResNet
        model.load_state_dict(torch.load(buf, weights_only=True, map_location='cpu'))
        model.eval()
        st.session_state.sup_model = model
        st.sidebar.success("✅ Supervised model loaded")
    except Exception as e:
        st.sidebar.error(f"Error loading supervised model: {e}")

# =============================================================================
# MAIN
# =============================================================================

st.markdown("# 🤖 INFERENCE")                            # page title
st.markdown("Load trained models and watch them navigate. Compare RL vs Supervised side by side.")
st.markdown("---")

col_gen, col_run = st.columns([1, 1])                    # two action buttons
gen_btn = col_gen.button("🎲  Generate New Grid")         # make a grid
run_btn = col_run.button("▶  Run Inference")             # run the models

# ── generate grid ─────────────────────────────────────────────────────────────
if gen_btn:                                              # generate pressed
    result = generate_random_grid(rows, cols, density)   # solvable random grid
    _, numeric, _, robot_start, target = result
    if numeric is not None:
        st.session_state.inf_numeric = numeric           # stash grid
        st.session_state.inf_robot   = robot_start
        st.session_state.inf_target  = target
    else:
        st.error("Could not generate grid. Try lower obstacle density.")

# ── show grid if exists ───────────────────────────────────────────────────────
if st.session_state.inf_numeric is not None:             # we have a grid
    numeric = st.session_state.inf_numeric
    cell_sz = max(16, 400 // max(rows, cols))            # px per cell

    if run_btn:                                          # run pressed
        # ── validate path exists ──────────────────────────────────────────
        bfs_path = find_shortest_path_bfs(               # is it solvable at all?
            numeric,
            st.session_state.inf_robot,
            st.session_state.inf_target,
            ACTIONS_4
        )
        if bfs_path is None:                             # no path -> abort
            st.error("❌ No valid path exists on this grid. Generate a new one.")
            st.stop()

        bfs_steps = len(bfs_path) - 1                    # optimal step count

        st.markdown("---")
        st.markdown("## COMPARISON")

        left_col, right_col = st.columns(2)              # RL | Supervised

        with left_col:
            st.markdown("### 🧠 GridNav-RL (Reinforcement)")
            rl_grid_display = st.empty()                 # RL animation slot

        with right_col:
            st.markdown("### 📚 GridNav-AI (Supervised)")
            sup_grid_display = st.empty()                # supervised animation slot

        # ── run RL inference ──────────────────────────────────────────────
        rl_path    = [st.session_state.inf_robot]        # default path = just start
        rl_success = False
        rl_reward  = 0

        if st.session_state.rl_model is not None:        # have an RL model?
            env = GridEnvironmentRL(                     # build env on this grid
                numeric,
                st.session_state.inf_robot,
                st.session_state.inf_target
            )
            rl_path, rl_reward, rl_success = run_rl_inference(  # greedy rollout
                st.session_state.rl_model, env, rows, cols
            )
        else:
            left_col.warning("No RL model loaded. Upload a .pth file in sidebar.")

        # ── run Supervised inference ──────────────────────────────────────
        sup_path    = [st.session_state.inf_robot]       # default path
        sup_success = False

        if st.session_state.sup_model is not None:       # have a supervised model?
            sup_path, sup_success = run_supervised_inference(  # rollout
                st.session_state.sup_model,
                numeric,
                st.session_state.inf_robot,
                st.session_state.inf_target,
                rows, cols
            )
        else:
            right_col.warning("No supervised model loaded. Upload a .pth file in sidebar.")

        # ── animate both simultaneously ───────────────────────────────────
        max_steps = max(len(rl_path), len(sup_path))     # longest of the two paths

        for i in range(max_steps):                       # step both animations together
            rl_pos  = rl_path[min(i, len(rl_path) - 1)]  # clamp to last frame when done
            sup_pos = sup_path[min(i, len(sup_path) - 1)]

            rl_img = render_grid_image(                  # draw RL frame
                numeric, rl_pos,
                st.session_state.inf_target,
                path_taken=rl_path[:i + 1],
                cell_size=cell_sz
            )
            sup_img = render_grid_image(                 # draw supervised frame
                numeric, sup_pos,
                st.session_state.inf_target,
                path_taken=sup_path[:i + 1],
                cell_size=cell_sz
            )

            rl_grid_display.image(image_to_bytes(rl_img),   width='stretch')  # show frames
            sup_grid_display.image(image_to_bytes(sup_img), width='stretch')

            time.sleep(anim_speed / 1000)                # pause between frames

        # ── metrics ───────────────────────────────────────────────────────
        rl_steps  = len(rl_path) - 1                     # moves taken
        sup_steps = len(sup_path) - 1
        rl_eff    = (bfs_steps / rl_steps  * 100) if rl_steps  > 0 else 0  # vs optimal %
        sup_eff   = (bfs_steps / sup_steps * 100) if sup_steps > 0 else 0

        with left_col:                                   # RL metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Steps",      str(rl_steps))
            c2.metric("BFS Opt",    str(bfs_steps))
            c3.metric("Efficiency", f"{rl_eff:.0f}%")
            st.metric("Success", "✅ Yes" if rl_success else "❌ No")
            st.metric("Reward",  f"{rl_reward:.3f}")

        with right_col:                                  # supervised metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Steps",      str(sup_steps))
            c2.metric("BFS Opt",    str(bfs_steps))
            c3.metric("Efficiency", f"{sup_eff:.0f}%")
            st.metric("Success", "✅ Yes" if sup_success else "❌ No")

        # ── summary ───────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 SUMMARY")
        sc1, sc2, sc3 = st.columns(3)                    # side-by-side step counts
        sc1.metric("BFS Optimal", str(bfs_steps))
        sc2.metric("RL Steps",    str(rl_steps))
        sc3.metric("Sup Steps",   str(sup_steps))

    else:
        # show grid without running
        img = render_grid_image(                         # static grid preview
            numeric,
            st.session_state.inf_robot,
            st.session_state.inf_target,
            cell_size=cell_sz
        )
        st.image(image_to_bytes(img), width='stretch',
                 caption="Generated grid — press Run Inference to start")

else:                                                    # no grid yet
    st.info("Press **Generate New Grid** to create a grid, then **Run Inference** to compare models.")
    st.markdown("""
    **How to use:**
    1. Upload RL model (.pth) and/or Supervised model (.pth) in sidebar
    2. Set grid size to match what the model was trained on
    3. Generate a grid
    4. Click Run Inference to watch both models navigate simultaneously
    """)
