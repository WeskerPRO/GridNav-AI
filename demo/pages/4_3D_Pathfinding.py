import streamlit as st                                  # the web UI framework
import sys                                               # to fix the import path
import os                                                # path helpers
import io                                                # in-memory bytes for model download
import time                                              # sleep between animation frames
import numpy as np                                       # arrays / means
import torch                                             # to save the trained model
from matplotlib import pyplot as plt                     # reward curve figure

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make 'core' importable
from core.grid_utils import render_reward_curve          # reused 2D helper (just plots a list)
from core.grid_utils_3d import (generate_random_grid_3d,  # random solvable 3D grid
                                find_shortest_path_bfs_3d,  # BFS optimal / validation
                                render_grid_3d_plotly,    # interactive voxel view
                                GRID_3D)                  # default (8,8,8)
from core.rl_model_3d import (train_rl_3d_live,          # training generator
                              run_rl_inference_3d,        # greedy rollout
                              GridEnvironmentRL3D,        # the 3D env
                              load_rl_model_3d,           # load a .pth
                              diagnose_eval_3d,           # density/failure breakdown
                              INPUT_DIM_3D, NUM_ACTIONS_3D)  # 132, 6
from core.rl_model import DQN_LSTM                        # the (shared) network class
from core.supervised_model_3d import (train_supervised_3d_live,  # supervised training generator
                                      run_supervised_inference_3d,  # greedy rollout
                                      PathPredictionResNet3D)   # Conv3d ResNet

# =============================================================================
# PAGE CONFIG + THEME  (matches the other pages)
# =============================================================================

st.set_page_config(page_title="GridNav — 3D", page_icon="🧊", layout="wide")  # browser tab + layout

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&family=Orbitron:wght@400;700&display=swap');
    .stApp { background-color: #0D0D1A; color: #E0E0FF; }
    section[data-testid="stSidebar"] { background-color: #0F0F20; border-right: 1px solid #1A1A3A; }
    h1, h2, h3 { font-family: 'Orbitron', monospace !important; }
    h1 { color: #E94560 !important; } h2 { color: #00B4D8 !important; } h3 { color: #8888AA !important; }
    p, li, label { font-family: 'JetBrains Mono', monospace; color: #AAAACC; font-size:0.85rem; }
    .stButton > button { background: linear-gradient(135deg, #E94560, #533483); color:white; border:none; border-radius:4px; font-family:'JetBrains Mono',monospace; font-weight:700; }
    [data-testid="metric-container"] { background:#1A1A2E; border:1px solid #1A1A3A; border-radius:8px; padding:12px; }
</style>
""", unsafe_allow_html=True)                              # inject the dark theme

# =============================================================================
# SIDEBAR — config
# =============================================================================

st.sidebar.markdown("## 🧊 3D CONFIG")                    # sidebar title
st.sidebar.markdown("---")
D = st.sidebar.slider("Depth (Z)",  5, 10, GRID_3D[0])    # grid depth
H = st.sidebar.slider("Height (Y)", 5, 10, GRID_3D[1])    # grid height
W = st.sidebar.slider("Width (X)",  5, 10, GRID_3D[2])    # grid width
density_min = st.sidebar.slider("Min density", 0.05, 0.25, 0.10, 0.05)  # easiest grids
density_max = st.sidebar.slider("Max density", 0.10, 0.35, 0.20, 0.05)  # hardest grids
episodes    = st.sidebar.slider("Episodes", 500, 8000, 4000, 500)       # training length
speed       = st.sidebar.slider("Update every N eps", 10, 200, 50)      # UI refresh rate (RL)
sup_epochs  = st.sidebar.slider("Supervised epochs", 5, 40, 15)         # passes over data (supervised)
sup_samples = st.sidebar.slider("Supervised samples", 500, 5000, 2000, 500)  # BFS samples (supervised)

steps_est = ((D-1)+(H-1)+(W-1)) * 3 + 5                   # MAX_STEPS the agent gets per episode
st.sidebar.info(f"{D}×{H}×{W} = {D*H*W} voxels · step budget {steps_est}\n\n"
                f"⚠️ {D*H*W} voxels is sparse — expect to need several thousand episodes.")  # honest expectation

# =============================================================================
# SESSION STATE  (survives Streamlit reruns)
# =============================================================================

ss = st.session_state                                    # short alias
ss.setdefault("g3_grid",   None)                         # current numeric grid
ss.setdefault("g3_robot",  None)                         # robot start
ss.setdefault("g3_target", None)                         # target
ss.setdefault("g3_rl_model",  None)                      # trained / loaded RL model
ss.setdefault("g3_sup_model", None)                      # trained / loaded supervised model
ss.setdefault("g3_stop",   False)                        # STOP-button flag

# =============================================================================
# HEADER + TABS
# =============================================================================

st.markdown("# 🧊 3D PATHFINDING")                        # page title
st.markdown("Navigate a robot through a **3D voxel grid** with a DQN-LSTM agent (6-connected moves).")
st.markdown("---")

tab_gen, tab_train, tab_infer = st.tabs(["🏗️ Generate", "⚡ Train", "▶️ Infer"])  # three sections

# -----------------------------------------------------------------------------
# TAB 1 — GENERATE
# -----------------------------------------------------------------------------
with tab_gen:
    st.markdown("## 🏗️ GENERATE A GRID")
    if st.button("🎲  New random grid"):                  # make a fresh solvable grid
        grid, path, robot, target = generate_random_grid_3d(D, H, W, density_max)  # BFS-checked inside
        if grid is None:                                 # too dense to solve?
            st.error("Couldn't generate a solvable grid — lower the density.")
        else:
            ss.g3_grid, ss.g3_robot, ss.g3_target = grid, robot, target  # remember it
            st.success(f"Grid ready · BFS optimal = {len(path)-1} steps")  # show difficulty

    if ss.g3_grid is not None:                            # something to show?
        fig = render_grid_3d_plotly(ss.g3_grid, ss.g3_robot, ss.g3_target)  # build the 3D figure
        st.plotly_chart(fig, width='stretch', key="gen3d")   # interactive (rotate/zoom); key avoids ID clash
    else:
        st.info("Press **New random grid** to create one. Drag to rotate the 3D view.")

# -----------------------------------------------------------------------------
# TAB 2 — TRAIN
# -----------------------------------------------------------------------------
with tab_train:
    st.markdown("## ⚡ LIVE TRAINING")
    algo = st.radio("Algorithm", ["🧠 RL (DQN-LSTM)", "📚 Supervised (Conv3d)"],  # pick what to train
                    horizontal=True, key="g3_algo")
    c1, c2, _ = st.columns([1, 1, 3])                    # button row
    start = c1.button("▶  START")                        # begin training
    if c2.button("⏹  STOP"):                              # request a stop
        ss.g3_stop = True

    left, right = st.columns([1, 1])                     # layout: live view | metrics
    with left:
        st.markdown("### 🗺️ LIVE VIEW")
        view_box = st.empty()                            # 3D grid (RL) or info (supervised)
    with right:
        st.markdown("### 📈 METRICS")
        m1, m2 = st.columns(2); m3, m4 = st.columns(2)   # 2×2 metric grid
        a_box, b_box = m1.empty(), m2.empty()
        c_box, d_box = m3.empty(), m4.empty()
        st.markdown("### 📊 CURVE")
        curve_box = st.empty()                           # reward (RL) or loss (supervised)
    prog = st.progress(0)                                # progress bar

    if start:                                            # user pressed START
        ss.g3_stop = False                               # clear any old stop

        if "RL" in algo:                                 # ── RL training ──────────────
            for tick, u in enumerate(train_rl_3d_live(D, H, W,  # consume the generator
                                      density_min=density_min, density_max=density_max,
                                      episodes=episodes, progress_every=speed)):  # tick = unique counter
                if ss.g3_stop:                           # STOP mid-run?
                    st.warning("Stopped by user."); break
                prog.progress(u["episode"] / u["total_episodes"])           # advance bar
                a_box.metric("Episode", f'{u["episode"]}/{u["total_episodes"]}')
                b_box.metric("Avg Reward", f'{u["avg_reward"]:.3f}')
                c_box.metric("Epsilon", f'{u["epsilon"]:.3f}')
                d_box.metric("Success", "—" if u["success_rate"] is None else f'{u["success_rate"]:.0f}%')
                fig = render_grid_3d_plotly(u["current_grid"], u["last_path"][-1],  # robot = end of path
                                            u["current_target"], path_taken=u["last_path"])
                view_box.plotly_chart(fig, width='stretch', key=f"train3d_{tick}")  # tick is always unique
                curve_box.pyplot(render_reward_curve(u["rewards_history"], title="3D Reward"))
                plt.close("all")                         # free matplotlib memory
                if u["done_training"]:                   # generator finished
                    ss.g3_rl_model = u["model"]          # keep the RL model
                    st.success("✅ RL training complete — saved to session.")

        else:                                            # ── Supervised training ──────
            view_box.info("Supervised learns from BFS labels — watch the loss curve.")
            density = (density_min + density_max) / 2    # single density for the dataset
            for u in train_supervised_3d_live(D, H, W, obstacle_density=density,
                                              epochs=sup_epochs, num_samples=sup_samples):
                if ss.g3_stop:                           # STOP mid-run?
                    st.warning("Stopped by user."); break
                prog.progress(u["epoch"] / u["total_epochs"])               # advance bar
                a_box.metric("Epoch", f'{u["epoch"]}/{u["total_epochs"]}')
                b_box.metric("Val Acc", f'{u["accuracy"]:.1f}%')
                c_box.metric("Train Loss", f'{u["train_loss"]:.3f}')
                d_box.metric("Val Loss", f'{u["val_loss"]:.3f}')
                fig, ax = plt.subplots(figsize=(10, 3), facecolor="#0D0D1A")  # loss curve
                ax.set_facecolor("#0D0D1A")
                ax.plot(u["train_losses"], color="#E94560", label="train")
                ax.plot(u["val_losses"],   color="#00B4D8", label="val")
                ax.tick_params(colors="#8888AA"); ax.legend(labelcolor="#8888AA")
                curve_box.pyplot(fig); plt.close("all")
                if u["done_training"]:                   # generator finished
                    ss.g3_sup_model = u["model"]         # keep the supervised model
                    st.success(f'✅ Supervised complete — val acc {u["accuracy"]:.1f}%.')

    # ── downloads ─────────────────────────────────────────────────────────────
    if ss.g3_rl_model is not None:                       # RL model download
        buf = io.BytesIO(); torch.save(ss.g3_rl_model.state_dict(), buf); buf.seek(0)
        st.download_button("⬇ Download RL model", buf, file_name="gridnav_3d_rl.pth",
                           mime="application/octet-stream", key="dl_rl")
    if ss.g3_sup_model is not None:                      # supervised model download
        buf = io.BytesIO(); torch.save(ss.g3_sup_model.state_dict(), buf); buf.seek(0)
        st.download_button("⬇ Download Supervised model", buf, file_name="gridnav_3d_sup.pth",
                           mime="application/octet-stream", key="dl_sup")

    # ── RL eval diagnostic (where do failures concentrate?) ────────────────────
    if ss.g3_rl_model is not None:                       # only if an RL model exists
        with st.expander("🔍 Diagnose RL — success / failure by density"):
            if st.button("Run diagnostic (60 grids)", key="diag_btn"):  # on demand
                with st.spinner("Evaluating on 60 random grids..."):
                    rows = diagnose_eval_3d(ss.g3_rl_model, D, H, W, density_min, density_max)
                st.table(rows)                           # bucketed success/timeout/deadlock table
                st.caption("deadlock-heavy dense rows → observability · timeout-heavy → raise step budget")

# -----------------------------------------------------------------------------
# TAB 3 — INFER
# -----------------------------------------------------------------------------
with tab_infer:
    st.markdown("## ▶️ INFERENCE")
    cu1, cu2 = st.columns(2)                             # two uploaders
    rl_up  = cu1.file_uploader("RL model (.pth)",         type=["pth"], key="g3_rl_up")
    sup_up = cu2.file_uploader("Supervised model (.pth)", type=["pth"], key="g3_sup_up")

    if rl_up is not None:                                # load an RL upload
        try:
            m = DQN_LSTM(INPUT_DIM_3D, 128, NUM_ACTIONS_3D)            # build matching net
            m.load_state_dict(torch.load(io.BytesIO(rl_up.read()), weights_only=True, map_location="cpu"))
            m.eval(); ss.g3_rl_model = m; st.success("RL model loaded.")
        except Exception as e:                           # wrong dims / bad file
            st.error(f"RL load failed: {e}")
    if sup_up is not None:                               # load a supervised upload
        try:
            m = PathPredictionResNet3D()                 # size-agnostic -> no dims needed
            m.load_state_dict(torch.load(io.BytesIO(sup_up.read()), weights_only=True, map_location="cpu"))
            m.eval(); ss.g3_sup_model = m; st.success("Supervised model loaded.")
        except Exception as e:
            st.error(f"Supervised load failed: {e}")

    if ss.g3_rl_model is None and ss.g3_sup_model is None:  # need at least one model
        st.info("Train a model in the **Train** tab, or upload a `.pth` here.")
    elif ss.g3_grid is None:                             # need a grid first
        st.info("Generate a grid in the **Generate** tab first.")
    else:
        anim = st.slider("Frame delay (ms)", 50, 500, 150, 50)  # animation speed
        if st.button("▶  Run inference"):                # go
            bfs = find_shortest_path_bfs_3d(ss.g3_grid, ss.g3_robot, ss.g3_target)  # optimal reference

            results = {}                                 # name -> (path, success)
            if ss.g3_rl_model is not None:               # run RL if available
                env = GridEnvironmentRL3D(ss.g3_grid, ss.g3_robot, ss.g3_target)
                rl_path, _, rl_ok = run_rl_inference_3d(ss.g3_rl_model, env, D, H, W)
                results["🧠 RL"] = (rl_path, rl_ok)
            if ss.g3_sup_model is not None:              # run supervised if available
                sup_path, sup_ok = run_supervised_inference_3d(
                    ss.g3_sup_model, ss.g3_grid, ss.g3_robot, ss.g3_target, D, H, W)
                results["📚 Supervised"] = (sup_path, sup_ok)

            for midx, (name, (path, ok)) in enumerate(results.items()):  # animate each solver
                st.markdown(f"### {name} — {'✅ reached' if ok else '❌ failed'}")
                box = st.empty()
                for i in range(len(path)):               # step through the path
                    fig = render_grid_3d_plotly(ss.g3_grid, path[i], ss.g3_target, path_taken=path[:i+1])
                    box.plotly_chart(fig, width='stretch', key=f"infer_{midx}_{i}")  # unique key
                    time.sleep(anim / 1000)              # pause between frames

            cols = st.columns(len(results) + 1)          # metrics row: BFS + each solver
            cols[0].metric("BFS optimal", len(bfs) - 1)  # shortest possible
            for j, (name, (path, ok)) in enumerate(results.items(), start=1):
                cols[j].metric(f"{name} steps", len(path) - 1)  # steps each took
