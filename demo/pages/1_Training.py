from matplotlib import pyplot as plt
import streamlit as st
import sys
import os
import io
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.grid_utils import render_grid_image, render_reward_curve, image_to_bytes
from core.rl_model import train_rl_live
from core.supervised_model import train_supervised_live

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="GridNav — Training",
    page_icon="⚡",
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
    p, li, label { font-family: 'JetBrains Mono', monospace; color: #AAAACC; font-size:0.85rem; }
    .stButton > button { background: linear-gradient(135deg, #E94560, #533483); color:white; border:none; border-radius:4px; font-family:'JetBrains Mono',monospace; font-weight:700; letter-spacing:1px; }
    [data-testid="metric-container"] { background:#1A1A2E; border:1px solid #1A1A3A; border-radius:8px; padding:12px; }
    [data-testid="metric-container"] label { color:#8888AA !important; font-family:'JetBrains Mono',monospace !important; font-size:0.75rem !important; }
    [data-testid="metric-container"] [data-testid="metric-value"] { color:#00B4D8 !important; font-family:'JetBrains Mono',monospace !important; }
    .stProgress > div > div > div { background: linear-gradient(90deg, #E94560, #00B4D8); }
    .stSlider label { font-family:'JetBrains Mono',monospace; color:#8888AA; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR — training config
# =============================================================================

st.sidebar.markdown("## ⚡ TRAINING CONFIG")
st.sidebar.markdown("---")

# ── algorithm selector ────────────────────────────────────────────────────────
train_mode = st.sidebar.radio(
    "Algorithm:",
    ["🤖 RL (DQN-LSTM)", "📚 Supervised (ResNet)"]
)

st.sidebar.markdown("---")

# ── shared config ─────────────────────────────────────────────────────────────
rows = st.sidebar.slider("Grid rows", 8,  25, 15)
cols = st.sidebar.slider("Grid cols", 8,  25, 15)

# ── mode-specific config ──────────────────────────────────────────────────────
if "RL" in train_mode:
    density_min = st.sidebar.slider("Min obstacle density", 0.05, 0.30, 0.10, 0.05)
    density_max = st.sidebar.slider("Max obstacle density", 0.15, 0.45, 0.35, 0.05)
    episodes    = st.sidebar.slider("Episodes", 1000, 8000, 5000, 500)
    speed       = st.sidebar.slider("Update every N episodes", 5, 100, 20)
    st.sidebar.markdown(f"""
    <p style="color:#533483; font-size:0.75rem;">
    DQN-LSTM agent:<br>
    • 5×5 vision window<br>
    • Random grids every episode<br>
    • Density: {density_min:.0%} – {density_max:.0%}<br>
    • Reward shaping + target network<br>
    • Early stopping at 90% success
    </p>
    """, unsafe_allow_html=True)
else:
    density     = st.sidebar.slider("Obstacle density", 0.10, 0.40, 0.20, 0.05)
    epochs      = st.sidebar.slider("Epochs",           5,   50,  20)
    num_samples = st.sidebar.slider("Training samples", 1000, 10000, 3000, 500)
    st.sidebar.markdown("""
    <p style="color:#533483; font-size:0.75rem;">
    ResNet supervised:<br>
    • Imitates BFS optimal paths<br>
    • 3-channel grid input<br>
    • 8 directional actions<br>
    • CrossEntropyLoss + Dropout
    </p>
    """, unsafe_allow_html=True)

# =============================================================================
# MAIN PAGE
# =============================================================================

st.markdown("# ⚡ LIVE TRAINING")
if "RL" in train_mode:
    st.markdown(
        f"Watch the **RL agent** learn to navigate from scratch using DQN-LSTM. "
        f"Training on random **{rows}×{cols} grids** with obstacle density "
        f"**{density_min:.0%}–{density_max:.0%}**."
    )
else:
    st.markdown(
        "Train the **ResNet** to imitate BFS optimal paths via supervised learning."
    )
st.markdown("---")

# ── layout ────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1])

with left_col:
    if "RL" in train_mode:
        st.markdown("## 🗺️ CURRENT EPISODE GRID")
    else:
        st.markdown("## 📉 LOSS CURVE")
    grid_display  = st.empty()
    st.markdown("## 📊 REWARD CURVE" if "RL" in train_mode else "")
    curve_display = st.empty()

with right_col:
    st.markdown("## 📈 METRICS")
    m1, m2 = st.columns(2)
    m3, m4 = st.columns(2)
    metric_episode = m1.empty()
    metric_reward  = m2.empty()
    metric_epsilon = m3.empty()
    metric_loss    = m4.empty()
    st.markdown("---")
    st.markdown("## 📋 STATUS LOG")
    status_log = st.empty()

progress_bar = st.progress(0)

# =============================================================================
# CONTROLS
# =============================================================================

ctrl1, ctrl2, _ = st.columns([1, 1, 3])
start_btn = ctrl1.button("▶  START TRAINING")
stop_btn  = ctrl2.button("⏹  STOP")

if "training_active" not in st.session_state:
    st.session_state.training_active = False
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None

if stop_btn:
    st.session_state.stop_requested = True

# =============================================================================
# TRAINING LOOP
# =============================================================================

if start_btn:
    st.session_state.stop_requested  = False
    st.session_state.training_active = True
    st.session_state.trained_model   = None

    log_lines = []

    # =========================================================================
    # RL TRAINING
    # =========================================================================
    if "RL" in train_mode:

        cell_sz = max(16, 400 // max(rows, cols))

        trainer = train_rl_live(
            rows            = rows,
            cols            = cols,
            density_min     = density_min,
            density_max     = density_max,
            episodes        = episodes,
            progress_every  = speed,
        )

        for update in trainer:
            if st.session_state.stop_requested:
                log_lines.append("⏹ Training stopped by user.")
                status_log.markdown("```\n" + "\n".join(log_lines[-8:]) + "\n```")
                break

            ep      = update["episode"]
            total   = update["total_episodes"]
            avg_r   = update["avg_reward"]
            best_r  = update["best_avg"]
            eps     = update["epsilon"]
            loss    = update["loss"]
            path    = update["last_path"]
            rewards = update["rewards_history"]
            model   = update["model"]
            sr      = update["success_rate"]
            cs      = update["consecutive_success"]
            pat     = update["early_stop_patience"]

            cur_numeric = update["current_numeric"]
            cur_robot   = update["current_robot"]
            cur_target  = update["current_target"]

            # progress
            progress_bar.progress(ep / total)

            # metrics
            metric_episode.metric("Episode",   f"{ep} / {total}")
            metric_reward.metric("Avg Reward", f"{avg_r:.3f}")
            metric_epsilon.metric("Epsilon ε", f"{eps:.3f}")
            metric_loss.metric("Loss",         f"{loss:.6f}")

            # grid — shows current episode's random grid
            if cur_numeric is not None:
                img = render_grid_image(
                    cur_numeric,
                    path[-1] if path else cur_robot,
                    cur_target,
                    path_taken = path,
                    cell_size  = cell_sz
                )
                grid_display.image(image_to_bytes(img), width='stretch')

            # reward curve
            fig = render_reward_curve(rewards, title=f"RL Reward — Episode {ep}")
            curve_display.pyplot(fig)
            plt.close(fig)

            # log
            if sr is not None:
                log_lines.append(
                    f"📊 eval ep {ep}: success={sr:.0f}% | "
                    f"early_stop {cs}/{pat}"
                )
            if ep % (speed * 5) == 0:
                log_lines.append(
                    f"ep {ep:>5} | avg_r={avg_r:>7.3f} | "
                    f"ε={eps:.3f} | loss={loss:.6f}"
                )
            status_log.markdown("```\n" + "\n".join(log_lines[-10:]) + "\n```")

            if update["done_training"]:
                if cs >= pat and sr is not None:
                    msg = (f"✅ Early stopped at ep {ep}! "
                           f"success={sr:.0f}% for {pat} evals")
                else:
                    msg = f"✅ RL Training complete! Best avg reward: {best_r:.3f}"
                log_lines.append(msg)
                status_log.markdown("```\n" + "\n".join(log_lines[-10:]) + "\n```")
                st.session_state.trained_model = model
                st.success(msg)
                break

    # =========================================================================
    # SUPERVISED TRAINING
    # =========================================================================
    else:

        grid_display.info(
            "Supervised training generates random grids internally "
            "and learns to imitate BFS optimal paths.\n\n"
            "No live grid animation — watch the loss curve instead."
        )

        with st.spinner(
            f"Generating {num_samples} training samples from BFS... "
            "(this may take 30-60 seconds)"
        ):
            trainer = train_supervised_live(
                rows,
                cols,
                obstacle_density = density,
                epochs           = epochs,
                num_samples      = num_samples,
                progress_every   = 1
            )

        for update in trainer:
            if st.session_state.stop_requested:
                log_lines.append("⏹ Training stopped by user.")
                status_log.markdown("```\n" + "\n".join(log_lines[-8:]) + "\n```")
                break

            ep       = update["epoch"]
            total    = update["total_epochs"]
            t_loss   = update["train_loss"]
            v_loss   = update["val_loss"]
            accuracy = update["accuracy"]
            best_v   = update["best_val"]
            model    = update["model"]

            # progress
            progress_bar.progress(ep / total)

            # metrics
            metric_episode.metric("Epoch",        f"{ep} / {total}")
            metric_reward.metric("Val Accuracy",  f"{accuracy:.1f}%")
            metric_epsilon.metric("Train Loss",   f"{t_loss:.4f}")
            metric_loss.metric("Val Loss",        f"{v_loss:.4f}")

            # loss curve
            fig, ax = plt.subplots(figsize=(10, 3), facecolor='#0D0D1A')
            ax.set_facecolor('#0D0D1A')
            ax.plot(update["train_losses"], color='#E94560',
                    linewidth=2, label='Train Loss')
            ax.plot(update["val_losses"],   color='#00B4D8',
                    linewidth=2, label='Val Loss')
            ax.set_xlabel("Epoch",  color='#8888AA')
            ax.set_ylabel("Loss",   color='#8888AA')
            ax.set_title("Supervised — Loss Curve",
                         color='#E0E0FF', fontfamily='monospace')
            ax.tick_params(colors='#8888AA')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#333355')
            ax.spines['left'].set_color('#333355')
            ax.legend(facecolor='#1A1A2E', labelcolor='#8888AA')
            plt.tight_layout()
            curve_display.pyplot(fig)
            plt.close(fig)

            # log
            log_lines.append(
                f"epoch {ep:>3} | train={t_loss:.4f} | "
                f"val={v_loss:.4f} | acc={accuracy:.1f}%"
            )
            status_log.markdown("```\n" + "\n".join(log_lines[-10:]) + "\n```")

            if update["done_training"]:
                msg = (
                    f"✅ Supervised training complete! "
                    f"Val accuracy: {accuracy:.1f}% | "
                    f"Best val loss: {best_v:.4f}"
                )
                log_lines.append(msg)
                status_log.markdown("```\n" + "\n".join(log_lines[-10:]) + "\n```")
                st.session_state.trained_model = model
                st.success(msg)
                break

    st.session_state.training_active = False

# =============================================================================
# SAVE MODEL
# =============================================================================

if st.session_state.trained_model is not None:
    st.markdown("---")
    st.markdown("### 💾 SAVE MODEL")

    is_rl = isinstance(
        st.session_state.trained_model,
        __import__('core.rl_model', fromlist=['DQN_LSTM']).DQN_LSTM
    )
    filename = "gridnav_rl_trained.pth" if is_rl else "gridnav_supervised_trained.pth"

    buf = io.BytesIO()
    torch.save(st.session_state.trained_model.state_dict(), buf)
    buf.seek(0)
    st.download_button(
        label     = f"⬇ Download trained model ({filename})",
        data      = buf,
        file_name = filename,
        mime      = "application/octet-stream"
    )
    if is_rl:
        st.info(
            f"RL model trained on {rows}×{cols} grids with "
            f"{density_min:.0%}–{density_max:.0%} obstacle density. "
            f"Works on any grid of this size."
        )
    else:
        st.info(
            f"Supervised model trained on {rows}×{cols} grids. "
            f"Upload to Inference page to compare with RL."
        )