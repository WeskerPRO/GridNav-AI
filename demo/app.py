import streamlit as st

# =============================================================================
# PAGE CONFIG — must be first Streamlit call
# =============================================================================

st.set_page_config(
    page_title="GridNav Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS — dark industrial aesthetic
# =============================================================================

st.markdown("""
<style>
    /* Import monospace font */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&family=Orbitron:wght@400;700;900&display=swap');

    /* Global dark theme */
    .stApp {
        background-color: #0D0D1A;
        color: #E0E0FF;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0F0F20;
        border-right: 1px solid #1A1A3A;
    }

    /* Main content */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'Orbitron', monospace !important;
        color: #E94560 !important;
        letter-spacing: 2px;
    }

    h2 { color: #00B4D8 !important; }
    h3 { color: #8888AA !important; }

    /* Metrics */
    [data-testid="metric-container"] {
        background: #1A1A2E;
        border: 1px solid #1A1A3A;
        border-radius: 8px;
        padding: 12px;
    }

    [data-testid="metric-container"] label {
        color: #8888AA !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.75rem !important;
    }

    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #00B4D8 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #E94560, #533483);
        color: white;
        border: none;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4);
    }

    /* Sliders */
    .stSlider [data-baseweb="slider"] {
        color: #E94560;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background: #1A1A2E;
        border-color: #333355;
        color: #E0E0FF;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Text */
    p, li {
        font-family: 'JetBrains Mono', monospace;
        color: #AAAACC;
        font-size: 0.85rem;
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #E94560, #00B4D8);
    }

    /* Divider */
    hr {
        border-color: #1A1A3A;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #0F0F20;
        border-bottom: 1px solid #1A1A3A;
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        color: #8888AA;
        background: transparent;
        border: 1px solid #1A1A3A;
        border-radius: 4px 4px 0 0;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        color: #00B4D8 !important;
        border-color: #00B4D8 !important;
        background: #1A1A2E !important;
    }

    /* Info boxes */
    .stInfo {
        background: #1A1A2E;
        border-left: 3px solid #00B4D8;
        color: #AAAACC;
    }

    /* Code */
    code {
        background: #1A1A2E;
        color: #E94560;
        font-family: 'JetBrains Mono', monospace;
        padding: 2px 6px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================

st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem 0;">
    <h1 style="font-size:3rem; margin:0; letter-spacing:6px;">
        GRIDNAV
    </h1>
    <p style="color:#533483; font-family:'JetBrains Mono',monospace;
              font-size:0.9rem; letter-spacing:3px; margin-top:0.5rem;">
        ROBOT NAVIGATION ∙ SUPERVISED vs REINFORCEMENT LEARNING
    </p>
</div>
<hr>
""", unsafe_allow_html=True)

# =============================================================================
# NAVIGATION
# =============================================================================

st.markdown("""
<div style="text-align:center; padding: 1rem 0 2rem 0;">
    <p style="color:#8888AA; font-family:'JetBrains Mono',monospace;">
        Use the <strong style="color:#E94560;">sidebar</strong> to navigate between modes,
        or select a page below.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="background:#1A1A2E; border:1px solid #E94560;
                border-radius:8px; padding:1.5rem; text-align:center;">
        <div style="font-size:2rem;">⚡</div>
        <h3 style="color:#E94560 !important; margin:0.5rem 0;">TRAINING</h3>
        <p style="color:#8888AA; font-size:0.8rem;">
            Watch the RL agent learn in real-time.
            Live reward curves and grid animation.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background:#1A1A2E; border:1px solid #00B4D8;
                border-radius:8px; padding:1.5rem; text-align:center;">
        <div style="font-size:2rem;">🤖</div>
        <h3 style="color:#00B4D8 !important; margin:0.5rem 0;">INFERENCE</h3>
        <p style="color:#8888AA; font-size:0.8rem;">
            Load a trained model and watch it navigate.
            Compare Supervised vs RL side by side.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background:#1A1A2E; border:1px solid #533483;
                border-radius:8px; padding:1.5rem; text-align:center;">
        <div style="font-size:2rem;">🏗️</div>
        <h3 style="color:#533483 !important; margin:0.5rem 0;">GRID BUILDER</h3>
        <p style="color:#8888AA; font-size:0.8rem;">
            Draw your own grid. Place obstacles, robot, target.
            Let AI and RL solve it.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================================
# ABOUT SECTION
# =============================================================================

with st.expander("ℹ️  About GridNav", expanded=False):
    st.markdown("""
    **GridNav** explores two fundamentally different approaches to robot navigation:

    | | GridNav-AI (Supervised) | GridNav-RL (Reinforcement) |
    |---|---|---|
    | **How it learns** | Imitates BFS optimal paths | Trial and error with rewards |
    | **Needs labels?** | ✅ Yes (BFS solutions) | ❌ No |
    | **Model** | ResNet CNN | DQN-LSTM |
    | **State** | Full grid (3-channel tensor) | 5×5 vision window |
    | **Actions** | 8 directions | 4 directions |
    | **Convergence** | Fast | Slower but more general |

    **Navigate using the sidebar pages:**
    - **Training** — watch the RL agent learn from scratch
    - **Inference** — load pretrained models, compare approaches
    - **Grid Builder** — design custom grids and benchmark both models
    """)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem 0;
            color:#333355; font-family:'JetBrains Mono',monospace;
            font-size:0.7rem; letter-spacing:2px;">
    GRIDNAV ∙ SUPERVISED + REINFORCEMENT LEARNING ∙ PYTORCH
</div>
""", unsafe_allow_html=True)