import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from collections import deque
from tqdm import tqdm

# =============================================================================
# SHARED COMPONENTS
# =============================================================================

CELL_TYPES = {
    0: 0.0,
    1: 0.0,
    3: 1.0,
    "R": 2.0,
    "T": 3.0
}

ACTIONS = {
    0: (-1,  0),  # UP
    1: ( 1,  0),  # DOWN
    2: ( 0, -1),  # LEFT
    3: ( 0,  1),  # RIGHT
}

NUM_ACTIONS = len(ACTIONS)
WINDOW_SIZE  = 5
WINDOW_CELLS = WINDOW_SIZE * WINDOW_SIZE   # = 25
INPUT_DIM = WINDOW_CELLS + 5            # 25 (window) + pos_r + pos_c + tgt_r + tgt_c + explored
                                           # was 28 (window + pos_r + pos_c + explored)
                                           # now 30 — target coords added for generalization

def get_grid_numeric(grid_data):
    numeric_grid = np.zeros((len(grid_data), len(grid_data[0])), dtype=np.float32)
    for r in range(len(grid_data)):
        for c in range(len(grid_data[0])):
            numeric_grid[r, c] = CELL_TYPES.get(grid_data[r][c], 0.0)
    return numeric_grid


def find_shortest_path_bfs(grid_numeric, start_pos, target_pos):
    queue = deque([(start_pos, [start_pos])])
    visited = {start_pos}
    rows, cols = grid_numeric.shape
    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == target_pos:
            return path
        for dr, dc in ACTIONS.values():
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and grid_numeric[nr, nc] != 1.0
                    and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))
    return None


def generate_random_grid(rows, cols, obstacle_density=0.2):
    for _ in range(100):
        grid_data = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if random.random() < obstacle_density:
                    grid_data[r][c] = 3

        all_coords = [(r, c) for r in range(rows) for c in range(cols)]
        random.shuffle(all_coords)

        robot_pos = target_pos = None
        for rc in all_coords:
            if grid_data[rc[0]][rc[1]] == 0:
                robot_pos = rc
                grid_data[rc[0]][rc[1]] = "R"
                break
        for rc in all_coords:
            if grid_data[rc[0]][rc[1]] == 0 and rc != robot_pos:
                target_pos = rc
                grid_data[rc[0]][rc[1]] = "T"
                break

        if robot_pos is None or target_pos is None:
            continue

        numeric_grid = get_grid_numeric(grid_data)

        # ── strip R and T — numeric_grid must only have 0.0 and 1.0 ──────────
        # robot and target tracked separately via robot_pos / target_pos
        # 2.0 and 3.0 in numeric_grid create ghost values after robot moves
        numeric_grid[robot_pos[0], robot_pos[1]] = 0.0
        numeric_grid[target_pos[0], target_pos[1]] = 0.0
        # ─────────────────────────────────────────────────────────────────────

        path = find_shortest_path_bfs(numeric_grid, robot_pos, target_pos)
        if path:
            return grid_data, numeric_grid, path, robot_pos, target_pos

    return None, None, None, None, None


def recommended_episodes(rows, cols, obstacle_density, target_visible=True, has_lstm=False, quality=10):
    free_cells = rows * cols * (1 - obstacle_density)
    max_steps = ((rows - 1) + (cols - 1)) * 3 + 5
    base_coverage = 50 + (obstacle_density * 300)
    visibility_mult = 1.5 if target_visible else 1.0
    lstm_mult = 1.5 if has_lstm else 1.0
    coverage = base_coverage * visibility_mult * lstm_mult
    return int((free_cells / max_steps) * coverage * quality)


# =============================================================================
# ENVIRONMENT
# =============================================================================

class GridEnvironmentStage3:
    def __init__(self, numeric_grid, robot_start, target_pos):
        self.numeric_grid = numeric_grid
        self.robot_start = robot_start
        self.target_pos= target_pos
        self.robot_pos = robot_start
        self.visited = set()

    def reset(self):
        self.robot_pos = self.robot_start
        self.visited = {self.robot_start}
        return self.robot_pos

    def step(self, action):
        dr, dc = ACTIONS[action]
        r, c = self.robot_pos
        nr, nc = r + dr, c + dc
        rows, cols = self.numeric_grid.shape

        if not (0 <= nr < rows and 0 <= nc < cols):
            return self.robot_pos, -0.01, False
        if self.numeric_grid[nr, nc] == 1.0:
            return self.robot_pos, -0.01, False

        prev_dist = abs(r  - self.target_pos[0]) + abs(c  - self.target_pos[1])
        self.robot_pos = (nr, nc)
        new_dist = abs(nr - self.target_pos[0]) + abs(nc - self.target_pos[1])

        if self.robot_pos == self.target_pos:
            return self.robot_pos, +1.0, True

        shaping = (prev_dist - new_dist) * 0.01

        if self.robot_pos in self.visited:
            revisit_penalty = -0.02
        else:
            revisit_penalty = 0.0

        self.visited.add(self.robot_pos)

        return self.robot_pos, -0.005 + shaping + revisit_penalty, False


# =============================================================================
# VISION WINDOW
# =============================================================================

def get_vision_window(numeric_grid, robot_pos, target_pos, window_size=WINDOW_SIZE):
    """
    Extract window_size×window_size area around robot.

    Values:
        0.0 = free cell (including robot's own cell — no ghost values)
        0.5 = out of bounds
        1.0 = obstacle
        3.0 = target (if inside window)
    """
    rows, cols = numeric_grid.shape
    r, c = robot_pos
    half = window_size // 2

    window = []
    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                window.append(0.5)             # out of bounds
            elif (nr, nc) == target_pos:
                window.append(3.0)             # target visible
            elif numeric_grid[nr, nc] == 1.0:
                window.append(1.0)             # obstacle
            else:
                window.append(0.0)             # free (no ghosts from 2.0/3.0)
    return window


# =============================================================================
# STATE
# =============================================================================

def get_state(robot_pos, numeric_grid, target_pos, rows, cols, visited):
    """
    State = 5×5 window (25) + pos_r + pos_c + tgt_r + tgt_c + explored = 30 values

    Target coordinates added (tgt_r, tgt_c) so model can generalize
    across random grids — without them reward shaping sends contradictory
    signals on different grids and model never converges.

    INPUT_DIM = 30
    """
    window = get_vision_window(numeric_grid, robot_pos, target_pos)
    pos_r = robot_pos[0] / rows
    pos_c = robot_pos[1] / cols
    tgt_r = target_pos[0] / rows    # ← target row normalized
    tgt_c = target_pos[1] / cols    # ← target col normalized
    explored = len(visited) / (rows * cols)
    return window + [pos_r, pos_c, tgt_r, tgt_c, explored]


# =============================================================================
# DQN-LSTM MODEL
# =============================================================================

class DQN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN_LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size  = 128,
            hidden_size = hidden_dim,
            num_layers  = 1,
            batch_first = True
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, h, c):
        encoded = self.encoder(x)
        lstm_in = encoded.unsqueeze(1)
        out, (h_n, c_n) = self.lstm(lstm_in, (h, c))
        out = out.squeeze(1)
        return self.decoder(out), h_n, c_n

    def init_hidden(self, batch_size=1, device='cpu'):
        h = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        c = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        return h, c


# =============================================================================
# REPLAY BUFFER
# =============================================================================

class ReplayBuffer:
    def __init__(self, maxlen=10000):
        self.buffer = deque(maxlen=maxlen)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = torch.tensor([e[0] for e in batch], dtype=torch.float32)
        actions = torch.tensor([e[1] for e in batch], dtype=torch.long)
        rewards = torch.tensor([e[2] for e in batch], dtype=torch.float32)
        next_states = torch.tensor([e[3] for e in batch], dtype=torch.float32)
        dones = torch.tensor([e[4] for e in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# EVALUATION SET — fixed grids, equivalent to validation set
# =============================================================================

def create_eval_set(rows, cols, n_grids=50, density_min=0.10, density_max=0.35):
    """
    Generate N fixed grids for evaluation — never change during training.
    Equivalent to a fixed validation set in supervised learning.
    Success rate on this set is the true learning metric.
    """
    eval_grids = []
    while len(eval_grids) < n_grids:
        density = random.uniform(density_min, density_max)
        result  = generate_random_grid(rows, cols, density)
        _, numeric, _, robot_start, target = result

        if numeric is not None:
            eval_grids.append((numeric, robot_start, target))
    print(f"Eval set created: {n_grids} fixed grids (density {density_min:.0%}–{density_max:.0%})")
    return eval_grids


def evaluate_on_fixed_set(model, eval_grids, rows, cols):
    """
    Evaluate on fixed held-out grids.
    Returns success_rate % — equivalent to validation accuracy.

    Per-episode reward on random grids is noisy (grid difficulty varies).
    This fixed eval set gives a stable, trustworthy learning signal.
    """
    device = next(model.parameters()).device
    model.eval()

    success_count = 0
    path_length = (rows - 1) + (cols - 1)
    MAX_STEPS = path_length * 3 + 5

    for numeric, robot_start, target in eval_grids:
        env = GridEnvironmentStage3(numeric, robot_start, target)
        raw_pos = env.reset()
        state = get_state(raw_pos, env.numeric_grid, env.target_pos, rows, cols, env.visited)
        h, c  = model.init_hidden(batch_size=1, device=device)
        recent = deque(maxlen=10)

        for _ in range(MAX_STEPS):
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                q_vals, h, c = model(state_t, h, c)
                action = torch.argmax(q_vals).item()
            h = h.detach()
            c = c.detach()

            new_raw, _, done = env.step(action)
            state = get_state(new_raw, env.numeric_grid, env.target_pos, rows, cols, env.visited)
            recent.append(env.robot_pos)

            if len(recent) == 10 and len(set(recent)) <= 2:
                break

            if done:
                success_count += 1
                break

    model.train()
    return success_count / len(eval_grids) * 100


# =============================================================================
# ANIMATION HELPERS
# =============================================================================

def _build_frame(numeric_grid, robot_pos, target_pos, path_taken):
    frame = np.copy(numeric_grid)
    for p_r, p_c in path_taken[:-1]:
        frame[p_r, p_c] = 0.5
    frame[robot_pos[0],  robot_pos[1]]  = CELL_TYPES["R"]
    frame[target_pos[0], target_pos[1]] = CELL_TYPES["T"]
    return frame


def animate_path(frames, title="Robot Path", save_path="animation.gif"):
    if not frames:
        print("No frames to animate.")
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ['#FFFFFF', '#DDDDDD', '#111111', '#E74C3C', '#2ECC71']
    cmap   = plt.matplotlib.colors.ListedColormap(colors)
    bounds = [0.0, 0.25, 0.75, 1.5, 2.5, 3.5]
    norm   = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    def update(i):
        frame_data = frames[i]
        ax.clear()
        ax.imshow(frame_data, cmap=cmap, norm=norm, origin='upper',
                  extent=[0, frame_data.shape[1], frame_data.shape[0], 0])
        ax.set_xticks(np.arange(frame_data.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(frame_data.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title} — Step {i}", fontsize=13)
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor='#E74C3C', label='Robot'),
            Patch(facecolor='#2ECC71', label='Target'),
            Patch(facecolor='#111111', label='Obstacle'),
            Patch(facecolor='#DDDDDD', label='Visited'),
        ], loc='upper right', fontsize=8)

    ani = animation.FuncAnimation(fig, update, frames=range(len(frames)),
                                  repeat=False, interval=500)
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    ani.save(save_path, writer='pillow', fps=2)
    print(f"Animation saved → {save_path}")
    plt.show()


def plot_rewards(rewards, success_rates, eval_every,
                 rows, cols, window=100,
                 title="Training Curves", save_path="rewards.png"):
    """
    Plots both reward curve AND success rate curve.

    Per-episode reward: noisy (grid difficulty varies) — like batch loss
    Success rate:       smooth (fixed eval set)        — like val accuracy
    """
    optimal_steps = (rows - 1) + (cols - 1)
    theoretical_max = 1.0 + (optimal_steps * -0.005)

    smoothed = [
        np.mean(rewards[max(0, i - window):i + 1])
        for i in range(len(rewards))
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # ── top: reward curve ────────────────────────────────────────────────
    ax1.plot(rewards,  alpha=0.3, color='steelblue', label='Raw reward')
    ax1.plot(smoothed, color='steelblue', linewidth=2,
             label=f'Rolling avg (window={window})')
    ax1.axhline(y=theoretical_max, color='green', linestyle='--',
                linewidth=1, label=f'Theoretical max ({theoretical_max:.3f})')
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title(f"{title} — Reward (noisy — grid difficulty varies each episode)")
    ax1.legend()
    ax1.grid(True)

    # ── bottom: success rate — the real metric ───────────────────────────
    if success_rates:
        eval_episodes = [(i + 1) * eval_every for i in range(len(success_rates))]
        ax2.plot(eval_episodes, success_rates,
                 color='tomato', linewidth=2, marker='o', markersize=4,
                 label='Success rate on fixed eval set')
        ax2.axhline(y=90, color='green', linestyle='--', linewidth=1,
                    label='Early stop threshold (90%)')
        ax2.set_ylim(0, 105)
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Success Rate (%)")
        ax2.set_title("Success Rate on Fixed Eval Set ← TRUE learning metric (like val accuracy)")
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training curves saved → {save_path}")
    plt.show()


# =============================================================================
# TRAINING
# =============================================================================

def train_stage3(rows, cols, density_min=0.10, density_max=0.35):
    """
    Trains on NEW random grid every episode with random obstacle density.
    Uses fixed eval set for reliable success rate tracking.

    Key changes from fixed-grid version:
        1. Random grid every episode → model generalizes
        2. Target in state → stable learning signal across grids
        3. Fixed eval set → reliable success rate (like val accuracy)
        4. Save model based on success rate, not avg reward
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    OUTPUT_DIM = NUM_ACTIONS
    HIDDEN_DIM = 128
    EPISODES = max(8000, recommended_episodes(rows, cols, (density_min + density_max) / 2, target_visible=True, has_lstm=True, quality=15))
    path_length = (rows - 1) + (cols - 1)
    MAX_STEPS = path_length * 3 + 5
    ALPHA = 5e-4
    GAMMA = 0.95
    EPSILON = 1.0
    EPSILON_MIN = 0.05
    target_ep = int(EPISODES * (2 / 3)) # 2 / 3
    EPSILON_DECAY = (EPSILON_MIN / EPSILON) ** (1 / target_ep)
    BATCH_SIZE = 32
    BUFFER_SIZE = 50000
    MIN_BUFFER = 1000
    EVAL_EVERY = 250
    UPDATE_TARGET_EVERY = 100
    EARLY_STOP_SUCCESS = 90.0
    EARLY_STOP_PATIENCE = 3
    WD = 1e-5

    model = DQN_LSTM(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
    target_model = DQN_LSTM(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=ALPHA, weight_decay=WD)
    loss_fn = nn.MSELoss()
    replay_buffer = ReplayBuffer(maxlen=BUFFER_SIZE)

    # ── create fixed eval set ONCE before training ────────────────────────────
    # equivalent to fixed validation set in supervised learning
    eval_grids = create_eval_set(rows, cols, n_grids=50, density_min=density_min, density_max=density_max)

    rewards_per_episode = []
    success_rates = []        # tracked on fixed eval set
    best_success_rate   = 0.0       # save model based on this, not avg_reward
    consecutive_success = 0

    print("\n" + "="*60)
    print("   STAGE 3 DQN-LSTM — RANDOM GRIDS + FIXED EVAL SET")
    print("="*60)
    print(f"Device:      {device}")
    print(f"State:       window({WINDOW_CELLS}) + pos + target + explored = {INPUT_DIM} inputs")
    print(f"Model:       Encoder → LSTM({HIDDEN_DIM}) → Decoder")
    print(f"Grids:       NEW random grid every episode")
    print(f"Density:     {density_min:.0%} – {density_max:.0%} random each episode")
    print(f"Eval set:    20 fixed grids — success rate is true metric")
    print(f"Episodes:    {EPISODES} | MAX_STEPS: {MAX_STEPS} | Batch: {BATCH_SIZE}")
    print(f"α={ALPHA} | γ={GAMMA} | ε_decay={EPSILON_DECAY:.6f}")
    print(f"Explore: 2/3 ({target_ep} ep) | Exploit: 1/3 ({EPISODES-target_ep} ep)\n")

    episode_bar = tqdm(range(EPISODES), desc="Stage3 Training", unit="ep")

    model.train()

    for episode_num, _ in enumerate(episode_bar):

        # ── new random grid every episode ────────────────────────────────────
        density = random.uniform(density_min, density_max)
        while True:
            result = generate_random_grid(rows, cols, density)
            _, numeric, _, robot_start, target = result
            if numeric is not None:
                break

        env = GridEnvironmentStage3(numeric, robot_start, target)
        raw_pos = env.reset()
        state = get_state(raw_pos, env.numeric_grid, env.target_pos, rows, cols, env.visited)
        # ─────────────────────────────────────────────────────────────────────

        h, c = model.init_hidden(batch_size=1, device=device)
        total_reward = 0
        episode_loss = 0
        train_steps  = 0
        done = False

        for _ in range(MAX_STEPS):
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            if random.random() < EPSILON:
                action = random.randint(0, OUTPUT_DIM - 1)
                model.eval()
                with torch.no_grad():
                    _, h, c = model(state_t, h, c)
                model.train()
            else:
                model.eval()
                with torch.no_grad():
                    q_vals, h, c = model(state_t, h, c)
                    action = torch.argmax(q_vals).item()
                model.train()

            h = h.detach()
            c = c.detach()

            new_state_raw, reward, done = env.step(action)
            new_state = get_state(new_state_raw, env.numeric_grid, env.target_pos, rows, cols, env.visited)
            total_reward += reward

            replay_buffer.push(state, action, reward, new_state, done)
            state = new_state

            if len(replay_buffer) >= MIN_BUFFER:
                states_b, actions_b, rewards_b, next_states_b, dones_b = \
                    replay_buffer.sample(BATCH_SIZE)

                states_b = states_b.to(device)
                actions_b = actions_b.to(device)
                rewards_b = rewards_b.to(device)
                next_states_b = next_states_b.to(device)
                dones_b = dones_b.to(device)

                h_train,  c_train  = model.init_hidden(BATCH_SIZE, device)
                h_target, c_target = target_model.init_hidden(BATCH_SIZE, device)

                q_all, _, _ = model(states_b, h_train, c_train)
                current_q = q_all.gather(1, actions_b.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q_all, _, _ = target_model(next_states_b, h_target, c_target)
                    max_next_q = next_q_all.max(1)[0]
                    target_q = rewards_b + GAMMA * max_next_q * (1 - dones_b)

                loss = loss_fn(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                episode_loss += loss.item()
                train_steps += 1

            if done:
                break

        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
        rewards_per_episode.append(total_reward)

        avg_reward = np.mean(rewards_per_episode[-100:])
        avg_loss   = episode_loss / train_steps if train_steps > 0 else 0.0

        episode_bar.set_postfix({
            "avg_r":   f"{avg_reward:6.3f}",
            "best_sr": f"{best_success_rate:.0f}%",
            "loss":    f"{avg_loss:.6f}",
            "ε":       f"{EPSILON:.3f}",
            "buf":     f"{len(replay_buffer)}",
        })

        # ── sync target network ───────────────────────────────────────────────
        if (episode_num + 1) % UPDATE_TARGET_EVERY == 0:
            target_model.load_state_dict(model.state_dict())
            tqdm.write(f"[ep {episode_num+1}] Target network updated")

        # ── evaluate on fixed set — the real metric ───────────────────────────
        if (episode_num + 1) % EVAL_EVERY == 0:
            success_rate = evaluate_on_fixed_set(model, eval_grids, rows, cols)
            success_rates.append(success_rate)

            tqdm.write(f"\n[ep {episode_num+1}] Eval success: {success_rate:.0f}%"
                       f"  ← val accuracy equivalent")

            # save model based on success rate, not avg_reward
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), "models/stage3_best.pth")
                tqdm.write(f"  → New best! Model saved ({best_success_rate:.0f}%)")

            # early stopping
            if success_rate >= EARLY_STOP_SUCCESS:
                consecutive_success += 1
                tqdm.write(f"Early stop: {consecutive_success}/{EARLY_STOP_PATIENCE}")
                if consecutive_success >= EARLY_STOP_PATIENCE:
                    tqdm.write(f"\n✅ Early stopping at episode {episode_num+1}!")
                    break
            else:
                consecutive_success = 0

    print(f"\n[Stage 3] Training complete!")
    print(f"Best success rate: {best_success_rate:.0f}%")
    print(f"Final avg reward (last 100): {np.mean(rewards_per_episode[-100:]):.3f}")

    if os.path.exists("models/stage3_best.pth"):
        model.load_state_dict(torch.load("models/stage3_best.pth", weights_only=True))
        print("Best model loaded.")

    return model, rewards_per_episode, success_rates


# =============================================================================
# EVALUATION (full, with prints)
# =============================================================================

def evaluate_model(model, eval_grids, rows, cols, n_episodes=None):
    """
    Full evaluation with prints.
    Uses fixed eval set for consistent results.
    """
    if n_episodes is not None:
        # use random grids if n_episodes specified separately
        grids_to_eval = []
        density = 0.20
        while len(grids_to_eval) < n_episodes:
            result = generate_random_grid(rows, cols, density)
            _, numeric, _, robot_start, target = result
            if numeric is not None:
                grids_to_eval.append((numeric, robot_start, target))
    else:
        grids_to_eval = eval_grids

    device        = next(model.parameters()).device
    model.eval()

    success_count = 0
    total_rewards = []
    total_steps = []
    path_length = (rows - 1) + (cols - 1)
    MAX_STEPS = path_length * 3 + 5

    print(f"\n--- Evaluating over {len(grids_to_eval)} grids ---")

    for numeric, robot_start, target in grids_to_eval:
        env = GridEnvironmentStage3(numeric, robot_start, target)
        raw_pos = env.reset()
        state  = get_state(raw_pos, env.numeric_grid, env.target_pos,
                           rows, cols, env.visited)
        h, c = model.init_hidden(batch_size=1, device=device)
        total_reward = 0
        steps = 0
        recent = deque(maxlen=10)

        for step in range(MAX_STEPS):
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                q_vals, h, c = model(state_t, h, c)
                action = torch.argmax(q_vals).item()
            h = h.detach()
            c = c.detach()

            new_raw, reward, done = env.step(action)
            state = get_state(new_raw, env.numeric_grid, env.target_pos, rows, cols, env.visited)
            total_reward += reward
            steps = step + 1
            recent.append(env.robot_pos)

            if len(recent) == 10 and len(set(recent)) <= 2:
                break

            if done:
                success_count += 1
                break

        total_rewards.append(total_reward)
        total_steps.append(steps)

    model.train()

    success_rate = success_count / len(grids_to_eval) * 100
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)

    print(f"Success rate:  {success_rate:.1f}%   ← main metric (val accuracy equivalent)")
    print(f"Avg reward:    {avg_reward:.3f}")
    print(f"Avg steps:     {avg_steps:.1f}")
    print(f"Best episode:  {np.max(total_rewards):.3f}")
    print(f"Worst episode: {np.min(total_rewards):.3f}")

    return success_rate, avg_reward, avg_steps


# =============================================================================
# SIMULATION
# =============================================================================

def simulate_stage3(model, env, rows, cols, max_steps=None):
    if max_steps is None:
        max_steps = ((rows - 1) + (cols - 1)) * 3 + 5

    device = next(model.parameters()).device
    raw_pos = env.reset()
    state = get_state(raw_pos, env.numeric_grid, env.target_pos, rows, cols, env.visited)
    h, c = model.init_hidden(batch_size=1, device=device)
    path_taken = [env.robot_pos]
    frames = [_build_frame(env.numeric_grid, env.robot_pos, env.target_pos, path_taken)]
    done = False
    recent = deque(maxlen=10)
    model.eval()

    for step in range(max_steps):
        if env.robot_pos == env.target_pos:
            print(f"[Stage 3] Target reached in {step} steps!")
            break

        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            q_vals, h, c = model(state_t, h, c)

        h = h.detach()
        c = c.detach()

        # inference revisit penalty — prevents oscillation
        q_adjusted = q_vals.clone().squeeze(0)
        for action_idx, (dr, dc) in ACTIONS.items():
            nr = env.robot_pos[0] + dr
            nc = env.robot_pos[1] + dc
            if (nr, nc) in env.visited:
                q_adjusted[action_idx] -= 0.5
        action = torch.argmax(q_adjusted).item()

        new_state_raw, _, done = env.step(action)
        state = get_state(new_state_raw, env.numeric_grid, env.target_pos, rows, cols, env.visited)
        path_taken.append(env.robot_pos)
        frames.append(_build_frame(env.numeric_grid, env.robot_pos, env.target_pos, path_taken))

        recent.append(env.robot_pos)
        if len(recent) == 10 and len(set(recent)) <= 2:
            print(f"[Stage 3] Oscillation at step {step} — stopping.")
            break

        if done:
            print(f"[Stage 3] Target reached in {step + 1} steps!")
            break

    if not done:
        print("[Stage 3] Max steps reached. Target not found.")

    return path_taken, frames


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    ROWS, COLS = 15, 15
    DENSITY_MIN = 0.10
    DENSITY_MAX = 0.35
    N_SIM = 5
    
    print(f"Grid: {ROWS}×{COLS} | Density: {DENSITY_MIN:.0%}–{DENSITY_MAX:.0%}")
    print(f"INPUT_DIM: {INPUT_DIM} (window={WINDOW_CELLS} + pos=2 + target=2 + explored=1)")

    os.makedirs("examples", exist_ok=True)
    model, rewards, success_rates = train_stage3(ROWS, COLS, density_min = DENSITY_MIN, density_max = DENSITY_MAX)
    plot_rewards(rewards, success_rates, eval_every=250, rows=ROWS, cols=COLS, title="Stage 3 DQN-LSTM — Random Grids", save_path="examples/stage3_rewards.png")

    # simulate on a fresh random grid
    print(f"\nSimulating on {N_SIM} random grids...")
    success_count = 0

    for sim_idx in range(N_SIM):
        density = random.uniform(DENSITY_MIN, DENSITY_MAX)
        while True:
            result = generate_random_grid(ROWS, COLS, density)
            _, numeric, _, robot_start, target = result
            if numeric is not None:
                break

        print(f"\n{'='*50}")
        print(f"Simulation {sim_idx + 1}/{N_SIM}")
        print(f"Robot={robot_start} | Target={target} | Density={density:.0%}")

        env   = GridEnvironmentStage3(numeric, robot_start, target)
        path, frames = simulate_stage3(model, env, ROWS, COLS)
        print(f"Path length: {len(path) - 1} steps")

        reached = (env.robot_pos == env.target_pos)
        if reached:
            success_count += 1
            print("Result: ✅ Target reached")
        else:
            print("Result: ❌ Target not reached")

        animate_path(
            frames,
            title=f"Stage 3 — Sim {sim_idx+1} | robot={robot_start} target={target}",
            save_path=f"examples/stage3_sim_{sim_idx + 1}.gif"
        )

    print(f"\n{'='*50}")
    print(f"Simulation results: {success_count}/{N_SIM} targets reached")
    print(f"Success rate:       {success_count / N_SIM * 100:.0f}%")