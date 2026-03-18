import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# =============================================================================
# CONSTANTS
# =============================================================================

ACTIONS = {
    0: (-1,  0),   # UP
    1: ( 1,  0),   # DOWN
    2: ( 0, -1),   # LEFT
    3: ( 0,  1),   # RIGHT
}
NUM_ACTIONS  = 4
WINDOW_SIZE  = 5
WINDOW_CELLS = WINDOW_SIZE * WINDOW_SIZE   # 25
INPUT_DIM    = WINDOW_CELLS + 3            # 28

CELL_OBSTACLE = 1.0
CELL_TARGET   = 3.0
CELL_OOB      = 0.5


# =============================================================================
# ENVIRONMENT
# =============================================================================

class GridEnvironmentRL:
    def __init__(self, numeric_grid, robot_start, target_pos):
        self.numeric_grid = numeric_grid
        self.robot_start  = robot_start
        self.target_pos   = target_pos
        self.robot_pos    = robot_start
        self.visited      = set()

    def reset(self):
        self.robot_pos = self.robot_start
        self.visited   = {self.robot_start}
        return self.robot_pos

    def step(self, action):
        dr, dc = ACTIONS[action]
        r, c   = self.robot_pos
        nr, nc = r + dr, c + dc
        rows, cols = self.numeric_grid.shape

        if not (0 <= nr < rows and 0 <= nc < cols):
            return self.robot_pos, -0.01, False
        if self.numeric_grid[nr, nc] == CELL_OBSTACLE:
            return self.robot_pos, -0.01, False

        prev_dist      = abs(r  - self.target_pos[0]) + abs(c  - self.target_pos[1])
        self.robot_pos = (nr, nc)
        new_dist       = abs(nr - self.target_pos[0]) + abs(nc - self.target_pos[1])

        if self.robot_pos == self.target_pos:
            return self.robot_pos, +1.0, True

        shaping         = (prev_dist - new_dist) * 0.01
        revisit_penalty = -0.01 if self.robot_pos in self.visited else 0.0
        self.visited.add(self.robot_pos)

        return self.robot_pos, -0.005 + shaping + revisit_penalty, False


# =============================================================================
# STATE
# =============================================================================

def get_vision_window(numeric_grid, robot_pos, target_pos):
    rows, cols = numeric_grid.shape
    r, c       = robot_pos
    half       = WINDOW_SIZE // 2
    window     = []
    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                window.append(CELL_OOB)             # out of bounds = 0.5
            elif (nr, nc) == target_pos:
                window.append(CELL_TARGET)           # target = 3.0
            elif numeric_grid[nr, nc] == CELL_OBSTACLE:
                window.append(CELL_OBSTACLE)         # obstacle = 1.0
            else:
                window.append(0.0)                   # free = 0.0 (no ghosts)
    return window


def get_state(robot_pos, numeric_grid, target_pos, rows, cols, visited):
    window   = get_vision_window(numeric_grid, robot_pos, target_pos)
    pos_r    = robot_pos[0] / rows
    pos_c    = robot_pos[1] / cols
    explored = len(visited) / (rows * cols)
    return torch.tensor(window + [pos_r, pos_c, explored],
                        dtype=torch.float32)


# =============================================================================
# MODEL
# =============================================================================

class DQN_LSTM(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=128,
                 output_dim=NUM_ACTIONS):
        super().__init__()
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
            input_size=128, hidden_size=hidden_dim,
            num_layers=1, batch_first=True
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, h, c):
        encoded         = self.encoder(x)
        lstm_in         = encoded.unsqueeze(1)
        out, (h_n, c_n) = self.lstm(lstm_in, (h, c))
        out             = out.squeeze(1)
        return self.decoder(out), h_n, c_n

    def init_hidden(self, batch_size=1, device='cpu'):
        h = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        c = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        return h, c


# =============================================================================
# REPLAY BUFFER
# =============================================================================

class ReplayBuffer:
    def __init__(self, maxlen=50000):
        self.buffer = deque(maxlen=maxlen)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch       = random.sample(self.buffer, batch_size)
        states      = torch.stack([e[0] for e in batch])
        actions     = torch.tensor([e[1] for e in batch], dtype=torch.long)
        rewards     = torch.tensor([e[2] for e in batch], dtype=torch.float32)
        next_states = torch.stack([e[3] for e in batch])
        dones       = torch.tensor([e[4] for e in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# EVALUATION — silent, used for early stopping
# =============================================================================

def evaluate_model_silent(model, rows, cols,
                           density_min=0.10,
                           density_max=0.35,
                           n_episodes=20):
    """
    Run n_episodes on random grids with random densities.
    Returns success rate (%).
    No prints — used internally during training.
    """
    from core.grid_utils import generate_random_grid

    device = next(model.parameters()).device
    model.eval()

    success_count = 0
    path_length   = (rows - 1) + (cols - 1)
    MAX_STEPS     = path_length * 3 + 5

    for _ in range(n_episodes):
        density = random.uniform(density_min, density_max)
        while True:
            result = generate_random_grid(rows, cols, density)
            _, numeric, _, robot_start, target = result
            if numeric is not None:
                break

        env    = GridEnvironmentRL(numeric, robot_start, target)
        raw_pos = env.reset()
        state  = get_state(raw_pos, env.numeric_grid, env.target_pos,
                           rows, cols, env.visited)
        h, c   = model.init_hidden(batch_size=1, device=device)
        recent = deque(maxlen=10)

        for _ in range(MAX_STEPS):
            state_t = state.unsqueeze(0).to(device)
            with torch.no_grad():
                q_vals, h, c = model(state_t, h, c)
                action = torch.argmax(q_vals).item()
            h = h.detach()
            c = c.detach()

            new_raw, _, done = env.step(action)
            state = get_state(new_raw, env.numeric_grid, env.target_pos,
                              rows, cols, env.visited)
            recent.append(env.robot_pos)
            if len(recent) == 10 and len(set(recent)) <= 2:
                break
            if done:
                success_count += 1
                break

    model.train()
    return success_count / n_episodes * 100


# =============================================================================
# TRAINING — generator, yields progress for Streamlit live updates
# =============================================================================

def train_rl_live(rows, cols,
                  density_min         = 0.10,
                  density_max         = 0.35,
                  episodes            = 5000,
                  progress_every      = 10,
                  early_stop_success  = 90.0,
                  early_stop_patience = 3,
                  eval_every          = 500):
    """
    Generator — yields training state every progress_every episodes.

    Key change from fixed-grid training:
        NEW random grid generated every episode
        density randomly chosen between density_min and density_max
        model learns general navigation, not one memorized path
        works on any grid at inference time
    """
    from core.grid_utils import generate_random_grid

    device = torch.device("cpu")

    path_length   = (rows - 1) + (cols - 1)
    MAX_STEPS     = path_length * 3 + 5
    ALPHA         = 1e-3
    GAMMA         = 0.95
    EPSILON       = 1.0
    EPSILON_MIN   = 0.01
    target_ep     = int(episodes * (2 / 3))
    EPSILON_DECAY = (EPSILON_MIN / EPSILON) ** (1 / target_ep)
    BATCH_SIZE    = 32
    MIN_BUFFER    = 500
    UPDATE_TARGET = 100
    HIDDEN_DIM    = 128

    model        = DQN_LSTM(INPUT_DIM, HIDDEN_DIM, NUM_ACTIONS).to(device)
    target_model = DQN_LSTM(INPUT_DIM, HIDDEN_DIM, NUM_ACTIONS).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer     = optim.Adam(model.parameters(), lr=ALPHA)
    loss_fn       = nn.MSELoss()
    replay_buffer = ReplayBuffer()

    rewards_history     = []
    best_avg            = -np.inf
    consecutive_success = 0
    success_rate        = None

    # these are yielded so Streamlit can display current episode grid
    current_numeric = None
    current_robot   = None
    current_target  = None

    model.train()

    for episode in range(episodes):

        # ── new random grid every episode ────────────────────────────────
        density = random.uniform(density_min, density_max)
        while True:
            result = generate_random_grid(rows, cols, density)
            _, numeric, _, robot_start, target = result
            if numeric is not None:
                break

        env = GridEnvironmentRL(numeric, robot_start, target)
        current_numeric = numeric
        current_robot   = robot_start
        current_target  = target
        # ─────────────────────────────────────────────────────────────────

        raw_pos = env.reset()
        state   = get_state(raw_pos, env.numeric_grid, env.target_pos,
                            rows, cols, env.visited)
        h, c    = model.init_hidden(batch_size=1, device=device)

        total_reward = 0
        episode_loss = 0
        train_steps  = 0
        done         = False
        last_path    = [env.robot_pos]

        for _ in range(MAX_STEPS):
            state_t = state.unsqueeze(0).to(device)

            if random.random() < EPSILON:
                action = random.randint(0, NUM_ACTIONS - 1)
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

            new_raw, reward, done = env.step(action)
            new_state    = get_state(new_raw, env.numeric_grid, env.target_pos,
                                     rows, cols, env.visited)
            total_reward += reward
            last_path.append(env.robot_pos)

            replay_buffer.push(state, action, reward, new_state, done)
            state = new_state

            if len(replay_buffer) >= MIN_BUFFER:
                s, a, r, ns, d = replay_buffer.sample(BATCH_SIZE)
                s  = s.to(device); a = a.to(device)
                r  = r.to(device); ns = ns.to(device); d = d.to(device)

                h_tr, c_tr = model.init_hidden(BATCH_SIZE, device)
                h_tg, c_tg = target_model.init_hidden(BATCH_SIZE, device)

                q_all, _, _ = model(s, h_tr, c_tr)
                current_q   = q_all.gather(1, a.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    nq, _, _ = target_model(ns, h_tg, c_tg)
                    target_q = r + GAMMA * nq.max(1)[0] * (1 - d)

                loss = loss_fn(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                episode_loss += loss.item()
                train_steps  += 1

            if done:
                break

        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
        rewards_history.append(total_reward)

        avg_reward = np.mean(rewards_history[-100:])
        avg_loss   = episode_loss / train_steps if train_steps > 0 else 0.0

        if avg_reward > best_avg:
            best_avg = avg_reward

        if (episode + 1) % UPDATE_TARGET == 0:
            target_model.load_state_dict(model.state_dict())

        # ── early stopping check ──────────────────────────────────────────
        early_stop_triggered = False

        if (episode + 1) % eval_every == 0:
            success_rate = evaluate_model_silent(
                model, rows, cols,
                density_min = density_min,
                density_max = density_max,
                n_episodes  = 20
            )
            if success_rate >= early_stop_success:
                consecutive_success += 1
                if consecutive_success >= early_stop_patience:
                    early_stop_triggered = True
            else:
                consecutive_success = 0

        # ── yield update ─────────────────────────────────────────────────
        if (episode + 1) % progress_every == 0 or early_stop_triggered:
            yield {
                "episode":             episode + 1,
                "total_episodes":      episodes,
                "avg_reward":          avg_reward,
                "best_avg":            best_avg,
                "epsilon":             EPSILON,
                "loss":                avg_loss,
                "rewards_history":     rewards_history.copy(),
                "last_path":           last_path,
                "model":               model,
                "done_training":       False,
                "success_rate":        success_rate,
                "consecutive_success": consecutive_success,
                "early_stop_patience": early_stop_patience,
                "current_numeric":     current_numeric,
                "current_robot":       current_robot,
                "current_target":      current_target,
            }

        if early_stop_triggered:
            break

    # final yield
    yield {
        "episode":             len(rewards_history),
        "total_episodes":      episodes,
        "avg_reward":          np.mean(rewards_history[-100:]),
        "best_avg":            best_avg,
        "epsilon":             EPSILON,
        "loss":                0.0,
        "rewards_history":     rewards_history,
        "last_path":           last_path,
        "model":               model,
        "done_training":       True,
        "success_rate":        success_rate,
        "consecutive_success": consecutive_success,
        "early_stop_patience": early_stop_patience,
        "current_numeric":     current_numeric,
        "current_robot":       current_robot,
        "current_target":      current_target,
    }


# =============================================================================
# INFERENCE
# =============================================================================

def run_rl_inference(model, env, rows, cols):
    """
    Run trained model on environment.
    Uses visited penalty during inference to prevent oscillation.
    Returns path, total_reward, success (bool).
    """
    device = next(model.parameters()).device
    model.eval()

    raw_pos = env.reset()
    state   = get_state(raw_pos, env.numeric_grid, env.target_pos,
                        rows, cols, env.visited)
    h, c    = model.init_hidden(batch_size=1, device=device)

    path_length  = (rows - 1) + (cols - 1)
    MAX_STEPS    = path_length * 3 + 5

    steps  = [env.robot_pos]
    total_reward = 0
    recent  = deque(maxlen=10)
    success      = False
    visited_inf  = {env.robot_pos}   # track visited for inference penalty

    for _ in range(MAX_STEPS):
        state_t = state.unsqueeze(0).to(device)
        with torch.no_grad():
            q_vals, h, c = model(state_t, h, c)

        h = h.detach()
        c = c.detach()

        # ── revisit penalty during inference ─────────────────────────────
        # prevents oscillation without retraining
        q_adjusted = q_vals.clone().squeeze(0)
        for action_idx, (dr, dc) in ACTIONS.items():
            nr = env.robot_pos[0] + dr
            nc = env.robot_pos[1] + dc
            if (nr, nc) in visited_inf:
                q_adjusted[action_idx] -= 0.5
        action = torch.argmax(q_adjusted).item()
        # ─────────────────────────────────────────────────────────────────

        new_raw, reward, done = env.step(action)
        visited_inf.add(env.robot_pos)

        state = get_state(new_raw, env.numeric_grid, env.target_pos,
                          rows, cols, env.visited)
        total_reward += reward
        steps.append(env.robot_pos)

        recent.append(env.robot_pos)
        if len(recent) == 10 and len(set(recent)) <= 2:
            break
        if done:
            success = True
            break

    return steps, total_reward, success


def load_rl_model(path, device='cpu'):
    """Load saved RL model from .pth file."""
    model = DQN_LSTM(INPUT_DIM, 128, NUM_ACTIONS).to(device)
    model.load_state_dict(torch.load(path, weights_only=True,
                                      map_location=device))
    model.eval()
    return model