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

# ==== BPTT ==== 
# BPTT = Backpropagation Through Time
# https://github.com/pytorch/tutorials/blob/gh-pages/_downloads/seq2seq_translation_tutorial.py

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
    2: (0, -1),  # LEFT
    3: (0,  1),  # RIGHT
}
NUM_ACTIONS = len(ACTIONS)

WINDOW_SIZE  = 5
WINDOW_CELLS = WINDOW_SIZE * WINDOW_SIZE   # = 25
INPUT_DIM = WINDOW_CELLS + 3            # = 28

def get_grid_numeric(grid_data):
    numeric_grid = np.zeros((len(grid_data), len(grid_data[0])), dtype=np.float32)
    for r in range(len(grid_data)):
        for c in range(len(grid_data[0])):
            numeric_grid[r, c] = CELL_TYPES.get(grid_data[r][c], 0.0)
    return numeric_grid


def find_shortest_path_bfs(grid_numeric, start_pos, target_pos):
    queue   = deque([(start_pos, [start_pos])])
    visited = {start_pos}
    rows, cols = grid_numeric.shape
    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == target_pos:
            return path
        for dr, dc in ACTIONS.values():
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and grid_numeric[nr, nc] != CELL_TYPES[3]
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
    return max(3000, int((free_cells / max_steps) * coverage * quality))


# =============================================================================
# ENVIRONMENT
# =============================================================================

class GridEnvironmentStage4:
    """
    Identical to Stage 3 environment.
    Same reward structure, same hidden target logic.
    The only improvement in Stage 4 is in the training algorithm (BPTT),
    not the environment.
    """
    def __init__(self, numeric_grid, robot_start, target_pos):
        self.numeric_grid = numeric_grid
        self.robot_start = robot_start
        self.target_pos = target_pos
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
        if self.numeric_grid[nr, nc] == CELL_TYPES[3]:
            return self.robot_pos, -0.01, False

        prev_dist = abs(r - self.target_pos[0]) + abs(c - self.target_pos[1])
        self.robot_pos = (nr, nc)
        new_dist = abs(nr - self.target_pos[0]) + abs(nc - self.target_pos[1])

        if self.robot_pos == self.target_pos:
            return self.robot_pos, +1.0, True

        shaping = (prev_dist - new_dist) * 0.01

        revisit_penalty = -0.01 if self.robot_pos in self.visited else 0.0
        self.visited.add(self.robot_pos)

        return self.robot_pos, -0.005 + shaping + revisit_penalty, False


# =============================================================================
# VISION WINDOW + STATE
# =============================================================================

def get_vision_window(numeric_grid, robot_pos, target_pos, window_size=WINDOW_SIZE):
    """
    Extract window_size × window_size area around robot.
    Target IS visible if it falls inside the window (3.0).
    Out of bounds cells treated as walls (0.5).
    """
    rows, cols = numeric_grid.shape
    r, c = robot_pos
    half = window_size // 2

    window = []
    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                window.append(0.5)
            elif (nr, nc) == target_pos:
                window.append(3.0)
            else:
                window.append(numeric_grid[nr, nc])
    return window


def get_state(robot_pos, numeric_grid, target_pos, rows, cols, visited):
    """
    State = 5×5 vision window (25) + pos_r + pos_c + explored% = 28 values
    Identical to Stage 3.
    """
    window = get_vision_window(numeric_grid, robot_pos, target_pos)
    pos_r = robot_pos[0] / rows
    pos_c = robot_pos[1] / cols
    explored = len(visited) / (rows * cols)
    return window + [pos_r, pos_c, explored]


# =============================================================================
# DQN-LSTM MODEL  (identical to Stage 3)
# =============================================================================

class DQN_LSTM(nn.Module):
    """
    Encoder → LSTM → Decoder.
    Identical architecture to Stage 3.
    The improvement in Stage 4 is in HOW we train this model (BPTT),
    not in the model itself.
    """
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
            input_size = 128,
            hidden_size = hidden_dim,
            num_layers = 1,
            batch_first = True
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, h, c):
        encoded = self.encoder(x)            # [batch, 128]
        lstm_in = encoded.unsqueeze(1)        # [batch, 1, 128]
        out, (h_n, c_n) = self.lstm(lstm_in, (h, c)) # carry memory
        out = out.squeeze(1)              # [batch, 128]
        q_vals = self.decoder(out)           # [batch, output_dim]
        return q_vals, h_n, c_n

    def init_hidden(self, batch_size=1, device='cpu'):
        h = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        c = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        return h, c


# =============================================================================
# SEQUENCE REPLAY BUFFER  (the core Stage 4 improvement)
# =============================================================================

class SequenceReplayBuffer:
    """
    Stage 3 stored individual transitions:
        (state, action, reward, next_state, done)
        hidden state reset to ZERO during training → loses temporal context

    Stage 4 stores sequences of SEQ_LEN consecutive steps:
        [(state_t,   action_t,   reward_t,   done_t,   h_t,   c_t  ),
         (state_t+1, action_t+1, reward_t+1, done_t+1, h_t+1, c_t+1),
         ...SEQ_LEN steps total]

        hidden state at sequence start is the REAL h,c from that moment
        → LSTM trained with actual temporal context
        → gradients flow back SEQ_LEN steps (BPTT)

    How sequences are built:
        - steps are added one at a time via push_step()
        - current_episode accumulates steps
        - when episode ends OR length reaches SEQ_LEN:
            slice current_episode into overlapping sequences → store in buffer
        - buffer holds maxlen sequences total (oldest deleted when full)
    """
    def __init__(self, maxlen, seq_len):
        self.buffer = deque(maxlen=maxlen)
        self.seq_len = seq_len
        self.current_episode = []    # accumulates steps for current episode

    def push_step(self, state, action, reward, done, h, c):
        self.current_episode.append((
            state,  # ← convert once here
            action, reward, done,
            h.cpu().clone(),
            c.cpu().clone()
        ))

        if done or len(self.current_episode) >= self.seq_len:
            self._store_sequences()
            if done:
                self.current_episode = []
            else:
                self.current_episode = self.current_episode[-(self.seq_len - 1):]

    def _store_sequences(self):
        """
        Slice completed episode into overlapping sequences of seq_len.

        Example: episode has 20 steps, seq_len=8
            sequence 1: steps 0-7   (h,c from step 0)
            sequence 2: steps 1-8   (h,c from step 1)
            sequence 3: steps 2-9   (h,c from step 2)
            ...
            sequence 13: steps 12-19 (h,c from step 12)

        Overlapping sequences = more training data from each episode
        Each sequence starts with the REAL hidden state from that moment
        """
        ep = self.current_episode
        if len(ep) < self.seq_len:
            # episode too short → pad with last step repeated
            while len(ep) < self.seq_len:
                ep.append(ep[-1])

        for i in range(len(ep) - self.seq_len + 1):
            self.buffer.append(ep[i:i + self.seq_len])

    def sample(self, batch_size):
        """
        Return batch_size random sequences.
        Each sequence is SEQ_LEN steps with real hidden states.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


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


def plot_rewards(rewards, rows, cols, window=100, title="Reward Curve", save_path="rewards.png"):
    optimal_steps = (rows - 1) + (cols - 1)
    theoretical_max = 1.0 + (optimal_steps * -0.005)
    realistic_max = 1.0 + (int(optimal_steps * 1.5) * -0.005)
    print("\nComparisons:")
    print(f"Theoretical max reward: {theoretical_max:.3f}")
    print(f"Realistic max reward:   {realistic_max:.3f}")

    smoothed = [
        np.mean(rewards[max(0, i - window):i + 1])
        for i in range(len(rewards))
    ]
    plt.figure(figsize=(10, 5))
    plt.plot(rewards,  alpha=0.3, color='mediumseagreen', label='Raw reward')
    plt.plot(smoothed, color='mediumseagreen', linewidth=2,
             label=f'Rolling avg (window={window})')
    plt.axhline(y=theoretical_max, color='green', linestyle='--',
                linewidth=1, label=f'Theoretical max ({theoretical_max:.3f})')
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Reward curve saved → {save_path}")
    plt.show()


# =============================================================================
# TRAINING
# =============================================================================

def train_stage4(env, rows, cols):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''
    ε=0 --> 100% exploitation
    ε=1 --> 100% exploration

    Key difference from Stage 3:
        Stage 3: individual transitions in buffer → h reset to zero in training
        Stage 4: sequences of SEQ_LEN steps      → h carried through sequence
                 gradients flow back SEQ_LEN steps (BPTT)
                 LSTM properly learns from temporal context
    '''

    # Hyperparameters
    OUTPUT_DIM = NUM_ACTIONS
    HIDDEN_DIM = 128
    SEQ_LEN = 8             # ← NEW: backprop through 8 steps
                                #   gradients flow: step t → t-1 → ... → t-7
                                #   robot learns how decisions N steps ago
                                #   led to current reward

    EPISODES = recommended_episodes(rows, cols, 0.20, target_visible=True, has_lstm=True, quality=10)
    path_length = (rows - 1) + (cols - 1)
    MAX_STEPS = path_length * 3 + 5
    ALPHA = 1e-3
    GAMMA = 0.95
    EPSILON = 1.0
    EPSILON_MIN = 0.01
    target_ep   = int(EPISODES * (2 / 3))
    EPSILON_DECAY = (EPSILON_MIN / EPSILON) ** (1 / target_ep)
    BATCH_SIZE = 32
    BUFFER_SIZE = 5000       # sequences not transitions → smaller is fine
                                     # 5000 sequences × 8 steps = 40000 transitions equiv
    MIN_BUFFER = 50        # sequences before training starts
    EVAL_EVERY = 250
    UPDATE_TARGET_EVERY = 100

    # Early stopping
    EARLY_STOP_SUCCESS  = 90.0       # % success rate
    EARLY_STOP_PATIENCE = 5          # consecutive evals above threshold
    consecutive_success = 0

    # Models — identical to Stage 3
    model = DQN_LSTM(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
    target_model = DQN_LSTM(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=ALPHA)
    loss_fn = nn.MSELoss()

    # ── SequenceReplayBuffer replaces ReplayBuffer ────────────────────────────
    seq_buffer = SequenceReplayBuffer(maxlen=BUFFER_SIZE, seq_len=SEQ_LEN)

    rewards_per_episode = []
    best_avg_reward = -np.inf

    print("\n" + "="*60)
    print("   STAGE 4 DQN-LSTM — BPTT (sequence training)")
    print("="*60)
    print(f"Device:      {device}")
    print(f"State:       5×5 window + pos + explored = {INPUT_DIM} inputs")
    print(f"Model:       Encoder → LSTM({HIDDEN_DIM}) → Decoder  [identical to Stage 3]")
    print(f"SEQ_LEN:     {SEQ_LEN} steps  ← gradients flow back {SEQ_LEN} steps")
    print(f"Buffer:      {BUFFER_SIZE} sequences × {SEQ_LEN} steps each")
    print(f"Episodes:    {EPISODES} | MAX_STEPS: {MAX_STEPS} | Batch: {BATCH_SIZE}")
    print(f"α={ALPHA} | γ={GAMMA} | ε_decay={EPSILON_DECAY:.6f}")
    print(f"Explore: 2/3 ({target_ep} ep) | Exploit: 1/3 ({EPISODES - target_ep} ep)")
    print(f"Early stop:  {EARLY_STOP_SUCCESS}% success for {EARLY_STOP_PATIENCE} evals\n")

    episode_bar = tqdm(range(EPISODES), desc="Stage4 Training", unit="ep")

    model.train()

    for episode_num, _ in enumerate(episode_bar):
        raw_pos = env.reset()
        state = torch.tensor(get_state(raw_pos, env.numeric_grid, env.target_pos, rows, cols, env.visited), dtype=torch.float32)
        h, c = model.init_hidden(batch_size=1, device=device)

        total_reward = 0
        episode_loss = 0
        train_steps  = 0
        done = False

        for _ in range(MAX_STEPS):

            state_t = state.unsqueeze(0).to(device)

            # ── Store h,c BEFORE this step ────────────────────────────────────
            # This is the hidden state the LSTM had when it saw this state
            # We store it so sequences can start with the REAL context
            h_store = h.detach().cpu().clone()
            c_store = c.detach().cpu().clone()

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
            new_state = torch.tensor(get_state(new_state_raw, env.numeric_grid, env.target_pos, rows, cols, env.visited), dtype=torch.float32)
            total_reward += reward

            # ── Push step to sequence buffer with REAL hidden state ───────────
            seq_buffer.push_step(state, action, reward, done, h_store, c_store)
            state = new_state

            # ── BPTT Training ─────────────────────────────────────────────────
            if len(seq_buffer) >= MIN_BUFFER:
                sequences = seq_buffer.sample(BATCH_SIZE)

                states_seq  = torch.stack([torch.stack([seq[t][0] for seq in sequences]) for t in range(SEQ_LEN)]).to(device) # [SEQ_LEN, BATCH, 28]
                actions_seq = torch.tensor([[seq[t][1] for seq in sequences] for t in range(SEQ_LEN)], dtype=torch.long).to(device) # [SEQ_LEN, BATCH]
                rewards_seq = torch.tensor([[seq[t][2] for seq in sequences] for t in range(SEQ_LEN)], dtype=torch.float32).to(device) # [SEQ_LEN, BATCH]
                dones_seq = torch.tensor([[seq[t][3] for seq in sequences] for t in range(SEQ_LEN)], dtype=torch.float32).to(device) # [SEQ_LEN, BATCH]
                h_batch = torch.stack([seq[0][4].squeeze() for seq in sequences]).unsqueeze(0).to(device)
                c_batch = torch.stack([seq[0][5].squeeze() for seq in sequences]).unsqueeze(0).to(device)

                # target model also starts with real hidden states
                h_tgt = h_batch.clone()
                c_tgt = c_batch.clone()

                total_loss = torch.tensor(0.0, device=device)

                # ── Loop through SEQ_LEN steps — the BPTT loop ────────────────
                for t in range(SEQ_LEN):
                    # extract step t from all sequences in batch
                    states_t = states_seq[t] # [BATCH, 28]
                    actions_t = actions_seq[t] # [BATCH]
                    rewards_t = rewards_seq[t] # [BATCH]
                    dones_t = dones_seq[t] # [BATCH]

                    # forward pass — h_batch carries memory from previous step
                    q_all, h_batch, c_batch = model(states_t, h_batch, c_batch)
                    current_q = q_all.gather(1, actions_t.unsqueeze(1)).squeeze(1) # [BATCH]

                    # target Q — use frozen target model
                    with torch.no_grad():
                        next_q_all, h_tgt, c_tgt = target_model(states_t, h_tgt, c_tgt)
                        max_next_q = next_q_all.max(1)[0]
                        target_q = rewards_t + GAMMA * max_next_q * (1 - dones_t)

                    # accumulate loss across all timesteps in sequence
                    total_loss = total_loss + loss_fn(current_q, target_q)

                    # ── Truncated BPTT ────────────────────────────────────────
                    # detach hidden state at each step
                    # gradients flow WITHIN the sequence (t=0 to SEQ_LEN-1)
                    # but NOT beyond the sequence boundary
                    # this is "truncated" BPTT — controlled gradient flow
                    h_batch = h_batch.detach()
                    c_batch = c_batch.detach()
                    h_tgt = h_tgt.detach()
                    c_tgt = c_tgt.detach()

                # average loss across sequence length
                total_loss = total_loss / SEQ_LEN

                # print(f"total_loss value: {total_loss.item():.6f}")

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                episode_loss += total_loss.item()
                train_steps += 1

            if done:
                break

        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
        rewards_per_episode.append(total_reward)

        avg_reward = np.mean(rewards_per_episode[-100:])
        avg_loss = episode_loss / train_steps if train_steps > 0 else 0.0

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/stage4_best.pth")

        episode_bar.set_postfix({
            "avg_r": f"{avg_reward:6.3f}",
            "best":  f"{best_avg_reward:6.3f}",
            "loss":  f"{avg_loss:.6f}",
            "ε":     f"{EPSILON:.3f}",
            "buf":   f"{len(seq_buffer)}",
        })

        # ── Sync target network ───────────────────────────────────────────────
        if (episode_num + 1) % UPDATE_TARGET_EVERY == 0:
            target_model.load_state_dict(model.state_dict())
            tqdm.write(f"[ep {episode_num+1}] Target network updated")

        # ── Periodic evaluation + early stopping ─────────────────────────────
        if (episode_num + 1) % EVAL_EVERY == 0:
            tqdm.write(f"\n--- Eval @ Episode {episode_num + 1} ---")
            success_rate, avg_r, avg_s = evaluate_model(model, env, rows, cols, n_episodes=20)

            tqdm.write(f"success={success_rate:.0f}% | "
                       f"avg_reward={avg_r:.3f} | "
                       f"avg_steps={avg_s:.1f}")
            model.train()

            # early stopping
            if success_rate >= EARLY_STOP_SUCCESS:
                consecutive_success += 1
                tqdm.write(f"Early stop progress: {consecutive_success}/{EARLY_STOP_PATIENCE}")

                if consecutive_success >= EARLY_STOP_PATIENCE:
                    tqdm.write(f"\n✅ Early stopping at episode {episode_num + 1}!")
                    tqdm.write(f"{success_rate:.0f}% success for {EARLY_STOP_PATIENCE} consecutive evals")
                    break
            else:
                consecutive_success = 0

    print(f"\n[Stage 4] Training complete!")
    print(f"Best avg reward:             {best_avg_reward:.3f}")
    print(f"Final avg reward (last 100): {np.mean(rewards_per_episode[-100:]):.3f}")

    if os.path.exists("models/stage4_best.pth"):
        model.load_state_dict(torch.load("models/stage4_best.pth", weights_only=True))
        print("Best model loaded.")

    return model, rewards_per_episode


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, env, rows, cols, n_episodes=100):
    """
    Run n_episodes with ε=0 (pure exploitation).
    LSTM hidden state reset at start of each episode.
    Loop detection prevents infinite oscillation.
    """
    device = next(model.parameters()).device
    model.eval()

    success_count = 0
    total_rewards = []
    total_steps = []

    path_length = (rows - 1) + (cols - 1)
    MAX_STEPS = path_length * 3 + 5

    print(f"\n--- Evaluating over {n_episodes} episodes ---")

    for ep in range(n_episodes):
        raw_pos = env.reset()
        state = get_state(raw_pos, env.numeric_grid, env.target_pos, rows, cols, env.visited)
        h, c = model.init_hidden(batch_size=1, device=device)
        total_reward = 0
        steps = 0
        recent = deque(maxlen=10)    # loop detection

        for step in range(MAX_STEPS):
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                q_vals, h, c = model(state_t, h, c)
                action = torch.argmax(q_vals).item()

            h = h.detach()
            c = c.detach()

            new_state_raw, reward, done = env.step(action)
            state = get_state(new_state_raw, env.numeric_grid, env.target_pos, rows, cols, env.visited)
            total_reward += reward
            steps = step + 1

            recent.append(env.robot_pos)
            if len(recent) == 10 and len(set(recent)) <= 2:
                break    # oscillation detected → end episode as failure

            if done:
                success_count += 1
                break

        total_rewards.append(total_reward)
        total_steps.append(steps)

    success_rate = success_count / n_episodes * 100
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)

    print(f"Success rate:  {success_rate:.1f}%   ← main metric")
    print(f"Avg reward:    {avg_reward:.3f}")
    print(f"Avg steps:     {avg_steps:.1f}")
    print(f"Best episode:  {np.max(total_rewards):.3f}")
    print(f"Worst episode: {np.min(total_rewards):.3f}")

    return success_rate, avg_reward, avg_steps


# =============================================================================
# SIMULATION
# =============================================================================

def simulate_stage4(model, env, rows, cols, max_steps=None):
    if max_steps is None:
        max_steps = ((rows - 1) + (cols - 1)) * 3 + 5

    device = next(model.parameters()).device
    raw_pos = env.reset()
    state = get_state(raw_pos, env.numeric_grid, env.target_pos, rows, cols, env.visited)
    h, c = model.init_hidden(batch_size=1, device=device)
    path_taken = [env.robot_pos]
    frames = [_build_frame(env.numeric_grid, env.robot_pos,
                               env.target_pos, path_taken)]
    done = False
    recent = deque(maxlen=10)    # loop detection

    model.eval()

    for step in range(max_steps):
        if env.robot_pos == env.target_pos:
            print(f"[Stage 4] Target reached in {step} steps!")
            break

        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            q_vals, h, c = model(state_t, h, c)
            action = torch.argmax(q_vals).item()

        h = h.detach()
        c = c.detach()

        new_state_raw, _, done = env.step(action)
        state = get_state(new_state_raw, env.numeric_grid, env.target_pos, rows, cols, env.visited)
        path_taken.append(env.robot_pos)
        frames.append(_build_frame(env.numeric_grid, env.robot_pos,
                                   env.target_pos, path_taken))

        recent.append(env.robot_pos)
        if len(recent) == 10 and len(set(recent)) <= 2:
            print(f"[Stage 4] Oscillation detected at step {step} — stopping.")
            print(f"Stuck between: {set(recent)}")
            break

        if done:
            print(f"[Stage 4] Target reached in {step + 1} steps!")
            break

    if not done:
        print("[Stage 4] Target not found.")

    return path_taken, frames


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    ROWS, COLS = 15, 15     # start small — verify BPTT works first
    OBSTACLE_DENSITY = 0.20       # then scale: 20×20, 25×25, 35×35

    print("Generating fixed grid...")
    while True:
        _, FIXED_GRID, _, ROBOT_START, TARGET = generate_random_grid(
            ROWS, COLS, obstacle_density=OBSTACLE_DENSITY
        )
        if FIXED_GRID is not None:
            break

    print(f"Robot start:  {ROBOT_START} | Target: {TARGET}")
    print(f"Grid size:    {ROWS}×{COLS} | Obstacles: {OBSTACLE_DENSITY*100:.0f}%")

    manhattan = abs(ROBOT_START[0] - TARGET[0]) + abs(ROBOT_START[1] - TARGET[1])
    episodes  = recommended_episodes(ROWS, COLS, OBSTACLE_DENSITY,
                                     target_visible=True, has_lstm=True, quality=10)
    print(f"Manhattan:    {manhattan} steps")
    print(f"Episodes:     {episodes} (auto-computed)")
    print(f"Vision:       {WINDOW_SIZE}×{WINDOW_SIZE} window, INPUT_DIM={INPUT_DIM}")

    env = GridEnvironmentStage4(FIXED_GRID, ROBOT_START, TARGET)

    os.makedirs("examples", exist_ok=True)

    model, rewards = train_stage4(env, ROWS, COLS)
    plot_rewards(rewards, ROWS, COLS,
                 title="Stage 4 DQN-LSTM — BPTT (sequence training)",
                 save_path="examples/stage4_rewards.png")

    path, frames = simulate_stage4(model, env, ROWS, COLS)
    print("Path taken:", path)
    animate_path(frames, title="Stage 4 — BPTT Robot",
                 save_path="examples/stage4_animation.gif")