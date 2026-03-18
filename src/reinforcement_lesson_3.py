import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from collections import deque
from tqdm import tqdm
# =============================================================================
# SHARED COMPONENTS (same in both Q-table and DQN)
# =============================================================================

CELL_TYPES = {
    0: 0.0,
    1: 0.0,
    3: 1.0,
    "R": 2.0,
    "T": 3.0
}

# No diagonals
ACTIONS = {
    0: (-1, 0),  # UP
    1: ( 1, 0),  # DOWN
    2: (0, -1),  # LEFT
    3: (0, 1),  # RIGHT
}
NUM_ACTIONS = len(ACTIONS)

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
        for rc in all_coords: # search space for robot to place
            if grid_data[rc[0]][rc[1]] == 0:
                robot_pos = rc
                grid_data[rc[0]][rc[1]] = "R"
                break # once found, no need to keep searching

        for rc in all_coords: # search space for target to place
            if grid_data[rc[0]][rc[1]] == 0 and rc != robot_pos:
                target_pos = rc
                grid_data[rc[0]][rc[1]] = "T"
                break # once found, no need to keep searching

        if robot_pos is None or target_pos is None:
            continue

        numeric_grid = get_grid_numeric(grid_data)
        path = find_shortest_path_bfs(numeric_grid, robot_pos, target_pos)
        if path:
            return grid_data, numeric_grid, path, robot_pos, target_pos

    return None, None, None, None, None


# =============================================================================
# ENVIRONMENT (same in both Q-table and DQN)
# =============================================================================

class GridEnvironment:
    def __init__(self, numeric_grid, robot_start, target_pos):
        self.numeric_grid = numeric_grid
        self.robot_start = robot_start
        self.target_pos = target_pos
        self.robot_pos = robot_start

    def reset(self):
        """Reset robot to starting position — same grid every episode."""
        self.robot_pos = self.robot_start
        return self.robot_pos

    def step(self, action):
        dr, dc = ACTIONS[action]
        r, c = self.robot_pos
        nr, nc = r + dr, c + dc
        rows, cols = self.numeric_grid.shape

        if not (0 <= nr < rows and 0 <= nc < cols):
            return self.robot_pos, -5, False        # out of bounds
        if self.numeric_grid[nr, nc] == CELL_TYPES[3]:
            return self.robot_pos, -5, False        # obstacle

        self.robot_pos = (nr, nc)

        if self.robot_pos == self.target_pos:
            return self.robot_pos, +100, True       # goal reached

        return self.robot_pos, -1, False            # normal step


# =============================================================================
# ANIMATION HELPERS (same in both Q-table and DQN)
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
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    bounds = [0.0, 0.25, 0.75, 1.5, 2.5, 3.5]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

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

def plot_rewards(rewards, window=100, title="Reward Curve", save_path="rewards.png"):
    smoothed = [
        np.mean(rewards[max(0, i - window):i + 1])
        for i in range(len(rewards))
    ]
    plt.figure(figsize=(10, 5))
    plt.plot(rewards,   alpha=0.3, color='steelblue', label='Raw reward')
    plt.plot(smoothed,  color='steelblue', linewidth=2, label=f'Rolling avg (window={window})')
    plt.axhline(y=0,    color='gray', linestyle='--', linewidth=0.8)
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
# ██████╗     ██╗      ██████╗  ██████╗ ██╗  ██╗
# ██╔═══██╗   ██║     ██╔═══██╗██╔════╝ ██║ ██╔╝
# ██║   ██║   ██║     ██║   ██║██║      █████╔╝
# ██║▄▄ ██║   ██║     ██║   ██║██║      ██╔═██╗
# ╚██████╔╝   ███████╗╚██████╔╝╚██████╗ ██║  ██╗
#  ╚══▀▀═╝    ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝
# Q-TABLE VERSION
# =============================================================================

def state_to_idx(state, cols): # convert state tuple to index for Q-table
    return state[0] * cols + state[1]

def simulate_ql(Q, env, max_steps=60):
    """Run trained Q-table agent, collect frames."""
    state = env.reset()
    path_taken = [state]
    frames = [_build_frame(env.numeric_grid, state, env.target_pos, path_taken)]
    done = False

    for step in range(max_steps):
        if state == env.target_pos:
            print(f"[Q-table] Target reached in {step} steps!")
            break

        s_idx = state_to_idx(state, env.numeric_grid.shape[1])
        action = np.argmax(Q[s_idx])
        state, _, done = env.step(action)
        path_taken.append(state)
        frames.append(_build_frame(env.numeric_grid, state, env.target_pos, path_taken))
        
        if done:
            print(f"[Q-table] Target reached in {step + 1} steps!")
            break

    if not done:
        print("[Q-table] Max steps reached. Target not reached.")
    return path_taken, frames

def train_qtable(env, rows, cols):
    """Full Q-table training."""

    def calculate_decay(episode, epsilon, epsilon_min, scalar_val):
        target_episodes = episode // scalar_val
        return (epsilon_min / epsilon) ** (1 / target_episodes)
    
    def calculate_max_steps(rows, cols, scalar_val, safety_factor):
        path_length = (rows - 1) + (cols - 1)
        return path_length * scalar_val + safety_factor
    
    # ── Hyperparameters ───────────────────────────────────────────────────────
    NUM_STATES = rows * cols # number of possible states
    EPISODES = 3000 # training episodes usually between 1000 and 10000
    ALPHA = 0.1 # learning rate (fixed for Q-Learning)
    GAMMA = 0.95 # discount factor (fixed for Q-Learning)
    EPSILON = 1.0 # start with full exploration (fixed for Q-Learning)
    EPSILON_MIN = 0.01 # allow full exploitation at the end (fixed for Q-Learning)
    EPSILON_DECAY = calculate_decay(EPISODES, EPSILON, EPSILON_MIN, 3) # decay rate formula: EPSILON_DECAY = (EPSILON_MIN / EPSILON) ** (1 / target_episodes) where target_episodes is (EPISODES / k) where k is a safety factor, usually between 2 and 4
    MAX_STEPS = calculate_max_steps(rows, cols, 3, 5) # formula: path length × N -- where N is a safety factor (usually between 2 and 5) and path length is (rows-1) + (cols-1)

    Q = np.zeros((NUM_STATES, NUM_ACTIONS))
    rewards_per_episode = []

    print("\n" + "="*60)
    print("   Q-TABLE TRAINING")
    print("="*60)
    print(f"Grid: {rows}×{cols} | States: {NUM_STATES} | Episodes: {EPISODES} | Max steps: {MAX_STEPS}")
    print(f"α={ALPHA} | γ={GAMMA} | ε decay={EPSILON_DECAY}")

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False

        for _ in range(MAX_STEPS):
            s_idx = state_to_idx(state, cols)

            # Epsilon-greedy
            if random.random() < EPSILON:
                action = random.randint(0, NUM_ACTIONS - 1)
            else:
                action = np.argmax(Q[s_idx])

            new_state, reward, done = env.step(action)
            total_reward += reward

            # Q update: Q(S,A) = Q(S,A) + α × (R + γ × max(Q(S')) - Q(S,A))
            ns_idx = state_to_idx(new_state, cols)
            best_future_q = np.max(Q[ns_idx])
            Q[s_idx, action] += ALPHA * (reward + GAMMA * best_future_q - Q[s_idx, action])

            state = new_state
            if done:
                break

        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
        rewards_per_episode.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode+1:4d} | Avg Reward: {avg:7.2f} | ε: {EPSILON:.3f}")

    print("\n[Q-table] Training complete!")
    print(f"Best episode reward:    {np.max(rewards_per_episode):.2f}")
    print(f"Avg reward (last 100):  {np.mean(rewards_per_episode[-100:]):.2f}")
    return Q, rewards_per_episode


# =============================================================================
# ██████╗  ██████╗ ███╗   ██╗
# ██╔══██╗██╔═══██╗████╗  ██║
# ██║  ██║██║   ██║██╔██╗ ██║
# ██║  ██║██║▄▄ ██║██║╚██╗██║
# ██████╔╝╚██████╔╝██║ ╚████║
# ╚═════╝  ╚══▀▀═╝ ╚═╝  ╚═══╝
# DQN VERSION
# =============================================================================

# ── 1. Neural Network ─────────────────────────────────────────────────────────
# Replaces the Q-table.
# Input:  [robot_r, robot_c]  (state as 2 numbers)
# Output: [Q_up, Q_down, Q_left, Q_right]  (one Q value per action)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            # nn.Dropout(0.1), --> we could use dropout but it's not necessary for now
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


# ── 2. Replay Buffer ──────────────────────────────────────────────────────────
# Stores past experiences. Sampled randomly to break correlation.
# Equivalent to: your fixed dataset in supervised learning.
class ReplayBuffer:
    def __init__(self, maxlen=10000):
        self.buffer = deque(maxlen=maxlen)  # auto-drops oldest when full

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


def simulate_dqn(model, env, rows, cols, max_steps=60):
    """Run trained DQN agent, collect frames."""
    device = next(model.parameters()).device
    state = get_norm_robot_state(env.reset(), env.target_pos, rows, cols)
    path_taken = [env.robot_pos]
    frames = [_build_frame(env.numeric_grid, env.robot_pos, env.target_pos, path_taken)]
    done = False

    model.eval()
    for step in range(max_steps):
        if env.robot_pos == env.target_pos:
            print(f"[DQN] Target reached in {step} steps!")
            break

        with torch.no_grad():
            q_values = model(torch.tensor(state, dtype=torch.float32).to(device))
            action = torch.argmax(q_values).item()

        new_state_raw, _, done = env.step(action) # new_state_raw, _, done = env.step(action)
        state = get_norm_robot_state(new_state_raw, env.target_pos, rows, cols)
        path_taken.append(env.robot_pos)
        frames.append(_build_frame(env.numeric_grid, env.robot_pos, env.target_pos, path_taken))

        if done:
            print(f"[DQN] Target reached in {step + 1} steps!")
            break

    if not done:
        print("[DQN] Max steps reached. Target not reached.")
    return path_taken, frames

def get_norm_robot_state(robot_pos, target_pos, rows, cols):
    return robot_pos[0] / rows, robot_pos[1] / cols, target_pos[0] / rows, target_pos[1] / cols   # normalized robot col
   

def train_dqn(env, rows, cols):
    """Full DQN training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def calculate_decay(episode, epsilon, epsilon_min, scalar_val):
        target_episodes = episode // scalar_val
        return (epsilon_min / epsilon) ** (1 / target_episodes)
    
    def calculate_max_steps(rows, cols, scalar_val, safety_factor):
        path_length = (rows - 1) + (cols - 1)
        return path_length * scalar_val + safety_factor
    
    # ── Hyperparameters ───────────────────────────────────────────────────────
    INPUT_DIM = 4      # [robot_r, robot_c]
    OUTPUT_DIM = NUM_ACTIONS
    EPISODES = 3000   # training episodes/epochs (usually between 1000 and 10000, depending upon the problem)
    MAX_STEPS = calculate_max_steps(rows, cols, 3, 5)  # max steps per episode - formula: path length × N -- where N is a safety factor (usually between 2 and 5) and path length is (rows-1) + (cols-1)
    ALPHA = 1e-3    # smaller than Q-table, Adam is sensitive (fixed for Q-Learning)
    GAMMA = 0.95    # discount factor (fixed for Q-Learning)
    EPSILON = 1.0    # start with full exploration (fixed for Q-Learning)
    EPSILON_MIN = 0.01     # allow full exploitation at the end (fixed for Q-Learning)
    EPSILON_DECAY = calculate_decay(EPISODES, EPSILON, EPSILON_MIN, 3)  # EPSILON_DECAY = (EPSILON_MIN / EPSILON) ** (1 / target_episodes) where target_episodes is (EPISODES / k) where k is a safety factor, usually between 2 and 4
    BATCH_SIZE = 32       # like supervised learning
    BUFFER_SIZE = 10000 
    MIN_BUFFER = 500      # explore first before training starts
    WEIGHT_DECAY = 0  # introduces L2 regularization (Ridges regression)
 
    # ── Initialize ────────────────────────────────────────────────────────────
    model = DQN(INPUT_DIM, OUTPUT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ALPHA, weight_decay=WEIGHT_DECAY) 
    loss_fn = nn.MSELoss()
    replay_buffer = ReplayBuffer(maxlen=BUFFER_SIZE)   # ← DEFINED HERE
    
    rewards_per_episode = []

    print("\n" + "="*60)
    print("   DQN TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Episodes: {EPISODES} | Batch: {BATCH_SIZE} | Buffer: {BUFFER_SIZE} | Max steps: {MAX_STEPS}")
    print(f"α={ALPHA} | γ={GAMMA} | ε decay={EPSILON_DECAY}")
    print(f"Training starts after {MIN_BUFFER} experiences in buffer")

    best_avg_reward = -np.inf
    model.train() # set model to training mode

    for episode in range(EPISODES):
        state = get_norm_robot_state(env.reset(), env.target_pos, ROWS, COLS)  # (r,c) → [r, c] for tensor
        total_reward = 0 # total reward per episode
        done = False
        episode_loss = 0
        train_steps = 0

        # step_bar = tqdm(range(MAX_STEPS), desc="Training", unit="ep")

        for _ in range(MAX_STEPS):
            # ── Epsilon-greedy (identical logic to Q-table) ───────────────────
            if random.random() < EPSILON:
                action = random.randint(0, OUTPUT_DIM - 1)           # explore
            else:
                model.eval() # set model to evaluation mode

                with torch.no_grad():
                    q_vals = model(torch.tensor(state, dtype=torch.float32).to(device)) # returns q values list = [q1, q2, q3, q4]
                    action = torch.argmax(q_vals).item() # returns maxium index of q values list            # exploit

                model.train() # set model to training mode

            # ── Take action ───────────────────────────────────────────────────
            new_state_raw, reward, done = env.step(action)
            new_state = get_norm_robot_state(new_state_raw, env.target_pos, ROWS, COLS)
            total_reward += reward

            # ── Push to replay buffer (NEW vs Q-table) ────────────────────────
            replay_buffer.push(state, action, reward, new_state, done)
            state = new_state

            # ── Train only when buffer has enough (NEW vs Q-table) ────────────
            if len(replay_buffer) >= MIN_BUFFER:
                states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(BATCH_SIZE)

                states_b = states_b.to(device)
                actions_b = actions_b.to(device)
                rewards_b = rewards_b.to(device)
                next_states_b = next_states_b.to(device)
                dones_b = dones_b.to(device)
                # Current Q values for the actions that were actually taken
                current_q = model(states_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)

                # Target Q values (Bellman equation — same as Q-table formula)
                # R + γ × max(Q(S')) × (1 - done)
                # (1 - done) ensures no future reward if episode ended
                with torch.no_grad():
                    max_next_q = model(next_states_b).max(1)[0]
                    target_q = rewards_b + GAMMA * max_next_q * (1 - dones_b)

                # Backprop — exactly like supervised learning
                loss = loss_fn(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                episode_loss += loss.item()
                train_steps  += 1

            if done:
                break

        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
        rewards_per_episode.append(total_reward)

        avg_reward = np.mean(rewards_per_episode[-100:])
        avg_loss = episode_loss / train_steps if train_steps > 0 else 0.0
        buffer_pct = min(100, int(len(replay_buffer) / MIN_BUFFER * 100))

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1:4d} | avg reward: {avg_reward:.2f} | ε: {EPSILON:.3f} | loss: {avg_loss:.4f} | buf: {buffer_pct}")

            
    print("\n[DQN] Training complete!")
    print(f"Best episode reward: {best_avg_reward:.2f}")
    print(f"Avg reward (last 100): {np.mean(rewards_per_episode[-100:]):.2f}")
    return model, rewards_per_episode


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    ROWS, COLS = 25, 25 # desired grid size

    # Generate ONE fixed grid — shared by both Q-table and DQN
    print("Generating fixed grid...")
    while True:
        _, FIXED_GRID, _, ROBOT_START, TARGET = generate_random_grid(
            ROWS, COLS, obstacle_density=0.70 # higher obstacle density
        )

        if FIXED_GRID is not None:
            break

    print(f"Robot start: {ROBOT_START} | Target: {TARGET}")
    print(FIXED_GRID)

    # Theoretical max reward
    optimal_steps = (ROWS - 1) + (COLS - 1) # path length
    theoretical_max = 100 + (optimal_steps * -1)
    realistic_max = 100 + (int(optimal_steps * 1.5) * -1)
    print("\nComparisons:")
    print(f"Theoretical max reward: {theoretical_max}")
    print(f"Realistic max reward: {realistic_max}")

    env = GridEnvironment(FIXED_GRID, ROBOT_START, TARGET)

    # ── Choose which to run ───────────────────────────────────────────────────
    RUN_QTABLE = False # run Q-table # no need to run both
    RUN_DQN = True # run DQN

    os.makedirs("examples", exist_ok=True)

    # ── Q-Table ───────────────────────────────────────────────────────────────
    if RUN_QTABLE:
        Q, ql_rewards = train_qtable(env, ROWS, COLS)

        plot_rewards(ql_rewards,
                     title="Q-Table: Reward over Episodes",
                     save_path="examples/ql_rewards.png")

        path, frames = simulate_ql(Q, env)
        print("Q-table path:", path)
        animate_path(frames,
                     title="Q-Table Robot",
                     save_path="examples/ql_animation.gif")

    # ── DQN ───────────────────────────────────────────────────────────────────
    if RUN_DQN:
        dqn_model, dqn_rewards = train_dqn(env, ROWS, COLS)

        plot_rewards(dqn_rewards,
                     title="DQN: Reward over Episodes",
                     save_path="examples/dqn_rewards.png")

        path, frames = simulate_dqn(dqn_model, env, ROWS, COLS)
        print("DQN path:", path)
        animate_path(frames,
                     title="DQN Robot",
                     save_path="examples/dqn_animation.gif")

    # ── Side by side reward comparison ───────────────────────────────────────
    if RUN_QTABLE and RUN_DQN:
        window = 100
        ql_smooth  = [np.mean(ql_rewards[max(0,  i-window):i+1]) for i in range(len(ql_rewards))]
        dqn_smooth = [np.mean(dqn_rewards[max(0, i-window):i+1]) for i in range(len(dqn_rewards))]

        plt.figure(figsize=(12, 5))
        plt.plot(ql_smooth,  color='steelblue', linewidth=2, label='Q-Table (rolling avg)')
        plt.plot(dqn_smooth, color='tomato',    linewidth=2, label='DQN (rolling avg)')
        plt.axhline(y=theoretical_max, color='green', linestyle='--',
                    linewidth=1, label=f'Theoretical max ({theoretical_max})')
        plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Q-Table vs DQN — Reward Comparison")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("examples/comparison.png")
        print("Comparison chart saved → examples/comparison.png")
        plt.show()