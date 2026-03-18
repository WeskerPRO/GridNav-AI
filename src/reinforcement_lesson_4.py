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
"""
SHARED COMPONENTS

Input features:     normalize to [0, 1] or [-1, 1]    ← get_state() does this
Rewards in RL:      scale to [-1, 1] range             ← what you just fixed
Target values:      same scale as predictions          ← MSELoss works correctly
Sine/Cosine:        naturally in [-1, 1]               ← perfect for neural nets
"""
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


# =============================================================================
# ENVIRONMENT — Stage 2: shaped reward, target hidden from robot
# =============================================================================

class GridEnvironmentStage2:
    def __init__(self, numeric_grid, robot_start, target_pos):
        self.numeric_grid = numeric_grid
        self.robot_start = robot_start
        self.target_pos = target_pos   # hidden from robot, used only for reward
        self.robot_pos = robot_start
        self.visited = set() # Set to track visited positions

    def reset(self):
        self.robot_pos = self.robot_start
        self.visited = {self.robot_start}  # ← reset visited each episode
        return self.robot_pos

    def step(self, action):
        dr, dc = ACTIONS[action]
        r, c = self.robot_pos
        nr, nc = r + dr, c + dc
        rows, cols = self.numeric_grid.shape

        # Out of bounds or obstacle
        if not (0 <= nr < rows and 0 <= nc < cols):
            return self.robot_pos, -0.05, False
        if self.numeric_grid[nr, nc] == CELL_TYPES[3]:
            return self.robot_pos, -0.05, False

        # Manhattan distance BEFORE move
        prev_dist = abs(r - self.target_pos[0]) + abs(c - self.target_pos[1])
        self.robot_pos = (nr, nc)
        new_dist = abs(nr - self.target_pos[0]) + abs(nc - self.target_pos[1])

        # Goal reached
        if self.robot_pos == self.target_pos:
            return self.robot_pos, +1, True

        shaping = (prev_dist - new_dist) * 0.01 # penalty for small steps calculation

        if self.robot_pos in self.visited:
            revisit_penalty = -0.02     # ← extra punishment for going back
        else:
            revisit_penalty = 0.0      # ← new cell, no penalty

        self.visited.add(self.robot_pos)

        return self.robot_pos, -0.005 + shaping + revisit_penalty, False
        # ↑ small step penalty still exists


# =============================================================================
# STATE — only robot position, no target
# =============================================================================

def get_state(robot_pos, rows, cols, visited):
    """
    Stage 2: robot is blind — only knows its own position.
    Target coordinates are NOT included.
    """
    return [
        robot_pos[0] / rows,   # normalized robot row
        robot_pos[1] / cols,   # normalized robot col
        len(visited) / (rows * cols),
    ]
    # INPUT_DIM = 2


# =============================================================================
# DQN MODEL
# =============================================================================

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, output_dim)   # no LayerNorm on output
        )

    def forward(self, x):
        return self.fc(x)


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

def plot_rewards(rewards, rows, cols, window=1, title="Reward Curve", save_path="rewards.png"):
    optimal_steps = (rows - 1) + (cols - 1)   # ← uses parameters ✅

    theoretical_max = 1 + (optimal_steps * -1)
    realistic_max = 1 + (int(optimal_steps * 1.5) * -1)
    print("\nComparisons:")
    print(f"Theoretical max reward: {theoretical_max}")
    print(f"Realistic max reward: {realistic_max}")

    smoothed = [
        np.mean(rewards[max(0, i - window):i + 1])
        for i in range(len(rewards))
    ]
    plt.figure(figsize=(10, 5))
    plt.plot(rewards,  alpha=0.3, color='tomato', label='Raw reward')
    plt.plot(smoothed, color='tomato', linewidth=2, label=f'Rolling avg (window={window})')
    plt.axhline(y=theoretical_max, color='green', linestyle='--', linewidth=1, label=f'Theoretical max ({theoretical_max})')
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

def train_stage2(env, rows, cols):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''
    ε=0 --> 100% exploitation
    ε=1 --> 100% exploration
    '''
    # Hyperparameters
    INPUT_DIM = 3         # only [robot_r, robot_c]
    OUTPUT_DIM = NUM_ACTIONS
    EPISODES = 3000
    path_length = (rows - 1) + (cols - 1)
    MAX_STEPS = path_length * 3 + 5
    ALPHA = 1e-3
    GAMMA = 0.95
    EPSILON = 1.0 
    EPSILON_MIN = 0.01 #
    target_ep = int(EPISODES * (2 / 3))   # linear decay coef
    EPSILON_DECAY = (EPSILON_MIN / EPSILON) ** (1 / target_ep)
    BATCH_SIZE = 32
    BUFFER_SIZE = 50000      # larger buffer — blind robot needs more diverse exp
    MIN_BUFFER = 1000       # more exploration before training
    EVAL_EVERY = 250    # ← evaluate every N episodes
    # extras_full_unlock.rpy
    model = DQN(INPUT_DIM, OUTPUT_DIM).to(device)
    # ── ADD THESE ──────────────────────────────────────────────
    target_model  = DQN(INPUT_DIM, OUTPUT_DIM).to(device)
    target_model.load_state_dict(model.state_dict())  # same weights initially
    target_model.eval()   # never trains, only predicts
    UPDATE_TARGET_EVERY = 100   # sync every 100 episodes
    # ───────────────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=ALPHA)

    loss_fn = nn.MSELoss()
    replay_buffer = ReplayBuffer(maxlen=BUFFER_SIZE)

    rewards_per_episode = []
    best_avg_reward = -np.inf

    print("\n" + "="*60)
    print("   STAGE 2 DQN — BLIND ROBOT (reward shaping)")
    print("="*60)
    print(f"Device: {device}")
    print(f"State: [robot_r, robot_c] only — target HIDDEN from robot")
    print(f"Reward shaping: ±1.0 per step based on distance change")
    print(f"Episodes: {EPISODES} | MAX_STEPS: {MAX_STEPS} | Batch: {BATCH_SIZE}")
    print(f"α={ALPHA} | γ={GAMMA} | ε_decay={EPSILON_DECAY:.6f}\n")

    episode_bar = tqdm(range(EPISODES), desc="Stage2 Training", unit="ep")

    model.train() # set model to training mode

    for episode_num, _ in enumerate(episode_bar):
        state = get_state(env.reset(), rows, cols, env.visited)
        total_reward = 0
        done = False
        episode_loss = 0
        train_steps  = 0

        for _ in range(MAX_STEPS):
            # Epsilon-greedy

            if random.random() < EPSILON: # explore
                action = random.randint(0, OUTPUT_DIM - 1)
            else:
                model.eval()

                with torch.no_grad(): # expoitation
                    q_vals = model(torch.tensor(state, dtype=torch.float32).to(device))
                    action = torch.argmax(q_vals).item()

                model.train()

            new_state_raw, reward, done = env.step(action)
            new_state = get_state(new_state_raw, rows, cols, env.visited)
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

                current_q = model(states_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    max_next_q = target_model(next_states_b).max(1)[0]  # ← target_model
                    target_q = rewards_b + GAMMA * max_next_q * (1 - dones_b)

                loss = loss_fn(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                episode_loss += loss.item()
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
            torch.save(model.state_dict(), "models/stage2_best.pth")

        episode_bar.set_postfix({
            "avg_r": f"{avg_reward:6.1f}",
            "best": f"{best_avg_reward:6.1f}",
            "loss": f"{avg_loss:.4f}",
            "ε": f"{EPSILON:.3f}",
            "buf": f"{len(replay_buffer)}",
        })

        if (episode_num + 1) % UPDATE_TARGET_EVERY == 0:
            target_model.load_state_dict(model.state_dict())
            tqdm.write(f"[ep {episode_num+1}] Target network updated")

        # ── Periodic evaluation — AFTER episode ends, OUTSIDE step loop ───────
        if (episode_num + 1) % EVAL_EVERY == 0:
            tqdm.write(f"\n--- Eval @ Episode {episode_num + 1} ---")
            success_rate, avg_r, avg_s = evaluate_model(
                model, env, rows, cols, n_episodes=20
            )
            tqdm.write(f"success={success_rate:.0f}% | "
                       f"avg_reward={avg_r:.2f} | "
                       f"avg_steps={avg_s:.1f}")
            model.train()   # ← switch back after evaluate_model sets eval mode


    print(f"\n[Stage 2] Training complete!")
    print(f"Best avg reward: {best_avg_reward:.2f}")
    print(f"Final avg reward (last 100): {np.mean(rewards_per_episode[-100:]):.2f}")

    if os.path.exists("models/stage2_best.pth"):
        model.load_state_dict(torch.load("models/stage2_best.pth", weights_only=True))
        print("Best model loaded.")

    return model, rewards_per_episode

def evaluate_model(model, env, rows, cols, n_episodes=100):
    """
    Test phase equivalent in RL.
    Run n_episodes with pure exploitation (ε=0).
    No training, no exploration, no gradient updates.
    """
    device = next(model.parameters()).device
    model.eval()

    success_count  = 0      # how many times robot reached target
    total_rewards = []
    total_steps = []

    path_length = (rows - 1) + (cols - 1)
    MAX_STEPS = path_length * 3 + 5

    print(f"\n--- Evaluating over {n_episodes} episodes ---")

    for ep in range(n_episodes):
        state = get_state(env.reset(), rows, cols, env.visited)
        total_reward = 0
        done = False
        steps = 0

        for step in range(MAX_STEPS):
            with torch.no_grad():
                q_vals = model(torch.tensor(state, dtype=torch.float32).to(device))
                action = torch.argmax(q_vals).item()   # ε=0, pure greedy

            new_state_raw, reward, done = env.step(action)
            state = get_state(new_state_raw, rows, cols, env.visited)
            total_reward += reward
            steps = step + 1

            if done:
                success_count += 1
                break

        total_rewards.append(total_reward)
        total_steps.append(steps)

    # Metrics — equivalent to test accuracy in supervised learning
    success_rate = success_count / n_episodes * 100
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)

    print(f"Success rate:     {success_rate:.1f}%   ← main metric (like test accuracy)")
    print(f"Avg reward:       {avg_reward:.2f}      ← reward quality")
    print(f"Avg steps:        {avg_steps:.1f}       ← efficiency")
    print(f"Best episode:     {np.max(total_rewards):.2f}")
    print(f"Worst episode:    {np.min(total_rewards):.2f}")

    return success_rate, avg_reward, avg_steps

# =============================================================================
# SIMULATION
# =============================================================================

def simulate_stage2(model, env, rows, cols, max_steps=None):
    if max_steps is None:
        max_steps = ((rows-1) + (cols-1)) * 3 + 5

    device = next(model.parameters()).device
    state = get_state(env.reset(), rows, cols, env.visited)
    path_taken = [env.robot_pos]
    frames = [_build_frame(env.numeric_grid, env.robot_pos, env.target_pos, path_taken)]
    done = False

    model.eval()

    for step in range(max_steps):
        if env.robot_pos == env.target_pos:
            print(f"[Stage 2] Target reached in {step} steps!")
            break

        with torch.no_grad():
            q_vals = model(torch.tensor(state, dtype=torch.float32).to(device))
            action = torch.argmax(q_vals).item()

        new_state_raw, _, done = env.step(action)
        state = get_state(new_state_raw, rows, cols, env.visited)
        path_taken.append(env.robot_pos)
        frames.append(_build_frame(env.numeric_grid, env.robot_pos, env.target_pos, path_taken))

        if done:
            print(f"[Stage 2] Target reached in {step + 1} steps!")
            break

    if not done:
        print("[Stage 2] Max steps reached. Target not reached.")

    return path_taken, frames


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Hyperparameters
    ROWS, COLS = 35, 35   # start small for Stage 2 — blind robot is harder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Generating fixed grid...")
    while True:
        _, FIXED_GRID, _, ROBOT_START, TARGET = generate_random_grid(
            ROWS, COLS, obstacle_density=0.35  # lower density — blind robot needs easier map
        )
        if FIXED_GRID is not None:
            break

    print(f"Robot start: {ROBOT_START} | Target: {TARGET}")
    print(FIXED_GRID)

    # Theoretical max reward with shaping
    # Each step toward target gives +0.5 (-0.5 + 1.0 shaping)
    # Each step away gives -1.5 (-0.5 - 1.0 shaping)
    optimal_steps = abs(ROBOT_START[0] - TARGET[0]) + abs(ROBOT_START[1] - TARGET[1])
    theoretical_max = 100 + (optimal_steps * 0.5)  # +0.5 per step toward target
    print(f"\nOptimal path length: {optimal_steps}")
    print(f"Theoretical max reward (shaped): {theoretical_max:.1f}")

    env = GridEnvironmentStage2(FIXED_GRID, ROBOT_START, TARGET)

    os.makedirs("examples", exist_ok=True)

    model, rewards = train_stage2(env, ROWS, COLS)
    plot_rewards(rewards, ROWS, COLS, title="Stage 2 DQN — Blind Robot (reward shaping)", save_path="examples/stage2_rewards.png")

    path, frames = simulate_stage2(model, env, ROWS, COLS)
    print("Path taken:", path)
    animate_path(frames, title="Stage 2 — Blind Robot", save_path="examples/stage2_animation.gif")
