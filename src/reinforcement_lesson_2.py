import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from matplotlib import animation
import os

## The full picture side by side
'''
Supervised Learning          Q-Learning
─────────────────────────    ──────────────────────────────
Loss function                TD Error: R + γQ' - Q
Learning rate α              Learning rate α  (same concept)
Weight decay                 No direct equivalent
Epochs                       Episodes
Batch size                   Single step updates (or replay buffer in DQN)
Overfitting                  Exploitation trap (too greedy too early)
Underfitting                 Underexploration (ε decays too fast)
Train/val split              Train performance = reward curve
Model checkpoint             Save Q-table when avg reward peaks


(1)
α — Learning Rate [Range: (0, 1), typical: 0.01 - 0.3]
Too high (> 0.5): Q values jump wildly, never converge, like gradient descent diverging
Too low (< 0.01): Learns correctly but needs millions of episodes, extremely slow
Sweet spot (0.1): Stable and reasonably fast convergence

(2)
γ — Discount Factor [Range: (0, 1), typical: 0.9 - 0.99]
Too high (> 0.99): Tiny Q errors propagate very far back in time, training becomes unstable
Too low (< 0.5): Robot only cares about immediate reward, completely blind to future, walks in circles if no reward nearby
Sweet spot (0.9-0.95): Robot plans ahead reasonably without instability

(3)
ε — Exploration Rate [Range: (0, 1), starts at 1.0, decays to ~ 0.01]
Stuck at 1.0: Pure random forever, never exploits what it learned, no improvement
Stuck at 0.0 early: Gets locked into first habit it finds, misses better paths, local optimum
Proper decay: Explores broadly early, gradually shifts to exploiting learned knowledge

(4)
ε — decay rate [Range: (0, 1), typical: 0.99 - 0.995]
Too aggressive (< 0.95): Robot stops exploring after ~100 episodes, hasn't seen enough of the grid yet, poor generalization
Too slow (> 0.9999): Robot stays random for thousands of wasted episodes, never learns
Sweet spot (0.995 - 0.999): Exploration fades out naturally over the full training run (balanced between exploration and exploitation)

(5)

The formula to estimate when ε hits minimum:
episodes_to_min = log(ε_min / ε_start) / log(decay_rate)

Example: log(0.01 / 1.0) / log(0.995) ≈ 919 episodes
So with 3000 total episodes, the robot spends ~1/3 exploring and ~2/3 exploiting. Good balance.

(6)
MAX_STEPS per episode [Range: depends on grid, typical: 2× – 5× optimal path length]
too low: Episode ends before robot reaches goal, never sees reward signal, Q values toward goal never update
too high: One bad episode wastes thousands of random steps, slows training badly
Sweet spot (2-5× optimal path length): Good balance between exploration and exploitation

(7)
Reward values [No fixed range — what matters is the RATIO between them]

Goal reward: 
Too low → drowns in step penalties, robot prefers standing still

Step penalty: 
Zero → robot finds goal but takes 200 unnecessary steps
Too high (-50) → robot freezes, standing still beats moving

Wall penalty: 
Zero → robot bounces off walls carelessly, slow convergence
Too high (-100) → robot is so afraid of walls it stops exploring

---->
The critical ratio to always check:
goal_reward  >>  MAX_STEPS × |step_penalty|

Good:  +100  >>  200 × 1  =  200  ✅  goal dominates
Bad:   +10   >>  200 × 1  =  200  ❌  penalties dominate, robot gives up

(8)

NUM_EPISODES [Range: hundreds to millions depending on problem, typical for small grids: 1000 – 10000]
Too low: Robot hasn't explored enough states, Q table mostly zeros, bad policy
Too high: Wastes time after convergence, no harm but unnecessary
How to know enough: Plot reward curve — when average reward plateaus, training is done

Parameter      Typical        Too High              Too Low
─────────────────────────────────────────────────────────────
α              0.1            Q diverges            Too slow
γ              0.9 – 0.95     Unstable propagation  Shortsighted
ε start        1.0            —                     Misses exploration
ε end          0.01           Never exploits        Over-exploits early
ε decay        0.995          Stops exploring early Wastes episodes randomly
MAX_STEPS      3×–5× BFS      Slow training         Never sees reward
Goal reward    +100           Overkill (fine)        Drowned by penalties
Step penalty   -1             Robot freezes         Inefficient paths
Wall penalty   -5             Robot stops moving    Careless bouncing
Episodes       2000–5000      Wasted compute        Undertrained

----- Best Practices:

(1) MAX_STEPS
Worst case Manhattan distance (no diagonals, no obstacles):
= (rows - 1) + (cols - 1) = path length

With obstacles blocking direct path, multiply by safety factor 3:
MAX_STEPS = ((rows-1) + (cols-1)) × 3 = path_length x (2 or 3 or 4 or 5)

This is exact math — Manhattan distance is provably the minimum
steps needed to cross a grid without diagonals

-----

(2) EPSILON_DECAY
You want ε to reach EPSILON_MIN after target_episode episodes:

EPSILON × decay^target_episodes = EPSILON_MIN
decay^target_episodes = EPSILON_MIN / EPSILON
decay = (EPSILON_MIN / EPSILON)^(1 / target_episodes) # EPSILON_DECAY formula

target_episodes is the episode number where you want ε to finish decaying. Using the rule of thumb of 1/3 of total episodes:
target_episodes = EPISODES / 3 (could be 1/2 or 2/3 or 3/4 deppending on your problem)

This is exact algebra — solving for decay rate given a target

-----

(3) ε hits minimum at episode N (inverse of above)
N = log(EPSILON_MIN / EPSILON) / log(EPSILON_DECAY)

This is exact — logarithm solving for exponential decay rate

(4) EPSILON_DECAY target = EPISODES / 3
Not a proven formula.
Origin: common practitioner advice — spend roughly:
  1/3 episodes exploring      (ε high)
  1/3 episodes mixed          (ε decaying)
  1/3 episodes exploiting     (ε ≈ minimum)

Some papers suggest 1/2 instead of 1/3.
Depends entirely on your problem.

-----

(4) Safety factor of 3× for MAX_STEPS
Not proven — just conservative padding.
With 20% obstacle density, BFS paths are rarely longer than
2× Manhattan distance. 3× gives extra room for the robot
to make a few mistakes before hitting the limit.

-----

Summary:

Formula type          Parameters
─────────────────────────────────────────────
Exact math ✅         MAX_STEPS, EPSILON_DECAY, ε decay episode estimate
Rules of thumb ⚠️     EPISODES, the 1/3 split, the 3× safety factor
Always fixed          ALPHA, GAMMA, EPSILON_START, EPSILON_MIN

-----

Max Reward Formula: Max Reward = GOAL_REWARD + (optimal_path_length × STEP_PENALTY)
Where:
optimal_path_length = (rows - 1) + (cols - 1)   ← Manhattan distance, no diagonals
                                                    best case, no obstacles

Your 6×7 grid
optimal_path_length = (6-1) + (7-1) = 5 + 6 = 11 steps

Max Reward = +100 + (11 × -1)
           = +100 - 11
           =  +89   ← theoretical best case (no obstacles)

-----

So DQN has:
```
batch_size    ✅  (sample from replay buffer)
shuffle       ✅  (random sampling breaks sequential correlation)
optimizer     ✅  (Adam, same as yours)
loss.backward ✅  (backprop through neural network)
DataLoader    ❌  (replaced by replay buffer sampling)
train/val     ❌  (no validation split, reward curve is your metric)
```

-----

```
Concept          Supervised        Q-Learning       DQN
────────────────────────────────────────────────────────────
Data source      fixed dataset     live experience  replay buffer
batch_size       ✅ DataLoader     ❌ step by step  ✅ random sample
shuffle          ✅                ❌               ✅ (implicit)
loss function    CrossEntropy      TD error         TD error
optimizer        Adam ✅           none (formula)   Adam ✅
backprop         ✅                ❌               ✅
val split        ✅                ❌               ❌
metric           loss + accuracy   reward curve     reward curve
early stopping   ✅ patience       ✅ reward plateau ✅ reward plateau
```

-----

## Why replay buffer exists in DQN

Without it, consecutive experiences are **highly correlated**:
```
step 1: robot at (0,0) → moves RIGHT
step 2: robot at (0,1) → moves RIGHT
step 3: robot at (0,2) → moves RIGHT

'''


# ── reuse your existing helpers ──────────────────────────────────────────────
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
    #4: (-1, -1),  # TOP_LEFT
    #5: (-1,  1),  # TOP_RIGHT
    #6: ( 1, -1),  # BOTTOM_LEFT
    #7: ( 1,  1),  # BOTTOM_RIGHT
}
NUM_ACTIONS = len(ACTIONS)

def get_grid_numeric(grid_data): 
    """Converts the symbolic grid data to a numeric NumPy array."""
    numeric_grid = np.zeros((len(grid_data), len(grid_data[0])), dtype=np.float32)
    for r in range(len(grid_data)):
        for c in range(len(grid_data[0])):
            numeric_grid[r, c] = CELL_TYPES.get(grid_data[r][c], 0.0)
    return numeric_grid

def find_shortest_path_bfs(grid_numeric, start_pos, target_pos):
    queue   = deque([(start_pos, [start_pos])])
    visited = {start_pos}
    obstacle_val = CELL_TYPES[3]
    rows, cols = grid_numeric.shape

    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == target_pos:
            return path
        for dr, dc in ACTIONS.values():
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and grid_numeric[nr, nc] != obstacle_val
                    and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))
    return None

def generate_random_grid(rows, cols, obstacle_density=0.2):
    for _ in range(100):
        grid_data = [[0]*cols for _ in range(rows)]

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


def simulate_robot_movement_ql(Q, env, max_steps=60):
    """Run trained Q-table agent and collect frames for animation."""
    state = env.reset()
    path_taken = [state]
    frames = []
    done = False

    # Initial frame
    frames.append(_build_frame(env.numeric_grid, state, env.target_pos, path_taken))

    for step in range(max_steps):
        if state == env.target_pos:
            print(f"Target reached in {step} steps!")
            break

        # Pure exploitation — no randomness
        s_idx  = state_to_idx(state, env.numeric_grid.shape[1])
        action = np.argmax(Q[s_idx])

        state, _, done = env.step(action)
        path_taken.append(state)

        frames.append(_build_frame(env.numeric_grid, state, env.target_pos, path_taken))

        if done:
            print(f"Target reached in {step + 1} steps!")
            break

    if not done:
        print("Max steps reached. Target not reached.")

    return path_taken, frames


def _build_frame(numeric_grid, robot_pos, target_pos, path_taken):
    """Build a single frame for animation."""
    frame = np.copy(numeric_grid)

    # Mark visited path (excluding current position)
    for p_r, p_c in path_taken[:-1]:
        frame[p_r, p_c] = 0.5  # light grey trail

    # Mark robot and target
    frame[robot_pos[0],  robot_pos[1]]  = CELL_TYPES["R"]  # 2.0 → red
    frame[target_pos[0], target_pos[1]] = CELL_TYPES["T"]  # 3.0 → green

    return frame


def animate_path_ql(frames, save_path="ql_animation.gif"):
    """Animate the Q-Learning robot's path and save as GIF."""
    if not frames:
        print("No frames to animate.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    # Color mapping:
    # 0.0 = empty (white)
    # 0.5 = visited trail (light grey)
    # 1.0 = obstacle (black)
    # 2.0 = robot (red)
    # 3.0 = target (green)
    colors = ['#FFFFFF', '#DDDDDD', '#111111', '#E74C3C', '#2ECC71']
    cmap   = plt.matplotlib.colors.ListedColormap(colors)
    bounds = [0.0, 0.25, 0.75, 1.5, 2.5, 3.5]
    norm   = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    def update(i):
        frame_data = frames[i]
        ax.clear()
        ax.imshow(
            frame_data, cmap=cmap, norm=norm,
            origin='upper',
            extent=[0, frame_data.shape[1], frame_data.shape[0], 0]
        )
        # Grid lines
        ax.set_xticks(np.arange(frame_data.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(frame_data.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Q-Learning Robot — Step {i}", fontsize=13)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#E74C3C', label='Robot'),
            Patch(facecolor='#2ECC71', label='Target'),
            Patch(facecolor='#111111', label='Obstacle'),
            Patch(facecolor='#DDDDDD', label='Visited'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    ani = animation.FuncAnimation(
        fig, update,
        frames=range(len(frames)),
        repeat=False,
        interval=500
    )

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    ani.save(save_path, writer='pillow', fps=2)
    print(f"Animation saved to {save_path}")
    plt.show()

# ── Environment ──────────────────────────────────────────────────────────────
ROWS, COLS = 35, 35

while True:
    _, FIXED_NUMERIC_GRID, _, FIXED_ROBOT_START, FIXED_TARGET = generate_random_grid(
        ROWS, COLS, obstacle_density=0.2
    )
    if FIXED_NUMERIC_GRID is not None:
        break

print(f"Fixed grid generated. Robot start: {FIXED_ROBOT_START}, Target: {FIXED_TARGET}")
print(FIXED_NUMERIC_GRID)

# ── Environment ───────────────────────────────────────────────────────────────
class GridEnvironment:
    def __init__(self, numeric_grid, robot_start, target_pos):
        self.numeric_grid = numeric_grid
        self.robot_start  = robot_start
        self.target_pos   = target_pos
        self.robot_pos    = robot_start

    def reset(self):
        """Reset robot to starting position — same grid every episode."""
        self.robot_pos = self.robot_start
        return self.robot_pos

    def step(self, action):
        """Apply action, return (new_state, reward, done)."""
        dr, dc = ACTIONS[action]
        r, c   = self.robot_pos
        nr, nc = r + dr, c + dc
        rows, cols = self.numeric_grid.shape

        # Out of bounds or obstacle → stay, punish
        if not (0 <= nr < rows and 0 <= nc < cols):
            return self.robot_pos, -5, False
        if self.numeric_grid[nr, nc] == CELL_TYPES[3]:
            return self.robot_pos, -5, False

        self.robot_pos = (nr, nc)

        if self.robot_pos == self.target_pos:
            return self.robot_pos, +100, True   # reached goal!

        return self.robot_pos, -1, False         # normal step

# ── Q-Table ───────────────────────────────────────────────────────────────────
def state_to_idx(state, cols):
    return state[0] * cols + state[1]

NUM_STATES = ROWS * COLS
Q = np.zeros((NUM_STATES, NUM_ACTIONS))

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPISODES      = 7500  # training episodes/epochs (usually between 1000 and 10000, depending upon the problem)
ALPHA         = 0.1  # learning rate (fixed for Q-Learning)
GAMMA         = 0.95  # discount factor (fixed for Q-Learning)
EPSILON       = 1.0  # start with full exploration (fixed for Q-Learning)
EPSILON_MIN   = 0.01  # allow full exploitation at the end (fixed for Q-Learning)
EPSILON_DECAY = 0.998 # ε hits minimum around episode 919 - formula: EPSILON_DECAY = (EPSILON_MIN / EPSILON) ** (1 / target_episodes) where target_episodes is (EPISODES / k) where k is a safety factor, usually between 2 and 4
MAX_STEPS     = 220  # max steps per episode - formula: path length × N -- where N is a safety factor (usually between 2 and 5) and path length is (rows-1) + (cols-1)

env = GridEnvironment(FIXED_NUMERIC_GRID, FIXED_ROBOT_START, FIXED_TARGET)
rewards_per_episode = []

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    done = False

    for _ in range(MAX_STEPS):
        s_idx = state_to_idx(state, COLS)

        # Epsilon-greedy
        if random.random() < EPSILON:
            action = random.randint(0, NUM_ACTIONS - 1)   # explore
        else:
            action = np.argmax(Q[s_idx]) # returns the index of the best action -- exploit

        new_state, reward, done = env.step(action)
        total_reward += reward

        # Q update
        ns_idx = state_to_idx(new_state, COLS)
        best_future_q  = np.max(Q[ns_idx])

        Q[s_idx, action] = Q[s_idx, action] + ALPHA * (reward + GAMMA * best_future_q - Q[s_idx, action])

        state = new_state

        if done:
            break

    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    rewards_per_episode.append(total_reward)

    if (episode + 1) % 100 == 0:
        avg = np.mean(rewards_per_episode[-100:])
        print(f"Episode {episode+1:4d} | Avg Reward: {avg:7.2f} | ε: {EPSILON:.3f}")

# ── Test the trained robot ────────────────────────────────────────────────────
print("\n--- Trained Robot Running ---")
state = env.reset()
path  = [state]
done  = False

for _ in range(MAX_STEPS):
    s_idx  = state_to_idx(state, COLS)
    action = np.argmax(Q[s_idx])              # pure exploitation
    state, reward, done = env.step(action)
    path.append(state)
    if done:
        break

print("Path taken:", path)
print("Target was:", env.target_pos)
print("Reached goal!" if done else "Failed — Q-table needs more training.")
print("\n--- Running trained Q-Learning robot ---")

print("Q-table (rows=states, cols=UP/DOWN/LEFT/RIGHT):")
print(np.round(Q, 1))

path_taken, frames = simulate_robot_movement_ql(Q, env, max_steps=MAX_STEPS)

print("Path taken:", path_taken)

avg_last_100 = np.mean(rewards_per_episode[-100:])
best_episode = np.max(rewards_per_episode)
worst_episode = np.min(rewards_per_episode[-100:])

print(f"Avg reward (last 100): {avg_last_100:.2f}")
print(f"Best episode:          {best_episode:.2f}")
print(f"Worst episode:         {worst_episode:.2f}")

animate_path_ql(frames, save_path="examples/ql_robot_animation.gif")

window = 100  # rolling average window

smoothed = [
    np.mean(rewards_per_episode[max(0, i - window):i + 1])
    for i in range(len(rewards_per_episode))
]

plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode, alpha=0.3, color='steelblue', label='Raw reward per episode')
plt.plot(smoothed, color='steelblue', linewidth=2, label=f'Rolling avg (window={window})')
plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)  # zero line reference

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning: Reward over Episodes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rl_reward_curve.png")
plt.show()