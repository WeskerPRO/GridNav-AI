'''
Agent (r):       the robot (the one making decisions) 
Environment (e): the grid (everything around the agent) 
State (s):       what the agent currently knows → its (x, y) position in the grid
Action (a):      what the agent can do → UP, DOWN, LEFT, RIGHT, diagonals etc
Reward (Q):      feedback signal after each action → how good or bad the action was
Policy (p):      the agent's strategy → "in state S, do action A" where S is the state and A is the action

Q-Learning:
Q(S, A) = Q(S, A) + n x [R + y x max(Q(S', all actions)) - Q(S, A)] where target = R + y x max(Q(S', all actions)) is the target Q-value

# Q(s, a) is the Q-value for state s and action a (current state and action)
# a is the learning rate, gamma is the discount factor
# reward is the immediate reward for taking action a in state s
# y is the discount factor
# Q(s', a') is the Q-value for the next state s' and the next action a' (next state and action)

Exploration vs Exploitation — the robot's dilemma
Early in training, Q values are all random/zero. If the robot always picks the best known action, 
it will keep doing the same thing forever and never discover better paths. This is called the exploitation trap.
Solution: ε-greedy (epsilon greedy)
With probability ε:      pick a RANDOM action  ← explore
With probability 1-ε:    pick the best known action ← exploit

ε starts at 1.0 (pure random)
ε decays over time to 0.01 (almost always greedy)
Early training = mostly exploring. Late training = mostly exploiting what it learned.


For your current setup — **probably not necessary**, but here's when it helps:
```
Add BatchNorm when:
- Network is deep (5+ layers)
- Training is unstable (loss oscillates wildly)
- You have large batches (64+)

Skip BatchNorm when:
- Network is shallow (2-3 layers like yours)
- Batch size is small (32)
- You already normalize inputs with get_state()
- Simulation runs one step at a time → batch=1 problem

'''

import numpy as np
import random

# Grid layout
# 0 = empty, 1 = obstacle, 2 = goal
GRID = [ # Map Grid for the agent to navigate
    [0, 0, 0, 1],
    [0, 1, 0, 0],
    [0, 0, 0, 2],
]

ROWS = len(GRID)
COLS = len(GRID[0])
print("rows:", ROWS)
print("cols:", COLS)

START = (0, 0)
GOAL  = (2, 3)

# Actions: (dr, dc)
ACTIONS = {
    0: (-1,  0),  # UP
    1: ( 1,  0),  # DOWN
    2: ( 0, -1),  # LEFT
    3: ( 0,  1),  # RIGHT
    # 4: (-1, -1),  # TOP_LEFT
    # 5: (-1,  1),  # TOP_RIGHT
    # 6: ( 1, -1),  # BOTTOM_LEFT
    #7: ( 1,  1),  # BOTTOM_RIGHT
}

NUM_ACTIONS = len(ACTIONS)

def step(state, action): # Function to take a step
    """
    Agent takes action in current state.
    Returns: (new_state, reward, done)
    """
    r, c = state
    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc

    # Hit a wall (out of bounds or obstacle) → stay in place, punish
    if not (0 <= nr < ROWS and 0 <= nc < COLS) or GRID[nr][nc] == 1:
        return state, -5, False  # stayed, punished

    new_state = (nr, nc)

    if GRID[nr][nc] == 2:        # Reached goal
        print("Target reached!")
        return new_state, +100, True  # Reached goal, big reward, Target reached (Yes)

    return new_state, -1, False  # Normal move, small penalty


# Q-table: rows=states, cols=actions
# State = (row, col) → flatten to single index
NUM_STATES = ROWS * COLS

def state_to_idx(state): # coverts state (row, col) to single index
    return state[0] * COLS + state[1]

# Initialize all Q values to 0
Q = np.zeros((NUM_STATES, NUM_ACTIONS))

print("Q-table shape:", Q.shape)  # (12, 4) → 12 states, 4 actions
print(Q)
print("random:", random.random())

# Hyperparameters
EPISODES    = 100    # how many times the robot attempts the maze
ALPHA       = 0.1     # learning rate
GAMMA       = 0.9     # discount factor (how much future rewards matter)
EPSILON     = 1.0     # starting exploration rate (100% random)
EPSILON_MIN = 0.01    # minimum exploration rate useful for late training
EPSILON_DECAY = 0.995 # multiply epsilon by this each episode

rewards_per_episode = [] # store rewards for each episode

# Main training loop
for episode in range(EPISODES):
    state = START # Starting position (0, 0) of the robot
    total_reward = 0 # Total reward for this episode
    done = False # Did we reach the goal?

    while not done:
        s_idx = state_to_idx(state) # Convert state (row, col) to single index and returns state[0] * COLS + state[1]
        random_epsilon = random.random() # Returns a random number in the range [0.0, 1.0)
        # random.random() returns random numbers within this range [0.0, 1.0)
        # --- Epsilon-greedy action selection ---
        if random_epsilon < EPSILON: # Exploration vs Exploitation (ε-greedy)
            action = random.randint(0, NUM_ACTIONS - 1)  # explore in early training
        else:
            action = np.argmax(Q[s_idx])                 # exploit # in late training

        # --- Take action, observe result ---
        new_state, reward, done = step(state, action) 
        total_reward += reward

        # --- Q update ---
        ns_idx = state_to_idx(new_state)
        best_future_q = np.max(Q[ns_idx])

        Q[s_idx, action] = Q[s_idx, action] + ALPHA * (reward + GAMMA * best_future_q - Q[s_idx, action]) # increases the Q-value for the current state-action pair if the reward is positive

        state = new_state

    # Decay epsilon after each episode
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY) # purpose of decay is to reduce the exploration rate over time
    rewards_per_episode.append(total_reward)

    if (episode + 1) % 10 == 0:
        avg = np.mean(rewards_per_episode[-100:])
        print(f"Episode {episode+1} | Avg Reward (last 100): {avg:.2f} | ε: {EPSILON:.3f}")

print("\nLearned Q-table (rows=states, cols=UP/DOWN/LEFT/RIGHT):")
print(np.round(Q, 1))

print("\nBest action per cell:")
action_names = [" UP ", " DOWN ", " LEFT ", " RIGHT "]

for r in range(ROWS):
    row_str = ""
    for c in range(COLS):
        if GRID[r][c] == 1: # obstacle
            row_str += "  X  " 
        elif (r, c) == GOAL:
            row_str += "  G  "  # goal
        else:
            idx = state_to_idx((r, c))
            best = np.argmax(Q[idx])
            row_str += f"{action_names[best]:^5}"
    print(row_str)

# Test phase of the trained robot, which is used for evaluation
print("\n--- Trained Robot Running ---")
state = START
path  = [state]
done  = False
steps = 0

while not done and steps < 20:
    s_idx  = state_to_idx(state)
    action = np.argmax(Q[s_idx])          # pure exploitation, no randomness
    state, reward, done = step(state, action)
    path.append(state)
    steps += 1

print("Path taken:", path)
print("Reached goal!" if done else "Failed.")