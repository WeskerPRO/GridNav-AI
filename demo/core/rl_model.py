import torch                                            # tensors + autograd
import torch.nn as nn                                    # layers + loss functions
import torch.optim as optim                              # Adam optimiser
import numpy as np                                       # means / arrays
import random                                            # epsilon-greedy + sampling
from collections import deque                            # replay buffer + oscillation window

# =============================================================================
# CONSTANTS
# =============================================================================

ACTIONS = {                                              # action index -> (row, col) move
    0: (-1,  0),   # UP
    1: ( 1,  0),   # DOWN
    2: ( 0, -1),   # LEFT
    3: ( 0,  1),   # RIGHT
}
NUM_ACTIONS  = 4                                         # 4 possible moves
WINDOW_SIZE  = 7                                         # 7x7 vision window around the robot
WINDOW_CELLS = WINDOW_SIZE * WINDOW_SIZE   # 49          # cells inside that window
INPUT_DIM = WINDOW_CELLS + 5            # 54 = 49 window + pos_r + pos_c + tgt_r + tgt_c + explored
                                        # must mirror reinforcement_lesson_5.py's get_state()
                                        # (target coords added so the model generalizes across grids)

CELL_OBSTACLE = 1.0                                      # encoding: obstacle
CELL_TARGET = 3.0                                        # encoding: target visible in window
CELL_OOB = 0.5                                           # encoding: outside the grid


# =============================================================================
# ENVIRONMENT
# =============================================================================

class GridEnvironmentRL:
    def __init__(self, numeric_grid, robot_start, target_pos):
        self.numeric_grid = numeric_grid                 # (rows,cols) array of 0.0/1.0
        self.robot_start  = robot_start                  # (r,c) start cell
        self.target_pos   = target_pos                   # (r,c) goal cell
        self.robot_pos    = robot_start                  # current position
        self.visited      = set()                        # cells seen this episode

    def reset(self):
        self.robot_pos = self.robot_start                # back to start
        self.visited   = {self.robot_start}              # only start is visited
        return self.robot_pos                            # caller builds first state

    def step(self, action):
        dr, dc = ACTIONS[action]                         # move vector for this action
        r, c   = self.robot_pos                          # where we are
        nr, nc = r + dr, c + dc                          # where we'd go
        rows, cols = self.numeric_grid.shape             # grid bounds

        if not (0 <= nr < rows and 0 <= nc < cols):      # off the grid?
            return self.robot_pos, -0.01, False          # stay put, tiny penalty
        if self.numeric_grid[nr, nc] == CELL_OBSTACLE:   # wall ahead?
            return self.robot_pos, -0.01, False          # stay put, tiny penalty

        prev_dist      = abs(r  - self.target_pos[0]) + abs(c  - self.target_pos[1])  # distance before
        self.robot_pos = (nr, nc)                        # commit the move
        new_dist       = abs(nr - self.target_pos[0]) + abs(nc - self.target_pos[1])  # distance after

        if self.robot_pos == self.target_pos:            # arrived?
            return self.robot_pos, +1.0, True            # big reward, done

        shaping         = (prev_dist - new_dist) * 0.01  # +ve if closer, -ve if farther
        revisit_penalty = -0.01 if self.robot_pos in self.visited else 0.0  # discourage loops
        self.visited.add(self.robot_pos)                 # mark visited

        return self.robot_pos, -0.005 + shaping + revisit_penalty, False  # step reward


# =============================================================================
# STATE
# =============================================================================

def get_vision_window(numeric_grid, robot_pos, target_pos):
    rows, cols = numeric_grid.shape                      # grid bounds
    r, c       = robot_pos                               # window centred on robot
    half       = WINDOW_SIZE // 2                        # = 3 for a 7-wide window
    window     = []                                      # will hold 49 values
    for dr in range(-half, half + 1):                    # sweep rows of the window
        for dc in range(-half, half + 1):                # sweep cols of the window
            nr, nc = r + dr, c + dc                      # absolute cell
            if not (0 <= nr < rows and 0 <= nc < cols):
                window.append(CELL_OOB)             # out of bounds = 0.5
            elif (nr, nc) == target_pos:
                window.append(CELL_TARGET)           # target = 3.0
            elif numeric_grid[nr, nc] == CELL_OBSTACLE:
                window.append(CELL_OBSTACLE)         # obstacle = 1.0
            else:
                window.append(0.0)                   # free = 0.0 (no ghosts)
    return window                                        # list of 49 floats


def get_state(robot_pos, numeric_grid, target_pos, rows, cols, visited):
    # state layout must match reinforcement_lesson_5.py exactly so checkpoints
    # (e.g. models/stage3_best.pth) load and behave identically here.
    window = get_vision_window(numeric_grid, robot_pos, target_pos)  # 49 local cells
    pos_r = robot_pos[0] / rows                           # robot row, normalised 0..1
    pos_c = robot_pos[1] / cols                           # robot col, normalised
    tgt_r = target_pos[0] / rows                          # target row, normalised
    tgt_c = target_pos[1] / cols                          # target col → lets it generalise
    explored = len(visited) / (rows * cols)              # fraction of grid seen
    return torch.tensor(window + [pos_r, pos_c, tgt_r, tgt_c, explored],  # concat -> 54
                        dtype=torch.float32)


# =============================================================================
# MODEL
# =============================================================================

class DQN_LSTM(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=128,
                 output_dim=NUM_ACTIONS):
        super().__init__()
        self.hidden_dim = hidden_dim                     # remember LSTM size

        self.encoder = nn.Sequential(                    # state -> 128-d features
            nn.Linear(input_dim, 128),                   # project input
            nn.LayerNorm(128),                           # stabilise activations
            nn.ReLU(),                                    # non-linearity
            nn.Linear(128, 128),                         # second layer
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(                             # carries memory across steps
            input_size=128, hidden_size=hidden_dim,
            num_layers=1, batch_first=True
        )
        self.decoder = nn.Sequential(                    # features -> Q-values
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)                    # one Q per action
        )

    def forward(self, x, h, c):
        encoded         = self.encoder(x)                # [B,128] features
        lstm_in         = encoded.unsqueeze(1)           # [B,1,128] (seq len = 1)
        out, (h_n, c_n) = self.lstm(lstm_in, (h, c))     # run one LSTM step
        out             = out.squeeze(1)                 # [B,hidden]
        return self.decoder(out), h_n, c_n               # Q-values + new memory

    def init_hidden(self, batch_size=1, device='cpu'):
        h = torch.zeros(1, batch_size, self.hidden_dim).to(device)  # zero hidden state
        c = torch.zeros(1, batch_size, self.hidden_dim).to(device)  # zero cell state
        return h, c


# =============================================================================
# REPLAY BUFFER
# =============================================================================

class ReplayBuffer:
    def __init__(self, maxlen=50000):
        self.buffer = deque(maxlen=maxlen)               # auto-drops oldest when full

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))  # store one transition

    def sample(self, batch_size):
        batch       = random.sample(self.buffer, batch_size)            # random minibatch
        states      = torch.stack([e[0] for e in batch])                # [B,54]
        actions     = torch.tensor([e[1] for e in batch], dtype=torch.long)     # [B]
        rewards     = torch.tensor([e[2] for e in batch], dtype=torch.float32)  # [B]
        next_states = torch.stack([e[3] for e in batch])                # [B,54]
        dones       = torch.tensor([e[4] for e in batch], dtype=torch.float32)  # [B]
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)                          # how many stored transitions


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
    from core.grid_utils import generate_random_grid    # imported here to avoid a cycle

    device = next(model.parameters()).device             # match model's device
    model.eval()                                         # disable dropout etc.

    success_count = 0                                    # grids solved
    path_length   = (rows - 1) + (cols - 1)              # min possible path length
    MAX_STEPS     = path_length * 3 + 5                  # step budget per grid

    for _ in range(n_episodes):
        density = random.uniform(density_min, density_max)  # vary difficulty
        while True:                                      # keep trying until solvable
            result = generate_random_grid(rows, cols, density)
            _, numeric, _, robot_start, target = result
            if numeric is not None:
                break

        env    = GridEnvironmentRL(numeric, robot_start, target)
        raw_pos = env.reset()
        state  = get_state(raw_pos, env.numeric_grid, env.target_pos,
                           rows, cols, env.visited)      # first state
        h, c   = model.init_hidden(batch_size=1, device=device)  # fresh memory
        recent = deque(maxlen=10)                        # oscillation detector

        for _ in range(MAX_STEPS):
            state_t = state.unsqueeze(0).to(device)      # add batch dim
            with torch.no_grad():                        # no gradients at eval
                q_vals, h, c = model(state_t, h, c)      # Q-values
                action = torch.argmax(q_vals).item()     # greedy action
            h = h.detach()                               # cut graph between steps
            c = c.detach()

            new_raw, _, done = env.step(action)          # act
            state = get_state(new_raw, env.numeric_grid, env.target_pos,
                              rows, cols, env.visited)
            recent.append(env.robot_pos)
            if len(recent) == 10 and len(set(recent)) <= 2:  # stuck looping?
                break
            if done:                                     # reached target
                success_count += 1
                break

    model.train()                                        # back to train mode
    return success_count / n_episodes * 100              # success rate %


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
    from core.grid_utils import generate_random_grid     # avoid import cycle

    device = torch.device("cpu")                         # demo runs on CPU

    path_length   = (rows - 1) + (cols - 1)              # min path length
    MAX_STEPS     = path_length * 3 + 5                  # step budget per episode
    ALPHA         = 1e-3                                 # learning rate
    GAMMA         = 0.95                                 # discount factor
    EPSILON       = 1.0                                  # start fully exploring
    EPSILON_MIN   = 0.01                                 # exploration floor
    target_ep     = int(episodes * (2 / 3))             # decay over first 2/3
    EPSILON_DECAY = (EPSILON_MIN / EPSILON) ** (1 / target_ep)  # per-episode factor
    BATCH_SIZE    = 32                                   # transitions per update
    MIN_BUFFER    = 500                                  # wait for this many first
    UPDATE_TARGET = 100                                  # sync target net every N eps
    HIDDEN_DIM    = 128                                  # LSTM size

    model        = DQN_LSTM(INPUT_DIM, HIDDEN_DIM, NUM_ACTIONS).to(device)  # online net
    target_model = DQN_LSTM(INPUT_DIM, HIDDEN_DIM, NUM_ACTIONS).to(device)  # stable target net
    target_model.load_state_dict(model.state_dict())     # start identical
    target_model.eval()                                  # target never trains directly

    optimizer     = optim.Adam(model.parameters(), lr=ALPHA)  # updates online net
    loss_fn       = nn.MSELoss()                         # Q vs target-Q error
    replay_buffer = ReplayBuffer()                       # experience memory

    rewards_history     = []                             # reward per episode
    best_avg            = -np.inf                        # best 100-ep avg so far
    consecutive_success = 0                              # streak of good evals
    success_rate        = None                           # last eval result

    # these are yielded so Streamlit can display current episode grid
    current_numeric = None                               # current grid
    current_robot   = None                               # current start
    current_target  = None                               # current target

    model.train()

    for episode in range(episodes):

        # ── new random grid every episode ────────────────────────────────
        density = random.uniform(density_min, density_max)  # vary difficulty
        while True:                                      # retry until solvable
            result = generate_random_grid(rows, cols, density)
            _, numeric, _, robot_start, target = result
            if numeric is not None:
                break

        env = GridEnvironmentRL(numeric, robot_start, target)
        current_numeric = numeric                        # stash for the UI
        current_robot   = robot_start
        current_target  = target
        # ─────────────────────────────────────────────────────────────────

        raw_pos = env.reset()
        state = get_state(raw_pos, env.numeric_grid, env.target_pos,
                            rows, cols, env.visited)      # first state
        h, c = model.init_hidden(batch_size=1, device=device)  # fresh memory

        total_reward = 0                                 # episode return
        episode_loss = 0                                 # summed loss
        train_steps  = 0                                 # gradient steps this episode
        done = False
        last_path = [env.robot_pos]                      # trajectory for the UI

        for _ in range(MAX_STEPS):
            state_t = state.unsqueeze(0).to(device)      # add batch dim

            if random.random() < EPSILON:                # explore?
                action = random.randint(0, NUM_ACTIONS - 1)  # random move
                model.eval()
                with torch.no_grad():
                    _, h, c = model(state_t, h, c)       # still advance memory
                model.train()
            else:                                        # exploit?
                model.eval()
                with torch.no_grad():
                    q_vals, h, c = model(state_t, h, c)
                    action = torch.argmax(q_vals).item() # best predicted move
                model.train()

            h = h.detach()                               # detach memory between steps
            c = c.detach()

            new_raw, reward, done = env.step(action)     # take the step
            new_state = get_state(new_raw, env.numeric_grid, env.target_pos, rows, cols, env.visited)
            total_reward += reward                        # accumulate
            last_path.append(env.robot_pos)

            replay_buffer.push(state, action, reward, new_state, done)  # store experience
            state = new_state                            # advance

            if len(replay_buffer) >= MIN_BUFFER:         # enough data to learn?
                s, a, r, ns, d = replay_buffer.sample(BATCH_SIZE)  # minibatch
                s  = s.to(device); a = a.to(device)
                r  = r.to(device); ns = ns.to(device); d = d.to(device)

                h_tr, c_tr = model.init_hidden(BATCH_SIZE, device)         # zero memory per pass
                h_tg, c_tg = target_model.init_hidden(BATCH_SIZE, device)

                q_all, _, _ = model(s, h_tr, c_tr)                          # Q(s, ·)
                current_q   = q_all.gather(1, a.unsqueeze(1)).squeeze(1)    # Q(s, a_taken)

                with torch.no_grad():
                    nq, _, _ = target_model(ns, h_tg, c_tg)                 # target Q(s', ·)
                    target_q = r + GAMMA * nq.max(1)[0] * (1 - d)           # Bellman target

                loss = loss_fn(current_q, target_q)      # prediction error
                optimizer.zero_grad()                    # clear old grads
                loss.backward()                          # backprop
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stabilise grads
                optimizer.step()                         # update weights

                episode_loss += loss.item()              # track loss
                train_steps  += 1

            if done:                                     # episode over
                break

        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)  # decay exploration
        rewards_history.append(total_reward)

        avg_reward = np.mean(rewards_history[-100:])     # smoothed performance
        avg_loss = episode_loss / train_steps if train_steps > 0 else 0.0  # mean loss

        if avg_reward > best_avg:                        # new best?
            best_avg = avg_reward

        if (episode + 1) % UPDATE_TARGET == 0:           # periodically...
            target_model.load_state_dict(model.state_dict())  # ...copy online -> target

        # ── early stopping check ──────────────────────────────────────────
        early_stop_triggered = False

        if (episode + 1) % eval_every == 0:              # time to evaluate?
            success_rate = evaluate_model_silent(
                model, rows, cols,
                density_min = density_min,
                density_max = density_max,
                n_episodes  = 20
            )
            if success_rate >= early_stop_success:       # good enough?
                consecutive_success += 1
                if consecutive_success >= early_stop_patience:  # consistently good?
                    early_stop_triggered = True
            else:
                consecutive_success = 0                  # reset streak

        # ── yield update ─────────────────────────────────────────────────
        if (episode + 1) % progress_every == 0 or early_stop_triggered:  # refresh UI
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

        if early_stop_triggered:                         # stop early if converged
            break

    # final yield
    yield {                                              # final payload for the UI
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
    device = next(model.parameters()).device             # model's device
    model.eval()                                         # eval mode

    raw_pos = env.reset()
    state   = get_state(raw_pos, env.numeric_grid, env.target_pos,
                        rows, cols, env.visited)          # first state
    h, c    = model.init_hidden(batch_size=1, device=device)  # fresh memory

    path_length  = (rows - 1) + (cols - 1)               # min path length
    MAX_STEPS    = path_length * 3 + 5                   # step budget

    steps  = [env.robot_pos]                             # trajectory to return
    total_reward = 0
    recent  = deque(maxlen=10)                           # oscillation detector
    success      = False
    visited_inf  = {env.robot_pos}   # track visited for inference penalty

    for _ in range(MAX_STEPS):
        state_t = state.unsqueeze(0).to(device)          # batch dim
        with torch.no_grad():
            q_vals, h, c = model(state_t, h, c)          # Q-values

        h = h.detach()                                   # detach memory
        c = c.detach()

        # ── revisit penalty during inference ─────────────────────────────
        # prevents oscillation without retraining
        q_adjusted = q_vals.clone().squeeze(0)           # copy we can tweak
        for action_idx, (dr, dc) in ACTIONS.items():     # check each neighbour
            nr = env.robot_pos[0] + dr
            nc = env.robot_pos[1] + dc

            if (nr, nc) in visited_inf:                  # already been there?
                q_adjusted[action_idx] -= 0.5            # lower its Q
        action = torch.argmax(q_adjusted).item()         # pick adjusted-best
        # ─────────────────────────────────────────────────────────────────

        new_raw, reward, done = env.step(action)         # act
        visited_inf.add(env.robot_pos)                   # remember this cell

        state = get_state(new_raw, env.numeric_grid, env.target_pos,
                          rows, cols, env.visited)
        total_reward += reward
        steps.append(env.robot_pos)

        recent.append(env.robot_pos)
        if len(recent) == 10 and len(set(recent)) <= 2:  # stuck?
            break
        if done:                                         # solved
            success = True
            break

    return steps, total_reward, success                  # path, return, did it reach target


def load_rl_model(path, device='cpu'):
    """Load saved RL model from .pth file."""
    model = DQN_LSTM(INPUT_DIM, 128, NUM_ACTIONS).to(device)  # must match trained dims
    model.load_state_dict(torch.load(path, weights_only=True,
                                      map_location=device))   # load weights
    model.eval()                                         # inference mode
    return model
