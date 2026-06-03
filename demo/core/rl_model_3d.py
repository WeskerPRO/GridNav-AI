"""
3D reinforcement-learning navigation — the 3D twin of core/rl_model.py.

Same DQN-LSTM recipe, just in three dimensions:
    - state  = 5x5x5 vision window (125) + pos(3) + target(3) + explored(1) = 132
      (free cells the robot already visited are marked 0.25 so it can SEE its trail
       and stop looping — deadlock is a memory problem, not a sight problem)
    - actions = 6 (face neighbours, from grid_utils_3d.ACTIONS_6)

The model (DQN_LSTM) and ReplayBuffer are REUSED from core.rl_model — they are
generic (parameterised by input_dim / output_dim), so we don't duplicate them.
See docs/3D_PATHFINDING_DESIGN.md.
"""

import torch                                              # tensors + autograd
import torch.nn as nn                                     # loss function lives here
import torch.optim as optim                               # Adam optimiser
import numpy as np                                        # grid maths
import random                                             # epsilon-greedy + sampling
from collections import deque                             # fixed-length "recent" buffer

from core.rl_model import DQN_LSTM, ReplayBuffer          # reused, not re-written
from core.grid_utils_3d import (ACTIONS_6, NUM_ACTIONS_3D,  # 6 moves
                                CELL_OBSTACLE,             # 1.0
                                generate_random_grid_3d)   # random solvable grids

# =============================================================================
# CONSTANTS
# =============================================================================

WINDOW_3D    = 5                                          # 5x5x5 cube around robot
WINDOW_CELLS = WINDOW_3D ** 3                             # = 125 voxels
INPUT_DIM_3D = WINDOW_CELLS + 7                           # 125 + pos(3) + target(3) + explored(1) = 132

CELL_OOB     = 0.5                                        # value for "outside the grid"
CELL_TARGET  = 3.0                                        # value for "target is here" inside window
CELL_VISITED = 0.25                                       # free cell already visited (the robot's own trail)

# ── reward shaping config (mirrors reinforcement_lesson_5.py) ─────────────────
REWARD_GOAL      = 1.0                                    # reached target (terminal)
REWARD_COLLISION = -0.01                                  # hit wall / obstacle / out of bounds
STEP_COST        = -0.005                                 # per-step cost → prefer shorter paths
SHAPING_COEF     = 0.01                                   # (prev_dist - new_dist) * COEF, potential-based
REVISIT_PENALTY  = -0.02                                  # stepping onto an already-visited voxel
TIMEOUT_PENALTY  = -0.25                                  # final transition if MAX_STEPS hit without goal


# =============================================================================
# ENVIRONMENT
# =============================================================================

class GridEnvironmentRL3D:
    def __init__(self, numeric_grid, robot_start, target_pos):
        self.numeric_grid = numeric_grid                 # (D,H,W) array of 0.0/1.0
        self.robot_start  = robot_start                  # (z,y,x) start
        self.target_pos   = target_pos                   # (z,y,x) goal
        self.robot_pos    = robot_start                  # current position
        self.visited      = set()                        # voxels seen this episode

    def reset(self):
        self.robot_pos = self.robot_start                # back to start
        self.visited   = {self.robot_start}              # only the start is "visited"
        return self.robot_pos                            # caller builds the first state

    def _manhattan(self, a, b):                          # 3D L1 distance
        return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])

    def step(self, action):
        dz, dy, dx = ACTIONS_6[action]                   # look up the move vector
        z, y, x    = self.robot_pos                      # where we are now
        nz, ny, nx = z + dz, y + dy, x + dx              # where we'd move to
        D, H, W    = self.numeric_grid.shape             # grid bounds

        if not (0 <= nz < D and 0 <= ny < H and 0 <= nx < W):  # off the grid?
            return self.robot_pos, REWARD_COLLISION, False     # stay put, small penalty
        if self.numeric_grid[nz, ny, nx] == CELL_OBSTACLE:     # wall ahead?
            return self.robot_pos, REWARD_COLLISION, False     # stay put, small penalty

        prev_dist      = self._manhattan(self.robot_pos, self.target_pos)  # distance before
        self.robot_pos = (nz, ny, nx)                    # commit the move
        new_dist       = self._manhattan(self.robot_pos, self.target_pos)  # distance after

        if self.robot_pos == self.target_pos:            # arrived?
            return self.robot_pos, REWARD_GOAL, True     # big reward, episode done

        shaping = (prev_dist - new_dist) * SHAPING_COEF  # +ve if we got closer, -ve if farther
        revisit = REVISIT_PENALTY if self.robot_pos in self.visited else 0.0  # discourage loops
        self.visited.add(self.robot_pos)                 # mark this voxel visited

        return self.robot_pos, STEP_COST + shaping + revisit, False  # combined step reward


# =============================================================================
# STATE  (132-dim vector)
# =============================================================================

def get_vision_window_3d(numeric_grid, robot_pos, target_pos, visited=None, window=WINDOW_3D):
    D, H, W = numeric_grid.shape                         # grid bounds
    z, y, x = robot_pos                                  # window is centred on robot
    half    = window // 2                                # = 2 for a 5-wide window
    visited = visited or set()                           # cells already stepped on
    cells   = []                                         # will hold 125 values

    for dz in range(-half, half + 1):                    # sweep z offset
        for dy in range(-half, half + 1):                # sweep y offset
            for dx in range(-half, half + 1):            # sweep x offset
                nz, ny, nx = z + dz, y + dy, x + dx      # absolute voxel coords
                if not (0 <= nz < D and 0 <= ny < H and 0 <= nx < W):
                    cells.append(CELL_OOB)               # 0.5 outside the grid
                elif (nz, ny, nx) == target_pos:
                    cells.append(CELL_TARGET)            # 3.0 if target is visible here
                elif numeric_grid[nz, ny, nx] == CELL_OBSTACLE:
                    cells.append(CELL_OBSTACLE)          # 1.0 obstacle
                elif (nz, ny, nx) in visited:
                    cells.append(CELL_VISITED)           # 0.25 been here -> the trail
                else:
                    cells.append(0.0)                    # 0.0 fresh free
    return cells                                         # list of 125 floats


def get_state_3d(robot_pos, numeric_grid, target_pos, D, H, W, visited):
    window   = get_vision_window_3d(numeric_grid, robot_pos, target_pos, visited)  # 125 local voxels (trail-aware)
    pos      = [robot_pos[0]/D, robot_pos[1]/H, robot_pos[2]/W]   # robot coords, normalised 0..1
    tgt      = [target_pos[0]/D, target_pos[1]/H, target_pos[2]/W]  # target coords → lets it generalise
    explored = [len(visited) / (D * H * W)]              # fraction of grid already seen
    return torch.tensor(window + pos + tgt + explored,  # concat → length 132
                        dtype=torch.float32)


# =============================================================================
# EVALUATION — silent, used for early stopping
# =============================================================================

def evaluate_model_silent_3d(model, D, H, W, density_min=0.10, density_max=0.35,
                             n_episodes=20):
    device = next(model.parameters()).device            # run eval on model's device
    model.eval()                                         # disable dropout etc.
    success = 0                                          # count solved grids
    MAX_STEPS = ((D-1)+(H-1)+(W-1)) * 3 + 5              # step budget per episode

    for _ in range(n_episodes):
        density = random.uniform(density_min, density_max)  # vary difficulty
        grid, _, robot, target = generate_random_grid_3d(D, H, W, density)  # solvable grid
        env   = GridEnvironmentRL3D(grid, robot, target)
        state = get_state_3d(env.reset(), env.numeric_grid, env.target_pos, D, H, W, env.visited)
        h, c  = model.init_hidden(batch_size=1, device=device)  # fresh LSTM memory
        recent = deque(maxlen=10)                        # for oscillation detection

        for _ in range(MAX_STEPS):
            with torch.no_grad():                        # no gradients at inference
                q, h, c = model(state.unsqueeze(0).to(device), h, c)  # Q-values for 6 actions
                action  = torch.argmax(q).item()         # greedy: best action
            h, c = h.detach(), c.detach()                # cut the graph between steps
            new_pos, _, done = env.step(action)          # act
            state = get_state_3d(new_pos, env.numeric_grid, env.target_pos, D, H, W, env.visited)
            recent.append(env.robot_pos)
            if len(recent) == 10 and len(set(recent)) <= 2:  # stuck oscillating?
                break                                    # give up this grid
            if done:                                     # reached target
                success += 1
                break

    model.train()                                        # back to train mode
    return success / n_episodes * 100                    # success rate %


# =============================================================================
# DIAGNOSTIC — success / failure type by grid density (3D twin of 2D diagnose_eval_set)
# =============================================================================

def diagnose_eval_3d(model, D, H, W, density_min=0.10, density_max=0.25,
                     n_grids=60, n_buckets=3):
    """
    Run the model on n_grids random grids, bucket the OUTCOME by density.
    Returns a list of row-dicts (per bucket + an OVERALL row) ready for st.table.

    outcome per grid: success / timeout / deadlock — so you can see WHERE 3D
    failures concentrate and WHY:
        deadlock-heavy dense buckets -> observability limit (bigger window)
        timeout-heavy buckets        -> step budget too small
    """
    device = next(model.parameters()).device             # model device
    was_training = model.training                        # restore mode afterwards
    model.eval()
    MAX_STEPS = ((D-1)+(H-1)+(W-1)) * 3 + 5               # step budget per grid

    records = []                                         # (density, outcome) per grid
    for _ in range(n_grids):
        density = random.uniform(density_min, density_max)  # vary difficulty
        grid, _, robot, target = generate_random_grid_3d(D, H, W, density)  # solvable grid
        env   = GridEnvironmentRL3D(grid, robot, target)
        state = get_state_3d(env.reset(), env.numeric_grid, env.target_pos, D, H, W, env.visited)
        h, c  = model.init_hidden(batch_size=1, device=device)  # fresh memory
        recent = deque(maxlen=10)                        # oscillation detector

        outcome = "timeout"                              # default if the loop exhausts
        for _ in range(MAX_STEPS):
            with torch.no_grad():
                q, h, c = model(state.unsqueeze(0).to(device), h, c)  # Q-values
                action  = torch.argmax(q).item()         # greedy
            h, c = h.detach(), c.detach()
            new_pos, _, done = env.step(action)          # act
            state = get_state_3d(new_pos, env.numeric_grid, env.target_pos, D, H, W, env.visited)
            recent.append(env.robot_pos)
            if len(recent) == 10 and len(set(recent)) <= 2:  # stuck oscillating
                outcome = "deadlock"; break
            if done:                                     # reached target
                outcome = "success"; break
        records.append((float(grid.mean()), outcome))   # density = obstacle fraction

    # bucket by density (equal-width bins over the observed range)
    densities = [d for d, _ in records]
    lo, hi = min(densities), max(densities)
    span = (hi - lo) or 1e-9                             # avoid divide-by-zero
    buckets = [[] for _ in range(n_buckets)]
    for d, outcome in records:
        idx = min(int((d - lo) / span * n_buckets), n_buckets - 1)  # clamp max into last bin
        buckets[idx].append(outcome)

    rows = []                                            # one dict per bucket
    tot_s = tot_t = tot_d = 0
    for i in range(n_buckets):
        b = buckets[i]
        s, t, dl = b.count("success"), b.count("timeout"), b.count("deadlock")  # tally outcomes
        tot_s += s; tot_t += t; tot_d += dl
        b_lo, b_hi = lo + span*i/n_buckets, lo + span*(i+1)/n_buckets  # bucket density range
        rows.append({"Density": f"{b_lo:.0%}-{b_hi:.0%}", "n": len(b),
                     "Success %": f"{(s/len(b)*100) if b else 0:.0f}",
                     "Timeout": t, "Deadlock": dl})
    N = len(records)
    rows.append({"Density": "OVERALL", "n": N,           # summary row
                 "Success %": f"{tot_s/N*100:.0f}" if N else "0",
                 "Timeout": tot_t, "Deadlock": tot_d})

    if was_training:                                    # restore train mode if needed
        model.train()
    return rows


# =============================================================================
# TRAINING — generator, yields progress for the Streamlit page
# =============================================================================

def train_rl_3d_live(D=8, H=8, W=8, density_min=0.10, density_max=0.35,
                     episodes=4000, progress_every=10,
                     early_stop_success=90.0, early_stop_patience=3,
                     eval_every=500):
    device = torch.device("cpu")                         # no CUDA here → CPU

    MAX_STEPS     = ((D-1)+(H-1)+(W-1)) * 3 + 5           # step budget per episode
    ALPHA         = 1e-3                                  # learning rate
    GAMMA         = 0.95                                  # discount on future reward
    EPSILON       = 1.0                                   # start fully exploring
    EPSILON_MIN   = 0.01                                  # floor on exploration
    target_ep     = int(episodes * (2/3))                # decay ε over first 2/3
    EPSILON_DECAY = (EPSILON_MIN / EPSILON) ** (1/target_ep)  # per-episode multiplier
    BATCH_SIZE    = 32                                    # transitions per gradient step
    MIN_BUFFER    = 500                                   # wait until buffer has this many
    UPDATE_TARGET = 100                                   # sync target net every N episodes
    HIDDEN_DIM    = 128                                   # LSTM hidden size

    model        = DQN_LSTM(INPUT_DIM_3D, HIDDEN_DIM, NUM_ACTIONS_3D).to(device)  # online net
    target_model = DQN_LSTM(INPUT_DIM_3D, HIDDEN_DIM, NUM_ACTIONS_3D).to(device)  # stable target net
    target_model.load_state_dict(model.state_dict())     # start identical
    target_model.eval()                                  # target net never trains directly

    optimizer = optim.Adam(model.parameters(), lr=ALPHA) # updates the online net
    loss_fn   = nn.MSELoss()                             # Q vs target-Q error
    buffer    = ReplayBuffer()                           # experience replay

    rewards_history = []                                 # total reward per episode
    best_avg        = -np.inf                            # best 100-ep average so far
    consec_success  = 0                                  # consecutive good evals
    success_rate    = None                               # last eval result

    model.train()

    for episode in range(episodes):
        density = random.uniform(density_min, density_max)  # new difficulty each episode
        grid, _, robot, target = generate_random_grid_3d(D, H, W, density)  # NEW grid each episode → generalise
        env = GridEnvironmentRL3D(grid, robot, target)
        state = get_state_3d(env.reset(), env.numeric_grid, env.target_pos, D, H, W, env.visited)
        h, c  = model.init_hidden(batch_size=1, device=device)  # fresh memory

        total_reward = 0                                 # episode return
        last_path    = [env.robot_pos]                   # for live animation
        done         = False

        for step_i in range(MAX_STEPS):
            if random.random() < EPSILON:                # explore?
                action = random.randint(0, NUM_ACTIONS_3D - 1)  # random move
                with torch.no_grad():
                    _, h, c = model(state.unsqueeze(0).to(device), h, c)  # still advance LSTM memory
            else:                                        # exploit?
                with torch.no_grad():
                    q, h, c = model(state.unsqueeze(0).to(device), h, c)
                    action = torch.argmax(q).item()      # best predicted move
            h, c = h.detach(), c.detach()                # detach memory between steps

            new_pos, reward, done = env.step(action)     # take the step
            new_state = get_state_3d(new_pos, env.numeric_grid, env.target_pos, D, H, W, env.visited)

            if not done and step_i == MAX_STEPS - 1:     # ran out of steps?
                reward += TIMEOUT_PENALTY                # punish the timeout
                done = True                              # mark terminal so penalty isn't bootstrapped away

            total_reward += reward                       # accumulate return
            last_path.append(env.robot_pos)
            buffer.push(state, action, reward, new_state, done)  # store experience
            state = new_state                            # advance

            if len(buffer) >= MIN_BUFFER:                # enough data to learn?
                s, a, r, ns, d = buffer.sample(BATCH_SIZE)  # random minibatch
                s, a, r, ns, d = s.to(device), a.to(device), r.to(device), ns.to(device), d.to(device)

                h_tr, c_tr = model.init_hidden(BATCH_SIZE, device)   # zero memory for batch pass
                h_tg, c_tg = target_model.init_hidden(BATCH_SIZE, device)

                q_all, _, _ = model(s, h_tr, c_tr)                   # Q(s, ·)
                current_q   = q_all.gather(1, a.unsqueeze(1)).squeeze(1)  # Q(s, a_taken)

                with torch.no_grad():
                    nq, _, _ = target_model(ns, h_tg, c_tg)          # target Q(s', ·)
                    target_q = r + GAMMA * nq.max(1)[0] * (1 - d)    # Bellman target (0 future if done)

                loss = loss_fn(current_q, target_q)                  # how wrong were we
                optimizer.zero_grad()                                # clear old grads
                loss.backward()                                      # backprop
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stabilise (no exploding grads)
                optimizer.step()                                     # update weights

            if done:                                     # episode over
                break

        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)  # decay exploration
        rewards_history.append(total_reward)
        avg_reward = np.mean(rewards_history[-100:])     # smoothed performance
        best_avg = max(best_avg, avg_reward)

        if (episode + 1) % UPDATE_TARGET == 0:           # periodically...
            target_model.load_state_dict(model.state_dict())  # ...copy online → target

        early_stop = False
        if (episode + 1) % eval_every == 0:              # time to evaluate?
            success_rate = evaluate_model_silent_3d(model, D, H, W, density_min, density_max)
            if success_rate >= early_stop_success:       # good enough?
                consec_success += 1
                if consec_success >= early_stop_patience:  # consistently good?
                    early_stop = True
            else:
                consec_success = 0                       # reset the streak

        if (episode + 1) % progress_every == 0 or early_stop:  # yield to the UI
            yield {                                      # the Streamlit page reads these keys
                "episode": episode + 1, "total_episodes": episodes,
                "avg_reward": avg_reward, "best_avg": best_avg,
                "epsilon": EPSILON, "rewards_history": rewards_history.copy(),
                "last_path": last_path, "model": model,
                "success_rate": success_rate, "current_grid": grid,
                "current_robot": robot, "current_target": target,
                "done_training": False,
            }

        if early_stop:                                   # stop early if converged
            break

    yield {                                              # final payload
        "episode": len(rewards_history), "total_episodes": episodes,
        "avg_reward": np.mean(rewards_history[-100:]), "best_avg": best_avg,
        "epsilon": EPSILON, "rewards_history": rewards_history,
        "last_path": last_path, "model": model,
        "success_rate": success_rate, "current_grid": grid,
        "current_robot": robot, "current_target": target,
        "done_training": True,
    }


# =============================================================================
# INFERENCE
# =============================================================================

def run_rl_inference_3d(model, env, D, H, W):
    device = next(model.parameters()).device
    model.eval()
    state = get_state_3d(env.reset(), env.numeric_grid, env.target_pos, D, H, W, env.visited)
    h, c  = model.init_hidden(batch_size=1, device=device)

    MAX_STEPS   = ((D-1)+(H-1)+(W-1)) * 3 + 5            # step budget
    path        = [env.robot_pos]                       # trajectory to return
    total_reward = 0
    recent      = deque(maxlen=10)                      # oscillation guard
    visited_inf = {env.robot_pos}                       # for the inference revisit penalty

    for _ in range(MAX_STEPS):
        with torch.no_grad():
            q, h, c = model(state.unsqueeze(0).to(device), h, c)  # Q-values
        h, c = h.detach(), c.detach()

        q_adj = q.clone().squeeze(0)                     # copy we can tweak
        for a, (dz, dy, dx) in ACTIONS_6.items():        # softly discourage stepping back
            nb = (env.robot_pos[0]+dz, env.robot_pos[1]+dy, env.robot_pos[2]+dx)
            if nb in visited_inf:                        # already been there?
                q_adj[a] -= 0.5                          # lower its Q so we explore new voxels
        action = torch.argmax(q_adj).item()             # pick adjusted-best action

        new_pos, reward, done = env.step(action)
        visited_inf.add(env.robot_pos)
        state = get_state_3d(new_pos, env.numeric_grid, env.target_pos, D, H, W, env.visited)
        total_reward += reward
        path.append(env.robot_pos)

        recent.append(env.robot_pos)
        if len(recent) == 10 and len(set(recent)) <= 2:  # stuck?
            break
        if done:                                         # solved
            return path, total_reward, True
    return path, total_reward, False                     # ran out without solving


def load_rl_model_3d(path, device="cpu"):
    model = DQN_LSTM(INPUT_DIM_3D, 128, NUM_ACTIONS_3D).to(device)  # must match trained dims
    model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    model.eval()
    return model
