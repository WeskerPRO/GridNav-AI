# 3D Grid Pathfinding — Design Doc

Status: **proposal / not yet built**. This extends the existing 2D GridNav
(supervised ResNet + RL DQN-LSTM) into 3D. Goal: agree on architecture before
writing code.

---

## 1. Goal & scope

Navigate a robot from start → target through a 3D grid of free/obstacle voxels,
using the **same two approaches** as 2D so the demo can compare them:

- **Supervised** — imitate 3D BFS optimal paths (Conv3d ResNet).
- **RL** — DQN-LSTM with a 3D local vision window.

Non-goals (v1): continuous space, dynamic obstacles, multi-agent.

## 2. Design principles (carry over from 2D)

1. **BFS is ground truth** — generates supervised labels *and* validates that a
   random grid is solvable before use.
2. **RL sees a local window, not the whole grid** → size-agnostic, generalizes.
3. **Supervised sees the whole grid** but stays size-agnostic via adaptive
   pooling (same fix we just applied in 2D — `AdaptiveAvgPool3d`).
4. **Keep 2D code untouched.** New parallel modules, not edits to existing ones.

## 3. Grid representation

- `numeric_grid`: `np.float32` array shape `(D, H, W)`, values `0.0` free /
  `1.0` obstacle (mirrors 2D's 0/1 convention — no ghost values).
- Positions are `(z, y, x)` integer tuples. Robot/target tracked separately,
  never baked into the grid.
- `generate_random_grid_3d(D, H, W, density)` — same retry-until-solvable loop
  as 2D, using 3D BFS to confirm a path exists.

## 4. Action space

**Recommended v1: 6-connected** (face neighbors only):

| idx | move | (dz,dy,dx) |
|----|------|-----------|
| 0 | +Z up    | ( 1, 0, 0) |
| 1 | −Z down  | (−1, 0, 0) |
| 2 | +Y       | ( 0, 1, 0) |
| 3 | −Y       | ( 0,−1, 0) |
| 4 | +X       | ( 0, 0, 1) |
| 5 | −X       | ( 0, 0,−1) |

Keeps Manhattan distance clean for reward shaping. **26-connected** (incl.
diagonals) is a v2 option — richer paths but 26-way classification is harder and
diagonal moves complicate the obstacle-clipping rules.

## 5. Supervised model (Conv3d ResNet)

Direct 3D analogue of the current `PathPredictionResNet`:

- **Input:** 3-channel 3D tensor `[1, 3, D, H, W]` — obstacle / robot-onehot /
  target-onehot (3D version of `get_input_state_cnn`).
- **Body:** `Conv3d` + `BatchNorm3d` residual blocks (same 32→32→128 shape).
- **Head:** `AdaptiveAvgPool3d((P,P,P))` → flatten → FC → 6 logits.
  Use the *same* size-agnostic trick we just applied in 2D (P>1 to preserve
  position; start `POOL_SIZE_3D = 4`).
- **Cost note:** Conv3d + 3D pooling is memory-heavy. Keep grids modest in v1
  (e.g. ≤ 12³) and channels small.

## 6. RL model (DQN-LSTM, 3D window)

Architecture is unchanged — only the **state vector** grows:

- **3D vision window** `5×5×5 = 125` voxels around the robot (encoding mirrors
  2D: free 0.0, obstacle 1.0, OOB 0.5, target 3.0 if in window).
- + `pos (z,y,x)` normalized (3) + `target (z,y,x)` normalized (3) +
  `explored` (1) → **INPUT_DIM = 132**.
- Everything else (encoder → LSTM → decoder, replay buffer, target network,
  ε-greedy, early stopping) is identical to `reinforcement_lesson_5.py`.
- **Reward shaping:** same potential-based form, Manhattan distance in 3D
  (`|dz|+|dy|+|dx|`). Reuse the tunable reward constants pattern.

## 7. Visualization (the genuinely hard part)

2D rendered PIL images; 3D can't. Options:

| Option | Pros | Cons |
|--------|------|------|
| **Plotly 3D** (`go.Scatter3d` / `Volume`) | interactive rotate/zoom, native `st.plotly_chart`, animatable | obstacle clutter needs opacity tuning |
| matplotlib `voxels` | simple, static | no interactivity, slow to redraw per frame |

**Recommendation:** Plotly. Render obstacles as semi-transparent voxels, robot
& target as solid markers, path as a 3D line. Animate by re-pushing the figure
per step (like the 2D `st.empty()` loop).

## 8. File layout (keeps 2D intact)

```
demo/core/grid_utils_3d.py        # 3D grid gen, 3D BFS, plotly render
demo/core/rl_model_3d.py          # GridEnvironmentRL3D, get_state, DQN-LSTM, train/infer
demo/core/supervised_model_3d.py  # Conv3d ResNet, 3D dataset, train/infer
demo/pages/4_3D_Pathfinding.py    # new Streamlit page (build / train / infer / compare)
src/reinforcement_lesson_3d.py    # standalone RL script, mirrors lesson_5 (optional)
```

## 9. Milestones (RL-first)

1. **Core 3D mechanics** — `grid_utils_3d.py`: grid gen + 3D BFS + Plotly render.
   Verify visually with a hand-made grid.
2. **RL in 3D** — `rl_model_3d.py`: env + 132-dim state + DQN-LSTM, train on 8³,
   confirm it learns (reuse lesson_5 loop + tunable rewards + diagnostics).
3. **Streamlit page (RL only)** — `pages/4_3D_Pathfinding.py`: build → train →
   infer → animate. Ship something runnable early.
4. **Supervised in 3D** — `supervised_model_3d.py`: Conv3d ResNet + BFS dataset;
   add as a second tab on the same page.
5. **Polish** — density/failure-mode diagnostics, compare-tab vs BFS.

## 10. Decisions (LOCKED)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Grid size (v1) | **8×8×8** (512 voxels) | CPU-friendly; proves concept, scale later |
| Connectivity | **6-connected** | clean Manhattan shaping, simple 6-way head |
| Layout / order | **RL-first, one combined page** | RL is light on CPU; ship runnable early |
| Compute | **CPU only** | no CUDA available — keeps grids small, RL before Conv3d |

Concrete constants for v1:
- `GRID_3D = (8, 8, 8)`, `WINDOW_3D = 5` → window `5³ = 125`
- RL `INPUT_DIM_3D = 125 + 3 (pos) + 3 (target) + 1 (explored) = 132`
- `NUM_ACTIONS_3D = 6`
- Supervised `POOL_SIZE_3D = 4` → FC input `128 · 4³ = 8192`
</content>
