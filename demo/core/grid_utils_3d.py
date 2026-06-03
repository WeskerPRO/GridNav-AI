"""
3D grid utilities — the foundation for 3D pathfinding (RL + supervised).

Mirrors the 2D core/grid_utils.py conventions in three dimensions:
    - numeric_grid: np.float32 array, shape (D, H, W), values 0.0 free / 1.0 obstacle
    - positions are (z, y, x) integer tuples
    - robot / target tracked SEPARATELY, never baked into the grid (no ghost values)

Pure logic + visualization — no torch, no training. See docs/3D_PATHFINDING_DESIGN.md.
"""

import numpy as np
import random
from collections import deque

# =============================================================================
# CONSTANTS
# =============================================================================

GRID_3D = (8, 8, 8)          # default (D, H, W) for v1 — CPU-friendly
CELL_FREE     = 0.0
CELL_OBSTACLE = 1.0

# 6-connected moves: face neighbours only (±z, ±y, ±x)
ACTIONS_6 = {
    0: ( 1,  0,  0),   # +Z  up
    1: (-1,  0,  0),   # -Z  down
    2: ( 0,  1,  0),   # +Y
    3: ( 0, -1,  0),   # -Y
    4: ( 0,  0,  1),   # +X
    5: ( 0,  0, -1),   # -X
}
NUM_ACTIONS_3D = len(ACTIONS_6)


# =============================================================================
# PATHFINDING — BFS ground truth (labels for supervised + solvability check)
# =============================================================================

def find_shortest_path_bfs_3d(numeric_grid, start_pos, target_pos, actions=None):
    """
    Shortest path through a 3D grid via BFS.
    Returns list of (z, y, x) positions start→target, or None if unreachable.
    """
    if actions is None:
        actions = ACTIONS_6
    D, H, W = numeric_grid.shape
    queue   = deque([(start_pos, [start_pos])])
    visited = {start_pos}

    while queue:
        (z, y, x), path = queue.popleft()
        if (z, y, x) == target_pos:
            return path
        for dz, dy, dx in actions.values():
            nz, ny, nx = z + dz, y + dy, x + dx
            if (0 <= nz < D and 0 <= ny < H and 0 <= nx < W
                    and numeric_grid[nz, ny, nx] != CELL_OBSTACLE
                    and (nz, ny, nx) not in visited):
                visited.add((nz, ny, nx))
                queue.append(((nz, ny, nx), path + [(nz, ny, nx)]))
    return None


# =============================================================================
# GRID GENERATION — random, guaranteed solvable
# =============================================================================

def generate_random_grid_3d(D=GRID_3D[0], H=GRID_3D[1], W=GRID_3D[2],
                            obstacle_density=0.20, max_tries=100):
    """
    Generate a random solvable 3D grid.

    Returns (numeric_grid, path, robot_pos, target_pos), or
            (None, None, None, None) if no solvable grid found in max_tries.

    numeric_grid contains ONLY 0.0 (free) and 1.0 (obstacle); robot and target
    positions are returned separately and are guaranteed free.
    """
    for _ in range(max_tries):
        numeric_grid = (np.random.random((D, H, W)) < obstacle_density).astype(np.float32)

        free_cells = list(zip(*np.where(numeric_grid == CELL_FREE)))
        if len(free_cells) < 2:
            continue

        robot_pos, target_pos = random.sample(free_cells, 2)
        robot_pos, target_pos = tuple(map(int, robot_pos)), tuple(map(int, target_pos))

        path = find_shortest_path_bfs_3d(numeric_grid, robot_pos, target_pos)
        if path:
            return numeric_grid, path, robot_pos, target_pos

    return None, None, None, None


def grid_from_voxels(obstacle_set, D, H, W):
    """
    Build a numeric grid from a set of (z, y, x) obstacle voxels.
    For the 3D Grid Builder. Robot/target are NOT placed here (tracked separately).
    """
    numeric_grid = np.zeros((D, H, W), dtype=np.float32)
    for (z, y, x) in obstacle_set:
        if 0 <= z < D and 0 <= y < H and 0 <= x < W:
            numeric_grid[z, y, x] = CELL_OBSTACLE
    return numeric_grid


# =============================================================================
# VISUALIZATION — interactive Plotly voxel view
# =============================================================================

# dark industrial palette, matching the 2D demo
COLORS_3D = {
    "obstacle": "#0F3460",
    "robot":    "#E94560",
    "target":   "#00B4D8",
    "path":     "#533483",
    "bg":       "#0D0D1A",
    "axis":     "#1A1A3A",
}


def render_grid_3d_plotly(numeric_grid, robot_pos, target_pos,
                          path_taken=None, cube_size=12):
    """
    Render a 3D grid as an interactive Plotly figure (for st.plotly_chart).

    Obstacles  → semi-transparent cubes
    Robot      → solid marker (red)
    Target     → solid diamond (cyan)
    path_taken → purple line + markers

    plotly is imported lazily so this module stays importable without it.
    """
    import plotly.graph_objects as go

    D, H, W = numeric_grid.shape
    fig = go.Figure()

    # ── obstacles ────────────────────────────────────────────────────────────
    ob = np.argwhere(numeric_grid == CELL_OBSTACLE)
    if len(ob):
        oz, oy, ox = ob[:, 0], ob[:, 1], ob[:, 2]
        fig.add_trace(go.Scatter3d(
            x=ox, y=oy, z=oz, mode="markers", name="obstacle",
            marker=dict(size=cube_size, symbol="square",
                        color=COLORS_3D["obstacle"], opacity=0.30),
            hoverinfo="skip",
        ))

    # ── path ─────────────────────────────────────────────────────────────────
    if path_taken:
        pz, py, px = zip(*path_taken)
        fig.add_trace(go.Scatter3d(
            x=px, y=py, z=pz, mode="lines+markers", name="path",
            line=dict(color=COLORS_3D["path"], width=6),
            marker=dict(size=cube_size * 0.35, color=COLORS_3D["path"]),
        ))

    # ── robot & target ───────────────────────────────────────────────────────
    if robot_pos is not None:
        fig.add_trace(go.Scatter3d(
            x=[robot_pos[2]], y=[robot_pos[1]], z=[robot_pos[0]],
            mode="markers", name="robot",
            marker=dict(size=cube_size * 1.4, color=COLORS_3D["robot"], symbol="circle"),
        ))
    if target_pos is not None:
        fig.add_trace(go.Scatter3d(
            x=[target_pos[2]], y=[target_pos[1]], z=[target_pos[0]],
            mode="markers", name="target",
            marker=dict(size=cube_size * 1.4, color=COLORS_3D["target"], symbol="diamond"),
        ))

    # ── styling ──────────────────────────────────────────────────────────────
    axis = dict(showbackground=True, backgroundcolor=COLORS_3D["bg"],
                gridcolor=COLORS_3D["axis"], zerolinecolor=COLORS_3D["axis"],
                color="#8888AA")
    fig.update_layout(
        paper_bgcolor=COLORS_3D["bg"], plot_bgcolor=COLORS_3D["bg"],
        font=dict(color="#8888AA", family="JetBrains Mono"),
        scene=dict(
            xaxis=dict(axis, title="X", range=[-1, W]),
            yaxis=dict(axis, title="Y", range=[-1, H]),
            zaxis=dict(axis, title="Z", range=[-1, D]),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(font=dict(color="#8888AA")),
    )
    return fig
