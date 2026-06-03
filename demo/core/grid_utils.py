import numpy as np                                      # grids as arrays
import random                                            # random obstacles / positions
from collections import deque                            # BFS queue
import matplotlib.pyplot as plt                          # reward-curve figure
from PIL import Image                                    # render grid -> image
import io                                                # in-memory PNG bytes

# =============================================================================
# SHARED CONSTANTS
# =============================================================================

CELL_TYPES = {                                           # grid_data symbol -> numeric value
    0:   0.0,   # free
    1:   0.0,   # free
    3:   1.0,   # obstacle
    "R": 2.0,   # robot  (used in grid_data only, stripped from numeric_grid)
    "T": 3.0,   # target (used in grid_data only, stripped from numeric_grid)
}

ACTIONS_4 = {                                            # 4-connected moves (RL agent)
    0: (-1,  0),   # UP
    1: ( 1,  0),   # DOWN
    2: ( 0, -1),   # LEFT
    3: ( 0,  1),   # RIGHT
}

ACTIONS_8 = {                                            # 8-connected moves (supervised model)
    0: (-1,  0),   # UP
    1: ( 1,  0),   # DOWN
    2: ( 0, -1),   # LEFT
    3: ( 0,  1),   # RIGHT
    4: (-1, -1),   # UP-LEFT
    5: (-1,  1),   # UP-RIGHT
    6: ( 1, -1),   # DOWN-LEFT
    7: ( 1,  1),   # DOWN-RIGHT
}

# Color scheme — dark industrial aesthetic
COLORS = {                                               # hex colours for rendering
    "free":      "#1A1A2E",
    "visited":   "#16213E",
    "obstacle":  "#0F3460",
    "robot":     "#E94560",
    "target":    "#00B4D8",
    "path":      "#533483",
    "grid_line": "#0D0D1A",
}


# =============================================================================
# GRID GENERATION
# =============================================================================

def get_grid_numeric(grid_data):
    """Convert grid_data list to float32 numpy array."""
    numeric_grid = np.zeros(
        (len(grid_data), len(grid_data[0])), dtype=np.float32  # empty grid
    )
    for r in range(len(grid_data)):                      # each row
        for c in range(len(grid_data[0])):               # each col
            numeric_grid[r, c] = CELL_TYPES.get(grid_data[r][c], 0.0)  # map symbol -> value
    return numeric_grid


def find_shortest_path_bfs(grid_numeric, start_pos, target_pos, actions=None):
    """BFS shortest path. Returns list of positions or None if no path."""
    if actions is None:
        actions = ACTIONS_4                              # default to 4-connected
    queue   = deque([(start_pos, [start_pos])])          # (cell, path-so-far)
    visited = {start_pos}                                # cells already enqueued
    rows, cols = grid_numeric.shape
    while queue:
        (r, c), path = queue.popleft()                   # FIFO -> shortest first
        if (r, c) == target_pos:                         # reached goal
            return path
        for dr, dc in actions.values():                  # try each neighbour
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and grid_numeric[nr, nc] != 1.0   # not obstacle
                    and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))  # extend the path
    return None                                          # no path exists


def generate_random_grid(rows, cols, obstacle_density=0.2):
    """
    Generate random solvable grid.
    Returns (grid_data, numeric_grid, path, robot_pos, target_pos).

    numeric_grid contains ONLY 0.0 (free) and 1.0 (obstacle).
    Robot and target positions are stripped — tracked separately.
    """
    for _ in range(100):                                 # retry until solvable (or give up)
        grid_data = [[0] * cols for _ in range(rows)]    # start all-free
        for r in range(rows):
            for c in range(cols):
                if random.random() < obstacle_density:   # sprinkle obstacles
                    grid_data[r][c] = 3

        all_coords = [(r, c) for r in range(rows) for c in range(cols)]  # every cell
        random.shuffle(all_coords)                       # random order for placement

        robot_pos = target_pos = None
        for rc in all_coords:                            # first free cell -> robot
            if grid_data[rc[0]][rc[1]] == 0:
                robot_pos = rc
                grid_data[rc[0]][rc[1]] = "R"
                break
        for rc in all_coords:                            # another free cell -> target
            if grid_data[rc[0]][rc[1]] == 0 and rc != robot_pos:
                target_pos = rc
                grid_data[rc[0]][rc[1]] = "T"
                break

        if robot_pos is None or target_pos is None:      # grid too full
            continue

        numeric_grid = get_grid_numeric(grid_data)       # to numeric

        # ── strip R and T — numeric_grid must only have 0.0 and 1.0 ──────
        # robot and target tracked separately via robot_pos / target_pos
        # having 2.0 / 3.0 in numeric_grid causes ghost values after robot moves
        numeric_grid[robot_pos[0], robot_pos[1]] = 0.0
        numeric_grid[target_pos[0], target_pos[1]] = 0.0
        # ─────────────────────────────────────────────────────────────────

        path = find_shortest_path_bfs(numeric_grid, robot_pos, target_pos)  # solvable?
        if path:
            return grid_data, numeric_grid, path, robot_pos, target_pos     # success

    return None, None, None, None, None                  # failed after 100 tries


def grid_from_builder(cell_states, rows, cols, robot_pos, target_pos):
    """
    Convert Grid Builder cell states to numeric grid.
    cell_states: dict {(r,c): 'free'|'obstacle'}

    Returns (grid_data, numeric_grid).
    numeric_grid contains ONLY 0.0 and 1.0 — no R/T values.
    """
    grid_data = [[0] * cols for _ in range(rows)]        # all-free base
    for r in range(rows):
        for c in range(cols):
            if cell_states.get((r, c)) == 'obstacle':    # user-drawn wall
                grid_data[r][c] = 3

    # ── do NOT place "R" or "T" in grid_data ─────────────────────────────
    # robot and target passed separately to get_state() and get_vision_window()
    # placing them here creates 2.0/3.0 values that model was not trained on

    numeric_grid = get_grid_numeric(grid_data)   # only 0.0 and 1.0 ✅
    return grid_data, numeric_grid


# =============================================================================
# VISUALIZATION
# =============================================================================

def image_to_bytes(pil_image):
    """
    Convert PIL Image to PNG bytes.
    Prevents MediaFileStorageError when Streamlit loses PIL Image reference.
    Always use this before st.image() calls.
    """
    buf = io.BytesIO()                                   # in-memory buffer
    pil_image.save(buf, format='PNG')                    # encode as PNG
    buf.seek(0)                                          # rewind for reading
    return buf


def render_grid_image(numeric_grid, robot_pos, target_pos,
                      path_taken=None, cell_size=32):
    """
    Render grid as PIL Image with dark industrial aesthetic.
    numeric_grid must contain only 0.0 (free) and 1.0 (obstacle).
    Robot and target rendered from separate position arguments.
    """
    rows, cols = numeric_grid.shape                      # grid size
    img_h = rows * cell_size                             # image height in px
    img_w = cols * cell_size                             # image width in px

    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)    # blank RGB canvas

    def hex_to_rgb(h):                                   # "#RRGGBB" -> (r,g,b)
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    free_rgb     = hex_to_rgb(COLORS["free"])            # pre-convert colours
    obstacle_rgb = hex_to_rgb(COLORS["obstacle"])
    robot_rgb    = hex_to_rgb(COLORS["robot"])
    target_rgb   = hex_to_rgb(COLORS["target"])
    path_rgb     = hex_to_rgb(COLORS["path"])
    line_rgb     = hex_to_rgb(COLORS["grid_line"])

    path_set = set(path_taken[:-1]) if path_taken else set()  # trail cells (excl. current)

    for r in range(rows):                                # paint every cell
        for c in range(cols):
            y1, y2 = r * cell_size, (r + 1) * cell_size  # cell pixel bounds
            x1, x2 = c * cell_size, (c + 1) * cell_size

            if (r, c) == robot_pos:                      # priority: robot
                color = robot_rgb
            elif (r, c) == target_pos:                   # then target
                color = target_rgb
            elif numeric_grid[r, c] == 1.0:       # obstacle
                color = obstacle_rgb
            elif (r, c) in path_set:                     # then trail
                color = path_rgb
            else:
                color = free_rgb                         # else free

            img[y1:y2, x1:x2] = color                    # fill the cell

            # grid lines
            img[y1,   x1:x2] = line_rgb                  # top edge
            img[y2-1, x1:x2] = line_rgb                  # bottom edge
            img[y1:y2, x1]   = line_rgb                  # left edge
            img[y1:y2, x2-1] = line_rgb                  # right edge

    # robot as circle
    if robot_pos:
        r, c   = robot_pos
        cy     = r * cell_size + cell_size // 2          # circle centre y
        cx     = c * cell_size + cell_size // 2          # circle centre x
        radius = cell_size // 2 - 3
        for dy in range(-radius, radius + 1):            # scan bounding box
            for dx in range(-radius, radius + 1):
                if dy*dy + dx*dx <= radius*radius:       # inside the circle?
                    py, px = cy + dy, cx + dx
                    if 0 <= py < img_h and 0 <= px < img_w:
                        img[py, px] = robot_rgb

    # target as diamond
    if target_pos:
        r, c = target_pos
        cy   = r * cell_size + cell_size // 2            # diamond centre
        cx   = c * cell_size + cell_size // 2
        half = cell_size // 2 - 3
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                if abs(dy) + abs(dx) <= half:            # Manhattan disk = diamond
                    py, px = cy + dy, cx + dx
                    if 0 <= py < img_h and 0 <= px < img_w:
                        img[py, px] = target_rgb

    return Image.fromarray(img)                          # numpy -> PIL Image


def render_reward_curve(rewards, window=50, title="Reward Curve"):
    """
    Returns matplotlib figure for st.pyplot().
    Call plt.close(fig) immediately after st.pyplot(fig) to prevent memory leak.
    """
    fig, ax = plt.subplots(figsize=(10, 3), facecolor='#0D0D1A')  # dark figure
    ax.set_facecolor('#0D0D1A')

    if len(rewards) > 0:                                 # raw reward (noisy)
        ax.plot(rewards, alpha=0.3, color='#E94560',
                linewidth=0.8, label='Raw')

    if len(rewards) >= window:                           # smoothed moving average
        smoothed = [
            np.mean(rewards[max(0, i - window):i + 1])   # mean of last `window`
            for i in range(len(rewards))
        ]
        ax.plot(smoothed, color='#00B4D8',
                linewidth=2, label=f'Avg ({window} ep)')

    ax.axhline(y=0, color='#533483',                     # zero reference line
               linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Episode",  color='#8888AA', fontsize=9)
    ax.set_ylabel("Reward",   color='#8888AA', fontsize=9)
    ax.set_title(title,       color='#E0E0FF', fontsize=11,
                 fontfamily='monospace')
    ax.tick_params(colors='#8888AA')                     # axis tick colour
    ax.spines['bottom'].set_color('#333355')
    ax.spines['left'].set_color('#333355')
    ax.spines['top'].set_visible(False)                  # hide top/right borders
    ax.spines['right'].set_visible(False)
    ax.legend(facecolor='#1A1A2E', labelcolor='#8888AA', fontsize=8)
    plt.tight_layout()
    return fig
