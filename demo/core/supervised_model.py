import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from torch.utils.data import Dataset, DataLoader, random_split

# =============================================================================
# CONSTANTS
# =============================================================================

ACTIONS_8 = {
    0: (-1,  0),   # UP
    1: ( 1,  0),   # DOWN
    2: ( 0, -1),   # LEFT
    3: ( 0,  1),   # RIGHT
    4: (-1, -1),   # UP-LEFT
    5: (-1,  1),   # UP-RIGHT
    6: ( 1, -1),   # DOWN-LEFT
    7: ( 1,  1),   # DOWN-RIGHT
}
NUM_ACTIONS_8 = 8
CELL_OBSTACLE = 1.0


# =============================================================================
# MODEL — your exact ResNet architecture from path_finder.py
# =============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.final_relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out      = self.main_path(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.final_relu(out)


class PathPredictionResNet(nn.Module):
    """
    ResNet CNN — exact architecture from path_finder.py.
    Input:  3-channel grid tensor (obstacles, robot, target)
    Output: 8 action logits
    Uses Dropout(0.6) to prevent overfitting.
    """
    def __init__(self, rows, cols, num_actions=NUM_ACTIONS_8):
        super(PathPredictionResNet, self).__init__()
        self.rows = rows
        self.cols = cols

        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # res_block1: 32 → 32, no channel change, no downsample needed
        self.res_block1 = ResidualBlock(32, 32)

        # res_block2: 32 → 128, channel change needs 1×1 downsample
        self.res_block2 = ResidualBlock(
            32, 128,
            downsample=nn.Sequential(
                nn.Conv2d(32, 128, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(128),
            )
        )

        self._flattened_dim = 128 * rows * cols

        self.fc = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(self._flattened_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = x.view(-1, self._flattened_dim)
        return self.fc(x)


# =============================================================================
# STATE PREPARATION
# =============================================================================

def get_input_state_cnn(numeric_grid, robot_pos, target_pos):
    """
    Convert grid state to 3-channel tensor for ResNet.
    Channel 0: obstacle map  (1.0 where obstacle, 0.0 elsewhere)
    Channel 1: robot position (1.0 at robot cell, 0.0 elsewhere)
    Channel 2: target position (1.0 at target cell, 0.0 elsewhere)
    """
    rows, cols = numeric_grid.shape

    obstacle_ch = (numeric_grid == CELL_OBSTACLE).astype(np.float32)

    robot_ch = np.zeros((rows, cols), dtype=np.float32)
    if robot_pos:
        robot_ch[robot_pos[0], robot_pos[1]] = 1.0

    target_ch = np.zeros((rows, cols), dtype=np.float32)
    if target_pos:
        target_ch[target_pos[0], target_pos[1]] = 1.0

    return torch.tensor(
        np.stack([obstacle_ch, robot_ch, target_ch]),
        dtype=torch.float32
    ).unsqueeze(0)   # [1, 3, rows, cols]


# =============================================================================
# DATASET — BFS imitation learning
# =============================================================================

class PathfindingDataset(Dataset):
    """
    Generates (state_tensor, action) pairs from BFS optimal paths.
    Keeps generating random grids until num_samples collected.
    Uses random_split for unbiased train/val split.
    Actions stored as torch.long tensors (correct for CrossEntropyLoss).
    """
    def __init__(self, rows, cols, obstacle_density=0.20, num_samples=3000):
        from core.grid_utils import generate_random_grid

        self.data = []
        maps_tried = 0

        print(f"Generating dataset — target: {num_samples} samples...")

        while len(self.data) < num_samples:
            result = generate_random_grid(rows, cols, obstacle_density)
            _, numeric, path, _, target = result
            maps_tried += 1

            if path is None:
                continue

            for i in range(len(path) - 1):
                curr  = path[i]
                next_ = path[i + 1]
                dr    = next_[0] - curr[0]
                dc    = next_[1] - curr[1]

                action = None
                for a, (adr, adc) in ACTIONS_8.items():
                    if adr == dr and adc == dc:
                        action = a
                        break

                if action is None:
                    continue

                state = get_input_state_cnn(numeric, curr, target).squeeze(0)
                self.data.append((
                    state,
                    torch.tensor(action, dtype=torch.long)
                ))

            if maps_tried % 200 == 0:
                print(f"  {len(self.data)} samples from {maps_tried} maps...")

        if len(self.data) == 0:
            raise ValueError("No training data generated. Check parameters.")

        print(f"Dataset ready: {len(self.data)} samples from {maps_tried} maps.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# =============================================================================
# TRAINING — generator, yields progress for Streamlit live updates
# =============================================================================

def train_supervised_live(rows, cols,
                           obstacle_density = 0.20,
                           epochs           = 20,
                           num_samples      = 3000,
                           batch_size       = 16,
                           lr               = 5e-4,
                           weight_decay     = 5e-3,
                           val_split        = 0.2,
                           progress_every   = 1):
    """
    Generator — yields training state every epoch.
    Uses your exact hyperparameters from path_finder.py:
        lr=5e-4, weight_decay=5e-3, Dropout(0.6), batch_size=16
        random_split for unbiased train/val split
        ReduceLROnPlateau scheduler
    """
    device    = torch.device("cpu")
    model     = PathPredictionResNet(rows, cols).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3
    )
    criterion = nn.CrossEntropyLoss()

    # generate dataset
    dataset    = PathfindingDataset(rows, cols, obstacle_density, num_samples)

    # random_split — unbiased, shuffled split
    val_size   = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses   = []
    best_val     = float('inf')

    for epoch in range(epochs):

        # ── training ──────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0
        for states, actions in train_loader:
            states  = states.to(device)
            actions = actions.to(device)
            logits  = model(states)
            loss    = criterion(logits, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)

        # ── validation ────────────────────────────────────────────────────
        model.eval()
        val_loss = 0
        correct  = 0
        total    = 0
        with torch.no_grad():
            for states, actions in val_loader:
                states  = states.to(device)
                actions = actions.to(device)
                logits  = model(states)
                val_loss += criterion(logits, actions).item()
                preds    = torch.argmax(logits, dim=1)
                correct  += (preds == actions).sum().item()
                total    += len(actions)

        avg_val  = val_loss / len(val_loader)
        accuracy = correct / total * 100
        val_losses.append(avg_val)

        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val

        if (epoch + 1) % progress_every == 0:
            yield {
                "epoch":        epoch + 1,
                "total_epochs": epochs,
                "train_loss":   avg_train,
                "val_loss":     avg_val,
                "accuracy":     accuracy,
                "best_val":     best_val,
                "train_losses": train_losses.copy(),
                "val_losses":   val_losses.copy(),
                "model":        model,
                "done_training": False,
            }

    yield {
        "epoch":        epochs,
        "total_epochs": epochs,
        "train_loss":   avg_train,
        "val_loss":     avg_val,
        "accuracy":     accuracy,
        "best_val":     best_val,
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "model":        model,
        "done_training": True,
    }


# =============================================================================
# INFERENCE
# =============================================================================

def run_supervised_inference(model, numeric_grid, robot_start,
                              target_pos, rows, cols):
    """
    Run trained ResNet on grid.
    Returns (path, success).
    Loop detection prevents infinite oscillation.
    """
    device    = next(model.parameters()).device
    model.eval()

    path_length = (rows - 1) + (cols - 1)
    MAX_STEPS   = path_length * 3 + 5

    robot_pos = robot_start
    path      = [robot_pos]
    recent    = deque(maxlen=10)

    for _ in range(MAX_STEPS):
        if robot_pos == target_pos:
            return path, True

        state_t = get_input_state_cnn(
            numeric_grid, robot_pos, target_pos
        ).to(device)

        with torch.no_grad():
            logits = model(state_t)
            action = torch.argmax(logits, dim=1).item()

        dr, dc = ACTIONS_8[action]
        r, c   = robot_pos
        nr, nc = r + dr, c + dc

        # only move if valid — stay in place if model picks wall/OOB
        if (0 <= nr < rows and 0 <= nc < cols
                and numeric_grid[nr, nc] != CELL_OBSTACLE):
            robot_pos = (nr, nc)

        path.append(robot_pos)

        recent.append(robot_pos)
        if len(recent) == 10 and len(set(recent)) <= 2:
            break   # oscillation → stop

    success = (robot_pos == target_pos)
    return path, success


def load_supervised_model(path, rows, cols, device='cpu'):
    """Load saved supervised model from .pth file."""
    model = PathPredictionResNet(rows, cols, NUM_ACTIONS_8).to(device)
    model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    model.eval()
    return model