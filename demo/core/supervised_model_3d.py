"""
3D supervised pathfinding — the 3D twin of core/supervised_model.py.

Same idea, in three dimensions:
    - input  = 3-channel 3D tensor (obstacles, robot one-hot, target one-hot)
    - labels = 3D BFS optimal moves (6-connected)
    - model  = Conv3d ResNet + AdaptiveAvgPool3d head (size-agnostic, like the 2D fix)

Channels are kept small (3 -> 16 -> 64) because Conv3d on CPU is memory-heavy.
See docs/3D_PATHFINDING_DESIGN.md.
"""

import torch                                              # tensors + autograd
import torch.nn as nn                                     # layers + loss
import torch.optim as optim                               # Adam + LR scheduler
import numpy as np                                        # build the input channels
from collections import deque                             # oscillation window at inference
from torch.utils.data import Dataset, DataLoader, random_split  # data pipeline + split

from core.grid_utils_3d import (ACTIONS_6, NUM_ACTIONS_3D,  # 6 moves
                                CELL_OBSTACLE,             # 1.0
                                generate_random_grid_3d,   # random solvable grids
                                find_shortest_path_bfs_3d) # BFS labels / validation

# =============================================================================
# CONSTANTS
# =============================================================================

POOL_SIZE_3D = 4                                          # any grid pooled to 4x4x4 before FC (size-agnostic)


# =============================================================================
# MODEL — Conv3d ResNet
# =============================================================================

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None):
        super().__init__()
        self.main_path = nn.Sequential(                  # two 3D convs (the residual function)
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),                # 3D batch-norm
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )
        self.final_relu = nn.ReLU(inplace=True)          # after the skip add
        self.downsample = downsample                     # reshapes skip if channels change

    def forward(self, x):
        identity = x                                     # skip path
        out      = self.main_path(x)                     # learned residual
        if self.downsample is not None:
            identity = self.downsample(x)                # match channels
        out += identity                                  # residual connection
        return self.final_relu(out)


class PathPredictionResNet3D(nn.Module):
    """
    3D ResNet. Input [B,3,D,H,W] -> 6 action logits.
    Size-agnostic: AdaptiveAvgPool3d pools any grid to POOL_SIZE_3D**3 before the FC head.
    """
    def __init__(self, num_actions=NUM_ACTIONS_3D):
        super().__init__()
        self.initial_conv = nn.Sequential(               # 3 channels -> 16 feature maps
            nn.Conv3d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.res_block1 = ResidualBlock3D(16, 16)        # 16 -> 16 (no channel change)
        self.res_block2 = ResidualBlock3D(               # 16 -> 64 (needs 1x1x1 skip)
            16, 64,
            downsample=nn.Sequential(
                nn.Conv3d(16, 64, kernel_size=1, bias=False),
                nn.BatchNorm3d(64),
            )
        )
        self.spatial_pool = nn.AdaptiveAvgPool3d((POOL_SIZE_3D,) * 3)  # any grid -> fixed cube
        self._flat = 64 * POOL_SIZE_3D ** 3              # fixed FC input size

        self.fc = nn.Sequential(                         # classifier head -> 6 logits
            nn.Dropout(0.3),
            nn.Linear(self._flat, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        x = self.initial_conv(x)    # [B, 16, D, H, W]
        x = self.res_block1(x)      # [B, 16, D, H, W]
        x = self.res_block2(x)      # [B, 64, D, H, W]
        x = self.spatial_pool(x)    # [B, 64, P, P, P]
        x = x.view(x.size(0), -1)   # [B, 64 * P**3] — fixed regardless of grid size
        return self.fc(x)           # action logits


# =============================================================================
# STATE PREPARATION
# =============================================================================

def get_input_state_cnn_3d(numeric_grid, robot_pos, target_pos):
    """3-channel 3D tensor: obstacles, robot one-hot, target one-hot. Shape [1,3,D,H,W]."""
    obstacle_ch = (numeric_grid == CELL_OBSTACLE).astype(np.float32)  # channel 0: walls
    robot_ch    = np.zeros_like(obstacle_ch)             # channel 1: robot
    target_ch   = np.zeros_like(obstacle_ch)             # channel 2: target
    if robot_pos:
        robot_ch[robot_pos] = 1.0                        # mark robot voxel
    if target_pos:
        target_ch[target_pos] = 1.0                      # mark target voxel
    return torch.tensor(np.stack([obstacle_ch, robot_ch, target_ch]),  # stack channels
                        dtype=torch.float32).unsqueeze(0)  # add batch dim


# =============================================================================
# DATASET — 3D BFS imitation learning
# =============================================================================

class PathfindingDataset3D(Dataset):
    """(state, action) pairs from 3D BFS optimal paths. action = which of the 6 moves."""
    def __init__(self, D, H, W, obstacle_density=0.15, num_samples=2000):
        self.data = []                                   # (state, action) pairs
        maps_tried = 0
        print(f"[3D] Generating dataset — target: {num_samples} samples...")

        while len(self.data) < num_samples:              # until enough samples
            grid, path, _, target = generate_random_grid_3d(D, H, W, obstacle_density)
            maps_tried += 1
            if path is None:                             # unsolvable -> skip
                continue

            for i in range(len(path) - 1):               # each step on the path = one example
                curr, nxt = path[i], path[i + 1]
                move = (nxt[0]-curr[0], nxt[1]-curr[1], nxt[2]-curr[2])  # the BFS move vector

                action = None                            # which of the 6 actions is `move`?
                for a, vec in ACTIONS_6.items():
                    if vec == move:
                        action = a
                        break
                if action is None:                       # shouldn't happen (BFS is 6-connected)
                    continue

                state = get_input_state_cnn_3d(grid, curr, target).squeeze(0)  # input tensor
                self.data.append((state, torch.tensor(action, dtype=torch.long)))  # + label

            if maps_tried % 100 == 0:
                print(f"  {len(self.data)} samples from {maps_tried} maps...")

        if not self.data:                                # safety
            raise ValueError("No 3D training data generated — check parameters.")
        print(f"[3D] Dataset ready: {len(self.data)} samples from {maps_tried} maps.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# =============================================================================
# TRAINING — generator, yields progress for the Streamlit page
# =============================================================================

def train_supervised_3d_live(D, H, W, obstacle_density=0.15, epochs=15,
                             num_samples=2000, batch_size=16, lr=5e-4,
                             weight_decay=5e-3, val_split=0.2, progress_every=1):
    device    = torch.device("cpu")                      # CPU only
    model     = PathPredictionResNet3D().to(device)      # fresh model
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # Adam + L2
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)  # LR drop
    criterion = nn.CrossEntropyLoss()                    # 6-way classification

    dataset = PathfindingDataset3D(D, H, W, obstacle_density, num_samples)  # BFS samples
    val_size   = int(len(dataset) * val_split)           # 20% validation
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])  # shuffled split
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    train_losses, val_losses, best_val = [], [], float('inf')  # tracking

    for epoch in range(epochs):
        model.train()                                    # enable dropout
        epoch_loss = 0
        for states, actions in train_loader:             # minibatches
            states, actions = states.to(device), actions.to(device)
            loss = criterion(model(states), actions)     # forward + loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()  # update
            epoch_loss += loss.item()
        avg_train = epoch_loss / len(train_loader)       # mean train loss
        train_losses.append(avg_train)

        model.eval()                                     # disable dropout
        val_loss = correct = total = 0
        with torch.no_grad():                            # no grads for val
            for states, actions in val_loader:
                states, actions = states.to(device), actions.to(device)
                logits = model(states)
                val_loss += criterion(logits, actions).item()
                correct += (torch.argmax(logits, 1) == actions).sum().item()  # matches
                total   += len(actions)
        avg_val  = val_loss / len(val_loader)            # mean val loss
        accuracy = correct / total * 100                 # action-match %
        val_losses.append(avg_val)
        scheduler.step(avg_val)                          # maybe lower LR
        best_val = min(best_val, avg_val)                # track best

        if (epoch + 1) % progress_every == 0:            # yield to UI
            yield {"epoch": epoch+1, "total_epochs": epochs, "train_loss": avg_train,
                   "val_loss": avg_val, "accuracy": accuracy, "best_val": best_val,
                   "train_losses": train_losses.copy(), "val_losses": val_losses.copy(),
                   "model": model, "done_training": False}

    yield {"epoch": epochs, "total_epochs": epochs, "train_loss": avg_train,  # final payload
           "val_loss": avg_val, "accuracy": accuracy, "best_val": best_val,
           "train_losses": train_losses, "val_losses": val_losses,
           "model": model, "done_training": True}


# =============================================================================
# INFERENCE
# =============================================================================

def run_supervised_inference_3d(model, numeric_grid, robot_start, target_pos, D, H, W):
    """Greedy rollout of the 3D ResNet. Returns (path, success)."""
    device = next(model.parameters()).device
    model.eval()
    MAX_STEPS = ((D-1)+(H-1)+(W-1)) * 3 + 5              # step budget
    robot_pos = robot_start                              # current voxel
    path   = [robot_pos]                                 # trajectory
    recent = deque(maxlen=10)                            # oscillation window

    for _ in range(MAX_STEPS):
        if robot_pos == target_pos:                      # arrived
            return path, True
        state = get_input_state_cnn_3d(numeric_grid, robot_pos, target_pos).to(device)  # input
        with torch.no_grad():
            action = torch.argmax(model(state), dim=1).item()  # best of 6 moves
        dz, dy, dx = ACTIONS_6[action]                   # move vector
        nz, ny, nx = robot_pos[0]+dz, robot_pos[1]+dy, robot_pos[2]+dx  # candidate voxel

        if (0 <= nz < D and 0 <= ny < H and 0 <= nx < W  # in bounds...
                and numeric_grid[nz, ny, nx] != CELL_OBSTACLE):  # ...and not a wall
            robot_pos = (nz, ny, nx)                     # accept the move
        path.append(robot_pos)

        recent.append(robot_pos)
        if len(recent) == 10 and len(set(recent)) <= 2:  # stuck oscillating
            break
    return path, (robot_pos == target_pos)               # path + did it reach target


def load_supervised_model_3d(path, device="cpu"):
    model = PathPredictionResNet3D().to(device)          # size-agnostic -> no rows/cols needed
    model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    model.eval()
    return model
