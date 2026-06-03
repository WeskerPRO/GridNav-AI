import torch                                            # tensors + autograd
import torch.nn as nn                                    # layers + loss
import torch.optim as optim                              # Adam + LR scheduler
import numpy as np                                       # build the input channels
from collections import deque                            # oscillation window at inference
from torch.utils.data import Dataset, DataLoader, random_split  # data pipeline + train/val split

# =============================================================================
# CONSTANTS
# =============================================================================

ACTIONS_8 = {                                            # 8-connected moves (incl. diagonals)
    0: (-1,  0),   # UP
    1: ( 1,  0),   # DOWN
    2: ( 0, -1),   # LEFT
    3: ( 0,  1),   # RIGHT
    4: (-1, -1),   # UP-LEFT
    5: (-1,  1),   # UP-RIGHT
    6: ( 1, -1),   # DOWN-LEFT
    7: ( 1,  1),   # DOWN-RIGHT
}
NUM_ACTIONS_8 = 8                                        # supervised model picks 1 of 8
CELL_OBSTACLE = 1.0                                      # obstacle encoding

# Adaptive spatial pooling target. Makes the ResNet size-agnostic: any rows×cols
# grid is pooled to POOL_SIZE×POOL_SIZE before the FC head, so a single trained
# model runs on any grid size. Kept >1 (not global pooling) so the robot/target
# POSITIONS survive — global 1×1 pooling would average position away and break
# pathfinding. Larger POOL_SIZE = more spatial fidelity, bigger FC layer.
POOL_SIZE = 8


# =============================================================================
# MODEL — your exact ResNet architecture from path_finder.py
# =============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.main_path = nn.Sequential(                  # two conv layers (the residual function)
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),                # normalise feature maps
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.final_relu = nn.ReLU(inplace=True)          # applied after the skip add
        self.downsample = downsample                     # reshapes the skip path if channels change

    def forward(self, x):
        identity = x                                     # the "skip" / shortcut
        out      = self.main_path(x)                     # the learned residual
        if self.downsample is not None:
            identity = self.downsample(x)                # match shape before adding
        out += identity                                  # residual connection
        return self.final_relu(out)                      # activate the sum


class PathPredictionResNet(nn.Module):
    """
    ResNet CNN — exact architecture from path_finder.py.
    Input:  3-channel grid tensor (obstacles, robot, target)
    Output: 8 action logits
    Uses Dropout(0.3) to prevent overfitting.
    """
    def __init__(self, rows, cols, num_actions=NUM_ACTIONS_8):
        super(PathPredictionResNet, self).__init__()
        self.rows = rows                                 # kept for reference (not used for sizing now)
        self.cols = cols

        self.initial_conv = nn.Sequential(               # 3 channels -> 32 feature maps
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # res_block1: 32 → 32, no channel change, no downsample needed
        self.res_block1 = ResidualBlock(32, 32)

        # res_block2: 32 → 128, channel change needs 1×1 downsample
        self.res_block2 = ResidualBlock(
            32, 128,
            downsample=nn.Sequential(                    # 1×1 conv to lift skip 32 -> 128
                nn.Conv2d(32, 128, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(128),
            )
        )

        # size-agnostic head: pool any rows×cols feature map to POOL_SIZE²
        self.spatial_pool = nn.AdaptiveAvgPool2d((POOL_SIZE, POOL_SIZE))  # any grid -> fixed grid
        self._flattened_dim = 128 * POOL_SIZE * POOL_SIZE                 # fixed FC input size

        self.fc = nn.Sequential(                         # classifier head -> 8 logits
            nn.Dropout(0.3),                             # regularisation
            nn.Linear(self._flattened_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_actions)                  # one logit per action
        )

    def forward(self, x):
        x = self.initial_conv(x)        # [B, 32, rows, cols]
        x = self.res_block1(x)          # [B, 32, rows, cols]
        x = self.res_block2(x)          # [B, 128, rows, cols]
        x = self.spatial_pool(x)        # [B, 128, POOL_SIZE, POOL_SIZE]
        x = x.view(x.size(0), -1)       # [B, 128 * POOL_SIZE²] — fixed regardless of grid size
        return self.fc(x)               # action logits


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
    rows, cols = numeric_grid.shape                      # grid size

    obstacle_ch = (numeric_grid == CELL_OBSTACLE).astype(np.float32)  # channel 0: walls

    robot_ch = np.zeros((rows, cols), dtype=np.float32)  # channel 1: robot one-hot
    if robot_pos:
        robot_ch[robot_pos[0], robot_pos[1]] = 1.0       # mark robot cell

    target_ch = np.zeros((rows, cols), dtype=np.float32) # channel 2: target one-hot
    if target_pos:
        target_ch[target_pos[0], target_pos[1]] = 1.0    # mark target cell

    return torch.tensor(
        np.stack([obstacle_ch, robot_ch, target_ch]),    # stack the 3 channels
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
        from core.grid_utils import generate_random_grid  # local import avoids a cycle

        self.data = []                                   # (state, action) pairs
        maps_tried = 0                                   # how many grids generated

        print(f"Generating dataset — target: {num_samples} samples...")

        while len(self.data) < num_samples:              # until we have enough samples
            result = generate_random_grid(rows, cols, obstacle_density)
            _, numeric, path, _, target = result         # BFS path is the label source
            maps_tried += 1

            if path is None:                             # unsolvable grid -> skip
                continue

            for i in range(len(path) - 1):               # each consecutive pair = one example
                curr  = path[i]                          # current cell
                next_ = path[i + 1]                      # next cell on optimal path
                dr    = next_[0] - curr[0]               # row delta
                dc    = next_[1] - curr[1]               # col delta

                action = None                            # which of the 8 moves is (dr,dc)?
                for a, (adr, adc) in ACTIONS_8.items():
                    if adr == dr and adc == dc:
                        action = a
                        break

                if action is None:                       # delta not in action set -> skip
                    continue

                state = get_input_state_cnn(numeric, curr, target).squeeze(0)  # input tensor
                self.data.append((
                    state,
                    torch.tensor(action, dtype=torch.long)  # label = the BFS move
                ))

            if maps_tried % 200 == 0:                    # occasional progress print
                print(f"  {len(self.data)} samples from {maps_tried} maps...")

        if len(self.data) == 0:                          # safety check
            raise ValueError("No training data generated. Check parameters.")

        print(f"Dataset ready: {len(self.data)} samples from {maps_tried} maps.")

    def __len__(self):
        return len(self.data)                            # dataset size

    def __getitem__(self, idx):
        return self.data[idx]                            # one (state, action) pair


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
        lr=5e-4, weight_decay=5e-3, Dropout(0.3), batch_size=16
        random_split for unbiased train/val split
        ReduceLROnPlateau scheduler
    """
    device = torch.device("cpu")                         # demo trains on CPU
    model = PathPredictionResNet(rows, cols).to(device)  # fresh model
    optimizer = optim.Adam(model.parameters(),           # Adam + L2 (weight decay)
                           lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(    # drop LR when val loss stalls
        optimizer, mode='min', factor=0.1, patience=3
    )
    criterion = nn.CrossEntropyLoss()                    # 8-way classification loss

    # generate dataset
    dataset = PathfindingDataset(rows, cols, obstacle_density, num_samples)  # BFS samples

    # random_split — unbiased, shuffled split
    val_size   = int(len(dataset) * val_split)           # 20% for validation
    train_size = len(dataset) - val_size                 # rest for training
    train_set, val_set = random_split(dataset, [train_size, val_size])  # shuffled split

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)   # shuffle train
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)      # fixed val order

    train_losses = []                                    # per-epoch train loss
    val_losses = []                                      # per-epoch val loss
    best_val = float('inf')                              # best val loss seen

    for epoch in range(epochs):

        # ── training ──────────────────────────────────────────────────────
        model.train()                                    # enable dropout
        epoch_loss = 0
        for states, actions in train_loader:             # minibatches
            states = states.to(device)
            actions = actions.to(device)
            logits = model(states)                       # forward
            loss = criterion(logits, actions)            # how wrong
            optimizer.zero_grad()                        # clear grads
            loss.backward()                              # backprop
            optimizer.step()                             # update weights
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(train_loader)       # mean train loss
        train_losses.append(avg_train)

        # ── validation ────────────────────────────────────────────────────
        model.eval()                                     # disable dropout
        val_loss = 0
        correct  = 0
        total    = 0
        with torch.no_grad():                            # no gradients for val
            for states, actions in val_loader:
                states  = states.to(device)
                actions = actions.to(device)
                logits  = model(states)
                val_loss += criterion(logits, actions).item()  # accumulate loss
                preds    = torch.argmax(logits, dim=1)   # predicted action
                correct  += (preds == actions).sum().item()    # count matches
                total    += len(actions)

        avg_val  = val_loss / len(val_loader)            # mean val loss
        accuracy = correct / total * 100                 # action-match accuracy %
        val_losses.append(avg_val)

        scheduler.step(avg_val)                          # maybe lower the LR

        if avg_val < best_val:                           # track best
            best_val = avg_val

        if (epoch + 1) % progress_every == 0:            # yield to the UI
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

    yield {                                              # final payload
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
    device    = next(model.parameters()).device          # model device
    model.eval()                                         # inference mode

    path_length = (rows - 1) + (cols - 1)                # min path length
    MAX_STEPS   = path_length * 3 + 5                    # step budget

    robot_pos = robot_start                              # current position
    path      = [robot_pos]                              # trajectory
    recent    = deque(maxlen=10)                         # oscillation window

    for _ in range(MAX_STEPS):
        if robot_pos == target_pos:                      # already there?
            return path, True

        state_t = get_input_state_cnn(                   # build 3-channel input
            numeric_grid, robot_pos, target_pos
        ).to(device)

        with torch.no_grad():
            logits = model(state_t)                      # predict
            action = torch.argmax(logits, dim=1).item()  # best action

        dr, dc = ACTIONS_8[action]                       # move vector
        r, c   = robot_pos
        nr, nc = r + dr, c + dc                          # candidate cell

        # only move if valid — stay in place if model picks wall/OOB
        if (0 <= nr < rows and 0 <= nc < cols
                and numeric_grid[nr, nc] != CELL_OBSTACLE):
            robot_pos = (nr, nc)                         # accept the move

        path.append(robot_pos)

        recent.append(robot_pos)
        if len(recent) == 10 and len(set(recent)) <= 2:
            break   # oscillation → stop

    success = (robot_pos == target_pos)                  # reached the goal?
    return path, success


def load_supervised_model(path, rows, cols, device='cpu'):
    """Load saved supervised model from .pth file."""
    model = PathPredictionResNet(rows, cols, NUM_ACTIONS_8).to(device)  # build net
    model.load_state_dict(torch.load(path, weights_only=True, map_location=device))  # load weights
    model.eval()                                         # inference mode
    return model
