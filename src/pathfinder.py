import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from torch.utils.data import DataLoader, random_split # Import for data splitting
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

'''
initial_grid_data = [
    [0, 0, 0, 0 , 0, 0, "T"],
    [0, 0, 0, 0,  0,  3,  0],
    [0, 0, 0, 1,  0,  3,  0],
    [0, 0, 0, "R",  0,  0,  0],
    [0, 0, 0, 0,  0,  0,  0],
    [0, 0, 0, 0,  0,  0,  0],
]
'''

CELL_TYPES = {
    0: 0.0,  # Traversable (empty)
    1: 0.0,  # Traversable (already visited, treat as empty for pathfinding)
    3: 1.0,  # Obstacle
    "R": 2.0, # Robot Start
    "T": 3.0  # Target
}

# Define possible actions (moves)
ACTIONS = {
    "DOWN": (1, 0),
    "BOTTOM_RIGHT": (1, 1),  
    "BOTTOM_LEFT": (1, -1),
    "TOP_RIGHT": (-1, 1),
    "TOP_LEFT": (-1, -1),
    "RIGHT": (0, 1),
    "LEFT": (0, -1),
    "UP": (-1, 0),
}

ACTION_NAMES = list(ACTIONS.keys())
ACTION_VECTORS = list(ACTIONS.values())

def get_grid_numeric(grid_data):
    """Converts the symbolic grid data to a numeric NumPy array."""
    numeric_grid = np.zeros((len(grid_data), len(grid_data[0])), dtype=np.float32)
    for r in range(len(grid_data)):
        for c in range(len(grid_data[0])):
            cell_val = grid_data[r][c]
            numeric_grid[r, c] = CELL_TYPES.get(cell_val, 0.0)
    return numeric_grid

def find_elements(grid_numeric, robot_val=CELL_TYPES["R"], target_val=CELL_TYPES["T"]):
    """Finds the coordinates of the robot and target."""
    robot_pos = None
    target_pos = None
    rows, cols = grid_numeric.shape
    for r in range(rows):
        for c in range(cols):
            if grid_numeric[r, c] == robot_val:
                robot_pos = (r, c)
            elif grid_numeric[r, c] == target_val:
                target_pos = (r, c)
    return robot_pos, target_pos

def is_valid_move(grid_numeric, r, c, obstacle_val=CELL_TYPES[3]):
    """Checks if a move to (r, c) is valid (within bounds and not an obstacle)."""
    rows, cols = grid_numeric.shape
    if not (0 <= r < rows and 0 <= c < cols):
        return False # Out of bounds
    if grid_numeric[r, c] == obstacle_val:
        return False # Is an obstacle
    return True

def find_shortest_path_bfs(grid_numeric, start_pos, target_pos):
    """
    Finds the shortest path from start_pos to target_pos using BFS.
    Returns a list of (row, col) tuples representing the path.
    """
    rows, cols = grid_numeric.shape
    
    queue = deque([(start_pos, [start_pos])]) # (current_pos, current_path)
    visited = {start_pos}

    while queue:
        (r, c), path = queue.popleft()

        if (r, c) == target_pos:
            return path # Found the shortest path

        for dr, dc in ACTION_VECTORS:
            nr, nc = r + dr, c + dc
            if is_valid_move(grid_numeric, nr, nc) and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))
    
    return None # No path found

def get_input_state_cnn(grid_numeric, robot_current_pos, target_pos):
    """
    Constructs the input tensor for the CNN from the current grid state.
    Returns a tensor with shape: [1, num_channels, height, width]
    """
    rows, cols = grid_numeric.shape
    
    # Initialize channels
    map_channel = np.zeros_like(grid_numeric)
    robot_channel = np.zeros_like(grid_numeric)
    target_channel = np.zeros_like(grid_numeric)

    # Populate map channel (obstacles)
    map_channel[grid_numeric == CELL_TYPES[3]] = 1.0 # Obstacles are 1.0

    # Populate robot channel
    robot_channel[robot_current_pos[0], robot_current_pos[1]] = 1.0

    # Populate target channel
    target_channel[target_pos[0], target_pos[1]] = 1.0

    # Stack channels to form the input (num_channels, height, width)
    input_channels = np.stack([map_channel, robot_channel, target_channel], axis=0)

    # Add batch dimension: (1, num_channels, height, width)
    return torch.tensor(input_channels, dtype=torch.float32).unsqueeze(0)



# --- 2. Random Grid Generator for Dataset ---
def generate_random_grid(rows, cols, obstacle_density=0.2):
    """
    Generates a random grid with obstacles, robot, and target.
    Ensures a path exists from robot to target.
    """
    max_attempts = 100 # Prevent infinite loops for unsolvable grids
    
    for _ in range(max_attempts):
        grid_data = [[0 for _ in range(cols)] for _ in range(rows)]
        
        # Place obstacles
        for r in range(rows):
            for c in range(cols):
                if random.random() < obstacle_density:
                    grid_data[r][c] = 3 # Mark as obstacle

        # Place Robot and Target randomly
        all_coords = [(r, c) for r in range(rows) for c in range(cols)]
        random.shuffle(all_coords)

        robot_pos = None
        target_pos = None

        for r_c in all_coords:
            if grid_data[r_c[0]][r_c[1]] == 0: # If empty
                robot_pos = r_c
                grid_data[r_c[0]][r_c[1]] = "R"
                break
        
        for t_c in all_coords:
            if grid_data[t_c[0]][t_c[1]] == 0 and t_c != robot_pos: # If empty and not robot pos
                target_pos = t_c
                grid_data[t_c[0]][t_c[1]] = "T"
                break
        
        if robot_pos is None or target_pos is None:
            continue # Couldn't place both, try again

        numeric_grid = get_grid_numeric(grid_data)
        
        # Ensure a path exists between robot and target
        path = find_shortest_path_bfs(numeric_grid, robot_pos, target_pos)
        if path:
            return grid_data, numeric_grid, path, robot_pos, target_pos
            
    # print("Warning: Could not generate a solvable grid after many attempts.")
    return None, None, None, None, None

# --- 3. PyTorch Dataset Class ---
class PathfindingDataset(torch.utils.data.Dataset):
    def __init__(self, num_maps=1000, map_rows=6, map_cols=7, obstacle_density=0.2):
        self.data = [] # List of (input_state_tensor, optimal_action_label) pairs
        print(f"Generating {num_maps} random maps for dataset...")
        
        for i in range(num_maps):
            grid_data, numeric_grid, optimal_path, robot_start_pos, target_pos = \
                generate_random_grid(map_rows, map_cols, obstacle_density)
            
            if optimal_path is None:
                continue # Skip if no path found
            
            # Iterate through the optimal path to generate (state, action) pairs
            for j in range(len(optimal_path) - 1):
                current_robot_pos = optimal_path[j]
                next_optimal_pos = optimal_path[j+1]

                dr = next_optimal_pos[0] - current_robot_pos[0]
                dc = next_optimal_pos[1] - current_robot_pos[1]

                optimal_action_label = -1
                for idx, (adir, adic) in enumerate(ACTION_VECTORS):
                    if dr == adir and dc == adic:
                        optimal_action_label = idx
                        break
                
                if optimal_action_label != -1:
                    # Note: get_input_state returns unsqueezed tensor, but Dataset expects 1D.
                    # We'll unsqueeze in DataLoader's collate_fn or directly in forward pass for batch.
                    # For simplicity, here, we get the 1D tensor:
                    input_state = get_input_state_cnn(numeric_grid, current_robot_pos, target_pos).squeeze(0)
                    self.data.append((input_state, torch.tensor(optimal_action_label, dtype=torch.long)))
            
            if (i + 1) % 500 == 0:
                print(f"Generated {len(self.data)} examples from {i+1} maps so far.")

        print(f"Finished data generation. Total examples: {len(self.data)}")
        if len(self.data) == 0:
             raise ValueError("No training data generated. Check map generation parameters.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# No separate ResidualBlock class needed for this unified approach

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        # Unify the main path's two convolutional layers and their BN/ReLU
        # The final ReLU of the block is applied *after* the skip connection
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # ReLU after the first conv-bn pair

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels) # No ReLU here, as the final ReLU for the block comes after addition
        )
        
        self.final_relu = nn.ReLU(inplace=True) # This ReLU applies after the skip connection (out + identity)
        self.downsample = downsample # Downsample path for the skip connection, if needed

    def forward(self, x):
        identity = x # Save the input for the skip connection
        out = self.main_path(x) # Process through the main path (two conv-bn-relu layers)

        if self.downsample is not None: # Apply downsample to the identity if it exists
            identity = self.downsample(x)

        out += identity # Add the skip connection
        out = self.final_relu(out) # Apply the final ReLU for the block
        return out
    
class PathPredictionResNet(nn.Module):
    def __init__(self, in_channels, height, width, output_dim):
        super(PathPredictionResNet, self).__init__()

        # --- Initial Convolutional Layer ---
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # --- Residual Block 1 ---
        # Channels: 32 -> 32, no downsample needed for identity
        self.res_block1 = ResidualBlock(32, 32) 

        # --- Residual Block 2 ---
        # Channels: 32 -> 64. Need a downsample for the skip connection to match channels.
        self.res_block2 = ResidualBlock(
            32, 128, 
            downsample=nn.Sequential(
                nn.Conv2d(32, 128, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(128),
            )
        )

        # --- Flatten and Fully Connected Layers ---
        # Calculate flattened dimension AFTER all conv/pooling operations.
        # Assuming no pooling, output feature map size remains height x width
        final_conv_channels = 128 # From the last residual block
        self._flattened_features_dim = final_conv_channels * height * width 
        
        self.fc = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(self._flattened_features_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, output_dim) 
        )

    def forward(self, x):
        # Initial Convolution
        x = self.initial_conv(x)
        
        # Residual Blocks
        x = self.res_block1(x) # Call the first residual block
        x = self.res_block2(x) # Call the second residual block
        
        # Flatten and Fully Connected Layers
        x = x.view(-1, self._flattened_features_dim)
        x = self.fc(x) 
        return x

# --- 5. Training Function ---
def train_model(model, dataset, epochs=50, batch_size=32, learning_rate=0.001, validation_split=0.2, patience=7, weight_decay=0, model_save_path="path_prediction_mlp.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Split dataset into training and validation sets
    train_size = int((1 - validation_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # No shuffle for validation
        
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience-2)
    loss_criterion = nn.CrossEntropyLoss()

    print(f"\n--- Starting Training on {device} ---")
    print(f"Training examples: {len(train_dataset)}, Validation examples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")

    best_val_loss = np.inf
    counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        correct_train_predictions = 0
        total_train_examples = 0

        train_loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Train", leave=False)

        for batch_input, batch_label in train_loop:
            batch_input, batch_label = batch_input.to(device), batch_label.to(device)

            optimizer.zero_grad()
            predictions = model(batch_input)
            
            loss = loss_criterion(predictions, batch_label)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(predictions.data, 1)
            total_train_examples += batch_label.size(0)
            correct_train_predictions += (predicted == batch_label).sum().item()

        train_loss = total_train_loss / len(train_dataloader)
        train_acc = correct_train_predictions / total_train_examples * 100

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        total_val_loss = 0
        correct_val_predictions = 0
        total_val_examples = 0

        val_loop = tqdm(val_dataloader, desc=f"Epoch {epoch+1} Val", leave=False)

        with torch.no_grad(): # No gradient calculations in validation
            for batch_input, batch_label in val_loop:
                batch_input, batch_label = batch_input.to(device), batch_label.to(device)
                
                predictions = model(batch_input)
                loss = loss_criterion(predictions, batch_label)

                total_val_loss += loss.item()
                _, predicted = torch.max(predictions.data, 1)
                total_val_examples += batch_label.size(0)
                correct_val_predictions += (predicted == batch_label).sum().item()

        val_loss = total_val_loss / len(val_dataloader)
        val_acc = correct_val_predictions / total_val_examples * 100

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        scheduler.step(val_loss) # Tell the scheduler the current validation loss
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.6f}") # Print with more precision

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save the model if validation accuracy improved
        if best_val_loss > val_loss:
            counter = 0
            print(f"Validation loss went down from {best_val_loss:.2f} to {val_loss:.2f}. Saving model.")
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
        else:
            counter += 1
        
        if counter >= patience:
            print(f"Early stopping at epoch {epoch} after {patience} epochs without improvement.")
            break

    print("\nTraining complete!")
    print(f"Best model (based on validation accuracy) saved to {model_save_path}")
    return model, history

# --- 6. Inference and Visualization (Same as before) ---
def simulate_robot_movement(model, initial_grid_data_list, max_steps=50, model_name=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    numeric_grid = get_grid_numeric(initial_grid_data_list)
    # rows, cols = numeric_grid.shape
    
    current_robot_pos, target_pos = find_elements(numeric_grid)
    
    if not current_robot_pos or not target_pos:
        print("Error: Robot or Target not found in the map.")
        return None, None

    print(f"\n--- Simulating movement with {model_name} model ---")
    print(f"Starting simulation from {current_robot_pos} to {target_pos}")

    path_taken = [current_robot_pos]
    frames = [] # To store grid states for animation

    # Initial frame
    current_map_display = np.copy(numeric_grid)
    current_map_display[current_robot_pos[0], current_robot_pos[1]] = CELL_TYPES["R"]
    frames.append(current_map_display)

    model.eval() # Set model to evaluation mode
    stuck_counter = 0

    for step in range(max_steps):
        if current_robot_pos == target_pos:
            print(f"Target reached in {step} steps!")
            break

        input_state = get_input_state_cnn(numeric_grid, current_robot_pos, target_pos).to(device)
        
        with torch.no_grad():
            action_logits = model(input_state)
            predicted_action_idx = torch.argmax(action_logits, dim=1).item()
        
        dr, dc = ACTION_VECTORS[predicted_action_idx]
        next_r, next_c = current_robot_pos[0] + dr, current_robot_pos[1] + dc

        if is_valid_move(numeric_grid, next_r, next_c):
            current_robot_pos = (next_r, next_c)
            path_taken.append(current_robot_pos)
            stuck_counter = 0
        else:
            print(f"Step {step}: Model predicted an INVALID move to ({next_r}, {next_c}). Staying at {current_robot_pos}")
            stuck_counter += 1
            if stuck_counter > 5:
                print("Robot is stuck or looping. Aborting simulation.")
                break
            
        current_map_display = np.copy(numeric_grid)
        for p_r, p_c in path_taken[:-1]:
            current_map_display[p_r, p_c] = 0.5
        current_map_display[current_robot_pos[0], current_robot_pos[1]] = CELL_TYPES["R"]
        current_map_display[target_pos[0], target_pos[1]] = CELL_TYPES["T"]
        frames.append(current_map_display)

    if current_robot_pos != target_pos:
        print("Max steps reached or robot got stuck. Target not reached.")
    
    return path_taken, frames

def animate_path(frames, title_suffix="", gif_index=0):
    """Creates a matplotlib animation of the robot's path."""
    if not frames:
        print("No frames to animate.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Define colormap and normalization for visualization
    colors = ['#FFFFFF', '#DDDDDD', '#000000', '#FF0000', '#00FF00'] # White, Light Grey, Black, Red, Green
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    bounds = [0.0, 0.25, 0.75, 1.5, 2.5, 3.5]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    # --- MODIFIED 'update' FUNCTION ---
    # The 'i' argument will be the frame index (0, 1, 2, ...)
    def update(i): 
        frame_data = frames[i] # Get the actual frame data using the index
        ax.clear()
        ax.imshow(frame_data, cmap=cmap, norm=norm, origin='upper', extent=[0, frame_data.shape[1], frame_data.shape[0], 0])
        ax.set_xticks(np.arange(frame_data.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(frame_data.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        ax.set_xticks([])
        ax.set_yticks([])
        # Use the index 'i' directly for the step number in the title
        ax.set_title(f"Robot Path - Step {i} {title_suffix}") 

    # --- MODIFIED 'FuncAnimation' CALL ---
    # Pass 'range(len(frames))' as the frames argument so 'update' receives indices
    ani = animation.FuncAnimation(fig, update, frames=range(len(frames)), repeat=False, interval=500)
    animation_path = os.path.join(os.path.dirname(__file__), '..', 'examples', f'robot_animation_{gif_index}.gif')
    ani.save(animation_path, writer='pillow', fps=2)
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration for Training ---
    NUM_TRAINING_MAPS = 5000 # Increased for better training
    MAP_ROWS, MAP_COLS = 15, 15 # Increased map size for more complex pathfinding
    OBSTACLE_DENSITY = 0.25
    NUM_EPOCHS = 50 # Can increase if needed
    BATCH_SIZE = 16 # 2^N where N > 1
    LEARNING_RATE = 5e-4
    VALIDATION_SPLIT = 0.2 # 20% of data for validation
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'models'), exist_ok=True) # Ensure models directory exists
    MODEL_SAVE_FILE = os.path.join(os.path.dirname(__file__), '..', 'models', 'path_prediction_mlp.pth')
    PATIENCE = 7 # Early stopping patience
    WEIGHT_DECAY = 5e-3
    # --- 1. Generate Dataset ---
    dataset = PathfindingDataset(NUM_TRAINING_MAPS, MAP_ROWS, MAP_COLS, OBSTACLE_DENSITY)
    # --- 2. Initialize Model ---
    # input_dim = MAP_ROWS * MAP_COLS + 4 
    output_dim = len(ACTIONS)
    IN_CHANNELS = 3 
    model = PathPredictionResNet(IN_CHANNELS, MAP_ROWS, MAP_COLS, output_dim)

    # --- 3. Train Model ---
    trained_model, history = train_model(
        model, 
        dataset, 
        NUM_EPOCHS, 
        BATCH_SIZE, 
        LEARNING_RATE, 
        VALIDATION_SPLIT, 
        PATIENCE, 
        WEIGHT_DECAY,
        MODEL_SAVE_FILE,
    )

    # --- 4. Test Trained Model on Original Example ---
    # Load the best model weights found during training
    model_loaded = PathPredictionResNet(IN_CHANNELS, MAP_ROWS, MAP_COLS, output_dim)
    if os.path.exists(MODEL_SAVE_FILE):
        model_loaded.load_state_dict(torch.load(MODEL_SAVE_FILE))
        print(f"\nLoaded trained model from {MODEL_SAVE_FILE}")
    else:
        print(f"\nWarning: Trained model file not found at {MODEL_SAVE_FILE}. Using the last state of trained_model.")
        model_loaded.load_state_dict(trained_model.state_dict())

    print("\n--- Testing Trained Model on a NEW Random Grid ---")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label="Training Loss")
    plt.plot(history['val_loss'], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Losses Over Epochs")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label="Training Accuracy")
    plt.plot(history['val_acc'], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracies Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'examples'), exist_ok=True) # Ensure examples directory exists
    training_path = os.path.join(os.path.dirname(__file__), '..', 'examples', 'training_history.png')
    plt.savefig(training_path)
    plt.show()

    # Generate a new solvable map for testing
    for i in range(10): # Repeat 10 times so I want to see how error is processing
        temp_grid_list, _, _, _, _ = generate_random_grid(MAP_ROWS, MAP_COLS, OBSTACLE_DENSITY)
        
        if temp_grid_list is None:
            print("Could not generate a solvable new random test grid. Skipping this test.")
        else:
            path_taken_new, animation_frames_new = simulate_robot_movement(model_loaded, temp_grid_list, 100, model_name="TRAINED")
            if path_taken_new:
                print("Path taken by TRAINED model on NEW random grid:")
                print(path_taken_new)
                animate_path(animation_frames_new, title_suffix="(New Random Grid)", gif_index=i)
            else:
                print("Trained model failed to find a path on new random grid or simulation aborted.")