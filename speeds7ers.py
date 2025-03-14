#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
Speeds7ers – Deep Q-Network Version
================================================================================

This is a Pygame-based game that pits a player against an AI whose behavior is 
learned using a deep Q-network (DQN). All the gameplay features (power-ups, 
trails, dynamic menu, etc.) have been preserved. The reinforcement learning 
model now uses a neural network with experience replay and a target network.

--------------------------------------------------------------------------------
Author: OpenAI ChatGPT
Date: 2025-03-13
--------------------------------------------------------------------------------
"""

# =============================================================================
#                              IMPORTS & SETUP
# =============================================================================

import pygame
import random
import numpy as np
import os
import atexit
from collections import deque

# Import PyTorch for the deep Q-learning model
import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
#                             PYGAME INITIALIZATION
# =============================================================================

pygame.init()
screen = pygame.display.set_mode((1280, 720))

# =============================================================================
#                             GAME CONSTANTS
# =============================================================================

WIDTH, HEIGHT = 1280, 720
GRID_SIZE = 10
COLS, ROWS = WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE
FPS = 45

# Power-Up size: reduced from 4 times the grid size (40x40) to 3 times the grid size (30x30)
POWERUP_SIZE = GRID_SIZE * 3

# Define the scale factor for power-up sprites (set to 7) and update the collision size accordingly.
SCALE_FACTOR = 7
POWERUP_COLLISION_SIZE = POWERUP_SIZE * SCALE_FACTOR

# NEW: Define a pickup scale to reduce the collision area (without reducing the sprite size)
POWERUP_PICKUP_SCALE = 0.5  # For example, 50% of the drawn sprite area
POWERUP_PICKUP_SIZE = int(POWERUP_COLLISION_SIZE * POWERUP_PICKUP_SCALE)

# Colors dictionary for use in drawing game elements
COLORS = {
    "Red": (255, 0, 0), "Blue": (0, 0, 255), "Green": (0, 255, 0),
    "Yellow": (255, 255, 0), "Purple": (128, 0, 128), "Cyan": (0, 255, 255),
    "White": (255, 255, 255), "Grey": (100, 100, 100)
}

# Reinforcement Learning and DQN Parameters
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
ALPHA = 0.001  # Learning rate for the optimizer (used in the DQN)
GAMMA = 0.98  # Discount factor for future rewards
EPSILON_START = 0.7  # Starting exploration rate
EPSILON_END = 0.01  # Minimum exploration rate
EPSILON_DECAY = 0.9995  # Epsilon decay factor per training update

BATCH_SIZE = 64  # Experience replay batch size

# Global difficulty settings (persist across rounds until app restart)
difficulty_levels = ["Easy", "Medium", "Hard", "Expert", "Nightmare"]
current_difficulty = 1  # Default to Medium difficulty

# =============================================================================
#                     LOAD POWER-UP SPRITES
# =============================================================================

def load_sprite(filename, scale_factor=1):
    sprite = pygame.image.load(filename).convert_alpha()
    new_size = (int(POWERUP_SIZE * scale_factor), int(POWERUP_SIZE * scale_factor))
    sprite = pygame.transform.scale(sprite, new_size)
    return sprite

powerup_sprites = {
    "double_speed": load_sprite("sprites/speed.png", scale_factor=SCALE_FACTOR),
    "invisibility": load_sprite("sprites/invis.png", scale_factor=SCALE_FACTOR),
    "wall_phase": load_sprite("sprites/wallphase.png", scale_factor=SCALE_FACTOR),
    "extra_life": load_sprite("sprites/life.png", scale_factor=SCALE_FACTOR),
    "ai_confusion": load_sprite("sprites/ai.png", scale_factor=SCALE_FACTOR),
    "teleport": load_sprite("sprites/tp.png", scale_factor=SCALE_FACTOR),
    "slow_time": load_sprite("sprites/time.png", scale_factor=SCALE_FACTOR)
}

# =============================================================================
#                         EXPERIENCE REPLAY BUFFER CLASS
# =============================================================================

class ExperienceReplay:
    """
    Experience Replay buffer that stores past transitions.
    Each transition is a tuple: (state, action, reward, next_state, done)
    """
    def __init__(self, capacity=20000):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Return a random sample of transitions from the buffer."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)

# =============================================================================
#                         INITIALIZE REPLAY BUFFER
# =============================================================================

replay_buffer = ExperienceReplay()

# =============================================================================
#                   DEEP Q-NETWORK (DQN) IMPLEMENTATION
# =============================================================================

# Set up device for PyTorch (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNNet(nn.Module):
    """
    A simple feed-forward neural network for approximating Q-values.
    Input dimension: size of state (15 features expected)
    Output dimension: number of actions (len(ACTIONS))
    """
    def __init__(self, input_dim, output_dim):
        super(DQNNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        # Forward pass through the network using ReLU activations.
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    """
    The DQN Agent that encapsulates the policy network, target network,
    optimizer, and learning functions.
    """
    def __init__(self, input_dim, output_dim, device):
        self.device = device
        self.policy_net = DQNNet(input_dim, output_dim).to(device)
        self.target_net = DQNNet(input_dim, output_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=ALPHA)
        self.epsilon = EPSILON_START
        self.epsilon_end = EPSILON_END
        self.epsilon_decay = EPSILON_DECAY
        self.gamma = GAMMA
        self.update_count = 0

    def select_action(self, state, available_actions):
        """
        Choose an action based on the current state.
        The state is a tuple of features that is converted into a tensor.
        The agent uses an epsilon-greedy policy.
        """
        # Convert the state tuple to a torch tensor
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        # Epsilon-greedy: with probability epsilon choose a random action
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        # Otherwise, use the policy network to predict Q-values
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy().flatten()
        best_action = None
        best_q = -float("inf")
        # Select the best action among the available ones
        for action in available_actions:
            idx = ACTIONS.index(action)
            if q_values[idx] > best_q:
                best_q = q_values[idx]
                best_action = action
        if best_action is None:
            best_action = random.choice(available_actions)
        return best_action

    def optimize_model(self, replay_buffer, batch_size):
        """
        Sample a batch of transitions from the replay buffer and perform
        a single step of optimization on the policy network.
        """
        if len(replay_buffer) < batch_size:
            return 0
        transitions = replay_buffer.sample(batch_size)
        # Unpack the transitions into separate batches
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
        batch_state = torch.tensor(batch_state, dtype=torch.float32, device=self.device)
        # Convert the action tuples to indices
        batch_action = torch.tensor([ACTIONS.index(a) for a in batch_action], dtype=torch.int64,
                                    device=self.device).unsqueeze(1)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        batch_next_state = torch.tensor(batch_next_state, dtype=torch.float32, device=self.device)
        batch_done = torch.tensor(batch_done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Compute Q values for the current states using the policy network
        current_q_values = self.policy_net(batch_state).gather(1, batch_action)

        # Compute next state Q values using the target network
        with torch.no_grad():
            next_q_values = self.target_net(batch_next_state).max(1)[0].unsqueeze(1)
        # Compute expected Q values using the Bellman equation
        expected_q_values = batch_reward + (1 - batch_done) * self.gamma * next_q_values

        # Calculate loss (MSE loss)
        loss = nn.MSELoss()(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Increase update count and update the target network periodically
        self.update_count += 1
        if self.update_count % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return loss.item()

# =============================================================================
#                       INITIALIZE DQN AGENT & MODEL SAVE/LOAD
# =============================================================================

agent = DQNAgent(input_dim=15, output_dim=len(ACTIONS), device=device)

def save_model():
    """Save the DQN model weights to a file."""
    try:
        torch.save(agent.policy_net.state_dict(), "dqn_model.pth")
        print("Saved DQN model.")
    except Exception as e:
        print("Error saving DQN model:", e)

def load_model():
    """Load the DQN model weights from a file if available."""
    if os.path.exists("dqn_model.pth"):
        try:
            agent.policy_net.load_state_dict(torch.load("dqn_model.pth", map_location=device))
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            print("Loaded DQN model.")
        except Exception as e:
            print("Error loading DQN model:", e)

# Attempt to load an existing model on startup
load_model()
# Register the model saving function to be called at exit.
atexit.register(save_model)

# =============================================================================
#                       STATE REPRESENTATION FUNCTION
# =============================================================================

def get_state(ai_pos, player_pos, ai_dir, player_trail, ai_trail, invisibility_active):
    """
    Returns an enriched state representation as a tuple.
    The state consists of relative direction, distance, danger indicators,
    and some extra features like trail density.
    """
    # Calculate relative position differences
    rel_x = player_pos[0][0] - ai_pos[0][0]
    rel_y = player_pos[0][1] - ai_pos[0][1]

    # More granular direction representation (8 directions)
    if abs(rel_x) > abs(rel_y) * 2:
        direction = 2 if rel_x > 0 else 6
    elif abs(rel_y) > abs(rel_x) * 2:
        direction = 0 if rel_y < 0 else 4
    elif rel_x > 0 and rel_y < 0:
        direction = 1
    elif rel_x > 0 and rel_y > 0:
        direction = 3
    elif rel_x < 0 and rel_y > 0:
        direction = 5
    else:
        direction = 7

    # More granular distance representation
    if abs(rel_x) > 15 or abs(rel_y) > 15:
        dist = 3
    elif abs(rel_x) > 10 or abs(rel_y) > 10:
        dist = 2
    elif abs(rel_x) > 5 or abs(rel_y) > 5:
        dist = 1
    else:
        dist = 0  # Very close

    # Detect immediate and secondary dangers
    dangers = []
    secondary_dangers = []
    for action in ACTIONS:
        next_x = ai_pos[0][0] + action[0]
        next_y = ai_pos[0][1] + action[1]
        danger = 1 if (next_x < 0 or next_x >= COLS or next_y < 0 or next_y >= ROWS or
                       (next_x, next_y) in player_trail or (next_x, next_y) in ai_trail) else 0
        dangers.append(danger)

        next2_x = next_x + action[0]
        next2_y = next_y + action[1]
        secondary_danger = 1 if (next2_x < 0 or next2_x >= COLS or next2_y < 0 or next2_y >= ROWS or
                                 (next2_x, next2_y) in player_trail or (next2_x, next2_y) in ai_trail) else 0
        secondary_dangers.append(secondary_danger)

    # Determine the current movement direction index
    dir_value = ACTIONS.index(ai_dir)

    # Determine player's movement direction
    player_dir_x = player_pos[0][0] - (player_trail[0][0] if player_trail else player_pos[0][0])
    player_dir_y = player_pos[0][1] - (player_trail[0][1] if player_trail else player_pos[0][1])
    player_dir_value = -1  # Default if no movement detected
    if (player_dir_x, player_dir_y) in ACTIONS:
        player_dir_value = ACTIONS.index((player_dir_x, player_dir_y))

    # Count how many trail positions are near the AI and the player
    ai_trail_near = sum(1 for pos in ai_trail if abs(ai_pos[0][0] - pos[0]) + abs(ai_pos[0][1] - pos[1]) < 5)
    player_trail_near = sum(1 for pos in player_trail if abs(ai_pos[0][0] - pos[0]) + abs(ai_pos[0][1] - pos[1]) < 5)

    # Return a tuple representing the current state
    return (direction, dist, dangers[0], dangers[1], dangers[2], dangers[3],
            secondary_dangers[0], secondary_dangers[1], secondary_dangers[2], secondary_dangers[3],
            dir_value, player_dir_value, int(invisibility_active), ai_trail_near, player_trail_near)

# =============================================================================
#                   COUNT OPEN CELLS FUNCTION (LOOKAHEAD)
# =============================================================================

def count_open_cells(x, y, direction, player_trail, ai_trail, depth=5):
    """
    Count the number of open cells from a given (x,y) position
    using a breadth-first search up to a specified depth.
    """
    from collections import deque as local_deque
    visited = set()
    queue = local_deque([(x, y, depth)])
    count = 0
    while queue:
        cx, cy, d = queue.popleft()
        if (cx, cy) in visited:
            continue
        visited.add((cx, cy))
        count += 1
        if d <= 0:
            continue
        for dx, dy in ACTIONS:
            nx, ny = cx + dx, cy + dy
            if (nx < 0 or nx >= COLS or ny < 0 or ny >= ROWS or
                    (nx, ny) in player_trail or (nx, ny) in ai_trail or
                    (nx, ny) in visited):
                continue
            queue.append((nx, ny, d - 1))
    return count

# =============================================================================
#                     MOVE EVALUATION WITH LOOKAHEAD
# =============================================================================

def evaluate_move(pos, direction, player_pos, player_dir, player_trail, ai_trail, depth=3):
    if depth == 0:
        return 0

    next_x = pos[0] + direction[0]
    next_y = pos[1] + direction[1]

    # Collision detection: penalize moves that hit a wall or trail
    if (next_x < 0 or next_x >= COLS or next_y < 0 or next_y >= ROWS or
            (next_x, next_y) in player_trail or (next_x, next_y) in ai_trail):
        return -1000

    # Base score based on available open space
    open_space = count_open_cells(next_x, next_y, direction, player_trail, ai_trail, depth=5)
    score = open_space * 2

    # Predict player's next position
    predicted_player_x = player_pos[0][0] + player_dir[0]
    predicted_player_y = player_pos[0][1] + player_dir[1]

    # Calculate Manhattan distance to the player
    distance_to_player = abs(next_x - player_pos[0][0]) + abs(next_y - player_pos[0][1])

    # Strong bonus for moving towards the player
    prev_distance = abs(pos[0] - player_pos[0][0]) + abs(pos[1] - player_pos[0][1])
    new_distance = abs(next_x - player_pos[0][0]) + abs(next_y - player_pos[0][1])
    if new_distance < prev_distance:
        score += 50  # Reward for moving closer to the player
    else:
        score -= 30  # Penalize moving away from the player

    # Bonus for potential interception if close to the player
    if distance_to_player < 10:
        for d in range(1, 5):
            intercept_x = predicted_player_x + player_dir[0] * d
            intercept_y = predicted_player_y + player_dir[1] * d
            if abs(next_x - intercept_x) + abs(next_y - intercept_y) <= d:
                score += 100  # Bonus for interception
                break

    # Evaluate follow-up moves recursively to avoid dead ends
    follow_up_scores = []
    for next_dir in get_available_actions((direction[0], direction[1])):
        temp_ai_trail = list(ai_trail)
        temp_ai_trail.append((next_x, next_y))
        follow_up_score = evaluate_move((next_x, next_y), next_dir, player_pos, player_dir,
                                        player_trail, temp_ai_trail, depth - 1)
        follow_up_scores.append(follow_up_score)
    if follow_up_scores:
        best_follow_up = max(follow_up_scores)
        score += best_follow_up * 0.7

    # Edge strategy: prefer moving along the wall but penalize corners
    if (next_x <= 2 or next_x >= COLS - 3 or next_y <= 2 or next_y >= ROWS - 3):
        wall_count = 0
        if next_x <= 2: wall_count += 1
        if next_x >= COLS - 3: wall_count += 1
        if next_y <= 2: wall_count += 1
        if next_y >= ROWS - 3: wall_count += 1
        if wall_count == 1:
            score += 15
        elif wall_count > 1:
            score -= 20

    # Bonus for momentum: reward continuing in the same direction
    score += 10

    # NEW: Additional penalty for being too close to the AI's own trail
    if ai_trail:
        min_trail_distance = min(abs(next_x - t[0]) + abs(next_y - t[1]) for t in ai_trail)
        # If too close (e.g., less than 3 cells away), impose a heavy penalty.
        if min_trail_distance < 3:
            score -= (3 - min_trail_distance) * 200

    return score

def get_available_actions(ai_dir):
    """
    Return a list of actions that are not the reverse of the current direction.
    """
    opposite_dir = (-ai_dir[0], -ai_dir[1])
    return [action for action in ACTIONS if action != opposite_dir]

# =============================================================================
#                       UTILITY FUNCTIONS FOR UI
# =============================================================================

def create_button(screen, text, position, width, height, color, hover_color):
    """
    Create a clickable button for the game menu or messages.
    """
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    button_rect = pygame.Rect(position[0], position[1], width, height)
    is_hovered = button_rect.collidepoint(mouse)
    pygame.draw.rect(screen, hover_color if is_hovered else color, button_rect)
    font = pygame.font.Font(None, 36)
    text_surface = font.render(text, True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=button_rect.center)
    screen.blit(text_surface, text_rect)
    if is_hovered and click[0] == 1:
        pygame.time.delay(200)
        return True
    return False

def display_message(screen, message):
    """
    Display a full-screen message (e.g., game over) with options to restart,
    return to the menu, or quit the game.
    """
    font = pygame.font.Font(None, 72)
    screen.fill((0, 0, 0))
    text = font.render(message, True, (255, 255, 255))
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 3))
    screen.blit(text, text_rect)
    small_font = pygame.font.Font(None, 36)
    key_text = small_font.render("Press R to Restart | Q to Quit", True, (255, 255, 255))
    key_text_rect = key_text.get_rect(center=(WIDTH // 2, HEIGHT // 3 + 80))
    screen.blit(key_text, key_text_rect)
    save_model()

    # Loop until the user makes a choice.
    while True:
        restart_clicked = create_button(screen, "Restart", (WIDTH // 2 - 300, HEIGHT // 2), 200, 50, (0, 128, 0),
                                        (0, 200, 0))
        menu_clicked = create_button(screen, "Main Menu", (WIDTH // 2 - 100, HEIGHT // 2), 200, 50, (0, 0, 128),
                                     (0, 0, 200))
        quit_clicked = create_button(screen, "Quit", (WIDTH // 2 + 100, HEIGHT // 2), 200, 50, (128, 0, 0), (200, 0, 0))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_model()
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return "restart"
                elif event.key == pygame.K_m:
                    return "menu"
                elif event.key == pygame.K_q:
                    save_model()
                    pygame.quit()
                    exit()
        if restart_clicked:
            return "restart"
        elif menu_clicked:
            return "menu"
        elif quit_clicked:
            save_model()
            pygame.quit()
            exit()

# =============================================================================
#                             MENU SCREEN
# =============================================================================

def menu_screen():
    """
    Display the main menu where the player can select colors and start the game.
    """
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.Font(None, 36)
    colors_list = list(COLORS.keys())
    player_color, ai_color = random.sample(colors_list, 2)
    title_font = pygame.font.Font(None, 72)
    while True:
        screen.fill((0, 0, 0))
        title_text = title_font.render("Speeds7ers", True, (255, 255, 255))
        screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, HEIGHT // 8))
        select_text = font.render("Select Player and AI Colors", True, (255, 255, 255))
        screen.blit(select_text, (WIDTH // 2 - select_text.get_width() // 2, HEIGHT // 4))
        player_text = font.render(f"Player Color: {player_color}", True, COLORS[player_color])
        ai_text = font.render(f"AI Color: {ai_color}", True, COLORS[ai_color])
        screen.blit(player_text, (WIDTH // 2 - player_text.get_width() // 2, HEIGHT // 3))
        screen.blit(ai_text, (WIDTH // 2 - ai_text.get_width() // 2, HEIGHT // 2))
        instruction_text = font.render("Click to select colors", True, (200, 200, 200))
        screen.blit(instruction_text, (WIDTH // 2 - instruction_text.get_width() // 2, int(HEIGHT // 1.5)))
        key_instr_text = font.render("Press Enter to Start | Q to Quit", True, (200, 200, 200))
        screen.blit(key_instr_text, (WIDTH // 2 - key_instr_text.get_width() // 2, int(HEIGHT // 1.5) + 40))

        # Added gap between text and buttons
        instruction_gap = 60  # gap in pixels
        button_y = int(HEIGHT // 1.5) + 40 + instruction_gap
        start_clicked = create_button(screen, "Start Game", (WIDTH // 2 - 200, button_y), 200, 50, (0, 128, 0),
                                      (0, 200, 0))
        quit_clicked = create_button(screen, "Quit Game", (WIDTH // 2 + 50, button_y), 200, 50, (128, 0, 0),
                                     (200, 0, 0))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_model()
                pygame.quit()
                return None, None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    return player_color, ai_color
                elif event.key == pygame.K_q:
                    save_model()
                    pygame.quit()
                    exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if HEIGHT // 3 - 20 < y < HEIGHT // 3 + 20:
                    player_color = random.choice([c for c in colors_list if c != ai_color])
                elif HEIGHT // 2 - 20 < y < HEIGHT // 2 + 20:
                    ai_color = random.choice([c for c in colors_list if c != player_color])
        if start_clicked:
            return player_color, ai_color
        elif quit_clicked:
            save_model()
            pygame.quit()
            exit()

# =============================================================================
#                             GAME LOOP FUNCTION
# =============================================================================

def game_loop(player_color, ai_color):
    """
    The main game loop that updates the game state, handles user inputs,
    moves the player and the AI, processes power-ups, and trains the DQN agent.
    """
    global current_difficulty
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    running = True

    # Variables for power-up message display
    powerup_message = ""
    powerup_message_timer = 0  # frame counter for message duration

    # -------------------- Power-Up System Setup --------------------
    powerups = []
    max_powerups = 3
    powerup_spawn_probability = 0.1
    powerup_types = ["double_speed", "invisibility", "wall_phase", "extra_life", "ai_confusion", "teleport",
                     "slow_time"]
    POWERUP_DURATIONS = {
        "double_speed": 150,
        "invisibility": 150,
        "wall_phase": 150,
        "ai_confusion": 150,
        "slow_time": 150
    }
    # POWERUP_COLORS is still defined if a type is not in the sprites dictionary
    POWERUP_COLORS = {
        "double_speed": (255, 165, 0),
        "invisibility": (128, 128, 128),
        "wall_phase": (75, 0, 130),
        "extra_life": (255, 20, 147),
        "ai_confusion": (64, 224, 208),
        "teleport": (255, 255, 0),
        "slow_time": (0, 191, 255)
    }
    active_effects = {
        "double_speed": 0,
        "invisibility": 0,
        "wall_phase": 0,
        "ai_confusion": 0,
        "slow_time": 0
    }
    extra_lives = 0
    ai_move_counter = 0
    # ----------------------------------------------------------------

    # Initialize trails and positions using deque for fixed-length history
    player_trail = deque(maxlen=50)
    ai_trail = deque(maxlen=50)
    player_pos = [(5, ROWS // 2)]
    ai_pos = [(COLS - 5, ROWS // 2)]
    player_dir, ai_dir = (1, 0), (-1, 0)

    print(f"Initial Player pos: {player_pos[0]}, AI pos: {ai_pos[0]}")
    game_over = False

    # For DQN, the state is always computed with invisibility off (False)
    current_state = get_state(ai_pos, player_pos, ai_dir, player_trail, ai_trail, False)
    steps_survived = 0
    total_games = 0
    training_mode = True
    pygame.time.delay(1000)

    # NEW: AI memory for tracking player patterns (not used in DQN but preserved)
    player_history = deque(maxlen=10)
    player_pattern = {}

    # Main game loop
    while running:
        clock.tick(FPS)
        screen.fill((0, 0, 0))

        # -------------------- Power-Up Spawning --------------------
        if len(powerups) < max_powerups and random.random() < powerup_spawn_probability:
            pu_type = random.choice(powerup_types)
            pu_x = random.randint(0, COLS - 1)
            pu_y = random.randint(0, ROWS - 1)
            if (pu_x, pu_y) not in player_trail and (pu_x, pu_y) not in ai_trail \
                    and (pu_x, pu_y) != player_pos[0] and (pu_x, pu_y) != ai_pos[0]:
                powerups.append({"type": pu_type, "pos": (pu_x, pu_y)})
        # Draw power-ups using sprites (if available) or fallback to rectangle
        for pu in powerups:
            if pu["type"] in powerup_sprites:
                screen.blit(powerup_sprites[pu["type"]], (pu["pos"][0] * GRID_SIZE, pu["pos"][1] * GRID_SIZE))
            else:
                pygame.draw.rect(screen, POWERUP_COLORS[pu["type"]],
                                 (pu["pos"][0] * GRID_SIZE, pu["pos"][1] * GRID_SIZE, POWERUP_SIZE, POWERUP_SIZE))
        # -------------------------------------------------------------

        # -------------------- Display Power-Up Message --------------------
        if powerup_message_timer > 0:
            msg_font = pygame.font.Font(None, 36)
            msg_surface = msg_font.render(powerup_message, True, (255, 255, 0))
            screen.blit(msg_surface, (WIDTH // 2 - msg_surface.get_width() // 2, 10))
            powerup_message_timer -= 1
        # ------------------------------------------------------------------

        # Decrement active power-up effect timers
        for effect in active_effects:
            if active_effects[effect] > 0:
                active_effects[effect] -= 1

        # -------------------- Event Handling --------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_model()
                running = False
                pygame.quit()
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    training_mode = not training_mode
                    print(f"Training mode: {'ON' if training_mode else 'OFF'}")
                elif event.key == pygame.K_UP:
                    current_difficulty = min(current_difficulty + 1, len(difficulty_levels) - 1)
                    print(f"Difficulty set to: {difficulty_levels[current_difficulty]}")
                elif event.key == pygame.K_DOWN:
                    current_difficulty = max(current_difficulty - 1, 0)
                    print(f"Difficulty set to: {difficulty_levels[current_difficulty]}")
                elif event.key == pygame.K_ESCAPE:
                    save_model()
                    return "menu"
        # ---------------------------------------------------------

        # -------------------- Player Movement Update --------------------
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] and player_dir != (0, 1):
            player_dir = (0, -1)
        elif keys[pygame.K_s] and player_dir != (0, -1):
            player_dir = (0, 1)
        elif keys[pygame.K_a] and player_dir != (1, 0):
            player_dir = (-1, 0)
        elif keys[pygame.K_d] and player_dir != (-1, 0):
            player_dir = (1, 0)

        num_player_moves = 2 if active_effects["double_speed"] > 0 else 1
        for move in range(num_player_moves):
            prev_player_pos = player_pos[0]
            new_player_pos = (player_pos[0][0] + player_dir[0], player_pos[0][1] + player_dir[1])
            if active_effects["wall_phase"] > 0:
                new_player_pos = (new_player_pos[0] % COLS, new_player_pos[1] % ROWS)
            player_pos.insert(0, new_player_pos)
            player_trail.append(prev_player_pos)

            # Update player history and movement pattern tracking
            player_history.append(player_dir)
            if len(player_history) >= 3:
                pattern_key = (player_history[-3], player_history[-2])
                if pattern_key not in player_pattern:
                    player_pattern[pattern_key] = {}
                current_move = player_history[-1]
                player_pattern[pattern_key][current_move] = player_pattern[pattern_key].get(current_move, 0) + 1

            # Collision detection with power-ups using rect collision
            player_rect = pygame.Rect(player_pos[0][0] * GRID_SIZE, player_pos[0][1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            for pu in powerups[:]:
                # Instead of using the full collision area, we create a smaller collision rect.
                drawn_x = pu["pos"][0] * GRID_SIZE
                drawn_y = pu["pos"][1] * GRID_SIZE
                offset = (POWERUP_COLLISION_SIZE - POWERUP_PICKUP_SIZE) // 2
                pu_rect = pygame.Rect(drawn_x + offset, drawn_y + offset, POWERUP_PICKUP_SIZE, POWERUP_PICKUP_SIZE)
                if player_rect.colliderect(pu_rect):
                    pu_type = pu["type"]
                    powerup_message = f"Power Up: {pu_type.replace('_', ' ').title()}!"
                    powerup_message_timer = FPS * 3
                    if pu_type in ["double_speed", "invisibility", "wall_phase", "ai_confusion", "slow_time"]:
                        active_effects[pu_type] = POWERUP_DURATIONS[pu_type]
                    elif pu_type == "extra_life":
                        extra_lives += 1
                    elif pu_type == "teleport":
                        valid = False
                        while not valid:
                            rand_pos = (random.randint(0, COLS - 1), random.randint(0, ROWS - 1))
                            if rand_pos not in player_trail and rand_pos not in ai_trail:
                                valid = True
                        player_pos[0] = rand_pos
                        current_state = get_state(ai_pos, player_pos, ai_dir, player_trail, ai_trail, False)
                    powerups.remove(pu)
            if active_effects["wall_phase"] <= 0 and (new_player_pos[0] < 0 or new_player_pos[0] >= COLS or
                                                      new_player_pos[1] < 0 or new_player_pos[1] >= ROWS):
                if extra_lives > 0:
                    extra_lives -= 1
                    player_pos = [(5, ROWS // 2)]
                    player_trail.clear()
                else:
                    game_over = True
                    game_over_message = "Game Over! You hit the wall!"
                    break
            if new_player_pos in list(player_trail)[1:]:
                if extra_lives > 0:
                    extra_lives -= 1
                    player_pos = [(5, ROWS // 2)]
                    player_trail.clear()
                else:
                    game_over = True
                    game_over_message = "Game Over! You hit your own trail!"
                    break
            if new_player_pos in ai_trail:
                if extra_lives > 0:
                    extra_lives -= 1
                    player_pos = [(5, ROWS // 2)]
                    player_trail.clear()
                else:
                    game_over = True
                    game_over_message = "Game Over! You collided with the AI trail!"
                    break
        if game_over:
            action = display_message(screen, game_over_message)
            if action == "restart":
                return "restart"
            elif action == "menu":
                return "menu"
            else:
                return action
        # --------------------------------------------------------------

        # -------------------- AI Movement Update --------------------
        if not (active_effects["slow_time"] > 0 and ai_move_counter % 2 == 1):
            prev_ai_pos = ai_pos[0]
            available_actions = get_available_actions(ai_dir)
            safe_actions = []
            action_scores = []
            for action in available_actions:
                next_x = ai_pos[0][0] + action[0]
                next_y = ai_pos[0][1] + action[1]
                # Only consider actions that don’t lead to an immediate collision
                if (next_x < 0 or next_x >= COLS or next_y < 0 or next_y >= ROWS or
                        (next_x, next_y) in player_trail or (next_x, next_y) in ai_trail):
                    continue
                score = evaluate_move(ai_pos[0], action, player_pos, player_dir, player_trail, ai_trail)
                # Penalize moves that are too close to AI's own trail
                for pos in ai_trail:
                    if abs(next_x - pos[0]) + abs(next_y - pos[1]) < 12:
                        score -= 100
                        break
                prev_distance = abs(player_pos[0][0] - ai_pos[0][0]) + abs(player_pos[0][1] - ai_pos[0][1])
                distance_to_player = abs(next_x - player_pos[0][0]) + abs(next_y - player_pos[0][1])
                distance_bonus = 0 if active_effects["invisibility"] > 0 else (prev_distance - distance_to_player) * 1.0
                score += distance_bonus
                if action == ai_dir:
                    score += 10  # Bonus for continuing in the same direction
                safe_actions.append(action)
                action_scores.append((action, score))

            # Choose an action from safe_actions if possible; otherwise, use fallback logic.
            if safe_actions:
                if active_effects["ai_confusion"] > 0 and random.random() < 0.2:
                    chosen_action = random.choice(safe_actions)
                else:
                    if not training_mode:
                        difficulty_epsilon = [0.4, 0.2, 0.1, 0.05, 0.01]
                        epsilon_backup = agent.epsilon
                        agent.epsilon = difficulty_epsilon[current_difficulty]
                        chosen_action = agent.select_action(current_state, safe_actions)
                        agent.epsilon = epsilon_backup
                    else:
                        chosen_action = agent.select_action(current_state, safe_actions)
            else:
                # Fallback: even if moves appear unsafe, pick the one with the highest (least negative) score.
                fallback_scores = []
                for action in available_actions:
                    score = evaluate_move(ai_pos[0], action, player_pos, player_dir, player_trail, ai_trail)
                    fallback_scores.append((action, score))
                if fallback_scores:
                    chosen_action = max(fallback_scores, key=lambda x: x[1])[0]
                else:
                    # No action available: keep moving in the current direction
                    chosen_action = ai_dir

            ai_dir = chosen_action

            # Compute the new position and double-check that it’s safe.
            new_ai_pos = (ai_pos[0][0] + ai_dir[0], ai_pos[0][1] + ai_dir[1])
            if (new_ai_pos[0] < 0 or new_ai_pos[0] >= COLS or new_ai_pos[1] < 0 or new_ai_pos[1] >= ROWS or
                    new_ai_pos in player_trail or new_ai_pos in ai_trail):
                # Try to find an alternative from safe_actions
                for action in safe_actions:
                    candidate = (ai_pos[0][0] + action[0], ai_pos[0][1] + action[1])
                    if (0 <= candidate[0] < COLS and 0 <= candidate[1] < ROWS and
                            candidate not in player_trail and candidate not in ai_trail):
                        new_ai_pos = candidate
                        ai_dir = action
                        break
            ai_pos.insert(0, new_ai_pos)
            ai_trail.append(prev_ai_pos)
        ai_move_counter += 1

        # -------------------- Reward Calculation --------------------
        prev_distance = abs(player_pos[0][0] - ai_pos[0][0]) + abs(player_pos[0][1] - ai_pos[0][1])
        new_distance = abs(player_pos[0][0] - ai_pos[0][0]) + abs(player_pos[0][1] - ai_pos[0][1])
        reward = 0.05 + 1.0 if new_distance < prev_distance else 0.05 - 0.5  # Stronger reward/penalty for distance
        steps_survived += 1
        player_head = player_pos[0]
        ai_head = ai_pos[0]
        player_space = count_open_cells(player_head[0], player_head[1], player_dir, player_trail, ai_trail, depth=3)
        ai_space = count_open_cells(ai_head[0], ai_head[1], ai_dir, player_trail, ai_trail, depth=3)
        space_ratio = ai_space / max(1, player_space)
        if space_ratio > 1.5:
            reward += 0.5  # Stronger reward for having more space
        elif space_ratio < 0.7:
            reward -= 0.3  # Stronger penalty for being in a tight space
        if steps_survived % 10 == 0:
            reward += 0.5  # Reward for surviving longer
        # --------------------------------------------------------------

        # Check for game-end conditions and assign terminal rewards/messages
        game_end = False
        if (player_pos[0][0] < 0 or player_pos[0][0] >= COLS or
                player_pos[0][1] < 0 or player_pos[0][1] >= ROWS):
            game_end = True
            reward = 50 + steps_survived * 0.2
            game_over_message = "Game Over! You hit the wall!"
        elif player_pos[0] in ai_trail:
            game_end = True
            reward = 50 + steps_survived * 0.2
            game_over_message = "Game Over! You collided with the AI trail!"
        elif new_ai_pos in player_trail:
            game_end = True
            reward = -50
            game_over_message = "You Win! AI hit your trail!"
        elif new_ai_pos in ai_trail:
            game_end = True
            reward = -50
            game_over_message = "You Win! AI hit its own trail!"
        elif (new_ai_pos[0] < 0 or new_ai_pos[0] >= COLS or
              new_ai_pos[1] < 0 or new_ai_pos[1] >= ROWS):
            game_end = True
            reward = -50
            game_over_message = "You Win! AI hit the wall!"
        elif player_pos[0] == new_ai_pos:
            game_end = True
            reward = 0
            game_over_message = "Draw! Head-on collision!"

        # -------------------- DQN Agent Training --------------------
        # Push the transition (state, action, reward, next_state, done) into replay buffer
        done_flag = 1 if game_end else 0
        replay_buffer.push(current_state, ai_dir, reward,
                           get_state(ai_pos, player_pos, ai_dir, player_trail, ai_trail, False), done_flag)
        if training_mode and steps_survived % 5 == 0:
            loss = agent.optimize_model(replay_buffer, BATCH_SIZE)
            # Optionally, you can print the loss value for debugging:
            # print(f"Loss: {loss:.4f}")
        current_state = get_state(ai_pos, player_pos, ai_dir, player_trail, ai_trail, False)
        # --------------------------------------------------------------

        if game_end:
            total_games += 1
            if total_games % 10 == 0:
                print(
                    f"Games played: {total_games}, Current epsilon: {agent.epsilon:.4f}, Updates: {agent.update_count}")
            action = display_message(screen, game_over_message)
            if action == "restart":
                return "restart"
            elif action == "menu":
                return "menu"
            else:
                return action

        # -------------------- Drawing the Trails and Heads --------------------
        for pos in player_trail:
            pygame.draw.rect(screen, COLORS[player_color],
                             (pos[0] * GRID_SIZE, pos[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        for pos in ai_trail:
            pygame.draw.rect(screen, COLORS[ai_color],
                             (pos[0] * GRID_SIZE, pos[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(screen, COLORS[player_color],
                         (player_pos[0][0] * GRID_SIZE - 1, player_pos[0][1] * GRID_SIZE - 1,
                          GRID_SIZE + 2, GRID_SIZE + 2))
        pygame.draw.rect(screen, COLORS[ai_color],
                         (ai_pos[0][0] * GRID_SIZE - 1, ai_pos[0][1] * GRID_SIZE - 1,
                          GRID_SIZE + 2, GRID_SIZE + 2))
        # ---------------------------------------------------------------------

        # -------------------- Display Game Info --------------------
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Player: {len(player_trail)}  AI: {len(ai_trail)}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))
        mode_text = font.render(f"Mode: {'Training' if training_mode else 'Playing'}", True, (255, 255, 255))
        screen.blit(mode_text, (10, 50))
        diff_text = font.render(f"Difficulty: {difficulty_levels[current_difficulty]}", True, (255, 255, 255))
        screen.blit(diff_text, (10, 90))
        if training_mode:
            eps_text = font.render(f"Epsilon: {agent.epsilon:.4f}", True, (255, 255, 255))
            screen.blit(eps_text, (10, 130))
            update_text = font.render(f"Updates: {agent.update_count}", True, (255, 255, 255))
            screen.blit(update_text, (10, 170))
        extra_text = font.render(f"Extra Lives: {extra_lives}", True, (255, 255, 255))
        screen.blit(extra_text, (10, 210))
        controls_text = font.render("ESC: Main Menu | T: Toggle Training | Up/Down: Change Difficulty", True,
                                    (200, 200, 200))
        screen.blit(controls_text, (WIDTH - controls_text.get_width() - 10, HEIGHT - 40))
        pygame.display.flip()
        # ---------------------------------------------------------------------
    return "menu"

# =============================================================================
#                                 MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function to start the game. It shows the menu screen and then enters
    the game loop.
    """
    pygame.display.set_caption("Speeds7ers")
    while True:
        player_color, ai_color = menu_screen()
        if player_color and ai_color:
            result = game_loop(player_color, ai_color)
            if result == "quit":
                break
        else:
            break

if __name__ == "__main__":
    try:
        main()
    finally:
        save_model()
        pygame.quit()
