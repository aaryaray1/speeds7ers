import pygame
import random
import numpy as np
import os

# Initialize Pygame
pygame.init()

# Game Constants
WIDTH, HEIGHT = 1280, 720
GRID_SIZE = 10
COLS, ROWS = WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE
FPS = 45

# Colors
COLORS = {"Red": (255, 0, 0), "Blue": (0, 0, 255), "Green": (0, 255, 0), "Yellow": (255, 255, 0),
          "Purple": (128, 0, 128), "Cyan": (0, 255, 255), "White": (255, 255, 255), "Grey": (100, 100, 100)}

# Reinforcement Learning Parameters
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
ALPHA = 0.1  # Learning rate
GAMMA = 0.95  # Discount factor (increased to value future rewards more)
EPSILON_START = 0.3  # Starting exploration rate
EPSILON_END = 0.05  # Minimum exploration rate
EPSILON_DECAY = 0.9999  # Rate at which exploration decreases
EPSILON = EPSILON_START  # Current exploration rate
Q_TABLE_FILE = "q_table.npy"

# Load or Initialize Q-Table
if os.path.exists(Q_TABLE_FILE):
    q_table = np.load(Q_TABLE_FILE, allow_pickle=True).item()
    print(f"Loaded Q-table with {len(q_table)} states")
else:
    q_table = {}
    print("Initialized new Q-table")


# Decay epsilon over time to reduce exploration as AI learns
def decay_epsilon():
    global EPSILON
    EPSILON = max(EPSILON_END, EPSILON * EPSILON_DECAY)


# Function to get state representation for Q-learning
def get_state(ai_pos, player_pos, ai_dir, player_trail, ai_trail):
    # Get relative position of player to AI (with more granularity)
    rel_x = player_pos[0][0] - ai_pos[0][0]
    rel_y = player_pos[0][1] - ai_pos[0][1]

    # Discretize relative position with more detail
    if abs(rel_x) > 10:
        rel_x = 2 if rel_x > 0 else -2  # Far right/left
    elif abs(rel_x) > 3:
        rel_x = 1 if rel_x > 0 else -1  # Medium right/left
    else:
        rel_x = 0  # Close horizontally

    if abs(rel_y) > 10:
        rel_y = 2 if rel_y > 0 else -2  # Far up/down
    elif abs(rel_y) > 3:
        rel_y = 1 if rel_y > 0 else -1  # Medium up/down
    else:
        rel_y = 0  # Close vertically

    # Check obstacles in four directions with distance
    obstacles = []
    for action in ACTIONS:
        # Look ahead up to 3 steps in this direction
        for distance in range(1, 4):
            next_x = ai_pos[0][0] + action[0] * distance
            next_y = ai_pos[0][1] + action[1] * distance

            # Check if next position is a wall
            wall = (next_x < 0 or next_x >= COLS or next_y < 0 or next_y >= ROWS)

            # Check if next position contains a trail
            trail = (next_x, next_y) in player_trail or (next_x, next_y) in ai_trail

            if wall or trail:
                obstacles.append(distance)  # Record distance to obstacle
                break
        else:
            obstacles.append(4)  # No obstacle within 3 steps

    # Direction of AI movement
    dir_value = ACTIONS.index(ai_dir)

    # Calculate open space in each direction (simple flood fill approximation)
    space_metrics = []
    for action in ACTIONS:
        open_count = count_open_cells(ai_pos[0][0], ai_pos[0][1], action, player_trail, ai_trail, depth=5)
        space_metrics.append(min(3, open_count // 3))  # Discretize to reduce state space

    # Return state as a tuple (can be used as dictionary key)
    return (
        rel_x, rel_y,
        obstacles[0], obstacles[1], obstacles[2], obstacles[3],
        space_metrics[0], space_metrics[1], space_metrics[2], space_metrics[3],
        dir_value
    )


# Helper function to count open cells in a direction (limited depth flood fill)
def count_open_cells(x, y, direction, player_trail, ai_trail, depth=5):
    if depth <= 0:
        return 0

    next_x = x + direction[0]
    next_y = y + direction[1]

    # Check boundaries and trails
    if (next_x < 0 or next_x >= COLS or next_y < 0 or next_y >= ROWS or
            (next_x, next_y) in player_trail or (next_x, next_y) in ai_trail):
        return 0

    # Count this cell plus recursive exploration in all directions
    count = 1
    for act in ACTIONS:
        if act != (-direction[0], -direction[1]):  # Don't go back the way we came
            count += count_open_cells(next_x, next_y, act, player_trail, ai_trail, depth - 1)

    return count


# Function to choose action based on Q-values and exploration
def choose_action(state, available_actions):
    if state not in q_table:
        # Initialize with small random values instead of zeros
        q_table[state] = np.random.uniform(0, 0.1, len(ACTIONS))

    # No valid actions, return random
    if not available_actions:
        return random.choice(ACTIONS)

    # Exploration (random action)
    if random.random() < EPSILON:
        return random.choice(available_actions)

    # Exploitation (best action)
    q_values = q_table[state]
    # Filter q_values to only include available actions
    available_q = [(i, q_values[i]) for i in range(len(ACTIONS)) if ACTIONS[i] in available_actions]

    if available_q:
        # Get action with highest Q-value among available actions
        return ACTIONS[max(available_q, key=lambda x: x[1])[0]]
    else:
        return random.choice(available_actions)


# Function to update Q-values
def update_q_value(state, action, reward, next_state):
    action_idx = ACTIONS.index(action)

    if state not in q_table:
        q_table[state] = np.random.uniform(0, 0.1, len(ACTIONS))

    if next_state not in q_table:
        q_table[next_state] = np.random.uniform(0, 0.1, len(ACTIONS))

    # Q-learning update formula
    q_table[state][action_idx] = q_table[state][action_idx] + ALPHA * (
            reward + GAMMA * np.max(q_table[next_state]) - q_table[state][action_idx]
    )


# Function to get available actions (prevent 180-degree turns)
def get_available_actions(ai_dir):
    # Cannot move in the opposite direction
    opposite_dir = (-ai_dir[0], -ai_dir[1])
    return [action for action in ACTIONS if action != opposite_dir]


# Enhanced function to evaluate how good a move is
def evaluate_move(pos, direction, player_trail, ai_trail):
    next_x = pos[0] + direction[0]
    next_y = pos[1] + direction[1]

    # Check if move is valid
    if (next_x < 0 or next_x >= COLS or next_y < 0 or next_y >= ROWS or
            (next_x, next_y) in player_trail or (next_x, next_y) in ai_trail):
        return -100  # Invalid move

    # Count open cells from this position (flood fill)
    open_space = count_open_cells(next_x, next_y, direction, player_trail, ai_trail, depth=8)

    return open_space


def display_message(screen, message):
    font = pygame.font.Font(None, 72)
    screen.fill((0, 0, 0))
    text = font.render(message, True, (255, 255, 255))
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 3))
    screen.blit(text, text_rect)

    button_font = pygame.font.Font(None, 48)
    retry_text = button_font.render("Press R to Restart or Q to Quit", True, (255, 255, 255))
    retry_rect = retry_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(retry_text, retry_rect)

    # Save Q-table when game ends
    np.save(Q_TABLE_FILE, q_table)
    print(f"Saved Q-table with {len(q_table)} states")

    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    main()
                    return
                elif event.key == pygame.K_q:
                    pygame.quit()
                    exit()


def menu_screen():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.Font(None, 36)

    colors_list = list(COLORS.keys())
    player_color, ai_color = random.sample(colors_list, 2)

    while True:
        screen.fill((0, 0, 0))
        title_text = font.render("Select Player and AI Colors", True, (255, 255, 255))
        screen.blit(title_text, (WIDTH // 6, HEIGHT // 6))

        player_text = font.render(f"Player Color: {player_color}", True, COLORS[player_color])
        ai_text = font.render(f"AI Color: {ai_color}", True, COLORS[ai_color])
        screen.blit(player_text, (WIDTH // 4, HEIGHT // 3))
        screen.blit(ai_text, (WIDTH // 4, HEIGHT // 2))

        instruction_text = font.render("Click to select, Press ENTER to start", True, (255, 255, 255))
        screen.blit(instruction_text, (WIDTH // 6, HEIGHT // 1.5))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None, None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    return player_color, ai_color
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if HEIGHT // 3 < y < HEIGHT // 3 + 30:
                    player_color = random.choice([c for c in colors_list if c != ai_color])
                elif HEIGHT // 2 < y < HEIGHT // 2 + 30:
                    ai_color = random.choice([c for c in colors_list if c != player_color])


def game_loop(player_color, ai_color):
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    running = True

    player_pos = [(5, ROWS // 2)]
    ai_pos = [(COLS - 5, ROWS // 2)]
    player_dir, ai_dir = (1, 0), (-1, 0)
    player_trail, ai_trail = set(), set()

    # DEBUG: Print initial positions
    print(f"Initial Player pos: {player_pos[0]}, AI pos: {ai_pos[0]}")

    # Track previous positions to add to trails
    prev_player_pos = player_pos[0]
    prev_ai_pos = ai_pos[0]

    # Game state
    game_over = False

    # AI learning variables
    current_state = get_state(ai_pos, player_pos, ai_dir, player_trail, ai_trail)
    steps_survived = 0
    total_games = 0

    # Display training info
    training_mode = True

    # AI difficulty settings
    difficulty_levels = ["Easy", "Medium", "Hard", "Expert"]
    current_difficulty = 1  # Medium by default

    # Add a delay before starting the game (1 second)
    pygame.time.delay(1000)

    while running:
        screen.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Save Q-table before quitting
                np.save(Q_TABLE_FILE, q_table)
                print(f"Saved Q-table with {len(q_table)} states")
                running = False
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    # Toggle training mode
                    training_mode = not training_mode
                    print(f"Training mode: {'ON' if training_mode else 'OFF'}")
                elif event.key == pygame.K_UP:
                    # Increase difficulty
                    current_difficulty = min(current_difficulty + 1, len(difficulty_levels) - 1)
                    print(f"Difficulty set to: {difficulty_levels[current_difficulty]}")
                elif event.key == pygame.K_DOWN:
                    # Decrease difficulty
                    current_difficulty = max(current_difficulty - 1, 0)
                    print(f"Difficulty set to: {difficulty_levels[current_difficulty]}")

        if not game_over:
            # Add previous positions to trails (not the initial positions or current positions)
            if steps_survived > 0:  # Only after the first move
                player_trail.add(prev_player_pos)
                ai_trail.add(prev_ai_pos)

            # Store current positions for next iteration's trail
            prev_player_pos = player_pos[0]
            prev_ai_pos = ai_pos[0]

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w] and player_dir != (0, 1):  # Prevent 180 degree turns
                player_dir = (0, -1)
            elif keys[pygame.K_s] and player_dir != (0, -1):
                player_dir = (0, 1)
            elif keys[pygame.K_a] and player_dir != (1, 0):
                player_dir = (-1, 0)
            elif keys[pygame.K_d] and player_dir != (-1, 0):
                player_dir = (1, 0)

            # Update player position
            new_player_pos = (player_pos[0][0] + player_dir[0], player_pos[0][1] + player_dir[1])
            player_pos.insert(0, new_player_pos)

            # AI decision making
            available_actions = get_available_actions(ai_dir)

            # Filter out actions that would lead to immediate collision
            safe_actions = []
            action_scores = []

            for action in available_actions:
                next_x = ai_pos[0][0] + action[0]
                next_y = ai_pos[0][1] + action[1]

                # Check if next position is safe
                if (next_x < 0 or next_x >= COLS or next_y < 0 or next_y >= ROWS or
                        (next_x, next_y) in player_trail or (next_x, next_y) in ai_trail):
                    # Skip unsafe actions
                    continue

                safe_actions.append(action)
                # Calculate a score for this action
                score = evaluate_move(ai_pos[0], action, player_trail, ai_trail)
                action_scores.append((action, score))

            # Adjust exploration based on difficulty
            local_epsilon = EPSILON
            if not training_mode:
                difficulty_epsilon = [0.4, 0.2, 0.1, 0.05]  # Easy, Medium, Hard, Expert
                local_epsilon = difficulty_epsilon[current_difficulty]

            # Choose action based on current state and difficulty
            if safe_actions:
                # In expert mode, sometimes use the best scored action instead of Q-learning
                if current_difficulty == 3 and random.random() < 0.3 and action_scores:
                    # Use the best scored action 30% of the time in expert mode
                    ai_dir = max(action_scores, key=lambda x: x[1])[0]
                else:
                    # Use Q-learning with appropriate epsilon
                    if random.random() < local_epsilon:
                        ai_dir = random.choice(safe_actions)
                    else:
                        ai_dir = choose_action(current_state, safe_actions)
            else:
                # No safe moves, try any available move
                ai_dir = choose_action(current_state, available_actions)

            # Update AI position
            new_ai_pos = (ai_pos[0][0] + ai_dir[0], ai_pos[0][1] + ai_dir[1])
            ai_pos.insert(0, new_ai_pos)

            # Calculate reward
            reward = 0.1  # Small positive reward for surviving
            steps_survived += 1

            # Add additional reward for gaining territory
            if len(ai_trail) > len(player_trail):
                reward += 0.05  # Slight bonus for having more territory

            # Bigger reward for surviving longer
            if steps_survived % 10 == 0:
                reward += 0.5

            # Calculate new state
            next_state = get_state(ai_pos, player_pos, ai_dir, player_trail, ai_trail)

            # Check all collision scenarios
            game_end = False

            # 1. Wall collision
            if (new_player_pos[0] < 0 or new_player_pos[0] >= COLS or
                    new_player_pos[1] < 0 or new_player_pos[1] >= ROWS):
                display_message(screen, "Game Over! You hit the wall!")
                game_end = True
                reward = 5  # Bigger reward for AI when player hits wall

            # 2. Player hitting AI trail
            elif new_player_pos in ai_trail:
                display_message(screen, "Game Over! You collided with the AI trail!")
                # Big reward for AI if player hits it
                reward = 10
                game_end = True

            # 3. AI hitting player trail
            elif new_ai_pos in player_trail:
                display_message(screen, "You Win! AI hit your trail!")
                # Big penalty for hitting player trail
                reward = -10
                game_end = True

            # 4. Player hitting own trail
            elif new_player_pos in player_trail:
                display_message(screen, "Game Over! You hit your own trail!")
                # Small reward for AI if player makes mistake
                reward = 3
                game_end = True

            # 5. AI hitting own trail
            elif new_ai_pos in ai_trail:
                display_message(screen, "You Win! AI hit its own trail!")
                # Big penalty for hitting own trail
                reward = -10
                game_end = True

            # 6. AI hitting wall
            elif (new_ai_pos[0] < 0 or new_ai_pos[0] >= COLS or
                  new_ai_pos[1] < 0 or new_ai_pos[1] >= ROWS):
                display_message(screen, "You Win! AI hit the wall!")
                # Big penalty for hitting wall
                reward = -10
                game_end = True

            # 7. Head-on collision (both lose)
            elif new_player_pos == new_ai_pos:
                display_message(screen, "Draw! Head-on collision!")
                reward = 0  # Neutral reward for draw
                game_end = True

            # Update Q-values if in training mode
            if training_mode:
                update_q_value(current_state, ai_dir, reward, next_state)
                # Decay epsilon over time
                decay_epsilon()

            # Update current state for next iteration
            current_state = next_state

            if game_end:
                total_games += 1
                if total_games % 10 == 0:
                    print(f"Games played: {total_games}, Current epsilon: {EPSILON:.4f}")
                return

        # Draw trails
        for pos in player_trail:
            pygame.draw.rect(screen, COLORS[player_color],
                             (pos[0] * GRID_SIZE, pos[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        for pos in ai_trail:
            pygame.draw.rect(screen, COLORS[ai_color], (pos[0] * GRID_SIZE, pos[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        # Draw player and AI heads with a slightly larger size to distinguish them
        pygame.draw.rect(screen, COLORS[player_color],
                         (player_pos[0][0] * GRID_SIZE - 1, player_pos[0][1] * GRID_SIZE - 1,
                          GRID_SIZE + 2, GRID_SIZE + 2))
        pygame.draw.rect(screen, COLORS[ai_color],
                         (ai_pos[0][0] * GRID_SIZE - 1, ai_pos[0][1] * GRID_SIZE - 1,
                          GRID_SIZE + 2, GRID_SIZE + 2))

        # Display score and game info
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Player: {len(player_trail)}  AI: {len(ai_trail)}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))

        # Display training status
        mode_text = font.render(f"Mode: {'Training' if training_mode else 'Playing'}", True, (255, 255, 255))
        screen.blit(mode_text, (10, 50))

        # Display difficulty
        diff_text = font.render(f"Difficulty: {difficulty_levels[current_difficulty]}", True, (255, 255, 255))
        screen.blit(diff_text, (10, 90))

        # Display epsilon value if in training mode
        if training_mode:
            eps_text = font.render(f"Epsilon: {EPSILON:.4f}", True, (255, 255, 255))
            screen.blit(eps_text, (10, 130))

        pygame.display.flip()
        clock.tick(FPS)


def main():
    pygame.display.set_caption("Tron Game with Adaptive AI")
    player_color, ai_color = menu_screen()
    if player_color and ai_color:
        game_loop(player_color, ai_color)


if __name__ == "__main__":
    main()