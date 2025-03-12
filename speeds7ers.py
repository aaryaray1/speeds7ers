import pygame
import random
import numpy as np
import os
from collections import deque
import atexit

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
ALPHA = 0.1   # Learning rate
GAMMA = 0.95  # Discount factor
EPSILON_START = 0.7   # Starting exploration rate
EPSILON_END = 0.05    # Minimum exploration rate
EPSILON_DECAY = 0.999 # (Decay logic is in decay_epsilon)
EPSILON = EPSILON_START  # Current exploration rate
Q_TABLE_FILE = "q_table.npy"
BATCH_SIZE = 32  # Experience replay batch size

# Global difficulty settings (persist across rounds until app restart)
difficulty_levels = ["Easy", "Medium", "Hard", "Expert"]
current_difficulty = 1  # Default to Medium

# Experience Replay buffer
class ExperienceReplay:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)

# Initialize replay buffer
replay_buffer = ExperienceReplay()

# Load or Initialize Q-Table (persistent training)
if os.path.exists(Q_TABLE_FILE):
    q_table = np.load(Q_TABLE_FILE, allow_pickle=True).item()
    print(f"Loaded Q-table with {len(q_table)} states")
else:
    q_table = {}
    print("Initialized new Q-table")

# Ensure Q-table is saved on exit
atexit.register(lambda: np.save(Q_TABLE_FILE, q_table))

# Improved epsilon decay function
def decay_epsilon():
    global EPSILON
    if len(q_table) < 1000:  # Early in training
        EPSILON = max(EPSILON_END, EPSILON * 0.995)  # Faster decay early
    else:
        EPSILON = max(EPSILON_END, EPSILON * 0.999)  # Slower decay later

# Simplified state representation for Q-learning
def get_state(ai_pos, player_pos, ai_dir, player_trail, ai_trail):
    rel_x = player_pos[0][0] - ai_pos[0][0]
    rel_y = player_pos[0][1] - ai_pos[0][1]

    if abs(rel_x) > abs(rel_y) * 2:
        direction = 2 if rel_x > 0 else 6  # E or W
    elif abs(rel_y) > abs(rel_x) * 2:
        direction = 0 if rel_y < 0 else 4  # N or S
    elif rel_x > 0 and rel_y < 0:
        direction = 1  # NE
    elif rel_x > 0 and rel_y > 0:
        direction = 3  # SE
    elif rel_x < 0 and rel_y > 0:
        direction = 5  # SW
    else:
        direction = 7  # NW

    distance = 0  # close
    if abs(rel_x) > 10 or abs(rel_y) > 10:
        distance = 2  # far
    elif abs(rel_x) > 5 or abs(rel_y) > 5:
        distance = 1  # medium

    dangers = []
    for action in ACTIONS:
        next_x = ai_pos[0][0] + action[0]
        next_y = ai_pos[0][1] + action[1]
        danger = 1 if (next_x < 0 or next_x >= COLS or next_y < 0 or next_y >= ROWS or
                       (next_x, next_y) in player_trail or (next_x, next_y) in ai_trail) else 0
        dangers.append(danger)

    dir_value = ACTIONS.index(ai_dir)
    return (direction, distance, dangers[0], dangers[1], dangers[2], dangers[3], dir_value)

# Optimized function to count open cells
def count_open_cells(x, y, direction, player_trail, ai_trail, depth=5):
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

# Function to choose action based on Q-values and exploration
def choose_action(state, available_actions):
    if state not in q_table:
        q_table[state] = np.random.uniform(0, 0.1, len(ACTIONS))
    if not available_actions:
        return random.choice(ACTIONS)
    if random.random() < EPSILON:
        return random.choice(available_actions)
    q_values = q_table[state]
    available_q = [(i, q_values[i]) for i in range(len(ACTIONS)) if ACTIONS[i] in available_actions]
    if available_q:
        return ACTIONS[max(available_q, key=lambda x: x[1])[0]]
    else:
        return random.choice(available_actions)

# Prioritized sweeping for more efficient updates
def prioritized_sweeping(state, action, reward, next_state, threshold=0.5):
    action_idx = ACTIONS.index(action)
    if state not in q_table:
        q_table[state] = np.random.uniform(0, 0.1, len(ACTIONS))
    if next_state not in q_table:
        q_table[next_state] = np.random.uniform(0, 0.1, len(ACTIONS))
    td_error = abs(reward + GAMMA * np.max(q_table[next_state]) - q_table[state][action_idx])
    if td_error > threshold:
        q_table[state][action_idx] += ALPHA * (reward + GAMMA * np.max(q_table[next_state]) - q_table[state][action_idx])
        return True
    return False

# Learn from experience replay buffer
def learn_from_experience(batch_size=BATCH_SIZE):
    if len(replay_buffer) < batch_size:
        return
    experiences = replay_buffer.sample(batch_size)
    updates_made = 0
    for state, action, reward, next_state in experiences:
        if prioritized_sweeping(state, action, reward, next_state):
            updates_made += 1
    return updates_made

# Function to get available actions (prevent 180-degree turns)
def get_available_actions(ai_dir):
    opposite_dir = (-ai_dir[0], -ai_dir[1])
    return [action for action in ACTIONS if action != opposite_dir]

# Enhanced function to evaluate how good a move is
def evaluate_move(pos, direction, player_trail, ai_trail):
    next_x = pos[0] + direction[0]
    next_y = pos[1] + direction[1]
    if (next_x < 0 or next_x >= COLS or next_y < 0 or next_y >= ROWS or
            (next_x, next_y) in player_trail or (next_x, next_y) in ai_trail):
        return -100  # Invalid move
    open_space = count_open_cells(next_x, next_y, direction, player_trail, ai_trail, depth=5)
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
                    main()  # Restart game loop without resetting difficulty
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
    global current_difficulty  # Use global difficulty so it persists
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    running = True

    # Use deque for trails so that old segments disappear
    player_trail = deque(maxlen=50)
    ai_trail = deque(maxlen=50)

    player_pos = [(5, ROWS // 2)]
    ai_pos = [(COLS - 5, ROWS // 2)]
    player_dir, ai_dir = (1, 0), (-1, 0)

    print(f"Initial Player pos: {player_pos[0]}, AI pos: {ai_pos[0]}")
    prev_player_pos = player_pos[0]
    prev_ai_pos = ai_pos[0]
    game_over = False

    current_state = get_state(ai_pos, player_pos, ai_dir, player_trail, ai_trail)
    steps_survived = 0
    total_games = 0
    learning_updates = 0
    training_mode = True
    pygame.time.delay(1000)

    while running:
        screen.fill((0, 0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                np.save(Q_TABLE_FILE, q_table)
                print(f"Saved Q-table with {len(q_table)} states")
                running = False
                pygame.quit()
                return
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

        if not game_over:
            if steps_survived > 0:
                player_trail.append(prev_player_pos)
                ai_trail.append(prev_ai_pos)
            prev_player_pos = player_pos[0]
            prev_ai_pos = ai_pos[0]

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w] and player_dir != (0, 1):
                player_dir = (0, -1)
            elif keys[pygame.K_s] and player_dir != (0, -1):
                player_dir = (0, 1)
            elif keys[pygame.K_a] and player_dir != (1, 0):
                player_dir = (-1, 0)
            elif keys[pygame.K_d] and player_dir != (-1, 0):
                player_dir = (1, 0)

            new_player_pos = (player_pos[0][0] + player_dir[0], player_pos[0][1] + player_dir[1])
            player_pos.insert(0, new_player_pos)

            prev_distance = abs(player_pos[0][0] - ai_pos[0][0]) + abs(player_pos[0][1] - ai_pos[0][1])
            available_actions = get_available_actions(ai_dir)
            safe_actions = []
            action_scores = []

            for action in available_actions:
                next_x = ai_pos[0][0] + action[0]
                next_y = ai_pos[0][1] + action[1]
                if (next_x < 0 or next_x >= COLS or next_y < 0 or next_y >= ROWS or
                        (next_x, next_y) in player_trail or (next_x, next_y) in ai_trail):
                    continue
                # Evaluate move score using evaluate_move
                score = evaluate_move(ai_pos[0], action, player_trail, ai_trail)
                # Enforce an 8-block gap from its own trail
                for pos in ai_trail:
                    if abs(next_x - pos[0]) + abs(next_y - pos[1]) < 8:
                        score -= 50
                        break
                # Bonus: reward moves that reduce the distance to the player's head
                distance_to_player = abs(next_x - player_pos[0][0]) + abs(next_y - player_pos[0][1])
                distance_bonus = (prev_distance - distance_to_player) * 0.5
                score += distance_bonus
                # Bonus for maintaining current direction (to reduce zigzagging)
                if action == ai_dir:
                    score += 5

                safe_actions.append(action)
                action_scores.append((action, score))

            local_epsilon = EPSILON
            if not training_mode:
                difficulty_epsilon = [0.4, 0.2, 0.1, 0.05]
                local_epsilon = difficulty_epsilon[current_difficulty]

            if safe_actions:
                if current_difficulty == 3 and random.random() < 0.3 and action_scores:
                    ai_dir = max(action_scores, key=lambda x: x[1])[0]
                else:
                    if random.random() < local_epsilon:
                        ai_dir = random.choice(safe_actions)
                    else:
                        ai_dir = choose_action(current_state, safe_actions)
            else:
                ai_dir = choose_action(current_state, available_actions)

            new_ai_pos = (ai_pos[0][0] + ai_dir[0], ai_pos[0][1] + ai_dir[1])
            ai_pos.insert(0, new_ai_pos)

            new_distance = abs(player_pos[0][0] - ai_pos[0][0]) + abs(player_pos[0][1] - ai_pos[0][1])
            # Increase bonus for moving closer to the player
            if new_distance < prev_distance:
                reward = 0.05 + 0.5
            else:
                reward = 0.05 - 0.2

            steps_survived += 1
            player_head = player_pos[0]
            ai_head = ai_pos[0]
            player_space = count_open_cells(player_head[0], player_head[1], player_dir, player_trail, ai_trail, depth=3)
            ai_space = count_open_cells(ai_head[0], ai_head[1], ai_dir, player_trail, ai_trail, depth=3)
            space_ratio = ai_space / max(1, player_space)
            if space_ratio > 1.5:
                reward += 0.2
            elif space_ratio < 0.7:
                reward -= 0.1
            if steps_survived % 10 == 0:
                reward += 0.3

            next_state = get_state(ai_pos, player_pos, ai_dir, player_trail, ai_trail)
            game_end = False

            if (new_player_pos[0] < 0 or new_player_pos[0] >= COLS or
                    new_player_pos[1] < 0 or new_player_pos[1] >= ROWS):
                display_message(screen, "Game Over! You hit the wall!")
                game_end = True
                reward = 50 + steps_survived * 0.2
            elif new_player_pos in ai_trail:
                display_message(screen, "Game Over! You collided with the AI trail!")
                reward = 50 + steps_survived * 0.2
                game_end = True
            elif new_player_pos in player_trail:
                display_message(screen, "Game Over! You hit your own trail!")
                reward = 50 + steps_survived * 0.2
                game_end = True
            elif new_ai_pos in player_trail:
                display_message(screen, "You Win! AI hit your trail!")
                reward = -50
                game_end = True
            elif new_ai_pos in ai_trail:
                display_message(screen, "You Win! AI hit its own trail!")
                reward = -50
                game_end = True
            elif (new_ai_pos[0] < 0 or new_ai_pos[0] >= COLS or
                  new_ai_pos[1] < 0 or new_ai_pos[1] >= ROWS):
                display_message(screen, "You Win! AI hit the wall!")
                reward = -50
                game_end = True
            elif new_player_pos == new_ai_pos:
                display_message(screen, "Draw! Head-on collision!")
                reward = 0
                game_end = True

            replay_buffer.push(current_state, ai_dir, reward, next_state)
            if training_mode:
                prioritized_sweeping(current_state, ai_dir, reward, next_state)
                if steps_survived % 5 == 0:
                    updates = learn_from_experience()
                    if updates:
                        learning_updates += updates
                decay_epsilon()
            current_state = next_state
            if game_end:
                total_games += 1
                if total_games % 10 == 0:
                    print(f"Games played: {total_games}, Current epsilon: {EPSILON:.4f}, Learning updates: {learning_updates}")
                return

        # Draw trails and players
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
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Player: {len(player_trail)}  AI: {len(ai_trail)}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))
        mode_text = font.render(f"Mode: {'Training' if training_mode else 'Playing'}", True, (255, 255, 255))
        screen.blit(mode_text, (10, 50))
        diff_text = font.render(f"Difficulty: {difficulty_levels[current_difficulty]}", True, (255, 255, 255))
        screen.blit(diff_text, (10, 90))
        if training_mode:
            eps_text = font.render(f"Epsilon: {EPSILON:.4f}", True, (255, 255, 255))
            screen.blit(eps_text, (10, 130))
            qtable_text = font.render(f"States: {len(q_table)} | Updates: {learning_updates}", True, (255, 255, 255))
            screen.blit(qtable_text, (10, 170))
        pygame.display.flip()
        clock.tick(FPS)

def main():
    pygame.display.set_caption("Speeds7ers")
    player_color, ai_color = menu_screen()
    if player_color and ai_color:
        game_loop(player_color, ai_color)

if __name__ == "__main__":
    main()
