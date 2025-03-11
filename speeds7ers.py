import pygame
import random
import numpy as np
import os

# Initialize Pygame
pygame.init()

# Game Constants
WIDTH, HEIGHT = 1500, 1500
GRID_SIZE = 10
COLS, ROWS = WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE
FPS = 45

# Colors
COLORS = {"Red": (255, 0, 0), "Blue": (0, 0, 255), "Green": (0, 255, 0), "Yellow": (255, 255, 0),
          "Purple": (128, 0, 128), "Cyan": (0, 255, 255), "White": (255, 255, 255), "Grey": (100, 100, 100)}

# Reinforcement Learning Parameters
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.1  # Exploration rate
Q_TABLE_FILE = "q_table.npy"

# Load or Initialize Q-Table
if os.path.exists(Q_TABLE_FILE):
    q_table = np.load(Q_TABLE_FILE, allow_pickle=True).item()
else:
    q_table = {}


def menu_screen():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.Font(None, 36)

    player_color = list(COLORS.keys())[0]
    ai_color = list(COLORS.keys())[1]
    colors_list = list(COLORS.keys())

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
    player_trail, ai_trail = [], []

    while running:
        screen.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            player_dir = (0, -1)
        elif keys[pygame.K_s]:
            player_dir = (0, 1)
        elif keys[pygame.K_a]:
            player_dir = (-1, 0)
        elif keys[pygame.K_d]:
            player_dir = (1, 0)

        player_pos.insert(0, (player_pos[0][0] + player_dir[0], player_pos[0][1] + player_dir[1]))
        ai_pos.insert(0, (ai_pos[0][0] + ai_dir[0], ai_pos[0][1] + ai_dir[1]))

        pygame.draw.rect(screen, COLORS[player_color],
                         (player_pos[0][0] * GRID_SIZE, player_pos[0][1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(screen, COLORS[ai_color],
                         (ai_pos[0][0] * GRID_SIZE, ai_pos[0][1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        pygame.display.flip()
        clock.tick(FPS)


def main():
    player_color, ai_color = menu_screen()
    if player_color and ai_color:
        game_loop(player_color, ai_color)


if __name__ == "__main__":
    main()
