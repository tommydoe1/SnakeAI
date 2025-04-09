import pygame
import numpy as np
from snake_env import SnakeGameEnv

def play_and_record():
    env = SnakeGameEnv(record_data=True)
    state = env.reset()
    done = False
    action_map = {
        pygame.K_LEFT: 0,
        pygame.K_RIGHT: 1,
        pygame.K_UP: 2,
        pygame.K_DOWN: 3
    }

    last_action = None  # Keep track of the last action to maintain movement

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.save_gameplay_data()
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key in action_map:
                    last_action = action_map[event.key]
            elif event.type == pygame.KEYUP:
                # When key is released, stop the movement in that direction
                if event.key in action_map and last_action == action_map[event.key]:
                    last_action = None

        # If there's a valid action (key is being held down), continue moving in that direction
        if last_action is not None:
            state, _, done = env.step(last_action)

        # Render the environment to show the game state
        env.render()

    # Save the gameplay data after the game ends
    env.save_gameplay_data()

if __name__ == "__main__":
    play_and_record()
