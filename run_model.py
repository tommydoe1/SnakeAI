import torch
import pygame
import sys
import os  # Import os
from snake_env import SnakeGameEnv
from dqn_model import DQN, Agent

def run_trained_model(model_file="best_snake_model2.pth"):
    # Initialize environment
    env = SnakeGameEnv()
    state_dim = len(env.get_state())
    action_dim = 4

    # Initialize agent with no exploration (epsilon=0)
    agent = Agent(state_dim, action_dim, gamma=0.99, epsilon_start=0.0, epsilon_end=0.0, epsilon_decay=0)

    # Load trained model
    if not model_file or not os.path.exists(model_file):
        print("Model file not found!")
        return

    checkpoint = torch.load(model_file)
    agent.model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from {model_file}")

    state = env.reset()
    done = False

    clock = pygame.time.Clock()

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        action = agent.select_action(state, train=False)  # Greedy action
        state, _, done = env.step(action)
        env.render()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    while True:
        run_trained_model()
        retry = input("Run again? (y/n): ").strip().lower()
        if retry != 'y':
            break

