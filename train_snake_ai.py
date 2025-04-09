import torch
import torch.optim as optim
import os
import random
import numpy as np
import pygame
from snake_env import SnakeGameEnv
from dqn_model import DQN, ReplayMemory, Agent
import datetime 

# Hyperparameters
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.2
epsilon_decay = 20000  # Increase decay period for more exploration
learning_rate = 0.001  # Slower learning rate for stability
memory_capacity = 10000
batch_size = 128
num_episodes = 10000
target_update_freq = 5  # Update target network every 5 episodes

# File to save best model
best_model_file = "best_snake_model.pth"
best_score = float('-inf')  # Initialize best score

# Initialize environment
env = SnakeGameEnv()
state_dim = len(env.get_state())  
action_dim = 4  

# Initialize agent and networks
agent = Agent(state_dim, action_dim, gamma=gamma, epsilon_start=epsilon_start,
              epsilon_end=epsilon_end, epsilon_decay=epsilon_decay, lr=learning_rate)

# Load best model if available
if os.path.exists(best_model_file):
    checkpoint = torch.load(best_model_file)
    agent.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    agent.epsilon = checkpoint["epsilon"]
    best_score = checkpoint["best_score"]
    print(f"Loaded model with best score: {best_score}")

# Function to save the best model
def save_best_model(agent, score):
    torch.save({
        "model_state_dict": agent.model.state_dict(),
        "epsilon": agent.epsilon,
        "best_score": score
    }, best_model_file)

# Initialize clock for controlling frame rate
clock = pygame.time.Clock()

# Track last 100 rewards for monitoring learning
last_100_rewards = []

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    snake_moves = 0  # Track moves for each episode

    while not done:
        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                print("Training terminated by user.")
                pygame.quit()
                exit()

        # Select action using the agent's policy
        action = agent.select_action(state)

        # Take action, observe new state and reward
        new_state, reward, done = env.step(action)

        # Store transition in replay memory
        agent.store_transition(state, action, reward, new_state, done)

        # Train the model only if enough experiences are stored
        if len(agent.memory) > batch_size * 10:  
            agent.learn(batch_size)

        # Move to next state
        state = new_state
        total_reward += reward
        snake_moves += 1  # Increment moves count

        # Render environment (for visualization only)
        render_flag = True  # Set to True if you want to visualize the agent's behavior
        if render_flag:
            env.render()
        clock.tick(30)  # Control frame rate only when rendering

    # Decay epsilon using exponential decay
    agent.epsilon = agent.epsilon_end + (epsilon_start - agent.epsilon_end) * np.exp(-1.0 * episode / epsilon_decay)

    # Store reward for monitoring progress
    last_100_rewards.append(total_reward)
    if len(last_100_rewards) > 100:
        last_100_rewards.pop(0)

    # Get the snake's score from the environment
    snake_score = env.score if hasattr(env, 'score') else 'N/A'  

    # Print episode performance with new details
    avg_reward = np.mean(last_100_rewards)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get current time
    print(f"Episode {episode}: Total Reward: {total_reward}, Avg (last 100): {avg_reward}, Moves: {snake_moves}, Score: {snake_score}, Time: {current_time}, Epsilon: {agent.epsilon:.4f}")

    # Save best model if new high score is reached
    if total_reward > best_score:
        best_score = total_reward
        save_best_model(agent, best_score)
        print(f"New best score! Model saved with score: {best_score}")

    # Update target network periodically
    if episode % target_update_freq == 0:
        agent.update_target_model()
        print("Updated target network.")

print("Training complete.")
pygame.quit()

