import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# Define the Deep Q-Network (DQN)
class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# Experience Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# DQN Agent
class Agent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=500, lr=0.001):  # Added 'lr' as a parameter
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        
        self.memory = ReplayMemory(capacity=10000)
        self.model = DQN(state_dim, hidden_dim, action_dim)
        self.target_model = DQN(state_dim, hidden_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)  # Use 'lr' instead of fixed value
        self.update_target_model()

    def update_target_model(self):
        """ Copies weights from the main model to the target model. """
        self.target_model.load_state_dict(self.model.state_dict())

    # def select_action(self, state, train=True):
    #     """ Uses epsilon-greedy policy to select an action. """
    #     if train:
    #         self.steps_done += 1
    #         epsilon = self.epsilon_end + (self.epsilon - self.epsilon_end) * np.exp(-self.steps_done / self.epsilon_decay)

    #         if random.random() < epsilon:
    #             return random.randint(0, self.action_dim - 1)
    #     else:
    #         epsilon = 0  # No randomness during inference

    #     state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    #     with torch.no_grad():
    #         return torch.argmax(self.model(state_tensor)).item()

    def select_action(self, state, train=True):
        if train:
            self.steps_done += 1
            # Exponential decay for better exploration-exploitation trade-off
            epsilon = self.epsilon_end + (self.epsilon - self.epsilon_end) * np.exp(-1.0 * self.steps_done / self.epsilon_decay)
        else:
            epsilon = 0  # No randomness during inference

        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                return torch.argmax(self.model(state_tensor)).item()


    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))

    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        transitions = self.memory.sample(batch_size)
        batch = list(zip(*transitions))
        states, actions, rewards, next_states, dones = batch

        states = torch.tensor(np.array(states), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

        q_values = self.model(states).gather(1, actions).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (~dones))

        loss = nn.SmoothL1Loss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
