import pygame
import random
import numpy as np
import os

class SnakeGameEnv:
    def __init__(self, width=600, height=400, snake_block=10, record_data=False):
        pygame.init()
        pygame.font.init()

        self.width = width
        self.height = height
        self.snake_block = snake_block
        self.win = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.time_alive = 0

        self.font = pygame.font.SysFont("arial", 20)
        self.high_score_file = "high_score.txt"
        self.high_score = self.load_high_score()
        
        self.record_data = record_data
        self.gameplay_data = [] 

        self.reset()

    def load_high_score(self):
        if os.path.exists(self.high_score_file):
            with open(self.high_score_file, "r") as file:
                return int(file.read().strip())
        return 0

    def save_high_score(self):
        with open(self.high_score_file, "w") as file:
            file.write(str(self.high_score))

    def reset(self):
        self.snake_pos = [self.width // 2, self.height // 2]
        self.snake_body = [self.snake_pos[:]]
        self.food_pos = self.get_random_food_position()
        self.direction = "RIGHT"  # Default direction
        self.score = 0
        self.done = False
        self.time_alive = 0
        return self.get_state()

    def get_random_food_position(self):
        while True:
            food_pos = [
                random.randrange(0, self.width, self.snake_block),
                random.randrange(0, self.height, self.snake_block)
            ]
            if food_pos not in self.snake_body:
                return food_pos

    def step(self, action):
        self.time_alive += 1

        # Handle direction change
        directions = ["LEFT", "RIGHT", "UP", "DOWN"]
        if action < len(directions):
            # Prevent the snake from going directly opposite to the current direction
            if not ((self.direction == "LEFT" and directions[action] == "RIGHT") or
                    (self.direction == "RIGHT" and directions[action] == "LEFT") or
                    (self.direction == "UP" and directions[action] == "DOWN") or
                    (self.direction == "DOWN" and directions[action] == "UP")):
                self.direction = directions[action]

        # Move the snake based on current direction
        move = {"LEFT": (-self.snake_block, 0), "RIGHT": (self.snake_block, 0), "UP": (0, -self.snake_block), "DOWN": (0, self.snake_block)}
        dx, dy = move[self.direction]
        self.snake_pos[0] += dx
        self.snake_pos[1] += dy

        reward = -0.01  # Small penalty per move to prevent endless loops

        prev_dist = np.linalg.norm(np.array(self.snake_body[-1]) - np.array(self.food_pos))
        new_dist = np.linalg.norm(np.array(self.snake_pos) - np.array(self.food_pos))
        reward += 1 if new_dist < prev_dist else -0.5  # Encourage getting closer to food

        if self.snake_pos == self.food_pos:
            reward = 20 + 5 * self.score
            self.food_pos = self.get_random_food_position()
            self.snake_body.append(self.snake_pos[:])
            self.score += 1
            if self.score > self.high_score:
                self.high_score = self.score
                self.save_high_score()
                reward += 50
        else:
            self.snake_body.append(self.snake_pos[:])
            self.snake_body.pop(0)

        # Check if snake hits the wall or itself
        if (self.snake_pos[0] < 0 or self.snake_pos[0] >= self.width or
            self.snake_pos[1] < 0 or self.snake_pos[1] >= self.height or
            self.snake_pos in self.snake_body[:-1]):
            self.done = True
            reward = -100

        # Save the state, action, and reward in the gameplay data
        if self.record_data:
            self.gameplay_data.append((self.get_state(), action, reward))

        return self.get_state(), reward, self.done

    # def get_state(self):
    #     head = np.array(self.snake_pos) / [self.width, self.height]
    #     food = (np.array(self.food_pos) - np.array(self.snake_pos)) / [self.width, self.height]

    #     walls = [
    #         self.snake_pos[0] / self.width,
    #         (self.width - self.snake_pos[0]) / self.width,
    #         self.snake_pos[1] / self.height,
    #         (self.height - self.snake_pos[1]) / self.height
    #     ]

    #     body_representation = [head]
    #     for i in range(1, 4):
    #         if i < len(self.snake_body):
    #             body_representation.append(np.array(self.snake_body[i]) / [self.width, self.height])
    #         else:
    #             body_representation.append(np.zeros(2))

    #     tail = np.array(self.snake_body[-1]) / [self.width, self.height] if len(self.snake_body) > 3 else np.zeros(2)
    #     body_representation.append(tail)

    #     direction = [
    #         int(self.direction == "LEFT"),
    #         int(self.direction == "RIGHT"),
    #         int(self.direction == "UP"),
    #         int(self.direction == "DOWN")
    #     ]

    #     return np.concatenate((head, food, walls, np.concatenate(body_representation), direction))

    def get_state(self):
        head = np.array(self.snake_pos) / [self.width, self.height]
        food = (np.array(self.food_pos) - np.array(self.snake_pos)) / [self.width, self.height]

        walls = [
            self.snake_pos[0] / self.width,
            (self.width - self.snake_pos[0]) / self.width,
            self.snake_pos[1] / self.height,
            (self.height - self.snake_pos[1]) / self.height
        ]

        body_representation = [head]
        
        # Limit body size to the last 3 segments (or pad with zeros if less)
        for i in range(1, 4):
            if i < len(self.snake_body):
                body_representation.append(np.array(self.snake_body[i]) / [self.width, self.height])
            else:
                body_representation.append(np.zeros(2))  # Padding with zero for missing body segments

        tail = np.array(self.snake_body[-1]) / [self.width, self.height] if len(self.snake_body) > 3 else np.zeros(2)
        body_representation.append(tail)

        direction = [
            int(self.direction == "LEFT"),
            int(self.direction == "RIGHT"),
            int(self.direction == "UP"),
            int(self.direction == "DOWN")
        ]

        return np.concatenate((head, food, walls, np.concatenate(body_representation), direction))

    def render(self):
        self.win.fill((30, 30, 30))
        pygame.draw.rect(self.win, (0, 255, 0), [*self.food_pos, self.snake_block, self.snake_block])
        for segment in self.snake_body:
            pygame.draw.rect(self.win, (255, 0, 0), [*segment, self.snake_block, self.snake_block])
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        high_score_text = self.font.render(f"High Score: {self.high_score}", True, (255, 255, 255))
        self.win.blit(score_text, (10, 10))
        self.win.blit(high_score_text, (10, 30))
        pygame.display.update()
        self.clock.tick(30)

    def save_gameplay_data(self, filename="human_play_data.npy"):
        if self.record_data and self.gameplay_data:
            # Unzip the gameplay data into states, actions, rewards
            states, actions, rewards = zip(*self.gameplay_data)
            
            # Convert to numpy arrays
            states = np.array(states)  # Ensure this has consistent shape
            actions = np.array(actions)
            rewards = np.array(rewards)

            # Make sure actions and rewards are 2D (if necessary)
            actions = actions.reshape(-1, 1)  # Make sure actions are a column vector
            rewards = rewards.reshape(-1, 1)  # Make sure rewards are a column vector

            # Ensure all arrays have the same shape before stacking
            if states.shape[0] == actions.shape[0] == rewards.shape[0]:
                # Stack them into a single array (or save them separately)
                gameplay_data = np.hstack((states, actions, rewards))  # Stack horizontally
                np.save(filename, gameplay_data)
                print(f"Gameplay data saved to {filename} with score {self.score}")
            else:
                print("Error: Mismatched shapes in gameplay data.")


