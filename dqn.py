import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Hyperparameters
GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Represents right, down, left, and up

# Q-Network architecture
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN Agent for managing robots
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.gamma = GAMMA
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    # Remember experiences
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Choose action using epsilon-greedy policy
    # def act(self, state):
    #     if np.random.rand() <= self.epsilon:
    #         return random.randrange(self.action_size)
    #     state = torch.FloatTensor(state).unsqueeze(0)
    #     act_values = self.model(state)
    #     return torch.argmax(act_values, dim=1).item()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0)  # Ensure state has shape (1, state_size)
        act_values = self.model(state)  # Output shape: (1, action_size)
        return torch.argmax(act_values).item()  # No need to specify dim, as it is a 1D tensor


    # Train the DQN network
    # def replay(self):
    #     if len(self.memory) < BATCH_SIZE:
    #         return
        
    #     minibatch = random.sample(self.memory, BATCH_SIZE)
    #     for state, action, reward, next_state, done in minibatch:
    #         state = torch.FloatTensor(state).unsqueeze(0)
    #         next_state = torch.FloatTensor(next_state).unsqueeze(0)
    #         target = reward
    #         if not done:
    #             target = reward + self.gamma * torch.max(self.model(next_state)).item()
    #         target_f = self.model(state)
    #         target_f[0, action] = target
    #         self.optimizer.zero_grad()
    #         loss = self.criterion(target_f, self.model(state))
    #         loss.backward()
    #         self.optimizer.step()
        
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay

    # def replay(self):
    #     if len(self.memory) < BATCH_SIZE:
    #         return
        
    #     minibatch = random.sample(self.memory, BATCH_SIZE)
    #     for state, action, reward, next_state, done in minibatch:
    #         state = torch.FloatTensor(state).unsqueeze(0)
    #         next_state = torch.FloatTensor(next_state).unsqueeze(0)
    #         target = reward
    #         if not done:
    #             target = reward + self.gamma * torch.max(self.model(next_state)).item()
    #         target_f = self.model(state)

    #         # Ensure the target_f tensor has the correct shape (1, action_size)
    #         target_f = target_f.squeeze(0)  # Remove extra batch dimension
    #         target_f[action] = target  # Update the Q-value for the chosen action
            
    #         self.optimizer.zero_grad()
    #         loss = self.criterion(target_f.unsqueeze(0), self.model(state))  # Reshape back to (1, action_size)
    #         loss.backward()
    #         self.optimizer.step()

    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            # Reshape state and next_state to ensure they are of shape (state_size,)
            state = np.array(state).reshape(-1)  # Flatten any extra dimensions
            next_state = np.array(next_state).reshape(-1)

            state = torch.FloatTensor(state)  # Shape: (state_size,)
            next_state = torch.FloatTensor(next_state)  # Shape: (state_size,)

            # Reshape to (1, state_size) before feeding to the model
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)

            # print(f"State shape: {state.shape}, Next state shape: {next_state.shape}")

            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state)).item()

            # Get the predicted Q-values for the current state
            target_f = self.model(state)  # Shape: (1, action_size)
            # print(f"target_f shape before squeeze: {target_f.shape}")

            # Ensure target_f is of shape (action_size,)
            target_f = target_f.squeeze(0)  # Shape: (action_size,)
            # print(f"target_f shape after squeeze: {target_f.shape}")

            # Make sure the action is valid
            if action >= self.action_size or action < 0:
                print(f"Error: action {action} is out of bounds for action_size {self.action_size}")
                continue

            # Update the Q-value for the chosen action
            target_f[action] = target  # Update the Q-value for the chosen action

            # Train the model
            self.optimizer.zero_grad()
            loss = self.criterion(target_f.unsqueeze(0), self.model(state))  # Reshape back to (1, action_size)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay






    # Load and save the model
    def load(self, name):
        self.model.load_state_dict(torch.load(name))
    
    def save(self, name):
        torch.save(self.model.state_dict(), name)

# Define the environment: Warehouse grid, robots, tasks
class WarehouseEnv_:
    def __init__(self, grid, robots, tasks):
        self.grid = grid
        self.robots = robots
        self.tasks = tasks
        self.current_step = 0
        self.done = False
        self.reward = 0

    def reset(self):
        # Reset the environment for a new episode
        self.current_step = 0
        self.done = False
        self.reward = 0
        # Reset robot and task positions
        # Return initial state (robot position, task position)
        return self.get_state()

    def get_state(self):
        # Combine robot and task positions into a state representation
        # Add more features as needed (other robots' positions, obstacles, etc.)
        return np.array([self.robots[0]['start'][0], self.robots[0]['start'][1], self.tasks[0]['end'][0], self.tasks[0]['end'][1]])

    # def step(self, action):
    #     # Perform the action (move robot), update state, calculate reward
    #     next_state = self.get_state()
    #     self.reward = -1  # default reward for a time step (penalizing longer paths)
        
    #     # Check if the robot reached the task
    #     if (next_state[:2] == next_state[2:]).all():
    #         self.done = True
    #         self.reward = 100  # large reward for completing the task
        
    #     return next_state, self.reward, self.done, {}

    def step(self, action):
    # Example: Move robot based on the action (up, down, left, right)
        direction = DIRECTIONS[action]
        self.robots[0]['start'] = (self.robots[0]['start'][0] + direction[0], self.robots[0]['start'][1] + direction[1])

        # Check if the robot moves out of bounds or hits an obstacle
        x, y = self.robots[0]['start']
        if x < 0 or y < 0 or x >= len(self.grid) or y >= len(self.grid[0]) or self.grid[x][y] == 1:
            self.done = True
            self.reward = -500  # Large penalty for invalid move (out of bounds or into obstacle)
            return self.get_state(), self.reward, self.done, {}

        next_state = self.get_state()
        self.reward = -1  # Default time-step penalty

        # Check if the robot reached the task's end position
        if (next_state[:2] == next_state[2:]).all():
            self.done = True
            self.reward = 100  # Reward for completing the task

        return next_state, self.reward, self.done, {}

class WarehouseEnvMono:
    def __init__(self, grid, robots, tasks):
        self.grid = grid
        self.robots = robots
        self.tasks = tasks
        self.current_step = 0
        self.done = False
        self.reward = 0

    def reset(self):
        # Reset the environment for a new episode
        self.current_step = 0
        self.done = False
        self.reward = 0
        # Reset robot and task positions
        return self.get_state()

    def get_state(self):
        # Combine robot and task positions into a state representation
        # Add more features as needed (other robots' positions, obstacles, etc.)
        return np.array([self.robots[0]['start'][0], self.robots[0]['start'][1], self.tasks[0]['end'][0], self.tasks[0]['end'][1]])

    def step(self, action):
        robot_pos = list(self.robots[0]['start'])  # Current position of the robot

        # Define movement directions (right, down, left, up)
        move = DIRECTIONS[action]

        # Apply the movement (check for boundary and grid validity)
        new_x = robot_pos[0] + move[0]
        new_y = robot_pos[1] + move[1]

        # Check boundaries and obstacles
        if new_x < 0 or new_x >= len(self.grid) or new_y < 0 or new_y >= len(self.grid[0]):
            self.reward = -100  # Penalize for moving outside the grid
        elif self.grid[new_x][new_y] == 1:  # Check for obstacles
            self.reward = -100  # Penalize for hitting an obstacle
        else:
            # Valid move, update the robot's position
            robot_pos = [new_x, new_y]
            self.robots[0]['start'] = tuple(robot_pos)
            self.reward = -1  # Default reward for a valid move

        # Check if the robot has reached the task
        if robot_pos == list(self.tasks[0]['end']):
            self.done = True
            self.reward = 100  # Large reward for completing the task

        next_state = self.get_state()
        return next_state, self.reward, self.done, {}


class WarehouseEnv:
    def __init__(self, grid, robots, tasks):
        self.grid = grid
        self.robots = robots
        self.tasks = tasks
        self.current_step = 0
        self.done = [False] * len(robots)  # Track each robot's task completion
        self.reward = [0] * len(robots)    # Track reward for each robot

    def reset(self):
        # Reset for all robots
        self.current_step = 0
        self.done = [False] * len(self.robots)
        self.reward = [0] * len(self.robots)
        return self.get_state()

    def get_state(self):
        # Combine positions of all robots and their respective tasks into a state representation
        state = []
        for i, robot in enumerate(self.robots):
            state.extend([robot['start'][0], robot['start'][1], self.tasks[i]['end'][0], self.tasks[i]['end'][1]])
        return np.array(state)

    def step(self, robot_id, action):
        robot_pos = list(self.robots[robot_id]['start'])
        move = DIRECTIONS[action]
        new_x = robot_pos[0] + move[0]
        new_y = robot_pos[1] + move[1]

        # Check boundaries and obstacles for this robot
        if new_x < 0 or new_x >= len(self.grid) or new_y < 0 or new_y >= len(self.grid[0]):
            self.reward[robot_id] = -100  # Penalize for moving outside the grid
        elif self.grid[new_x][new_y] == 1:
            self.reward[robot_id] = -100  # Penalize for hitting an obstacle
        else:
            self.robots[robot_id]['start'] = (new_x, new_y)  # Valid move
            self.reward[robot_id] = -1  # Default reward for valid move

        # Check if robot reached its task
        if self.robots[robot_id]['start'] == self.tasks[robot_id]['end']:
            self.done[robot_id] = True
            self.reward[robot_id] = 100  # Large reward for completing task

        next_state = self.get_state()
        return next_state, self.reward[robot_id], self.done[robot_id], {}

# Main function for training the DQN agent in the warehouse environment
# def train_fms_dqn(grid, robots, tasks, episodes=1000):
#     state_size = 4  # Example: robot's (x, y) and task's (x, y)
#     action_size = len(DIRECTIONS)  # up, down, left, right
#     agent = DQNAgent(state_size, action_size)
#     env = WarehouseEnv(grid, robots, tasks)

#     total_rewards = [] 

#     for e in range(episodes):
#         print("Episode : ",e)
#         state = env.reset()
#         state = np.reshape(state, [1, state_size])

#         for time in range(500):  # Limiting each episode to 500 steps
#             action = agent.act(state)
#             next_state, reward, done, _ = env.step(action)
#             next_state = np.reshape(next_state, [1, state_size])
#             agent.remember(state, action, reward, next_state, done)
#             state = next_state

#             if done:
#                 print(f"Episode {e}/{episodes} - Time: {time} - Reward: {reward} - Epsilon: {agent.epsilon}")
#                 break

#         agent.replay()  # Train the agent after each episode

#     agent.save("dqn_fms_model.pth")




# def train_fms_dqn(grid, robots, tasks, episodes=1000):
#     state_size = 4  # Example: robot's (x, y) and task's (x, y)
#     action_size = len(DIRECTIONS)  # up, down, left, right
#     agent = DQNAgent(state_size, action_size)
#     env = WarehouseEnv(grid, robots, tasks)

#     total_rewards = []  # To keep track of rewards for each episode

#     for e in range(episodes):
#         state = env.reset()
#         state = np.reshape(state, [1, state_size])
#         total_reward = 0

#         for time in range(500):  # Limiting each episode to 500 steps
#             action = agent.act(state)
#             next_state, reward, done, _ = env.step(action)
#             next_state = np.reshape(next_state, [1, state_size])
#             agent.remember(state, action, reward, next_state, done)
#             state = next_state

#             total_reward += reward  # Accumulate reward for the episode

#             if done:
#                 break

#         # Store total reward for this episode
#         total_rewards.append(total_reward)

#         # Log progress
#         print(f"Episode {e + 1}/{episodes} - Total Reward: {total_reward} - Epsilon: {agent.epsilon}")

#         # Train the agent after each episode
#         agent.replay()  

#     # Save the trained model
#     agent.save("dqn_fms_model.pth")

#     # Optional: Log average reward over last 10 episodes
#     if len(total_rewards) > 10:
#         avg_reward = np.mean(total_rewards[-10:])
#         print(f"Average Reward over last 10 episodes: {avg_reward}")



def train_fms_dqn(grid, robots, tasks, episodes=1000):
    # state_size = 4  # Robot's (x, y) and task's (x, y)
    state_size = len(robots) * 4  # 4 for each robot's (x, y) and task's (x, y)

    action_size = len(DIRECTIONS)  # up, down, left, right
    agent = DQNAgent(state_size, action_size)
    env = WarehouseEnv(grid, robots, tasks)

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        
        total_reward = 0  # Track total reward per episode

        for time in range(500):  # Limiting each episode to 500 steps
            action = agent.act(state)

            # Log state, action and reward
            print(f"Episode {e}, Step {time}, State: {state}, Action: {action}")
            
            # Take action in environment
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            # Add to memory and update state
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Log rewards and progress
            print(f"Episode {e}, Step {time}, Reward: {reward}, Next State: {next_state}, Done: {done}")
            
            if done:
                print(f"Episode {e}/{episodes} - Total Reward: {total_reward} - Epsilon: {agent.epsilon}")
                break

        # Train the agent after each episode
        agent.replay()

    # Save the model after training
    agent.save("dqn_fms_model.pth")







# Example usage
# grid = [
#     [0, 0, 0, 0], 
#     [0, 1, 0, 1], 
#     [0, 0, 0, 0], 
#     [1, 0, 1, 0]]  # 0: free, 1: obstacle
# robots = [
#     {'id': 1, 
#      'start': (0, 0)}
#      ]
# tasks = [
#     {'id': 1, 
#      'start': (0, 0), 
#      'end': (3, 3), 
#      'priority': 1}
#      ]

# train_fms_dqn(grid, robots, tasks, episodes=100)



# grid = [
#     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 0: free space, 1: obstacle
#     [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],  # A mix of obstacles and open paths
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
#     [0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
#     [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [0, 1, 0, 1, 0, 1, 0, 0, 0, 0]
# ]

# robots = [
#     {'id': 1, 'start': (0, 0)}  # Robot starts at top-left corner
# ]

# tasks = [
#     {'id': 1, 'start': (0, 0), 'end': (9, 9), 'priority': 1}  # Task at bottom-right corner
# ]

# train_fms_dqn(grid, robots, tasks, episodes=1000)

grid = [
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 0: free space, 1: obstacle
    [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],  # A mix of obstacles and open paths
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 1, 0, 1, 0, 1, 0, 0, 0, 0]
]

robots = [
    {'id': 1, 'start': (0, 0)},  # Robot 1 starts at top-left corner
    {'id': 2, 'start': (9, 0)},  # Robot 2 starts at bottom-left corner
    {'id': 3, 'start': (0, 9)}   # Robot 3 starts at top-right corner
]

tasks = [
    {'id': 1, 'start': (0, 0), 'end': (9, 9), 'priority': 1},  # Task at bottom-right corner
    {'id': 2, 'start': (9, 0), 'end': (0, 0), 'priority': 2},  # Task at top-left corner
    {'id': 3, 'start': (0, 9), 'end': (9, 0), 'priority': 3}   # Task at bottom-left corner
]

train_fms_dqn(grid, robots, tasks, episodes=100)

