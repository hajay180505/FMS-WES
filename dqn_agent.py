import random
from typing import Tuple, Any, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from numpy import ndarray
from numpy._typing import _64Bit


class Rewards:
    # negative rewards
    HIT_WALLS_NEAR_TARGET = -300
    HIT_WALLS_AWAY_TARGET = -450
    HITS_ROBOT = -500
    MOVING_AWAY_TARGET = -100
    ITERATION_PENALTY = -1

    # positive rewards
    TARGET_REACHED = 900
    MOVING_TOWARDS_TARGET = 450
#   TARGET_REACHED_FEW_STEPS = 1000


#Hyperparameters
class Config:
    GAMMA : float = 0.95
    EPSILON : float = 1.0
    EPSILON_MIN : float = 0.01
    EPSILON_DECAY : float = 0.995
    LEARNING_RATE : float = 0.001
    BATCH_SIZE : int =  64
    MEMORY_SIZE :int = 10000

    DISTANCE_THRESHOLD = 20


# Q-Network architecture
class DQN(nn.Module):
    def __init__(self, state_size : int , action_size : int) -> None :
        super(DQN, self).__init__()
        self.fc1 : nn.Linear = nn.Linear(state_size, 24)
        self.fc2 : nn.Linear = nn.Linear(24, 24)
        self.fc3 : nn.Linear = nn.Linear(24, action_size)

    def forward(self, state) -> nn.Linear :
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size : int, action_size : int) -> None:
        self.state_size : int= state_size
        self.action_size : int = action_size
        self.memory : deque = deque(maxlen=Config.MEMORY_SIZE)
        self.epsilon : float = Config.EPSILON
        self.gamma : float = Config.GAMMA
        self.epsilon_min : float = Config.EPSILON_MIN
        self.epsilon_decay : float = Config.EPSILON_DECAY
        self.learning_rate : float = Config.LEARNING_RATE
        self.model : DQN = DQN(state_size, action_size)
        self.optimizer :optim.Adam = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    # Remember experiences
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Choose action using epsilon-greedy policy
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Random action for exploration

        # Forward pass through the network to get Q-values for the given state
        state = torch.FloatTensor(state).unsqueeze(0)  # Shape [1, state_size]
        act_values = self.model(state)  # Shape [1, action_size]

        # Debugging: Check shape of act_values
        # print(f"act_values shape: {act_values.shape}")

        # Get the index of the action with the highest Q-value
        return torch.argmax(act_values).item()  # Returns scalar action index

    def replay(self):
        if len(self.memory) < Config.BATCH_SIZE:
            return

        minibatch = random.sample(self.memory, Config.BATCH_SIZE)
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

# Warehouse environment
class WarehouseEnv:
    def __init__(self,  obstacles , grid_size=(10, 10), num_robots=1):
        self.grid_size :Tuple[int] = grid_size
        self.num_robots :int = num_robots
        self.grid : np.ndarray[Any, np.dtype[np.floating[_64Bit]]]= np.zeros(self.grid_size)
        self.robot_positions : Dict[int, Dict[str, Any] ] = {}
        self.tasks = {}  # Dictionary to hold tasks for each robot
        self.obstacles : List[Tuple[int]] = obstacles if obstacles is not None else []  # Store obstacles as a list

        # Place obstacles in the grid
        for obs in self.obstacles:
            self.grid[obs[0], obs[1]] = -1

    def register_robot(self, robot_id : int , start : Tuple[int], end : Tuple[int], priority : int):
        """Register a robot in the environment with its start and end positions."""
        if robot_id in self.robot_positions:
            print(f"Robot {robot_id} is already registered.")
            return False  # Robot is already registered, so skip

        for other_robot_id, robot_data in self.robot_positions.items():
            if (other_robot_id!= robot_id) and tuple(robot_data['current_position']) == start:
                print(
                    f"Error: Cannot register robot {robot_id}. Another robot {other_robot_id} is already at position {start}.")
                return False

        self.robot_positions[robot_id] = {
            'start': start,  # Store the start position
            'current_position': list(start),  # Ensure current_position is a list
            'end_position': end,
            'priority': priority
        }
        print(f"Robot {robot_id} registered at position {start}, heading to {end}.")
        return True

    def is_collision(self,calling_robot_id, new_position):
        """Check if the new position causes a collision with another robot."""
        for robot_id, robot_data in self.robot_positions.items():
            if (calling_robot_id != robot_id) and tuple(robot_data['current_position']) == tuple(new_position):
                return True, f"Other robot {robot_id} at {new_position} "

        if self.grid[new_position[0], new_position[1]] == -1:
            return True, "Hit walls"

        return False,""

    def step_og(self, robot_id : int, action) -> tuple[list[int], int, bool ]:
        """Takes an action and moves the robot in the environment."""
        current_position : List[int] = list(self.robot_positions[robot_id]['current_position'])
        end_position : Tuple[int] = self.robot_positions[robot_id]['end_position']

        # Determine the new position based on the action
        new_position = current_position.copy()  # Copy current position to avoid modifying it
        if action == 0:  # Move right
            new_position[0] += 1
        elif action == 1:  # Move down
            new_position[1] += 1
        elif action == 2:  # Move left
            new_position[0] -= 1
        elif action == 3:  # Move up
            new_position[1] -= 1

        # Boundary checks
        new_position[0] = np.clip(new_position[0], 0, self.grid_size[0] - 1)
        new_position[1] = np.clip(new_position[1], 0, self.grid_size[1] - 1)

        # Check for collisions
        is_collided, cause = self.is_collision(robot_id, new_position)
        if is_collided:
            if cause == "Hit walls":
                reward = -250
            else:
                reward = -500 # Severe penalty for collision
            done = True  # End the episode if a collision occurs
            # print(f"Collision detected for robot {robot_id} at position {new_position} due to {cause}.")
            return current_position, reward, done

        # Move the robot to the new position
        self.robot_positions[robot_id]['current_position'] = new_position
        self.render()
        done = (new_position == list(end_position))  # Check if the robot reached the end

        # Reward for reaching the goal
        if done:
            reward : int = 100
            print(f"Robot {robot_id} has reached its destination.")
        else:
            reward = -1  # Small penalty for each step

        return new_position, reward, done

    def step(self, robot_id: int, action) -> tuple[list[int], int, bool]:
        """Takes an action and moves the robot in the environment."""
        current_position: List[int] = list(self.robot_positions[robot_id]['current_position'])
        end_position: Tuple[int] = self.robot_positions[robot_id]['end_position']

        # Determine the new position based on the action
        new_position = current_position.copy()  # Copy current position to avoid modifying it
        if action == 0:  # Move right
            new_position[0] += 1
        elif action == 1:  # Move down
            new_position[1] += 1
        elif action == 2:  # Move left
            new_position[0] -= 1
        elif action == 3:  # Move up
            new_position[1] -= 1

        # Boundary checks
        new_position[0] = np.clip(new_position[0], 0, self.grid_size[0] - 1)
        new_position[1] = np.clip(new_position[1], 0, self.grid_size[1] - 1)

        # Check for collisions
        is_collided, cause = self.is_collision(robot_id, new_position)
        if is_collided:
            if cause == "Hit walls":
                # Calculate distance to the goal
                current_distance = abs(current_position[0] - end_position[0]) + abs(
                    current_position[1] - end_position[1])
                new_distance = abs(new_position[0] - end_position[0]) + abs(new_position[1] - end_position[1])

                if abs(current_distance - new_distance) < Config.DISTANCE_THRESHOLD:
                    reward = Rewards.HIT_WALLS_NEAR_TARGET  # Penalty for hitting a wall
                else:
                    reward = Rewards.HIT_WALLS_AWAY_TARGET  # Penalty for hitting a wall

            else:
                reward = Rewards.HITS_ROBOT # Severe penalty for colliding with other robots/obstacles
            done = True  # End the episode if a collision occurs
            # print(f"Collision detected for robot {robot_id} at position {new_position} due to {cause}.")
            return current_position, reward, done

        # Calculate distance to the goal
        current_distance = abs(current_position[0] - end_position[0]) + abs(current_position[1] - end_position[1])
        new_distance = abs(new_position[0] - end_position[0]) + abs(new_position[1] - end_position[1])

        # Move the robot to the new position
        self.robot_positions[robot_id]['current_position'] = new_position
        self.render()

        # Check if the robot reached its destination
        done = (new_position == list(end_position))
        if done:
            reward = Rewards.TARGET_REACHED  # Reward for reaching the goal
            print(f"Robot {robot_id} has reached its destination.")
        else:
            # Reward for getting closer, penalty for moving away
            if new_distance < current_distance:
                reward = Rewards.MOVING_TOWARDS_TARGET  # Reward for reducing the distance to the goal
            elif new_distance > current_distance:
                reward = Rewards.MOVING_AWAY_TARGET  # Penalty for moving farther from the goal
            else:
                reward = Rewards.ITERATION_PENALTY  # Small step penalty

        return new_position, reward, done

    # def reset(self):
    #     """Reset the environment for all robots."""
    #     self.robot_positions = {}
    #     self.grid = np.zeros(self.grid_size)
    #     print("Warehouse environment has been reset.")

    def reset(self, robot_id : int = None) -> None:
        """Reset the environment or a specific robot."""
        if robot_id is None:
            # Reset the entire environment
            self.robot_positions = {}
            self.grid = np.zeros(self.grid_size)
            # print("Warehouse environment has been reset.")
        else:
            # Reset the specific robot
            if robot_id in self.robot_positions:
                self.robot_positions[robot_id]['current_position'] = self.robot_positions[robot_id]['start']
                # print(f"Robot {robot_id} has been reset to its start position: {self.robot_positions[robot_id]['start']}.")
            else:
                print(f"Robot {robot_id} not found.")

    def render(self, display : bool = False) ->None:
        """Render the grid with robot positions and obstacles."""
        self.grid.fill(0)  # Clear the grid

        # Mark obstacles on the grid with -1
        for obs in self.obstacles:
            self.grid[obs[0], obs[1]] = -1

        # Mark robot positions on the grid with 1
        for robot_id, robot_data in self.robot_positions.items():
            pos = robot_data['current_position']
            self.grid[pos[0], pos[1]] = robot_id  # Mark robot's current position on the grid

        # Display the grid
        if display :
            print("Grid state:")
            print(self.grid)

    def get_state(self, robot_id) -> ndarray :
        """Return the current state of the robot for DQN."""
        if robot_id in self.robot_positions:
            current_position = list(self.robot_positions[robot_id]['current_position'])
            end_position = list(self.robot_positions[robot_id]['end_position'])  # Convert to tuple
            state : np.ndarray = np.array(current_position + list(end_position))  # Concatenate positions
            # print(f"State for robot {robot_id}: {state}")  # Debugging line
            return state  # Ensure this returns a flat array of the correct size
        else:
            print(f"Robot {robot_id} not found.")
            return np.zeros((4,))  # Return a default state of size 4 if robot not found
