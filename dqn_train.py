from typing import List, Dict, Tuple

import numpy as np
from dqn_agent import DQNAgent, WarehouseEnv  # Import the WarehouseEnv class
import logging
import datetime

presentDate = datetime.datetime.now()
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Define your warehouse grid and robots
grid_size = (10, 10)  # Define the grid size
obstacles = [
    (1, 1), (1, 8), (2, 3), (2, 6), (3, 4),
    (4, 2), (4, 7), (5, 5), (6, 3), (6, 8),
    (7, 2), (7, 9), (8, 1), (8, 5), (9, 4)
]

robots : List[Dict[str, int| Tuple[int]]]= [
    {'id': 1, 'start': (0, 0), 'end': (9, 9), 'priority': 1},  # Robot 1
    {'id': 2, 'start': (9, 0), 'end': (0, 9), 'priority': 2},   # Robot 2
    {'id': 3, 'start': (4, 9), 'end': (0, 4), 'priority': 3},  # Robot 2
]

# Hyperparameters
episodes : int = 10000 # Number of episodes to train
state_size : int = 4   # (robot_x, robot_y, task_x, task_y)
action_size : int = 4  # Move up, down, left, right

# Initialize the DQNAgent and WarehouseEnv
agent = DQNAgent(state_size, action_size)
env = WarehouseEnv(obstacles= obstacles, grid_size=grid_size,  num_robots= 2)

# # Training loop
# def train_fms_dqn(episodes):
#     for e in range(episodes):
#         # Register each robot in the environment
#         for robot in robots:
#             env.register_robot(robot['id'], robot['start'], robot['end'], robot['priority'])
#
#             # Reset environment and get initial state for this robot
#             state = env.reset(robot['id']) #FIXME
#             logging.info(f'Episode: {e}, Robot ID: {robot["id"]}, Initial State: {state}')
#
#             # print(f'Initial State for Robot {robot["id"]}: {state}')  # Debugging line
#             state = np.reshape(state, [1, state_size])  # Reshape the state correctly
#             total_reward = 0
#
#             for time in range(500):  # Limiting each episode to 500 steps
#                 # Agent takes an action for the robot
#                 action = agent.act(state)
#
#                 # Environment responds to the action of the robot
#                 next_state, reward, done, _ = env.step(robot['id'], action)
#                 next_state = np.reshape(next_state, [1, state_size])
#
#                 # Store experience and train
#                 agent.remember(state, action, reward, next_state, done)
#                 state = next_state
#                 total_reward += reward
#
#                 if done:
#                     logging.info(
#                         f"Episode {e}/{episodes} - Total Reward for Robot {robot['id']}: {total_reward} - Epsilon: {agent.epsilon}"
#                     )
#                     # print(f"Episode {e}/{episodes} - Total Reward for Robot {robot['id']}: {total_reward} - Epsilon: {agent.epsilon}")
#                     break
#
#         # Train the agent using the replay buffer
#         agent.replay()
#
#     # Save the trained model
#     agent.save("dqn_fms_model.pth")
#

def train_fms_dqn(episodes):
    for robot in robots:
        _ = env.register_robot(robot['id'], robot['start'], robot['end'], robot['priority'])
    for e in range(episodes):
        for robot in robots:
            # is_registered = env.register_robot(robot['id'], robot['start'], robot['end'], robot['priority'])
            # if not is_registered :
            #     continue
            env.reset(robot['id'])
            state = env.get_state(robot['id'])  # Get the initial state after resetting

            # Check if state size is correct
            if state.shape[0] != state_size:
                print(f"Warning: State size for robot {robot['id']} is {state.shape[0]}, expected {state_size}.")
                state = np.zeros((state_size,))  # Fallback to zero state

            state = np.reshape(state, [1, state_size])  # Reshape the state correctly
            total_reward = 0

            for time in range(500):
                action = agent.act(state)
                # print(f"ACTION = {action}")
                next_state, reward, done = env.step(robot['id'], action)
                next_state = np.reshape(next_state, [1, state_size]) if len(next_state) == state_size else np.zeros(
                    (1, state_size))

                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if done:
                    print(
                        f"Episode {e}/{episodes} - Total Reward for Robot {robot['id']}: {total_reward} - Epsilon: {agent.epsilon}")
                    break
            # env.reset()
        agent.replay()
    agent.save(f"dqn_fms_model_consec.pth")


# Run training
if __name__ == "__main__":
    train_fms_dqn(episodes)
