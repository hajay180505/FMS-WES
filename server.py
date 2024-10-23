from flask import Flask, request, jsonify
import numpy as np
import threading
import queue  # Use priority queue for tasks
import time
from dqn_agent import DQNAgent  # Import your DQNAgent class
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


app = Flask(__name__)

# Global variables
robots = {}  # Dictionary to store robot states
grid_size = (10, 10)  # Example grid size
lock = threading.Lock()  # Lock for thread safety

# Task queue with priority (lower number means higher priority)
task_queue = queue.PriorityQueue()

# DQN agent setup
state_size = 4  # As per your DQN setup (x, y for robot, x, y for target)
action_size = 4  # Actions: up, down, left, right
agent = DQNAgent(state_size, action_size)  # Load trained model
agent.load("dqn_fms_model.pth")  # Load your trained model


@app.route('/register_robot', methods=['POST'])
def register_robot():
    data = request.json

    if 'id' not in data or 'start' not in data or 'end' not in data or 'priority' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    priority = data['priority']
    robot_id = data['id']
    start_coord = tuple(data['start'])
    end_coord = tuple(data['end'])

    if not (0 <= start_coord[0] < grid_size[0] and 0 <= start_coord[1] < grid_size[1]):
        return jsonify({'error': 'Start coordinates out of bounds'}), 400

    with lock:
        if robot_id in robots:
            return jsonify({'error': 'Robot already registered'}), 400  # Prevent duplicate registrations

        robots[robot_id] = {
            'start': start_coord,
            'end': end_coord,
            'priority': priority,
            'current_position': start_coord
        }

    return jsonify({'message': f'Robot {robot_id} registered successfully.'})


@app.route('/get_action', methods=['POST'])
def get_action():
    data = request.json
    robot_id = data['id']

    with lock:
        if robot_id not in robots:
            return jsonify({'error': 'Robot not registered'}), 404

        # Add the action request to the task queue with the robot's priority
        robot = robots[robot_id]
        task_queue.put((robot['priority'], robot_id))  # Priority queue stores tuples (priority, robot_id)

    return jsonify({'message': f'Action request for robot {robot_id} added to the queue.'})


def update_position(current_position, action):
    x, y = current_position
    if action == 0 and x + 1 < grid_size[0]:  # Move right
        return x + 1, y
    elif action == 1 and y + 1 < grid_size[1]:  # Move down
        return x, y + 1
    elif action == 2 and x - 1 >= 0:  # Move left
        return x - 1, y
    elif action == 3 and y - 1 >= 0:  # Move up
        return x, y - 1
    return current_position


def process_task_queue():
    """Thread function to process tasks from the queue based on priority."""
    while True:
        if not task_queue.empty():
            priority, robot_id = task_queue.get()

            with lock:
                if robot_id not in robots:
                    logging.info(f"Robot {robot_id} is no longer registered.")
                    continue

                robot = robots[robot_id]
                current_position = robot['current_position']
                goal_position = robot['end']

                # Prepare state: [robot_x, robot_y, goal_x, goal_y]
                state = np.array([current_position[0], current_position[1], goal_position[0], goal_position[1]])
                state = np.reshape(state, [1, state_size])

                # Get action from DQN model
                action = agent.act(state)

                # Update robot's position based on action
                new_position = update_position(current_position, action)
                robots[robot_id]['current_position'] = new_position

                logging.info(f"Processed robot {robot_id} with action {action}, new position: {new_position}")

        # Sleep for the time quantum (1 second)
        time.sleep(1)


if __name__ == '__main__':
    # Start task processing thread
    task_processing_thread = threading.Thread(target=process_task_queue, daemon=True)
    task_processing_thread.start()

    # Run Flask server
    app.run(host='0.0.0.0', port=5000, threaded=True)
