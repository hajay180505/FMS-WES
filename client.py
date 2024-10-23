import requests
import numpy as np
import argparse
import time

# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Robot Client')
    parser.add_argument('--id', type=str, required=True, help='Robot ID')
    parser.add_argument('--start', type=int, nargs=2, required=True, help='Start coordinates (x y)')
    parser.add_argument('--end', type=int, nargs=2, required=True, help='End coordinates (x y)')
    parser.add_argument('--priority', type=int, required=True, help='Priority of the robot')
    parser.add_argument('--server', type=str, default='http://localhost:5000', help='Server URL')

    return parser.parse_args()

# Function to register the robot with the server
def register_robot(robot_id, start, end, priority):
    # Register robot on the server
    data = {
        'id': robot_id,
        'start': start,
        'end': end,
        'priority': priority
    }

    try:
        response = requests.post('http://127.0.0.1:5000/register_robot', json=data, timeout=5)
        response.raise_for_status()  # Raise an error for bad responses
        print(f"Robot {robot_id} registered successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to register robot {robot_id}: {e}")


# Function to get action from the server
def get_next_action(robot_id, state):
    response = requests.post('http://127.0.0.1:5000/get_action', json={'id': robot_id, 'state': state.tolist()})
    if response.status_code == 200:
        return response.json()['action']
    else:
        print(f"Failed to get action for robot {robot_id}.")
        return None

# Main function
def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Example robot state (start_x, start_y, end_x, end_y)
    state = np.array([args.start[0], args.start[1], args.end[0], args.end[1]])

    print(f'Robot ID: {args.id}, Start: {args.start}, End: {args.end}, Priority: {args.priority}')

    # Register the robot with the server
    register_robot(args.id, args.start, args.end, args.priority)

    try :
        while True:
            # Get the next action from the server based on the current state
            action = get_next_action(args.id, state)
            if action is None:
                break  # Exit the loop if there was a failure in getting the action
            actions = {
                0: "Move right",
                1: "Move down",
                2: "Move left",
                3: "Move up"
            }

            print(f'Robot ID: {args.id} - Next action: {actions[action]}')

            # Update robot state based on the action
            if action == 0:  # Move right
                state[0] += 1
            elif action == 1:  # Move down
                state[1] += 1
            elif action == 2:  # Move left
                state[0] -= 1
            elif action == 3:  # Move up
                state[1] -= 1

        # Check if the robot has reached the destination
            if (state[0], state[1]) == tuple(args.end):
                print(f'Robot {args.id} has reached its destination.')
                break

            # Add a delay for readability
            time.sleep(1)

    except KeyboardInterrupt:
        print(f"Robot {args.id} is shutting down.")


if __name__ == '__main__':
    main()
