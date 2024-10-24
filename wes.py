import csv
import heapq
import subprocess

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from concurrent.futures import ThreadPoolExecutor

DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

def astar(start, goal, grid):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    if grid[start[0]][start[1]] == 1 or grid[goal[0]][goal[1]] == 1:
        return None
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    closed_set = set()
    while open_set:
        current = heapq.heappop(open_set)[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
        closed_set.add(current)
        for direction in DIRECTIONS:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            if (0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0])
                    and grid[neighbor[0]][neighbor[1]] == 0):
                if neighbor in closed_set:
                    continue
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None

def calculate_expected_time(start, end, grid):
    path = astar(start, end, grid)
    if path is None:
        return float('inf')
    return len(path)

def reassign_priorities(tasks):
    distances = [(i, np.linalg.norm(np.array(task['end']) - np.array(task['start']))) for i, task in enumerate(tasks)]
    distances.sort(key=lambda x: x[1])
    for idx, (i, dist) in enumerate(distances):
        tasks[i]['priority'] = idx + 1
    return tasks

def optimize_robot_assignment(tasks, robots, expected_times):
    model = gp.Model()
    x = model.addVars(len(tasks), len(robots), vtype=GRB.BINARY, name="x")
    valid_tasks = [i for i in range(len(tasks)) if expected_times[i] != float('inf')]
    if len(valid_tasks) == 0:
        print("No valid tasks available for optimization.")
        return []
    objective = gp.quicksum(x[i, j] * expected_times[i] / tasks[i]['priority']
                            for i in valid_tasks for j in range(len(robots)))
    model.setObjective(objective, GRB.MINIMIZE)
    for i in valid_tasks:
        model.addConstr(gp.quicksum(x[i, j] for j in range(len(robots))) == 1)
    model.optimize()
    assignment = []
    for i in valid_tasks:
        for j in range(len(robots)):
            if x[i, j].x > 0.5:
                assignment.append((tasks[i], robots[j]))
    return assignment

def read_tasks_from_csv(tasks_csv):
    tasks = []
    with open(tasks_csv, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            task = {
                'id': int(row[0]),
                'start': (int(row[1]), int(row[2])),
                'end': (int(row[3]), int(row[4])),
                'priority': int(row[5])
            }
            tasks.append(task)
    return tasks

def read_robots_from_csv(robots_csv):
    robots = []
    with open(robots_csv, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            robot = {
                'id': int(row[0]),
                'start': (int(row[1]), int(row[2]))
            }
            robots.append(robot)
    return robots

def read_grid_from_csv(grid_csv):
    grid = []
    with open(grid_csv, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            grid.append([int(cell) for cell in row])
    return grid

def main():

    tasks = read_tasks_from_csv('tasks.csv')
    robots = read_robots_from_csv('robots.csv')
    grid = read_grid_from_csv('grid.csv')
    tasks = reassign_priorities(tasks)
    expected_times = [calculate_expected_time(task['start'], task['end'], grid) for task in tasks]
    robot_assignment = optimize_robot_assignment(tasks, robots, expected_times)
    
    print("\nTask Assignments:")
    for task, robot in robot_assignment:
        print(f"Task {task['id']} assigned to Robot {robot['id']}")
        print(f"  Start: {task['start']}, End: {task['end']}, Priority: {task['priority']}\n")

        # Prepare arguments for client.py
        client_args = [
            'python', 'client_no_gui.py',
            '--id', str(task['id']),
            '--start', str(task['start'][0]), str(task['start'][1]),
            '--end', str(task['end'][0]), str(task['end'][1]),
            '--priority', str(task['priority'])
        ]

        # Run client.py with the task details
        subprocess.run(client_args)

if __name__ == "__main__":
    main()
