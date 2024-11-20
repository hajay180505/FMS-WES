import streamlit as st
import numpy as np
import time
import heapq  # Import heapq for A* usage

# Icons for the grid
ROBOT_ICON = "🤖"
GOAL_ICON = "🎯"
OBSTACLE_ICON = "🟥"
EMPTY_ICON = "⬜"

# Grid size (Adjustable)
GRID_SIZE = 10

# Obstacles (Coordinates as provided)
OBSTACLES = [
    (1, 1), (1, 8), (2, 3), (2, 6), (3, 4),
    (4, 2), (4, 7), (5, 5), (6, 3), (6, 8),
    (7, 2), (7, 9), (8, 1), (8, 5), (9, 4)
]

def a_star(grid, start, goal):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

    def neighbors(node):
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for move in moves:
            neighbor = (node[0] + move[0], node[1] + move[1])
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and grid[neighbor[0]][neighbor[1]] == 0:
                yield neighbor

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        for neighbor in neighbors(current):
            tentative_g_score = g_score[current] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return []

def initialize_grid():
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for obs in OBSTACLES:
        grid[obs] = 1
    return grid

def render_grid(grid, path):
    st.write("")  # Create space
    for i in range(GRID_SIZE):
        row = []
        for j in range(GRID_SIZE):
            icon = EMPTY_ICON
            if (i, j) in OBSTACLES:
                icon = OBSTACLE_ICON
            elif (i, j) == path[0]:  # Start
                icon = ROBOT_ICON
            elif (i, j) == path[-1]:  # Goal
                icon = GOAL_ICON
            elif (i, j) in path:  # Path
                icon = "🟩"
            row.append(icon)
        st.write("".join(row))

def validate_positions(start, end):
    if not (0 <= start[0] < GRID_SIZE and 0 <= start[1] < GRID_SIZE):
        return "Start position is out of bounds."
    if not (0 <= end[0] < GRID_SIZE and 0 <= end[1] < GRID_SIZE):
        return "End position is out of bounds."
    if tuple(start) in OBSTACLES:
        return "Start position cannot be on an obstacle."
    if tuple(end) in OBSTACLES:
        return "End position cannot be on an obstacle."
    if start == end:
        return "Start and end positions cannot be the same."
    return None

def main():
    st.title("🤖 A* Pathfinding Robot Client")
    st.write("Manage and monitor robots with a live grid display.")

    with st.form(key="robot_form"):
        st.subheader("Register a Robot")

        col1, col2, col3 = st.columns(3)
        with col1:
            start = st.text_input("Start Position (x, y)", value="0, 0").split(",")
        with col2:
            end = st.text_input("End Position (x, y)", value="9, 9").split(",")

        submit_button = st.form_submit_button("Find Path")

        if submit_button:
            try:
                start = [int(start[0]), int(start[1])]
                end = [int(end[0]), int(end[1])]

                error_message = validate_positions(start, end)
                if error_message:
                    st.error(f"⚠️ {error_message}")
                else:
                    grid = initialize_grid()
                    path = a_star(grid, tuple(start), tuple(end))
                    if path:
                        st.success(f"🎯 Path found: {path}")
                        render_grid(grid, path)
                    else:
                        st.error("⚠️ No valid path found.")
            except ValueError:
                st.error("⚠️ Please enter valid integer coordinates.")

if __name__ == "__main__":
    main()
