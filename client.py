import streamlit as st
import numpy as np
import time
import requests

# Streamlit page configuration
st.set_page_config(page_title="Visual Robot Movement", layout="wide")

# Server URL configuration
SERVER_URL = "http://127.0.0.1:5000"

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

def set_ui_style():
    st.markdown(
        """
        <style>
        .big-font { font-size:24px !important; }
        .robot-card { border-radius: 10px; padding: 15px; background-color: #f5f7fa; margin-bottom: 15px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def register_robot(robot_id, start, end, priority):
    data = {"id": robot_id, "start": start, "end": end, "priority": priority}
    try:
        response = requests.post(f"{SERVER_URL}/register_robot", json=data, timeout=5)
        response.raise_for_status()
        st.success(f"✅ Robot {robot_id} registered successfully!")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to register Robot {robot_id}: {e}")

def get_next_action(robot_id, state):
    try:
        response = requests.post(
            f"{SERVER_URL}/get_action", json={"id": robot_id, "state": state.tolist()}
        )
        if response.status_code == 200:
            return response.json().get("action")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error getting action for Robot {robot_id}: {e}")
    return None

def render_grid(grid, robot_position, goal_position):
    grid_display = st.empty()
    grid[goal_position[1], goal_position[0]] = GOAL_ICON
    for obstacle in OBSTACLES:
        grid[obstacle[1], obstacle[0]] = OBSTACLE_ICON
    grid[robot_position[1], robot_position[0]] = ROBOT_ICON

    with grid_display.container():
        for row in grid:
            st.write("".join(row))

def calculate_manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def monitor_robot(robot_id, start, end):
    state = np.array([start[0], start[1], end[0], end[1]])
    actions_map = {0: "Move Right", 1: "Move Down", 2: "Move Left", 3: "Move Up"}

    grid = np.full((GRID_SIZE, GRID_SIZE), EMPTY_ICON)

    total_distance = calculate_manhattan_distance(start, end)
    progress_bar = st.progress(0.0)

    step_count = 0
    max_steps = GRID_SIZE * 100

    try:
        while step_count < max_steps:
            action = get_next_action(robot_id, state)
            if action is None:
                st.error(f"❌ Failed to get action for Robot {robot_id}.")
                break

            st.markdown(
                f'<div class="robot-card">🚀 <b>Robot {robot_id}</b> - Action: <span class="big-font">{actions_map[action]}</span></div>',
                unsafe_allow_html=True,
            )

            new_x, new_y = state[0], state[1]
            if action == 0 and is_valid_position(new_x + 1, new_y):
                new_x += 1
            elif action == 1 and is_valid_position(new_x, new_y + 1):
                new_y += 1
            elif action == 2 and is_valid_position(new_x - 1, new_y):
                new_x -= 1
            elif action == 3 and is_valid_position(new_x, new_y - 1):
                new_y -= 1

            grid[state[1], state[0]] = EMPTY_ICON
            state[0], state[1] = new_x, new_y

            render_grid(grid, (new_x, new_y), tuple(end))

            remaining_distance = calculate_manhattan_distance((new_x, new_y), end)
            progress = max(0.0, min(1.0, (total_distance - remaining_distance) / total_distance))
            progress_bar.progress(progress)

            if (new_x, new_y) == tuple(end):
                st.success(f"🎯 Robot {robot_id} has reached its destination!")
                break

            step_count += 1
            time.sleep(1)

    except KeyboardInterrupt:
        st.info(f"⚠️ Robot {robot_id} monitoring interrupted.")

def is_valid_position(x, y):
    return (
        0 <= x < GRID_SIZE and
        0 <= y < GRID_SIZE and
        (x, y) not in OBSTACLES
    )

def validate_positions(start, end):
    if not (0 <= start[0] < GRID_SIZE and 0 <= start[1] < GRID_SIZE):
        return "Start position is out of bounds."
    if not (0 <= end[0] < GRID_SIZE and 0 <= end[1] < GRID_SIZE):
        return "End position is out of bounds."
    if (start[0], start[1]) in OBSTACLES:
        return "Start position cannot be on an obstacle."
    if (end[0], end[1]) in OBSTACLES:
        return "End position cannot be on an obstacle."
    if start == end:
        return "Start and end positions cannot be the same."

    return None

def main():
    set_ui_style()
    st.title("🤖 Visual Robot Client with Grid Movement")
    st.write("Manage and monitor robots with a live grid display.")

    with st.form(key="robot_form"):
        st.subheader("Register a Robot")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            robot_id = st.text_input("Robot ID", value="Robot_1")
        with col2:
            start = st.text_input("Start Position (x, y)", value="0, 0").split(",")
        with col3:
            end = st.text_input("End Position (x, y)", value="9, 9").split(",")
        with col4:
            priority = st.slider("Priority", 1, 10, value=1)

        submit_button = st.form_submit_button("Register and Monitor")

        if submit_button:
            try:
                start = [int(start[0]), int(start[1])]
                end = [int(end[0]), int(end[1])]
                error_message = validate_positions(start, end)
                
                if error_message:
                    st.error(f"⚠️ {error_message}")
                else:
                    register_robot(robot_id, start, end, priority)
                    monitor_robot(robot_id, start, end)
            except ValueError:
                st.error("⚠️ Please enter valid integer coordinates.")

if __name__ == "__main__":
    main()
