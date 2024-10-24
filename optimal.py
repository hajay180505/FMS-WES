import heapq


def a_star(grid, start, goal):
    def heuristic(a, b):
        # Manhattan distance heuristic
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def neighbors(node):
        # Valid moves: up, down, left, right
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for move in moves:
            neighbor = (node[0] + move[0], node[1] + move[1])
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and grid[neighbor[0]][neighbor[1]] == 0:
                yield neighbor

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}  # Keeps track of the path
    g_score = {start: 0}  # Distance from start to the node
    f_score = {start: heuristic(start, goal)}  # Estimated distance to goal

    while open_set:
        _, current = heapq.heappop(open_set)

        # If we reach the goal, reconstruct the path
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Return the path from start to goal

        for neighbor in neighbors(current):
            tentative_g_score = g_score[current] + 1  # Each move has a cost of 1

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []  # Return an empty path if no path is found


# Example usage:
grid = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
]

start = (0, 0)
goal = (9, 9)

path = a_star(grid, start, goal)
print("Path followed:", path)
