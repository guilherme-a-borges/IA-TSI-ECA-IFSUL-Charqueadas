import heapq
import os
import time
from dataclasses import dataclass
from itertools import count
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle


# ============================================================
# CONTENT MATRIX VALUES
# ============================================================
# This matrix represents what exists inside each cell.

EMPTY = 0
FINN = 1
JAKE = 2
TRAP = 3


# ============================================================
# VALID MOVEMENT BIT MASKS
# ============================================================
# These values can be combined.
#
# Example:
# RIGHT | DOWN means the agent can move right and down.

UP = 1
RIGHT = 2
DOWN = 4
LEFT = 8


ROWS = 5
COLS = 5


# ============================================================
# MATRIX 1: CONTENT OF THE MAP
# ============================================================
# Human coordinates:
#
# Finn starts at row 1, column 1
# Jake starts at row 1, column 5
#
# Python uses zero-based indexing:
# Finn = content[0][0]
# Jake = content[0][4]

CONTENT = np.array([
    [FINN,  EMPTY, EMPTY, EMPTY, JAKE],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
])


Position = Tuple[int, int]  # row, col


@dataclass
class Step:
    action: str
    position: Position


# ============================================================
# VALID MOVEMENT MATRIX
# ============================================================

def create_valid_moves() -> np.ndarray:
    """
    Creates a matrix where each cell stores the allowed movements.
    Walls are represented by removing movement between adjacent cells.
    """
    moves = np.zeros((ROWS, COLS), dtype=int)

    # First, allow all movements that stay inside the grid.
    for row in range(ROWS):
        for col in range(COLS):
            value = 0

            if row > 0:
                value |= UP
            if col < COLS - 1:
                value |= RIGHT
            if row < ROWS - 1:
                value |= DOWN
            if col > 0:
                value |= LEFT

            moves[row, col] = value

    # Then, block specific passages according to the maze walls.
    #
    # Human coordinates use rows/columns 1..5.
    # Python coordinates use rows/columns 0..4.

    # Vertical walls in the first row.
    block_passage(moves, 0, 1, RIGHT)  # between (1,2) and (1,3)
    block_passage(moves, 0, 2, RIGHT)  # between (1,3) and (1,4)

    # Vertical wall between columns 1 and 2, from rows 2 to 4.
    block_passage(moves, 1, 0, RIGHT)  # between (2,1) and (2,2)
    block_passage(moves, 2, 0, RIGHT)  # between (3,1) and (3,2)
    block_passage(moves, 3, 0, RIGHT)  # between (4,1) and (4,2)

    # Vertical wall between columns 2 and 3, from rows 2 to 3.
    block_passage(moves, 1, 1, RIGHT)  # between (2,2) and (2,3)
    block_passage(moves, 2, 1, RIGHT)  # between (3,2) and (3,3)

    # Other internal vertical walls.
    block_passage(moves, 2, 2, RIGHT)  # between (3,3) and (3,4)
    block_passage(moves, 3, 3, RIGHT)  # between (4,4) and (4,5)

    # Horizontal walls.
    block_passage(moves, 0, 0, DOWN)   # between (1,1) and (2,1)
    block_passage(moves, 1, 4, DOWN)   # between (2,5) and (3,5)
    block_passage(moves, 2, 3, DOWN)   # between (3,4) and (4,4)
    block_passage(moves, 3, 1, DOWN)   # between (4,2) and (5,2)
    block_passage(moves, 3, 2, DOWN)   # between (4,3) and (5,3)

    return moves


def block_passage(moves: np.ndarray, row: int, col: int, direction: int) -> None:
    """
    Blocks movement from one cell to its neighbor.
    It also blocks the opposite movement from the neighbor back to this cell.
    """
    moves[row, col] = remove_direction(moves[row, col], direction)

    neighbor_row = row + delta_row(direction)
    neighbor_col = col + delta_col(direction)

    if is_inside(neighbor_row, neighbor_col):
        opposite_direction = opposite(direction)
        moves[neighbor_row, neighbor_col] = remove_direction(
            moves[neighbor_row, neighbor_col],
            opposite_direction
        )


def remove_direction(value: int, direction: int) -> int:
    return value & ~direction


def is_inside(row: int, col: int) -> bool:
    return 0 <= row < ROWS and 0 <= col < COLS


def delta_row(direction: int) -> int:
    if direction == UP:
        return -1
    if direction == DOWN:
        return 1
    return 0


def delta_col(direction: int) -> int:
    if direction == LEFT:
        return -1
    if direction == RIGHT:
        return 1
    return 0


def opposite(direction: int) -> int:
    if direction == UP:
        return DOWN
    if direction == DOWN:
        return UP
    if direction == LEFT:
        return RIGHT
    if direction == RIGHT:
        return LEFT
    raise ValueError(f"Invalid direction: {direction}")


def has_direction(moves: int, direction: int) -> bool:
    return (moves & direction) != 0


VALID_MOVES = create_valid_moves()


# ============================================================
# SEARCH FUNCTIONS
# ============================================================

def find_cell(content: np.ndarray, value: int) -> Position:
    positions = np.argwhere(content == value)

    if len(positions) == 0:
        raise ValueError(f"Value {value} not found in content matrix.")

    row, col = positions[0]
    return int(row), int(col)


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(valid_moves: np.ndarray, pos: Position) -> List[Step]:
    row, col = pos
    moves = valid_moves[row, col]

    neighbors = []

    if has_direction(moves, UP):
        neighbors.append(Step("UP", (row - 1, col)))

    if has_direction(moves, RIGHT):
        neighbors.append(Step("RIGHT", (row, col + 1)))

    if has_direction(moves, DOWN):
        neighbors.append(Step("DOWN", (row + 1, col)))

    if has_direction(moves, LEFT):
        neighbors.append(Step("LEFT", (row, col - 1)))

    return neighbors


def reconstruct_path(
    came_from: Dict[Position, Position],
    action_from: Dict[Position, str],
    current: Position
) -> List[Step]:
    path = []

    while current in came_from:
        action = action_from[current]
        path.append(Step(action, current))
        current = came_from[current]

    path.reverse()
    return path


def astar_search(
    content: np.ndarray,
    valid_moves: np.ndarray,
    start: Position,
    goal: Position
) -> List[Step]:
    frontier = []
    tie_breaker = count()

    heapq.heappush(frontier, (0, next(tie_breaker), start))

    came_from: Dict[Position, Position] = {}
    action_from: Dict[Position, str] = {}
    cost_so_far: Dict[Position, int] = {start: 0}

    while frontier:
        _, _, current = heapq.heappop(frontier)

        if current == goal:
            return reconstruct_path(came_from, action_from, current)

        for step in get_neighbors(valid_moves, current):
            next_pos = step.position

            new_cost = cost_so_far[current] + 1

            # Optional example: trap cells are more expensive.
            if content[next_pos] == TRAP:
                new_cost += 5

            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost

                priority = new_cost + manhattan(next_pos, goal)
                heapq.heappush(frontier, (priority, next(tie_breaker), next_pos))

                came_from[next_pos] = current
                action_from[next_pos] = step.action

    return []


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def load_image(filename: str):
    if os.path.exists(filename):
        return mpimg.imread(filename)
    return None


def get_cell_background(
    pos: Position,
    goal: Position,
    visited: Set[Position],
    planned_path: Set[Position],
    content: np.ndarray
) -> str:
    if pos == goal:
        return "#D2FFD2"  # light green

    if pos in visited:
        return "#FFEB78"  # yellow

    if pos in planned_path:
        return "#B4E1FF"  # light blue

    if content[pos] == TRAP:
        return "#FFB4B4"  # light red

    return "white"


def draw_maze(
    ax,
    content: np.ndarray,
    valid_moves: np.ndarray,
    agent_position: Position,
    goal: Position,
    visited: Set[Position],
    planned_path: Set[Position],
    step_number: int,
    current_action: str,
    jake_image=None,
    finn_image=None
) -> None:
    ax.clear()

    rows, cols = content.shape

    # Draw cell backgrounds.
    for row in range(rows):
        for col in range(cols):
            pos = (row, col)

            color = get_cell_background(
                pos=pos,
                goal=goal,
                visited=visited,
                planned_path=planned_path,
                content=content
            )

            rect = Rectangle(
                (col, row),
                1,
                1,
                facecolor=color,
                edgecolor="lightgray",
                linewidth=1,
                zorder=1
            )
            ax.add_patch(rect)

    # Draw thick walls.
    for row in range(rows):
        for col in range(cols):
            x = col
            y = row
            moves = valid_moves[row, col]

            if not has_direction(moves, UP):
                ax.plot([x, x + 1], [y, y], color="black", linewidth=6, zorder=3)

            if not has_direction(moves, RIGHT):
                ax.plot([x + 1, x + 1], [y, y + 1], color="black", linewidth=6, zorder=3)

            if not has_direction(moves, DOWN):
                ax.plot([x, x + 1], [y + 1, y + 1], color="black", linewidth=6, zorder=3)

            if not has_direction(moves, LEFT):
                ax.plot([x, x], [y, y + 1], color="black", linewidth=6, zorder=3)

    # Draw Finn.
    goal_row, goal_col = goal
    if finn_image is not None:
        ax.imshow(
            finn_image,
            extent=(goal_col + 0.15, goal_col + 0.85, goal_row + 0.85, goal_row + 0.15),
            zorder=5
        )
    else:
        ax.add_patch(
            Circle(
                (goal_col + 0.5, goal_row + 0.5),
                0.30,
                color="green",
                zorder=5
            )
        )
        ax.text(
            goal_col + 0.5,
            goal_row + 0.5,
            "F",
            ha="center",
            va="center",
            color="white",
            fontsize=16,
            fontweight="bold",
            zorder=6
        )

    # Draw Jake.
    agent_row, agent_col = agent_position
    if jake_image is not None:
        ax.imshow(
            jake_image,
            extent=(agent_col + 0.15, agent_col + 0.85, agent_row + 0.85, agent_row + 0.15),
            zorder=6
        )
    else:
        ax.add_patch(
            Circle(
                (agent_col + 0.5, agent_row + 0.5),
                0.30,
                color="royalblue",
                zorder=6
            )
        )
        ax.text(
            agent_col + 0.5,
            agent_row + 0.5,
            "J",
            ha="center",
            va="center",
            color="white",
            fontsize=16,
            fontweight="bold",
            zorder=7
        )

    # Draw coordinate labels.
    for col in range(cols):
        ax.text(
            col + 0.5,
            -0.35,
            str(col + 1),
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold"
        )

    for row in range(rows):
        ax.text(
            -0.35,
            row + 0.5,
            str(row + 1),
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold"
        )

    ax.set_title(f"Step: {step_number} | Actuation: {current_action}", fontsize=14)

    ax.set_xlim(-0.6, cols)
    ax.set_ylim(rows + 0.2, -0.6)
    ax.set_aspect("equal")
    ax.axis("off")


# ============================================================
# MAIN SIMULATION
# ============================================================

def main() -> None:
    content = CONTENT.copy()
    valid_moves = VALID_MOVES.copy()

    start = find_cell(content, JAKE)
    goal = find_cell(content, FINN)

    solution = astar_search(content, valid_moves, start, goal)

    if not solution:
        print("No path found.")
        return

    planned_path = {step.position for step in solution}
    visited = {start}

    jake_image = load_image("jake.png")
    finn_image = load_image("finn.png")

    if jake_image is None:
        print("jake.png not found. Drawing Jake as a blue circle.")

    if finn_image is None:
        print("finn.png not found. Drawing Finn as a green circle.")

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))

    agent_position = start

    draw_maze(
        ax=ax,
        content=content,
        valid_moves=valid_moves,
        agent_position=agent_position,
        goal=goal,
        visited=visited,
        planned_path=planned_path,
        step_number=0,
        current_action="Initial state",
        jake_image=jake_image,
        finn_image=finn_image
    )

    plt.pause(0.8)

    for step_number, step in enumerate(solution, start=1):
        agent_position = step.position
        visited.add(agent_position)

        draw_maze(
            ax=ax,
            content=content,
            valid_moves=valid_moves,
            agent_position=agent_position,
            goal=goal,
            visited=visited,
            planned_path=planned_path,
            step_number=step_number,
            current_action=step.action,
            jake_image=jake_image,
            finn_image=finn_image
        )

        plt.pause(0.7)
        time.sleep(0.1)

    plt.ioff()
    plt.show()

    print("Goal reached!")
    print("Path:")
    for step in solution:
        print(f"{step.action} -> {step.position}")


if __name__ == "__main__":
    main()