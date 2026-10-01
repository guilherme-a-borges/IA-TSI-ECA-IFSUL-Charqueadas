import heapq
import os
from itertools import count
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle


# ============================================================
# CONTENT MATRIX VALUES
# ============================================================

EMPTY = 0
FINN = 1
JAKE = 2
TRAP = 3


# ============================================================
# VALID MOVEMENT BIT MASKS
# ============================================================

UP = 1
RIGHT = 2
DOWN = 4
LEFT = 8

ACTION_BITS = [UP, RIGHT, DOWN, LEFT]
ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]
ACTION_SYMBOLS = ["↑", "→", "↓", "←"]

ROWS = 5
COLS = 5

Position = Tuple[int, int]


# ============================================================
# MATRIX 1: CONTENT OF THE MAP
# ============================================================
# 0 = empty
# 1 = Finn agent initial position
# 2 = Jake agent initial position
# 3 = trap

CONTENT = np.array([
    [FINN,  EMPTY, EMPTY, EMPTY, JAKE],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    [EMPTY, EMPTY, TRAP,  EMPTY, EMPTY],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
])


# ============================================================
# MATRIX 2: VALID MOVEMENTS
# ============================================================

def create_valid_moves() -> np.ndarray:
    moves = np.zeros((ROWS, COLS), dtype=int)

    # First, allow all movements inside the grid.
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

    # Then block passages to create walls.
    # Python coordinates: row 0..4, col 0..4.

    block_passage(moves, 0, 2, RIGHT)
    block_passage(moves, 0, 3, DOWN)

    block_passage(moves, 1, 1, RIGHT)
    block_passage(moves, 2, 1, RIGHT)
    block_passage(moves, 2, 2, RIGHT)

    block_passage(moves, 1, 4, DOWN)
    block_passage(moves, 3, 3, RIGHT)

    block_passage(moves, 3, 1, DOWN)
    block_passage(moves, 3, 2, DOWN)

    return moves


def block_passage(moves: np.ndarray, row: int, col: int, direction: int) -> None:
    moves[row, col] = remove_direction(moves[row, col], direction)

    neighbor_row = row + delta_row(direction)
    neighbor_col = col + delta_col(direction)

    if is_inside(neighbor_row, neighbor_col):
        moves[neighbor_row, neighbor_col] = remove_direction(
            moves[neighbor_row, neighbor_col],
            opposite(direction)
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
# BASIC ENVIRONMENT FUNCTIONS
# ============================================================

def find_cell(content: np.ndarray, value: int) -> Position:
    positions = np.argwhere(content == value)

    if len(positions) == 0:
        raise ValueError(f"Value {value} not found in content matrix.")

    row, col = positions[0]
    return int(row), int(col)


def action_to_delta(action_index: int) -> Tuple[int, int]:
    action_bit = ACTION_BITS[action_index]
    return delta_row(action_bit), delta_col(action_bit)


def move_agent(state: Position, action_index: int) -> Position:
    dr, dc = action_to_delta(action_index)
    return state[0] + dr, state[1] + dc


def get_valid_actions(valid_moves: np.ndarray, state: Position) -> List[int]:
    moves = valid_moves[state]
    actions = []

    for action_index, action_bit in enumerate(ACTION_BITS):
        if has_direction(moves, action_bit):
            actions.append(action_index)

    return actions


def get_neighbors(valid_moves: np.ndarray, state: Position) -> List[Tuple[int, Position]]:
    neighbors = []

    for action_index in get_valid_actions(valid_moves, state):
        next_state = move_agent(state, action_index)
        neighbors.append((action_index, next_state))

    return neighbors


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def found_each_other(
    old_jake: Position,
    old_finn: Position,
    new_jake: Position,
    new_finn: Position
) -> bool:
    same_cell = new_jake == new_finn
    crossed_paths = new_jake == old_finn and new_finn == old_jake
    return same_cell or crossed_paths


# ============================================================
# A* SEARCH
# ============================================================

def movement_cost(content: np.ndarray, next_state: Position) -> float:
    """
    Basic movement cost.

    Normal cell: cost 1
    Trap cell: higher cost, so A* tends to avoid it if possible.
    """
    if content[next_state] == TRAP:
        return 6.0

    return 1.0


def astar_search(
    content: np.ndarray,
    valid_moves: np.ndarray,
    start: Position,
    goal: Position
) -> List[Tuple[int, Position]]:
    """
    Returns a path as a list of:
    [(action_index, next_position), ...]
    """

    if start == goal:
        return []

    frontier = []
    tie_breaker = count()

    heapq.heappush(frontier, (0.0, next(tie_breaker), start))

    came_from: Dict[Position, Position] = {}
    action_from: Dict[Position, int] = {}
    cost_so_far: Dict[Position, float] = {start: 0.0}

    while frontier:
        _, _, current = heapq.heappop(frontier)

        if current == goal:
            return reconstruct_path(came_from, action_from, current)

        for action_index, next_state in get_neighbors(valid_moves, current):
            new_cost = cost_so_far[current] + movement_cost(content, next_state)

            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                cost_so_far[next_state] = new_cost

                priority = new_cost + manhattan(next_state, goal)
                heapq.heappush(frontier, (priority, next(tie_breaker), next_state))

                came_from[next_state] = current
                action_from[next_state] = action_index

    return []


def reconstruct_path(
    came_from: Dict[Position, Position],
    action_from: Dict[Position, int],
    current: Position
) -> List[Tuple[int, Position]]:
    path = []

    while current in came_from:
        action_index = action_from[current]
        path.append((action_index, current))
        current = came_from[current]

    path.reverse()
    return path


# ============================================================
# VISUALIZATION
# ============================================================

def load_image(filename: str):
    if os.path.exists(filename):
        return mpimg.imread(filename)
    return None


def draw_thick_walls(ax, valid_moves: np.ndarray) -> None:
    for row in range(ROWS):
        for col in range(COLS):
            x = col
            y = row
            moves = valid_moves[row, col]

            if not has_direction(moves, UP):
                ax.plot([x, x + 1], [y, y], color="black", linewidth=6, zorder=4)

            if not has_direction(moves, RIGHT):
                ax.plot([x + 1, x + 1], [y, y + 1], color="black", linewidth=6, zorder=4)

            if not has_direction(moves, DOWN):
                ax.plot([x, x + 1], [y + 1, y + 1], color="black", linewidth=6, zorder=4)

            if not has_direction(moves, LEFT):
                ax.plot([x, x], [y, y + 1], color="black", linewidth=6, zorder=4)


def draw_agent(
    ax,
    position: Position,
    image,
    fallback_color: str,
    label: str,
    offset_x: float = 0.0
) -> None:
    row, col = position

    if image is not None:
        ax.imshow(
            image,
            extent=(
                col + 0.15 + offset_x,
                col + 0.85 + offset_x,
                row + 0.85,
                row + 0.15
            ),
            zorder=8
        )
    else:
        ax.add_patch(
            Circle(
                (col + 0.5 + offset_x, row + 0.5),
                0.30,
                color=fallback_color,
                zorder=8
            )
        )
        ax.text(
            col + 0.5 + offset_x,
            row + 0.5,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=16,
            fontweight="bold",
            zorder=9
        )


def draw_transition_arrow(ax, old_pos: Optional[Position], new_pos: Optional[Position], color: str) -> None:
    if old_pos is None or new_pos is None:
        return

    ax.annotate(
        "",
        xy=(new_pos[1] + 0.5, new_pos[0] + 0.5),
        xytext=(old_pos[1] + 0.5, old_pos[0] + 0.5),
        arrowprops=dict(arrowstyle="->", color=color, linewidth=3),
        zorder=6
    )


def draw_planned_path(
    ax,
    path: List[Tuple[int, Position]],
    color: str,
    offset_y: float
) -> None:
    if not path:
        return

    xs = []
    ys = []

    for _, position in path:
        row, col = position
        xs.append(col + 0.5)
        ys.append(row + 0.5 + offset_y)

    ax.plot(
        xs,
        ys,
        linestyle="--",
        linewidth=2,
        color=color,
        zorder=5,
        alpha=0.8
    )


def draw_maze(
    ax,
    content: np.ndarray,
    valid_moves: np.ndarray,
    jake: Position,
    finn: Position,
    previous_jake: Optional[Position],
    previous_finn: Optional[Position],
    jake_visited: Set[Position],
    finn_visited: Set[Position],
    jake_path: List[Tuple[int, Position]],
    finn_path: List[Tuple[int, Position]],
    step_number: int,
    jake_image=None,
    finn_image=None
) -> None:
    ax.clear()

    for row in range(ROWS):
        for col in range(COLS):
            pos = (row, col)

            if content[pos] == TRAP:
                color = "#FFB4B4"
            elif pos in jake_visited and pos in finn_visited:
                color = "#E6D2FF"
            elif pos in jake_visited:
                color = "#BEDCFF"
            elif pos in finn_visited:
                color = "#FFE6AA"
            else:
                color = "white"

            ax.add_patch(
                Rectangle(
                    (col, row),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="lightgray",
                    linewidth=1,
                    zorder=1
                )
            )

    draw_planned_path(ax, jake_path, color="blue", offset_y=-0.12)
    draw_planned_path(ax, finn_path, color="darkorange", offset_y=0.12)

    draw_transition_arrow(ax, previous_jake, jake, "blue")
    draw_transition_arrow(ax, previous_finn, finn, "darkorange")

    draw_thick_walls(ax, valid_moves)

    if jake == finn:
        draw_agent(ax, jake, jake_image, "royalblue", "J", offset_x=-0.13)
        draw_agent(ax, finn, finn_image, "orange", "F", offset_x=0.13)
    else:
        draw_agent(ax, jake, jake_image, "royalblue", "J")
        draw_agent(ax, finn, finn_image, "orange", "F")

    for col in range(COLS):
        ax.text(
            col + 0.5,
            -0.35,
            str(col + 1),
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold"
        )

    for row in range(ROWS):
        ax.text(
            -0.35,
            row + 0.5,
            str(row + 1),
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold"
        )

    ax.set_title(
        f"Multi-agent A* planning | Simultaneous execution | Step {step_number}",
        fontsize=14
    )

    ax.set_xlim(-0.6, COLS)
    ax.set_ylim(ROWS + 0.2, -0.6)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_stats(
    ax,
    step_number: int,
    jake: Position,
    finn: Position,
    jake_action: str,
    finn_action: str,
    jake_path: List[Tuple[int, Position]],
    finn_path: List[Tuple[int, Position]],
    status_text: str
) -> None:
    ax.clear()
    ax.axis("off")

    text = (
        f"Method: A* search\n"
        f"Environment: multi-agent\n"
        f"Planning mode: dynamic replanning\n\n"
        f"Step: {step_number}\n"
        f"Status: {status_text}\n\n"
        f"Jake position: ({jake[0] + 1}, {jake[1] + 1})\n"
        f"Finn position: ({finn[0] + 1}, {finn[1] + 1})\n"
        f"Distance Jake-Finn: {manhattan(jake, finn)}\n\n"
        f"Jake target: Finn current position\n"
        f"Finn target: Jake current position\n\n"
        f"Jake action: {jake_action}\n"
        f"Finn action: {finn_action}\n\n"
        f"Jake planned path length: {len(jake_path)}\n"
        f"Finn planned path length: {len(finn_path)}\n\n"
        f"Goal condition:\n"
        f"same cell or crossing paths"
    )

    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )


def draw_path_table(
    ax,
    jake_path: List[Tuple[int, Position]],
    finn_path: List[Tuple[int, Position]]
) -> None:
    ax.clear()
    ax.axis("off")
    ax.set_title("Current A* plans", fontsize=12)

    lines = []

    lines.append("Jake plan toward Finn:")
    if not jake_path:
        lines.append("  No movement / already at target")
    else:
        for i, (action, pos) in enumerate(jake_path[:10], start=1):
            lines.append(
                f"  {i:02d}. {ACTION_SYMBOLS[action]} {ACTION_NAMES[action]:>5} -> ({pos[0] + 1}, {pos[1] + 1})"
            )

    lines.append("")
    lines.append("Finn plan toward Jake:")
    if not finn_path:
        lines.append("  No movement / already at target")
    else:
        for i, (action, pos) in enumerate(finn_path[:10], start=1):
            lines.append(
                f"  {i:02d}. {ACTION_SYMBOLS[action]} {ACTION_NAMES[action]:>5} -> ({pos[0] + 1}, {pos[1] + 1})"
            )

    ax.text(
        0.02,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )


def draw_frame(
    ax_maze,
    ax_stats,
    ax_plans,
    content: np.ndarray,
    valid_moves: np.ndarray,
    jake: Position,
    finn: Position,
    previous_jake: Optional[Position],
    previous_finn: Optional[Position],
    jake_visited: Set[Position],
    finn_visited: Set[Position],
    jake_path: List[Tuple[int, Position]],
    finn_path: List[Tuple[int, Position]],
    step_number: int,
    jake_action: str,
    finn_action: str,
    status_text: str,
    jake_image=None,
    finn_image=None
) -> None:
    draw_maze(
        ax=ax_maze,
        content=content,
        valid_moves=valid_moves,
        jake=jake,
        finn=finn,
        previous_jake=previous_jake,
        previous_finn=previous_finn,
        jake_visited=jake_visited,
        finn_visited=finn_visited,
        jake_path=jake_path,
        finn_path=finn_path,
        step_number=step_number,
        jake_image=jake_image,
        finn_image=finn_image
    )

    draw_stats(
        ax=ax_stats,
        step_number=step_number,
        jake=jake,
        finn=finn,
        jake_action=jake_action,
        finn_action=finn_action,
        jake_path=jake_path,
        finn_path=finn_path,
        status_text=status_text
    )

    draw_path_table(
        ax=ax_plans,
        jake_path=jake_path,
        finn_path=finn_path
    )


# ============================================================
# SIMULATION
# ============================================================

def execute_two_agents_astar(
    content: np.ndarray,
    valid_moves: np.ndarray,
    ax_maze,
    ax_stats,
    ax_plans,
    jake_image=None,
    finn_image=None,
    max_steps: int = 80
) -> None:
    jake = find_cell(content, JAKE)
    finn = find_cell(content, FINN)

    previous_jake = None
    previous_finn = None

    jake_visited = {jake}
    finn_visited = {finn}

    status = "Both agents plan with A* toward each other."

    for step_number in range(0, max_steps + 1):
        # Dynamic replanning.
        # Jake plans to Finn's current position.
        # Finn plans to Jake's current position.
        jake_path = astar_search(content, valid_moves, start=jake, goal=finn)
        finn_path = astar_search(content, valid_moves, start=finn, goal=jake)

        if step_number == 0:
            draw_frame(
                ax_maze=ax_maze,
                ax_stats=ax_stats,
                ax_plans=ax_plans,
                content=content,
                valid_moves=valid_moves,
                jake=jake,
                finn=finn,
                previous_jake=previous_jake,
                previous_finn=previous_finn,
                jake_visited=jake_visited,
                finn_visited=finn_visited,
                jake_path=jake_path,
                finn_path=finn_path,
                step_number=step_number,
                jake_action="-",
                finn_action="-",
                status_text=status,
                jake_image=jake_image,
                finn_image=finn_image
            )

            plt.pause(1.0)
            continue

        old_jake = jake
        old_finn = finn

        if not jake_path and not finn_path:
            status = "No path available for both agents."
            break

        # Each agent executes only the first actuation of its current A* plan.
        if jake_path:
            jake_action_index, new_jake = jake_path[0]
            jake_action_name = ACTION_NAMES[jake_action_index]
        else:
            new_jake = jake
            jake_action_name = "WAIT"

        if finn_path:
            finn_action_index, new_finn = finn_path[0]
            finn_action_name = ACTION_NAMES[finn_action_index]
        else:
            new_finn = finn
            finn_action_name = "WAIT"

        previous_jake = old_jake
        previous_finn = old_finn

        jake = new_jake
        finn = new_finn

        jake_visited.add(jake)
        finn_visited.add(finn)

        if found_each_other(old_jake, old_finn, new_jake, new_finn):
            status = "Jake and Finn found each other."
        else:
            status = "Both agents replanned and moved simultaneously."

        # Recompute paths after the movement, only for display.
        display_jake_path = astar_search(content, valid_moves, start=jake, goal=finn)
        display_finn_path = astar_search(content, valid_moves, start=finn, goal=jake)

        draw_frame(
            ax_maze=ax_maze,
            ax_stats=ax_stats,
            ax_plans=ax_plans,
            content=content,
            valid_moves=valid_moves,
            jake=jake,
            finn=finn,
            previous_jake=previous_jake,
            previous_finn=previous_finn,
            jake_visited=jake_visited,
            finn_visited=finn_visited,
            jake_path=display_jake_path,
            finn_path=display_finn_path,
            step_number=step_number,
            jake_action=jake_action_name,
            finn_action=finn_action_name,
            status_text=status,
            jake_image=jake_image,
            finn_image=finn_image
        )

        plt.pause(0.8)

        if found_each_other(old_jake, old_finn, new_jake, new_finn):
            return

    print(status)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    jake_image = load_image("jake.png")
    finn_image = load_image("finn.png")

    if jake_image is None:
        print("jake.png not found. Drawing Jake as a blue circle.")

    if finn_image is None:
        print("finn.png not found. Drawing Finn as an orange circle.")

    print("\nMulti-agent A* simulation:")
    print("Jake is an agent and Finn is an agent.")
    print("Jake plans toward Finn's current position.")
    print("Finn plans toward Jake's current position.")
    print("At each time step, both agents move simultaneously.")
    print("The task is complete when they reach the same cell or cross paths.\n")

    plt.ion()

    fig = plt.figure(figsize=(13, 7))

    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.35, 1.0, 0.95],
        height_ratios=[1.0, 1.0]
    )

    ax_maze = fig.add_subplot(gs[:, 0])
    ax_stats = fig.add_subplot(gs[:, 1])
    ax_plans = fig.add_subplot(gs[:, 2])

    execute_two_agents_astar(
        content=CONTENT.copy(),
        valid_moves=VALID_MOVES.copy(),
        ax_maze=ax_maze,
        ax_stats=ax_stats,
        ax_plans=ax_plans,
        jake_image=jake_image,
        finn_image=finn_image,
        max_steps=80
    )

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()