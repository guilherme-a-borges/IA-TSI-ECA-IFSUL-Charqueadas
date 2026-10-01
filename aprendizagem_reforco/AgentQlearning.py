import os
import random
import time
from dataclasses import dataclass
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

ACTIONS = [UP, RIGHT, DOWN, LEFT]

ACTION_NAMES = {
    UP: "UP",
    RIGHT: "RIGHT",
    DOWN: "DOWN",
    LEFT: "LEFT",
}

ROWS = 5
COLS = 5

Position = Tuple[int, int]


# ============================================================
# MATRIX 1: CONTENT OF THE MAP
# ============================================================
# Human coordinates:
# Finn starts at row 1, column 1
# Jake starts at row 1, column 5

CONTENT = np.array([
    [FINN,  EMPTY, EMPTY, EMPTY, JAKE],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
])


@dataclass
class Step:
    action: str
    position: Position


# ============================================================
# MATRIX 2: VALID MOVEMENTS
# ============================================================

def create_valid_moves() -> np.ndarray:
    moves = np.zeros((ROWS, COLS), dtype=int)

    # First, allow all movements that stay inside the map.
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

    # Then, block passages to create the maze.
    # Human coordinates are 1..5, Python coordinates are 0..4.

    # Vertical walls in first row.
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
# BASIC ENVIRONMENT FUNCTIONS
# ============================================================

def find_cell(content: np.ndarray, value: int) -> Position:
    positions = np.argwhere(content == value)

    if len(positions) == 0:
        raise ValueError(f"Value {value} not found in content matrix.")

    row, col = positions[0]
    return int(row), int(col)


def get_valid_actions(valid_moves: np.ndarray, state: Position) -> List[int]:
    row, col = state
    moves = valid_moves[row, col]

    actions = []

    for action in ACTIONS:
        if has_direction(moves, action):
            actions.append(action)

    return actions


def move(state: Position, action: int) -> Position:
    row, col = state
    return row + delta_row(action), col + delta_col(action)


def get_reward(content: np.ndarray, state: Position, goal: Position) -> float:
    if state == goal:
        return 100.0

    if content[state] == TRAP:
        return -20.0

    return -1.0


# ============================================================
# Q-LEARNING
# ============================================================

def create_q_table() -> Dict[Tuple[Position, int], float]:
    q_table = {}

    for row in range(ROWS):
        for col in range(COLS):
            state = (row, col)

            for action in ACTIONS:
                q_table[(state, action)] = 0.0

    return q_table


def best_action(
    q_table: Dict[Tuple[Position, int], float],
    valid_moves: np.ndarray,
    state: Position
) -> Optional[int]:
    valid_actions = get_valid_actions(valid_moves, state)

    if not valid_actions:
        return None

    return max(valid_actions, key=lambda action: q_table[(state, action)])


def choose_action_epsilon_greedy(
    q_table: Dict[Tuple[Position, int], float],
    valid_moves: np.ndarray,
    state: Position,
    epsilon: float
) -> Optional[int]:
    valid_actions = get_valid_actions(valid_moves, state)

    if not valid_actions:
        return None

    if random.random() < epsilon:
        return random.choice(valid_actions)

    return max(valid_actions, key=lambda action: q_table[(state, action)])


def train_q_learning(
    content: np.ndarray,
    valid_moves: np.ndarray,
    start: Position,
    goal: Position,
    episodes: int = 4000,
    max_steps_per_episode: int = 100,
    alpha: float = 0.2,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995
) -> Dict[Tuple[Position, int], float]:
    """
    Q-Learning update:

    Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max Q(s',a') - Q(s,a)]
    """

    q_table = create_q_table()
    epsilon = epsilon_start

    for episode in range(episodes):
        state = start

        for _ in range(max_steps_per_episode):
            action = choose_action_epsilon_greedy(
                q_table=q_table,
                valid_moves=valid_moves,
                state=state,
                epsilon=epsilon
            )

            if action is None:
                break

            next_state = move(state, action)
            reward = get_reward(content, next_state, goal)

            next_valid_actions = get_valid_actions(valid_moves, next_state)

            if next_valid_actions:
                max_next_q = max(q_table[(next_state, next_action)] for next_action in next_valid_actions)
            else:
                max_next_q = 0.0

            old_q = q_table[(state, action)]

            q_table[(state, action)] = old_q + alpha * (
                reward + gamma * max_next_q - old_q
            )

            state = next_state

            if state == goal:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return q_table


def extract_learned_path(
    q_table: Dict[Tuple[Position, int], float],
    valid_moves: np.ndarray,
    start: Position,
    goal: Position,
    max_steps: int = 100
) -> List[Step]:
    """
    After training, follow the best learned action from each state.
    """
    path = []
    state = start
    visited_states = set()

    for _ in range(max_steps):
        if state == goal:
            break

        if state in visited_states:
            print("Warning: learned policy entered a loop.")
            break

        visited_states.add(state)

        action = best_action(q_table, valid_moves, state)

        if action is None:
            print("Warning: no valid action from state:", state)
            break

        next_state = move(state, action)

        path.append(Step(ACTION_NAMES[action], next_state))
        state = next_state

    return path


# ============================================================
# VISUALIZATION
# ============================================================

def load_image(filename: str):
    if os.path.exists(filename):
        return mpimg.imread(filename)
    return None


def get_cell_background(
    pos: Position,
    goal: Position,
    visited: Set[Position],
    learned_path: Set[Position],
    content: np.ndarray
) -> str:
    if pos == goal:
        return "#D2FFD2"  # light green

    if pos in visited:
        return "#FFEB78"  # yellow

    if pos in learned_path:
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
    learned_path: Set[Position],
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
                learned_path=learned_path,
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
        ax.add_patch(Circle((goal_col + 0.5, goal_row + 0.5), 0.30, color="green", zorder=5))
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
        ax.add_patch(Circle((agent_col + 0.5, agent_row + 0.5), 0.30, color="royalblue", zorder=6))
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

    ax.set_title(
        f"Q-Learning | Step: {step_number} | Actuation: {current_action}",
        fontsize=14
    )

    ax.set_xlim(-0.6, cols)
    ax.set_ylim(rows + 0.2, -0.6)
    ax.set_aspect("equal")
    ax.axis("off")


# ============================================================
# MAIN SIMULATION
# ============================================================

def main() -> None:
    random.seed(42)
    np.random.seed(42)

    content = CONTENT.copy()
    valid_moves = VALID_MOVES.copy()

    start = find_cell(content, JAKE)
    goal = find_cell(content, FINN)

    print("Training Q-Learning agent...")

    q_table = train_q_learning(
        content=content,
        valid_moves=valid_moves,
        start=start,
        goal=goal,
        episodes=4000,
        max_steps_per_episode=100,
        alpha=0.2,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995
    )

    learned_solution = extract_learned_path(
        q_table=q_table,
        valid_moves=valid_moves,
        start=start,
        goal=goal,
        max_steps=100
    )

    if not learned_solution:
        print("No learned path found. Try increasing the number of episodes.")
        return

    learned_path_cells = {step.position for step in learned_solution}

    jake_image = load_image("jake.png")
    finn_image = load_image("finn.png")

    if jake_image is None:
        print("jake.png not found. Drawing Jake as a blue circle.")

    if finn_image is None:
        print("finn.png not found. Drawing Finn as a green circle.")

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))

    agent_position = start
    visited = {start}

    draw_maze(
        ax=ax,
        content=content,
        valid_moves=valid_moves,
        agent_position=agent_position,
        goal=goal,
        visited=visited,
        learned_path=learned_path_cells,
        step_number=0,
        current_action="Initial state",
        jake_image=jake_image,
        finn_image=finn_image
    )

    plt.pause(1.0)

    for step_number, step in enumerate(learned_solution, start=1):
        agent_position = step.position
        visited.add(agent_position)

        draw_maze(
            ax=ax,
            content=content,
            valid_moves=valid_moves,
            agent_position=agent_position,
            goal=goal,
            visited=visited,
            learned_path=learned_path_cells,
            step_number=step_number,
            current_action=step.action,
            jake_image=jake_image,
            finn_image=finn_image
        )

        plt.pause(0.7)
        time.sleep(0.1)

    plt.ioff()
    plt.show()

    print("\nGoal reached using learned Q-Learning policy!")
    print("Learned path:")
    for step in learned_solution:
        print(f"{step.action} -> {step.position}")


if __name__ == "__main__":
    main()