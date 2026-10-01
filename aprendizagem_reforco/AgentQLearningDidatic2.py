import os
import random
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

ACTION_NAMES = {
    0: "UP",
    1: "RIGHT",
    2: "DOWN",
    3: "LEFT",
}

ACTION_SYMBOLS = {
    0: "↑",
    1: "→",
    2: "↓",
    3: "←",
}

ROWS = 5
COLS = 5

Position = Tuple[int, int]


# ============================================================
# MATRIX 1: CONTENT OF THE MAP
# ============================================================
# 0 = empty
# 1 = Finn / goal
# 2 = Jake / agent
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
    """
    Each cell stores the movements allowed from that cell.

    Example:
    RIGHT | DOWN means that, from that cell, the agent can move
    right and down.

    Walls are represented by blocking movements between cells.
    """
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

    # Then block passages to create the maze.
    # Python coordinates: row 0..4, col 0..4.
    #
    # The path is intentionally possible:
    # Jake starts at (0,4), Finn is at (0,0).

    # Top internal walls.
    block_passage(moves, 0, 1, RIGHT)
    block_passage(moves, 0, 2, RIGHT)
    block_passage(moves, 0, 3, DOWN)

    # Right side control.
    block_passage(moves, 1, 4, DOWN)
    block_passage(moves, 2, 4, DOWN)

    # Left-side vertical barriers between columns 0 and 1.
    block_passage(moves, 1, 0, RIGHT)
    block_passage(moves, 2, 0, RIGHT)
    block_passage(moves, 3, 0, RIGHT)

    # Internal vertical barriers.
    block_passage(moves, 1, 1, RIGHT)
    block_passage(moves, 2, 1, RIGHT)
    block_passage(moves, 3, 1, RIGHT)

    block_passage(moves, 2, 2, RIGHT)
    block_passage(moves, 3, 3, RIGHT)

    # Internal horizontal barriers.
    block_passage(moves, 0, 0, DOWN)
    block_passage(moves, 3, 2, DOWN)

    return moves


def block_passage(moves: np.ndarray, row: int, col: int, direction: int) -> None:
    """
    Blocks movement from one cell to its neighbor.
    Also blocks the opposite movement from the neighbor back.
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


def move(state: Position, action_index: int) -> Position:
    dr, dc = action_to_delta(action_index)
    return state[0] + dr, state[1] + dc


def get_valid_actions(valid_moves: np.ndarray, state: Position) -> List[int]:
    row, col = state
    moves = valid_moves[row, col]

    valid_actions = []

    for action_index, action_bit in enumerate(ACTION_BITS):
        if has_direction(moves, action_bit):
            valid_actions.append(action_index)

    return valid_actions


def get_reward(content: np.ndarray, state: Position, goal: Position) -> float:
    if state == goal:
        return 100.0

    if content[state] == TRAP:
        return -20.0

    return -1.0


# ============================================================
# REWARD MATRIX
# ============================================================

def create_reward_matrix(content: np.ndarray, goal: Position) -> np.ndarray:
    """
    Reward received when the agent enters each cell.

    +100 = goal
    -20  = trap
    -1   = normal cell
    """
    reward_matrix = np.full((ROWS, COLS), -1.0)

    for row in range(ROWS):
        for col in range(COLS):
            state = (row, col)

            if state == goal:
                reward_matrix[row, col] = 100.0
            elif content[state] == TRAP:
                reward_matrix[row, col] = -20.0
            else:
                reward_matrix[row, col] = -1.0

    return reward_matrix


def print_reward_matrix(reward_matrix: np.ndarray) -> None:
    print("\nReward matrix R(s):")
    print("Each value is the reward received when Jake enters that cell.\n")

    for row in range(ROWS):
        line = ""

        for col in range(COLS):
            line += f"{reward_matrix[row, col]:8.1f}"

        print(line)

    print()


# ============================================================
# Q-LEARNING FUNCTIONS
# ============================================================

def create_q_table() -> np.ndarray:
    """
    Q-table shape:
    rows x cols x actions

    q_table[row, col, action]
    """
    return np.zeros((ROWS, COLS, len(ACTION_BITS)), dtype=float)


def choose_action_epsilon_greedy(
    q_table: np.ndarray,
    valid_moves: np.ndarray,
    state: Position,
    epsilon: float
) -> Tuple[Optional[int], str]:

    valid_actions = get_valid_actions(valid_moves, state)

    if not valid_actions:
        return None, "NO_ACTION"

    if random.random() < epsilon:
        return random.choice(valid_actions), "EXPLORE"

    q_values = [q_table[state[0], state[1], action] for action in valid_actions]
    max_q = max(q_values)

    best_actions = [
        action for action in valid_actions
        if q_table[state[0], state[1], action] == max_q
    ]

    return random.choice(best_actions), "EXPLOIT"


def best_action(
    q_table: np.ndarray,
    valid_moves: np.ndarray,
    state: Position
) -> Optional[int]:

    valid_actions = get_valid_actions(valid_moves, state)

    if not valid_actions:
        return None

    q_values = [q_table[state[0], state[1], action] for action in valid_actions]
    max_q = max(q_values)

    best_actions = [
        action for action in valid_actions
        if q_table[state[0], state[1], action] == max_q
    ]

    return random.choice(best_actions)


def extract_learned_path(
    q_table: np.ndarray,
    valid_moves: np.ndarray,
    start: Position,
    goal: Position,
    max_steps: int = 100
) -> List[Tuple[str, Position]]:

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
            break

        next_state = move(state, action)
        path.append((ACTION_NAMES[action], next_state))
        state = next_state

    return path


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(filename: str):
    if os.path.exists(filename):
        return mpimg.imread(filename)
    return None


# ============================================================
# Q-TABLE VISUALIZATION
# ============================================================

def q_values_for_action(
    q_table: np.ndarray,
    valid_moves: np.ndarray,
    action_index: int
) -> np.ndarray:
    """
    Returns a 2D matrix with Q-values for one action.

    Invalid actions are represented as NaN.
    """
    q_matrix = np.full((ROWS, COLS), np.nan)

    for row in range(ROWS):
        for col in range(COLS):
            state = (row, col)
            valid_actions = get_valid_actions(valid_moves, state)

            if action_index in valid_actions:
                q_matrix[row, col] = q_table[row, col, action_index]

    return q_matrix


def draw_q_table_panel(
    q_axes,
    q_table: np.ndarray,
    valid_moves: np.ndarray,
    current_state: Position
) -> None:
    """
    Draws the Q-table as four matrices:
    Q(s, UP), Q(s, RIGHT), Q(s, DOWN), Q(s, LEFT).
    """

    action_titles = {
        0: f"Q(s, {ACTION_SYMBOLS[0]}) UP",
        1: f"Q(s, {ACTION_SYMBOLS[1]}) RIGHT",
        2: f"Q(s, {ACTION_SYMBOLS[2]}) DOWN",
        3: f"Q(s, {ACTION_SYMBOLS[3]}) LEFT",
    }

    valid_q_values = []

    for row in range(ROWS):
        for col in range(COLS):
            valid_actions = get_valid_actions(valid_moves, (row, col))
            for action in valid_actions:
                valid_q_values.append(q_table[row, col, action])

    if valid_q_values:
        vmin = min(valid_q_values)
        vmax = max(valid_q_values)

        if abs(vmax - vmin) < 1e-9:
            vmin -= 1
            vmax += 1
    else:
        vmin, vmax = -1, 1

    for action_index, ax in enumerate(q_axes):
        ax.clear()

        q_matrix = q_values_for_action(
            q_table=q_table,
            valid_moves=valid_moves,
            action_index=action_index
        )

        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="lightgray")

        ax.imshow(
            q_matrix,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )

        ax.set_title(action_titles[action_index], fontsize=10)

        for row in range(ROWS):
            for col in range(COLS):
                value = q_matrix[row, col]

                if np.isnan(value):
                    text = "X"
                    color = "black"
                else:
                    text = f"{value:.1f}"
                    color = "white" if value > (vmin + vmax) / 2 else "black"

                ax.text(
                    col,
                    row,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color
                )

        current_row, current_col = current_state

        rect = Rectangle(
            (current_col - 0.5, current_row - 0.5),
            1,
            1,
            fill=False,
            edgecolor="red",
            linewidth=2.5
        )
        ax.add_patch(rect)

        ax.set_xticks(range(COLS))
        ax.set_yticks(range(ROWS))
        ax.set_xticklabels(range(1, COLS + 1))
        ax.set_yticklabels(range(1, ROWS + 1))
        ax.tick_params(labelsize=7)


# ============================================================
# MAIN VISUALIZATION
# ============================================================

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


def draw_policy_arrows(ax, q_table: np.ndarray, valid_moves: np.ndarray, goal: Position) -> None:
    """
    Draws the currently learned best action for each cell.
    """
    for row in range(ROWS):
        for col in range(COLS):
            state = (row, col)

            if state == goal:
                continue

            valid_actions = get_valid_actions(valid_moves, state)

            if not valid_actions:
                continue

            q_values = [q_table[row, col, action] for action in valid_actions]
            max_q = max(q_values)

            if abs(max_q) < 1e-9:
                continue

            action = valid_actions[q_values.index(max_q)]
            dr, dc = action_to_delta(action)

            start_x = col + 0.5
            start_y = row + 0.5

            end_x = start_x + 0.25 * dc
            end_y = start_y + 0.25 * dr

            ax.annotate(
                "",
                xy=(end_x, end_y),
                xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle="->", color="gray", linewidth=1.8),
                zorder=5
            )


def draw_agent(ax, position: Position, jake_image=None) -> None:
    row, col = position

    if jake_image is not None:
        ax.imshow(
            jake_image,
            extent=(col + 0.15, col + 0.85, row + 0.85, row + 0.15),
            zorder=8
        )
    else:
        ax.add_patch(Circle((col + 0.5, row + 0.5), 0.30, color="royalblue", zorder=8))
        ax.text(
            col + 0.5,
            row + 0.5,
            "J",
            ha="center",
            va="center",
            color="white",
            fontsize=16,
            fontweight="bold",
            zorder=9
        )


def draw_goal(ax, goal: Position, finn_image=None) -> None:
    row, col = goal

    if finn_image is not None:
        ax.imshow(
            finn_image,
            extent=(col + 0.15, col + 0.85, row + 0.85, row + 0.15),
            zorder=7
        )
    else:
        ax.add_patch(Circle((col + 0.5, row + 0.5), 0.30, color="green", zorder=7))
        ax.text(
            col + 0.5,
            row + 0.5,
            "F",
            ha="center",
            va="center",
            color="white",
            fontsize=16,
            fontweight="bold",
            zorder=8
        )


def draw_training_frame(
    ax_maze,
    ax_stats,
    q_axes,
    content: np.ndarray,
    valid_moves: np.ndarray,
    q_table: np.ndarray,
    agent_position: Position,
    goal: Position,
    current_episode_path: Set[Position],
    last_transition: Optional[Tuple[Position, Position]],
    reward_history: List[float],
    metrics: Dict[str, float],
    jake_image=None,
    finn_image=None
) -> None:

    ax_maze.clear()
    ax_stats.clear()

    for row in range(ROWS):
        for col in range(COLS):
            pos = (row, col)

            if pos == goal:
                color = "#D2FFD2"
            elif pos == agent_position:
                color = "#FFFFFF"
            elif pos in current_episode_path:
                color = "#FFEB78"
            elif content[pos] == TRAP:
                color = "#FFB4B4"
            else:
                color = "white"

            rect = Rectangle(
                (col, row),
                1,
                1,
                facecolor=color,
                edgecolor="lightgray",
                linewidth=1,
                zorder=1
            )
            ax_maze.add_patch(rect)

    draw_policy_arrows(ax_maze, q_table, valid_moves, goal)

    if last_transition is not None:
        previous, current = last_transition

        px, py = previous[1] + 0.5, previous[0] + 0.5
        cx, cy = current[1] + 0.5, current[0] + 0.5

        if metrics["last_decision"] == "EXPLORE":
            arrow_color = "orange"
        else:
            arrow_color = "purple"

        ax_maze.annotate(
            "",
            xy=(cx, cy),
            xytext=(px, py),
            arrowprops=dict(arrowstyle="->", color=arrow_color, linewidth=3),
            zorder=6
        )

    draw_thick_walls(ax_maze, valid_moves)
    draw_goal(ax_maze, goal, finn_image)
    draw_agent(ax_maze, agent_position, jake_image)

    for col in range(COLS):
        ax_maze.text(
            col + 0.5,
            -0.35,
            str(col + 1),
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold"
        )

    for row in range(ROWS):
        ax_maze.text(
            -0.35,
            row + 0.5,
            str(row + 1),
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold"
        )

    ax_maze.set_title(
        f"Q-Learning exploration | Episode {int(metrics['episode'])}/{int(metrics['total_episodes'])}",
        fontsize=14
    )

    ax_maze.set_xlim(-0.6, COLS)
    ax_maze.set_ylim(ROWS + 0.2, -0.6)
    ax_maze.set_aspect("equal")
    ax_maze.axis("off")

    ax_stats.set_title("Training information", fontsize=14)

    if reward_history:
        ax_stats.plot(reward_history, linewidth=1.5)
        ax_stats.set_xlabel("Training round / episode")
        ax_stats.set_ylabel("Reward")
        ax_stats.grid(True, alpha=0.3)

    text = (
        f"Training rounds done: {int(metrics['episode'])}\n"
        f"Total rounds planned: {int(metrics['total_episodes'])}\n"
        f"Training steps done: {int(metrics['total_training_steps'])}\n\n"
        f"Current episode step: {int(metrics['episode_step'])}\n"
        f"Current episode reward: {metrics['episode_reward']:.1f}\n"
        f"Epsilon: {metrics['epsilon']:.3f}\n\n"
        f"Last decision: {metrics['last_decision']}\n"
        f"Explore count: {int(metrics['explore_count'])}\n"
        f"Exploit count: {int(metrics['exploit_count'])}\n"
        f"Success count: {int(metrics['success_count'])}"
    )

    ax_stats.text(
        0.02,
        0.98,
        text,
        transform=ax_stats.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    draw_q_table_panel(
        q_axes=q_axes,
        q_table=q_table,
        valid_moves=valid_moves,
        current_state=agent_position
    )


# ============================================================
# Q-LEARNING TRAINING
# ============================================================

def train_q_learning_visual(
    content: np.ndarray,
    valid_moves: np.ndarray,
    start: Position,
    goal: Position,
    ax_maze,
    ax_stats,
    q_axes,
    jake_image=None,
    finn_image=None,
    episodes: int = 300,
    max_steps_per_episode: int = 80,
    alpha: float = 0.2,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.985,
    visual_pause: float = 0.03
) -> Tuple[np.ndarray, List[float]]:

    q_table = create_q_table()

    epsilon = epsilon_start
    reward_history = []

    explore_count = 0
    exploit_count = 0
    success_count = 0
    total_training_steps = 0

    for episode in range(1, episodes + 1):
        state = start
        current_episode_path = {start}
        episode_reward = 0.0
        last_transition = None
        last_decision = "START"

        for episode_step in range(1, max_steps_per_episode + 1):
            action, decision = choose_action_epsilon_greedy(
                q_table=q_table,
                valid_moves=valid_moves,
                state=state,
                epsilon=epsilon
            )

            if action is None:
                break

            if decision == "EXPLORE":
                explore_count += 1
            else:
                exploit_count += 1

            next_state = move(state, action)
            reward = get_reward(content, next_state, goal)

            valid_next_actions = get_valid_actions(valid_moves, next_state)

            if valid_next_actions:
                max_next_q = max(
                    q_table[next_state[0], next_state[1], next_action]
                    for next_action in valid_next_actions
                )
            else:
                max_next_q = 0.0

            old_q = q_table[state[0], state[1], action]

            q_table[state[0], state[1], action] = old_q + alpha * (
                reward + gamma * max_next_q - old_q
            )

            previous_state = state
            state = next_state

            current_episode_path.add(state)
            episode_reward += reward
            total_training_steps += 1
            last_transition = (previous_state, state)
            last_decision = decision

            should_draw = episode <= 15 or episode % 10 == 0

            if should_draw:
                metrics = {
                    "episode": episode,
                    "total_episodes": episodes,
                    "episode_step": episode_step,
                    "episode_reward": episode_reward,
                    "epsilon": epsilon,
                    "last_decision": last_decision,
                    "explore_count": explore_count,
                    "exploit_count": exploit_count,
                    "success_count": success_count,
                    "total_training_steps": total_training_steps,
                }

                draw_training_frame(
                    ax_maze=ax_maze,
                    ax_stats=ax_stats,
                    q_axes=q_axes,
                    content=content,
                    valid_moves=valid_moves,
                    q_table=q_table,
                    agent_position=state,
                    goal=goal,
                    current_episode_path=current_episode_path,
                    last_transition=last_transition,
                    reward_history=reward_history,
                    metrics=metrics,
                    jake_image=jake_image,
                    finn_image=finn_image
                )

                plt.pause(visual_pause)

            if state == goal:
                success_count += 1
                break

        reward_history.append(episode_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    metrics = {
        "episode": episodes,
        "total_episodes": episodes,
        "episode_step": 0,
        "episode_reward": reward_history[-1] if reward_history else 0,
        "epsilon": epsilon,
        "last_decision": "TRAINING FINISHED",
        "explore_count": explore_count,
        "exploit_count": exploit_count,
        "success_count": success_count,
        "total_training_steps": total_training_steps,
    }

    draw_training_frame(
        ax_maze=ax_maze,
        ax_stats=ax_stats,
        q_axes=q_axes,
        content=content,
        valid_moves=valid_moves,
        q_table=q_table,
        agent_position=start,
        goal=goal,
        current_episode_path={start},
        last_transition=None,
        reward_history=reward_history,
        metrics=metrics,
        jake_image=jake_image,
        finn_image=finn_image
    )

    plt.pause(1.0)

    return q_table, reward_history


# ============================================================
# FINAL LEARNED POLICY EXECUTION
# ============================================================

def draw_final_execution(
    ax_maze,
    ax_stats,
    q_axes,
    content: np.ndarray,
    valid_moves: np.ndarray,
    q_table: np.ndarray,
    learned_path: List[Tuple[str, Position]],
    start: Position,
    goal: Position,
    reward_history: List[float],
    jake_image=None,
    finn_image=None
) -> None:

    visited = {start}
    agent_position = start

    for step_number, (action_name, next_position) in enumerate(learned_path, start=1):
        previous = agent_position
        agent_position = next_position
        visited.add(agent_position)

        metrics = {
            "episode": 0,
            "total_episodes": 0,
            "episode_step": step_number,
            "episode_reward": 0,
            "epsilon": 0,
            "last_decision": f"EXECUTING {action_name}",
            "explore_count": 0,
            "exploit_count": 0,
            "success_count": 0,
            "total_training_steps": 0,
        }

        draw_training_frame(
            ax_maze=ax_maze,
            ax_stats=ax_stats,
            q_axes=q_axes,
            content=content,
            valid_moves=valid_moves,
            q_table=q_table,
            agent_position=agent_position,
            goal=goal,
            current_episode_path=visited,
            last_transition=(previous, agent_position),
            reward_history=reward_history,
            metrics=metrics,
            jake_image=jake_image,
            finn_image=finn_image
        )

        ax_maze.set_title(
            f"Final learned policy | Step {step_number} | {action_name}",
            fontsize=14
        )

        plt.pause(0.7)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    random.seed(7)
    np.random.seed(7)

    content = CONTENT.copy()
    valid_moves = VALID_MOVES.copy()

    start = find_cell(content, JAKE)
    goal = find_cell(content, FINN)

    reward_matrix = create_reward_matrix(content, goal)
    print_reward_matrix(reward_matrix)

    jake_image = load_image("jake.png")
    finn_image = load_image("finn.png")

    if jake_image is None:
        print("jake.png not found. Drawing Jake as a blue circle.")

    if finn_image is None:
        print("finn.png not found. Drawing Finn as a green circle.")

    plt.ion()

    fig = plt.figure(figsize=(16, 8))

    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.3, 1.0, 1.0, 1.0],
        height_ratios=[1.0, 1.0]
    )

    ax_maze = fig.add_subplot(gs[:, 0])
    ax_stats = fig.add_subplot(gs[:, 1])

    ax_q_up = fig.add_subplot(gs[0, 2])
    ax_q_right = fig.add_subplot(gs[0, 3])
    ax_q_down = fig.add_subplot(gs[1, 2])
    ax_q_left = fig.add_subplot(gs[1, 3])

    q_axes = [ax_q_up, ax_q_right, ax_q_down, ax_q_left]

    print("Training Q-Learning agent...")

    q_table, reward_history = train_q_learning_visual(
        content=content,
        valid_moves=valid_moves,
        start=start,
        goal=goal,
        ax_maze=ax_maze,
        ax_stats=ax_stats,
        q_axes=q_axes,
        jake_image=jake_image,
        finn_image=finn_image,
        episodes=300,
        max_steps_per_episode=80,
        alpha=0.2,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.985,
        visual_pause=0.03
    )

    learned_path = extract_learned_path(
        q_table=q_table,
        valid_moves=valid_moves,
        start=start,
        goal=goal,
        max_steps=100
    )

    print("\nTraining finished.")
    print(f"Total training rounds: {len(reward_history)}")

    if not learned_path:
        print("No learned path found. Try increasing the number of episodes.")
        plt.ioff()
        plt.show()
        return

    print("\nLearned path:")
    for action_name, position in learned_path:
        print(f"{action_name} -> {position}")

    print("\nExecuting final learned policy...")

    draw_final_execution(
        ax_maze=ax_maze,
        ax_stats=ax_stats,
        q_axes=q_axes,
        content=content,
        valid_moves=valid_moves,
        q_table=q_table,
        learned_path=learned_path,
        start=start,
        goal=goal,
        reward_history=reward_history,
        jake_image=jake_image,
        finn_image=finn_image
    )

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()