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
ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]
ACTION_SYMBOLS = ["↑", "→", "↓", "←"]

ROWS = 5
COLS = 5
ACTION_COUNT = 4

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
# REWARD SYSTEM
# ============================================================

def print_base_reward_matrix(content: np.ndarray) -> None:
    print("\nDynamic multi-agent reward system:")
    print("+100  if Jake and Finn meet or cross paths")
    print("-1    per movement step")
    print("+2    for each Manhattan-distance unit reduced")
    print("-2    for each Manhattan-distance unit increased")
    print("-20   if the agent enters a trap cell\n")

    print("Base cell reward matrix:")
    print("Normal cell = -1, trap = -20")
    print("The +100 goal reward is dynamic because each agent is the goal of the other.\n")

    base_reward = np.full((ROWS, COLS), -1.0)
    base_reward[content == TRAP] = -20.0

    for row in range(ROWS):
        print("".join(f"{base_reward[row, col]:8.1f}" for col in range(COLS)))
    print()


def reward_for_agent(
    content: np.ndarray,
    old_own: Position,
    old_target: Position,
    new_own: Position,
    new_target: Position,
    success: bool
) -> float:
    if success:
        return 100.0

    reward = -1.0

    old_distance = manhattan(old_own, old_target)
    new_distance = manhattan(new_own, new_target)

    reward += 2.0 * (old_distance - new_distance)

    if content[new_own] == TRAP:
        reward -= 20.0

    return reward


# ============================================================
# Q-LEARNING FUNCTIONS
# ============================================================

def create_q_table() -> np.ndarray:
    """
    Dynamic-goal Q-table:

    Q[own_row][own_col][target_row][target_col][action]

    The state is:
    my current position + the other agent's current position.
    """
    return np.zeros((ROWS, COLS, ROWS, COLS, ACTION_COUNT), dtype=float)


def best_action(
    q_table: np.ndarray,
    valid_moves: np.ndarray,
    own: Position,
    target: Position,
    rng: random.Random
) -> Optional[int]:
    valid_actions = get_valid_actions(valid_moves, own)

    if not valid_actions:
        return None

    values = [
        q_table[own[0], own[1], target[0], target[1], action]
        for action in valid_actions
    ]

    max_value = max(values)

    best_actions = [
        action for action in valid_actions
        if abs(q_table[own[0], own[1], target[0], target[1], action] - max_value) < 1e-9
    ]

    return rng.choice(best_actions)


def choose_action(
    q_table: np.ndarray,
    valid_moves: np.ndarray,
    own: Position,
    target: Position,
    epsilon: float,
    rng: random.Random
) -> Tuple[Optional[int], str]:
    valid_actions = get_valid_actions(valid_moves, own)

    if not valid_actions:
        return None, "NO_ACTION"

    if rng.random() < epsilon:
        return rng.choice(valid_actions), "EXPLORE"

    action = best_action(q_table, valid_moves, own, target, rng)
    return action, "EXPLOIT"


def max_next_q(
    q_table: np.ndarray,
    valid_moves: np.ndarray,
    own: Position,
    target: Position
) -> float:
    valid_actions = get_valid_actions(valid_moves, own)

    if not valid_actions:
        return 0.0

    return max(
        q_table[own[0], own[1], target[0], target[1], action]
        for action in valid_actions
    )


def update_q(
    q_table: np.ndarray,
    valid_moves: np.ndarray,
    old_own: Position,
    old_target: Position,
    action: int,
    new_own: Position,
    new_target: Position,
    reward: float,
    alpha: float,
    gamma: float,
    terminal: bool
) -> None:
    old_q = q_table[old_own[0], old_own[1], old_target[0], old_target[1], action]
    next_q = 0.0 if terminal else max_next_q(q_table, valid_moves, new_own, new_target)

    q_table[old_own[0], old_own[1], old_target[0], old_target[1], action] = (
        old_q + alpha * (reward + gamma * next_q - old_q)
    )


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
    phase: str,
    step_or_episode: int,
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

    if phase == "TRAINING":
        title = f"Multi-agent Q-Learning | Training episode {step_or_episode}"
    else:
        title = f"Simultaneous multi-agent execution | Step {step_or_episode}"

    ax.set_title(title, fontsize=14)
    ax.set_xlim(-0.6, COLS)
    ax.set_ylim(ROWS + 0.2, -0.6)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_reward_chart(ax, reward_history: List[float]) -> None:
    ax.clear()
    ax.set_title("Combined reward by episode", fontsize=12)

    if reward_history:
        ax.plot(reward_history, linewidth=1.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No reward history yet", ha="center", va="center")

    ax.tick_params(labelsize=8)


def draw_stats(
    ax,
    phase: str,
    episode: int,
    episodes_planned: int,
    episode_step: int,
    execution_step: int,
    reward_history: List[float],
    epsilon: float,
    success_count: int,
    total_training_steps: int,
    jake: Position,
    finn: Position,
    last_jake_action: str,
    last_finn_action: str,
    last_jake_decision: str,
    last_finn_decision: str,
    explore_count_jake: int,
    exploit_count_jake: int,
    explore_count_finn: int,
    exploit_count_finn: int,
    status_text: str
) -> None:
    ax.clear()
    ax.axis("off")

    text = (
        f"Phase: {phase}\n"
        f"Status: {status_text}\n\n"
        f"Training rounds done: {len(reward_history)}\n"
        f"Total training rounds: {episodes_planned}\n"
        f"Training steps done: {total_training_steps}\n"
        f"Success count: {success_count}\n"
        f"Epsilon: {epsilon:.3f}\n\n"
        f"Current training episode: {episode}\n"
        f"Current episode step: {episode_step}\n"
        f"Execution step: {execution_step}\n\n"
        f"Jake position: ({jake[0] + 1}, {jake[1] + 1})\n"
        f"Finn position: ({finn[0] + 1}, {finn[1] + 1})\n"
        f"Distance Jake-Finn: {manhattan(jake, finn)}\n\n"
        f"Jake action: {last_jake_action} / {last_jake_decision}\n"
        f"Finn action: {last_finn_action} / {last_finn_decision}\n\n"
        f"Jake explore/exploit: {explore_count_jake}/{exploit_count_jake}\n"
        f"Finn explore/exploit: {explore_count_finn}/{exploit_count_finn}\n\n"
        "Goal condition:\n"
        "same cell or crossing paths"
    )

    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )


def draw_current_q_values(
    ax,
    q_jake: np.ndarray,
    q_finn: np.ndarray,
    valid_moves: np.ndarray,
    jake: Position,
    finn: Position
) -> None:
    ax.clear()
    ax.axis("off")
    ax.set_title("Current dynamic Q-values", fontsize=12)

    lines = []

    lines.append("Jake: Q(own=Jake, target=Finn, action)")
    for action in range(ACTION_COUNT):
        valid = action in get_valid_actions(valid_moves, jake)
        value = q_jake[jake[0], jake[1], finn[0], finn[1], action]
        value_text = "X" if not valid else f"{value:8.2f}"
        lines.append(f"  {ACTION_SYMBOLS[action]} {ACTION_NAMES[action]:>5}: {value_text}")

    lines.append("")
    lines.append("Finn: Q(own=Finn, target=Jake, action)")
    for action in range(ACTION_COUNT):
        valid = action in get_valid_actions(valid_moves, finn)
        value = q_finn[finn[0], finn[1], jake[0], jake[1], action]
        value_text = "X" if not valid else f"{value:8.2f}"
        lines.append(f"  {ACTION_SYMBOLS[action]} {ACTION_NAMES[action]:>5}: {value_text}")

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
    ax_reward,
    ax_q,
    content: np.ndarray,
    valid_moves: np.ndarray,
    q_jake: np.ndarray,
    q_finn: np.ndarray,
    reward_history: List[float],
    jake: Position,
    finn: Position,
    previous_jake: Optional[Position],
    previous_finn: Optional[Position],
    jake_visited: Set[Position],
    finn_visited: Set[Position],
    phase: str,
    step_or_episode: int,
    episode: int,
    episodes_planned: int,
    episode_step: int,
    execution_step: int,
    epsilon: float,
    success_count: int,
    total_training_steps: int,
    last_jake_action: str,
    last_finn_action: str,
    last_jake_decision: str,
    last_finn_decision: str,
    explore_count_jake: int,
    exploit_count_jake: int,
    explore_count_finn: int,
    exploit_count_finn: int,
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
        phase=phase,
        step_or_episode=step_or_episode,
        jake_image=jake_image,
        finn_image=finn_image
    )

    draw_stats(
        ax=ax_stats,
        phase=phase,
        episode=episode,
        episodes_planned=episodes_planned,
        episode_step=episode_step,
        execution_step=execution_step,
        reward_history=reward_history,
        epsilon=epsilon,
        success_count=success_count,
        total_training_steps=total_training_steps,
        jake=jake,
        finn=finn,
        last_jake_action=last_jake_action,
        last_finn_action=last_finn_action,
        last_jake_decision=last_jake_decision,
        last_finn_decision=last_finn_decision,
        explore_count_jake=explore_count_jake,
        exploit_count_jake=exploit_count_jake,
        explore_count_finn=explore_count_finn,
        exploit_count_finn=exploit_count_finn,
        status_text=status_text
    )

    draw_reward_chart(ax_reward, reward_history)
    draw_current_q_values(ax_q, q_jake, q_finn, valid_moves, jake, finn)


# ============================================================
# TRAINING
# ============================================================

def random_cell(rng: random.Random) -> Position:
    return rng.randrange(ROWS), rng.randrange(COLS)


def random_cell_different_from(rng: random.Random, other: Position) -> Position:
    candidate = random_cell(rng)

    while candidate == other:
        candidate = random_cell(rng)

    return candidate


def train_two_agents(
    content: np.ndarray,
    valid_moves: np.ndarray,
    ax_maze,
    ax_stats,
    ax_reward,
    ax_q,
    jake_image=None,
    finn_image=None,
    episodes_planned: int = 1500,
    max_steps_per_episode: int = 80,
    alpha: float = 0.20,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.990,
    visual_pause: float = 0.02
):
    rng = random.Random(7)

    q_jake = create_q_table()
    q_finn = create_q_table()

    epsilon = epsilon_start
    reward_history: List[float] = []

    explore_count_jake = 0
    exploit_count_jake = 0
    explore_count_finn = 0
    exploit_count_finn = 0
    success_count = 0
    total_training_steps = 0

    for episode in range(1, episodes_planned + 1):
        jake = random_cell(rng)
        finn = random_cell_different_from(rng, jake)

        previous_jake = None
        previous_finn = None

        jake_visited = {jake}
        finn_visited = {finn}

        episode_reward = 0.0
        last_jake_decision = "START"
        last_finn_decision = "START"
        last_jake_action = "-"
        last_finn_action = "-"

        for episode_step in range(1, max_steps_per_episode + 1):
            old_jake = jake
            old_finn = finn

            jake_action, jake_decision = choose_action(
                q_jake, valid_moves, old_jake, old_finn, epsilon, rng
            )

            finn_action, finn_decision = choose_action(
                q_finn, valid_moves, old_finn, old_jake, epsilon, rng
            )

            if jake_action is None or finn_action is None:
                break

            new_jake = move_agent(old_jake, jake_action)
            new_finn = move_agent(old_finn, finn_action)

            success = found_each_other(old_jake, old_finn, new_jake, new_finn)

            reward_jake = reward_for_agent(
                content, old_jake, old_finn, new_jake, new_finn, success
            )

            reward_finn = reward_for_agent(
                content, old_finn, old_jake, new_finn, new_jake, success
            )

            update_q(
                q_table=q_jake,
                valid_moves=valid_moves,
                old_own=old_jake,
                old_target=old_finn,
                action=jake_action,
                new_own=new_jake,
                new_target=new_finn,
                reward=reward_jake,
                alpha=alpha,
                gamma=gamma,
                terminal=success
            )

            update_q(
                q_table=q_finn,
                valid_moves=valid_moves,
                old_own=old_finn,
                old_target=old_jake,
                action=finn_action,
                new_own=new_finn,
                new_target=new_jake,
                reward=reward_finn,
                alpha=alpha,
                gamma=gamma,
                terminal=success
            )

            if jake_decision == "EXPLORE":
                explore_count_jake += 1
            else:
                exploit_count_jake += 1

            if finn_decision == "EXPLORE":
                explore_count_finn += 1
            else:
                exploit_count_finn += 1

            previous_jake = old_jake
            previous_finn = old_finn

            jake = new_jake
            finn = new_finn

            jake_visited.add(jake)
            finn_visited.add(finn)

            episode_reward += reward_jake + reward_finn
            total_training_steps += 1

            last_jake_decision = jake_decision
            last_finn_decision = finn_decision
            last_jake_action = ACTION_NAMES[jake_action]
            last_finn_action = ACTION_NAMES[finn_action]

            should_draw = episode <= 15 or episode % 20 == 0

            if should_draw:
                draw_frame(
                    ax_maze=ax_maze,
                    ax_stats=ax_stats,
                    ax_reward=ax_reward,
                    ax_q=ax_q,
                    content=content,
                    valid_moves=valid_moves,
                    q_jake=q_jake,
                    q_finn=q_finn,
                    reward_history=reward_history,
                    jake=jake,
                    finn=finn,
                    previous_jake=previous_jake,
                    previous_finn=previous_finn,
                    jake_visited=jake_visited,
                    finn_visited=finn_visited,
                    phase="TRAINING",
                    step_or_episode=episode,
                    episode=episode,
                    episodes_planned=episodes_planned,
                    episode_step=episode_step,
                    execution_step=0,
                    epsilon=epsilon,
                    success_count=success_count,
                    total_training_steps=total_training_steps,
                    last_jake_action=last_jake_action,
                    last_finn_action=last_finn_action,
                    last_jake_decision=last_jake_decision,
                    last_finn_decision=last_finn_decision,
                    explore_count_jake=explore_count_jake,
                    exploit_count_jake=exploit_count_jake,
                    explore_count_finn=explore_count_finn,
                    exploit_count_finn=exploit_count_finn,
                    status_text="Training",
                    jake_image=jake_image,
                    finn_image=finn_image
                )
                plt.pause(visual_pause)

            if success:
                success_count += 1
                break

        reward_history.append(episode_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print("\nTraining finished.")
    print(f"Training rounds: {len(reward_history)}")
    print(f"Success count: {success_count}")
    print(f"Jake explore count: {explore_count_jake}")
    print(f"Jake exploit count: {exploit_count_jake}")
    print(f"Finn explore count: {explore_count_finn}")
    print(f"Finn exploit count: {exploit_count_finn}")

    return {
        "q_jake": q_jake,
        "q_finn": q_finn,
        "reward_history": reward_history,
        "epsilon": epsilon,
        "success_count": success_count,
        "total_training_steps": total_training_steps,
        "explore_count_jake": explore_count_jake,
        "exploit_count_jake": exploit_count_jake,
        "explore_count_finn": explore_count_finn,
        "exploit_count_finn": exploit_count_finn,
        "episodes_planned": episodes_planned,
    }


# ============================================================
# FINAL EXECUTION
# ============================================================

def execute_two_agents(
    content: np.ndarray,
    valid_moves: np.ndarray,
    training_result: Dict,
    ax_maze,
    ax_stats,
    ax_reward,
    ax_q,
    jake_image=None,
    finn_image=None,
    max_execution_steps: int = 80
) -> None:
    rng = random.Random(10)

    q_jake = training_result["q_jake"]
    q_finn = training_result["q_finn"]

    jake = find_cell(content, JAKE)
    finn = find_cell(content, FINN)

    previous_jake = None
    previous_finn = None

    jake_visited = {jake}
    finn_visited = {finn}

    last_jake_action = "-"
    last_finn_action = "-"
    last_jake_decision = "LEARNED POLICY"
    last_finn_decision = "LEARNED POLICY"

    for execution_step in range(1, max_execution_steps + 1):
        old_jake = jake
        old_finn = finn

        jake_action = best_action(q_jake, valid_moves, old_jake, old_finn, rng)
        finn_action = best_action(q_finn, valid_moves, old_finn, old_jake, rng)

        if jake_action is None or finn_action is None:
            status = "No valid action available."
            break

        new_jake = move_agent(old_jake, jake_action)
        new_finn = move_agent(old_finn, finn_action)

        previous_jake = old_jake
        previous_finn = old_finn

        jake = new_jake
        finn = new_finn

        jake_visited.add(jake)
        finn_visited.add(finn)

        last_jake_action = ACTION_NAMES[jake_action]
        last_finn_action = ACTION_NAMES[finn_action]

        success = found_each_other(old_jake, old_finn, new_jake, new_finn)

        if success:
            status = "Jake and Finn found each other."
        else:
            status = "Both agents are moving simultaneously."

        draw_frame(
            ax_maze=ax_maze,
            ax_stats=ax_stats,
            ax_reward=ax_reward,
            ax_q=ax_q,
            content=content,
            valid_moves=valid_moves,
            q_jake=q_jake,
            q_finn=q_finn,
            reward_history=training_result["reward_history"],
            jake=jake,
            finn=finn,
            previous_jake=previous_jake,
            previous_finn=previous_finn,
            jake_visited=jake_visited,
            finn_visited=finn_visited,
            phase="EXECUTION",
            step_or_episode=execution_step,
            episode=training_result["episodes_planned"],
            episodes_planned=training_result["episodes_planned"],
            episode_step=0,
            execution_step=execution_step,
            epsilon=training_result["epsilon"],
            success_count=training_result["success_count"],
            total_training_steps=training_result["total_training_steps"],
            last_jake_action=last_jake_action,
            last_finn_action=last_finn_action,
            last_jake_decision=last_jake_decision,
            last_finn_decision=last_finn_decision,
            explore_count_jake=training_result["explore_count_jake"],
            exploit_count_jake=training_result["exploit_count_jake"],
            explore_count_finn=training_result["explore_count_finn"],
            exploit_count_finn=training_result["exploit_count_finn"],
            status_text=status,
            jake_image=jake_image,
            finn_image=finn_image
        )

        plt.pause(0.7)

        if success:
            return


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print_base_reward_matrix(CONTENT)

    jake_image = load_image("jake.png")
    finn_image = load_image("finn.png")

    if jake_image is None:
        print("jake.png not found. Drawing Jake as a blue circle.")

    if finn_image is None:
        print("finn.png not found. Drawing Finn as an orange circle.")

    plt.ion()

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.35, 1.0, 0.9],
        height_ratios=[1.0, 1.0]
    )

    ax_maze = fig.add_subplot(gs[:, 0])
    ax_stats = fig.add_subplot(gs[0, 1])
    ax_reward = fig.add_subplot(gs[1, 1])
    ax_q = fig.add_subplot(gs[:, 2])

    training_result = train_two_agents(
        content=CONTENT.copy(),
        valid_moves=VALID_MOVES.copy(),
        ax_maze=ax_maze,
        ax_stats=ax_stats,
        ax_reward=ax_reward,
        ax_q=ax_q,
        jake_image=jake_image,
        finn_image=finn_image,
        episodes_planned=700,
        max_steps_per_episode=80,
        alpha=0.20,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.990,
        visual_pause=0.02
    )

    execute_two_agents(
        content=CONTENT.copy(),
        valid_moves=VALID_MOVES.copy(),
        training_result=training_result,
        ax_maze=ax_maze,
        ax_stats=ax_stats,
        ax_reward=ax_reward,
        ax_q=ax_q,
        jake_image=jake_image,
        finn_image=finn_image,
        max_execution_steps=80
    )

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()