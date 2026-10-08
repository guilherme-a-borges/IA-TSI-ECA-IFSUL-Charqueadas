import heapq
import os
import time
from collections import deque
from dataclasses import dataclass
from itertools import count
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle


# ============================================================
# Definir Valores no mapa
# ============================================================
EMPTY = 0
FINN = 1
JAKE = 2
TRAP = 3


# ============================================================
# Ações permitidas para o agente
# ============================================================
UP = 1
RIGHT = 2
DOWN = 4
LEFT = 8


# ============================================================
# MATRIZ 1: Definição do conteúdo do mapa
# ============================================================
CONTENT = np.array([
    [FINN,  EMPTY, EMPTY, EMPTY, JAKE],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    [EMPTY, EMPTY, TRAP,  EMPTY, EMPTY],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
])


# ============================================================
# Definição de posição e passos
# ============================================================
ROWS = 5
COLS = 5

Position = Tuple[int, int]  # row, col


@dataclass
class Step:
    action: str
    position: Position


# ============================================================
# Validação de movimentos e paredes
# ============================================================
def create_valid_moves() -> np.ndarray:
    moves = np.zeros((ROWS, COLS), dtype=int)

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

    block_passage(moves, 0, 1, RIGHT)
    block_passage(moves, 0, 2, RIGHT)

    block_passage(moves, 1, 0, RIGHT)
    block_passage(moves, 2, 0, RIGHT)
    block_passage(moves, 3, 0, RIGHT)

    block_passage(moves, 1, 1, RIGHT)
    block_passage(moves, 2, 1, RIGHT)

    block_passage(moves, 2, 2, RIGHT)
    block_passage(moves, 3, 3, RIGHT)

    block_passage(moves, 0, 0, DOWN)
    block_passage(moves, 1, 4, DOWN)
    block_passage(moves, 2, 3, DOWN)
    block_passage(moves, 3, 1, DOWN)
    block_passage(moves, 3, 2, DOWN)

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


def find_cell(content: np.ndarray, value: int) -> Position:
    positions = np.argwhere(content == value)
    if len(positions) == 0:
        raise ValueError(f"Value {value} not found in content matrix.")
    row, col = positions[0]
    return int(row), int(col)


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


def get_step_cost(content: np.ndarray, pos: Position) -> int:
    if content[pos] == TRAP:
        return 15
    return 1


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


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ============================================================
# ALGORITMOS DE BUSCA (Retornam: Caminho, Nós Explorados e Nome)
# ============================================================

def bfs_search(
    content: np.ndarray,
    valid_moves: np.ndarray,
    start: Position,
    goal: Position
) -> Tuple[List[Step], Set[Position], str]:
    name = "Busca em Largura (BFS)"
    if start == goal:
        return [], {start}, name

    frontier = deque([start])
    visited: Set[Position] = {start}
    came_from: Dict[Position, Position] = {}
    action_from: Dict[Position, str] = {}

    while frontier:
        current = frontier.popleft()

        if current == goal:
            return reconstruct_path(came_from, action_from, current), visited, name

        for step in get_neighbors(valid_moves, current):
            next_pos = step.position
            if next_pos not in visited:
                visited.add(next_pos)
                frontier.append(next_pos)
                came_from[next_pos] = current
                action_from[next_pos] = step.action

    return [], visited, name


def uniform_cost_search(
    content: np.ndarray,
    valid_moves: np.ndarray,
    start: Position,
    goal: Position
) -> Tuple[List[Step], Set[Position], str]:
    name = "Custo Uniforme (UCS / Dijkstra)"
    frontier = []
    tie_breaker = count()

    heapq.heappush(frontier, (0, next(tie_breaker), start))
    came_from: Dict[Position, Position] = {}
    action_from: Dict[Position, str] = {}
    cost_so_far: Dict[Position, int] = {start: 0}
    explored: Set[Position] = set()

    while frontier:
        current_cost, _, current = heapq.heappop(frontier)
        explored.add(current)

        if current == goal:
            return reconstruct_path(came_from, action_from, current), explored, name

        if current_cost > cost_so_far[current]:
            continue

        for step in get_neighbors(valid_moves, current):
            next_pos = step.position
            new_cost = cost_so_far[current] + get_step_cost(content, next_pos)

            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost
                heapq.heappush(frontier, (new_cost, next(tie_breaker), next_pos))
                came_from[next_pos] = current
                action_from[next_pos] = step.action

    return [], explored, name


def dfs_search(
    content: np.ndarray,
    valid_moves: np.ndarray,
    start: Position,
    goal: Position
) -> Tuple[List[Step], Set[Position], str]:
    name = "Busca em Profundidade (DFS)"
    if start == goal:
        return [], {start}, name

    frontier = [start]
    visited: Set[Position] = {start}
    came_from: Dict[Position, Position] = {}
    action_from: Dict[Position, str] = {}

    while frontier:
        current = frontier.pop()

        if current == goal:
            return reconstruct_path(came_from, action_from, current), visited, name

        for step in get_neighbors(valid_moves, current):
            next_pos = step.position
            if next_pos not in visited:
                visited.add(next_pos)
                frontier.append(next_pos)
                came_from[next_pos] = current
                action_from[next_pos] = step.action

    return [], visited, name


def depth_limited_search(
    content: np.ndarray,
    valid_moves: np.ndarray,
    start: Position,
    goal: Position,
    limit: int = 12
) -> Tuple[List[Step], Set[Position], str]:
    name = f"Busca Limitada em Profundidade (DLS | Limite={limit})"
    if start == goal:
        return [], {start}, name

    frontier = [(start, 0)]
    depth_reached: Dict[Position, int] = {start: 0}
    came_from: Dict[Position, Position] = {}
    action_from: Dict[Position, str] = {}
    explored: Set[Position] = {start}

    while frontier:
        current, depth = frontier.pop()
        explored.add(current)

        if current == goal:
            return reconstruct_path(came_from, action_from, current), explored, name

        if depth < limit:
            for step in get_neighbors(valid_moves, current):
                next_pos = step.position
                next_depth = depth + 1

                if next_pos not in depth_reached or next_depth < depth_reached[next_pos]:
                    depth_reached[next_pos] = next_depth
                    frontier.append((next_pos, next_depth))
                    came_from[next_pos] = current
                    action_from[next_pos] = step.action

    return [], explored, name


def iterative_deepening_search(
    content: np.ndarray,
    valid_moves: np.ndarray,
    start: Position,
    goal: Position,
    max_depth: int = 25
) -> Tuple[List[Step], Set[Position], str]:
    name = "Aprofundamento Iterativo (IDDFS)"
    if start == goal:
        return [], {start}, name

    all_explored: Set[Position] = set()

    for depth in range(max_depth + 1):
        path_set = set()

        def dls(current: Position, current_depth: int) -> Optional[List[Step]]:
            all_explored.add(current)
            if current == goal:
                return []
            if current_depth == depth:
                return None

            path_set.add(current)
            for step in get_neighbors(valid_moves, current):
                next_pos = step.position
                if next_pos not in path_set:
                    res = dls(next_pos, current_depth + 1)
                    if res is not None:
                        return [step] + res
            path_set.remove(current)
            return None

        result = dls(start, 0)
        if result is not None:
            return result, all_explored, name

    return [], all_explored, name


def greedy_best_first_search(
    content: np.ndarray,
    valid_moves: np.ndarray,
    start: Position,
    goal: Position
) -> Tuple[List[Step], Set[Position], str]:
    name = "Busca Gulosa (Greedy Best-First | f = h)"
    frontier = []
    tie_breaker = count()

    initial_priority = manhattan(start, goal)
    heapq.heappush(frontier, (initial_priority, next(tie_breaker), start))

    came_from: Dict[Position, Position] = {}
    action_from: Dict[Position, str] = {}
    visited: Set[Position] = {start}

    while frontier:
        _, _, current = heapq.heappop(frontier)

        if current == goal:
            return reconstruct_path(came_from, action_from, current), visited, name

        for step in get_neighbors(valid_moves, current):
            next_pos = step.position
            if next_pos not in visited:
                visited.add(next_pos)
                priority = manhattan(next_pos, goal)
                heapq.heappush(frontier, (priority, next(tie_breaker), next_pos))
                came_from[next_pos] = current
                action_from[next_pos] = step.action

    return [], visited, name


def astar_search(
    content: np.ndarray,
    valid_moves: np.ndarray,
    start: Position,
    goal: Position
) -> Tuple[List[Step], Set[Position], str]:
    name = "Busca A* (A-Star | f = g + h)"
    frontier = []
    tie_breaker = count()

    heapq.heappush(frontier, (0, next(tie_breaker), start))

    came_from: Dict[Position, Position] = {}
    action_from: Dict[Position, str] = {}
    cost_so_far: Dict[Position, int] = {start: 0}
    explored: Set[Position] = set()

    while frontier:
        _, _, current = heapq.heappop(frontier)
        explored.add(current)

        if current == goal:
            return reconstruct_path(came_from, action_from, current), explored, name

        for step in get_neighbors(valid_moves, current):
            next_pos = step.position
            new_cost = cost_so_far[current] + get_step_cost(content, next_pos)

            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost
                priority = new_cost + manhattan(next_pos, goal)
                heapq.heappush(frontier, (priority, next(tie_breaker), next_pos))
                came_from[next_pos] = current
                action_from[next_pos] = step.action

    return [], explored, name


# ============================================================
# DESENHO DO MAPA
# ============================================================
def load_image(filename: str):
    if os.path.exists(filename):
        return mpimg.imread(filename)
    return None


def get_cell_background(
    pos: Position,
    goal: Position,
    path_visited: Set[Position],
    planned_path: Set[Position],
    explored_nodes: Set[Position],
    content: np.ndarray
) -> str:
    if pos == goal:
        return "#D2FFD2"  # Verde Claro (Objetivo)
    if pos in path_visited:
        return "#FFEB78"  # Amarelo (Trilha percorrida pela animação)
    if pos in planned_path:
        return "#B4E1FF"  # Azul Claro (Caminho Solução)
    if content[pos] == TRAP:
        return "#FFB4B4"  # Vermelho Claro (Armadilha)
    if pos in explored_nodes:
        return "#E2D0F9"  # Lilás / Roxo Claro (Estados Expandidos)

    return "white"


def draw_maze(
    ax,
    content: np.ndarray,
    valid_moves: np.ndarray,
    agent_position: Position,
    goal: Position,
    path_visited: Set[Position],
    planned_path: Set[Position],
    explored_nodes: Set[Position],
    step_number: int,
    current_action: str,
    accumulated_cost: int,
    algorithm_name: str,
    jake_image=None,
    finn_image=None
) -> None:
    ax.clear()
    rows, cols = content.shape

    for row in range(rows):
        for col in range(cols):
            pos = (row, col)
            color = get_cell_background(
                pos=pos,
                goal=goal,
                path_visited=path_visited,
                planned_path=planned_path,
                explored_nodes=explored_nodes,
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

    goal_row, goal_col = goal
    if finn_image is not None:
        ax.imshow(
            finn_image,
            extent=(goal_col + 0.15, goal_col + 0.85, goal_row + 0.85, goal_row + 0.15),
            zorder=5
        )
    else:
        ax.add_patch(Circle((goal_col + 0.5, goal_row + 0.5), 0.30, color="green", zorder=5))
        ax.text(goal_col + 0.5, goal_row + 0.5, "F", ha="center", va="center", color="white", fontsize=16, fontweight="bold", zorder=6)

    agent_row, agent_col = agent_position
    if jake_image is not None:
        ax.imshow(
            jake_image,
            extent=(agent_col + 0.15, agent_col + 0.85, agent_row + 0.85, agent_row + 0.15),
            zorder=6
        )
    else:
        ax.add_patch(Circle((agent_col + 0.5, agent_row + 0.5), 0.30, color="royalblue", zorder=6))
        ax.text(agent_col + 0.5, agent_row + 0.5, "J", ha="center", va="center", color="white", fontsize=16, fontweight="bold", zorder=7)

    for col in range(cols):
        ax.text(col + 0.5, -0.35, str(col + 1), ha="center", va="center", fontsize=18, fontweight="bold")
    for row in range(rows):
        ax.text(-0.35, row + 0.5, str(row + 1), ha="center", va="center", fontsize=18, fontweight="bold")

    title = f"Algoritmo: {algorithm_name}\n"
    title += f"Passo: {step_number} | Ação: {current_action} | Custo: {accumulated_cost} | Nós Expandidos: {len(explored_nodes)}"
    ax.set_title(title, fontsize=11, fontweight="bold")

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

    # ========================================================
    # BASTA DESCOMENTAR QUALQUER LINHA:
    # ========================================================
    # solution, explored, algorithm_name = bfs_search(content, valid_moves, start, goal)
    # solution, explored, algorithm_name = uniform_cost_search(content, valid_moves, start, goal)
    # solution, explored, algorithm_name = dfs_search(content, valid_moves, start, goal)
    # solution, explored, algorithm_name = depth_limited_search(content, valid_moves, start, goal, limit=12)
    # solution, explored, algorithm_name = iterative_deepening_search(content, valid_moves, start, goal)
    solution, explored, algorithm_name = greedy_best_first_search(content, valid_moves, start, goal)
    #solution, explored, algorithm_name = astar_search(content, valid_moves, start, goal)

    if not solution:
        print(f"[{algorithm_name}] Caminho não encontrado.")
        return

    planned_path = {step.position for step in solution}
    path_visited = {start}

    jake_image = load_image("jake.png")
    finn_image = load_image("finn.png")

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    agent_position = start
    accumulated_cost = 0

    draw_maze(
        ax=ax,
        content=content,
        valid_moves=valid_moves,
        agent_position=agent_position,
        goal=goal,
        path_visited=path_visited,
        planned_path=planned_path,
        explored_nodes=explored,
        step_number=0,
        current_action="Inicio",
        accumulated_cost=accumulated_cost,
        algorithm_name=algorithm_name,
        jake_image=jake_image,
        finn_image=finn_image
    )

    plt.pause(0.8)

    for step_number, step in enumerate(solution, start=1):
        agent_position = step.position
        path_visited.add(agent_position)
        accumulated_cost += get_step_cost(content, agent_position)

        draw_maze(
            ax=ax,
            content=content,
            valid_moves=valid_moves,
            agent_position=agent_position,
            goal=goal,
            path_visited=path_visited,
            planned_path=planned_path,
            explored_nodes=explored,
            step_number=step_number,
            current_action=step.action,
            accumulated_cost=accumulated_cost,
            algorithm_name=algorithm_name,
            jake_image=jake_image,
            finn_image=finn_image
        )

        plt.pause(0.7)
        time.sleep(0.1)

    plt.ioff()
    plt.show()

    print("--- RESULTADOS ---")
    print(f"Algoritmo: {algorithm_name}")
    print(f"Total de Nós Expandidos: {len(explored)}")
    print(f"Número de Passos no Caminho Final: {len(solution)}")
    print(f"Custo Total do Caminho: {accumulated_cost}")


if __name__ == "__main__":
    main()