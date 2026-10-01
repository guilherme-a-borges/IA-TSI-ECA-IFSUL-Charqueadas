import heapq
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


# ============================================================
# Definir o mapa do labirinto (ambiente) e sua legenda
# ============================================================
# 0 = free path (caminho livre)
# 1 = wall (parede)
# 2 = agent (agente)
# 3 = goal (objetivo)
# 4 = trap / special cell (armadilha / célula especial)
# 5 = visited path (caminho visitado)
# 6 = planned path (caminho planejado)

# Labirinto de exemplo
MAZE = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 0, 0, 1, 0, 0, 3, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 4, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
])

# ============================================================
# Definir estrutura de dado para posição 2D
# ============================================================

# Type alias para Position (linha, coluna), o qual é usado para representacar as posições do Agente, objetivos e objetos. Em Java, isso seria uma nova classe ou record.
Position = Tuple[int, int]  

# ============================================================
# Ações do agente
# ============================================================

# ações do agente: (delta_linha, delta_coluna)
ACTIONS = {
    "UP":    (-1, 0), # ir para cima
    "DOWN":  (1, 0),  # ir para baixo
    "LEFT":  (0, -1), # ir para a esquerda
    "RIGHT": (0, 1),  # ir para a direita
}


# ============================================================
# Classes e funções para representar o agente
# ============================================================


# uma classe para representar o agente
@dataclass
class Agent:
    position: Position
    goal: Position
    path_taken: List[Position]

# função para encontrar a posição de um valor específico no labirinto
def find_cell(maze: np.ndarray, value: int) -> Position:
    positions = np.argwhere(maze == value)
    if len(positions) == 0:
        raise ValueError(f"Cell with value {value} not found in maze.")
    row, col = positions[0]
    return int(row), int(col)

# função para calcular a distância de Manhattan entre duas posições, usada como heurística no A*
def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def is_valid_position(maze: np.ndarray, pos: Position) -> bool:
    row, col = pos

    if row < 0 or row >= maze.shape[0]:
        return False
    if col < 0 or col >= maze.shape[1]:
        return False

    # Wall is not walkable
    if maze[row, col] == 1:
        return False

    return True


# função para obter os vizinhos válidos de uma posição no labirinto
def get_neighbors(maze: np.ndarray, pos: Position) -> List[Tuple[str, Position]]:
    neighbors = []

    for action_name, (dr, dc) in ACTIONS.items():
        new_pos = (pos[0] + dr, pos[1] + dc)

        if is_valid_position(maze, new_pos):
            neighbors.append((action_name, new_pos))

    return neighbors

# função para reconstruir o caminho a partir do dicionário de "came_from" gerado pelo A*
def reconstruct_path(
    came_from: Dict[Position, Tuple[Position, str]],
    current: Position
) -> List[Tuple[str, Position]]:
    path = []

    while current in came_from:
        previous, action = came_from[current]
        path.append((action, current))
        current = previous

    path.reverse()
    return path

# ============================================================
# Função usada para  fazer planejamento
# ============================================================


# função para realizar a busca A* no labirinto
def astar_search(maze: np.ndarray, start: Position, goal: Position) -> List[Tuple[str, Position]]:
    frontier = []
    heapq.heappush(frontier, (0, start))

    came_from: Dict[Position, Tuple[Position, str]] = {}
    cost_so_far: Dict[Position, int] = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            return reconstruct_path(came_from, current)

        for action, next_pos in get_neighbors(maze, current):
            new_cost = cost_so_far[current] + 1

            # Optional: make trap more expensive
            if maze[next_pos] == 4:
                new_cost += 5

            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost
                priority = new_cost + manhattan(next_pos, goal)
                heapq.heappush(frontier, (priority, next_pos))
                came_from[next_pos] = (current, action)

    return []

# ============================================================
# Funções para dezenhar o mapa
# ============================================================


# Desenha o mapa do labirinto, o agente, o caminho planejado e o caminho visitado usando Matplotlib
def draw_maze(
    base_maze: np.ndarray,
    agent: Agent,
    planned_path: List[Position],
    action: Optional[str],
    step: int
) -> None:
    visual_maze = base_maze.copy()

    # Remove original agent from base map
    visual_maze[visual_maze == 2] = 0

    if(planned_path != None):
        # Draw planned path
        for pos in planned_path:
            if visual_maze[pos] == 0:
                visual_maze[pos] = 6

    # Draw visited path
    for pos in agent.path_taken:
        if visual_maze[pos] == 0 or visual_maze[pos] == 6:
            visual_maze[pos] = 5

    # Draw goal again
    visual_maze[agent.goal] = 3

    # Draw agent
    visual_maze[agent.position] = 2

    cmap = ListedColormap([
        "white",       # 0 free path
        "black",       # 1 wall
        "royalblue",   # 2 agent
        "limegreen",   # 3 goal
        "red",         # 4 trap
        "gold",        # 5 visited path
        "lightskyblue" # 6 planned path
    ])

    plt.clf()
    plt.imshow(visual_maze, cmap=cmap, vmin=0, vmax=6)

    # Grid lines
    plt.xticks(np.arange(-0.5, base_maze.shape[1], 1), [])
    plt.yticks(np.arange(-0.5, base_maze.shape[0], 1), [])
    plt.grid(color="gray", linewidth=1)

    title = f"Step: {step}"
    if action is not None:
        title += f" | Actuation: {action}"

    plt.title(title)
    plt.pause(0.4)



# ============================================================
# Função usada para executar o agente e interagir com o mapa
# ============================================================


# função principal para executar o agente no labirinto
def main() -> None:

    # Faz uma cópia do labirinto para não modificar o original durante a execução
    maze = MAZE.copy()

    # encontra a posição (Position) inicial do agente (2) e do objetivo (3) no labirinto
    posicaoInicialDoAgente = find_cell(maze, 2)
    posicaoObjetivo = find_cell(maze, 3)

    # instancia o Agente
    agent = Agent(
        position=posicaoInicialDoAgente,
        goal=posicaoObjetivo,
        path_taken=[posicaoInicialDoAgente]
    )

    # Exibe o labirinto com o estado inicial do agente e do objetivo antes de começar a busca
    draw_maze(
        base_maze=maze,
        agent=agent,
        planned_path=None,
        action=None,
        step=0
    )
    # wait two minuts
    time.sleep(2)

    # cria uma solução (ou plano de ações) usando a busca A* para encontrar o caminho do agente até o objetivo
    solution = astar_search(maze, posicaoInicialDoAgente, posicaoObjetivo)

    if not solution:
        print("No path found.")
        return

    planned_positions = [pos for _, pos in solution]

    plt.figure(figsize=(8, 5))
    plt.ion()

    # Execute one actuation at a time
    for step, (action, new_position) in enumerate(solution, start=1):
        agent.position = new_position
        agent.path_taken.append(new_position)

        draw_maze(
            base_maze=maze,
            agent=agent,
            planned_path=planned_positions,
            action=action,
            step=step
        )

        time.sleep(0.2)

    plt.ioff()
    plt.show()

    print("Goal reached!")
    print("Path:")
    for action, pos in solution:
        print(f"{action} -> {pos}")


if __name__ == "__main__":
    main()