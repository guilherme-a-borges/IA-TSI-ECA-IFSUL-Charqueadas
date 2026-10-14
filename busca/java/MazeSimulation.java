import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.List;
import java.util.Queue;
import javax.imageio.ImageIO;
import javax.swing.*;

// ============================================================
// Definição de Posição, Passo e Nós auxiliares
// ============================================================
record Position(int row, int col) {}

record Step(String action, Position position) {}

record PQNode(int priority, long tieBreaker, Position pos) implements Comparable<PQNode> {
    @Override
    public int compareTo(PQNode o) {
        int cmp = Integer.compare(this.priority, o.priority);
        if (cmp != 0) return cmp;
        return Long.compare(this.tieBreaker, o.tieBreaker);
    }
}

record DepthNode(Position pos, int depth) {}

public class MazeSimulation {

    // ============================================================
    // Definir Valores no mapa
    // ============================================================
    public static final int EMPTY = 0;
    public static final int FINN = 1;
    public static final int JAKE = 2;
    public static final int TRAP = 3;

    // ============================================================
    // Ações permitidas para o cliente
    // ============================================================
    public static final int UP = 1;
    public static final int RIGHT = 2;
    public static final int DOWN = 4;
    public static final int LEFT = 8;

    // ============================================================
    // MATRIZ 1: Definição do conteúdo do mapa
    // ============================================================
    public static final int ROWS = 5;
    public static final int COLS = 5;

    public static final int[][] CONTENT = {
        {FINN,  EMPTY, EMPTY, EMPTY, JAKE},
        {EMPTY, EMPTY, EMPTY, EMPTY, EMPTY},
        {EMPTY, EMPTY, TRAP,  EMPTY, EMPTY},
        {EMPTY, EMPTY, EMPTY, EMPTY, EMPTY},
        {EMPTY, EMPTY, EMPTY, EMPTY, EMPTY}
    };

    public static final int[][] VALID_MOVES = createValidMoves();

    // ============================================================
    // Funções para validar movimentos e declarar paredes
    // ============================================================
    public static int[][] createValidMoves() {
        int[][] moves = new int[ROWS][COLS];

        for (int row = 0; row < ROWS; row++) {
            for (int col = 0; col < COLS; col++) {
                int value = 0;
                if (row > 0) value |= UP;
                if (col < COLS - 1) value |= RIGHT;
                if (row < ROWS - 1) value |= DOWN;
                if (col > 0) value |= LEFT;
                moves[row][col] = value;
            }
        }

        // Bloqueios das paredes do labirinto
        blockPassage(moves, 0, 1, RIGHT);
        blockPassage(moves, 0, 2, RIGHT);

        blockPassage(moves, 1, 0, RIGHT);
        blockPassage(moves, 2, 0, RIGHT);
        blockPassage(moves, 3, 0, RIGHT);

        blockPassage(moves, 1, 1, RIGHT);
        blockPassage(moves, 2, 1, RIGHT);

        blockPassage(moves, 2, 2, RIGHT);
        blockPassage(moves, 3, 3, RIGHT);

        blockPassage(moves, 0, 0, DOWN);
        blockPassage(moves, 1, 4, DOWN);
        blockPassage(moves, 2, 3, DOWN);
        blockPassage(moves, 3, 1, DOWN);
        blockPassage(moves, 3, 2, DOWN);

        return moves;
    }

    public static void blockPassage(int[][] moves, int row, int col, int direction) {
        moves[row][col] = removeDirection(moves[row][col], direction);
        int neighborRow = row + deltaRow(direction);
        int neighborCol = col + deltaCol(direction);

        if (isInside(neighborRow, neighborCol)) {
            int oppositeDirection = opposite(direction);
            moves[neighborRow][neighborCol] = removeDirection(
                moves[neighborRow][neighborCol],
                oppositeDirection
            );
        }
    }

    public static int removeDirection(int value, int direction) {
        return value & ~direction;
    }

    public static boolean isInside(int row, int col) {
        return row >= 0 && row < ROWS && col >= 0 && col < COLS;
    }

    public static int deltaRow(int direction) {
        if (direction == UP) return -1;
        if (direction == DOWN) return 1;
        return 0;
    }

    public static int deltaCol(int direction) {
        if (direction == LEFT) return -1;
        if (direction == RIGHT) return 1;
        return 0;
    }

    public static int opposite(int direction) {
        if (direction == UP) return DOWN;
        if (direction == DOWN) return UP;
        if (direction == LEFT) return RIGHT;
        if (direction == RIGHT) return LEFT;
        throw new IllegalArgumentException("Invalid direction: " + direction);
    }

    public static boolean hasDirection(int moves, int direction) {
        return (moves & direction) != 0;
    }

    // ============================================================
    // Funções para validar movimento e encontrar célula do agente
    // ============================================================
    public static Position findCell(int[][] content, int value) {
        for (int row = 0; row < ROWS; row++) {
            for (int col = 0; col < COLS; col++) {
                if (content[row][col] == value) {
                    return new Position(row, col);
                }
            }
        }
        throw new IllegalArgumentException("Value " + value + " not found in content matrix.");
    }

    public static List<Step> getNeighbors(int[][] validMoves, Position pos) {
        int row = pos.row();
        int col = pos.col();
        int moves = validMoves[row][col];
        List<Step> neighbors = new ArrayList<>();

        if (hasDirection(moves, UP)) neighbors.add(new Step("UP", new Position(row - 1, col)));
        if (hasDirection(moves, RIGHT)) neighbors.add(new Step("RIGHT", new Position(row, col + 1)));
        if (hasDirection(moves, DOWN)) neighbors.add(new Step("DOWN", new Position(row + 1, col)));
        if (hasDirection(moves, LEFT)) neighbors.add(new Step("LEFT", new Position(row, col - 1)));

        return neighbors;
    }

    public static int getStepCost(int[][] content, Position pos) {
        if (content[pos.row()][pos.col()] == TRAP) {
            return 15;
        }
        return 1;
    }

    public static List<Step> reconstructPath(
        Map<Position, Position> cameFrom,
        Map<Position, String> actionFrom,
        Position current
    ) {
        List<Step> path = new ArrayList<>();
        while (cameFrom.containsKey(current)) {
            String action = actionFrom.get(current);
            path.add(new Step(action, current));
            current = cameFrom.get(current);
        }
        Collections.reverse(path);
        return path;
    }

    public static int manhattan(Position a, Position b) {
        return Math.abs(a.row() - b.row()) + Math.abs(a.col() - b.col());
    }

    // ============================================================
    // 1. BUSCAS NÃO INFORMADAS (CLÁSSICAS)
    // ============================================================

    public static List<Step> bfsSearch(
        int[][] content,
        int[][] validMoves,
        Position start,
        Position goal
    ) {
        if (start.equals(goal)) return Collections.emptyList();

        Queue<Position> frontier = new ArrayDeque<>();
        Set<Position> visited = new HashSet<>();
        Map<Position, Position> cameFrom = new HashMap<>();
        Map<Position, String> actionFrom = new HashMap<>();

        frontier.add(start);
        visited.add(start);

        while (!frontier.isEmpty()) {
            Position current = frontier.poll();

            if (current.equals(goal)) {
                return reconstructPath(cameFrom, actionFrom, current);
            }

            for (Step step : getNeighbors(validMoves, current)) {
                Position nextPos = step.position();

                if (!visited.contains(nextPos)) {
                    visited.add(nextPos);
                    frontier.add(nextPos);

                    cameFrom.put(nextPos, current);
                    actionFrom.put(nextPos, step.action());
                }
            }
        }
        return Collections.emptyList();
    }

    public static List<Step> uniformCostSearch(
        int[][] content,
        int[][] validMoves,
        Position start,
        Position goal
    ) {
        PriorityQueue<PQNode> frontier = new PriorityQueue<>();
        long tieBreaker = 0;

        frontier.add(new PQNode(0, tieBreaker++, start));

        Map<Position, Position> cameFrom = new HashMap<>();
        Map<Position, String> actionFrom = new HashMap<>();
        Map<Position, Integer> costSoFar = new HashMap<>();
        costSoFar.put(start, 0);

        while (!frontier.isEmpty()) {
            PQNode current = frontier.poll();
            Position currentPos = current.pos();

            if (currentPos.equals(goal)) {
                return reconstructPath(cameFrom, actionFrom, currentPos);
            }

            if (current.priority() > costSoFar.get(currentPos)) {
                continue;
            }

            for (Step step : getNeighbors(validMoves, currentPos)) {
                Position nextPos = step.position();
                int newCost = costSoFar.get(currentPos) + getStepCost(content, nextPos);

                if (!costSoFar.containsKey(nextPos) || newCost < costSoFar.get(nextPos)) {
                    costSoFar.put(nextPos, newCost);
                    frontier.add(new PQNode(newCost, tieBreaker++, nextPos));

                    cameFrom.put(nextPos, currentPos);
                    actionFrom.put(nextPos, step.action());
                }
            }
        }
        return Collections.emptyList();
    }

    public static List<Step> dfsSearch(
        int[][] content,
        int[][] validMoves,
        Position start,
        Position goal
    ) {
        if (start.equals(goal)) return Collections.emptyList();

        Deque<Position> frontier = new ArrayDeque<>();
        Set<Position> visited = new HashSet<>();
        Map<Position, Position> cameFrom = new HashMap<>();
        Map<Position, String> actionFrom = new HashMap<>();

        frontier.push(start);
        visited.add(start);

        while (!frontier.isEmpty()) {
            Position current = frontier.pop();

            if (current.equals(goal)) {
                return reconstructPath(cameFrom, actionFrom, current);
            }

            for (Step step : getNeighbors(validMoves, current)) {
                Position nextPos = step.position();

                if (!visited.contains(nextPos)) {
                    visited.add(nextPos);
                    frontier.push(nextPos);

                    cameFrom.put(nextPos, current);
                    actionFrom.put(nextPos, step.action());
                }
            }
        }
        return Collections.emptyList();
    }

    public static List<Step> depthLimitedSearch(
        int[][] content,
        int[][] validMoves,
        Position start,
        Position goal,
        int limit
    ) {
        if (start.equals(goal)) return Collections.emptyList();

        Deque<DepthNode> frontier = new ArrayDeque<>();
        Map<Position, Integer> depthReached = new HashMap<>();
        Map<Position, Position> cameFrom = new HashMap<>();
        Map<Position, String> actionFrom = new HashMap<>();

        frontier.push(new DepthNode(start, 0));
        depthReached.put(start, 0);

        while (!frontier.isEmpty()) {
            DepthNode node = frontier.pop();
            Position current = node.pos();
            int depth = node.depth();

            if (current.equals(goal)) {
                return reconstructPath(cameFrom, actionFrom, current);
            }

            if (depth < limit) {
                for (Step step : getNeighbors(validMoves, current)) {
                    Position nextPos = step.position();
                    int nextDepth = depth + 1;

                    if (!depthReached.containsKey(nextPos) || nextDepth < depthReached.get(nextPos)) {
                        depthReached.put(nextPos, nextDepth);
                        frontier.push(new DepthNode(nextPos, nextDepth));

                        cameFrom.put(nextPos, current);
                        actionFrom.put(nextPos, step.action());
                    }
                }
            }
        }
        return Collections.emptyList();
    }

    public static List<Step> iterativeDeepeningSearch(
        int[][] content,
        int[][] validMoves,
        Position start,
        Position goal,
        int maxDepth
    ) {
        if (start.equals(goal)) return Collections.emptyList();

        for (int depth = 0; depth <= maxDepth; depth++) {
            Set<Position> pathSet = new HashSet<>();
            List<Step> result = dlsRecursive(validMoves, start, goal, depth, 0, pathSet);
            if (result != null) {
                return result;
            }
        }
        return Collections.emptyList();
    }

    private static List<Step> dlsRecursive(
        int[][] validMoves,
        Position current,
        Position goal,
        int limit,
        int currentDepth,
        Set<Position> pathSet
    ) {
        if (current.equals(goal)) return new ArrayList<>();
        if (currentDepth == limit) return null;

        pathSet.add(current);
        for (Step step : getNeighbors(validMoves, current)) {
            Position nextPos = step.position();
            if (!pathSet.contains(nextPos)) {
                List<Step> res = dlsRecursive(validMoves, nextPos, goal, limit, currentDepth + 1, pathSet);
                if (res != null) {
                    res.add(0, step);
                    pathSet.remove(current);
                    return res;
                }
            }
        }
        pathSet.remove(current);
        return null;
    }

    // ============================================================
    // 2. BUSCAS HEURÍSTICAS (INFORMADAS)
    // ============================================================

    public static List<Step> greedyBestFirstSearch(
        int[][] content,
        int[][] validMoves,
        Position start,
        Position goal
    ) {
        PriorityQueue<PQNode> frontier = new PriorityQueue<>();
        long tieBreaker = 0;

        int initialPriority = manhattan(start, goal);
        frontier.add(new PQNode(initialPriority, tieBreaker++, start));

        Map<Position, Position> cameFrom = new HashMap<>();
        Map<Position, String> actionFrom = new HashMap<>();
        Set<Position> visited = new HashSet<>();
        visited.add(start);

        while (!frontier.isEmpty()) {
            PQNode current = frontier.poll();
            Position currentPos = current.pos();

            if (currentPos.equals(goal)) {
                return reconstructPath(cameFrom, actionFrom, currentPos);
            }

            for (Step step : getNeighbors(validMoves, currentPos)) {
                Position nextPos = step.position();

                if (!visited.contains(nextPos)) {
                    visited.add(nextPos);
                    int priority = manhattan(nextPos, goal);
                    frontier.add(new PQNode(priority, tieBreaker++, nextPos));

                    cameFrom.put(nextPos, currentPos);
                    actionFrom.put(nextPos, step.action());
                }
            }
        }
        return Collections.emptyList();
    }

    public static List<Step> astarSearch(
        int[][] content,
        int[][] validMoves,
        Position start,
        Position goal
    ) {
        PriorityQueue<PQNode> frontier = new PriorityQueue<>();
        long tieBreaker = 0;

        frontier.add(new PQNode(0, tieBreaker++, start));

        Map<Position, Position> cameFrom = new HashMap<>();
        Map<Position, String> actionFrom = new HashMap<>();
        Map<Position, Integer> costSoFar = new HashMap<>();
        costSoFar.put(start, 0);

        while (!frontier.isEmpty()) {
            PQNode current = frontier.poll();
            Position currentPos = current.pos();

            if (currentPos.equals(goal)) {
                return reconstructPath(cameFrom, actionFrom, currentPos);
            }

            for (Step step : getNeighbors(validMoves, currentPos)) {
                Position nextPos = step.position();
                int newCost = costSoFar.get(currentPos) + getStepCost(content, nextPos);

                if (!costSoFar.containsKey(nextPos) || newCost < costSoFar.get(nextPos)) {
                    costSoFar.put(nextPos, newCost);
                    int priority = newCost + manhattan(nextPos, goal);
                    frontier.add(new PQNode(priority, tieBreaker++, nextPos));

                    cameFrom.put(nextPos, currentPos);
                    actionFrom.put(nextPos, step.action());
                }
            }
        }
        return Collections.emptyList();
    }

    // ============================================================
    // Painel GUI em Java Swing para Renderização do Labirinto
    // ============================================================
    static class MazePanel extends JPanel {
        private final int[][] content;
        private final int[][] validMoves;
        private final Position goal;
        private final Set<Position> plannedPath;

        private Position agentPosition;
        private Set<Position> visited;
        private int stepNumber;
        private String currentAction;
        private int accumulatedCost;

        private BufferedImage jakeImage;
        private BufferedImage finnImage;

        public MazePanel(int[][] content, int[][] validMoves, Position goal, Set<Position> plannedPath) {
            this.content = content;
            this.validMoves = validMoves;
            this.goal = goal;
            this.plannedPath = plannedPath;
            this.visited = new HashSet<>();

            this.jakeImage = loadImage("jake.png");
            this.finnImage = loadImage("finn.png");
        }

        private BufferedImage loadImage(String filename) {
            try {
                File file = new File(filename);
                if (file.exists()) {
                    return ImageIO.read(file);
                }
            } catch (IOException ignored) {}
            return null;
        }

        public void updateState(Position agentPosition, Set<Position> visited, int stepNumber, String action, int cost) {
            this.agentPosition = agentPosition;
            this.visited = new HashSet<>(visited);
            this.stepNumber = stepNumber;
            this.currentAction = action;
            this.accumulatedCost = cost;
            repaint();
        }

        private Color getCellBackground(Position pos) {
            if (pos.equals(goal)) return new Color(210, 255, 210);       // Light green
            if (content[pos.row()][pos.col()] == TRAP) return new Color(255, 180, 180); // Light red
            if (visited.contains(pos)) return new Color(255, 235, 120);   // Yellow
            if (plannedPath.contains(pos)) return new Color(180, 225, 255); // Light blue
            return Color.WHITE;
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2 = (Graphics2D) g;
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            // Definição das margens para evitar sobreposição
            int marginTop = 85;    // Espaço para a legenda (topo) + números de coluna (meio)
            int marginLeft = 60;   // Espaço para os números de linha (esquerda)
            int paddingRight = 40;
            int paddingBottom = 40;

            int cellSize = Math.min(
                (getWidth() - marginLeft - paddingRight) / COLS,
                (getHeight() - marginTop - paddingBottom) / ROWS
            );

            // 1. Legenda no topo da janela
            g2.setColor(Color.BLACK);
            g2.setFont(new Font("SansSerif", Font.BOLD, 16));
            String title = String.format("Step: %d | Action: %s | Total Cost: %d", stepNumber, currentAction, accumulatedCost);
            FontMetrics fm = g2.getFontMetrics();
            g2.drawString(title, (getWidth() - fm.stringWidth(title)) / 2, 30);

            // 2. Números das colunas (Abaixo da legenda e acima do labirinto)
            g2.setFont(new Font("SansSerif", Font.BOLD, 18));
            for (int c = 0; c < COLS; c++) {
                int x = marginLeft + c * cellSize + cellSize / 2 - 5;
                int y = marginTop - 15;
                g2.drawString(String.valueOf(c + 1), x, y);
            }

            // 3. Números das linhas (À esquerda do labirinto)
            for (int r = 0; r < ROWS; r++) {
                int x = marginLeft / 2 - 10;
                int y = marginTop + r * cellSize + cellSize / 2 + 6;
                g2.drawString(String.valueOf(r + 1), x, y);
            }

            // 4. Desenhar Células e Fundos
            for (int r = 0; r < ROWS; r++) {
                for (int c = 0; c < COLS; c++) {
                    int x = marginLeft + c * cellSize;
                    int y = marginTop + r * cellSize;
                    Position pos = new Position(r, c);

                    g2.setColor(getCellBackground(pos));
                    g2.fillRect(x, y, cellSize, cellSize);

                    g2.setColor(Color.LIGHT_GRAY);
                    g2.drawRect(x, y, cellSize, cellSize);
                }
            }

            // 5. Desenhar Paredes Espessas
            g2.setColor(Color.BLACK);
            g2.setStroke(new BasicStroke(6));

            for (int r = 0; r < ROWS; r++) {
                for (int c = 0; c < COLS; c++) {
                    int x = marginLeft + c * cellSize;
                    int y = marginTop + r * cellSize;
                    int moves = validMoves[r][c];

                    if (!hasDirection(moves, UP)) g2.drawLine(x, y, x + cellSize, y);
                    if (!hasDirection(moves, RIGHT)) g2.drawLine(x + cellSize, y, x + cellSize, y + cellSize);
                    if (!hasDirection(moves, DOWN)) g2.drawLine(x, y + cellSize, x + cellSize, y + cellSize);
                    if (!hasDirection(moves, LEFT)) g2.drawLine(x, y, x, y + cellSize);
                }
            }

            // 6. Desenhar Finn
            int goalX = marginLeft + goal.col() * cellSize;
            int goalY = marginTop + goal.row() * cellSize;
            if (finnImage != null) {
                g2.drawImage(finnImage, goalX + (int)(cellSize * 0.15), goalY + (int)(cellSize * 0.15),
                             (int)(cellSize * 0.7), (int)(cellSize * 0.7), null);
            } else {
                g2.setColor(new Color(34, 139, 34));
                g2.fillOval(goalX + (int)(cellSize * 0.2), goalY + (int)(cellSize * 0.2), (int)(cellSize * 0.6), (int)(cellSize * 0.6));
                g2.setColor(Color.WHITE);
                g2.setFont(new Font("SansSerif", Font.BOLD, cellSize / 2));
                g2.drawString("F", goalX + cellSize / 3, goalY + (int)(cellSize * 0.7));
            }

            // 7. Desenhar Jake
            if (agentPosition != null) {
                int agentX = marginLeft + agentPosition.col() * cellSize;
                int agentY = marginTop + agentPosition.row() * cellSize;
                if (jakeImage != null) {
                    g2.drawImage(jakeImage, agentX + (int)(cellSize * 0.15), agentY + (int)(cellSize * 0.15),
                                 (int)(cellSize * 0.7), (int)(cellSize * 0.7), null);
                } else {
                    g2.setColor(new Color(65, 105, 225));
                    g2.fillOval(agentX + (int)(cellSize * 0.2), agentY + (int)(cellSize * 0.2), (int)(cellSize * 0.6), (int)(cellSize * 0.6));
                    g2.setColor(Color.WHITE);
                    g2.setFont(new Font("SansSerif", Font.BOLD, cellSize / 2));
                    g2.drawString("J", agentX + cellSize / 3, agentY + (int)(cellSize * 0.7));
                }
            }
        }
    }

    // ============================================================
    // MAIN SIMULATION
    // ============================================================
    public static void main(String[] args) {
        int[][] content = CONTENT;
        int[][] validMoves = VALID_MOVES;

        Position start = findCell(content, JAKE);
        Position goal = findCell(content, FINN);

        // Algoritmos disponíveis para seleção:
        // List<Step> solution = bfsSearch(content, validMoves, start, goal);
        // List<Step> solution = uniformCostSearch(content, validMoves, start, goal);
        // List<Step> solution = dfsSearch(content, validMoves, start, goal);
        // List<Step> solution = depthLimitedSearch(content, validMoves, start, goal, 12);
        // List<Step> solution = iterativeDeepeningSearch(content, validMoves, start, goal, 25);

        // List<Step> solution = greedyBestFirstSearch(content, validMoves, start, goal);
        List<Step> solution = astarSearch(content, validMoves, start, goal);

        if (solution.isEmpty()) {
            System.out.println("No path found.");
            return;
        }

        Set<Position> plannedPath = new HashSet<>();
        for (Step s : solution) {
            plannedPath.add(s.position());
        }

        JFrame frame = new JFrame("Maze Simulation - Java");
        MazePanel panel = new MazePanel(content, validMoves, goal, plannedPath);
        frame.add(panel);
        frame.setSize(600, 650);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

        Set<Position> visited = new HashSet<>();
        visited.add(start);
        Position agentPosition = start;
        int accumulatedCost = 0;

        panel.updateState(agentPosition, visited, 0, "Initial state", accumulatedCost);

        try {
            Thread.sleep(800);
            int stepNumber = 1;
            for (Step step : solution) {
                agentPosition = step.position();
                visited.add(agentPosition);
                accumulatedCost += getStepCost(content, agentPosition);

                panel.updateState(agentPosition, visited, stepNumber++, step.action(), accumulatedCost);
                Thread.sleep(700);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        System.out.println("Goal reached!");
        System.out.println("Total Path Cost: " + accumulatedCost);
        System.out.println("Path:");
        for (Step step : solution) {
            System.out.println(step.action() + " -> " + "(" + step.position().row() + ", " + step.position().col() + ")");
        }
    }
}