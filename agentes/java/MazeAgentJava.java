import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.Set;

public class MazeAgentJava extends JFrame {

    // ============================================================
    // MAZE LEGEND
    // ============================================================
    // 0 = free path
    // 1 = wall
    // 2 = agent initial position
    // 3 = goal
    // 4 = trap / special cell
    // 5 = visited path, used only for visualization
    // 6 = planned path, used only for visualization

    private static final int[][] MAZE = {
            {1, 1, 1, 1, 1, 1, 1, 1, 1},
            {1, 2, 0, 0, 1, 0, 0, 3, 1},
            {1, 0, 1, 0, 1, 0, 1, 0, 1},
            {1, 0, 1, 0, 0, 0, 1, 0, 1},
            {1, 0, 0, 0, 1, 4, 1, 0, 1},
            {1, 1, 1, 1, 1, 1, 1, 1, 1}
    };

    public MazeAgentJava() {
        setTitle("Maze Agent - A* Search");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setResizable(false);

        MazePanel panel = new MazePanel(MAZE);
        add(panel);

        pack();
        setLocationRelativeTo(null);
        setVisible(true);

        panel.startSimulation();
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(MazeAgentJava::new);
    }

    // ============================================================
    // POSITION CLASS
    // ============================================================

    static class Pos {
        int row;
        int col;

        Pos(int row, int col) {
            this.row = row;
            this.col = col;
        }

        @Override
        public boolean equals(Object obj) {
            if (!(obj instanceof Pos)) return false;
            Pos other = (Pos) obj;
            return this.row == other.row && this.col == other.col;
        }

        @Override
        public int hashCode() {
            return Objects.hash(row, col);
        }

        @Override
        public String toString() {
            return "(" + row + ", " + col + ")";
        }
    }

    // ============================================================
    // STEP CLASS
    // ============================================================

    static class Step {
        String action;
        Pos position;

        Step(String action, Pos position) {
            this.action = action;
            this.position = position;
        }
    }

    // ============================================================
    // NODE CLASS FOR A*
    // ============================================================

    static class Node {
        Pos position;
        int priority;

        Node(Pos position, int priority) {
            this.position = position;
            this.priority = priority;
        }
    }

    // ============================================================
    // MAZE PANEL
    // ============================================================

    static class MazePanel extends JPanel {

        private final int[][] maze;
        private final int rows;
        private final int cols;
        private final int cellSize = 80;

        private Pos start;
        private Pos goal;
        private Pos agentPosition;

        private final List<Pos> visitedPath = new ArrayList<>();
        private final List<Step> solutionPath;
        private final Set<Pos> plannedPathCells = new HashSet<>();

        private int currentStepIndex = 0;
        private String currentAction = "Initial state";

        private Image agentImage;

        MazePanel(int[][] maze) {
            this.maze = maze;
            this.rows = maze.length;
            this.cols = maze[0].length;

            setPreferredSize(new Dimension(cols * cellSize, rows * cellSize + 40));
            setBackground(Color.WHITE);

            start = findCell(2);
            goal = findCell(3);
            agentPosition = start;
            visitedPath.add(start);

            loadAgentImage();

            solutionPath = astarSearch(start, goal);

            for (Step step : solutionPath) {
                plannedPathCells.add(step.position);
            }
        }

        private void loadAgentImage() {
            try {
                agentImage = ImageIO.read(new File("jake.png"));
            } catch (IOException e) {
                System.out.println("Could not load jake.png. A blue circle will be used instead.");
                agentImage = null;
            }
        }

        public void startSimulation() {
            if (solutionPath.isEmpty()) {
                System.out.println("No path found.");
                return;
            }

            javax.swing.Timer timer = new javax.swing.Timer(600, event -> {
                if (currentStepIndex < solutionPath.size()) {
                    Step step = solutionPath.get(currentStepIndex);

                    currentAction = step.action;
                    agentPosition = step.position;
                    visitedPath.add(agentPosition);

                    currentStepIndex++;
                    repaint();
                } else {
                    ((javax.swing.Timer) event.getSource()).stop();
                    currentAction = "Goal reached!";
                    repaint();

                    System.out.println("Goal reached!");
                    System.out.println("Path:");
                    for (Step step : solutionPath) {
                        System.out.println(step.action + " -> " + step.position);
                    }
                }
            });

            timer.start();
        }

        private Pos findCell(int value) {
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    if (maze[row][col] == value) {
                        return new Pos(row, col);
                    }
                }
            }
            throw new RuntimeException("Cell value " + value + " not found.");
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);

            Graphics2D g2 = (Graphics2D) g;
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            drawMaze(g2);
            drawInfoText(g2);
        }

        private void drawMaze(Graphics2D g2) {
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {

                    Pos current = new Pos(row, col);

                    Color cellColor = getCellColor(row, col, current);

                    int x = col * cellSize;
                    int y = row * cellSize;

                    g2.setColor(cellColor);
                    g2.fillRect(x, y, cellSize, cellSize);

                    g2.setColor(Color.GRAY);
                    g2.drawRect(x, y, cellSize, cellSize);
                }
            }

            drawAgent(g2);
        }

        private Color getCellColor(int row, int col, Pos current) {
            int value = maze[row][col];

            if (value == 1) {
                return Color.BLACK;
            }

            if (current.equals(goal)) {
                return Color.GREEN;
            }

            if (value == 4) {
                return Color.RED;
            }

            if (visitedPath.contains(current)) {
                return Color.YELLOW;
            }

            if (plannedPathCells.contains(current)) {
                return new Color(135, 206, 250); // light blue
            }

            return Color.WHITE;
        }

        private void drawAgent(Graphics2D g2) {
            int x = agentPosition.col * cellSize;
            int y = agentPosition.row * cellSize;

            if (agentImage != null) {
                g2.drawImage(agentImage, x + 5, y + 5, cellSize - 10, cellSize - 10, null);
            } else {
                g2.setColor(Color.BLUE);
                g2.fillOval(x + 10, y + 10, cellSize - 20, cellSize - 20);
            }
        }

        private void drawInfoText(Graphics2D g2) {
            g2.setColor(Color.BLACK);
            g2.setFont(new Font("Arial", Font.BOLD, 16));

            String text = "Step: " + currentStepIndex + " | Actuation: " + currentAction;
            g2.drawString(text, 10, rows * cellSize + 25);
        }

        // ============================================================
        // A* SEARCH
        // ============================================================

        private List<Step> astarSearch(Pos start, Pos goal) {
            PriorityQueue<Node> frontier = new PriorityQueue<>(Comparator.comparingInt(n -> n.priority));
            frontier.add(new Node(start, 0));

            Map<Pos, Pos> cameFrom = new HashMap<>();
            Map<Pos, String> actionFrom = new HashMap<>();
            Map<Pos, Integer> costSoFar = new HashMap<>();

            costSoFar.put(start, 0);

            while (!frontier.isEmpty()) {
                Pos current = frontier.poll().position;

                if (current.equals(goal)) {
                    return reconstructPath(cameFrom, actionFrom, current);
                }

                for (Step neighbor : getNeighbors(current)) {
                    Pos next = neighbor.position;

                    int newCost = costSoFar.get(current) + 1;

                    // Optional: trap has higher cost
                    if (maze[next.row][next.col] == 4) {
                        newCost += 5;
                    }

                    if (!costSoFar.containsKey(next) || newCost < costSoFar.get(next)) {
                        costSoFar.put(next, newCost);

                        int priority = newCost + manhattan(next, goal);
                        frontier.add(new Node(next, priority));

                        cameFrom.put(next, current);
                        actionFrom.put(next, neighbor.action);
                    }
                }
            }

            return new ArrayList<>();
        }

        private List<Step> reconstructPath(
                Map<Pos, Pos> cameFrom,
                Map<Pos, String> actionFrom,
                Pos current
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

        private int manhattan(Pos a, Pos b) {
            return Math.abs(a.row - b.row) + Math.abs(a.col - b.col);
        }

        private List<Step> getNeighbors(Pos pos) {
            List<Step> neighbors = new ArrayList<>();

            addNeighbor(neighbors, pos, -1, 0, "UP");
            addNeighbor(neighbors, pos, 1, 0, "DOWN");
            addNeighbor(neighbors, pos, 0, -1, "LEFT");
            addNeighbor(neighbors, pos, 0, 1, "RIGHT");

            return neighbors;
        }

        private void addNeighbor(List<Step> neighbors, Pos pos, int dr, int dc, String action) {
            Pos next = new Pos(pos.row + dr, pos.col + dc);

            if (isValidPosition(next)) {
                neighbors.add(new Step(action, next));
            }
        }

        private boolean isValidPosition(Pos pos) {
            if (pos.row < 0 || pos.row >= rows) return false;
            if (pos.col < 0 || pos.col >= cols) return false;

            // Wall is not walkable
            return maze[pos.row][pos.col] != 1;
        }
    }
}