import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.Timer;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.RenderingHints;
import java.awt.Stroke;

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

public class MazeTwoMatrices extends JFrame {

    // ============================================================
    // CONTENT MATRIX VALUES
    // ============================================================
    // This matrix represents what exists inside each cell.
    private static final int EMPTY = 0;
    private static final int FINN = 1;
    private static final int JAKE = 2;
    private static final int TRAP = 3;

    // ============================================================
    // VALID MOVEMENT BIT MASKS
    // ============================================================
    // These values can be combined.
    // Example:
    // RIGHT + DOWN means the agent can move right and down.
    private static final int UP = 1;
    private static final int RIGHT = 2;
    private static final int DOWN = 4;
    private static final int LEFT = 8;

    private static final int ROWS = 5;
    private static final int COLS = 5;

    // ============================================================
    // MATRIX 1: CONTENT OF THE MAP
    // ============================================================
    // Human visualization:
    //
    // Finn starts at row 1, column 1
    // Jake starts at row 1, column 5
    //
    // Java uses zero-based indexing:
    // Finn = CONTENT[0][0]
    // Jake = CONTENT[0][4]

    private static final int[][] CONTENT = {
            {FINN,  EMPTY, EMPTY, EMPTY, JAKE},
            {EMPTY, EMPTY, EMPTY, EMPTY, EMPTY},
            {EMPTY, EMPTY, EMPTY, EMPTY, EMPTY},
            {EMPTY, EMPTY, EMPTY, EMPTY, EMPTY},
            {EMPTY, EMPTY, EMPTY, EMPTY, EMPTY}
    };

    // ============================================================
    // MATRIX 2: VALID MOVEMENTS
    // ============================================================
    // This matrix is generated from allowed directions.
    // Walls are represented by removing movements between cells.

    private static final int[][] VALID_MOVES = createValidMoves();

    public MazeTwoMatrices() {
        setTitle("Maze with Two Matrices - Content + Valid Movements");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setResizable(false);

        MazePanel panel = new MazePanel(CONTENT, VALID_MOVES);
        add(panel);

        pack();
        setLocationRelativeTo(null);
        setVisible(true);

        panel.startSimulation();
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(MazeTwoMatrices::new);
    }

    // ============================================================
    // CREATE VALID MOVEMENTS MATRIX
    // ============================================================

    private static int[][] createValidMoves() {
        int[][] moves = new int[ROWS][COLS];

        // First, allow all movements that stay inside the grid.
        for (int row = 0; row < ROWS; row++) {
            for (int col = 0; col < COLS; col++) {
                int value = 0;

                if (row > 0) value += UP;
                if (col < COLS - 1) value += RIGHT;
                if (row < ROWS - 1) value += DOWN;
                if (col > 0) value += LEFT;

                moves[row][col] = value;
            }
        }

        // Then, block specific passages according to the maze walls.
        //
        // The comments below use human coordinates:
        // row 1..5, column 1..5.
        //
        // Java coordinates are zero-based:
        // row 0..4, column 0..4.

        // Vertical walls in the first row.
        blockPassage(moves, 0, 1, RIGHT); // between (1,2) and (1,3)
        blockPassage(moves, 0, 2, RIGHT); // between (1,3) and (1,4)

        // Vertical wall between columns 1 and 2, from rows 2 to 4.
        blockPassage(moves, 1, 0, RIGHT); // between (2,1) and (2,2)
        blockPassage(moves, 2, 0, RIGHT); // between (3,1) and (3,2)
        blockPassage(moves, 3, 0, RIGHT); // between (4,1) and (4,2)

        // Vertical wall between columns 2 and 3, from rows 2 to 3.
        blockPassage(moves, 1, 1, RIGHT); // between (2,2) and (2,3)
        blockPassage(moves, 2, 1, RIGHT); // between (3,2) and (3,3)

        // Other internal vertical walls.
        blockPassage(moves, 2, 2, RIGHT); // between (3,3) and (3,4)
        blockPassage(moves, 3, 3, RIGHT); // between (4,4) and (4,5)

        // Horizontal walls.
        blockPassage(moves, 0, 0, DOWN);  // between (1,1) and (2,1)
        blockPassage(moves, 1, 4, DOWN);  // between (2,5) and (3,5)
        blockPassage(moves, 2, 3, DOWN);  // between (3,4) and (4,4)
        blockPassage(moves, 3, 1, DOWN);  // between (4,2) and (5,2)
        blockPassage(moves, 3, 2, DOWN);  // between (4,3) and (5,3)

        return moves;
    }

    private static void blockPassage(int[][] moves, int row, int col, int direction) {
        // Remove direction from the current cell.
        moves[row][col] = removeDirection(moves[row][col], direction);

        // Remove the opposite direction from the neighboring cell.
        int neighborRow = row + dRow(direction);
        int neighborCol = col + dCol(direction);

        if (isInside(neighborRow, neighborCol)) {
            int opposite = opposite(direction);
            moves[neighborRow][neighborCol] = removeDirection(moves[neighborRow][neighborCol], opposite);
        }
    }

    private static int removeDirection(int value, int direction) {
        return value & ~direction;
    }

    private static boolean isInside(int row, int col) {
        return row >= 0 && row < ROWS && col >= 0 && col < COLS;
    }

    private static int dRow(int direction) {
        if (direction == UP) return -1;
        if (direction == DOWN) return 1;
        return 0;
    }

    private static int dCol(int direction) {
        if (direction == LEFT) return -1;
        if (direction == RIGHT) return 1;
        return 0;
    }

    private static int opposite(int direction) {
        if (direction == UP) return DOWN;
        if (direction == DOWN) return UP;
        if (direction == LEFT) return RIGHT;
        if (direction == RIGHT) return LEFT;

        throw new IllegalArgumentException("Invalid direction: " + direction);
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
            return "(" + (row + 1) + ", " + (col + 1) + ")";
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

        private final int[][] content;
        private final int[][] validMoves;

        private final int rows;
        private final int cols;

        private final int cellSize = 110;
        private final int offsetX = 70;
        private final int offsetY = 70;
        private final int infoHeight = 50;

        private Pos start;
        private Pos goal;
        private Pos agentPosition;

        private final List<Pos> visitedPath = new ArrayList<>();
        private final List<Step> solutionPath;
        private final Set<Pos> plannedPathCells = new HashSet<>();

        private int currentStepIndex = 0;
        private String currentAction = "Initial state";

        private Image jakeImage;
        private Image finnImage;

        MazePanel(int[][] content, int[][] validMoves) {
            this.content = content;
            this.validMoves = validMoves;

            this.rows = content.length;
            this.cols = content[0].length;

            setPreferredSize(new Dimension(
                    offsetX + cols * cellSize + 30,
                    offsetY + rows * cellSize + infoHeight
            ));

            setBackground(Color.WHITE);

            start = findCell(JAKE);
            goal = findCell(FINN);

            agentPosition = start;
            visitedPath.add(start);

            loadImages();

            solutionPath = astarSearch(start, goal);

            for (Step step : solutionPath) {
                plannedPathCells.add(step.position);
            }
        }

        private void loadImages() {
            try {
                jakeImage = ImageIO.read(new File("jake.png"));
            } catch (IOException e) {
                System.out.println("jake.png not found. Drawing Jake as a blue circle.");
                jakeImage = null;
            }

            try {
                finnImage = ImageIO.read(new File("finn.png"));
            } catch (IOException e) {
                System.out.println("finn.png not found. Drawing Finn as a green circle.");
                finnImage = null;
            }
        }

        public void startSimulation() {
            if (solutionPath.isEmpty()) {
                System.out.println("No path found.");
                return;
            }

            Timer timer = new Timer(700, event -> {
                if (currentStepIndex < solutionPath.size()) {
                    Step step = solutionPath.get(currentStepIndex);

                    currentAction = step.action;
                    agentPosition = step.position;
                    visitedPath.add(agentPosition);

                    currentStepIndex++;
                    repaint();
                } else {
                    ((Timer) event.getSource()).stop();
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
                    if (content[row][col] == value) {
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

            g2.setRenderingHint(
                    RenderingHints.KEY_ANTIALIASING,
                    RenderingHints.VALUE_ANTIALIAS_ON
            );

            drawCoordinates(g2);
            drawCells(g2);
            drawThinGrid(g2);
            drawThickWalls(g2);
            drawFinn(g2);
            drawJake(g2);
            drawInfoText(g2);
        }

        // ============================================================
        // DRAWING
        // ============================================================

        private void drawCoordinates(Graphics2D g2) {
            g2.setColor(Color.BLACK);
            g2.setFont(new Font("Arial", Font.BOLD, 28));

            // Column labels: 1, 2, 3, 4, 5
            for (int col = 0; col < cols; col++) {
                int x = offsetX + col * cellSize + cellSize / 2 - 8;
                int y = offsetY - 25;
                g2.drawString(String.valueOf(col + 1), x, y);
            }

            // Row labels: 1, 2, 3, 4, 5
            for (int row = 0; row < rows; row++) {
                int x = offsetX - 40;
                int y = offsetY + row * cellSize + cellSize / 2 + 10;
                g2.drawString(String.valueOf(row + 1), x, y);
            }
        }

        private void drawCells(Graphics2D g2) {
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    Pos current = new Pos(row, col);

                    int x = offsetX + col * cellSize;
                    int y = offsetY + row * cellSize;

                    g2.setColor(getCellBackground(current));
                    g2.fillRect(x, y, cellSize, cellSize);
                }
            }
        }

        private Color getCellBackground(Pos current) {
            if (current.equals(goal)) {
                return new Color(210, 255, 210);
            }

            if (visitedPath.contains(current)) {
                return new Color(255, 235, 120);
            }

            if (plannedPathCells.contains(current)) {
                return new Color(180, 225, 255);
            }

            if (content[current.row][current.col] == TRAP) {
                return new Color(255, 180, 180);
            }

            return Color.WHITE;
        }

        private void drawThinGrid(Graphics2D g2) {
            g2.setColor(Color.LIGHT_GRAY);
            g2.setStroke(new BasicStroke(1));

            for (int row = 0; row <= rows; row++) {
                int y = offsetY + row * cellSize;
                g2.drawLine(offsetX, y, offsetX + cols * cellSize, y);
            }

            for (int col = 0; col <= cols; col++) {
                int x = offsetX + col * cellSize;
                g2.drawLine(x, offsetY, x, offsetY + rows * cellSize);
            }
        }

        private void drawThickWalls(Graphics2D g2) {
            Stroke oldStroke = g2.getStroke();

            g2.setColor(Color.BLACK);
            g2.setStroke(new BasicStroke(8));

            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    int x = offsetX + col * cellSize;
                    int y = offsetY + row * cellSize;

                    int moves = validMoves[row][col];

                    // If movement is not allowed, draw a thick wall.
                    if (!hasDirection(moves, UP)) {
                        g2.drawLine(x, y, x + cellSize, y);
                    }

                    if (!hasDirection(moves, RIGHT)) {
                        g2.drawLine(x + cellSize, y, x + cellSize, y + cellSize);
                    }

                    if (!hasDirection(moves, DOWN)) {
                        g2.drawLine(x, y + cellSize, x + cellSize, y + cellSize);
                    }

                    if (!hasDirection(moves, LEFT)) {
                        g2.drawLine(x, y, x, y + cellSize);
                    }
                }
            }

            g2.setStroke(oldStroke);
        }

        private boolean hasDirection(int moves, int direction) {
            return (moves & direction) != 0;
        }

        private void drawJake(Graphics2D g2) {
            int x = offsetX + agentPosition.col * cellSize;
            int y = offsetY + agentPosition.row * cellSize;

            if (jakeImage != null) {
                g2.drawImage(jakeImage, x + 12, y + 12, cellSize - 24, cellSize - 24, null);
            } else {
                g2.setColor(Color.BLUE);
                g2.fillOval(x + 25, y + 25, cellSize - 50, cellSize - 50);

                g2.setColor(Color.WHITE);
                g2.setFont(new Font("Arial", Font.BOLD, 20));
                g2.drawString("J", x + cellSize / 2 - 6, y + cellSize / 2 + 7);
            }
        }

        private void drawFinn(Graphics2D g2) {
            int x = offsetX + goal.col * cellSize;
            int y = offsetY + goal.row * cellSize;

            if (finnImage != null) {
                g2.drawImage(finnImage, x + 12, y + 12, cellSize - 24, cellSize - 24, null);
            } else {
                g2.setColor(Color.GREEN.darker());
                g2.fillOval(x + 25, y + 25, cellSize - 50, cellSize - 50);

                g2.setColor(Color.WHITE);
                g2.setFont(new Font("Arial", Font.BOLD, 20));
                g2.drawString("F", x + cellSize / 2 - 6, y + cellSize / 2 + 7);
            }
        }

        private void drawInfoText(Graphics2D g2) {
            g2.setColor(Color.BLACK);
            g2.setFont(new Font("Arial", Font.BOLD, 16));

            String text = "Step: " + currentStepIndex + " | Actuation: " + currentAction;
            g2.drawString(text, offsetX, offsetY + rows * cellSize + 35);
        }

        // ============================================================
        // A* SEARCH USING THE VALID_MOVES MATRIX
        // ============================================================

        private List<Step> astarSearch(Pos start, Pos goal) {
            PriorityQueue<Node> frontier =
                    new PriorityQueue<>(Comparator.comparingInt(n -> n.priority));

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

        private List<Step> getNeighbors(Pos pos) {
            List<Step> neighbors = new ArrayList<>();

            int moves = validMoves[pos.row][pos.col];

            if (hasDirection(moves, UP)) {
                neighbors.add(new Step("UP", new Pos(pos.row - 1, pos.col)));
            }

            if (hasDirection(moves, RIGHT)) {
                neighbors.add(new Step("RIGHT", new Pos(pos.row, pos.col + 1)));
            }

            if (hasDirection(moves, DOWN)) {
                neighbors.add(new Step("DOWN", new Pos(pos.row + 1, pos.col)));
            }

            if (hasDirection(moves, LEFT)) {
                neighbors.add(new Step("LEFT", new Pos(pos.row, pos.col - 1)));
            }

            return neighbors;
        }

        private int manhattan(Pos a, Pos b) {
            return Math.abs(a.row - b.row) + Math.abs(a.col - b.col);
        }
    }
}