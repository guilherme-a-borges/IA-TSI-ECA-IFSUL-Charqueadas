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
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.Set;

public class MazeQLearningVisual extends JFrame {

    // ============================================================
    // CONTENT MATRIX VALUES
    // ============================================================

    private static final int EMPTY = 0;
    private static final int FINN = 1;
    private static final int JAKE = 2;
    private static final int TRAP = 3;

    // ============================================================
    // VALID MOVEMENT BIT MASKS
    // ============================================================

    private static final int UP = 1;
    private static final int RIGHT = 2;
    private static final int DOWN = 4;
    private static final int LEFT = 8;

    private static final int[] ACTION_BITS = {UP, RIGHT, DOWN, LEFT};

    private static final String[] ACTION_NAMES = {
            "UP", "RIGHT", "DOWN", "LEFT"
    };

    private static final int ROWS = 5;
    private static final int COLS = 5;
    private static final int ACTION_COUNT = 4;

    // ============================================================
    // MATRIX 1: CONTENT OF THE MAP
    // ============================================================
    // 0 = empty
    // 1 = Finn / goal
    // 2 = Jake / agent
    // 3 = trap

    private static final int[][] CONTENT = {
            {FINN,  EMPTY, EMPTY, EMPTY, JAKE},
            {EMPTY, EMPTY, EMPTY, EMPTY, EMPTY},
            {EMPTY, EMPTY, TRAP,  EMPTY, EMPTY},
            {EMPTY, EMPTY, EMPTY, EMPTY, EMPTY},
            {EMPTY, EMPTY, EMPTY, EMPTY, EMPTY}
    };

    // ============================================================
    // MATRIX 2: VALID MOVEMENTS
    // ============================================================

    private static final int[][] VALID_MOVES = createValidMoves();

    public MazeQLearningVisual() {
        setTitle("Q-Learning Maze - Two Matrices");
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
        Pos goal = findCell(CONTENT, FINN);
        double[][] rewardMatrix = createRewardMatrix(CONTENT, goal);
        printRewardMatrix(rewardMatrix);

        SwingUtilities.invokeLater(MazeQLearningVisual::new);
    }

    // ============================================================
    // VALID MOVES CREATION
    // ============================================================

    private static int[][] createValidMoves() {
        int[][] moves = new int[ROWS][COLS];

        // First, allow all movements that remain inside the grid.
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

        // Then block passages to create the maze.
        // Java coordinates: row 0..4, col 0..4.

        // Top internal walls.
        blockPassage(moves, 0, 1, RIGHT);
        blockPassage(moves, 0, 2, RIGHT);
        blockPassage(moves, 0, 3, DOWN);

        // Right-side control.
        blockPassage(moves, 1, 4, DOWN);
        blockPassage(moves, 2, 4, DOWN);

        // Left-side vertical barriers between columns 0 and 1.
        blockPassage(moves, 1, 0, RIGHT);
        blockPassage(moves, 2, 0, RIGHT);
        blockPassage(moves, 3, 0, RIGHT);

        // Internal vertical barriers.
        blockPassage(moves, 1, 1, RIGHT);
        blockPassage(moves, 2, 1, RIGHT);
        blockPassage(moves, 3, 1, RIGHT);

        blockPassage(moves, 2, 2, RIGHT);
        blockPassage(moves, 3, 3, RIGHT);

        // Internal horizontal barriers.
        blockPassage(moves, 0, 0, DOWN);
        blockPassage(moves, 3, 2, DOWN);

        return moves;
    }

    private static void blockPassage(int[][] moves, int row, int col, int direction) {
        moves[row][col] = removeDirection(moves[row][col], direction);

        int neighborRow = row + deltaRow(direction);
        int neighborCol = col + deltaCol(direction);

        if (isInside(neighborRow, neighborCol)) {
            int oppositeDirection = opposite(direction);
            moves[neighborRow][neighborCol] =
                    removeDirection(moves[neighborRow][neighborCol], oppositeDirection);
        }
    }

    private static int removeDirection(int value, int direction) {
        return value & ~direction;
    }

    private static boolean isInside(int row, int col) {
        return row >= 0 && row < ROWS && col >= 0 && col < COLS;
    }

    private static int deltaRow(int direction) {
        if (direction == UP) return -1;
        if (direction == DOWN) return 1;
        return 0;
    }

    private static int deltaCol(int direction) {
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

    private static boolean hasDirection(int moves, int direction) {
        return (moves & direction) != 0;
    }

    // ============================================================
    // BASIC ENVIRONMENT FUNCTIONS
    // ============================================================

    private static Pos findCell(int[][] content, int value) {
        for (int row = 0; row < ROWS; row++) {
            for (int col = 0; col < COLS; col++) {
                if (content[row][col] == value) {
                    return new Pos(row, col);
                }
            }
        }

        throw new RuntimeException("Value " + value + " not found in content matrix.");
    }

    private static int deltaRowFromAction(int actionIndex) {
        return deltaRow(ACTION_BITS[actionIndex]);
    }

    private static int deltaColFromAction(int actionIndex) {
        return deltaCol(ACTION_BITS[actionIndex]);
    }

    private static Pos moveAgent(Pos state, int actionIndex) {
        return new Pos(
                state.row + deltaRowFromAction(actionIndex),
                state.col + deltaColFromAction(actionIndex)
        );
    }

    private static List<Integer> getValidActions(int[][] validMoves, Pos state) {
        List<Integer> actions = new ArrayList<>();

        int moves = validMoves[state.row][state.col];

        for (int actionIndex = 0; actionIndex < ACTION_COUNT; actionIndex++) {
            if (hasDirection(moves, ACTION_BITS[actionIndex])) {
                actions.add(actionIndex);
            }
        }

        return actions;
    }

    private static double getReward(int[][] content, Pos state, Pos goal) {
        if (state.equals(goal)) {
            return 100.0;
        }

        if (content[state.row][state.col] == TRAP) {
            return -20.0;
        }

        return -1.0;
    }

    // ============================================================
    // REWARD MATRIX
    // ============================================================

    private static double[][] createRewardMatrix(int[][] content, Pos goal) {
        double[][] rewardMatrix = new double[ROWS][COLS];

        for (int row = 0; row < ROWS; row++) {
            for (int col = 0; col < COLS; col++) {
                Pos state = new Pos(row, col);

                if (state.equals(goal)) {
                    rewardMatrix[row][col] = 100.0;
                } else if (content[row][col] == TRAP) {
                    rewardMatrix[row][col] = -20.0;
                } else {
                    rewardMatrix[row][col] = -1.0;
                }
            }
        }

        return rewardMatrix;
    }

    private static void printRewardMatrix(double[][] rewardMatrix) {
        System.out.println("\nReward matrix R(s):");
        System.out.println("Each value is the reward received when Jake enters that cell.\n");

        for (int row = 0; row < ROWS; row++) {
            for (int col = 0; col < COLS; col++) {
                System.out.printf("%8.1f", rewardMatrix[row][col]);
            }
            System.out.println();
        }

        System.out.println();
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

        public String toHumanString() {
            return "(" + (row + 1) + ", " + (col + 1) + ")";
        }
    }

    // ============================================================
    // STEP CLASS
    // ============================================================

    static class Step {
        int actionIndex;
        Pos position;

        Step(int actionIndex, Pos position) {
            this.actionIndex = actionIndex;
            this.position = position;
        }
    }

    // ============================================================
    // ACTION CHOICE CLASS
    // ============================================================

    static class ActionChoice {
        Integer actionIndex;
        String decision;

        ActionChoice(Integer actionIndex, String decision) {
            this.actionIndex = actionIndex;
            this.decision = decision;
        }
    }

    // ============================================================
    // MAZE PANEL
    // ============================================================

    static class MazePanel extends JPanel {

        private final int[][] content;
        private final int[][] validMoves;

        private final double[][][] qTable = new double[ROWS][COLS][ACTION_COUNT];

        private final Random random = new Random(7);

        private final Pos start;
        private final Pos goal;

        private Pos state;
        private Pos previousState;
        private Pos agentPosition;

        private final Set<Pos> currentEpisodePath = new HashSet<>();
        private final Set<Pos> finalExecutionVisited = new HashSet<>();

        private final List<Double> rewardHistory = new ArrayList<>();
        private List<Step> learnedPath = new ArrayList<>();

        private Image jakeImage;
        private Image finnImage;

        private Timer timer;

        // Training parameters
        private final int episodesPlanned = 600;
        private final int maxStepsPerEpisode = 80;

        private final double alpha = 0.20;
        private final double gamma = 0.95;
        private final double epsilonMin = 0.05;
        private final double epsilonDecay = 0.985;

        private double epsilon = 1.0;

        private int episode = 1;
        private int episodeStep = 0;
        private double episodeReward = 0.0;

        private int exploreCount = 0;
        private int exploitCount = 0;
        private int successCount = 0;
        private int totalTrainingSteps = 0;

        private boolean episodeEndPending = false;

        private String lastDecision = "START";

        private int executionStepIndex = 0;

        private Phase phase = Phase.TRAINING;

        // Drawing layout
        private final int panelWidth = 1400;
        private final int panelHeight = 760;

        private final int mazeX = 55;
        private final int mazeY = 70;
        private final int cellSize = 90;

        private final int statsX = 545;
        private final int statsY = 70;

        private final int qStartX = 850;
        private final int qStartY = 70;
        private final int qCellSize = 42;
        private final int qGapX = 245;
        private final int qGapY = 300;

        MazePanel(int[][] content, int[][] validMoves) {
            this.content = content;
            this.validMoves = validMoves;

            setPreferredSize(new Dimension(panelWidth, panelHeight));
            setBackground(Color.WHITE);

            this.start = findCell(content, JAKE);
            this.goal = findCell(content, FINN);

            this.state = start;
            this.agentPosition = start;

            currentEpisodePath.add(start);
            finalExecutionVisited.add(start);

            loadImages();
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
            timer = new Timer(25, event -> onTimerTick());
            timer.start();
        }

        private void onTimerTick() {
            if (phase == Phase.TRAINING) {
                trainingTick();
            } else if (phase == Phase.EXECUTION) {
                executionTick();
            }
        }

        private void trainingTick() {
            if (episode > episodesPlanned && !episodeEndPending) {
                finishTraining();
                repaint();
                return;
            }

            if (episodeEndPending) {
                finishEpisode();
                repaint();
                return;
            }

            if (shouldVisualizeEpisode(episode)) {
                doOneTrainingStep();
                repaint();
            } else {
                while (episode <= episodesPlanned && !shouldVisualizeEpisode(episode)) {
                    if (episodeEndPending) {
                        finishEpisode();
                    } else {
                        doOneTrainingStep();
                    }
                }
                repaint();
            }
        }

        private boolean shouldVisualizeEpisode(int episodeNumber) {
            return episodeNumber <= 15 || episodeNumber % 10 == 0;
        }

        private void doOneTrainingStep() {
            if (episodeStep >= maxStepsPerEpisode) {
                markEpisodeEnd(false);
                return;
            }

            ActionChoice choice = chooseActionEpsilonGreedy(state, epsilon);

            if (choice.actionIndex == null) {
                markEpisodeEnd(false);
                return;
            }

            int action = choice.actionIndex;

            if ("EXPLORE".equals(choice.decision)) {
                exploreCount++;
            } else if ("EXPLOIT".equals(choice.decision)) {
                exploitCount++;
            }

            Pos nextState = moveAgent(state, action);
            double reward = getReward(content, nextState, goal);

            List<Integer> nextValidActions = getValidActions(validMoves, nextState);

            double maxNextQ = 0.0;
            if (!nextValidActions.isEmpty()) {
                maxNextQ = Double.NEGATIVE_INFINITY;
                for (int nextAction : nextValidActions) {
                    maxNextQ = Math.max(maxNextQ, qTable[nextState.row][nextState.col][nextAction]);
                }
            }

            double oldQ = qTable[state.row][state.col][action];

            qTable[state.row][state.col][action] =
                    oldQ + alpha * (reward + gamma * maxNextQ - oldQ);

            previousState = state;
            state = nextState;

            currentEpisodePath.add(state);

            episodeStep++;
            episodeReward += reward;
            totalTrainingSteps++;
            lastDecision = choice.decision;

            if (state.equals(goal)) {
                markEpisodeEnd(true);
            } else if (episodeStep >= maxStepsPerEpisode) {
                markEpisodeEnd(false);
            }
        }

        private void markEpisodeEnd(boolean success) {
            if (episodeEndPending) return;

            if (success) {
                successCount++;
            }

            rewardHistory.add(episodeReward);
            episodeEndPending = true;
        }

        private void finishEpisode() {
            episodeEndPending = false;

            episode++;
            epsilon = Math.max(epsilonMin, epsilon * epsilonDecay);

            if (episode <= episodesPlanned) {
                state = start;
                previousState = null;
                currentEpisodePath.clear();
                currentEpisodePath.add(start);
                episodeReward = 0.0;
                episodeStep = 0;
                lastDecision = "START";
            }
        }

        private ActionChoice chooseActionEpsilonGreedy(Pos currentState, double epsilon) {
            List<Integer> validActions = getValidActions(validMoves, currentState);

            if (validActions.isEmpty()) {
                return new ActionChoice(null, "NO_ACTION");
            }

            if (random.nextDouble() < epsilon) {
                int action = validActions.get(random.nextInt(validActions.size()));
                return new ActionChoice(action, "EXPLORE");
            }

            int action = bestAction(currentState);
            return new ActionChoice(action, "EXPLOIT");
        }

        private int bestAction(Pos currentState) {
            List<Integer> validActions = getValidActions(validMoves, currentState);

            double bestValue = Double.NEGATIVE_INFINITY;
            List<Integer> bestActions = new ArrayList<>();

            for (int action : validActions) {
                double value = qTable[currentState.row][currentState.col][action];

                if (value > bestValue) {
                    bestValue = value;
                    bestActions.clear();
                    bestActions.add(action);
                } else if (Math.abs(value - bestValue) < 1e-9) {
                    bestActions.add(action);
                }
            }

            return bestActions.get(random.nextInt(bestActions.size()));
        }

        private List<Step> extractLearnedPath() {
            List<Step> path = new ArrayList<>();

            Pos current = start;
            Set<Pos> visited = new HashSet<>();

            for (int i = 0; i < 100; i++) {
                if (current.equals(goal)) {
                    break;
                }

                if (visited.contains(current)) {
                    System.out.println("Warning: learned policy entered a loop.");
                    break;
                }

                visited.add(current);

                List<Integer> validActions = getValidActions(validMoves, current);
                if (validActions.isEmpty()) {
                    break;
                }

                int action = bestAction(current);
                Pos next = moveAgent(current, action);

                path.add(new Step(action, next));
                current = next;
            }

            return path;
        }

        private void finishTraining() {
            lastDecision = "TRAINING FINISHED";

            learnedPath = extractLearnedPath();

            System.out.println("\nTraining finished.");
            System.out.println("Total training rounds: " + rewardHistory.size());
            System.out.println("Explore count: " + exploreCount);
            System.out.println("Exploit count: " + exploitCount);
            System.out.println("Success count: " + successCount);

            if (learnedPath.isEmpty()) {
                System.out.println("No learned path found. Try increasing the number of episodes.");
                phase = Phase.DONE;
                timer.stop();
                return;
            }

            System.out.println("\nLearned path:");
            for (Step step : learnedPath) {
                System.out.println(ACTION_NAMES[step.actionIndex] + " -> " + step.position.toHumanString());
            }

            phase = Phase.EXECUTION;
            timer.setDelay(700);

            agentPosition = start;
            previousState = null;
            finalExecutionVisited.clear();
            finalExecutionVisited.add(start);
            executionStepIndex = 0;
        }

        private void executionTick() {
            if (executionStepIndex < learnedPath.size()) {
                Step step = learnedPath.get(executionStepIndex);

                previousState = agentPosition;
                agentPosition = step.position;
                finalExecutionVisited.add(agentPosition);

                lastDecision = "EXECUTING " + ACTION_NAMES[step.actionIndex];

                executionStepIndex++;
                repaint();
            } else {
                phase = Phase.DONE;
                lastDecision = "GOAL REACHED";
                repaint();
                timer.stop();
            }
        }

        // ============================================================
        // DRAWING
        // ============================================================

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);

            Graphics2D g2 = (Graphics2D) g;

            g2.setRenderingHint(
                    RenderingHints.KEY_ANTIALIASING,
                    RenderingHints.VALUE_ANTIALIAS_ON
            );

            Pos positionToDraw;
            Set<Pos> pathToDraw;

            if (phase == Phase.TRAINING) {
                positionToDraw = state;
                pathToDraw = currentEpisodePath;
            } else {
                positionToDraw = agentPosition;
                pathToDraw = finalExecutionVisited;
            }

            drawMaze(g2, positionToDraw, pathToDraw);
            drawStats(g2);
            drawQTablePanel(g2, positionToDraw);
        }

        private void drawMaze(Graphics2D g2, Pos currentPosition, Set<Pos> path) {
            drawCellBackgrounds(g2, currentPosition, path);
            drawPolicyArrows(g2);
            drawLastTransitionArrow(g2);
            drawThickWalls(g2);
            drawGoal(g2);
            drawAgent(g2, currentPosition);
            drawCoordinates(g2);

            g2.setColor(Color.BLACK);
            g2.setFont(new Font("Arial", Font.BOLD, 16));

            String title;
            if (phase == Phase.TRAINING) {
                title = "Q-Learning exploration | Episode " +
                        Math.min(episode, episodesPlanned) + "/" + episodesPlanned;
            } else {
                title = "Final learned policy | Step " + executionStepIndex;
            }

            g2.drawString(title, mazeX, 35);
        }

        private void drawCellBackgrounds(Graphics2D g2, Pos currentPosition, Set<Pos> path) {
            for (int row = 0; row < ROWS; row++) {
                for (int col = 0; col < COLS; col++) {
                    Pos pos = new Pos(row, col);

                    Color color;

                    if (pos.equals(goal)) {
                        color = new Color(210, 255, 210);
                    } else if (pos.equals(currentPosition)) {
                        color = Color.WHITE;
                    } else if (path.contains(pos)) {
                        color = new Color(255, 235, 120);
                    } else if (content[row][col] == TRAP) {
                        color = new Color(255, 180, 180);
                    } else {
                        color = Color.WHITE;
                    }

                    int x = mazeX + col * cellSize;
                    int y = mazeY + row * cellSize;

                    g2.setColor(color);
                    g2.fillRect(x, y, cellSize, cellSize);

                    g2.setColor(Color.LIGHT_GRAY);
                    g2.drawRect(x, y, cellSize, cellSize);
                }
            }
        }

        private void drawCoordinates(Graphics2D g2) {
            g2.setColor(Color.BLACK);
            g2.setFont(new Font("Arial", Font.BOLD, 18));

            for (int col = 0; col < COLS; col++) {
                int x = mazeX + col * cellSize + cellSize / 2 - 5;
                int y = mazeY - 20;
                g2.drawString(String.valueOf(col + 1), x, y);
            }

            for (int row = 0; row < ROWS; row++) {
                int x = mazeX - 30;
                int y = mazeY + row * cellSize + cellSize / 2 + 7;
                g2.drawString(String.valueOf(row + 1), x, y);
            }
        }

        private void drawThickWalls(Graphics2D g2) {
            Stroke oldStroke = g2.getStroke();

            g2.setColor(Color.BLACK);
            g2.setStroke(new BasicStroke(6));

            for (int row = 0; row < ROWS; row++) {
                for (int col = 0; col < COLS; col++) {
                    int x = mazeX + col * cellSize;
                    int y = mazeY + row * cellSize;

                    int moves = validMoves[row][col];

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

        private void drawPolicyArrows(Graphics2D g2) {
            g2.setColor(Color.GRAY);
            g2.setStroke(new BasicStroke(2));

            for (int row = 0; row < ROWS; row++) {
                for (int col = 0; col < COLS; col++) {
                    Pos pos = new Pos(row, col);

                    if (pos.equals(goal)) continue;

                    List<Integer> validActions = getValidActions(validMoves, pos);
                    if (validActions.isEmpty()) continue;

                    double bestValue = Double.NEGATIVE_INFINITY;
                    int bestAction = -1;

                    for (int action : validActions) {
                        double value = qTable[row][col][action];

                        if (value > bestValue) {
                            bestValue = value;
                            bestAction = action;
                        }
                    }

                    if (Math.abs(bestValue) < 1e-9) continue;

                    int centerX = mazeX + col * cellSize + cellSize / 2;
                    int centerY = mazeY + row * cellSize + cellSize / 2;

                    int dx = deltaColFromAction(bestAction) * 22;
                    int dy = deltaRowFromAction(bestAction) * 22;

                    drawArrow(g2, centerX, centerY, centerX + dx, centerY + dy, Color.GRAY, 2);
                }
            }
        }

        private void drawLastTransitionArrow(Graphics2D g2) {
            if (previousState == null) return;

            Pos current;

            if (phase == Phase.TRAINING) {
                current = state;
            } else {
                current = agentPosition;
            }

            int x1 = mazeX + previousState.col * cellSize + cellSize / 2;
            int y1 = mazeY + previousState.row * cellSize + cellSize / 2;

            int x2 = mazeX + current.col * cellSize + cellSize / 2;
            int y2 = mazeY + current.row * cellSize + cellSize / 2;

            Color color;

            if ("EXPLORE".equals(lastDecision)) {
                color = Color.ORANGE;
            } else {
                color = new Color(128, 0, 128);
            }

            drawArrow(g2, x1, y1, x2, y2, color, 4);
        }

        private void drawArrow(Graphics2D g2, int x1, int y1, int x2, int y2, Color color, int thickness) {
            Stroke oldStroke = g2.getStroke();

            g2.setColor(color);
            g2.setStroke(new BasicStroke(thickness));

            g2.drawLine(x1, y1, x2, y2);

            double angle = Math.atan2(y2 - y1, x2 - x1);
            int arrowSize = 10;

            int xArrow1 = (int) (x2 - arrowSize * Math.cos(angle - Math.PI / 6));
            int yArrow1 = (int) (y2 - arrowSize * Math.sin(angle - Math.PI / 6));

            int xArrow2 = (int) (x2 - arrowSize * Math.cos(angle + Math.PI / 6));
            int yArrow2 = (int) (y2 - arrowSize * Math.sin(angle + Math.PI / 6));

            g2.drawLine(x2, y2, xArrow1, yArrow1);
            g2.drawLine(x2, y2, xArrow2, yArrow2);

            g2.setStroke(oldStroke);
        }

        private void drawAgent(Graphics2D g2, Pos position) {
            int x = mazeX + position.col * cellSize;
            int y = mazeY + position.row * cellSize;

            if (jakeImage != null) {
                g2.drawImage(jakeImage, x + 12, y + 12, cellSize - 24, cellSize - 24, null);
            } else {
                g2.setColor(Color.BLUE);
                g2.fillOval(x + 22, y + 22, cellSize - 44, cellSize - 44);

                g2.setColor(Color.WHITE);
                g2.setFont(new Font("Arial", Font.BOLD, 20));
                g2.drawString("J", x + cellSize / 2 - 6, y + cellSize / 2 + 7);
            }
        }

        private void drawGoal(Graphics2D g2) {
            int x = mazeX + goal.col * cellSize;
            int y = mazeY + goal.row * cellSize;

            if (finnImage != null) {
                g2.drawImage(finnImage, x + 12, y + 12, cellSize - 24, cellSize - 24, null);
            } else {
                g2.setColor(Color.GREEN.darker());
                g2.fillOval(x + 22, y + 22, cellSize - 44, cellSize - 44);

                g2.setColor(Color.WHITE);
                g2.setFont(new Font("Arial", Font.BOLD, 20));
                g2.drawString("F", x + cellSize / 2 - 6, y + cellSize / 2 + 7);
            }
        }

        private void drawStats(Graphics2D g2) {
            g2.setColor(Color.BLACK);
            g2.setFont(new Font("Arial", Font.BOLD, 18));
            g2.drawString("Training information", statsX, statsY - 30);

            drawRewardChart(g2);

            g2.setFont(new Font("Arial", Font.PLAIN, 14));

            int completedRounds = rewardHistory.size();

            int y = statsY + 190;
            int line = 22;

            g2.drawString("Current episode: " + Math.min(episode, episodesPlanned), statsX, y);
            y += line;

            g2.drawString("Training rounds done: " + completedRounds, statsX, y);
            y += line;

            g2.drawString("Total rounds planned: " + episodesPlanned, statsX, y);
            y += line;

            g2.drawString("Training steps done: " + totalTrainingSteps, statsX, y);
            y += line * 2;

            g2.drawString("Current episode step: " + episodeStep, statsX, y);
            y += line;

            g2.drawString(String.format("Current episode reward: %.1f", episodeReward), statsX, y);
            y += line;

            g2.drawString(String.format("Epsilon: %.3f", epsilon), statsX, y);
            y += line * 2;

            g2.drawString("Last decision: " + lastDecision, statsX, y);
            y += line;

            g2.drawString("Explore count: " + exploreCount, statsX, y);
            y += line;

            g2.drawString("Exploit count: " + exploitCount, statsX, y);
            y += line;

            g2.drawString("Success count: " + successCount, statsX, y);
        }

        private void drawRewardChart(Graphics2D g2) {
            int x = statsX;
            int y = statsY;
            int width = 260;
            int height = 140;

            g2.setColor(Color.WHITE);
            g2.fillRect(x, y, width, height);

            g2.setColor(Color.BLACK);
            g2.drawRect(x, y, width, height);

            g2.setFont(new Font("Arial", Font.PLAIN, 11));
            g2.drawString("Reward by episode", x + 5, y + 15);

            if (rewardHistory.isEmpty()) {
                return;
            }

            double min = Double.POSITIVE_INFINITY;
            double max = Double.NEGATIVE_INFINITY;

            for (double value : rewardHistory) {
                min = Math.min(min, value);
                max = Math.max(max, value);
            }

            if (Math.abs(max - min) < 1e-9) {
                max += 1.0;
                min -= 1.0;
            }

            int n = rewardHistory.size();

            g2.setColor(new Color(40, 90, 180));
            g2.setStroke(new BasicStroke(2));

            int previousX = x;
            int previousY = y + height - scaleReward(rewardHistory.get(0), min, max, height);

            for (int i = 1; i < n; i++) {
                int currentX = x + (int) ((double) i / Math.max(1, n - 1) * width);
                int currentY = y + height - scaleReward(rewardHistory.get(i), min, max, height);

                g2.drawLine(previousX, previousY, currentX, currentY);

                previousX = currentX;
                previousY = currentY;
            }

            g2.setStroke(new BasicStroke(1));
        }

        private int scaleReward(double value, double min, double max, int height) {
            double normalized = (value - min) / (max - min);
            return (int) (normalized * (height - 25)) + 10;
        }

        // ============================================================
        // Q-TABLE DRAWING
        // ============================================================

        private void drawQTablePanel(Graphics2D g2, Pos currentState) {
            drawQMatrix(g2, 0, qStartX, qStartY, currentState, "Q(s, UP)");
            drawQMatrix(g2, 1, qStartX + qGapX, qStartY, currentState, "Q(s, RIGHT)");
            drawQMatrix(g2, 2, qStartX, qStartY + qGapY, currentState, "Q(s, DOWN)");
            drawQMatrix(g2, 3, qStartX + qGapX, qStartY + qGapY, currentState, "Q(s, LEFT)");
        }

        private void drawQMatrix(Graphics2D g2, int actionIndex, int startX, int startY, Pos currentState, String title) {
            double[] minMax = getMinMaxValidQ();
            double min = minMax[0];
            double max = minMax[1];

            if (Math.abs(max - min) < 1e-9) {
                max += 1.0;
                min -= 1.0;
            }

            g2.setColor(Color.BLACK);
            g2.setFont(new Font("Arial", Font.BOLD, 14));
            g2.drawString(title, startX, startY - 12);

            for (int row = 0; row < ROWS; row++) {
                for (int col = 0; col < COLS; col++) {
                    Pos state = new Pos(row, col);
                    List<Integer> validActions = getValidActions(validMoves, state);

                    int x = startX + col * qCellSize;
                    int y = startY + row * qCellSize;

                    if (!validActions.contains(actionIndex)) {
                        g2.setColor(Color.LIGHT_GRAY);
                        g2.fillRect(x, y, qCellSize, qCellSize);

                        g2.setColor(Color.BLACK);
                        g2.drawRect(x, y, qCellSize, qCellSize);
                        g2.drawString("X", x + qCellSize / 2 - 4, y + qCellSize / 2 + 5);
                    } else {
                        double value = qTable[row][col][actionIndex];
                        Color color = valueToColor(value, min, max);

                        g2.setColor(color);
                        g2.fillRect(x, y, qCellSize, qCellSize);

                        g2.setColor(Color.BLACK);
                        g2.drawRect(x, y, qCellSize, qCellSize);

                        g2.setFont(new Font("Arial", Font.PLAIN, 10));
                        g2.drawString(String.format("%.1f", value), x + 6, y + qCellSize / 2 + 4);
                    }

                    if (currentState.row == row && currentState.col == col) {
                        Stroke oldStroke = g2.getStroke();
                        g2.setColor(Color.RED);
                        g2.setStroke(new BasicStroke(3));
                        g2.drawRect(x + 2, y + 2, qCellSize - 4, qCellSize - 4);
                        g2.setStroke(oldStroke);
                    }
                }
            }
        }

        private double[] getMinMaxValidQ() {
            double min = Double.POSITIVE_INFINITY;
            double max = Double.NEGATIVE_INFINITY;

            for (int row = 0; row < ROWS; row++) {
                for (int col = 0; col < COLS; col++) {
                    Pos state = new Pos(row, col);
                    List<Integer> validActions = getValidActions(validMoves, state);

                    for (int action : validActions) {
                        double value = qTable[row][col][action];

                        min = Math.min(min, value);
                        max = Math.max(max, value);
                    }
                }
            }

            if (Double.isInfinite(min) || Double.isInfinite(max)) {
                return new double[]{-1.0, 1.0};
            }

            return new double[]{min, max};
        }

        private Color valueToColor(double value, double min, double max) {
            double t = (value - min) / (max - min);
            t = Math.max(0.0, Math.min(1.0, t));

            int r = (int) (40 + t * 200);
            int g = (int) (80 + t * 160);
            int b = (int) (160 - t * 120);

            return new Color(r, g, b);
        }
    }

    enum Phase {
        TRAINING,
        EXECUTION,
        DONE
    }
}