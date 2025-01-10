import logging
import numpy as np
from MazeGenerator import MazeGenerator
from MazeEnvironment import MazeEnvironment
from MazeSolver import MazeSolver
from MazeVisualiser import MazeVisualizer
from Persistence import QTablePersistence
from Constants import *

def main():
    # Initialize MazeGenerator
    generator = MazeGenerator(height=20, width=20, openings=5)
    maze_array = generator.generate_maze()

    # Initialize environment and solver
    env = MazeEnvironment(maze_array)
    solver = MazeSolver(env)
    visualizer = MazeVisualizer(maze_array)

    # Attempt to load Q-tables
    q_learning_table = QTablePersistence.load(Q_LEARNING_FILE)
    sarsa_table = QTablePersistence.load(SARSA_FILE)

    # If tables are not available, initialize them dynamically based on the maze dimensions
    if q_learning_table is None:
        q_learning_table = np.zeros((generator.height, generator.width, 4))
    if sarsa_table is None:
        sarsa_table = np.zeros((generator.height, generator.width, 4))

    # Solve the maze using A*
    a_star_rewards, a_star_steps, came_from = solver.solve_astar()
    visualizer.show_maze("Original Maze")
    visualizer.visualize_path(came_from, *env.get_start_and_goal())

    # Solve the maze using Q-Learning
    print("\n--- Starting Q-Learning (10,000 episodes) ---")
    q_rewards, q_steps, q_table = solver.solve_qlearning(
        episodes=10000, log_interval=500, save_filename=Q_LEARNING_FILE
    )
    visualizer.plot_rewards_and_steps(q_rewards, q_steps, Q_LEARNING)
    start, goal = env.get_start_and_goal()
    visualizer.visualize_agent_run(
        q_table,
        start,
        goal,
        lambda s, a: env.step(s, a)[0]
    )

    # Solve the maze using SARSA
    print("\n--- Starting SARSA (10,000 episodes) ---")
    sarsa_rewards, sarsa_steps, sarsa_table = solver.solve_sarsa(
        episodes=10000, log_interval=500, save_filename=SARSA_FILE
    )
    visualizer.plot_rewards_and_steps(sarsa_rewards, sarsa_steps, SARSA)
    visualizer.visualize_agent_run(
        sarsa_table,
        start,
        goal,
        lambda s, a: env.step(s, a)[0]
    )

    # Visualize BFS perimeter
    perimeter_cells = solver.perimeter_bfs()
    logging.info(f"Perimeter cells visited: {perimeter_cells}")
    visualizer.visualise_perimeter_cells(perimeter_cells)

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    main()
