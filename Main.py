import logging
import numpy as np
from MazeGenerator import MazeGenerator
from MazeEnvironment import MazeEnvironment
from MazeSolver import MazeSolver
from MazeVisualiser import MazeVisualizer

from Persistence import QTablePersistence

from BfsManagerReverse import BFSManagerReverse
from BfsManager import BfsManager
from BfsRlManager import BfandRlMananger
from Constants import *

def main():

    # Maze + Environment
    env, maze_array, generator = create_maze_and_env(height=20, width=20, openings=5)

    bfs_mgr = BfsManager(env)
    q_table = np.zeros((env.height, env.width, 4), dtype=float)

    manager = BfandRlMananger(env, bfs_mgr, q_table)
    final_q, expansions_log, reward_log, step_log = manager.run_bfs_and_rl(
        episodes=1000,
        max_steps_per_episode=500,
        batch_size=10
    )

    logging.info("[Main] BFS+RL synergy run complete.")
    logging.info(f"Expansions log: {expansions_log}")
    logging.info(f"Episode Rewards: {reward_log}")
    logging.info(f"Episode Steps: {step_log}")

    # MazeSolver, MazeVisualizer
    solver = MazeSolver(env)
    visualizer = MazeVisualizer(maze_array)

    start, goal = env.get_start_and_goal()
    if start and goal:
        visualizer.visualize_agent_run(final_q, start, goal, lambda s,a: env.step(s,a)[0])
    else:
        logging.info("No valid start/goal to visualize final path.")

    # Load or create Q-learning / SARSA tables
    # q_learning_table, sarsa_table = load_qtables(env, generator)

    # Solve Maze with A*
    run_astar_and_visualize(env, solver, visualizer)

    # Solve with Q-learning
    run_qlearning_and_visualize(env, solver, visualizer)

    # Solve with SARSA
    run_sarsa_and_visualize(env, solver, visualizer)

    # Visualize BFS perimeter
    visualize_boundary_bfs(solver, visualizer)

    perimeter_visited, boundary_openings = solver.perimeter_bfs()
    MazeVisualizer.visualize_bfs_paths_from_openings(solver, visualizer, boundary_openings)
    MazeVisualizer.visualize_agent_run_multiple_openings(final_q, boundary_openings, env, visualizer)    

    start, _ = env.get_start_and_goal()
    state = start

    agent_states = []
    steps = 0
    done = False
    max_steps = 500
    while not done and steps < max_steps:
        agent_states.append(state)
        # e-greedy using final_q:
        if np.random.rand() < 0.1:
            action = np.random.randint(4)
        else:
            action = np.argmax(final_q[state[0], state[1]])
        next_state, r, done = env.step(state, action)
        state = next_state
        steps += 1

    # BFS visited cells
    bfs_visited = list(bfs_mgr.dist_goal.keys())  # all cells BFS discovered

    # 6) Visualize BFS visited + agent path
    visualizer = MazeVisualizer(maze_array)
    visualizer.visualise_bfs_and_agent_path(bfs_visited, agent_states)

def create_maze_and_env(height=20, width=20, openings=5):
    """
    Create a MazeGenerator, generate the maze array, then create MazeEnvironment.
    Returns (env, maze_array, generator).
    """
    generator = MazeGenerator(height=height, width=width, openings=openings)
    maze_array = generator.generate_maze()
    env = MazeEnvironment(maze_array)
    return env, maze_array, generator

def run_bfs_rl_manager(env):
    """
    Creates BFSManager, Q-table, and calls the BFS+RL manager logic.
    Returns final_q, expansions_log, reward_log, step_log
    """
    bfs_mgr = BfsManager(env)
    q_table = np.zeros((env.height, env.width, 4), dtype=float)

    final_q, expansions_log, reward_log, step_log = bfs_mgr.run_bfs_and_rl(
        env,
        bfs_mgr,
        q_table,
        episodes=1000,
        iters_per_episode=1000,
        batch_size=10
    )

    return final_q, expansions_log, reward_log, step_log

def expand_bfs_reverse(bfs_mgr):
    """
    Optionally expand BFS in increments and log progress.
    """
    while not bfs_mgr.is_finished_goal():
        processed = bfs_mgr.expand_next_batch_reverse(batch_size=20)
        print(f"Expanded BFS by {processed} states...")
    print("BFS done. Visited states count:", len(bfs_mgr.visited))

def load_qtables(env, generator):
    """
    Attempt to load Q-tables for Q-learning and SARSA from disk.
    If not found, create new ones of the correct shape.
    Returns (q_learning_table, sarsa_table).
    """
    q_learning_table = QTablePersistence.load(Q_LEARNING_FILE)
    sarsa_table = QTablePersistence.load(SARSA_FILE)

    if q_learning_table is None:
        q_learning_table = np.zeros((generator.height, generator.width, 4))
    if sarsa_table is None:
        sarsa_table = np.zeros((generator.height, generator.width, 4))
    return q_learning_table, sarsa_table

def run_astar_and_visualize(env, solver, visualizer):
    """
    Solve the maze with A* and visualize the path.
    """
    a_star_rewards, a_star_steps, came_from = solver.solve_astar()
    visualizer.show_maze("Original Maze")
    start, goal = env.get_start_and_goal()
    visualizer.visualize_path(came_from, start, goal)

def run_qlearning_and_visualize(env, solver, visualizer):
    """
    Solve using Q-learning, log results, and visualize the agent.
    """
    print("\n--- Starting Q-Learning (100,000 episodes) ---")
    q_rewards, q_steps, q_table = solver.solve_qlearning(
        episodes=100000, log_interval=10000, save_filename=Q_LEARNING_FILE
    )
    visualizer.plot_rewards_and_steps(q_rewards, q_steps, Q_LEARNING)

    start, goal = env.get_start_and_goal()
    visualizer.visualize_agent_run(
        q_table,
        start,
        goal,
        lambda s, a: env.step(s, a)[0]
    )

def run_sarsa_and_visualize(env, solver, visualizer):
    """
    Solve using SARSA, log results, and visualize.
    """
    print("\n--- Starting SARSA (100,000 episodes) ---")
    sarsa_rewards, sarsa_steps, sarsa_table = solver.solve_sarsa(
        episodes=100000, log_interval=10000, save_filename=SARSA_FILE
    )
    visualizer.plot_rewards_and_steps(sarsa_rewards, sarsa_steps, SARSA)

    start, goal = env.get_start_and_goal()
    visualizer.visualize_agent_run(
        sarsa_table,
        start,
        goal,
        lambda s, a: env.step(s, a)[0]
    )

def visualize_boundary_bfs(solver, visualizer):
    """
    Run the perimeter BFS in MazeSolver, log it, visualize results.
    """
    perimeter_visited, boundary_openings = solver.perimeter_bfs()

    logging.info(f"Visited set: {perimeter_visited}")
    logging.info(f"Openings: {boundary_openings}")

    # If you want to visualize *only* perimeter-connected cells:
    visualizer.visualise_perimeter_cells(perimeter_visited)

    # If you also want to highlight only the boundary-edge openings:
    visualizer.visualise_perimeter_cells(boundary_openings)

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    main()