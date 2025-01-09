from MazeGenerator import MazeGenerator
from MazeEnvironment import MazeEnvironment
from MazeSolver import MazeSolver
from MazeVisualiser import MazeVisualizer

def main():
    # Maze setup
    generator = MazeGenerator(height=20, width=20, openings=1)
    maze_array = generator.generate_maze()

    env = MazeEnvironment(maze_array)
    solver = MazeSolver(env)
    visualizer = MazeVisualizer(maze_array)

    # A* once
    a_star_rewards, a_star_steps, came_from = solver.solve_astar()
    visualizer.show_maze("Original Maze")
    visualizer.visualize_path(came_from, *env.get_start_and_goal())

    # Q-Learning
    # -----------
    print("\n--- Starting Q-Learning (10,000 episodes) ---")
    q_rewards, q_steps, q_table = solver.solve_qlearning(episodes=10000, log_interval=500)
    visualizer.plot_rewards_and_steps(q_rewards, q_steps, "Q-Learning")
    start, goal = env.get_start_and_goal()
    # Animate agent run with final Q-table
    visualizer.visualize_agent_run(
        q_table,
        start,
        goal,
        lambda s, a: env.step(s, a)[0]
    )

    # SARSA
    # -----------
    print("\n--- Starting SARSA (10,000 episodes) ---")
    sarsa_rewards, sarsa_steps, sarsa_table = solver.solve_sarsa(episodes=10000, log_interval=500)
    visualizer.plot_rewards_and_steps(sarsa_rewards, sarsa_steps, "SARSA")
    start, goal = env.get_start_and_goal()
    # Animate agent run with final SARSA Q-table
    visualizer.visualize_agent_run(
        sarsa_table,
        start,
        goal,
        lambda s, a: env.step(s, a)[0]
    )

    # Optional: Show a perimeter BFS or perimeter “scan”
    # Uncomment below if you want a demonstration of perimeter exploration
    # perimeter_cells = solver.perimeter_bfs()  # We'll define in MazeSolver or somewhere
    # visualizer.visualize_perimeter_cells(perimeter_cells)

if __name__ == "__main__":
    main()
