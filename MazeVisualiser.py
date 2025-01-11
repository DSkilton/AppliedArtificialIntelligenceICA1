import datetime, os, logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np

class MazeVisualiser:

    def __init__(self, maze):
        self.logger = logging.getLogger(__name__)

        self.maze = maze
        self.height, self.width = maze.shape
        self.palette = mpl.cm.inferno.resampled(5).colors

        # Label them: 0=unfilled,1=wall,2=passage,3=agent,4=bfs
        self.labels = ["0: unfilled", "1: wall", "2: passage", "3: agent/solution", "4: bfs"]

    def show_maze(self, title="Generated Maze"):
        plt.figure(figsize=(7, 7))
        plt.imshow(self.palette[self.maze])
        patches = [
            mpatches.Patch(color=color, label=label)
            for color, label in zip(self.palette, self.labels)
        ]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2)
        plt.title(title)
        plt.show()

    def visualise_path(self, came_from, start, goal):
        self.logger.info(f"[visualise_path] -> came_from: {came_from}")
        maze_copy = np.copy(self.maze)
        current = goal
        while current and current != start:
            maze_copy[current] = 3
            current = came_from.get(current, None)
        plt.figure(figsize=(7, 7))
        plt.imshow(maze_copy)
        plt.title("A* Path")
        plt.show()

    def plot_rewards_and_steps(self, rewards, steps, algorithm_name):
        # create timestamped directory
        self.logger.info(f"[plot_rewards_and_steps] -> algorithm_name: {algorithm_name}, rewards size: {len(rewards)}, steps size: {len(steps)}")

        now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = os.path.join(".", now_str)
        os.makedirs(out_dir, exist_ok=True)
    
        plt.figure(figsize=(7, 4))
        plt.plot(rewards, label="Rewards")
        plt.title(f"{algorithm_name}: Rewards per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.show()
        out_file1 = os.path.join(out_dir, f"{algorithm_name}_rewards.png")
        plt.savefig(out_file1, dpi=150)

        plt.figure(figsize=(7, 4))
        plt.plot(steps, label="Steps")
        plt.title(f"{algorithm_name}: Steps per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.legend()
        plt.grid(True)
        plt.show()
        out_file1 = os.path.join(out_dir, f"{algorithm_name}_rewards.png")
        plt.savefig(out_file1, dpi=150)

    def visualise_agent_run(self, q_table, start, goal, get_next_state_func):
        maze_copy = np.copy(self.maze)

        plt.ion()
        fig, ax = plt.subplots(figsize=(7,7))
        current_state = start

        while current_state != goal:
            maze_copy[current_state] = 3
            ax.clear()
            ax.imshow(maze_copy)
            plt.title("Agent Running")
            plt.draw()
            plt.pause(0.2)

            maze_copy[current_state] = 2

            action = np.argmax(q_table[current_state[0], current_state[1]])
            next_state = get_next_state_func(current_state, action)
            if next_state == current_state:
                # stuck
                break
            current_state = next_state

        maze_copy[current_state] = 3
        ax.clear()
        ax.imshow(maze_copy)
        plt.title("Agent Reached Goal")
        plt.draw()
        plt.pause(1.0)
        plt.ioff()
        plt.show()

    # If you want to visualise BFS perimeter cells
    def visualise_perimeter_cells(self, perimeter_cells):
        """
        Color any perimeter BFS visited cells in '3' and show them.
        """
        maze_copy = np.copy(self.maze)
        for cell in perimeter_cells:
            maze_copy[cell] = 3

        plt.figure(figsize=(7,7))
        plt.imshow(self.palette[maze_copy])
        plt.title("Perimeter BFS Visited Cells")
        plt.show()

    @staticmethod
    def visualise_agent_run_multiple_openings(q_table, openings, env, visualiser):
        """
        Using the final Q-table, run the agent from each 'opening' to the environment's goal,
        calling 'visualiser.visualise_agent_run' for each route.
        """
        _, goal = env.get_start_and_goal()
        if goal is None:
            print("[visualise_agent_run_multiple_openings] No valid goal in environment.")
            return

        for open_cell in openings:
            print(f"** Visualizing agent from {open_cell} to {goal} **")
            visualiser.visualise_agent_run(
                q_table,
                open_cell,
                goal,
                lambda s,a: env.step(s,a)[0]
            )

    @staticmethod
    def visualise_bfs_paths_from_openings(solver, visualiser, openings):
        """
        For each opening in 'openings', run BFS from that opening to the real goal,
        and visualise the resulting path if found.
        """
        env = solver.env
        _, goal = env.get_start_and_goal()
        if goal is None:
            print("[visualise_bfs_paths_from_openings] No valid goal found in MazeEnvironment.")
            return

        print(f"[visualise_bfs_paths_from_openings] We have {len(openings)} openings to try.")
        for open_cell in openings:
            print(f"--- BFS from opening {open_cell} to {goal} ---")
            came_from = solver.solve_bfs_custom(open_cell, goal)  
            if came_from is not None:
                # Visualize the path from open_cell to goal
                visualiser.visualise_path(came_from, open_cell, goal)
            else:
                print(f"No BFS path from {open_cell} to {goal}")

    def visualise_bfs_and_agent_path(self, bfs_visited, agent_states):
        """
        BFS visited => color=4, agent path => color=3
        """
        maze_copy = np.copy(self.maze)

        # Mark BFS visited
        for (x, y) in bfs_visited:
            maze_copy[y, x] = 4

        # Mark agent path
        for (x, y) in agent_states:
            maze_copy[y, x] = 3

        plt.figure(figsize=(7,7))
        plt.imshow(self.palette[maze_copy])
        plt.title("BFS visited (color=4) + Agent path (color=3)")
        plt.show()

    def visualise_bfs_path(self, bfs_visited):
        """
        Visualize BFS traversal paths for educational purposes.
        :param bfs_visited: List of cells visited by BFS.
        """
        maze_copy = np.copy(self.maze)
        for cell in bfs_visited:
            maze_copy[cell] = 4  # Color code for BFS visited cells

        plt.figure(figsize=(7, 7))
        plt.imshow(self.palette[maze_copy])
        plt.title("BFS Traversal")
        plt.show()
