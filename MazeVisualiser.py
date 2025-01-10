import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

class MazeVisualizer:
    def __init__(self, maze):
        self.maze = maze
        self.height, self.width = maze.shape
        import matplotlib as mpl
        self.palette = mpl.cm.inferno.resampled(4).colors
        self.labels = ["0: unfilled", "1: wall", "2: passage", "3: agent/solution"]

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

    def visualize_path(self, came_from, start, goal):
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
        plt.figure(figsize=(7, 4))
        plt.plot(rewards, label="Rewards")
        plt.title(f"{algorithm_name}: Rewards per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(7, 4))
        plt.plot(steps, label="Steps")
        plt.title(f"{algorithm_name}: Steps per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_agent_run(self, q_table, start, goal, get_next_state_func):
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

    # If you want to visualize BFS perimeter cells
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
