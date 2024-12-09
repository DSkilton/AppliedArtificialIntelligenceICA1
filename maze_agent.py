import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
from skimage.segmentation import flood_fill

class MazeGenerator:
    """A class for generating and visualising mazes."""

    def __init__(self, height, width):
        """
        Initialize the maze generator with height and width.

        Args:
            height (int): The height of the maze grid.
            width (int): The width of the maze grid.
        """
        self.height = height
        self.width = width
        self.maze = None
        self.palette = mpl.cm.inferno.resampled(3).colors
        self.labels = ["0: unfilled", "1: wall", "2: passage"]


    def create_empty_maze(self):
        """
        Create an empty maze with walls and start/end passages.

        Returns:
            np.ndarray: A grid representing the initial maze.
        """
        maze = np.zeros((self.height, self.width), dtype=np.uint8)
        # Define walls: top, bottom, left and right
        maze[0] = maze[-1] = maze[:, 0] = maze[:, -1] = 1
        # Define start(top-left) and end (bottom-right) passages
        maze[0, 1] = maze[-1, -2] = 2
        return maze
    

    def generate_mask(self):
        """
        Generate a mask for neighbor cells above, below, left, and right.

        Returns:
            np.ndarray: A mask used for checking neighbor cells.
        """
        return np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    

    def generate_maze(self, maze):
        """
        Generate a solvable maze using randomized filling and flood-fill for validation.

        Args:
            maze (np.ndarray): The initial maze grid.

        Returns:
            np.ndarray: The completed maze with paths carved out.
        """
        unfilled_cells = np.swapaxes(np.where(maze == 0), 0, 1)
        np.random.shuffle(unfilled_cells)

        start = tuple(coord[0] for coord in np.where(maze == 2))
        maze = np.copy(maze)
        maze[maze == 2] = 0

        neighbours_mask = self.generate_mask()

        for cell in unfilled_cells:
            y, x = tuple(cell)

            # Prevent creating dead-end walls
            if np.sum(neighbours_mask * maze[y - 1:y + 2, x - 1: x + 2]) > 2:
                continue

            maze[y,x] = 1

            # Check connectivity using flood-fill
            test_maze = flood_fill(maze, start, 1, connectivity=1)
            if np.any(test_maze == 0):
                maze[y, x] = 0

        # Mark unfilled cells as passages
        maze[maze == 0] = 2
        return maze
        

    def visualise_maze(self, maze):
        """
        visualise the maze using Matplotlib.

        Args:
            maze (np.ndarray): The maze grid to visualise.
        """
        plt.figure(figsize=(9, 9))
        image = plt.imshow(self.palette[maze])

        # Create a legend
        patches = [
            mpatches.Patch(color=color, label=label)
            for color, label in zip(self.palette, self.labels)
        ]
        plt.legend(handles=patches, bbox_to_anchor=(1.1, 1), loc=2, borderaxespad = 0)
        plt.show()

    def generate_and_visualise_maze(self):
        """
        Generate and visualise a maze.
        """
        initial_maze = self.create_empty_maze()
        self.maze = self.generate_maze(initial_maze)
        self.visualise_maze(self.maze)

if __name__ == "__main__":
    generator = MazeGenerator(height=30, width=30)
    generator.generate_and_visualise_maze()
