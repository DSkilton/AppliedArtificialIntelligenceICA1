import numpy as np
from skimage.segmentation import flood_fill

class MazeGenerator:
    def __init__(self, height, width, openings=1):
        self.height = height
        self.width = width
        self.openings = openings

    def create_empty_maze(self):
        maze = np.zeros((self.height, self.width), dtype=np.uint8)
        maze[0] = maze[-1] = maze[:, 0] = maze[:, -1] = 1
        maze[0, 1] = 2
        maze[-1, -2] = 2
        return maze

    def generate_mask(self):
        return np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ], dtype=np.uint8)

    def generate_maze(self):
        maze = self.create_empty_maze()
        unfilled_cells = np.swapaxes(np.where(maze == 0), 0, 1)
        np.random.shuffle(unfilled_cells)

        start_tuple = tuple(coord[0] for coord in np.where(maze == 2))
        # Temporarily mark “2” as “0”
        maze[maze == 2] = 0

        neighbours_mask = self.generate_mask()

        for (y, x) in unfilled_cells:
            # If blocking a cell kills connectivity, revert
            if np.sum(neighbours_mask * maze[y - 1:y + 2, x - 1:x + 2]) > 2:
                continue

            maze[y, x] = 1
            test_maze = flood_fill(maze, start_tuple, 1, connectivity=1)
            if np.any(test_maze == 0):
                maze[y, x] = 0

        # Convert remaining 0 -> 2
        maze[maze == 0] = 2
        return maze
