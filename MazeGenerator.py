import random
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

        # Always have top-left and bottom-right
        maze[0, 1] = 2
        maze[-1, -2] = 2
        total_openings = 2

        # Add more openings up to self.openings
        while total_openings < self.openings:
            side = random.choice(["top","bottom","left","right"])
            if side == "top":
                x = random.randint(1, self.width-2)
                if maze[0, x] == 1:
                    maze[0, x] = 2
                    total_openings += 1
            elif side == "bottom":
                x = random.randint(1, self.width-2)
                if maze[self.height-1, x] == 1:
                    maze[self.height-1, x] = 2
                    total_openings += 1
            elif side == "left":
                y = random.randint(1, self.height-2)
                if maze[y, 0] == 1:
                    maze[y, 0] = 2
                    total_openings += 1
            elif side == "right":
                y = random.randint(1, self.height-2)
                if maze[y, self.width-1] == 1:
                    maze[y, self.width-1] = 2
                    total_openings += 1

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

        for (x, y) in unfilled_cells:
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
