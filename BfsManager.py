import logging
from collections import deque
import numpy as np

class BfsManager:
    """
    Handles incremental BFS expansions from boundary cells to find passable states
    """

    logger = logging.getLogger(__name__)

    def __init__(self, env):
        """
        env: MazeEnvironment
        """
        self.env = env
        self.height = env.height
        self.width = env.width

        # For BFS-from-goal (reverse BFS)
        self.frontier = deque()    # frontier for BFS from goal
        self.dist_goal = {}             # (y,x) -> distance to goal
        self.goal_inited = False

        # For BFS-from-boundary
        self.frontier_boundary = deque()
        self.visited_boundary = {}          # (y,x) -> distance from boundary
        self.boundary_inited = False

        self.visited = {}

    def init_bfs_from_boundary(self):
        """
        This can find distance from each cell to the goal if you BFS backwards from goal.
        """
        self.frontier_boundary.clear()
        self.dist_goal.clear()
        self.visited_boundary.clear()
        self.initialised = True

        # We'll do something like "distGoal = 0 at boundary" if you want
        # or you can store in 'visited' if that’s better.
        # For now, let's store in distGoal for consistency.
        boundary_cells = []
        for x in range(self.width):
            if self.env.is_valid_state((0, x)):
                boundary_cells.append((0, x))
            if self.env.is_valid_state((self.height-1, x)):
                boundary_cells.append((self.height-1, x))

        for y in range(self.height):
            if self.env.is_valid_state((y, 0)):
                boundary_cells.append((y, 0))
            if self.env.is_valid_state((y, self.width-1)):
                boundary_cells.append((y, self.width-1))

        # enqueue them
        for cell in boundary_cells:
            self.visited_boundary[cell] = 0
            self.dist_goal[cell] = 0

        BfsManager.logger.info(f"[BfsManager] init_bfs_from_boundary: Enqueued {len(boundary_cells)} boundary cells.")

    def init_bfs_from_goal(self):
        """
        Clears BFS data and starts BFS from the goal cell i.e. reverse BFS
        """
        self.frontier.clear()
        self.dist_goal.clear()
        self.goal_inited = True

        start, goal = self.env.get_start_and_goal()
        if goal is None:
            BfsManager.logger.warning("[BfsManager] init_bfs_from_goal: No valid goal found. BFS won't do anything.")
            return

        self.frontier.append(goal)
        self.dist_goal[goal] = 0

        BfsManager.logger.info(f"[BfsManager] init_bfs_from_goal: Enqueued goal cell {goal} with distance=0")

    def expand_next_batch_reverse(self, batch_size=10):
        """
        Expand BFS up to 'batch_size' states. Return how many expansions performed.
        """

        expansions = 0
        directions = [(-1,0),(1,0),(0,-1),(0,1)]

        while expansions < batch_size and self.frontier:
            cell = self.frontier.popleft()
            current_dist = self.dist_goal[cell]

            for (dy, dx) in directions:
                ny, nx = cell[0] + dy, cell[1] + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    if self.env.is_valid_state((ny, nx)) and (ny, nx) not in self.dist_goal:
                        self.dist_goal[(ny, nx)] = current_dist + 1
                        self.frontier.append((ny, nx))
            expansions += 1
        return expansions

    def expand_next_batch_boundary(self, batch_size=10):
        """
        Process up to 'batch_size' BFS expansions.
        Return how many expansions actually processed (could be < batch_size if BFS done).
        """
        if not self.boundary_inited:
            BfsManager.logger.warning("[BfsManager] expand_next_batch_boundary called but BFS-from-boundary not inited.")
            return 0

        expansions = 0
        directions = [(-1,0),(1,0),(0,-1),(0,1)]

        while expansions < batch_size and self.frontier_boundary:
            cell = self.frontier_boundary.popleft()
            dist = self.visited_boundary[cell]

            for (dy, dx) in directions:
                ny, nx = cell[0] + dy, cell[1] + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    if self.env.is_valid_state((ny, nx)) and (ny,nx) not in self.visited_boundary:
                        self.visited_boundary[(ny, nx)] = dist + 1
                        self.frontier_boundary.append((ny, nx))
            expansions += 1

        return expansions

    def is_finished_boundary(self):
        """ Returns True if BFS frontier is empty i.e. all expansions are explored """
        # return len(self.frontier) == 0
        return (not self.frontier_boundary) and self.boundary_inited
    
    def is_finished_goal(self):
        """Returns True if BFS frontier for the goal-based BFS is empty."""
        return (not self.frontier) and self.goal_inited

    def min_distance(self):
        if not self.dist_goal:
            return None
        return min(self.dist_goal.values())

    def max_distance(self):
        if not self.dist_goal:
            return None
        return max(self.dist_goal.values())

    def update_qtable(self, q_table):
        """
        For each visited cell (x, y) = dist, set Q-values in q_table
        that reflect BFS knowledge. Possibly dist is from the goal, 
        so shorter dist => better Q-value. 
        """
        for (x, y), dist in self.dist_goal.items():
            for a in range(q_table.shape[2]):
                old_val = q_table[y, x, a]
                new_val = max(old_val, 10 - dist)
                q_table[y, x, a] = new_val

    def get_distGoal_map(self):
        return self.dist_goal

    def get_visitedBoundary_map(self):
        return self.visited_boundary

    def expand_next_batch(self, batch_size=10):
        """
        Process up to 'batch_size' BFS expansions.
        Return how many expansions actually processed (could be < batch_size if BFS done).
        """
        expansions = 0
        directions = [(-1,0),(1,0),(0,-1),(0,1)]

        while expansions < batch_size and self.frontier:
            cell = self.frontier.popleft()
            dist = self.visited[cell]

            for (dy, dx) in directions:
                ny, nx = cell[0] + dy, cell[1] + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    if self.env.is_valid_state((ny, nx)) and (ny, nx) not in self.visited:
                        self.visited[(ny, nx)] = dist + 1
                        self.frontier.append((ny, nx))
            expansions += 1

        return expansions