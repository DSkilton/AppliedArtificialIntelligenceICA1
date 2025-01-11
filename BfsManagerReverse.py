import logging
from collections import deque

class BFSManagerReverse:
    """
    A BFS manager that starts from the goal cell (reverse BFS).
    dist_goal[(x, y)] = number of steps from (x,y) to goal.
    Expands in small batches for incremental BFS.
    """

    def __init__(self, env):
        self.logger = logging.getLogger(__name__)
        self.env = env
        self.height = env.height
        self.width = env.width

        self.frontier = deque()
        self.dist_goal = {}  # (x, y) -> BFS distance
        self.initialized = False

    def init_bfs_from_goal(self):
        """Clear BFS data, pick the goal from env, enqueue at distance=0."""
        self.frontier.clear()
        self.dist_goal.clear()
        self.initialized = True

        start, goal = self.env.get_start_and_goal()
        if goal is None:
            self.logger.warning("[BFSManagerReverse] No valid goal found to init BFS.")
            return

        self.frontier.append(goal)
        self.dist_goal[goal] = 0
        self.logger.info(f"[BFSManagerReverse] init_bfs_from_goal: enqueued goal {goal} at dist=0")

    def expand_next_batch(self, batch_size=10):
        """Expand BFS by up to batch_size states, in a partial/incremental manner."""
        if not self.initialized:
            self.logger.warning("[BFSManagerReverse] BFS not initialized! No expansions.")
            return 0

        expansions = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while expansions < batch_size and self.frontier:
            cell = self.frontier.popleft()
            current_dist = self.dist_goal[cell]

            for dy, dx in directions:
                ny, nx = cell[0] + dy, cell[1] + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    if self.env.is_valid_state((ny, nx)) and (ny, nx) not in self.dist_goal:
                        self.dist_goal[(ny, nx)] = current_dist + 1
                        self.frontier.append((ny, nx))
            expansions += 1

        return expansions

    def is_finished(self):
        return (not self.frontier) and self.initialized

    def min_distance(self):
        if self.dist_goal:
            return min(self.dist_goal.values())
        return None

    def max_distance(self):
        if self.dist_goal:
            return max(self.dist_goal.values())
        return None