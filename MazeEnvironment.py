import numpy as np

class MazeEnvironment:
    def __init__(self, maze):
        self.maze = maze
        self.height, self.width = maze.shape

    def is_valid_state(self, state):
        y, x = state
        if 0 <= y < self.height and 0 <= x < self.width:
            return self.maze[y, x] == 2
        return False

    def get_start_and_goal(self):
        passages = np.argwhere(self.maze == 2)
        if len(passages) < 2:
            return None, None
        start = tuple(passages[0])
        goal = tuple(passages[-1])
        return start, goal

    def step(self, state, action):
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ny = state[0] + moves[action][0]
        nx = state[1] + moves[action][1]

        if self.is_valid_state((ny, nx)):
            next_state = (ny, nx)
        else:
            next_state = state

        start, goal = self.get_start_and_goal()
        if next_state == goal:
            r = 10
            done = True
        else:
            r = -1
            done = False
        return next_state, r, done
