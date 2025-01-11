import logging
import random
import numpy as np
import BfsManager as BfsManager
from AlgorithmHelpers import AlgorithmHelpers

class BfandRlMananger:

    def __init__(self, env, bfs_mgr, q_table, alpha=0.1, gamma_rl=0.9, epsilon=0.1, beta=0.5):
        """
        env: MazeEnvironment
        bfs_mgr: BFSManagerReverse
        q_table: np.ndarray shape (height, width, 4)
        """
        self.logger = logging.getLogger(__name__)
        self.env = env
        self.bfs_mgr = bfs_mgr
        self.q_table = q_table

        # RL Params
        self.alpha = alpha
        self.gamma_rl = gamma_rl
        self.epsilon = epsilon
        self.beta = beta

        # For logging results
        self.expansions_logs = []
        self.episode_rewards = []
        self.episode_steps = []

    def run_bfs_and_rl(self, episodes, max_steps_per_episode, batch_size):
        """
        Each episode:
          1) pick random perimeter cell as start
          2) run BFS partial expansions
          3) BFS -> Q init
          4) run RL sub-episode (up to max_steps_per_episode steps or until done)
        """
        # BFS init from goal
        self.bfs_mgr.init_bfs_from_goal()

        # Let BFS expand some initially
        expansions = self.bfs_mgr.expand_next_batch_reverse(batch_size=50)  # big initial chunk
        self.logger.info(f"Initial BFS expansions: {expansions}")

        # Gather perimeter openings for random start
        perimeter_openings = self.get_perimeter_cells()
        _, goal = self.env.get_start_and_goal()

        for ep in range(episodes):
            # pick random start
            if perimeter_openings:
                start = random.choice(perimeter_openings)
            else:
                # fallback to default environment start
                start, _ = self.env.get_start_and_goal()

            if not goal:
                self.logger.warning("[run_bfs_and_rl] No valid goal, skipping RL episodes.")
                break

            state = start
            done = False
            ep_reward = 0
            steps = 0

            while not done and steps < max_steps_per_episode:
                # BFS partial expansions
                if not self.bfs_mgr.is_finished_goal():
                    expansions = self.bfs_mgr.expand_next_batch_reverse(batch_size=batch_size)

                    # BFS -> Q
                    AlgorithmHelpers.bfs_init_qvalue_minmax_discount(self.q_table, self.bfs_mgr.dist_goal)

                # RL step
                steps += 1
                action = self.select_action(state)
                next_state, r, done = self.env.step(state, action)

                # Weighted Q-value update with BFS influence
                bfs_value = self.bfs_mgr.dist_goal.get(state, 0)  # Default to 0 if no BFS value
                current_value = self.q_table[state[0], state[1], action]
                next_max = np.max(self.q_table[next_state[0], next_state[1]])
                rl_update = self.alpha * (r + self.gamma_rl * next_max - current_value)
                self.q_table[state[0], state[1], action] = (1 - self.beta) * (current_value + rl_update) + self.beta * bfs_value

                state = next_state
                ep_reward += r
                self.beta = max(0.1, self.beta * 0.99) # Decay BFS influence 

            self.episode_rewards.append(ep_reward)
            self.episode_steps.append(steps)

            # Logging
            if ep % 10 == 0:
                distMin = self.bfs_mgr.min_distance()
                distMax = self.bfs_mgr.max_distance()
                frontier_sz = len(self.bfs_mgr.frontier)
                visited_ct = len(self.bfs_mgr.dist_goal)
                self.logger.info(f"Episode {ep}: BFS(frontier={frontier_sz}, visited={visited_ct}, distRange=({distMin},{distMax})) => RL Reward={ep_reward}, steps={steps}")

        return self.q_table, self.expansions_logs, self.episode_rewards, self.episode_steps

    def select_action(self, state):
        """Epsilon-greedy with the Q-table."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(4)
        else:
            return np.argmax(self.q_table[state[0], state[1]])

    def get_perimeter_cells(self):
        """
        Return a list of passable perimeter cells to randomize start.
        """
        perimeter = []
        for x in range(self.env.width):
            if self.env.is_valid_state((0, x)):
                perimeter.append((0, x))
            if self.env.is_valid_state((self.env.height-1, x)):
                perimeter.append((self.env.height-1, x))
        for y in range(self.env.height):
            if self.env.is_valid_state((y, 0)):
                perimeter.append((y, 0))
            if self.env.is_valid_state((y, self.env.width-1)):
                perimeter.append((y, self.env.width-1))
        return perimeter