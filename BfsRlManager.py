import logging
import numpy as np
import BfsManager as BfsManager
from AlgorithmHelpers import AlgorithmHelpers

class BfandRlMananger:

    logger = logging.getLogger(__name__)

    def run_bfs_and_rl(self, env, bfs_mgr, q_table, episodes=50, iters_per_episode=5, batch_size=10):
        """
        Time-sliced BFS + RL approach.
        - BFS expansions happen partially each iteration
        - Then BFS-based Q-initialization
        - Then RL does a partial episode

        env: MazeEnvironment
        bfs_mgr: BFSManagerReverse (with dist_goal)
        q_table: np.ndarray shape (height, width, 4)
        episodes: how many RL episodes we run
        iters_per_episode: how many BFS+RL sub-iterations per episode
        batch_size: BFS expansions per iteration

        We'll store logs in lists so we can observe the progress.
        """
        alpha = 0.1
        gamma_rl = 0.9  # RL discount
        epsilon = 0.1

        # BFS init from the goal
        bfs_mgr.init_bfs_from_goal()

        expansions_log = []
        reward_log = []
        step_log = []

        for ep in range(episodes):
            start, goal = env.get_start_and_goal()
            if not goal:
                print("[run_bfs_and_rl] No valid goal, skipping RL episodes.")
                break

            total_episode_reward = 0
            steps_in_episode = 0
            done = False
            state = start

            for it in range(iters_per_episode):
                if not bfs_mgr.is_finished_goal():
                    expanded = bfs_mgr.expand_next_batch_reverse(batch_size=batch_size)
                    expansions_log.append(expanded)

                    # BFS -> Q init, using MIN-MAX
                    AlgorithmHelpers.bfs_init_qvalue(q_table, bfs_mgr.dist_goal, r_goal=10, gamma=0.9)

                if not done:
                    steps_in_episode += 1

                    # Epsilon-greedy
                    if np.random.rand() < epsilon:
                        action = np.random.randint(4)
                    else:
                        action = np.argmax(q_table[state[0], state[1]])

                    next_state, r, done = env.step(state, action)

                    # Standard Q-learning update
                    old_q = q_table[state[0], state[1], action]
                    next_max = np.max(q_table[next_state[0], next_state[1]])
                    new_q = old_q + alpha * (r + gamma_rl * next_max - old_q)
                    q_table[state[0], state[1], action] = new_q

                    state = next_state
                    total_episode_reward += r

                    if done:
                        break

            reward_log.append(total_episode_reward)
            step_log.append(steps_in_episode)

            if ep % 10 == 0:
                min_dist = bfs_mgr.min_distance()
                max_dist = bfs_mgr.max_distance()
                frontier_size = len(bfs_mgr.frontier_goal)
                visited_ct = len(bfs_mgr.dist_goal)
                print(f"Episode {ep}: BFS(frontier={frontier_size}, visited={visited_ct}, distRange=({min_dist},{max_dist})) => RL Reward={total_episode_reward}, steps={steps_in_episode}")

        return q_table, expansions_log, reward_log, step_log
