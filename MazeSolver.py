from queue import PriorityQueue
from collections import deque
import numpy as np
from Persistence import QTablePersistence

class MazeSolver:
    def __init__(self, env):
        self.env = env
        self.maze = env.maze
        self.height, self.width = self.maze.shape

    # ------------ A* ------------ #
    def solve_astar(self):
        start, goal = self.env.get_start_and_goal()
        if not start or not goal:
            print("Not enough passages for A*.")
            return [], []

        frontier = PriorityQueue()
        frontier.put((0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        rewards = []
        steps = 0

        while not frontier.empty():
            _, current = frontier.get()
            steps += 1
            rewards.append(-1)

            if current == goal:
                rewards[-1] = 10
                break

            for dy, dx in directions:
                ny, nx = current[0] + dy, current[1] + dx
                if self.env.is_valid_state((ny, nx)):
                    new_cost = cost_so_far[current] + 1
                    if (ny, nx) not in cost_so_far or new_cost < cost_so_far[(ny, nx)]:
                        cost_so_far[(ny, nx)] = new_cost
                        priority = new_cost + self.heuristic((ny, nx), goal)
                        frontier.put((priority, (ny, nx)))
                        came_from[(ny, nx)] = current

        return rewards, list(range(steps)), came_from

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ------------ Q-Learning ------------ #
    def solve_qlearning(self, episodes=500, log_interval=0, save_filename=None):
        """
        log_interval: if > 0, print logs every 'log_interval' episodes
        """
        q_table = np.zeros((self.height, self.width, 4))
        alpha = 0.1
        gamma = 0.9
        epsilon = 0.1

        start, goal = self.env.get_start_and_goal()
        if not start or not goal:
            print("Not enough passages for Q-learning.")
            return [], [], q_table

        all_rewards = []
        all_steps = []

        for ep in range(episodes):
            state = start
            done = False
            episode_reward = 0
            step_count = 0

            while not done:
                step_count += 1
                # Epsilon-greedy
                if np.random.rand() < epsilon:
                    action = np.random.randint(4)
                else:
                    action = np.argmax(q_table[state[0], state[1]])

                next_state, r, done = self.env.step(state, action)

                old_q = q_table[state[0], state[1], action]
                next_max = np.max(q_table[next_state[0], next_state[1]])
                new_q = old_q + alpha * (r + gamma * next_max - old_q)
                q_table[state[0], state[1], action] = new_q

                state = next_state
                episode_reward += r

                if step_count > 5000:
                    break

            all_rewards.append(episode_reward)
            all_steps.append(step_count)

            # Log every 'log_interval' episodes
            if log_interval > 0 and (ep+1) % log_interval == 0:
                recent_avg = np.mean(all_rewards[-log_interval:])
                print(f"[Q-Learning] Ep {ep+1}, AvgReward(last {log_interval})={recent_avg:.2f}")

            if save_filename:
                QTablePersistence.save(q_table, save_filename)

        return all_rewards, all_steps, q_table

    # ------------ SARSA ------------ #
    def solve_sarsa(self, episodes=500, log_interval=0, save_filename=None): 
        """
        log_interval: if > 0, print logs every 'log_interval' episodes
        """
        q_table = np.zeros((self.height, self.width, 4))
        policy = np.ones((self.height, self.width, 4)) * 0.25  # Initial equal probabilities for all actions
        alpha = 0.1  # Learning rate
        gamma = 0.8  # Discount factor
        epsilon = 1.0  # Exploration rate
        min_epsilon = 0.1
        decay_rate = 0.001
        num_actions = 4

        start, goal = self.env.get_start_and_goal()
        if not start or not goal:
            print("Not enough passages for SARSA.")
            return [], [], q_table

        all_rewards = []
        all_steps = []

        for ep in range(episodes):
            state = start
            action = np.random.choice(num_actions, p=policy[state[0], state[1]])  # Choose action based on policy

            episode_reward = 0
            step_count = 0
            done = False

            while not done:
                step_count += 1
                next_state, r, done = self.env.step(state, action)

                # Choose next action based on policy
                next_action = np.random.choice(num_actions, p=policy[next_state[0], next_state[1]])

                # Compute Expected SARSA value
                expected_value = sum(
                    policy[next_state[0], next_state[1], a] * q_table[next_state[0], next_state[1], a]
                    for a in range(num_actions)
                )

                # Update Q-value using Expected SARSA
                old_q = q_table[state[0], state[1], action]
                new_q = old_q + alpha * (r + gamma * expected_value - old_q)
                q_table[state[0], state[1], action] = new_q

                # Update policy dynamically for current state
                best_action = np.argmax(q_table[state[0], state[1]])
                for a in range(num_actions):
                    if a == best_action:
                        policy[state[0], state[1], a] = 1 - epsilon + (epsilon / num_actions)
                    else:
                        policy[state[0], state[1], a] = epsilon / num_actions

                # Move to the next state and action
                state = next_state
                action = next_action
                episode_reward += r

                if step_count > 5000:  # Prevent infinite loops in training
                    break

            all_rewards.append(episode_reward)
            all_steps.append(step_count)

            # Log progress
            if log_interval > 0 and (ep + 1) % log_interval == 0:
                recent_avg = np.mean(all_rewards[-log_interval:])
                print(f"[Expected SARSA] Ep {ep+1}, AvgReward(last {log_interval})={recent_avg:.2f}")

            # Decay epsilon
            epsilon = max(min_epsilon, epsilon * np.exp(-decay_rate * ep))

            if save_filename:
                QTablePersistence.save(q_table, save_filename)

        return all_rewards, all_steps, q_table

    # ------------ Perimeter BFS Example ------------ #
    def perimeter_bfs(self):
        """
        Simple BFS from all perimeter openings to see how many cells it can reach.
        """
        queue = deque()
        visited = set()
        openings = []

        # Gather perimeter openings
        for x in range(self.width):
            if self.env.is_valid_state((0, x)):
                queue.append((0, x))
            if self.env.is_valid_state((self.height-1, x)):
                queue.append((self.height-1, x))

        for y in range(self.height):
            if self.env.is_valid_state((y, 0)):
                queue.append((y, 0))
            if self.env.is_valid_state((y, self.width-1)):
                queue.append((y, self.width-1))

        while queue:
            cell = queue.popleft()
            if cell in visited:
                continue
            visited.add(cell)

            if(cell[0] in (0, self.height-1) or cell[1] in (0, self.width-1)):
                openings.append(cell)

            # neighbors (4 in 2D (left, right, up, down))
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = cell[0]+dy, cell[1]+dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    if self.env.is_valid_state((ny,nx)) and (ny,nx) not in visited:
                        queue.append((ny,nx))

        return visited, openings

    def solve_bfs_custom(self, start, goal):
        """
        Simple BFS from 'start' to 'goal' in the current maze (no partial expansions).
        Returns 'came_from' dict if a path is found, or None if unreachable.
        """
        from collections import deque

        if not start or not goal:
            print("[solve_bfs_custom] Invalid start or goal.")
            return None

        queue = deque()
        queue.append(start)
        came_from = {start: None}  # track how we reached each cell
        found_goal = False

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            current = queue.popleft()
            if current == goal:
                found_goal = True
                break

            for dy, dx in directions:
                ny, nx = current[0] + dy, current[1] + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    if self.env.is_valid_state((ny, nx)) and (ny, nx) not in came_from:
                        came_from[(ny, nx)] = current
                        queue.append((ny, nx))

        if found_goal:
            return came_from
        else:
            return None
