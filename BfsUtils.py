class BfsUtils:

    @staticmethod
    def bfs_init_qvalue_minmax_discount(q_table, dist_goal):
        """
        Update the Q-table with BFS-based initial values using min-max discounting.
        :param q_table: The Q-table to update
        :param dist_goal: Dictionary mapping states to distances to the goal
        """
        for (y, x), distance in dist_goal.items():
            for action in range(4):  # Assuming 4 actions: up, down, left, right
                q_table[y, x, action] = 1 / (1 + distance)
