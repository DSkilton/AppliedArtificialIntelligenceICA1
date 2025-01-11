import logging


class AlgorithmHelpers:

    logger = logging.getLogger(__name__)

    @staticmethod
    def bfs_init_qvalue_minmax_discount(q_table, dist_map, r_goal=10, gamma=0.9):
        """
        For each cell in distGoal, do min-max scaling and discount:
        d_min = min(distGoal)
        d_max = max(distGoal)
        d_norm = (d - d_min)/(d_max - d_min)
        r_disc = r_goal * (gamma^d) * (1 - d_norm)

        Then Q[y,x,a] = max(Q[y,x,a], r_disc)
        """
        if not dist_map:
            AlgorithmHelpers.logger.warn(f"[bfs_init_qvalue_minmax_discount] No BFS distances found.")
            return

        d_min = min(dist_map.values())
        d_max = max(dist_map.values())

        for (x, y), d in dist_map.items():
            # Min Max Scale
            if d_max > d_min:
                d_norm = (d - d_min) / (d_max - d_min)
            else:
                d_norm = 0

            # Discount formula
            discounted_val = r_goal * (gamma ** d) * (1 - d_norm)

            for a in range(q_table.shape[2]):
                old_val = q_table[y, x, a]
                new_val = max(old_val, discounted_val)
                q_table[y, x, a] = new_val

    @staticmethod
    def bfs_init_qvalue(q_table, dist_goal, r_goal=10, gamma=0.9):
        """
        For each cell in distGoal, min-max scale the distances, then apply a discount-based formula
        and update Q-values in q_table.
        """
        if not dist_goal:
            AlgorithmHelpers.logger.warn(f"[bfs_init_qvalue] No BFS distances found.")
            return

        d_min = min(dist_goal.values())
        d_max = max(dist_goal.values())

        for (x, y), d in dist_goal.items():
            # Min-Max scale
            if d_max > d_min:
                d_norm = (d - d_min) / (d_max - d_min)
            else:
                d_norm = 0

            # Apply a discount
            r_disc = r_goal * (gamma**d) * (1 - d_norm)

            for a in range(q_table.shape[2]):
                old_val = q_table[y, x, a]
                new_val = max(old_val, r_disc)
                q_table[y, x, a] = new_val