import logging
import os
import numpy as np

class QTablePersistence:
    """Handles saving/loading Q-tables (numpy arrays) using compressed .npz files."""

    logger = logging.getLogger(__name__)

    @staticmethod
    def save(q_table, filename="q_table.npz"):
        """
        Saves the Q-table to compressed .npz file
        """
        try:
            np.savez_compressed(filename, q_table=q_table)
            # QTablePersistence.logger.info(f"Q-table saved to {filename}")
        except Exception as e:
            QTablePersistence.logger.warning(f"Error saving {filename}: {e}")

    @staticmethod
    def load(filename="q_table.npz"):
        """
        Loads a Q-table from .npz if it exists, else it returns none
        """
        if not os.path.exists(filename):
            QTablePersistence.logger.warning(f"File {filename} does not exist")
            return None

        try:
            data = np.load(filename)
            q_table = data["q_table"]
            QTablePersistence.logger.info(f"Q-table loaded from {filename}")
            return q_table
        except Exception as e:
            QTablePersistence.logger.info(f"Error loading Q-table {filename}: {e} ")
            return None
