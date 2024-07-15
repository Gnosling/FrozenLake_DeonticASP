from typing import Set, Dict, Any
import random

from .policy import Policy
from .q_table import QTable
from src.utils.constants import action_set


class EpsilonGreedyPolicy(Policy):
    """
    This is policy uses epsilon to choose random options for better exploration
    """

    def __init__(self, q_table: QTable, learning_rate: float, discount: float, epsilon: float):
        """
        Args:
        0 <= epsilon (float) <= 1
        """
        super().__init__(q_table, learning_rate, discount)
        self.epsilon = epsilon

    def suggest_action(self, state) -> Any:
        """
        Returns the best known action with prop 1-epsilon
        Returns a random action with prop epsilon
        """
        if random.random() < self.epsilon:
            return random.choice(list(action_set))
        else:
            return self.q_table.get_best_action_for_state(state)

