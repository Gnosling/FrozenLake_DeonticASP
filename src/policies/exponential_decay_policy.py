from typing import Set, Dict, Any
import random
import math

from .policy import Policy
from .q_table import QTable
from src.utils.constants import action_set

class ExponentialDecayPolicy(Policy):
    """
    This policy uses epsilon to compute the exponential decay rate
    """

    def __init__(self, q_table: QTable, learning_rate: float, learning_rate_strategy, learning_decay_rate, discount: float, epsilon: float):
        """
        Args:
        0 <= epsilon (float) <= 1
        """
        super().__init__(q_table, learning_rate, learning_rate_strategy, learning_decay_rate, discount)
        self.epsilon = epsilon

    def suggest_action(self, state) -> Any:
        """
        Returns the best known action with prop 1-epsilon
        Returns a random action with prop epsilon
        """
        # exponential decay
        chance = math.exp(self.epsilon * -1 * self.call_count)
        if random.random() < chance:
            return random.choice(list(action_set))
        else:
            return self.q_table.get_best_action_for_state(state)

    def reset_after_episode(self):
        pass
