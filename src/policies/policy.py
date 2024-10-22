from typing import Any
from .q_table import QTable
from src.utils.constants import ACTION_SET
from src.utils.constants import debug_print
from src.utils.utils import guardrail

import math


class Policy:
    """
    This is the most general policy, suggesting only the best known action (greedy)
    """

    def __init__(self, q_table: QTable, learning_rate: float, learning_rate_strategy: str, learning_decay_rate: float, discount: float):
        """
        Args:
        q_table                             the Q-Table to store learned values for state-action pairs
        0 <= learning_rate (float) <= 1     scales each update
        learning_rate_strategy              defines update of learning rate
        0 <= learning_decay_rate            values used for decay strategies
        0 <= discount (float) <= 1          defines importance of next-state value in the update
        """
        self.q_table = q_table
        self.learning_rate = learning_rate
        self.learning_rate_strategy = learning_rate_strategy
        self.learning_decay_rate = learning_decay_rate
        self.discount = discount
        self.call_count = 0
        self.last_performed_action = None
        self.last_proposed_action = None
        self.previous_state = None

    def initialize(self, states, available_actions, env):
        self.q_table.initialize_state(states, available_actions, env)

    def update_after_step(self, state, action, new_state, reward):
        self.update_learning_rate()
        delta = (self.learning_rate
                 * (reward + self.discount * self.value_of_state(new_state) - self.q_table.value_of(state, action)))
        self.q_table.update(state, action, delta)

    def update_after_end_of_episode(self, trail):
        """
        updates q-table in reversed step order
        """
        for state, action, new_state, reward in reversed(trail):
            self.update_after_step(state, action, new_state, reward)

    def value_of_state_action_pair(self, state, action) -> float:
        return self.q_table.value_of(state, action)

    def suggest_action(self, state, enforcing, env) -> Any:

        allowed_actions = ACTION_SET
        if enforcing:
            if "guardrail" in enforcing.get("strategy"):
                allowed_actions = guardrail(enforcing, state, self.previous_state, self.last_performed_action,
                                            self.last_proposed_action, env)

        return self.q_table.get_best_allowed_action_for_state(state, allowed_actions)

    def update_learning_rate(self):
        if self.learning_rate_strategy == "constant":
            pass
        elif self.learning_rate_strategy == "linear_decay":
            self.learning_rate = max(self.learning_rate - self.call_count * self.learning_decay_rate, 0.01)
        elif self.learning_rate_strategy == "exponential_decay":
            self.learning_rate = math.exp(self.learning_decay_rate * -1 * self.call_count)
        else:
            raise ValueError(f"invalid learning-rate strategy: {self.learning_rate_strategy}")

    def update_dynamic_env_aspects(self, last_performed_action, last_proposed_action, previous_state):
        self.last_performed_action = last_performed_action
        self.last_proposed_action = last_proposed_action
        self.previous_state = previous_state

    def value_of_state(self, state):
        """
        state-value is best known action value of that state
        """
        return self.q_table.max_value_of(state)

    def reset_after_episode(self):
        pass

    def get_printed_policy(self) -> str:
        return str(self.q_table.get_all_values())

    def get_q_table(self):
        return self.q_table

