from typing import Any
from .q_table import QTable


class Policy:
    """
    This is the most general policy, suggesting only the best known action (greedy)
    """

    def __init__(self, q_table: QTable, learning_rate: float, discount: float):
        """
        Args:
        q_table                             the Q-Table to store learned values for state-action pairs
        0 <= learning_rate (float) <= 1     scales each update
        0 <= discount (float) <= 1          defines importance of next-state value in the update
        """
        self.q_table = q_table
        self.learning_rate = learning_rate
        self.discount = discount

    def initialize(self, states, available_actions):
        for state in states:
            self.q_table.initialize_state(state, available_actions)

    def update_after_step(self, state, action, new_state, reward):
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

    def suggest_action(self, state) -> Any:
        return self.q_table.get_best_action_for_state(state)

    def value_of_state(self, state):
        """
        state-value is best known action value of that state
        """
        return self.q_table.max_value_of(state)

    # TODO: create helper function to get learned path of policy?
    def get_printed_policy(self) -> str:
        return str(self.q_table.get_all_values())
