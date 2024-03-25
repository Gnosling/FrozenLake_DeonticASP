from typing import Set, Dict, Any
import random
from src.policies.q_table import QTable


class Policy:

    def __init__(self):
        self.q_table = QTable()
        self.learning_rate = 0.3
        self.discount = 0.3

    def initialize(self, states, available_actions):
        for state in states:
            self.q_table.initialize_state(state, available_actions)

    def update_after_step(self, state, action, new_state, reward):
        delta = (self.learning_rate
                 * (reward + self.discount * self.value_of_state(new_state) - self.q_table.value_of(state, action)))
        self.q_table.update(state, action, delta)

    def update_after_end_of_episode(self, trail):
        for state, action, new_state, reward in reversed(trail):
            self.update_after_step(state, action, new_state, reward)

    def value_of_state_action_pair(self, state, action) -> float:
        return self.q_table.value_of(state, action)

    def get_best_action_for_state(self, state) -> Any:
        return self.q_table.get_best_action_for_state(state)

    def value_of_state(self, state):
        return self.q_table.max_value_of(state)

    def print_policy(self):
        print(str(self.q_table.get_all_values()))
