from typing import Set, Dict, Any
import random


# TODO: use initial estimations?
class QTable:

    def __init__(self):
        self.table = dict()

    def initialize_state(self, state, available_actions: Set):
        self.table[state] = {a: 0 for a in available_actions }

    def update(self, state, action, delta: float):
        self.table[state][action] += delta

    def value_of(self, state, action) -> float:
        if state is None or action is None:
            raise ValueError("Entry was not found in Q-Table!")
        return self.table[state][action]

    def get_best_action_for_state(self, state) -> Any:
        available_actions = self.table[state].items()

        if len(available_actions) == 0:
            return None

        current_maximal_estimate = max(v for _,v in available_actions)
        current_optimal_actions = [a for (a, v) in available_actions
                                   if v==current_maximal_estimate]

        return random.choice(current_optimal_actions)

    def max_value_of(self, state):
        return max(self.table[state].values())

    def get_all_values(self):
        return self.table