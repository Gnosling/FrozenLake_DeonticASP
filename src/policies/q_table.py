from typing import Set, Dict, Any
import random

class QTable:

    def __init__(self, initialization: bool = False):
        self.table = dict()
        self.initialization = initialization

    def _init_table_for_safe_states_and_action(self, available_actions, env):
        layout, width, height = env.get_layout()
        safe_tiles = b'FPSG' # Note: safe tiles get a value of 0.2, cracked are only half save so they get 0.1
        for state in range(width * height):
            row = int(state / height)
            col = state % width
            for action in available_actions:
                if action == "LEFT":
                    if col == 0:
                        self.table[state][action] = 0.2
                    else:
                        tile = env.get_tile_of_state_number(state -1)
                        if tile in safe_tiles:
                            self.table[state][action] = 0.2
                        elif tile in b'C':
                            self.table[state][action] = 0.1
                        else:
                            self.table[state][action] = 0

                elif action == "DOWN":
                    if row == height-1:
                        self.table[state][action] = 0.2
                    else:
                        tile = env.get_tile_of_state_number(state + height)
                        if tile in safe_tiles:
                            self.table[state][action] = 0.2
                        elif tile in b'C':
                            self.table[state][action] = 0.1
                        else:
                            self.table[state][action] = 0

                elif action == "RIGHT":
                    if col == width-1:
                        self.table[state][action] = 0.2
                    else:
                        tile = env.get_tile_of_state_number(state +1)
                        if tile in safe_tiles:
                            self.table[state][action] = 0.2
                        elif tile in b'C':
                            self.table[state][action] = 0.1
                        else:
                            self.table[state][action] = 0

                elif action == "UP":
                    if row == 0:
                        self.table[state][action] = 0.2
                    else:
                        tile = env.get_tile_of_state_number(state - height)
                        if tile in safe_tiles:
                            self.table[state][action] = 0.2
                        elif tile in b'C':
                            self.table[state][action] = 0.1
                        else:
                            self.table[state][action] = 0

    def initialize_state(self, states, available_actions: Set, env):
        if self.initialization == "random":
            for state in states:
                self.table[state] = {a: round(random.uniform(0, 0.2), 2) for a in available_actions}
        elif self.initialization == "safe":
            for state in states:
                self.table[state] = {a: 0 for a in available_actions}
            self._init_table_for_safe_states_and_action(available_actions, env)
        else:
            for state in states:
                self.table[state] = {a: 0 for a in available_actions}

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