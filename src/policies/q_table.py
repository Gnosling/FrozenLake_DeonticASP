from typing import Set, Dict, Any
import random

class QTable:
    """
    QTable stores state action values and contain logic for different initialization strategies.
    A state is defined as (current_position, current_traverser_position, [ordered_list_of_remaining_presents]).
    actions are always {"LEFT", "DOWN", "RIGHT", "UP"}.
    """

    def __init__(self, initialization: bool = False):
        self.table = dict()
        self.initialization = initialization

    def _init_table_for_safe_states_and_action(self, states, available_actions, env):
        layout, width, height = env.get_layout()
        safe_tiles = b'FPSG' # Note: safe tiles get a value of 0.2, cracked are only half save so they get 0.1
        for tile_number in range(width * height):
            row = int(tile_number / height)
            col = tile_number % width
            current_states = {(s,t,p) for (s,t,p) in states if s == tile_number}
            for action in available_actions:
                if action == "LEFT":
                    if col == 0:
                        for key in current_states:
                            self.table[key][action] = 0.2
                    else:
                        tile = env.get_tile_symbol_of_state_number(tile_number - 1)
                        if tile in safe_tiles:
                            for key in current_states:
                                self.table[key][action] = 0.2
                        elif tile in b'C':
                            for key in current_states:
                                self.table[key][action] = 0.1
                        else:
                            for key in current_states:
                                self.table[key][action] = 0

                elif action == "DOWN":
                    if row == height-1:
                        for key in current_states:
                            self.table[key][action] = 0.2
                    else:
                        tile = env.get_tile_symbol_of_state_number(tile_number + height)
                        if tile in safe_tiles:
                            for key in current_states:
                                self.table[key][action] = 0.2
                        elif tile in b'C':
                            for key in current_states:
                                self.table[key][action] = 0.1
                        else:
                            for key in current_states:
                                self.table[key][action] = 0

                elif action == "RIGHT":
                    if col == width-1:
                        for key in current_states:
                            self.table[key][action] = 0.2
                    else:
                        tile = env.get_tile_symbol_of_state_number(tile_number + 1)
                        if tile in safe_tiles:
                            for key in current_states:
                                self.table[key][action] = 0.2
                        elif tile in b'C':
                            for key in current_states:
                                self.table[key][action] = 0.1
                        else:
                            for key in current_states:
                                self.table[key][action] = 0

                elif action == "UP":
                    if row == 0:
                        for key in current_states:
                            self.table[key][action] = 0.2
                    else:
                        tile = env.get_tile_symbol_of_state_number(tile_number - height)
                        if tile in safe_tiles:
                            for key in current_states:
                                self.table[key][action] = 0.2
                        elif tile in b'C':
                            for key in current_states:
                                self.table[key][action] = 0.1
                        else:
                            for key in current_states:
                                self.table[key][action] = 0


    def initialize_state(self, states, available_actions: Set, env):
        if self.initialization == "random":
            for state in states:
                self.table[state] = {a: round(random.uniform(0, 0.2), 2) for a in available_actions}
        elif self.initialization == "safe":
            for state in states:
                self.table[state] = {a: 0 for a in available_actions}
            self._init_table_for_safe_states_and_action(states, available_actions, env)
        # TODO: also use super-safe init? and rename the other cautious?
        else:
            for state in states:
                self.table[state] = {a: 0 for a in available_actions}
        self.table = dict(sorted(self.table.items()))

    def update(self, state, action, delta: float):
        self.table[state][action] += delta

    def value_of(self, state, action) -> float:
        if state is None or action is None:
            raise ValueError("Entry was not found in Q-Table!")
        return self.table[state][action]

    def get_best_action_for_state(self, state) -> Any:
        available_actions = self.table[state].items()
        current_maximal_estimate = max(v for _,v in available_actions)
        current_optimal_actions = [a for (a, v) in available_actions
                                   if v==current_maximal_estimate]

        return random.choice(current_optimal_actions)

    def get_best_allowed_action_for_state(self, state, allowed_actions) -> Any:
        """
        returns the best action within the allowed_actions-set
        """
        available_actions = {action: self.table[state][action] for action in allowed_actions}
        current_maximal_estimate = max(v for _, v in available_actions)
        current_optimal_actions = [a for (a, v) in available_actions
                                   if v == current_maximal_estimate]

        return random.choice(current_optimal_actions)

    def max_value_of(self, state):
        return max(self.table[state].values())

    def get_all_values(self):
        return self.table

    def get_values_of_current_position(self, position: int):
        filtered = {(s,t,p): v for (s,t,p), v in self.table.items() if s == position}
        return dict(sorted(filtered.items()))
