from typing import Any

from .policy import Policy
from .q_table import QTable
from src.planning.planning import *

class PlannerPolicy(Policy):

    def __init__(self, q_table: QTable, learning_rate: float, discount: float, strategy: str, level: str):
        super().__init__(q_table, learning_rate, discount)
        self.strategy = strategy
        self.level = level
        self.visited_states = []


    def suggest_action(self, state) -> Any:
        """
        depending on the planning_strategy different suggestion are made using parts of ASP-solving and RL-Learning
        """
        action = None

        if self.strategy == "full_planning":
            action = plan_action(self.level, state)

        elif self.strategy == "plan_for_new_states":
            if state not in self.visited_states:
                action = plan_action(self.level, state)
            else:
                action = self.q_table.get_best_action_for_state(state)

        elif self.strategy == "epsilon_planning":
            #TODO: useful?
            action = self.q_table.get_best_action_for_state(state)

        if state not in self.visited_states:
            self.visited_states.append(state)

        return action


