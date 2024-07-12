from typing import Any

from .policy import Policy
from .q_table import QTable
from src.planning.planning import *

class PlannerPolicy(Policy):

    def __init__(self, q_table: QTable, learning_rate: float, discount: float, strategy: str, planning_horizon: int, level: str, norm_set: int, evaluation_function: int):
        super().__init__(q_table, learning_rate, discount)
        self.strategy = strategy
        self.planning_horizon = planning_horizon
        self.level = level
        self.norm_set = norm_set
        self.evaluation_function = evaluation_function
        self.visited_states = []
        self.current_state_of_traverser = -1
        self.last_performed_action = None


    def suggest_action(self, state) -> Any:
        """
        depending on the planning_strategy different suggestion are made using parts of ASP-solving and RL-Learning
        """
        action = None

        if self.strategy == "full_planning":
            action = plan_action(self.level, self.planning_horizon, self.current_state_of_traverser, self.last_performed_action, state, self.norm_set, self.evaluation_function)

        elif self.strategy == "plan_for_new_states":
            if state not in self.visited_states:
                action = plan_action(self.level, self.planning_horizon, self.current_state_of_traverser, self.last_performed_action, state, self.norm_set, self.evaluation_function)
            else:
                action = self.q_table.get_best_action_for_state(state)

        elif self.strategy == "epsilon_planning":
            # ToDo: activate planning based on epsilon? or have decreasing planning rate?
            action = self.q_table.get_best_action_for_state(state)

        if state not in self.visited_states:
            self.visited_states.append(state)

        return action

    def updated_dynamic_env_aspects(self, current_state_of_traverser, last_performed_action):
        self.current_state_of_traverser = current_state_of_traverser
        self.last_performed_action = last_performed_action


