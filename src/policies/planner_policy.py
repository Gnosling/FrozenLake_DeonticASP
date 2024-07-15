from typing import Any
import random
import math

from .policy import Policy
from .q_table import QTable
from src.planning.planning import *

class PlannerPolicy(Policy):

    def __init__(self, q_table: QTable, learning_rate: float, discount: float, epsilon: float, strategy: str, planning_horizon: int, level: str, norm_set: int, evaluation_function: int):
        super().__init__(q_table, learning_rate, discount)
        self.epsilon = epsilon
        self.strategy = strategy
        self.planning_horizon = planning_horizon
        self.level = level
        self.norm_set = norm_set
        self.evaluation_function = evaluation_function
        self.visited_states = []
        self.call_count = 0
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
            if random.random() < self.epsilon:
                action = plan_action(self.level, self.planning_horizon, self.current_state_of_traverser, self.last_performed_action, state, self.norm_set, self.evaluation_function)
            else:
                action = self.q_table.get_best_action_for_state(state)

        elif self.strategy == "decaying_planning":
            # exponential decay
            chance = math.exp(self.epsilon * -1 * self.call_count)
            if random.random() < chance:
                action = plan_action(self.level, self.planning_horizon, self.current_state_of_traverser, self.last_performed_action, state, self.norm_set, self.evaluation_function)
            else:
                action = self.q_table.get_best_action_for_state(state)

        else:
            raise ValueError(f"invalid planning strategy: {self.strategy}")

        if state not in self.visited_states:
            self.visited_states.append(state)
        self.call_count = self.call_count+1
        return action

    def updated_dynamic_env_aspects(self, current_state_of_traverser, last_performed_action):
        self.current_state_of_traverser = current_state_of_traverser
        self.last_performed_action = last_performed_action


