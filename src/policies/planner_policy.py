from typing import Any
import random
import math

from .policy import Policy
from .q_table import QTable
from src.planning.planning import *
from src.utils.constants import ACTION_SET


class PlannerPolicy(Policy):
    """
    This policy use ASP as planning, there are different strategies when the planning should be triggered
    """

    def __init__(self, q_table: QTable, learning_rate: float, learning_rate_strategy: str, learning_decay_rate: float, discount: float, epsilon: float, strategy: str, planning_horizon: int, level: str, norm_set: int, evaluation_function: int):
        super().__init__(q_table, learning_rate, learning_rate_strategy, learning_decay_rate, discount)
        self.epsilon = epsilon
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
        # TODO: since planning might yield short solutions, what happens is the way is too long,
        # ie. -1 for short failure is better than -20+10 for long way? --> use as reason to set goal rewards high

        if self.strategy == "full_planning":
            action = plan_action(self.level, self.planning_horizon, self.current_state_of_traverser, self.last_performed_action, state, self.current_presents, self.norm_set, self.evaluation_function)

        elif self.strategy == "plan_for_new_states":
            if state not in self.visited_states:
                action = plan_action(self.level, self.planning_horizon, self.current_state_of_traverser, self.last_performed_action, state, self.current_presents, self.norm_set, self.evaluation_function)
            else:
                action = self.q_table.get_best_action_for_state(state)

        elif self.strategy == "epsilon_planning":
            if random.random() < self.epsilon:
                action = plan_action(self.level, self.planning_horizon, self.current_state_of_traverser, self.last_performed_action, state, self.current_presents, self.norm_set, self.evaluation_function)
            else:
                action = self.q_table.get_best_action_for_state(state)

        # TODO: maybe combine explorations aspects here as well? -> only with the best exploration aspect --> depends on what is best in A*
        elif self.strategy == "epsilon_planning_with_epsilon_exploration":
            if random.random() < self.epsilon:
                action = plan_action(self.level, self.planning_horizon, self.current_state_of_traverser, self.last_performed_action, state, self.current_presents, self.norm_set, self.evaluation_function)
            else:
                if random.random() < self.epsilon:
                    action = random.choice(list(ACTION_SET))
                else:
                    action = self.q_table.get_best_action_for_state(state)

        elif self.strategy == "decaying_planning":
            # exponential decay
            chance = math.exp(self.epsilon * -1 * self.call_count)
            if random.random() < chance:
                action = plan_action(self.level, self.planning_horizon, self.current_state_of_traverser, self.last_performed_action, state, self.current_presents, self.norm_set, self.evaluation_function)
            else:
                action = self.q_table.get_best_action_for_state(state)

        else:
            raise ValueError(f"invalid planning strategy: {self.strategy}")

        if state not in self.visited_states:
            self.visited_states.append(state)
        self.call_count = self.call_count+1
        return action

    def updated_dynamic_env_aspects(self, current_state_of_traverser, last_performed_action, current_presents):
        self.current_state_of_traverser = current_state_of_traverser
        self.last_performed_action = last_performed_action
        self.current_presents = current_presents

    def reset_after_episode(self):
        self.visited_states = []
        pass


