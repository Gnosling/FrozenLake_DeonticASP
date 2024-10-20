from typing import Any
import random
import math

from .policy import Policy
from .q_table import QTable
from src.planning.planning import *
from src.utils.constants import ACTION_SET
from src.utils.constants import debug_print


class PlannerPolicy(Policy):
    """
    This policy uses ASP as planning and epsilon greedy exploration, there are different strategies when the planning should be triggered
    """

    def __init__(self, q_table: QTable, learning_rate: float, learning_rate_strategy: str, learning_decay_rate: float, discount: float, epsilon: float, strategy: str, planning_horizon: int, delta: int, level: str, norm_set: int, evaluation_function: int):
        super().__init__(q_table, learning_rate, learning_rate_strategy, learning_decay_rate, discount)
        self.epsilon = epsilon
        self.delta = delta
        self.strategy = strategy
        self.planning_horizon = planning_horizon
        self.level = level
        self.norm_set = norm_set
        self.evaluation_function = evaluation_function
        self.visited_states = []
        self.last_performed_action = None


    def suggest_action(self, state) -> Any:
        """
        depending on the planning_strategy different suggestion are made using parts of ASP-solving and RL-Learning
        """
        action = None
        # TODO: since planning might yield short solutions, what happens is the way is too long,
        # ie. -1 for short failure is better than -20+10 for long way? --> use as reason to set goal rewards high

        if self.strategy == "full_planning":
            debug_print("planning was triggered")
            action = plan_action(self.level, self.planning_horizon, self.last_performed_action, state, self.norm_set, self.evaluation_function)

        elif self.strategy == "plan_for_new_states":
            # TODO: like full new states or only for new positions?
            if state not in self.visited_states:
                debug_print("planning was triggered")
                action = plan_action(self.level, self.planning_horizon, self.last_performed_action, state, self.norm_set, self.evaluation_function)
            else:
                action = self._retrieve_action_from_table(state)

        elif self.strategy == "delta_greedy_planning":
            if random.random() < self.delta:
                debug_print("planning was triggered")
                action = plan_action(self.level, self.planning_horizon, self.last_performed_action, state, self.norm_set, self.evaluation_function)
            else:
                action = self._retrieve_action_from_table(state)

        elif self.strategy == "delta_decaying_planning":
            # exponential decay
            chance = math.exp(self.delta * -1 * self.call_count)
            if random.random() < chance:
                debug_print("planning was triggered")
                action = plan_action(self.level, self.planning_horizon, self.last_performed_action, state, self.norm_set, self.evaluation_function)
            else:
                action = self._retrieve_action_from_table(state)

        elif self.strategy == "no_planning":
            action = self._retrieve_action_from_table(state)

        else:
            raise ValueError(f"invalid planning strategy: {self.strategy}")

        if state not in self.visited_states:
            self.visited_states.append(state)
        self.call_count = self.call_count+1
        return action

    def _retrieve_action_from_table(self, state):
        if random.random() < self.epsilon:
            debug_print("exploration was triggered")
            return random.choice(list(ACTION_SET))
        else:
            return self.q_table.get_best_action_for_state(state)


    def updated_dynamic_env_aspects(self, last_performed_action):
        self.last_performed_action = last_performed_action

    def reset_after_episode(self):
        self.visited_states = []
        pass


