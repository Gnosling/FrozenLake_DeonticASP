from typing import Any
import random
import math

from .policy import Policy
from .q_table import QTable
from src.planning.planning import *
from src.utils.constants import ACTION_SET
from src.utils.constants import debug_print
from src.utils.utils import guardrail


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


    def suggest_action(self, state, enforcing, env) -> Any:
        """
        depending on the planning_strategy different suggestion are made using parts of ASP-solving and RL-Learning.
        Enforcing strategies are also handled here.
        """
        action = None

        allowed_actions = ACTION_SET
        if enforcing and enforcing.get("phase") == "during_training":
            if "guardrail" in enforcing.get("strategy"):
                allowed_actions = guardrail(enforcing, state, self.previous_state, self.last_performed_action, self.last_proposed_action, env)

        if self.strategy == "full_planning":
            debug_print("planning was triggered")
            action = plan_action(self.level, self.planning_horizon, self.last_performed_action, state, self.norm_set, self.evaluation_function, allowed_actions)

        elif self.strategy == "plan_for_new_states":
            # TODO: use full new states or only for new positions?
            if state not in self.visited_states:
                debug_print("planning was triggered")
                action = plan_action(self.level, self.planning_horizon, self.last_performed_action, state, self.norm_set, self.evaluation_function, allowed_actions)
            else:
                action = self._retrieve_action_from_table(state, allowed_actions)

        elif self.strategy == "delta_greedy_planning":
            if random.random() < self.delta:
                debug_print("planning was triggered")
                action = plan_action(self.level, self.planning_horizon, self.last_performed_action, state, self.norm_set, self.evaluation_function, allowed_actions)
            else:
                action = self._retrieve_action_from_table(state, allowed_actions)

        elif self.strategy == "delta_decaying_planning":
            # exponential decay
            chance = math.exp(self.delta * -1 * self.call_count)
            if random.random() < chance:
                debug_print("planning was triggered")
                action = plan_action(self.level, self.planning_horizon, self.last_performed_action, state, self.norm_set, self.evaluation_function, allowed_actions)
            else:
                action = self._retrieve_action_from_table(state, allowed_actions)

        elif self.strategy == "no_planning":
            action = self._retrieve_action_from_table(state, allowed_actions)

        else:
            raise ValueError(f"invalid planning strategy: {self.strategy}")

        if state not in self.visited_states:
            self.visited_states.append(state)
        self.call_count = self.call_count+1
        return action

    def _retrieve_action_from_table(self, state, allowed_actions):
        if random.random() < self.epsilon:
            debug_print("exploration was triggered")
            return random.choice(list(ACTION_SET))  # Note: exploration is never restricted
        else:
            return self.q_table.get_best_allowed_action_for_state(state, allowed_actions)

    def reset_after_episode(self):
        self.visited_states = []
        pass


