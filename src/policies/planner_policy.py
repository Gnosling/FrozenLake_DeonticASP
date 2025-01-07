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

    def __init__(self, exp_name, q_table: QTable, learning_rate: float, learning_rate_strategy: str, learning_decay_rate: float, discount: float, epsilon: float, strategy: str, planning_horizon: int, delta: int, level: str, norm_set: int, reward_set: int, evaluation_function: int, enforcing = None):
        super().__init__(exp_name, q_table, learning_rate, learning_rate_strategy, learning_decay_rate, discount, level, enforcing, norm_set)
        self.epsilon = epsilon
        self.delta = delta
        self.decay_value = 1
        self.strategy = strategy
        self.planning_horizon = planning_horizon
        self.planning_reward_set = reward_set
        self.suggestion_called_count = 0
        self.evaluation_function = evaluation_function
        self.visited_states = []


    def suggest_action(self, state, env) -> Any:
        """
        depending on the planning_strategy different suggestion are made using parts of ASP-solving and RL-Learning.
        Enforcing strategies are also handled here.
        """
        action = None

        allowed_actions = ACTION_SET
        if self.enforcing and self.enforcing.get("phase") == "during_training":
            if "guardrail" in self.enforcing.get("strategy"):
                allowed_actions = guardrail(self.enforcing, state, self.previous_state, self.last_performed_action, self.last_proposed_action, env)

            if "fixing" in self.enforcing.get("strategy"):
                return self._check_and_fix_path(state, env)

        if self.strategy == "full_planning":
            debug_print("planning was triggered")
            action = plan_action(self.exp_name, self.level, self.planning_horizon, self.last_performed_action, state, self.norm_set, self.planning_reward_set, self.evaluation_function, allowed_actions)

        elif self.strategy == "plan_for_new_states":
            if state not in self.visited_states:
                debug_print("planning was triggered")
                action = plan_action(self.exp_name, self.level, self.planning_horizon, self.last_performed_action, state, self.norm_set, self.planning_reward_set, self.evaluation_function, allowed_actions)
            else:
                action = self._retrieve_action_from_table(state, allowed_actions)

        elif self.strategy == "delta_greedy_planning":
            if random.random() < self.delta:
                debug_print("planning was triggered")
                action = plan_action(self.exp_name, self.level, self.planning_horizon, self.last_performed_action, state, self.norm_set, self.planning_reward_set, self.evaluation_function, allowed_actions)
            else:
                action = self._retrieve_action_from_table(state, allowed_actions)

        elif self.strategy == "delta_decaying_planning":
            # Note: decay_value should be initial 1 and delta should be <0.0005
            self.decay_value = self.decay_value * math.exp(self.delta * -1 * self.suggestion_called_count)
            if random.random() < self.decay_value:
                debug_print("planning was triggered")
                action = plan_action(self.exp_name, self.level, self.planning_horizon, self.last_performed_action, state, self.norm_set, self.planning_reward_set, self.evaluation_function, allowed_actions)
            else:
                action = self._retrieve_action_from_table(state, allowed_actions)

        elif self.strategy == "no_planning":
            action = self._retrieve_action_from_table(state, allowed_actions)

        else:
            raise ValueError(f"invalid planning strategy: {self.strategy}")

        if state not in self.visited_states:
            self.visited_states.append(state)
        self.suggestion_called_count = self.suggestion_called_count + 1
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


