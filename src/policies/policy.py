from typing import Any
from .q_table import QTable
from src.utils.constants import ACTION_SET
from src.utils.constants import debug_print
from src.utils.utils import guardrail
from src.utils.utils import compute_successor
from src.planning import validate_path, plan_action

import math




class Policy:
    """
    This is the most general policy, suggesting without enforcing only the best known action (greedy)
    """

    def __init__(self, q_table: QTable, learning_rate: float, learning_rate_strategy: str, learning_decay_rate: float, discount: float, level: str, enforcing = None):
        """
        Args:
        q_table                             the Q-Table to store learned values for state-action pairs
        0 <= learning_rate (float) <= 1     scales each update
        learning_rate_strategy              defines update of learning rate
        0 <= learning_decay_rate            values used for decay strategies
        0 <= discount (float) <= 1          defines importance of next-state value in the update
        """
        self.q_table = q_table
        self.learning_rate = learning_rate
        self.learning_rate_strategy = learning_rate_strategy
        self.learning_decay_rate = learning_decay_rate
        self.discount = discount
        self.call_count = 0
        self.level = level
        self.enforcing = enforcing
        self.last_performed_action = None
        self.last_proposed_action = None
        self.previous_state = None

    def initialize(self, states, available_actions, env):
        self.q_table.initialize_state(states, available_actions, env)

    def update_after_step(self, state, action, new_state, reward):
        self.update_learning_rate()
        delta = (self.learning_rate
                 * (reward + self.discount * self.value_of_state(new_state) - self.q_table.value_of(state, action)))
        self.q_table.update(state, action, delta)

    def update_after_end_of_episode(self, trail):
        """
        updates q-table in reversed step order
        """
        for state, action, new_state, reward in reversed(trail):
            self.update_after_step(state, action, new_state, reward)

    def value_of_state_action_pair(self, state, action) -> float:
        return self.q_table.value_of(state, action)

    def suggest_action(self, state, env) -> Any:

        allowed_actions = ACTION_SET
        if self.enforcing:
            if "guardrail" in self.enforcing.get("strategy"):
                allowed_actions = guardrail(self.enforcing, state, self.previous_state, self.last_performed_action,
                                            self.last_proposed_action, env)

            if "fixing" in self.enforcing.get("strategy"):
                return self._check_and_fix_path(state, env)

        return self.q_table.get_best_allowed_action_for_state(state, allowed_actions)


    def _check_and_fix_path(self, state, env):
        """
        Uses ASP checking to validate the proposed path from the policy.
        If violations are detected, then the planning component is triggered to 'fix' the first action
        """
        layout, width, height = env.get_layout()
        enforcing_horizon = self.enforcing.get("enforcing_horizon")
        enforcing_norm_set = self.enforcing.get("norm_set")
        action_sequence = []
        original_state = state
        for i in range(enforcing_horizon):
            action = self.q_table.get_best_action_for_state(state)
            action_sequence.append(action)
            successor = compute_successor(state[0], action, width, height)
            state = (successor, state[1], state[2])

        if validate_path(action_sequence, self.level, enforcing_horizon, self.last_performed_action, original_state, enforcing_norm_set):
            return action_sequence[0]
        else:
            # Note: evaluation_set 3 is used per default
            # TODO: maybe implement an evaluation set that does not care about rewards and use it here
            return plan_action(self.level, enforcing_horizon, self.last_performed_action, original_state, enforcing_norm_set, 3, ACTION_SET)


    def update_learning_rate(self):
        if self.learning_rate_strategy == "constant":
            pass
        elif self.learning_rate_strategy == "linear_decay":
            self.learning_rate = max(self.learning_rate - self.call_count * self.learning_decay_rate, 0.01)
        elif self.learning_rate_strategy == "exponential_decay":
            self.learning_rate = math.exp(self.learning_decay_rate * -1 * self.call_count)
        else:
            raise ValueError(f"invalid learning-rate strategy: {self.learning_rate_strategy}")

    def update_dynamic_env_aspects(self, last_performed_action, last_proposed_action, previous_state):
        self.last_performed_action = last_performed_action
        self.last_proposed_action = last_proposed_action
        self.previous_state = previous_state

    def value_of_state(self, state):
        """
        state-value is best known action value of that state
        """
        return self.q_table.max_value_of(state)

    def reset_after_episode(self):
        pass

    def set_enforcing(self, value):
        self.enforcing = value

    def get_printed_policy(self) -> str:
        return str(self.q_table.get_all_values())

    def get_q_table(self):
        return self.q_table

