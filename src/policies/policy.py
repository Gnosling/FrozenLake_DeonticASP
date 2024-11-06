from typing import Any
from .q_table import QTable
from src.utils.constants import ACTION_SET
from src.utils.constants import debug_print
from src.utils.utils import guardrail, get_shaped_rewards
from src.utils.utils import compute_expected_successor
from src.planning import validate_path, plan_action

import math




class Policy:
    """
    This is the most general policy, suggesting without enforcing only the best known action (greedy)
    """

    def __init__(self, q_table: QTable, learning_rate: float, learning_rate_strategy: str, learning_decay_rate: float, discount: float, level: str, enforcing=None, norm_set=None):
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
        self.update_called_count = 0
        self.level = level
        self.enforcing = enforcing
        self.norm_set = norm_set
        self.last_performed_action = None
        self.last_proposed_action = None
        self.previous_state = None

    def initialize(self, states, available_actions, env):
        self.q_table.initialize_state(states, available_actions, self.norm_set, env)

    def update_after_step(self, state, action, new_state, reward, trail, env, after_training=False):
        self._update_learning_rate() # TODO: test_learning rate updates
        if self.enforcing and "reward_shaping" in self.enforcing.get("strategy"):
            if (self.enforcing.get("phase") == "during_training" and not after_training) or (self.enforcing.get("phase") == "after_training" and after_training):
                reward = reward + get_shaped_rewards(self.enforcing, self.discount, state, new_state, trail, env)
        delta = (self.learning_rate
                 * (reward + self.discount * self.value_of_state(new_state) - self.q_table.value_of(state, action)))
        self.q_table.update(state, action, delta)
        self.update_called_count += 1

    def update_after_end_of_episode(self, trail, env):
        """
        updates q-table in reversed step order
        """
        counter = 0
        for state, action, new_state, reward in reversed(trail):
            if counter == 0:
                self.update_after_step(state, action, new_state, reward, trail, env)
            else:
                self.update_after_step(state, action, new_state, reward, trail[:counter], env)
            counter -= 1

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
            successor = compute_expected_successor(state[0], action, width, height)
            state = (successor, state[1], state[2])

        if validate_path(action_sequence, self.level, enforcing_horizon, self.last_performed_action, original_state, enforcing_norm_set):
            return action_sequence[0]
        else:
            # Note: evaluation_set 3 is used per default
            # TODO: maybe implement an evaluation set that does not care about rewards and use it here
            # TODO: use planning_horizon here?
            return plan_action(self.level, enforcing_horizon, self.last_performed_action, original_state, enforcing_norm_set, 3, ACTION_SET)


    def _update_learning_rate(self):
        if self.learning_rate_strategy == "constant":
            pass
        elif self.learning_rate_strategy == "linear_decay":
            self.learning_rate = max(self.learning_rate - self.update_called_count * self.learning_decay_rate, 0.01)
        elif self.learning_rate_strategy == "exponential_decay":
            self.learning_rate = math.exp(self.learning_decay_rate * -1 * self.update_called_count)
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

