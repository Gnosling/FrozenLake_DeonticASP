from typing import Any

from .policy import Policy
from .q_table import QTable
from src.planning.planning import *
class PlannerPolicy(Policy):

    def __init__(self, q_table: QTable, learning_rate: float, discount: float, strategy: str):
        super().__init__(q_table, learning_rate, discount)
        self.strategy = strategy


    def suggest_action(self, state) -> Any:
        return plan_action("FrozenLake_3x3_A", state)


