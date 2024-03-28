import os.path

import clingo
import telingo
import os

import sys
from io import StringIO

from src.utils.constants import *


def _fill_file_for_dynamic_parameters(current_state: int):
    with open(os.path.join(os.getcwd(), "src", "planning", "dynamic_parameters.lp"), "w") as file:
        file.write("#program initial.\n")
        file.write(f"currentState({current_state}).\n")


def _extract_first_action_from_telingo_output(output: str):
    action = ""
    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("act("):
            action = line
            break

    action = action.replace("(", "")
    action = action.replace(")", "")
    action = action.replace("act", "")
    action = action.replace("move", "")
    return action.upper()

def plan_action(level: str, current_state: int = 0):
    """
    calls potassco's telingo to perform ASP-solving.
    uses general_reasoner for RL, frozenlake_reasoner for frozenlake, level-data and a helper file to handle dynamic properties (eg. to parse the currrent-state in the solver)
    returns planned action
    """

    tel = telingo.TelApp()

    file1 = os.path.join(os.getcwd(), "src", "planning", "general_reasoning.lp")
    file2 = os.path.join(os.getcwd(), "src", "planning", "frozenlake_reasoner.lp")
    file3 = os.path.join(os.getcwd(), "src", "planning", "levels", f"{level}.lp")
    _fill_file_for_dynamic_parameters(current_state)
    file4 = os.path.join(os.getcwd(), "src", "planning", "dynamic_parameters.lp")

    output_buffer = StringIO()
    sys.stdout = output_buffer  # Redirect stdout to an in-memory buffer for storing results of solver

    # Note: Clingo maximizes by minimization of negated max, which means the reported optimization value is negated
    if DEBUG_MODE:
        clingo.clingo_main(tel, ['--time-limit=10', '--istop=sat', file1, file2, file3, file4])
    else:
        clingo.clingo_main(tel, ['--verbose=0', '--quiet=1,2,2', '--time-limit=10', '--istop=sat', file1, file2, file3, file4])

    printed_output = output_buffer.getvalue()
    sys.stdout = sys.__stdout__  # Reset sys.stdout to its original value

    return _extract_first_action_from_telingo_output(printed_output)
