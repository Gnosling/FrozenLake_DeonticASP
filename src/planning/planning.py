import os.path

import clingo
import telingo
import os

import sys
from io import StringIO

from src.utils.constants import *


def _fill_file_for_dynamic_parameters(current_state: int, current_state_of_traverser: int, last_performed_action: str):
    with open(os.path.join(os.getcwd(), "src", "planning", "dynamic_parameters.lp"), "w") as file:
        file.write("#program initial.\n")
        file.write(f"currentState({current_state}).\n")
        if current_state_of_traverser is not None:
            file.write(f"currentStateOfTraverser({current_state_of_traverser}).\n")
        if last_performed_action is not None:
            file.write(f"lastPerformedAction(move({last_performed_action.lower()})).\n")


def _extract_first_action_from_telingo_output(output: str):
    best_action = ""
    opt = 10000000
    for section in output.split("State 0"):
        action = ""
        value = 10000000
        for line in section.split("\n"):
            line = line.strip()
            if line.startswith("act(") and action == "":
                action = line

            if line.startswith("eval"):
                value = int(line[5:-1])

        if value < opt:
            opt = value
            best_action = action


    best_action = best_action.replace("(", "")
    best_action = best_action.replace(")", "")
    best_action = best_action.replace("act", "")
    best_action = best_action.replace("move", "")
    return best_action.upper()

def plan_action(level: str, planning_horizon: int, current_state_of_traverser: int, last_performed_action: str, current_state: int = 0, norm_set: int = 1, evaluation_function: int = 1):
    """
    calls potassco's telingo to perform ASP-solving.
    uses general_reasoner for RL, frozenlake_reasoner for frozenlake, level-data and a helper file to handle dynamic properties (eg. to parse the currrent-state in the solver)
    returns planned action
    """

    file1 = os.path.join(os.getcwd(), "src", "planning", "general_reasoning.lp")
    file2 = os.path.join(os.getcwd(), "src", "planning", "frozenlake_reasoning.lp")
    file3 = os.path.join(os.getcwd(), "src", "planning", "levels", f"{level}.lp")
    _fill_file_for_dynamic_parameters(current_state, current_state_of_traverser, last_performed_action)
    file4 = os.path.join(os.getcwd(), "src", "planning", "dynamic_parameters.lp")
    file5 = os.path.join(os.getcwd(), "src", "planning", "deontic_reasonings", f"deontic_reasoning_{norm_set}.lp")
    file6 = os.path.join(os.getcwd(), "src", "planning", "evaluations", f"evaluation_{evaluation_function}.lp")

    output_buffer = StringIO()
    sys.stdout = output_buffer  # Redirects stdout to an in-memory buffer for storing results of solver

    tel = telingo.TelApp()
    # Note: Clingo maximizes by minimization of negated max, which means the reported optimization value is always negated
    # Note: planning_horizon is needed to force telingo to explore that many states, ie. imin lowerbounds the states telingo unfolds
    # if the optimum lies within these steps, then the optimal action will be found (otw its the optimal with these limited steps)

    if DEBUG_MODE:
        clingo.clingo_main(tel, ['--quiet=1', f'--imin={planning_horizon}', f'--imax={planning_horizon}', '--time-limit=30', file1, file2, file3, file4, file5, file6])
    else:
        clingo.clingo_main(tel, ['--verbose=0', '--warn=none', '--quiet=1,2,2', f'--imin={planning_horizon}', f'--imax={planning_horizon}', '--time-limit=30', file1, file2, file3, file4, file5, file6])

    printed_output = output_buffer.getvalue()
    sys.stdout = sys.__stdout__  # Resets sys.stdout to its original value

    # print(f"{printed_output}")

    return _extract_first_action_from_telingo_output(printed_output)
