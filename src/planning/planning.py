import os
import os.path
import subprocess



#from src.utils.utils import debug_print


def _fill_file_for_dynamic_parameters(exp_name, current_state: int, current_state_of_traverser: int, last_performed_action: str, presents: list, allowed_actions: list = None, action_path: list = None):
    with open(os.path.join(os.getcwd(), "src", "planning", f"dynamic_parameters_{exp_name}.lp"), "w") as file:
        file.write("#program initial.\n")
        file.write(f"currentState({current_state}).\n")
        if current_state_of_traverser is not None:
            file.write(f"currentStateOfTraverser({current_state_of_traverser}).\n")
        if last_performed_action:
            file.write(f"lastPerformedAction(move({last_performed_action.lower()})).\n")
        if presents:
            for present in presents:
                file.write(f"present({present}).\n")
        if allowed_actions:
            for action in allowed_actions:
                file.write(f"allowedAction(move({action.lower()})).\n")
        if action_path:
            file.write("#program always.\n")
            for index, action in enumerate(action_path):
                file.write(f"givenAction({action.lower()},{index}).\n")

def _remove_file_with_dynamic_parameters(exp_name):
    path = os.path.join(os.getcwd(), "src", "planning", f"dynamic_parameters_{exp_name}.lp")
    if os.path.exists(path):
        os.remove(path)

def _value_better_than_optimization(value, opt):
    # first levels have highest prio
    for i in range(len(opt)):
        if value[i] < opt[i]:
            return True
        elif value[i] > opt[i]:
            return False
        # else there is a tie and next level is checked

    return False


def _extract_first_action_from_telingo_output(output: str):
    best_action = ""
    best_section = ""
    max_opt_levels = 20
    opt = [10000000 for _ in range(max_opt_levels)]
    sections = output.split("Solving...")
    skipped_first_section = False
    for section in sections[1:]:
        if not skipped_first_section and len(sections) >= 2:
            skipped_first_section = True
            continue
        action = ""
        value = [10000000 for _ in range(max_opt_levels)]
        for line in section.split("\n"):
            line = line.strip()
            if line.startswith("act(") and action == "":
                # first action
                action = line

            if line.startswith("Optimization:"):
                value = [int(num) for num in line.split(" ")[1:]]

        # if last eval of section is minimum, then take first action of that section
        if _value_better_than_optimization(value, opt):
            opt = value
            best_action = action
            best_section = section


    best_action = best_action.replace("(", "")
    best_action = best_action.replace(")", "")
    best_action = best_action.replace("act", "")
    best_action = best_action.replace("move", "")
    return best_action.upper()


def _extract_validation_result_from_telingo_output(output: str):
    if "UNSATISFIABLE" in output:
        return False
    elif "SATISFIABLE" in output:
        return True
    else:
        raise ValueError('Checking threw an unexpected error!')


def plan_action(exp_name, level: str, planning_horizon: int, last_performed_action: str, state: tuple, norm_set: int = 1, reward_set: int = 1, evaluation_function: int = 1, allowed_actions=None):
    """
    calls potassco's telingo to perform ASP-solving.
    uses general_reasoner for RL, frozenlake_reasoner for frozenlake, level-data and a helper file to handle dynamic properties (eg. to parse the currrent-state in the solver)
    returns planned action
    """

    file1 = os.path.join(os.getcwd(), "src", "planning", "general_reasoning.lp")
    file2 = os.path.join(os.getcwd(), "src", "planning", "frozenlake_reasoning.lp")
    file3 = os.path.join(os.getcwd(), "src", "planning", "levels", f"{level}.lp")
    _fill_file_for_dynamic_parameters(exp_name, state[0], state[1], last_performed_action, list(state[2]), allowed_actions, None)
    file4 = os.path.join(os.getcwd(), "src", "planning", f"dynamic_parameters_{exp_name}.lp")
    file5 = os.path.join(os.getcwd(), "src", "planning", "deontic_norms", f"deontic_norms_{norm_set}.lp")
    file6 = os.path.join(os.getcwd(), "src", "planning", "rewards", f"rewards_{reward_set}.lp")
    file7 = os.path.join(os.getcwd(), "src", "planning", "evaluations", f"evaluation_{evaluation_function}.lp")

    # Note: Clingo maximizes by minimization of negated max, which means the reported optimization value is always negated
    # Note: planning_horizon is needed to force telingo to explore that many states, ie. imin lowerbounds the states telingo unfolds
    #   if the optimum lies within these steps, then the optimal action will be found (otw its the optimal with these limited steps)
    # Note: Weak constraints must add positive penalties, such that telingo handles the return value correctly

    # Note: starts already in the active python environment
    command = [
        f'python', f'-m', f'telingo',
        f'--quiet=1,1,1',
        f'--imin={planning_horizon}', f'--imax={planning_horizon}',
        f'--time-limit=30',
        f'"{file1}"', f'"{file2}"', f'"{file3}"', f'"{file4}"', f'"{file5}"', f'"{file6}"', f'"{file7}"'
    ]

    bat_content = f"""
    @echo off
    {' '.join(command)}
    """

    bat_file_path = os.path.join(os.getcwd(), f"run_telingo_{exp_name}.bat")
    with open(bat_file_path, 'w') as bat_file:
        bat_file.write(bat_content)


    result = subprocess.run(['cmd', '/c', bat_file_path], shell=True, capture_output=True, text=True, env={**os.environ, 'PYTHONUNBUFFERED': '1'})

    output = result.stdout
    errors_and_warnings = result.stderr

    if 'traceback' in errors_and_warnings.lower() or 'error' in errors_and_warnings.lower():
        print("Telingo Errors and Warnings:")
        print(errors_and_warnings)

    os.remove(bat_file_path)
    _remove_file_with_dynamic_parameters(exp_name)

    return _extract_first_action_from_telingo_output(output)

def validate_path(exp_name, actions: list, level: str, enforcing_horizon: int, last_performed_action: str, state: tuple, enforcing_norm_set: int):
    """
    calls potassco's telingo to perform ASP-checking.
    uses frozenlake_checking, level-data and a helper file to handle dynamic properties
    returns True if and only if the checked path is valid meaning no norm-violations were identified
    """

    file1 = os.path.join(os.getcwd(), "src", "planning", "frozenlake_checking.lp")
    file2 = os.path.join(os.getcwd(), "src", "planning", "levels", f"{level}.lp")
    _fill_file_for_dynamic_parameters(exp_name, state[0], state[1], last_performed_action, list(state[2]), None, actions)
    file3 = os.path.join(os.getcwd(), "src", "planning", f"dynamic_parameters_{exp_name}.lp")
    file4 = os.path.join(os.getcwd(), "src", "planning", "deontic_norms", f"deontic_norms_{enforcing_norm_set}.lp")

    # Note: starts already in the active python environment
    command = [
        # f'call',  f'{ANACONDA_PATH}', f'{CONDA_ENV_NAME}', f'&&',
        f'python', f'-m', f'telingo',
        f'--quiet=1,1,1',
        f'--imin={enforcing_horizon}', f'--imax={enforcing_horizon}',
        f'--time-limit=30',
        f'"{file1}"', f'"{file2}"', f'"{file3}"', f'"{file4}"'
    ]

    bat_content = f"""
        @echo off
        {' '.join(command)}
        """

    bat_file_path = os.path.join(os.getcwd(), f"run_telingo_{exp_name}.bat")
    with open(bat_file_path, 'w') as bat_file:
        bat_file.write(bat_content)

    result = subprocess.run(['cmd', '/c', bat_file_path], shell=True, capture_output=True, text=True)

    output = result.stdout
    errors_and_warnings = result.stderr

    if 'traceback' in errors_and_warnings or 'error' in errors_and_warnings:
        print("Telingo Errors and Warnings:")
        print(errors_and_warnings)

    os.remove(bat_file_path)
    _remove_file_with_dynamic_parameters(exp_name)

    return _extract_validation_result_from_telingo_output(output)




