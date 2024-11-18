import os.path
from typing import Tuple, List, Any
import numpy as np
import pandas as pd
import random
import copy
import time
import matplotlib.pyplot as plt
import ast
import itertools
import seaborn as sns

from src.configs import configs
from src.policies.q_table import QTable
from . import constants
from .constants import *


    #     - 0: LEFT
    #     - 1: DOWN
    #     - 2: RIGHT
    #     - 3: UP
def action_number_to_string(number: int) -> str:
    if number == 0:
        return "LEFT"
    elif number == 1:
        return "DOWN"
    elif number == 2:
        return "RIGHT"
    elif number == 3:
        return "UP"
    else:
        return None

def action_name_to_number(name: str) -> int:
    if name == "LEFT":
        return 0
    elif name == "DOWN":
        return 1
    elif name == "RIGHT":
        return 2
    elif name == "UP":
        return 3
    else:
        return None

def get_level_data(level_name):
    path = os.path.join(os.getcwd(), "src", "planning", "levels", f"{level_name}.lp")
    with open(path, 'r', newline='') as file:
        content = file.read()

    width = None
    height = None
    goal = None
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("#const width"):
            width = int(line.split(" = ")[1][:-1])
        elif line.startswith("#const height"):
            height = int(line.split(" = ")[1][:-1])
        elif line.startswith("goal("):
            goal = int(line.split("goal(")[1][:-2])

    return width, height, goal

def extract_performed_action(current_position: int, new_position : int, width):
    diff = new_position - current_position
    if diff == 1:
        return "RIGHT"
    elif diff == -1:
        return "LEFT"
    elif diff == width:
        return "DOWN"
    elif diff == -width:
        return "UP"
    else:
        # Note: agent can slip on edges in the same tile
        return None

def compute_non_determinisic_successors(current_position: int, action, width, height):
    """
    returns all possible states the action could slip into
    """
    successors = []
    left_edge = current_position % width == 0
    right_edge = current_position % width == width - 1
    upper_edge = int(current_position / width) == 0
    lower_edge = int(current_position / width) == height - 1

    if action == "LEFT":
        if not left_edge:
            successors.append(current_position - 1)
        if not upper_edge:
            successors.append(current_position - width)
        if not lower_edge:
            successors.append(current_position + width)
        if left_edge or upper_edge or lower_edge:
            successors.append(current_position)
    elif action == "DOWN":
        if not left_edge:
            successors.append(current_position - 1)
        if not lower_edge:
            successors.append(current_position + width)
        if not right_edge:
            successors.append(current_position + 1)
        if left_edge or right_edge or lower_edge:
            successors.append(current_position)
    elif action == "RIGHT":
        if not upper_edge:
            successors.append(current_position - width)
        if not lower_edge:
            successors.append(current_position + width)
        if not right_edge:
            successors.append(current_position + 1)
        if right_edge or upper_edge or lower_edge:
            successors.append(current_position)
    elif action == "UP":
        if not left_edge:
            successors.append(current_position - 1)
        if not upper_edge:
            successors.append(current_position - width)
        if not right_edge:
            successors.append(current_position + 1)
        if left_edge or upper_edge or right_edge:
            successors.append(current_position)

    return successors

def compute_expected_successor(current_position: int, action, width, height) -> int:
    """
    returns the expected successor
    """
    left_edge = current_position % width == 0
    right_edge = current_position % width == width - 1
    upper_edge = int(current_position / width) == 0
    lower_edge = int(current_position / width) == height - 1

    if action == "LEFT":
        if not left_edge:
            return current_position-1
        else:
            return current_position
    elif action == "DOWN":
        if not lower_edge:
            return current_position+width
        else:
            return current_position
    elif action == "RIGHT":
        if not right_edge:
            return current_position+1
        else:
            return current_position
    elif action == "UP":
        if not upper_edge:
            return current_position-width
        else:
            return current_position

def compute_expected_predecessor(current_position: int, last_action, width, height) -> int:
    """
    returns the expected predecessor
    """
    left_edge = current_position % width == 0
    right_edge = current_position % width == width - 1
    upper_edge = int(current_position / width) == 0
    lower_edge = int(current_position / width) == height - 1
    previous_position = current_position

    if last_action == "LEFT":
        if not right_edge:
            return previous_position + 1
        else:
            return previous_position
    elif last_action == "DOWN":
        if not upper_edge:
            return previous_position - width
        else:
            return previous_position
    elif last_action == "RIGHT":
        if not left_edge:
            return previous_position - 1
        else:
            return previous_position
    elif last_action == "UP":
        if not lower_edge:
            return previous_position + width
        else:
            return previous_position

    return previous_position

def read_config_param(config_name: str) -> Tuple[int, int, int, dict, dict, dict, dict, dict]:
    if config_name in configs.keys():
        values = configs.get(config_name)
        repetitions = values.get("repetitions")
        episodes = values.get("episodes")
        max_steps = values.get("max_steps")
        learning = values.get("learning")
        frozenlake = values.get("frozenlake")
        planning = values.get("planning")
        deontic = values.get("deontic")
        enforcing = values.get("enforcing")
        return repetitions, episodes, max_steps, learning, frozenlake, planning, deontic, enforcing
    else:
        raise ValueError("Configuration was not found!")

def transform_to_state(current_tile: int, traverser_tile: int, presents: List):
    presents.sort()
    return (current_tile, traverser_tile, presents)


def build_policy(config: str, env):
    from src.policies.policy import Policy
    from src.policies.planner_policy import PlannerPolicy
    _, _, _, learning, frozenlake, planning, deontic, enforcing = read_config_param(config)

    if planning is None:
        behavior = Policy(QTable(learning.get("initialisation")), learning.get("learning_rate"), learning.get("learning_rate_strategy"), learning.get("learning_decay_rate"), learning.get("discount"), frozenlake.get("name"), enforcing)
    else:
        behavior = PlannerPolicy(QTable(learning.get("initialisation")), learning.get("learning_rate"), learning.get("learning_rate_strategy"), learning.get("learning_decay_rate"), learning.get("discount"), learning.get("epsilon"), planning.get("strategy"), planning.get("planning_horizon"), planning.get("delta"), frozenlake.get("name"), deontic.get("norm_set"), deontic.get("evaluation_function"), enforcing)

    presents = tuple(env.get_tiles_with_presents())
    subsets = [tuple(comb) for i in range(len(presents) + 1) for comb in itertools.combinations(presents, i)]
    behavior.initialize({ (s,t,subset) # a state is a tuple (current_position, traverser_position, presents_positions
                          for s in range(env.get_number_of_tiles())
                          for t in range(-1, env.get_number_of_tiles())
                          for subset in subsets},
                        constants.ACTION_SET, env)
    target = Policy(behavior.get_q_table(), learning.get("learning_rate"), learning.get("learning_rate_strategy"), learning.get("learning_decay_rate"), learning.get("discount"), frozenlake.get("name"), enforcing)
    return behavior, target


def compute_expected_return(discount_rate: float, rewards: List[float]) -> float:
    """
    returns discounted sum of rewards of single episode
    """
    G = [0] * len(rewards)

    # for t in reversed(range(T-1)):
    #     G[t] = rewards[t + 1] + discount_rate * G[t + 1]

    for t in range(len(rewards)):
        if t > 0:
            G[t] = rewards[t] + discount_rate * G[t-1]
        else:
            G[t] = rewards[t]

    return G[-1]


def update_state_visits(collection, results_from_one_iteration):
    if not collection:
        collection = {state: {action: 0 for action in actions} for state, actions in results_from_one_iteration.items()}

    for state, actions in results_from_one_iteration.items():
        for action, visit in actions.items():
            collection[state][action] += visit
    return collection


def get_average_numbers(results):
    average_numbers = []
    standard_deviations = []
    episodes = len(results[0])
    repetitions = len(results)
    for episode in range(episodes):
        values_per_episode = []
        for rep in range(repetitions):
            values_per_episode.append(results[rep][episode])
        average_numbers.append(sum(values_per_episode) / repetitions)
        standard_deviations.append(np.std(values_per_episode))

    return average_numbers, standard_deviations

def get_average_violations(results, norm_set):
    average_violations = []
    standard_deviations = []
    episodes = len(results[0])
    repetitions = len(results)
    for episode in range(episodes):
        values_per_episode = dict()
        norms = extract_norm_keys(norm_set)
        for norm in norms.keys():
            values_per_episode[norm] = list()
        avg_per_episode = dict()
        std_per_episode = dict()
        for rep in range(repetitions):
            for norm in values_per_episode.keys():
                values_per_episode[norm].append(results[rep][episode][norm])
        for norm in values_per_episode.keys():
            avg_per_episode[norm] = sum(values_per_episode[norm]) / repetitions
            std_per_episode[norm] = np.std(values_per_episode[norm])
        average_violations.append(avg_per_episode)
        standard_deviations.append(std_per_episode)

    return average_violations, standard_deviations


def append_violations(final_results, violations):
    if not final_results:
        final_results = {key: [] for key in violations.keys()}

    for norm, value in violations.items():
        final_results[norm].append(value)
    return final_results

def get_average_state_visits(results, reps):
    average_state_visits = {state: {action: 0 for action in actions} for state, actions in results.items()}
    for state, actions in results.items():
        for action, value in actions.items():
            average_state_visits[state][action] = value / reps
    return average_state_visits


def test_target(target, env, config, after_training):
    _, _, max_steps, _, frozenlake, _, deontic, enforcing = read_config_param(config)
    norm_violations = None
    if deontic:
        norm_violations = extract_norm_keys(deontic.get("norm_set"))
    trail_of_target = []
    state_visits = {state: {action: 0 for action in actions} for state, actions in target.q_table.get_all_values().items()}
    for state in state_visits.keys():
        state_visits[state]['VISITS'] = 0
    state, info = env.reset()
    state_visits[state]['VISITS'] += 1
    layout, width, height = env.get_layout()
    slips = 0
    last_performed_action = None
    action_name = None
    previous_state = None
    start_time = time.time()

    for step in range(max_steps):
        target.update_dynamic_env_aspects(last_performed_action, action_name, previous_state)
        action_name = target.suggest_action(state, env)
        new_state, reward, terminated, truncated, info = env.step(action_name_to_number(action_name))
        trail_of_target.append([state, action_name, new_state, reward])
        if False and after_training and enforcing and "reward_shaping" in enforcing.get("strategy"):
            target.update_after_step(state, action_name, new_state, reward, trail_of_target, env, after_training)
            # TODO: i guess reversed learning is not useful here also change with new enforcing control!
        previous_state = state
        last_performed_action = extract_performed_action(state[0], new_state[0], width)

        if last_performed_action:
            state_visits[state][last_performed_action] += 1
        state_visits[new_state]['VISITS'] += 1

        if info.get("prob") == 0.1:
            slips += 1

        if norm_violations is not None:
            _check_violations(norm_violations, trail_of_target, terminated or step == max_steps - 1, env)

        state = new_state

        if terminated or truncated:
            break

    end_time = time.time()
    return trail_of_target, norm_violations, slips, end_time-start_time, state_visits

def _tile_is_safe(tile: int, env):
    """
    checks if there are holes next to the input tile, if so returns false
    """
    if env.get_tile_symbol_of_state_number(tile) in b'H':
        return False

    layout, width, height = env.get_layout()

    row = tile/height
    col = tile%width
    adjacent_tiles = []
    if row > 0:
        adjacent_tiles.append(tile-width)
    if row < height-1:
        adjacent_tiles.append(tile+width)
    if col > 0:
        adjacent_tiles.append(tile-1)
    if col < width-1:
        adjacent_tiles.append(tile+1)

    for neighbour in adjacent_tiles:
        if env.get_tile_symbol_of_state_number(neighbour) in b'H':
            return False

    return True

def distance_to_goal(position, goal, width, height):
    return abs(position % width - goal % width) + abs(int(position / height) - int(goal / height))

def _check_violations(norm_violations, trail, terminated, env):
    """
    checks violations of norms in the current step and fills up norm_violations dictionary with the counter of the respective violation
    """
    layout, width, height = env.get_layout()
    goal = env.get_goal_tile()

    state, action_name, new_state, _ = trail[-1]
    last_performed_action = extract_performed_action(state[0], new_state[0], width)
    second_last_performed_action = None
    if len(trail) > 1:
        second_last_state, second_last_action_name, _, _ = trail[-2]
        if second_last_state:
            second_last_performed_action = extract_performed_action(second_last_state[0], state[0], width)
    old_position = state[0]
    new_position = new_state[0]
    old_traverser_position = state[1]
    new_traverser_position = new_state[1]
    previous_presents = list(state[2])
    remaining_presents = list(new_state[2])

    for norm in norm_violations.keys():
        if norm == "notReachedGoal":
            if terminated:
                if new_position != goal:
                    norm_violations[norm] += 1

        elif norm == "occupiedTraverserTile":
            if new_position == new_traverser_position:
                norm_violations[norm] += 1

        elif norm == "turnedOnTraverserTile":
            if len(trail) > 1:
                if old_position == old_traverser_position:
                    if second_last_performed_action is not None and last_performed_action is not None:
                        # Note: if one of them is 'no movement' then there was no turn
                        if second_last_performed_action != last_performed_action:
                            norm_violations[norm] += 1

        elif norm == "stolePresent":
            if len(previous_presents) != len(remaining_presents):
                difference = [p for p in previous_presents if p not in remaining_presents]
                # Note: Agent always picks up first
                if new_position in difference:
                    norm_violations[norm] += 1

        elif norm == "missedPresents":
            if terminated:
                remaining_presents = list(new_state[2])
                if len(remaining_presents) > 0:
                    norm_violations[norm] += 1

        elif norm == "movedAwayFromGoal":
            previous_distance = distance_to_goal(old_position, goal, width, height)
            new_distance = distance_to_goal(new_position, goal, width, height)
            if new_distance > previous_distance:
                norm_violations[norm] += 1

        elif norm == "leftSafeArea":
            # old was, new is not
            if _tile_is_safe(old_position, env) and not _tile_is_safe(new_position, env):
                norm_violations[norm] += 1

        elif norm == "didNotReturnToSafeArea":
            # both are not safe
            if not _tile_is_safe(old_position, env) and not _tile_is_safe(new_position, env):
                norm_violations[norm] += 1

        else:
            raise ValueError(f"Unexpected norm to check: {norm}!")

    return norm_violations


def extract_norm_keys(norm_set):
    if norm_set is None:
        return None

    norms = dict()
    reached_section = False
    with open(os.path.join(os.getcwd(), "src", "planning", "deontic_norms", f"deontic_norms_{norm_set}.lp"), 'r') as file:
        for line in file:
            if "norms:" in line.lower():
                reached_section = True
                continue
            if not reached_section:
                continue
            if line.strip() == "":
                break
            key = line.strip().split(" ")[-1]
            norms[key] = 0
    # TODO: define order of norms to be put in the dict here! such that plots always have same order
    return dict(sorted(norms.items()))

def extract_norm_levels(norm_set):
    """
    returns a dict of norms and their level in the deontic-files, if no level is given then 0 is returned as default-level
    """
    if norm_set is None:
        return None

    norms = dict()
    reached_section = False
    passed_norms = False
    with open(os.path.join(os.getcwd(), "src", "planning", "deontic_norms", f"deontic_norms_{norm_set}.lp"),
              'r') as file:
        for line in file:
            line = line.strip()
            if "norms:" in line.lower():
                reached_section = True
                continue
            if not reached_section:
                continue
            if line == "":
                continue
            if line.startswith("#program"):
                passed_norms = True
                continue

            if not passed_norms:
                key = line.strip().split(" ")[-1]
                norms[key] = 0
            else:
                if line.startswith("level"):
                    norm_string = line.split(",")[0].split("(")[1]
                    norm_level = int(line.split(",")[1].split(")")[0])
                    for key in norms.keys():
                        if key.lower() in norm_string.lower() or norm_string.lower() in key.lower():
                            norms[key] = norm_level

    return norms

def guardrail(enforcing_config, state, previous_state, last_performed_action, last_proposed_action, env):
    """
    guardrails the 'allowed' moves for the given state.
    returns the norm-confirm next actions selectable for this state
    (only considers direct violations in the successor and no future violations of all paths)
    """
    allowed_actions = {action for action in constants.ACTION_SET}
    if enforcing_config is None:
        return allowed_actions

    layout, width, height = env.get_layout()
    goal = env.get_goal_tile()
    holes = env.get_tiles_with_holes()
    sum_of_violations = dict()

    # Note: foreach action all non-deterministic successors are added to the trail and checked
    # If all actions are not allowed, then the one with the minimal sum is returned
    for action in constants.ACTION_SET:
        successor = compute_expected_successor(state[0], action, width, height)
        trail = [[previous_state, last_proposed_action, state, False]]
        norm_violations = extract_norm_keys(enforcing_config.get("norm_set"))
        terminated = successor == goal or successor in holes

        trail.append([state, action, (successor, state[1], state[2]), 0])
        _check_violations(norm_violations, trail, terminated, env)
        if any(value > 0 for value in norm_violations.values()):
            allowed_actions.remove(action)
            sum_of_violations[action] = sum(norm_violations.values())

    if not allowed_actions:
        minimal_violation = min(sum_of_violations.values())
        allowed_actions = [action for action, value in sum_of_violations.items() if value == minimal_violation]
        return allowed_actions
    else:
        return allowed_actions


def get_state_value(state, norms, level_of_norms, env):
    """
    Computes the state-value function under the given norms, CTDs cannot be expressed as state-functions.
    Violations are scaled with scaling_factor**(level_of_norms[norm]-1).
    """
    value = 0
    scaling_factor = 2
    layout, width, height = env.get_layout()
    goal = env.get_goal_tile()
    holes = env.get_tiles_with_holes()
    cracks = env.get_tiles_with_cracks()
    terminated = False
    if state[0] == goal or state[0] in holes or (state[0] == state[1] and state[0] in cracks): # TODO: test this
        terminated = True
    for norm in norms.keys():
        if norm == "notReachedGoal":
            if terminated and state[0] == goal:
                value += 1 * (scaling_factor**(level_of_norms[norm]-1))

        elif norm == "occupiedTraverserTile":
            if state[0] != state[1]:
                value += 1 * (scaling_factor**(level_of_norms[norm]-1))

        elif norm == "turnedOnTraverserTile":
            continue  # Note: CTD cannot be checked as state-function

        elif norm == "stolePresent":
            if len(state[2]) > 0:
                value += len(state[2]) * (scaling_factor**(level_of_norms[norm]-1))

        elif norm == "missedPresents":
            if terminated and len(state[2]) > 0:
                value += -len(state[2]) * (scaling_factor**(level_of_norms[norm]-1))

        elif norm == "movedAwayFromGoal":
            value += (-distance_to_goal(state[0], goal, width, height) / 0.8) * (scaling_factor ** (level_of_norms[norm] - 1))

        elif norm == "leftSafeArea":
            if _tile_is_safe(state[0], env):
                value += 1 * (scaling_factor**(level_of_norms[norm]-1))

        elif norm == "didNotReturnToSafeArea":
            continue  # Note: CTD cannot be checked as state-function

        else:
            raise ValueError(f"Unexpected norm to check: {norm}!")

    value = value / 100  # Note: this is to scale all rewards down
    return value


def get_state_action_penalty(trail, terminated, norms, level_of_norms, env):
    """
    Computes the state-action-penalty under the given norms.
    Violations are scaled with scaling_factor**(level_of_norms[norm]-1).
    """
    if not trail:
        return 0

    scaling_factor = 2
    # Note: trail[i] = [state, action, new_state, reward], but rewards don't matter for violations
    _check_violations(norms, trail, terminated, env)

    penalty = 0
    for norm, violations in norms.items():
        if violations > 0:
            penalty += violations * (scaling_factor**(level_of_norms[norm]-1))

    penalty = -penalty /100  # Note: this is to scale all rewards down, penalties are negative
    return penalty


def get_shaped_rewards(enforcing, discount, state, new_state, trail, env):
    norms = extract_norm_keys(enforcing.get("norm_set"))
    level_of_norms = extract_norm_levels(enforcing.get("norm_set"))
    terminated = env.is_terminated()
    shaped_rewards = 0
    if "optimal_reward_shaping" in enforcing.get("strategy"):
        shaped_rewards = discount * get_state_value(new_state, norms, level_of_norms, env) - get_state_value(state, norms, level_of_norms, env)
    elif "full_reward_shaping" in enforcing.get("strategy"):
        # NOTE: the full shaping uses _check_violations(..) for state_actions penalties, hence both need separate norms-dicts
        if len(trail) <= 1:
            shaped_rewards = ((discount * (get_state_action_penalty(trail, terminated, extract_norm_keys(enforcing.get("norm_set")), level_of_norms, env)))
                              - (get_state_action_penalty([[trail[0][0], None, trail[0][0], 0]], False, extract_norm_keys(enforcing.get("norm_set")), level_of_norms, env)))
        else:
            shaped_rewards = ((discount * (get_state_action_penalty(trail, terminated, extract_norm_keys(enforcing.get("norm_set")), level_of_norms, env)))
                              - (get_state_action_penalty(trail[:-1], False, extract_norm_keys(enforcing.get("norm_set")), level_of_norms, env)))
    return shaped_rewards


def store_results(config: str, training_returns_avg, training_returns_stddev, training_steps_avg, training_steps_stddev, training_slips_avg, training_slips_stddev, training_violations_avg, training_violations_stddev, training_fitting_times_avg, training_fitting_times_stddev, training_inference_times_avg, training_inference_times_stddev, training_state_visits,
                  final_returns, final_steps, final_slips, final_violations, final_inference_times, final_state_visits,
                  enforced_returns, enforced_steps, enforced_slips, enforced_violations, enforced_inference_times, enforced_state_visits):
    conf = configs.get(config)
    experiment_folder = os.path.join(os.getcwd(), "results", f"{config[0]}", f"{config}")
    training_folder = os.path.join(experiment_folder, "training")
    final_folder = os.path.join(experiment_folder, "final")
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    if not os.path.exists(training_folder):
        os.makedirs(training_folder)
    if not os.path.exists(final_folder):
        os.makedirs(final_folder)

    path = os.path.join(experiment_folder, f"{config}_config.txt")
    with open(path, 'w', newline='') as file:
        file.write(str(conf))
    print(f"Stored configuration in: \t {path}")

    # Training results
    path = os.path.join(training_folder, f"{config}_returns.txt")
    with open(path, 'w', newline='') as file:
        file.write(str(training_returns_avg) + "\n")
        file.write(str(training_returns_stddev))

    path = os.path.join(training_folder, f"{config}_steps.txt")
    with open(path, 'w', newline='') as file:
        file.write(str(training_steps_avg) + "\n")
        file.write(str(training_steps_stddev))

    path = os.path.join(training_folder, f"{config}_slips.txt")
    with open(path, 'w', newline='') as file:
        file.write(str(training_slips_avg) + "\n")
        file.write(str(training_slips_stddev))

    if training_violations_avg:
        path = os.path.join(training_folder, f"{config}_violations.txt")
        with open(path, 'w', newline='') as file:
            file.write(str(training_violations_avg) + "\n")
            file.write(str(training_violations_stddev))

    path = os.path.join(training_folder, f"{config}_fitting_times.txt")
    with open(path, 'w', newline='') as file:
        file.write(str(training_fitting_times_avg) + "\n")
        file.write(str(training_fitting_times_stddev))

    path = os.path.join(training_folder, f"{config}_inference_times.txt")
    with open(path, 'w', newline='') as file:
        file.write(str(training_inference_times_avg) + "\n")
        file.write(str(training_inference_times_stddev))

    path = os.path.join(training_folder, f"{config}_state_visits.txt")
    with open(path, 'w', newline='') as file:
        file.write(str(training_state_visits))


    # Final results
    path = os.path.join(final_folder, f"{config}_returns.txt")
    with open(path, 'w', newline='') as file:
        file.write(str(final_returns))

    path = os.path.join(final_folder, f"{config}_steps.txt")
    with open(path, 'w', newline='') as file:
        file.write(str(final_steps))

    path = os.path.join(final_folder, f"{config}_slips.txt")
    with open(path, 'w', newline='') as file:
        file.write(str(final_slips))

    path = os.path.join(final_folder, f"{config}_violations.txt")
    with open(path, 'w', newline='') as file:
        file.write(str(final_violations))

    path = os.path.join(final_folder, f"{config}_inference_times.txt")
    with open(path, 'w', newline='') as file:
        file.write(str(final_inference_times))

    path = os.path.join(final_folder, f"{config}_state_visits.txt")
    with open(path, 'w', newline='') as file:
        file.write(str(final_state_visits))


    # Enforced results
    if enforced_returns is not None:
        path = os.path.join(final_folder, f"{config}_returns_enforced.txt")
        with open(path, 'w', newline='') as file:
            file.write(str(enforced_returns))

    if enforced_steps is not None:
        path = os.path.join(final_folder, f"{config}_steps_enforced.txt")
        with open(path, 'w', newline='') as file:
            file.write(str(enforced_steps))

    if enforced_slips is not None:
        path = os.path.join(final_folder, f"{config}_slips_enforced.txt")
        with open(path, 'w', newline='') as file:
            file.write(str(enforced_slips))

    if enforced_violations:
        path = os.path.join(final_folder, f"{config}_violations_enforced.txt")
        with open(path, 'w', newline='') as file:
            file.write(str(enforced_violations))

    if enforced_inference_times is not None:
        path = os.path.join(final_folder, f"{config}_inference_times_enforced.txt")
        with open(path, 'w', newline='') as file:
            file.write(str(enforced_inference_times))

    if enforced_state_visits is not None:
        path = os.path.join(final_folder, f"{config}_state_visits_enforced.txt")
        with open(path, 'w', newline='') as file:
            file.write(str(enforced_state_visits))


def plot_experiment(config: str):

    # TODO: add std_deviations also in the plots?
    #  -> not for violations, state_visits, runtimes, step+slips
    #  -> for steps+slips (final->done), returns (training->done) (final->done),
    repetitions, episodes, max_steps, learning, frozenlake, planning, deontic, enforcing = read_config_param(config)
    maximum = 1
    experiment_folder = os.path.join(os.getcwd(), "results", f"{config[0]}", f"{config}")
    training_folder = os.path.join(experiment_folder, "training")
    final_folder = os.path.join(experiment_folder, "final")
    plot_folder = os.path.join(os.getcwd(), "plots", f"{config[0]}", f"{config}")
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)


    #  ---   ---   ---   plots: training_returns   ---   ---   ---
    path = os.path.join(training_folder, f"{config}_returns.txt")
    avg_returns = []
    std_returns = []
    with open(path, 'r', newline='') as file:
        content = file.read()
        avg_returns = np.array(ast.literal_eval(content.split("\n")[0]))
        std_returns = np.array(ast.literal_eval(content.split("\n")[1]))

    plt.figure(figsize=(10,6))
    plt.plot(list(range(1,episodes+1)), avg_returns, label='expected return', linewidth=1.7, color='royalblue', marker='o', markersize=4)
    plt.fill_between(list(range(1,episodes+1)), avg_returns - std_returns, avg_returns + std_returns, color='blue', alpha=0.2, label='standard deviation')
    plt.plot(list(range(1,episodes+1)), [maximum] * episodes, color='limegreen', linestyle='-.', linewidth=1.2, label=f'maximum = {maximum}')
    plt.axhline(y=0, color='dimgray', linestyle='-', linewidth=0.7)
    plt.grid(True, which='both', axis='y', linestyle='-', linewidth=0.2, color='grey')

    plt.title(f'{config} - Training returns')
    plt.figtext(0.5, 0.01, f'{frozenlake.get("name")}, {planning}, norm_set={deontic}\n', ha='center', va='center', fontsize=9)
    plt.xlabel('episodes')
    plt.ylabel('returns')
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,0]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left', framealpha=1.0)

    plt.xlim(1, episodes)
    plt.ylim(-0.02, 1.02)

    plt.savefig(os.path.join(plot_folder, f"{config}_returns_training.png"))
    # plt.show()
    plt.close()


    # ---   ---   ---   plots: final+enforced returns   ---   ---   ---
    path = os.path.join(final_folder, f"{config}_returns.txt")
    final_returns = None
    with open(path, 'r', newline='') as file:
        content = file.read()
        final_returns = ast.literal_eval(content)

    final_percentage = final_returns.count(1) / len(final_returns)
    final_std_dev = np.sqrt(final_percentage * (1 - final_percentage) / len(final_returns))
    final_bar_size = final_percentage

    path = os.path.join(final_folder, f"{config}_returns_enforced")
    enforced_returns = None
    if os.path.exists(path):
        with open(path, 'r', newline='') as file:
            content = file.read()
            enforced_returns = ast.literal_eval(content)

    if enforced_returns:
        enforced_percentage = enforced_returns.count(1) / len(enforced_returns)
        enforced_std_dev = np.sqrt(enforced_percentage * (1 - enforced_percentage) / len(enforced_returns))
        enforced_bar_size = enforced_percentage
        plt.figure(figsize=(10, 8))
        plt.bar(['final returns', 'enforced returns'], [final_bar_size, enforced_bar_size], label='Percentage of successes', color='skyblue', width=0.5, yerr=[final_std_dev, enforced_std_dev], capsize=15)
    else:
        plt.figure(figsize=(4.1, 8))
        plt.bar(['final returns'], [final_bar_size], label='Percentage of successes', color='skyblue', width=0.5, yerr=final_std_dev, capsize=30)

    plt.grid(True, which='both', axis='y', linestyle='-', linewidth=0.2, color='grey')
    plt.ylabel('Percentage')
    plt.title('Stacked Bar Plot of Binary Datasets (Relative Split)')
    plt.ylim(0, 1)
    plt.yticks([i/10 for i in range(11)])
    plt.legend()

    # data = pd.DataFrame({
    #     'Type': ['Returns'] * len(final_returns),
    #     'Value': final_returns
    # })
    #
    # if enforced_returns:
    #     data = pd.DataFrame({
    #         'Type': ['Returns'] * len(final_returns) + ['Enforced returns'] * len(enforced_returns),
    #         'Value': np.concatenate([final_returns, enforced_returns])
    #     })
    #
    # plt.figure(figsize=(14, 8))
    # # TODO: this does not really work for simple return!
    # sns.boxplot(x='Type', y='Value', data=data, palette='Set2', width=0.4)
    # sns.stripplot(x='Type', y='Value', data=data, jitter=True, color='black', alpha=0.5)
    # plt.title('Box Plot with Jittered Points for Return Distribution by Group')
    # plt.xlabel('Target Groups')
    # plt.ylabel('Return')
    # plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.savefig(os.path.join(plot_folder, f"{config}_returns_final.png"))
    # plt.show()
    plt.close()


    #  ---   ---   ---   plots: training_runtimes   ---   ---   ---
    path = os.path.join(training_folder, f"{config}_fitting_times.txt")
    fitting_times = []
    with open(path, 'r', newline='') as file:
        content = file.read()
        fitting_times = ast.literal_eval(content.split("\n")[0])

    inference_times = []
    path = os.path.join(training_folder, f"{config}_inference_times.txt")
    with open(path, 'r', newline='') as file:
        content = file.read()
        inference_times = ast.literal_eval(content.split("\n")[0])

    plt.figure(figsize=(10, 6))
    plt.plot(list(range(1, episodes + 1)), fitting_times, label='fitting', linewidth=1.7, color='royalblue',
             marker='o', markersize=4)
    plt.plot(list(range(1, episodes + 1)), inference_times, label='inference', linewidth=1.7, color='seagreen',
             marker='o', markersize=4)
    plt.axhline(y=0, color='dimgray', linestyle='-', linewidth=0.7)
    plt.grid(True, which='both', axis='y', linestyle='-', linewidth=0.2, color='grey')

    plt.title(f'{config} - Average time for training and inference of target policy')
    plt.figtext(0.5, 0.01,
                f'{frozenlake.get("name")}, {planning}, norm_set={deontic}\n',
                ha='center', va='center', fontsize=9)
    plt.xlabel('episodes')
    plt.ylabel('seconds')
    plt.legend(loc='upper right', framealpha=1.0)

    plt.xlim(1, episodes)
    plt.ylim(-0.02, max(fitting_times) + 1)

    plt.savefig(os.path.join(plot_folder, f"{config}_runtimes_training.png"))
    # plt.show()
    plt.close()


    #  ---   ---   ---   plots: final+enforced runtimes   ---   ---   ---
    path = os.path.join(final_folder, f"{config}_inference_times.txt")
    final_inference_times = None
    with open(path, 'r', newline='') as file:
        content = file.read()
        final_inference_times = ast.literal_eval(content)

    path = os.path.join(final_folder, f"{config}_inference_times_enforced")
    enforced_inference_times = None
    if os.path.exists(path):
        with open(path, 'r', newline='') as file:
            content = file.read()
            enforced_inference_times = ast.literal_eval(content)

    data = pd.DataFrame({
        'Type': ['Runtimes'] * len(final_inference_times),
        'Value': final_inference_times
    })
    plt.figure(figsize=(7, 8))

    if enforced_inference_times:
        data = pd.DataFrame({
            'Type': ['Runtimes'] * len(final_inference_times) + ['Enforced runtimes'] * len(enforced_inference_times),
            'Value': np.concatenate([final_inference_times, enforced_inference_times])
        })
        plt.figure(figsize=(14, 8))

    sns.boxplot(x='Type', y='Value', data=data, palette='Set2', width=0.3)
    sns.stripplot(x='Type', y='Value', data=data, jitter=True, color='black', alpha=0.5)
    plt.title('Box Plot with Jittered Points for Return Distribution by Group')
    plt.xlabel('Target Groups')
    plt.ylabel('Runtimes (s)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.savefig(os.path.join(plot_folder, f"{config}_runtimes_final.png"))
    # plt.show()
    plt.close()


    #  ---   ---   ---   plots: training steps and slips   ---   ---   ---
    path = os.path.join(training_folder, f"{config}_steps.txt")
    steps = []
    with open(path, 'r', newline='') as file:
        content = file.read()
        steps = ast.literal_eval(content.split("\n")[0])

    path = os.path.join(training_folder, f"{config}_slips.txt")
    slips = []
    with open(path, 'r', newline='') as file:
        content = file.read()
        slips = ast.literal_eval(content.split("\n")[0])

    plt.figure(figsize=(10, 6))
    plt.plot(list(range(1, episodes + 1)), steps, label='steps', linewidth=1.7, color='royalblue',
             marker='o', markersize=4)
    plt.plot(list(range(1, episodes + 1)), slips, label='slips', linewidth=1.7, color='seagreen',
             marker='o', markersize=4)
    plt.axhline(y=0, color='dimgray', linestyle='-', linewidth=0.7)
    plt.grid(True, which='both', axis='y', linestyle='-', linewidth=0.2, color='grey')

    plt.title(f'{config} - Average number of steps and of slips')
    plt.figtext(0.5, 0.01,
                f'{frozenlake.get("name")}, {planning}, norm_set={deontic}\n',
                ha='center', va='center', fontsize=9)
    plt.xlabel('episodes')
    plt.ylabel('counts')
    plt.legend(loc='upper right', framealpha=1.0)

    plt.xlim(1, episodes)
    plt.ylim(-0.02, max(steps)+2)

    plt.savefig(os.path.join(plot_folder, f"{config}_steps_training.png"))
    # plt.show()
    plt.close()


    #  ---   ---   ---   plots: final steps and slips   ---   ---   ---
    path = os.path.join(final_folder, f"{config}_steps.txt")
    final_steps = None
    with open(path, 'r', newline='') as file:
        content = file.read()
        final_steps = ast.literal_eval(content)

    path = os.path.join(final_folder, f"{config}_slips.txt")
    final_slips = None
    with open(path, 'r', newline='') as file:
        content = file.read()
        final_slips = ast.literal_eval(content)

    path = os.path.join(final_folder, f"{config}_steps_enforced")
    enforced_steps = None
    if os.path.exists(path):
        with open(path, 'r', newline='') as file:
            content = file.read()
            enforced_steps = ast.literal_eval(content)

    path = os.path.join(final_folder, f"{config}_slips_enforced")
    enforced_slips = None
    if os.path.exists(path):
        with open(path, 'r', newline='') as file:
            content = file.read()
            enforced_slips = ast.literal_eval(content)

    data = pd.DataFrame({
        'Type': ['Steps'] * len(final_steps) + ['Slips'] * len(final_slips),
        'Value': final_steps + final_slips
    })
    plt.figure(figsize=(7, 8))

    if enforced_steps and enforced_slips:
        data = pd.DataFrame({
            'Type': ['Steps'] * len(final_steps) + ['Enforced steps'] * len(enforced_steps) + ['Slips'] * len(final_slips) + ['Enforced slips'] * len(enforced_slips),
            'Value': final_steps + enforced_steps + final_slips + enforced_slips
        })
        plt.figure(figsize=(14, 8))

    sns.boxplot(x='Type', y='Value', data=data, palette='Set2', width=0.3)
    sns.stripplot(x='Type', y='Value', data=data, jitter=True, color='black', alpha=0.5)
    plt.title('Box Plot with Jittered Points for Return Distribution by Group')
    plt.ylabel('Counts')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.savefig(os.path.join(plot_folder, f"{config}_steps_final.png"))
    # plt.show()
    plt.close()


    #  ---   ---   ---   plots: training violations   ---   ---   ---
    if deontic is not None:
        colors_of_norms = {
            'occupiedTraverserTile': 'darkred',
            'turnedOnTraverserTile': 'red',
            'notReachedGoal': 'royalblue',
            'movedAwayFromGoal': 'mediumseagreen',
            'leftSafeArea': 'gold',
            'didNotReturnToSafeArea': 'darkorange',
            'stolePresent': 'deeppink',
            'missedPresents': 'mediumvioletred'
        }  # see https://matplotlib.org/stable/gallery/color/named_colors.html

        path = os.path.join(training_folder, f"{config}_violations.txt")
        violations = []
        with open(path, 'r', newline='') as file:
            content = file.read()
            violations = ast.literal_eval(content.split("\n")[0])

        plt.figure(figsize=(10, 6))

        norms = violations[0].keys()
        for index, norm in enumerate(norms):
            plt.plot(list(range(1,episodes+1)), [elem[norm] for elem in violations], label=f'{norm}', linewidth=1.5, color=colors_of_norms[norm], marker='o', markersize=3)

        plt.grid(True, which='both', axis='y', linestyle='-', linewidth=0.2, color='grey')

        plt.title(f'{config} - Violations')
        plt.figtext(0.5, 0.01, f'{frozenlake.get("name")}, {planning.get("planning_strategy")}, norm_set={deontic.get("norm_set")}\n', ha='center', va='center', fontsize=9)
        plt.xlabel('episodes')
        plt.ylabel('violations')
        plt.legend(loc='upper right', framealpha=1.0)

        plt.xlim(1, episodes)
        plt.ylim(0, 20)
        plt.yticks(range(0, 21, 1))

        plt.savefig(os.path.join(plot_folder, f"{config}_violations_training.png"))
        # plt.show()
        plt.close()


    #  ---   ---   ---   plots: final+enforced violations   ---   ---   ---
    if deontic is not None:
        path = os.path.join(final_folder, f"{config}_violations.txt")
        final_violations = dict()
        with open(path, 'r', newline='') as file:
            content = file.read()
            final_violations = ast.literal_eval(content)

        enforced_violations = None
        path = os.path.join(final_folder, f"{config}_violations_enforced.txt")
        if os.path.exists(path):
            with open(path, 'r', newline='') as file:
                content = file.read()
                enforced_violations = ast.literal_eval(content)

        group_labels = ['notReachedGoal', 'movedAwayFromGoal', 'occupiedTraverserTile', 'turnedOnTraverserTile', 'leftSafeArea', 'didNotReturnToSafeArea', 'stolePresent', 'missedPresents']

        if not enforced_violations: # TODO: either enable the enforced_half or simple set the sns.split later to false -> what looks better?
            enforced_violations = dict()
            enforced_violations['notReachedGoal'] = [-100]  # Note: this 'enables' it the chart
            enforced_violations['movedAwayFromGoal'] = [-100]  # Note: this 'enables' it the chart

        for norm in group_labels:
            final_violations.setdefault(norm, [])
            enforced_violations.setdefault(norm, [])

        data_list = []
        type_column = []
        group_column = []
        for norm in group_labels:
            data_list.append(final_violations[norm])
            type_column.extend(['violations'] * len(final_violations[norm]))
            group_column.extend([norm] * len(final_violations[norm]))

            if any(enforced_violations[norm]):
                data_list.append(enforced_violations[norm])
                type_column.extend(['violation after enforcing'] * len(enforced_violations[norm]))
                group_column.extend([norm] * len(enforced_violations[norm]))

        flattened_data_list = [item for sublist in data_list for item in sublist]

        data = pd.DataFrame({
            'Type': type_column,
            'Value': flattened_data_list,
            'Group': group_column
        })

        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Group', y='Value', hue='Type', data=data, split=True, inner='quart', palette='Set2', bw=0.4 , cut=0)

        plt.title('Violations')
        plt.xlabel('Group')
        plt.ylabel('Violations')
        plt.ylim(-0.5, 10)
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        plt.savefig(os.path.join(plot_folder, f"{config}_violations_final.png"))
        # plt.show()
        plt.close()

        for key, value_list in final_violations.items():
            if value_list:
                avg = sum(value_list) / len(value_list)
            else:
                avg = 0
            print(f"Finale average of {key}: {avg}")

        if os.path.exists(path):
            for key, value_list in enforced_violations.items():
                if value_list:
                    avg = sum(value_list) / len(value_list)
                else:
                    avg = 0
                print(f"Enforcing average of {key}: {avg}")


    #  ---   ---   ---   plots: state-visits & preferred actions (training)  ---   ---   ---
    path = os.path.join(training_folder, f"{config}_state_visits.txt")
    with open(path, 'r', newline='') as file:
        content = file.read()
        state_actions_and_visits = ast.literal_eval(content)

    width, height, goal = get_level_data(frozenlake.get("name"))
    position_visits = {tile: 0 for tile in range(width*height)}
    sum_of_executed_actions = {tile: {action: 0 for action in actions if action != 'VISITS'} for tile in range(width*height) for actions in state_actions_and_visits.values()}
    preferred_actions = {tile: 'NONE' for tile in range(width*height)}

    for states, actions in state_actions_and_visits.items():
        position = states[0]
        for action, value in actions.items():
            if action == 'VISITS':
                position_visits[position] += value # TODO: test the merging of position and filling of these lists more!
            else:
                sum_of_executed_actions[position][action] += value

    for tile, actions in sum_of_executed_actions.items():
        preferred_action = max([action for action in actions if actions[action] != 0], key=actions.get, default=None)
        if preferred_action:
            preferred_actions[tile] = preferred_action

    grid = np.zeros((4, 4))

    for cell, total_value in position_visits.items():
        row = cell // width
        col = cell % width
        grid[row, col] = total_value

    formatted_data = np.array([[f'{val:.2f}'.rstrip('0').rstrip('.') if val != 0 else '0' for val in row] for row in grid])
    plt.figure(figsize=(width+2, height+2))
    ax = sns.heatmap(grid, cmap="viridis", cbar=True, annot=formatted_data, fmt='')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    for i in range(height):
        for j in range(width):
            index = i*width + j
            ax.text(j + 0.5, i + 0.25, f'{preferred_actions[index]}', ha='center', va='bottom', fontweight='bold',
                    color='white' if grid[i, j] < grid.max() / 2 else 'black')

    plt.title(f'{config} - Visits of target policy', fontsize=16, pad=20)
    plt.figtext(0.5, 0.01,f'{frozenlake.get("name")}, bla bla bla, \n', ha='center', va='center', fontsize=9) # TODO: define titles and subtitles for each plot

    plt.savefig(os.path.join(plot_folder, f"{config}_states_training.png"))
    # plt.show()
    plt.close()


    #  ---   ---   ---   plots: state-visits & preferred actions (final)  ---   ---   ---
    path = os.path.join(final_folder, f"{config}_state_visits.txt")
    with open(path, 'r', newline='') as file:
        content = file.read()
        state_actions_and_visits = ast.literal_eval(content)

    width, height, goal = get_level_data(frozenlake.get("name"))
    position_visits = {tile: 0 for tile in range(width * height)}
    sum_of_executed_actions = {tile: {action: 0 for action in actions if action != 'VISITS'} for tile in
                               range(width * height) for actions in state_actions_and_visits.values()}
    preferred_actions = {tile: 'NONE' for tile in range(width * height)}

    for states, actions in state_actions_and_visits.items():
        position = states[0]
        for action, value in actions.items():
            if action == 'VISITS':
                position_visits[
                    position] += value  # TODO: test the merging of position and filling of these lists more!
            else:
                sum_of_executed_actions[position][action] += value

    for tile, actions in sum_of_executed_actions.items():
        preferred_action = max([action for action in actions if actions[action] != 0], key=actions.get, default=None)
        if preferred_action:
            preferred_actions[tile] = preferred_action

    grid = np.zeros((4, 4))

    for cell, total_value in position_visits.items():
        row = cell // width
        col = cell % width
        grid[row, col] = total_value

    formatted_data = np.array(
        [[f'{val:.2f}'.rstrip('0').rstrip('.') if val != 0 else '0' for val in row] for row in grid])
    plt.figure(figsize=(width + 2, height + 2))
    ax = sns.heatmap(grid, cmap="viridis", cbar=True, annot=formatted_data, fmt='')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    for i in range(height):
        for j in range(width):
            index = i * width + j
            ax.text(j + 0.5, i + 0.25, f'{preferred_actions[index]}', ha='center', va='bottom', fontweight='bold',
                    color='white' if grid[i, j] < grid.max() / 2 else 'black')

    plt.title(f'{config} - Final state visits', fontsize=16, pad=20)
    plt.figtext(0.5, 0.01, f'{frozenlake.get("name")}, bla bla bla, \n', ha='center', va='center',
                fontsize=9)  # TODO: define titles and subtitles for each plot

    plt.savefig(os.path.join(plot_folder, f"{config}_states_final.png"))
    # plt.show()
    plt.close()

    #  ---   ---   ---   plots: state-visits & preferred actions (enforced)  ---   ---   ---
    path = os.path.join(final_folder, f"{config}_state_visits_enforced.txt")
    if os.path.exists(path):
        with open(path, 'r', newline='') as file:
            content = file.read()
            state_actions_and_visits = ast.literal_eval(content)

        width, height, goal = get_level_data(frozenlake.get("name"))
        position_visits = {tile: 0 for tile in range(width * height)}
        sum_of_executed_actions = {tile: {action: 0 for action in actions if action != 'VISITS'} for tile in
                                   range(width * height) for actions in state_actions_and_visits.values()}
        preferred_actions = {tile: 'NONE' for tile in range(width * height)}

        for states, actions in state_actions_and_visits.items():
            position = states[0]
            for action, value in actions.items():
                if action == 'VISITS':
                    position_visits[
                        position] += value  # TODO: test the merging of position and filling of these lists more!
                else:
                    sum_of_executed_actions[position][action] += value

        for tile, actions in sum_of_executed_actions.items():
            preferred_action = max([action for action in actions if actions[action] != 0], key=actions.get, default=None)
            if preferred_action:
                preferred_actions[tile] = preferred_action

        grid = np.zeros((4, 4))

        for cell, total_value in position_visits.items():
            row = cell // width
            col = cell % width
            grid[row, col] = total_value

        formatted_data = np.array(
            [[f'{val:.2f}'.rstrip('0').rstrip('.') if val != 0 else '0' for val in row] for row in grid])
        plt.figure(figsize=(width + 2, height + 2))
        ax = sns.heatmap(grid, cmap="viridis", cbar=True, annot=formatted_data, fmt='')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        for i in range(height):
            for j in range(width):
                index = i * width + j
                ax.text(j + 0.5, i + 0.25, f'{preferred_actions[index]}', ha='center', va='bottom', fontweight='bold',
                        color='white' if grid[i, j] < grid.max() / 2 else 'black')

        plt.title(f'{config} - Enforced state visits', fontsize=16, pad=20)
        plt.figtext(0.5, 0.01, f'{frozenlake.get("name")}, bla bla bla, \n', ha='center', va='center',
                    fontsize=9)  # TODO: define titles and subtitles for each plot

        plt.savefig(os.path.join(plot_folder, f"{config}_states_enforced.png"))
        # plt.show()
        plt.close()

    print(f"Stored plots in: \t {plot_folder}")


def debug_print(msg: str) -> Any:
    if DEBUG_MODE:
        print(msg)