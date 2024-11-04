import os.path
from typing import Tuple, List, Any
import numpy as np
import random
import matplotlib.pyplot as plt
import ast
import itertools

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

def _compute_non_determinisic_successors(current_position: int, action, width, height):
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

def get_average_numbers(results):
    average_returns = []
    episodes = len(results[0])
    repetitions = len(results)
    for episode in range(episodes):
        total_per_episode = 0
        for rep in range(repetitions):
            total_per_episode += results[rep][episode]
        average_returns.append(total_per_episode / repetitions)

    return average_returns

def get_average_violations(results, norm_set):
    average_violations = []
    episodes = len(results[0])
    repetitions = len(results)
    for episode in range(episodes):
        total_per_episode = _extract_norm_keys(norm_set)
        avg_per_episode = dict()
        for rep in range(repetitions):
            for norm in total_per_episode.keys():
                total_per_episode[norm] += results[rep][episode][norm]
        for norm in total_per_episode.keys():
            avg_per_episode[norm] = total_per_episode[norm] / repetitions
        average_violations.append(avg_per_episode)

    return average_violations


def test_target(target, env, config, after_training):
    _, _, max_steps, _, frozenlake, _, deontic, enforcing = read_config_param(config)
    if enforcing:
        # enables / disables enforcing in the right phase
        if after_training and enforcing.get("phase") == "after_training":
            target.set_enforcing(enforcing)
        # elif not after_training and enforcing.get("phase") == "during_training":
        #     target.set_enforcing(enforcing)
        else:
            target.set_enforcing(None)
    norm_violations = _extract_norm_keys(deontic.get("norm_set"))
    trail_of_target = []
    state, info = env.reset()
    layout, width, height = env.get_layout()
    slips = 0
    last_performed_action = None
    action_name = None
    previous_state = None

    for step in range(max_steps):
        target.update_dynamic_env_aspects(last_performed_action, action_name, previous_state)
        action_name = target.suggest_action(state, env)
        new_state, reward, terminated, truncated, info = env.step(action_name_to_number(action_name))
        if after_training and enforcing and "reward_shaping" in enforcing.get("strategy"):
            target.update_after_step(state, action_name, new_state, reward, env, after_training)
            # TODO: i guess reversed learning is not useful here
        trail_of_target.append([state, action_name, new_state, reward])
        previous_state = state
        last_performed_action = extract_performed_action(state[0], new_state[0], width)

        if info.get("prob") == 0.1:
            slips += 1

        if norm_violations is not None:
            _check_violations(norm_violations, trail_of_target, last_performed_action, terminated or step == max_steps - 1, env)

        state = new_state

        if terminated or truncated:
            break

    return trail_of_target, norm_violations, slips

def _tile_is_safe(tile: int, env, width, height):
    """
    checks if there are holes next to the input tile, if so returns false
    """
    if env.get_tile_symbol_of_state_number(tile) in b'H':
        return False

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

def _distance_to_goal(position, goal, width, height):
    return abs(position % width - goal % width) + abs(int(position / height) - int(goal / height))

def _check_violations(norm_violations, trail_of_target, last_performed_action, terminated, env):
    """
    checks violations of norms in the current step and fills up norm_violations dictionary with the counter of the respective violation
    """
    layout, width, height = env.get_layout()
    goal = env.get_goal_tile()

    state, action_name, new_state, _ = trail_of_target[-1]
    old_position = state[0]
    new_position = new_state[0]
    traverser_state = state[0] # TODO: which traveser shouldt be taken here?
    previous_presents = list(state[2])
    remaining_presents = list(new_state[2])

    for norm in norm_violations.keys():
        if norm == "notReachedGoal":
            if terminated:
                if new_position != goal:
                    norm_violations[norm] += 1

        elif norm == "occupiedTraverserTile":
            if old_position == traverser_state: # TODO: old or new position?
                norm_violations[norm] += 1

        elif norm == "turnedOnTraverserTile":
            if len(trail_of_target) > 1:
                if old_position == traverser_state:
                    # _, previous_action, _, _ = trail_of_target[-2] # Note: last_proposed_action is not the same as last_performed_action
                    # TODO: test this violation!
                    if action_name != last_performed_action:
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
            previous_distance = _distance_to_goal(old_position, goal, width, height)
            new_distance = _distance_to_goal(new_position, goal, width, height)
            if new_distance > previous_distance:
                norm_violations[norm] += 1

        elif norm == "leftSafeArea":
            # old was, new is not
            if _tile_is_safe(old_position, env, width, height) and not _tile_is_safe(new_position, env, width, height):
                norm_violations[norm] += 1

        elif norm == "didNotReturnToSafeArea":
            # both are not safe
            if not _tile_is_safe(old_position, env, width, height) and not _tile_is_safe(new_position, env, width, height):
                norm_violations[norm] += 1

        else:
            raise ValueError(f"Unexpected norm to check: {norm}!")

    return norm_violations


def _extract_norm_keys(norm_set):
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

def _extract_norm_levels(norm_set):
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
    # TODO: should it consider all successors or only the expected?
    for action in constants.ACTION_SET:
        successor = compute_expected_successor(state[0], action, width, height)
        trail = [[previous_state, last_proposed_action, state, False]]
        norm_violations = _extract_norm_keys(enforcing_config.get("norm_set"))
        terminated = successor == goal or successor in holes

        trail.append([state, action, (successor, state[1], state[2]), 0])
        _check_violations(norm_violations, trail, last_performed_action, terminated, env)
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
    for norm in norms.keys():
        if norm == "notReachedGoal":
            if state[0] == goal:
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
            if len(state[2]) > 0:
                value += -len(state[2]) * (scaling_factor**(level_of_norms[norm]-1))

        elif norm == "movedAwayFromGoal":
            value += -_distance_to_goal(state[0], goal, width, height)/0.8 * (scaling_factor**(level_of_norms[norm]-1))

        elif norm == "leftSafeArea":
            if _tile_is_safe(state[0], env, width, height):
                value += 1 * (scaling_factor**(level_of_norms[norm]-1))

        elif norm == "didNotReturnToSafeArea":
            continue  # Note: CTD cannot be checked as state-function

        else:
            raise ValueError(f"Unexpected norm to check: {norm}!")

    value = value / 100  # Note: this is to scale all rewards down
    return value

def get_state_action_value(state, action, last_action, norms, level_of_norms, env):
    """
    Computes the state-action-value under the given norms.
    Violations are scaled with scaling_factor**(level_of_norms[norm]-1).
    """
    scaling_factor = 2
    # TODO: call use this function for the norm-initialisation?
    layout, width, height = env.get_layout()
    goal = env.get_goal_tile()

    expected_old_state = (compute_expected_predecessor(state[0], last_action, width, height), state[1], state[2])
    expected_successor = compute_expected_successor(state[0], action, width, height)
    expected_remaining_presents = [elem for elem in state[2] if elem != expected_successor]
    expected_new_state = (expected_successor, state[1], expected_remaining_presents)
    terminated = goal == expected_successor

    trail = [[expected_old_state, last_action, state, 0], [state, action, expected_new_state, terminated]]
    # Note: trail[i] = [state, action, new_state, reward], but rewards does not matter for violations

    _check_violations(norms, trail, last_action, terminated, env)

    value = 0
    for norm, violations in norms.items():
        if violations > 0:
            value += violations * (scaling_factor**(level_of_norms[norm]-1))

    value = value /100  # Note: this is to scale all rewards down
    return value


def get_shaped_rewards(enforcing, discount, last_action, state, action, new_state, env):
    norms = _extract_norm_keys(enforcing.get("norm_set"))
    level_of_norms = _extract_norm_levels(enforcing.get("norm_set"))
    shaped_rewards = 0
    if "optimal_reward_shaping" in enforcing.get("strategy"):
        shaped_rewards = discount * get_state_value(new_state, norms, level_of_norms, env) - get_state_value(state, norms, level_of_norms, env)
    elif "full_reward_shaping" in enforcing.get("strategy"):
        # NOTE: the full shaping uses _check_violations(..) for state_actions, hence both need separate norms-dicts
        shaped_rewards = (discount * (get_state_action_value(new_state, action, last_action, _extract_norm_keys(enforcing.get("norm_set")), level_of_norms, env))
                          -(get_state_action_value(state, action, last_action, _extract_norm_keys(enforcing.get("norm_set")), level_of_norms, env)))
    return shaped_rewards


def store_results(config: str, returns, steps, slips, violations, enforced_returns, enforced_steps, enforced_slips, enforced_violations):
    conf = configs.get(config)
    path = os.path.join(os.getcwd(), "results", f"{config[0]}", f"{config}_config.txt")
    with open(path, 'w', newline='') as file:
        file.write(str(conf))
    print(f"Stored configuration in: \t {path}")


    path = os.path.join(os.getcwd(), "results", f"{config[0]}", f"{config}_return.txt")
    with open(path, 'w', newline='') as file:
        file.write(str(returns))
    print(f"Stored return in: \t {path}")

    if enforced_returns is not None:
        path = os.path.join(os.getcwd(), "results", f"{config[0]}", f"{config}_return_enforced.txt")
        with open(path, 'w', newline='') as file:
            file.write(str(enforced_returns))
        print(f"Stored enforced return in: \t {path}")


    path = os.path.join(os.getcwd(), "results", f"{config[0]}", f"{config}_steps.txt")
    with open(path, 'w', newline='') as file:
        file.write(str(steps))
    print(f"Stored steps in: \t {path}")

    if enforced_steps is not None:
        path = os.path.join(os.getcwd(), "results", f"{config[0]}", f"{config}_steps_enforced.txt")
        with open(path, 'w', newline='') as file:
            file.write(str(enforced_steps))
        print(f"Stored enforced return in: \t {path}")


    path = os.path.join(os.getcwd(), "results", f"{config[0]}", f"{config}_slips.txt")
    with open(path, 'w', newline='') as file:
        file.write(str(slips))
    print(f"Stored slips in: \t {path}")

    if enforced_slips is not None:
        path = os.path.join(os.getcwd(), "results", f"{config[0]}", f"{config}_slips_enforced.txt")
        with open(path, 'w', newline='') as file:
            file.write(str(enforced_slips))
        print(f"Stored enforced slips in: \t {path}")


    if violations:
        path = os.path.join(os.getcwd(), "results", f"{config[0]}", f"{config}_violations.txt")
        with open(path, 'w', newline='') as file:
            file.write(str(violations))
        print(f"Stored violations in: \t {path}")

    if enforced_violations:
        path = os.path.join(os.getcwd(), "results", f"{config[0]}", f"{config}_violations_enforced.txt")
        with open(path, 'w', newline='') as file:
            file.write(str(enforced_violations))
        print(f"Stored enforced violations in: \t {path}")


def plot_experiment(config: str):
    # TODO: use seperate plots for enforcing?
    repetitions, episodes, max_steps, learning, frozenlake, planning, deontic, enforcing = read_config_param(config)
    optimum = 1

    path = os.path.join(os.getcwd(), "results", f"{config[0]}", f"{config}_return.txt")
    returns = []
    with open(path, 'r', newline='') as file:
        content = file.read()
        returns = ast.literal_eval(content)

    plt.figure(figsize=(10,6))
    plt.plot(list(range(1,episodes+1)), returns, label='expected return', linewidth=1.7, color='royalblue', marker='o', markersize=4)
    plt.plot(list(range(1,episodes+1)), [optimum] * episodes, color='limegreen', linestyle='-.', linewidth=1.2, label=f'optimum = {optimum}')
    plt.axhline(y=0, color='dimgray', linestyle='-', linewidth=0.7)
    plt.grid(True, which='both', axis='y', linestyle='-', linewidth=0.2, color='grey')

    plt.title(f'{config} - Return of target policy')
    plt.figtext(0.5, 0.01, f'{frozenlake.get("name")}, {planning.get("planning_strategy")}, norm_set={deontic.get("norm_set")}\n', ha='center', va='center', fontsize=9)
    plt.xlabel('episode')
    plt.ylabel('return')
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,0]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left', framealpha=1.0)

    plt.xlim(1, episodes)
    plt.ylim(-0.02, 1.02)

    plt.savefig(os.path.join(os.getcwd(), "plots", f"{config[0]}", f"{config}_return.png"))
    # plt.show()
    plt.close()


    path = os.path.join(os.getcwd(), "results", f"{config[0]}", f"{config}_steps.txt")
    steps = []
    with open(path, 'r', newline='') as file:
        content = file.read()
        steps = ast.literal_eval(content)

    path = os.path.join(os.getcwd(), "results", f"{config[0]}", f"{config}_slips.txt")
    slips = []
    with open(path, 'r', newline='') as file:
        content = file.read()
        slips = ast.literal_eval(content)

    plt.figure(figsize=(10, 6))
    plt.plot(list(range(1, episodes + 1)), steps, label='number of steps', linewidth=1.7, color='royalblue',
             marker='o', markersize=4)
    plt.plot(list(range(1, episodes + 1)), slips, label='number of slips', linewidth=1.7, color='yellow',
             marker='o', markersize=4)
    plt.axhline(y=0, color='dimgray', linestyle='-', linewidth=0.7)
    plt.grid(True, which='both', axis='y', linestyle='-', linewidth=0.2, color='grey')

    plt.title(f'{config} - Average number of steps and of slips of target policy')
    plt.figtext(0.5, 0.01,
                f'{frozenlake.get("name")}, {planning.get("planning_strategy")}, norm_set={deontic.get("norm_set")}\n',
                ha='center', va='center', fontsize=9)
    plt.xlabel('episode')
    plt.ylabel('number')
    plt.legend(loc='upper right', framealpha=1.0)

    plt.xlim(1, episodes)
    plt.ylim(-0.02, max(steps)+2)

    plt.savefig(os.path.join(os.getcwd(), "plots", f"{config[0]}", f"{config}_steps.png"))
    # plt.show()
    plt.close()


    if deontic.get("norm_set") is not None:
        # the notReachedGoal should be the inverse of return, thus update rewards to 0 / 1
        colors_of_norms = {
            'occupiedTraverserTile': 'darkred',
            'turnedOnTraverserTile': 'red',
            'notReachedGoal': 'royalblue',
            'movedAwayFromGoal': 'mediumseagreen',
            'leftSafeArea': 'gold',
            'didNotReturnToSafeArea': 'darkorange',
            'stolePresent': 'deeppink',
            'missedPresents': 'mediumvioletred'
        } # see https://matplotlib.org/stable/gallery/color/named_colors.html

        path = os.path.join(os.getcwd(), "results", f"{config[0]}", f"{config}_violations.txt")
        violations = []
        with open(path, 'r', newline='') as file:
            content = file.read()
            violations = ast.literal_eval(content)

        plt.figure(figsize=(10, 6))

        norms = violations[0].keys()
        for index, norm in enumerate(norms):
            plt.plot(list(range(1,episodes+1)), [elem[norm] for elem in violations], label=f'{norm}', linewidth=1.5, color=colors_of_norms[norm], marker='o', markersize=3)

        plt.grid(True, which='both', axis='y', linestyle='-', linewidth=0.2, color='grey')

        plt.title(f'{config} - Violations of target policy')
        plt.figtext(0.5, 0.01, f'{frozenlake.get("name")}, {planning.get("planning_strategy")}, norm_set={deontic.get("norm_set")}\n', ha='center', va='center', fontsize=9)
        plt.xlabel('episode')
        plt.ylabel('violations')
        plt.legend(loc='upper right', framealpha=1.0)

        plt.xlim(1, episodes)
        plt.ylim(0, 20)
        plt.yticks(range(0, 21, 1))

        plt.savefig(os.path.join(os.getcwd(), "plots", f"{config[0]}", f"{config}_violations.png"))
        # plt.show()
        plt.close()


    # TODO: plot head-map of visited states -> needs new results

def debug_print(msg: str) -> Any:
    if DEBUG_MODE:
        print(msg)