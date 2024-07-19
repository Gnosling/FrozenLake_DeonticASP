import os.path
from typing import Tuple, List, Any
import numpy as np
import csv

from src.configs import configs
from src.policies.policy import Policy
from src.policies.epsilon_greedy_policy import EpsilonGreedyPolicy
from src.policies.planner_policy import PlannerPolicy
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



def read_config_param(config_name: str) -> Tuple[int, int, int, float, float, bool, dict, bool, str, float, str, int, int, int]:
    if config_name in configs.keys():
        values = configs.get(config_name)
        reps = values.get("reps")
        episodes = values.get("episodes")
        max_steps = values.get("max_steps")
        discount = values.get("discount")
        learning_rate = values.get("learning_rate")
        reversed_q_learning = values.get("reversed_q_learning")
        frozenlake = values.get("frozenlake")
        policy = values.get("policy")
        epsilon = values.get("epsilon")
        planning_strategy = values.get("planning_strategy")
        planning_horizon = values.get("planning_horizon")
        norm_set = values.get("norm_set")
        evaluation_function = values.get("evaluation_function")
        return reps, episodes, max_steps, discount, learning_rate, reversed_q_learning,frozenlake, policy, epsilon, planning_strategy, planning_horizon, norm_set, evaluation_function
    else:
        raise ValueError("Configuration was not found!")


def build_policy(config: str):
    _, _, _, discount, learning_rate, _, frozenlake, policy, epsilon, planning_strategy, planning_horizon, norm_set, evaluation_function = read_config_param(config)

    if policy == "greedy":
        obj = Policy(QTable(), learning_rate, discount)
    elif policy == "eps_greedy":
        obj = EpsilonGreedyPolicy(QTable(), learning_rate, discount, epsilon)
    elif policy == "planning":
        obj = PlannerPolicy(QTable(), learning_rate, discount, epsilon, planning_strategy, planning_horizon, frozenlake.get("name"), norm_set, evaluation_function)
    else:
        raise ValueError(f"Wrong value of policy: {policy}!")

    obj.initialize({s for s in range(frozenlake.get("tiles"))}, constants.action_set)
    target = Policy(obj.get_q_table(), learning_rate, discount)
    return obj, target


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

def get_average_returns(results):
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
        total_per_episode = extract_norm_keys(norm_set)
        avg_per_episode = dict()
        for rep in range(repetitions):
            for norm in total_per_episode.keys():
                total_per_episode[norm] += results[rep][episode][norm]
        for norm in total_per_episode.keys():
            avg_per_episode[norm] = total_per_episode[norm] / repetitions
        average_violations.append(avg_per_episode)

    return average_violations


def test_target(target, env, config):
    _, _, max_steps, _, _, _, frozenlake, _, _, _, _, norm_set, evaluation_function = read_config_param(config)
    norm_violations = extract_norm_keys(norm_set)
    trail_of_target = []
    state, info = env.reset()
    traverser_state = env.get_current_traverser_state()  # this is -1 if there is no traverser
    layout, width, height = env.get_layout()

    for step in range(max_steps):
        action_name = target.suggest_action(state)
        new_state, reward, terminated, truncated, info = env.step(action_name_to_number(action_name))
        trail_of_target.append([state, action_name, new_state, reward])

        check_violations(norm_violations, trail_of_target, terminated or step == max_steps-1, traverser_state, layout, width, height)

        traverser_state = env.get_current_traverser_state()
        state = new_state

        if terminated or truncated:
            break

    return trail_of_target, norm_violations


def check_violations(norm_violations, trail_of_target, terminated, traverser_state, layout, width, height):
    """
    checks violations of norms:
        occupiedTraverserTile
        turnedOnTraverserTile
        notReachedGoal
    """
    state, action_name, new_state, reward = trail_of_target[-1]


    for norm in norm_violations.keys():

        if norm == "notReachedGoal":
            if terminated:
                if layout[int(new_state/height)][new_state%width] != b"G":
                    norm_violations[norm] += 1

        elif norm == "occupiedTraverserTile":
            if state == traverser_state:
                norm_violations[norm] += 1

        elif norm == "turnedOnTraverserTile":
            if len(trail_of_target) > 1:
                if state == traverser_state:
                    _, previous_action, _, _ = trail_of_target[-2]
                    if action_name != previous_action:
                        norm_violations[norm] += 1

        else:
            raise ValueError(f"Unexpected norm to check: {norm}!")

    return norm_violations


def extract_norm_keys(norm_set):
    norms = dict()
    with open(os.path.join(os.getcwd(), "src", "planning", "deontic_reasonings", f"deontic_reasoning_{norm_set}.lp"), 'r') as file:
        for line in file:
            if "checks" in line.lower():
                continue
            if line.strip() == "":
                break
            key = line.strip().split(" ")[-1]
            norms[key] = 0
    return norms


def store_results(config: str, data):
    conf = configs.get(config)
    path = os.path.join(os.getcwd(), "results", f"{config}_config.txt")
    with open(path, 'w', newline='') as file:
        file.write(str(conf))
    print(f"Stored configuration in: \t {path}")

    path = os.path.join(os.getcwd(), "results", f"{config}_results.csv")
    with open(path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(data.keys())
        writer.writerows(zip(*data.values()))
    print(f"Stored data in: \t\t {path}")

def debug_print(msg: str) -> Any:
    if DEBUG_MODE:
        print(msg)