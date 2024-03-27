import os.path
from typing import Tuple, List, Any
import numpy as np
import csv

from src.configs import configs
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



def read_config_param(config_name: str) -> Tuple[int, int, int, float, float, dict, bool, str, float, str]:
    if config_name in configs.keys():
        values = configs.get(config_name)
        reps = values.get("reps")
        episodes = values.get("episodes")
        max_steps = values.get("max_steps")
        discount = values.get("discount")
        learning_rate = values.get("learning_rate")
        frozenlake = values.get("frozenlake")
        policy = values.get("policy")
        epsilon = values.get("epsilon")
        planning = values.get("planning")
        return reps, episodes, max_steps, discount, learning_rate, frozenlake, policy, epsilon, planning
    else:
        raise ValueError("Configuration was not found!")


def compute_expected_return(discount_rate: float, rewards: List[float]) -> float:
    """
    returns discounted sum of rewards of single episode
    """

    T = len(rewards)
    G = [0] * T

    for t in reversed(range(T-1)):
        G[t] = rewards[t + 1] + discount_rate * G[t + 1]

    return G[0]

def get_average_results(results: dict):
    average_results = dict()
    for key in results.keys():
        values = results.get(key)
        if values:
            np_matrix = np.array(values)
            column_averages = np.mean(np_matrix, axis=0)
            average_results[key] = column_averages.tolist()

    return average_results

    # reps = len(args[0])
    # values = len(args[0][0])
    #
    # avg = []
    # for i in range(reps):
    #     avg_row = []
    #     for j in range(values):
    #         total = sum(lst[i][j] for lst in args)
    #         avg_row.append(total / len(args))
    #     avg.append(avg_row)
    # return avg

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