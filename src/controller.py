from typing import List

import gym
import time
import csv

from .utils import constants
from .utils.utils import *
from .policies.epsilon_greedy_policy import EpsilonGreedyPolicy
from .policies.planner_policy import PlannerPolicy
from .policies.q_table import QTable
from .policies.policy import Policy
from .service import *

class Controller:

    def __init__(self):
        self.discount_rate = 1.0

    # TODO: have a service.class for this

    # TODO: change the frozenLake env to have multiple targets, other elves and change properties?



    def run_experiment(self, config: str):

        print(f"Starting experiment {config} ...")

        # -----------------------------------------------------------------------------
        # Reading params
        # -----------------------------------------------------------------------------
        reps, episodes, max_steps, discount, learning_rate, frozenlake, policy, epsilon, planning_strategy = read_config_param(config)

        # -----------------------------------------------------------------------------
        # Initializations
        # -----------------------------------------------------------------------------

        env = gym.make(id=frozenlake.get("name"), traverser_path=frozenlake.get("traverser_path"), is_slippery=frozenlake.get("slippery"), render_mode='ansi')  # render_mode='human', render_mode='ansi'
        env.reset(seed=42)
        behavior = build_policy(config)
        behavior.initialize({s for s in range(frozenlake.get("tiles"))}, constants.action_set)

        # -----------------------------------------------------------------------------
        # Training
        # -----------------------------------------------------------------------------

        total_results = {"avg_behavior_return": [], "avg_target_return": []}
        for rep in range(reps):
            return_of_behavior = []
            return_of_target = []
            print(f"Performing repetition {rep+1}", end='\r', flush=True)
            for episode in range(episodes):
                debug_print("_____________________________________________")
                debug_print(f"    ----    ----    Episode {episode}    ----    ----    ")
                state, info = env.reset()  # this is to restart
                trail = []  # list of [state, action_name, new_state, rewards]

                for step in range(max_steps):
                    # debug_print("_____________________________________________")
                    debug_print(env.render())

                    action_name = behavior.suggest_action(state)
                    action = action_name_to_number(action_name)
                    debug_print(f'Action: {action_number_to_string(action)}')

                    new_state, reward, terminated, truncated, info = env.step(action)
                    # debug_print(f'new_state: {new_state}, reward: {reward}, terminated: {terminated}, info: {info}')

                    trail.append([state, action_name, new_state, reward])
                    behavior.update_after_step(state, action_name, new_state, reward)

                    state = new_state

                    # time.sleep(0.5)

                    if terminated or truncated:
                        debug_print(env.render())
                        env.reset() # this is to restart
                        break  # this is to terminate

                # behavior.update_after_end_of_episode(trail)
                expected_return = compute_expected_return(discount, [r for [_,_,_,r] in trail])
                return_of_behavior.append(expected_return)
                return_of_target.append(expected_return)
                debug_print(f"Expected return of ep {episode}: {expected_return}")
                debug_print("_____________________________________________")

            debug_print("_____________________________________________")
            # debug_print(env.render())
            debug_print(behavior.get_printed_policy())
            total_results.get("avg_behavior_return").append(return_of_behavior)
            total_results.get("avg_target_return").append(return_of_target)
            # time.sleep(1)
            env.close()

        print("Experiment completed!            ", end='\n', flush=True)
        # print("\n", flush=True)
        # print("\nTask completed!")
        # -----------------------------------------------------------------------------
        # Evaluation
        # -----------------------------------------------------------------------------
        avg_results = get_average_results(total_results)
        debug_print(f"Results:\n{avg_results}")

        # -----------------------------------------------------------------------------
        # Storing of results
        # -----------------------------------------------------------------------------
        store_results(config, avg_results)

        # print("Run finished!")

