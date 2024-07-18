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

class Controller:

    # TODO: have a service.class for this
    # TODO: change the frozenLake env to have multiple targets, other elves and change properties?

    def run_experiment(self, config: str):

        print(f"Starting experiment {config} ...")

        # -----------------------------------------------------------------------------
        # Reading params
        # -----------------------------------------------------------------------------
        reps, episodes, max_steps, discount, learning_rate, reversed_q_learning, frozenlake, policy, epsilon, planning_strategy, planning_horizon, norm_set, evaluation_function = read_config_param(config)

        # -----------------------------------------------------------------------------
        # Initializations
        # -----------------------------------------------------------------------------

        env = gym.make(id=frozenlake.get("name"), traverser_path=frozenlake.get("traverser_path"), is_slippery=frozenlake.get("slippery"), render_mode='ansi')  # render_mode='human', render_mode='ansi'
        env.reset(seed=42)
        behavior, target = build_policy(config)

        # -----------------------------------------------------------------------------
        # Training
        # -----------------------------------------------------------------------------

        total_results = []
        for rep in range(reps):
            return_of_target = []
            print(f"Performing repetition {rep+1}", end='\r', flush=True)
            for episode in range(episodes):
                debug_print("_____________________________________________")
                debug_print(f"    ----    ----    Episode {episode}    ----    ----    ")
                state, info = env.reset()  # this is to restart
                trail_of_behavior = []  # list of [state, action_name, new_state, rewards]
                action_name = None

                for step in range(max_steps):
                    debug_print("_____________________________________________")
                    debug_print(env.render())

                    behavior.updated_dynamic_env_aspects(env.get_current_traverser_state(), action_name)
                    action_name = behavior.suggest_action(state)
                    debug_print(f'Action: {action_name}')

                    new_state, reward, terminated, truncated, info = env.step(action_name_to_number(action_name))
                    debug_print(f'new_state: {new_state}, reward: {reward}, terminated: {terminated}, info: {info}')

                    trail_of_behavior.append([state, action_name, new_state, reward])
                    if not reversed_q_learning:
                        behavior.update_after_step(state, action_name, new_state, reward)

                    state = new_state

                    if terminated or truncated:
                        debug_print(env.render())
                        break  # this is to terminate

                if reversed_q_learning:
                    behavior.update_after_end_of_episode(trail_of_behavior)

                if type(behavior) == PlannerPolicy:
                    behavior.reset_after_episode()

                trail_of_target = generate_trail_of_target(target, env, max_steps)
                expected_return = compute_expected_return(discount, [r for [_,_,_,r] in trail_of_target])
                return_of_target.append(expected_return)
                # TODO: add violations of target, as dict() with countering?
                debug_print(f"Expected return of ep {episode}: {expected_return}")
                debug_print("_____________________________________________")

            debug_print("_____________________________________________")
            # debug_print(env.render())
            total_results.append(return_of_target)
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

