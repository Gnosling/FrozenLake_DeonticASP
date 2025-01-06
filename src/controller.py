import gym
import time
from tqdm import tqdm

from .utils.utils import *
from .policies.planner_policy import PlannerPolicy

import warnings
warnings.filterwarnings("ignore", message=".*is not within the observation space.*")


class Controller:

    def plot_experiment(self, config: str):
        plot_experiment(config)

    def plot_compare_of_experiments(self, configs: List, show_returns: bool = True, norm_set: int = 0):
        plot_compare_of_experiments(configs, show_returns, norm_set)

    def run_experiment(self, config: str):

        print(f"Starting experiment {config} ...")

        # -----------------------------------------------------------------------------
        # Reading params
        # -----------------------------------------------------------------------------
        repetitions, episodes, max_steps, evaluation_repetitions, learning, frozenlake, planning, deontic, enforcing = read_config_param(config)


        # -----------------------------------------------------------------------------
        # Training
        # -----------------------------------------------------------------------------
        total_returns = []
        total_violations = []
        total_steps = []
        total_slips = []
        total_training_times = []
        total_inference_times = []
        total_state_visits = dict()
        final_target_policies = []
        for rep in range(1, repetitions+1):
            env = gym.make(id=frozenlake.get("name"), traverser_path=frozenlake.get("traverser_path"),
                           is_slippery=frozenlake.get("slippery"),
                           render_mode='ansi')  # render_mode='human', render_mode='ansi'
            env.reset()  # Note: no seed given to use random one
            behavior, target = build_policy(config, env)

            return_of_target_per_episode = []
            violations_of_target_per_episode = []
            steps_of_target_per_episode = []
            slips_of_target_per_episode = []
            training_time_of_behavior_per_episode = []
            inference_time_of_target_per_episode = []
            state_visits_of_target = dict()
            for episode in range(1, episodes+1):
                if not is_debug_mode():
                    pbar = tqdm(total=max_steps, desc=f"Performing repetition {rep}:   {episode}/{episodes}", leave=False)

                debug_print(f"Performing repetition {rep}/{repetitions}: \t {episode}/{episodes}")
                debug_print("_____________________________________________")
                debug_print(f"    ----    ----    Episode {episode}    ----    ----    ")
                state, info = env.reset()  # this is to restart
                layout, width, height = env.get_layout()
                trail_of_behavior = []  # list of [state, proposed_action_name, new_state, rewards]
                # Note: a state is (current_position, traverser_position, list_of_presents)
                last_performed_action = None
                action_name = None
                previous_state = None
                start_time = time.time()

                for step in range(max_steps):
                    debug_print(env.render())

                    behavior.update_dynamic_env_aspects(last_performed_action, action_name, previous_state)
                    action_name = behavior.suggest_action(state, env)
                    debug_print(f'Action: {action_name}')

                    new_state, reward, terminated, _, info = env.step(action_name_to_number(action_name))
                    debug_print(f'new_state: {new_state}, reward: {reward}, terminated: {terminated}, info: {info}')
                    trail_of_behavior.append([state, action_name, new_state, reward])
                    if step == max_steps-1:
                        env.set_terminated(True)

                    if not learning.get("reversed_q_learning"):
                        behavior.update_after_step(state, action_name, new_state, reward, trail_of_behavior, env)

                    previous_state = state
                    last_performed_action = extract_performed_action(state[0], new_state[0], width)
                    state = new_state

                    if not is_debug_mode():
                        pbar.update(1)  # updates progress bar

                    if terminated:
                        debug_print(env.render())
                        break

                if not is_debug_mode():
                    if pbar.n < pbar.total:
                        pbar.update(pbar.total - pbar.n -1)

                if learning.get("reversed_q_learning"):
                    behavior.update_after_end_of_episode(trail_of_behavior, env)

                if isinstance(behavior, PlannerPolicy):
                    behavior.reset_after_episode()

                end_time = time.time()
                training_time = end_time-start_time
                trail_of_target, violations_of_target, slips_of_target, inference_time, state_visits = test_target(target, env, config)
                expected_return = compute_expected_return(learning.get("discount"), [r for [_,_,_,r] in trail_of_target])
                debug_print(f"Expected return of ep {episode}: {expected_return}")
                debug_print(f"Violations of ep {episode}: {violations_of_target}")
                debug_print("_____________________________________________")
                return_of_target_per_episode.append(expected_return)
                violations_of_target_per_episode.append(violations_of_target)
                steps_of_target_per_episode.append(len(trail_of_target))
                slips_of_target_per_episode.append(slips_of_target)
                training_time_of_behavior_per_episode.append(training_time)
                inference_time_of_target_per_episode.append(inference_time)
                state_visits_of_target = state_visits
                if not is_debug_mode():
                    pbar.close()

            debug_print(f"Finished repetition {rep}")
            debug_print("\n_____________________________________________")
            total_returns.append(return_of_target_per_episode)
            total_violations.append(violations_of_target_per_episode)
            total_steps.append(steps_of_target_per_episode)
            total_slips.append(slips_of_target_per_episode)
            total_training_times.append(training_time_of_behavior_per_episode)
            total_inference_times.append(inference_time_of_target_per_episode)
            final_target_policies.append(target)
            total_state_visits = update_state_visits(total_state_visits, state_visits_of_target)  # Note: takes only last values
            env.close()


        # -----------------------------------------------------------------------------
        # Evaluation: Training + Final + Enforced
        # -----------------------------------------------------------------------------
        training_returns_avg, _ = get_average_numbers(total_returns)
        training_returns_stderr = get_standard_error(total_returns)
        debug_print(f"Returns:\n{training_returns_avg}")
        training_steps_avg, training_steps_stddev = get_average_numbers(total_steps)
        training_slips_avg, training_slips_stddev = get_average_numbers(total_slips)
        training_fitting_times_avg, training_fitting_times_stddev = get_average_numbers(total_training_times)
        training_inference_times_avg, training_inference_times_stddev = get_average_numbers(total_inference_times)
        training_state_visits = get_average_state_visits(total_state_visits, repetitions)
        training_violations_avg = training_violations_stddev = None
        if deontic:
            training_violations_avg, training_violations_stddev = get_average_violations(total_violations, deontic.get("norm_set"))
            debug_print(f"Violations:\n{training_violations_avg}")


        final_returns = []
        final_steps = []
        final_slips = []
        final_inference_times = []
        final_state_visits = dict()
        final_violations = None
        for target in final_target_policies:
            target.set_enforcing(None)
            for i in range(evaluation_repetitions):
                trail_of_target, violations_of_target, slips_of_target, inference_time, state_visits = test_target(target, env, config)
                expected_return = compute_expected_return(learning.get("discount"), [r for [_, _, _, r] in trail_of_target])
                final_returns.append(expected_return)
                final_violations = append_violations(final_violations, violations_of_target)
                final_steps.append(len(trail_of_target))
                final_slips.append(slips_of_target)
                final_inference_times.append(inference_time)
                final_state_visits = update_state_visits(final_state_visits, state_visits)
        final_state_visits = get_average_state_visits(final_state_visits, repetitions*evaluation_repetitions)

        final_enforced_returns = []
        final_enforced_steps = []
        final_enforced_slips = []
        final_enforced_inference_times = []
        final_enforced_state_visits = dict()
        final_enforced_violations = None
        if enforcing and enforcing.get("phase") == "after_training":
            for target in final_target_policies:
                target.set_enforcing(enforcing)
                if enforcing.get("enforcing_horizon") and "reward_shaping" in enforcing.get("strategy"):
                    # Note: if rewards shaping is used, then the targets are 'shaped' before the actual evaluation
                    for i in range(enforcing.get("enforcing_horizon")[0]):
                        test_target(target, env, config)

                for i in range(evaluation_repetitions):
                    trail_of_target, violations_of_target, slips_of_target, inference_time, state_visits = test_target(target, env, config)
                    expected_return = compute_expected_return(learning.get("discount"),[r for [_, _, _, r] in trail_of_target])
                    final_enforced_returns.append(expected_return)
                    final_enforced_violations = append_violations(final_enforced_violations, violations_of_target)
                    final_enforced_steps.append(len(trail_of_target))
                    final_enforced_slips.append(slips_of_target)
                    final_enforced_inference_times.append(inference_time)
                    final_enforced_state_visits = update_state_visits(final_enforced_state_visits, state_visits)
            final_enforced_state_visits = get_average_state_visits(final_enforced_state_visits, repetitions * evaluation_repetitions)


        # -----------------------------------------------------------------------------
        # Storing of results
        # -----------------------------------------------------------------------------
        store_results(config,
                      training_returns_avg, training_returns_stderr, training_steps_avg, training_steps_stddev, training_slips_avg, training_slips_stddev, training_violations_avg, training_violations_stddev, training_fitting_times_avg, training_fitting_times_stddev, training_inference_times_avg, training_inference_times_stddev, training_state_visits,
                      final_returns, final_steps, final_slips, final_violations, final_inference_times, final_state_visits,
                      final_enforced_returns, final_enforced_steps, final_enforced_slips, final_enforced_violations, final_enforced_inference_times, final_enforced_state_visits)


        # -----------------------------------------------------------------------------
        # Plotting
        # -----------------------------------------------------------------------------
        plot_experiment(config)

        print(f"Completed experiment: {config}")

