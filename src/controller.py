import gym
import time

from .utils.utils import *
from .policies.planner_policy import PlannerPolicy

import warnings
warnings.filterwarnings("ignore", message=".*is not within the observation space.*")


class Controller:

    def plot_experiment(self, config: str):
        plot_experiment(config)

    def run_experiment(self, config: str):

        print(f"Starting experiment {config} ...")

        # -----------------------------------------------------------------------------
        # Reading params
        # -----------------------------------------------------------------------------
        repetitions, episodes, max_steps, learning, frozenlake, planning, deontic, enforcing = read_config_param(config)


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
        for rep in range(repetitions):
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
            print(f"Performing repetition {rep+1} ...", end='\r')
            for episode in range(episodes):
                debug_print("_____________________________________________")
                debug_print(f"    ----    ----    Episode {episode+1}    ----    ----    ")
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

                    if terminated:
                        debug_print(env.render())
                        break

                if learning.get("reversed_q_learning"):
                    behavior.update_after_end_of_episode(trail_of_behavior, env)

                if type(behavior) == PlannerPolicy:
                    behavior.reset_after_episode()

                end_time = time.time()
                training_time = end_time-start_time
                trail_of_target, violations_of_target, slips_of_target, inference_time, state_visits = test_target(target, env, config, False)
                expected_return = compute_expected_return(learning.get("discount"), [r for [_,_,_,r] in trail_of_target])
                debug_print(f"Expected return of ep {episode+1}: {expected_return}")
                debug_print(f"Violations of ep {episode+1}: {violations_of_target}")
                debug_print("_____________________________________________")
                return_of_target_per_episode.append(expected_return)
                violations_of_target_per_episode.append(violations_of_target)
                steps_of_target_per_episode.append(len(trail_of_target))
                slips_of_target_per_episode.append(slips_of_target)
                training_time_of_behavior_per_episode.append(training_time)
                inference_time_of_target_per_episode.append(inference_time)
                state_visits_of_target = state_visits

            debug_print(f"Finished repetition {rep + 1}")
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
        # Evaluation
        # -----------------------------------------------------------------------------
        training_returns_avg, training_returns_stddev = get_average_numbers(total_returns)
        debug_print(f"Returns:\n{training_returns_avg}")
        training_steps_avg, training_steps_stddev = get_average_numbers(total_steps)
        debug_print(f"Steps:\n{training_steps_avg}")
        training_slips_avg, training_slips_stddev = get_average_numbers(total_slips)
        training_fitting_times_avg, training_fitting_times_stddev = get_average_numbers(total_training_times)
        training_inference_times_avg, training_inference_times_stddev = get_average_numbers(total_inference_times)
        training_state_visits = get_average_state_visits(total_state_visits, repetitions)
        training_violations_avg = training_violations_stddev = None
        if deontic:
            training_violations_avg, training_violations_stddev = get_average_violations(total_violations, deontic.get("norm_set"))
            debug_print(f"Violations:\n{training_violations_avg}")

        final_returns = [] # TODO: these list just store any value, works because each target has the same number of evalutions (10 i guess)
        final_violations = None  # TODO: this is then a dict with lists as values forech norm
        final_steps = []
        final_slips = []
        final_inference_times = []
        final_state_visits = dict()  # TODO: make another plot for this should be sum as well
        final_enforced_returns = None
        final_enforced_steps = None
        final_enforced_slips = None
        final_enforced_inference_times = None
        final_enforced_state_visits = None # TODO: update enforcing values to also have be in std of last episode
        final_enforced_violations = None
        # TODO: should enforcements be applied multiple times and then averaged in both dimensions? -> change this to general evaluation!

        for target in final_target_policies:
            target.set_enforcing(None) # TODO: maybe do this in here
            for i in range(10):
                trail_of_target, violations_of_target, slips_of_target, inference_time, state_visits = test_target(target, env, config, True)
                expected_return = compute_expected_return(learning.get("discount"), [r for [_, _, _, r] in trail_of_target])
                final_returns.append(expected_return)
                final_violations = append_violations(final_violations, violations_of_target)
                final_steps.append(len(trail_of_target))
                final_slips.append(slips_of_target)
                final_inference_times.append(inference_time)
                final_state_visits = update_state_visits(final_state_visits, state_visits)


        if enforcing and enforcing.get("phase") == "after_training":
            return_of_targets = []
            violations_of_targets = []
            steps_of_targets = []
            slips_of_targets = []
            enforced_inference_time_of_target = []
            total_enforced_state_visits = dict()
            for target in final_target_policies:
                trail_of_target, violations_of_target, slips_of_target, inference_time, state_visits = test_target(target, env, config, True)
                expected_return = compute_expected_return(learning.get("discount"), [r for [_, _, _, r] in trail_of_target])
                return_of_targets.append(expected_return)
                violations_of_targets.append(violations_of_target)
                steps_of_targets.append(len(trail_of_target))
                slips_of_targets.append(slips_of_target)
                enforced_inference_time_of_target.append(inference_time)
                total_enforced_state_visits = update_state_visits(total_enforced_state_visits, state_visits)
            final_enforced_returns = sum(return_of_targets) / len(return_of_targets)
            final_enforced_violations = {norm: sum(violation[norm] for violation in violations_of_targets) / len(violations_of_targets) for norm in violations_of_targets[0]}
            final_enforced_steps = sum(steps_of_targets) / len(steps_of_targets)
            final_enforced_slips = sum(slips_of_targets) / len(slips_of_targets)
            final_enforced_inference_times = sum(enforced_inference_time_of_target) / len(enforced_inference_time_of_target)
            final_enforced_state_visits = get_average_state_visits(total_enforced_state_visits, repetitions)


        # -----------------------------------------------------------------------------
        # Storing of results
        # -----------------------------------------------------------------------------
        store_results(config,
                      training_returns_avg, training_returns_stddev, training_steps_avg, training_steps_stddev, training_slips_avg, training_slips_stddev, training_violations_avg, training_violations_stddev, training_fitting_times_avg, training_fitting_times_stddev, training_inference_times_avg, training_inference_times_stddev, training_state_visits,
                      final_returns, final_steps, final_slips, final_violations, final_inference_times, final_state_visits,
                      final_enforced_returns, final_enforced_steps, final_enforced_slips, final_enforced_violations, final_enforced_inference_times, final_enforced_state_visits)


        # -----------------------------------------------------------------------------
        # Plotting
        # -----------------------------------------------------------------------------
        plot_experiment(config)

        print(f"Completed experiment: {config}")

