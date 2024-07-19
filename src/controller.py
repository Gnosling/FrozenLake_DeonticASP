import gym

from .utils.utils import *
from .policies.planner_policy import PlannerPolicy


class Controller:

    # TODO: change the frozenLake env to have multiple targets, other elves and change properties?

    def run_experiment(self, config: str):

        print(f"Starting experiment {config} ...")

        # -----------------------------------------------------------------------------
        # Reading params
        # -----------------------------------------------------------------------------
        reps, episodes, max_steps, discount, learning_rate, reversed_q_learning, frozenlake, policy, epsilon, planning_strategy, planning_horizon, norm_set, evaluation_function = read_config_param(config)

        # -----------------------------------------------------------------------------
        # Training
        # -----------------------------------------------------------------------------
        total_returns = []
        total_violations = []
        for rep in range(reps):
            env = gym.make(id=frozenlake.get("name"), traverser_path=frozenlake.get("traverser_path"),
                           is_slippery=frozenlake.get("slippery"),
                           render_mode='ansi')  # render_mode='human', render_mode='ansi'
            env.reset(seed=42)
            behavior, target = build_policy(config)

            return_of_target_per_episode = []
            violations_of_target_per_episode = []
            print(f"Performing repetition {rep+1}", end='\r', flush=True)
            for episode in range(episodes):
                debug_print("_____________________________________________")
                debug_print(f"    ----    ----    Episode {episode}    ----    ----    ")
                state, info = env.reset()  # this is to restart
                trail_of_behavior = []  # list of [state, action_name, new_state, rewards]
                action_name = None

                for step in range(max_steps):
                    debug_print(env.render())

                    behavior.updated_dynamic_env_aspects(env.get_current_traverser_state(), action_name)
                    action_name = behavior.suggest_action(state)
                    debug_print(f'Action: {action_name}')

                    new_state, reward, terminated, truncated, info = env.step(action_name_to_number(action_name))
                    debug_print(f'new_state: {new_state}, reward: {reward}, terminated: {terminated}, info: {info}')

                    if not reversed_q_learning:
                        behavior.update_after_step(state, action_name, new_state, reward)

                    trail_of_behavior.append([state, action_name, new_state, reward])
                    state = new_state

                    if terminated or truncated:
                        debug_print(env.render())
                        break

                if reversed_q_learning:
                    behavior.update_after_end_of_episode(trail_of_behavior)

                if type(behavior) == PlannerPolicy:
                    behavior.reset_after_episode()

                trail_of_target, violations_of_target = test_target(target, env, config)
                expected_return = compute_expected_return(discount, [r for [_,_,_,r] in trail_of_target])
                debug_print(f"Expected return of ep {episode}: {expected_return}")
                debug_print(f"Violations of ep {episode}: {violations_of_target}")
                debug_print("_____________________________________________")
                return_of_target_per_episode.append(expected_return)
                violations_of_target_per_episode.append(violations_of_target)

            debug_print("\n_____________________________________________")
            total_returns.append(return_of_target_per_episode)
            total_violations.append(violations_of_target_per_episode)
            # debug_print(str(total_returns))
            # debug_print(str(total_violations))
            env.close()

        print("Experiment completed!")


        # -----------------------------------------------------------------------------
        # Evaluation
        # -----------------------------------------------------------------------------
        avg_returns = get_average_returns(total_returns)
        debug_print(f"Returns:\n{avg_returns}")
        avg_violations = get_average_violations(total_violations, norm_set)
        debug_print(f"Violations:\n{avg_violations}")


        # -----------------------------------------------------------------------------
        # Storing of results
        # -----------------------------------------------------------------------------
        store_results(config, avg_returns, avg_violations)

        # -----------------------------------------------------------------------------
        # Plotting
        # -----------------------------------------------------------------------------
        # Todo: plot the stored_results!
        plot_experiment(config)

        print("Experiment completed!")

