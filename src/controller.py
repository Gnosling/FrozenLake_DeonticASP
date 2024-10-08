import gym

from .utils.utils import *
from .policies.planner_policy import PlannerPolicy


class Controller:

    def plot_experiment(self, config: str):
        plot_experiment(config)

    def run_experiment(self, config: str):

        print(f"Starting experiment {config} ...")

        # -----------------------------------------------------------------------------
        # Reading params
        # -----------------------------------------------------------------------------
        repetitions, episodes, max_steps, learning, frozenlake, planning, deontic = read_config_param(config)

        # -----------------------------------------------------------------------------
        # Training
        # -----------------------------------------------------------------------------
        total_returns = []
        total_violations = []
        total_steps = []
        total_slips = []
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
            print(f"Performing repetition {rep+1} ...", end='\r')
            for episode in range(episodes):
                debug_print("_____________________________________________")
                debug_print(f"    ----    ----    Episode {episode}    ----    ----    ")
                state, info = env.reset()  # this is to restart
                trail_of_behavior = []  # list of [state, action_name, new_state, rewards]
                action_name = None

                for step in range(max_steps):
                    debug_print(env.render())

                    behavior.updated_dynamic_env_aspects(env.get_current_traverser_state(), action_name, env.get_states_with_presents())
                    action_name = behavior.suggest_action(state)
                    debug_print(f'Action: {action_name}')

                    new_state, reward, terminated, truncated, info = env.step(action_name_to_number(action_name))
                    debug_print(f'new_state: {new_state}, reward: {reward}, terminated: {terminated}, info: {info}')

                    if not learning.get("reversed_q_learning"):
                        behavior.update_after_step(state, action_name, new_state, reward)

                    trail_of_behavior.append([state, action_name, new_state, reward])
                    state = new_state

                    if terminated or truncated:
                        debug_print(env.render())
                        break

                if learning.get("reversed_q_learning"):
                    behavior.update_after_end_of_episode(trail_of_behavior)

                if type(behavior) == PlannerPolicy:
                    behavior.reset_after_episode()

                # TODO: analyse values of target learning might be too weak (works though but values are low), eithe increase episodes or learning rate
                trail_of_target, violations_of_target, slips_of_target = test_target(target, env, config)
                expected_return = compute_expected_return(learning.get("discount"), [r for [_,_,_,r] in trail_of_target])
                debug_print(f"Expected return of ep {episode}: {expected_return}")
                debug_print(f"Violations of ep {episode}: {violations_of_target}")
                debug_print("_____________________________________________")
                return_of_target_per_episode.append(expected_return)
                violations_of_target_per_episode.append(violations_of_target)
                steps_of_target_per_episode.append(len(trail_of_target))
                slips_of_target_per_episode.append(slips_of_target)

            debug_print(f"Finished repetition {rep + 1}")
            debug_print("\n_____________________________________________")
            total_returns.append(return_of_target_per_episode)
            total_violations.append(violations_of_target_per_episode)
            total_steps.append(steps_of_target_per_episode)
            total_slips.append(slips_of_target_per_episode)
            env.close()

        # -----------------------------------------------------------------------------
        # Evaluation
        # -----------------------------------------------------------------------------
        avg_returns = get_average_numbers(total_returns)
        debug_print(f"Returns:\n{avg_returns}")
        avg_steps = get_average_numbers(total_steps)
        debug_print(f"Steps:\n{avg_steps}")
        avg_slips = get_average_numbers(total_slips)
        avg_violations = None
        if deontic.get("norm_set") is not None:
            avg_violations = get_average_violations(total_violations, deontic.get("norm_set"))
            debug_print(f"Violations:\n{avg_violations}")

        # -----------------------------------------------------------------------------
        # Storing of results
        # -----------------------------------------------------------------------------
        store_results(config, avg_returns, avg_steps, avg_slips, avg_violations)

        # -----------------------------------------------------------------------------
        # Plotting
        # -----------------------------------------------------------------------------
        plot_experiment(config)

        print(f"Completed experiment: {config}")

