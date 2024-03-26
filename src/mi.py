import gym
import time

from .utils import constants
from .utils.env_translation import *
from .policies.epsilon_greedy_policy import EpsilonGreedyPolicy
from .policies.q_table import QTable

table = QTable()
learning_rate = 0.3
discount = 0.5
epsilon = 0.2

# Create the FrozenLake environment
env = gym.make('FrozenLake3x3', is_slippery=False, render_mode='ansi')  # You can also use 'FrozenLake-v0', render_mode='human', render_mode='ansi'
state, info = env.reset(seed=42)
behavior = EpsilonGreedyPolicy(table, learning_rate, discount, epsilon)
behavior.initialize({s for s in range(9)}, constants.action_set)

# initial_state = 0
# state = initial_state

for i in range(20):

    print("___________________________________________")
    print(f"    ----    ----    RUN {i}    ----    ----    ")
    state, info = env.reset()  # this is to restart
    trail = []

    for _ in range(50):

        print("_____________________________________________")
        print(env.render())

        action_name = behavior.suggest_action(state)
        action = action_name_to_number(action_name)
        print(f'Action: {action_number_to_string(action)}')

        new_state, reward, terminated, truncated, info = env.step(action)
        print(f'new_state: {new_state}, reward: {reward}, terminated: {terminated}, info: {info}')

        trail.append([state, action_name, new_state, reward])
        behavior.update_after_step(state, action_name, new_state, reward)

        state = new_state

        # time.sleep(0.5)

        if terminated or truncated:
            # state, info = env.reset() # this is to restart
            break # this is to terminate

    print(env.render())
    # behavior.update_after_end_of_episode(trail)

print("_____________________________________________")
print(env.render())
behavior.print_policy()
time.sleep(2)
env.close()
