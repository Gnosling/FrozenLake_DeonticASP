import gym
import time

from src.utils.EnvTranslation import action_number_to_string

# Create the FrozenLake environment
env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human')  # You can also use 'FrozenLake-v0', render_mode='human', render_mode='ansi'
observation, info = env.reset(seed=42)

for _ in range(10):

    print(env.render())
    action = env.action_space.sample() # randomly picks an action
    print(f'Action: {action_number_to_string(action)}')
    observation, reward, terminated, truncated, info = env.step(action)

    print(f'obs:{observation}, reward: {reward}, terminated: {terminated}, info: {info}')
    time.sleep(0.5)

    if terminated or truncated:
        # observation, info = env.reset() # this is to restart
        break

time.sleep(2)
env.close()
