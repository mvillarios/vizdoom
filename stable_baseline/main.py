import cv2
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import pandas as pd

import vizdoom.gymnasium_wrapper

ENV = "VizdoomDefendCenter-v0"
RESOLUTION = (60, 45)

# Params
TRAINING_TIMESTEPS = int(6e6)  # Aproximadamente 6000 episodios
N_STEPS = 128
N_ENVS = 1
FRAME_SKIP = 4

LOG_DIR = "logs/"

class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, shape=RESOLUTION):
        super().__init__(env)
        self.image_shape = shape
        self.image_shape_reverse = shape[::-1]

        # Create new observation space with the new shape
        num_channels = env.observation_space["screen"].shape[-1]
        new_shape = (shape[0], shape[1], num_channels)
        self.observation_space = gym.spaces.Box(
            0, 255, shape=new_shape, dtype=np.uint8
        )

    def observation(self, observation):
        observation = cv2.resize(observation["screen"], self.image_shape_reverse)
        return observation


if __name__ == "__main__":

    def wrap_env(env):
        env = ObservationWrapper(env)
        env = Monitor(env, LOG_DIR)
        return env

    env = make_vec_env(ENV, n_envs=N_ENVS, wrapper_class=wrap_env)
    env.envs[0].env.frame_skip = FRAME_SKIP  # Set frame skip for the environment

    model = PPO(
        "CnnPolicy", 
        env, 
        learning_rate=1e-2, 
        #buffer_size=10000,
        verbose=1,
        device='cuda'
    )
    model.learn(total_timesteps=TRAINING_TIMESTEPS)
    model.save("ppo_vizdoom")

    env.close()

    # Read the CSV file while handling malformed rows
    data = []
    with open(LOG_DIR + "monitor.csv", 'r') as file:
        for line in file.readlines()[2:]:  # Skip the first two lines
            try:
                r, l, t = map(float, line.split(','))
                data.append((r, l, t))
            except ValueError:
                continue  # Skip malformed rows

    # Convert to DataFrame
    results_df = pd.DataFrame(data, columns=['r', 'l', 't'])

    # Smooth the rewards
    results_df['r'] = results_df['r'].rolling(window=10).mean()

    # Plot the rewards
    rewards = results_df['r']

    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.savefig("reward_per_episode.png")
