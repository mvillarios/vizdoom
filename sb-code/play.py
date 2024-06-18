import cv2
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import vizdoom.gymnasium_wrapper

ENV = "VizdoomDefendCenter-v0"
RESOLUTION = (60, 45)
MODEL_PATH = "dqn_vizdoom.zip"  # Path to the saved model

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

def wrap_env(env):
    env = ObservationWrapper(env)
    return env

if __name__ == "__main__":
    env = make_vec_env(ENV, n_envs=1, wrapper_class=wrap_env)
    env.envs[0].env.frame_skip = 1  # Set frame skip for the environment
    env.envs[0].env.render_mode = "human" # Set render mode human

    model = DQN.load(MODEL_PATH)

    obs = env.reset()
    for _ in range(1000):  # Play for 1000 steps
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

        if dones:
            obs = env.reset()

    env.close()
