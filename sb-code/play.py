import cv2
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import callbacks

import vizdoom.gymnasium_wrapper

ENV = "VizdoomDefendCenter-v0"
RESOLUTION = (60, 45)

model = "ppo"
num = "1"
map = "defend-center"
MODEL_PATH = f"saves-tunning/{map}/{model}-{num}/saves/{model}_vizdoom"

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
        return env

    env = make_vec_env(ENV, wrapper_class=wrap_env, env_kwargs={"frame_skip": 1, "render_mode": "human"})

    # Load the trained agent
    if model == "dqn":
        agent = DQN.load(MODEL_PATH, print_system_info=True)
    else:
        agent = PPO.load(MODEL_PATH, print_system_info=True)

    #Esperar a que el usuario presione una tecla para comenzar
    input("Press Enter to start...")

    episode = 0
    for _ in range(50):
        obs = env.reset()
        done = False
        total_reward = 0
        episode+=1
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            env.render()

        print(f"Episode: {episode} Reward: {total_reward}")

    env.close()


