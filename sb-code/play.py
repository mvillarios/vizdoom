import os
import cv2
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import callbacks
from gymnasium import RewardWrapper

import vizdoom.gymnasium_wrapper

ENV = "VizdoomCorridor-v0"
RESOLUTION = (60, 45)

model = "ppo"
num = "stop-3-2"
map = "corridor"
#MODEL_PATH = f"trains/{map}/{model}-{num}/saves/{model}_vizdoom"
MODEL_PATH = f"trains/{map}/{model}-{num}/models/best_model"

class RewardShapingWrapper(RewardWrapper):
    def __init__(self, env, damage_reward=100, hit_taken_penalty=-5, ammo_penalty=-1):
        super(RewardShapingWrapper, self).__init__(env)
        self.damage_reward = damage_reward
        self.hit_taken_penalty = hit_taken_penalty
        self.ammo_penalty = ammo_penalty
        self.previous_damage_taken = 0
        self.previous_hitcount = 0
        self.previous_ammo = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        game_variables = self.env.unwrapped.game.get_state().game_variables

        self.previous_damage_taken = game_variables[1]  # DAMAGE_TAKEN
        self.previous_hitcount = game_variables[2]  # HITCOUNT
        self.previous_ammo = game_variables[3]  # SELECTED_WEAPON_AMMO

        #print(f"Game variables: {game_variables}")
        #print(f"Reset: Damage taken: {self.previous_damage_taken}, Hitcount: {self.previous_hitcount}, Ammo: {self.previous_ammo}")

        return obs, info

    def reward(self, reward):
        #print(f"Reward original: {reward}")
        custom_reward = reward
        game_state = self.env.unwrapped.game.get_state()

        if game_state:
            game_variables = game_state.game_variables
            current_damage_taken = game_variables[1]  # DAMAGE_TAKEN
            current_hitcount = game_variables[2]  # HITCOUNT
            current_ammo = game_variables[3]  # SELECTED_WEAPON_AMMO

            # Penalización por recibir daño
            # if current_damage_taken > self.previous_damage_taken:
            #     damage_taken_delta = current_damage_taken - self.previous_damage_taken
            #     penalty = damage_taken_delta * self.hit_taken_penalty
            #     custom_reward += penalty
            #     #print(f"Penalización por recibir daño: {penalty}, Daño recibido: {damage_taken_delta}, Recompensa actual: {custom_reward}")
            # self.previous_damage_taken = current_damage_taken

            # Recompensa por hacer daño (hitcount)
            if current_hitcount > self.previous_hitcount:
                hitcount_delta = current_hitcount - self.previous_hitcount
                reward_gain = hitcount_delta * self.damage_reward
                custom_reward += reward_gain
                #print(f"Recompensa por hacer daño: {reward_gain}, Hits hechos: {hitcount_delta}, Recompensa actual: {custom_reward}")
            self.previous_hitcount = current_hitcount

            # Penalización por gastar balas
            if current_ammo < self.previous_ammo:
                ammo_delta = self.previous_ammo - current_ammo
                penalty = ammo_delta * self.ammo_penalty
                custom_reward += penalty
                #print(f"Penalización por gastar balas: {penalty}, Balas gastadas: {ammo_delta}, Recompensa actual: {custom_reward}")
            self.previous_ammo = current_ammo

        #print(f"Reward custom: {custom_reward}")
        return custom_reward

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
        env = RewardShapingWrapper(env)
        #env = gym.wrappers.TransformReward(env, lambda r: r * 0.001)
        return env

    env = make_vec_env(ENV, wrapper_class=wrap_env, env_kwargs={"frame_skip": 4, "render_mode": "human"})

    # Load the trained agent
    if model == "dqn":
        agent = DQN.load(MODEL_PATH, print_system_info=True)
    else:
        agent = PPO.load(MODEL_PATH, print_system_info=True)

    os.system("clear")

    # print env and num
    print(f"Env: {ENV}")
    print(f"Model: {model}")
    print(f"Num: {num}")

    episode = 0
    for _ in range(50):
        obs = env.reset()
        done = False
        total_reward = 0
        episode+=1
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            #print(f"Reward: {reward}")
            total_reward += reward
            env.render()

        print(f"Episode: {episode} Reward: {total_reward}")

    env.close()


