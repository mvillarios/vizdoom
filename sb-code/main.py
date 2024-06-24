import os
import cv2
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import callbacks

import vizdoom.gymnasium_wrapper
from utils import plot_rewards

ENV = "VizdoomDefendCenter-v0"
RESOLUTION = (60, 45)

# Params
TRAINING_TIMESTEPS = int(6e5)  # 600k
N_STEPS = 4096
N_ENVS = 1
FRAME_SKIP = 4
BATCH_SIZE = 64

model = "dqn"
num = "3"
map = "defend-center"

LOG_DIR = f"saves/{map}/{model}-{num}"

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

class EpsilonLogger(callbacks.BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(EpsilonLogger, self).__init__(verbose)
        self.epsilons = []
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        self.epsilons.append(self.model.exploration_rate)
        return True

    def _on_training_end(self) -> None:
        np.save(os.path.join(self.log_dir, 'epsilons.npy'), self.epsilons)

if __name__ == "__main__":

    def wrap_env(env):
        env = ObservationWrapper(env)
        env = Monitor(env, f"{LOG_DIR}/")
        return env

    env_kwargs = {
        "frame_skip": FRAME_SKIP,
    }

    train_env = make_vec_env(ENV, n_envs=N_ENVS, wrapper_class=wrap_env, env_kwargs=env_kwargs)
    train_env = VecTransposeImage(train_env)

    eval_env = make_vec_env(ENV, n_envs=N_ENVS, wrapper_class=wrap_env, env_kwargs=env_kwargs)
    eval_env = VecTransposeImage(eval_env)

    evaluation_callback = callbacks.EvalCallback(
        eval_env,
        best_model_save_path=f"{LOG_DIR}/models",
        log_path=f"{LOG_DIR}/eval",
        eval_freq=5000,
        render=False,
        n_eval_episodes=10,
    )

    if model == "dqn":
        agent = DQN(
            "CnnPolicy", 
            train_env,
            batch_size=BATCH_SIZE,
            exploration_final_eps=1e-2,
            learning_starts=1e5,
            buffer_size=10000,
            verbose=1,
            device='cuda'
        )

        epsilon_logger = EpsilonLogger(LOG_DIR)
        
        agent.learn(total_timesteps=TRAINING_TIMESTEPS, tb_log_name="dqn", callback=[evaluation_callback, epsilon_logger])
        agent.save(f"{LOG_DIR}/saves/dqn_vizdoom")

    else:
        agent = PPO(
            "CnnPolicy", 
            train_env,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            verbose=1,
            device='cuda'
        )

        agent.learn(total_timesteps=TRAINING_TIMESTEPS, tb_log_name="ppo", callback=evaluation_callback)
        agent.save(f"{LOG_DIR}/saves/ppo_vizdoom")


    train_env.close()
    eval_env.close()

    plot_rewards(LOG_DIR)
