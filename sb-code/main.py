import os
import cv2
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import callbacks

import vizdoom.gymnasium_wrapper
from utils import plot_rewards
from params import DQN_PARAMS, PPO_PARAMS

ENV_LIST = [
    #"VizdoomDefendCenter-v0", 
    #"VizdoomDefendLine-v0",
    "VizdoomCorridor-v0",
    #"VizdoomMyWayHome-v0",
    #"VizdoomHealthGathering-v0"
    # "VizdoomPredictPosition-v0",
    # "VizdoomTakeCover-v0",
    # "VizdoomDeathmatch-v0",
]

MAP_LIST = [
    #"defend-center",
    #"defend-line",
    "corridor",
    #"my-way-home",
    #"health-gathering",
    # "predict-position",
    # "take-cover"
    # "deathmatch",
]

MODEL_LIST = [
    "dqn",
    "ppo"
]

RESOLUTION = (60, 45)
TRAINING_TIMESTEPS = int(6e6)  # 600k
N_ENVS = 1
FRAME_SKIP = 4

num = 2

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

class EnvWrapper:
    def __init__(self, log_dir):
        self.log_dir = log_dir

    def __call__(self, env):
        env = ObservationWrapper(env)
        env = Monitor(env, self.log_dir)
        return env

if __name__ == "__main__":
    env_kwargs = {
        "frame_skip": FRAME_SKIP,
    }

    start_index = 0  # comienza desde el tercer mapa a entrenar

    for model in MODEL_LIST:
        for map_name, env_name in zip(MAP_LIST[start_index:], ENV_LIST[start_index:]):
            LOG_DIR = f"saves-tunning/{map_name}/{model}-{num}"
            env_wrapper = EnvWrapper(log_dir=f"{LOG_DIR}/")

            train_env = make_vec_env(env_name, n_envs=N_ENVS, wrapper_class=env_wrapper, env_kwargs=env_kwargs)
            train_env = VecTransposeImage(train_env)

            eval_env = make_vec_env(env_name, n_envs=N_ENVS, wrapper_class=env_wrapper, env_kwargs=env_kwargs)
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
                params = DQN_PARAMS.get(env_name, {})
                agent = DQN(
                    "CnnPolicy",
                    train_env,
                    batch_size=params.get("batch_size", 64),
                    learning_rate=params.get("learning_rate", 0.0001),
                    buffer_size=params.get("buffer_size", 50000),
                    gamma=params.get("gamma", 0.99),
                    exploration_fraction=params.get("exploration_fraction", 0.1),
                    exploration_final_eps=params.get("exploration_final_eps", 0.02),
                    learning_starts=params.get("learning_starts", 10000),
                    verbose=1,
                    device='cuda'
                )

                epsilon_logger = EpsilonLogger(LOG_DIR)
                
                agent.learn(total_timesteps=TRAINING_TIMESTEPS, tb_log_name="dqn", callback=[evaluation_callback, epsilon_logger])
                agent.save(f"{LOG_DIR}/saves/dqn_vizdoom")

            else:
                params = PPO_PARAMS.get(env_name, {})
                agent = PPO(
                    "CnnPolicy",
                    train_env,
                    n_steps=params.get("n_steps", 2048),
                    batch_size=params.get("batch_size", 64),
                    learning_rate=params.get("learning_rate", 3e-4),
                    gamma=params.get("gamma", 0.99),
                    gae_lambda=params.get("gae_lambda", 0.95),
                    verbose=1,
                    device='cuda'
                )

                agent.learn(total_timesteps=TRAINING_TIMESTEPS, tb_log_name="ppo", callback=evaluation_callback)
                agent.save(f"{LOG_DIR}/saves/ppo_vizdoom")

            train_env.close()
            eval_env.close()

            plot_rewards(LOG_DIR, model)
