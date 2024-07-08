import cv2
import numpy as np
import gymnasium as gym
import optuna
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import callbacks
from stable_baselines3.common.evaluation import evaluate_policy

import vizdoom.gymnasium_wrapper

ENV = "VizdoomDefendCenter-v0"
RESOLUTION = (60, 45)
TRAINING_TIMESTEPS = int(6e4)
N_ENVS = 1
FRAME_SKIP = 4

model = "dqn-tunning"
num = "1"
map = "defend-center"

LOG_DIR = f"saves/{map}/{model}-{num}"

class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, shape=RESOLUTION):
        super().__init__(env)
        self.image_shape = shape
        self.image_shape_reverse = shape[::-1]
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
    env = Monitor(env, f"{LOG_DIR}/")
    return env

env_kwargs = {"frame_skip": FRAME_SKIP}

def objective_dqn(trial):
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    buffer_size = trial.suggest_int('buffer_size', int(1e4), int(1e5))
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    exploration_fraction = trial.suggest_float('exploration_fraction', 0.1, 0.5)
    exploration_final_eps = trial.suggest_float('exploration_final_eps', 0.01, 0.1)

    train_env = make_vec_env(ENV, n_envs=N_ENVS, wrapper_class=wrap_env, env_kwargs=env_kwargs)
    train_env = VecTransposeImage(train_env)
    eval_env = make_vec_env(ENV, n_envs=N_ENVS, wrapper_class=wrap_env, env_kwargs=env_kwargs)
    eval_env = VecTransposeImage(eval_env)

    evaluation_callback = callbacks.EvalCallback(
        eval_env, best_model_save_path=f"{LOG_DIR}/models",
        log_path=f"{LOG_DIR}/eval", eval_freq=5000,
        render=False, n_eval_episodes=10,
    )

    agent = DQN(
        "CnnPolicy", train_env, batch_size=batch_size,
        learning_rate=learning_rate, gamma=gamma, exploration_final_eps=exploration_final_eps,
        exploration_fraction=exploration_fraction, buffer_size=buffer_size,
        learning_starts=1e4, verbose=1, device='cuda'
    )

    agent.learn(total_timesteps=TRAINING_TIMESTEPS, tb_log_name="dqn", callback=evaluation_callback)
    mean_reward, _ = evaluate_policy(agent, eval_env, n_eval_episodes=10)

    train_env.close()
    eval_env.close()

    return mean_reward

def objective_ppo(trial):
    n_steps = trial.suggest_categorical('n_steps', [2048, 4096, 8192])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    gae_lambda = trial.suggest_float('gae_lambda', 0.8, 0.99)

    train_env = make_vec_env(ENV, n_envs=N_ENVS, wrapper_class=wrap_env, env_kwargs=env_kwargs)
    train_env = VecTransposeImage(train_env)
    eval_env = make_vec_env(ENV, n_envs=N_ENVS, wrapper_class=wrap_env, env_kwargs=env_kwargs)
    eval_env = VecTransposeImage(eval_env)

    evaluation_callback = callbacks.EvalCallback(
        eval_env, best_model_save_path=f"{LOG_DIR}/models",
        log_path=f"{LOG_DIR}/eval", eval_freq=5000,
        render=False, n_eval_episodes=10,
    )

    agent = PPO(
        "CnnPolicy", train_env, n_steps=n_steps, batch_size=batch_size,
        learning_rate=learning_rate, gamma=gamma, gae_lambda=gae_lambda,
        verbose=1, device='cuda'
    )

    agent.learn(total_timesteps=TRAINING_TIMESTEPS, tb_log_name="ppo", callback=evaluation_callback)
    mean_reward, _ = evaluate_policy(agent, eval_env, n_eval_episodes=10)

    train_env.close()
    eval_env.close()

    return mean_reward

if __name__ == "__main__":
    # Optimize DQN
    if model == "dqn-tunning":
        dqn_study = optuna.create_study(direction='maximize')
        dqn_study.optimize(objective_dqn, n_trials=50)
        print(f'DQN Best trial: {dqn_study.best_trial.value}')
        print(f'DQN Best hyperparameters: {dqn_study.best_trial.params}')

        print(f'Best Hyperparameters: {dqn_study.best_params}')
        print(f'Best Reward: {dqn_study.best_value}')

    # # Optimize PPO
    if model == "ppo-tunning":
        ppo_study = optuna.create_study(direction='maximize')
        ppo_study.optimize(objective_ppo, n_trials=50)
        print(f'PPO Best trial: {ppo_study.best_trial.value}')
        print(f'PPO Best hyperparameters: {ppo_study.best_trial.params}')

        print(f'Best Hyperparameters: {ppo_study.best_params}')
        print(f'Best Reward: {ppo_study.best_value}')
