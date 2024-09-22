import os
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

ENV = "VizdoomMyWayHome-v0"
RESOLUTION = (60, 45)
TRAINING_TIMESTEPS = int(3e5)# 300000
N_ENVS = 1
FRAME_SKIP = 4

model = "dqn"
num = "1"
map = "my-way-home"

LOG_DIR = f"tunning/{map}/{model}-{num}"

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


def custom_epsilon_schedule(initial_epsilon, final_epsilon, decay_start_steps, decay_end_steps):
    """
    Custom epsilon schedule that starts decaying epsilon after decay_start_steps.
    
    Parameters:
    - initial_epsilon: float, initial value of epsilon
    - final_epsilon: float, final value of epsilon
    - total_steps: int, total number of steps for the schedule
    - decay_start_steps: float, percentage of steps to delay the decay of epsilon
    
    Returns:
    - A function that takes the current step and returns the epsilon value.
    """
    
    def schedule(step_fraction):

        step = (1 - step_fraction) * TRAINING_TIMESTEPS

        if step < decay_start_steps:
            return initial_epsilon
        else:       
            fraction = (step - decay_start_steps) / (decay_end_steps - decay_start_steps)

            curvature = 1.5 # 1.0 for linear // >1.0 for slow decay // <1.0 for fast decay
            adjusted_fraction = fraction ** curvature

            return max(final_epsilon, initial_epsilon - adjusted_fraction * (initial_epsilon - final_epsilon))
    
    return schedule

env_kwargs = {"frame_skip": FRAME_SKIP}

def objective_dqn(trial):
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True) # 0.00001, 0.01
    buffer_size = trial.suggest_int('buffer_size', int(1e3), int(1.5e4))# 1000, 15000
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    exploration_fraction = trial.suggest_float('exploration_fraction', 0.1, 0.5)
    exploration_final_eps = trial.suggest_float('exploration_final_eps', 0.0001, 0.1)
    decay_start_steps = trial.suggest_float('decay_start_steps', 0.0, 0.5)
    decay_end_steps = trial.suggest_float('decay_end_steps', decay_start_steps, 1)

    epsilon_schedule_fn = custom_epsilon_schedule(
        initial_epsilon=1.0,
        final_epsilon=exploration_final_eps,
        decay_start_steps=decay_start_steps * TRAINING_TIMESTEPS,
        decay_end_steps=decay_end_steps * TRAINING_TIMESTEPS
    )

    train_env = make_vec_env(ENV, n_envs=N_ENVS, wrapper_class=wrap_env, env_kwargs=env_kwargs)
    train_env = VecTransposeImage(train_env)
    eval_env = make_vec_env(ENV, n_envs=N_ENVS, wrapper_class=wrap_env, env_kwargs=env_kwargs)
    eval_env = VecTransposeImage(eval_env)

    evaluation_callback = callbacks.EvalCallback(
        eval_env, best_model_save_path=f"{LOG_DIR}/models",
        log_path=f"{LOG_DIR}/eval", eval_freq=500,
        render=False, n_eval_episodes=50,
    )

    agent = DQN(
        "CnnPolicy", train_env, batch_size=batch_size,
        learning_rate=learning_rate, gamma=gamma, exploration_final_eps=exploration_final_eps,
        exploration_fraction=exploration_fraction, buffer_size=buffer_size,
        verbose=1, device='cuda'
    )

    agent.exploration_schedule = epsilon_schedule_fn

    agent.learn(total_timesteps=TRAINING_TIMESTEPS, tb_log_name="dqn", callback=evaluation_callback)
    mean_reward, _ = evaluate_policy(agent, eval_env, n_eval_episodes=100)

    train_env.close()
    eval_env.close()

    return mean_reward

def objective_ppo(trial):
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    gae_lambda = trial.suggest_float('gae_lambda', 0.8, 0.99)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
    ent_coef = trial.suggest_float('ent_coef', 0.00001, 0.01, log=True)
    vf_coef = trial.suggest_float('vf_coef', 0.1, 1.0)
    clip_range_vf = trial.suggest_categorical('clip_range_vf', [None, 0.1, 0.2, 0.3])
    target_kl = trial.suggest_float('target_kl', 0.01, 0.3)

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
        "CnnPolicy", train_env, n_steps=2048, batch_size=batch_size,
        learning_rate=learning_rate, gamma=gamma, gae_lambda=gae_lambda,
        clip_range=clip_range, clip_range_vf=clip_range_vf, target_kl=target_kl,
        ent_coef=ent_coef, vf_coef=vf_coef, verbose=1, device='cuda'
    )

    agent.learn(total_timesteps=TRAINING_TIMESTEPS, tb_log_name="ppo", callback=evaluation_callback)
    mean_reward, _ = evaluate_policy(agent, eval_env, n_eval_episodes=10)

    train_env.close()
    eval_env.close()

    return mean_reward

if __name__ == "__main__":
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    with open(f"{LOG_DIR}params_tunning.txt", "w") as f:
        # Optimize DQN
        if model == "dqn":
            dqn_study = optuna.create_study(direction='maximize')
            dqn_study.optimize(objective_dqn, n_trials=25)
            f.write(f'DQN Best trial: {dqn_study.best_trial.value}\n')
            f.write(f'DQN Best hyperparameters: {dqn_study.best_trial.params}\n')

        # Optimize PPO
        if model == "ppo":
            ppo_study = optuna.create_study(direction='maximize')
            ppo_study.optimize(objective_ppo, n_trials=25)
            f.write(f'PPO Best trial: {ppo_study.best_trial.value}\n')
            f.write(f'PPO Best hyperparameters: {ppo_study.best_trial.params}\n')

