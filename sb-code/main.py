import os
import cv2
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import callbacks
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_linear_fn
from gymnasium import RewardWrapper

from typing import Callable
import vizdoom.gymnasium_wrapper
from utils import plot_rewards
from params import DQN_PARAMS, PPO_PARAMS

ENV_LIST = [
    #"VizdoomDefendCenter-v0", 
    #"VizdoomDefendLine-v0",
    "VizdoomCorridor-v0",
    #"VizdoomMyWayHome-v0",
    #"VizdoomHealthGathering-v0"
    #"VizdoomPredictPosition-v0",
    #"VizdoomTakeCover-v0",
    #"VizdoomDeathmatch-v0",
]

MAP_LIST = [
    #"defend-center",
    #"defend-line",
    "corridor",
    #"my-way-home",
    #"health-gathering",
    #"predict-position",
    #"take-cover"
    #"deathmatch",
]

MODEL_LIST = [
    "dqn",
    #"ppo"
]

RESOLUTION = (60, 45)
TRAINING_TIMESTEPS = int(1e6)  # 600k 200k 1000k
N_ENVS = 1
FRAME_SKIP = 4

old_save = False
old_dir_dqn = "trains/corridor/dqn-1"
old_dir_ppo = "trains/corridor/ppo-stop-4-1"

#num = f"2-btn(menos)-fs({FRAME_SKIP})-steps({TRAINING_TIMESTEPS})"
#num = f"4-fs({FRAME_SKIP})-steps({TRAINING_TIMESTEPS})"
num = f"stop-1"

class RewardShapingWrapper(RewardWrapper):
    def __init__(self, env, damage_reward=100, hit_taken_penalty=-3, ammo_penalty=-1):
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

class RewardShapingWrapperDeathMatch(RewardWrapper):
    def __init__(self, env, item_reward=0.1, hit_reward=0.1, kill_reward=1.0):
        super(RewardShapingWrapperDeathMatch, self).__init__(env)
        self.item_reward = item_reward
        self.hit_reward = hit_reward
        self.kill_reward = kill_reward
        self.previous_itemcount = 0
        self.previous_hitcount = 0
        self.previous_killcount = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        game_state = self.env.unwrapped.game.get_state()

        if game_state:
            game_variables = game_state.game_variables
            self.previous_killcount = game_variables[0]  # KILLCOUNT
            self.previous_itemcount = game_variables[5]  # ITEMCOUNT
            self.previous_hitcount = game_variables[6]   # HITCOUNT

        print(f"Reset variables - Items: {self.previous_itemcount}, Hits: {self.previous_hitcount}, Kills: {self.previous_killcount}")
        return obs, info

    def reward(self, reward):
        print(f"Reward original: {reward}")
        custom_reward = reward
        game_state = self.env.unwrapped.game.get_state()

        if game_state:
            game_variables = game_state.game_variables
            current_itemcount = game_variables[5]  # ITEMCOUNT
            current_hitcount = game_variables[6]   # HITCOUNT
            current_killcount = game_variables[0]  # KILLCOUNT

            # Recompensa por recoger un ítem
            if current_itemcount > self.previous_itemcount:
                item_delta = current_itemcount - self.previous_itemcount
                reward_gain = item_delta * self.item_reward
                custom_reward += reward_gain
                print(f"Recompensa por ítem: {reward_gain}, Ítems recogidos: {item_delta}, Recompensa actual: {custom_reward}")
            self.previous_itemcount = current_itemcount

            # Recompensa por golpear a un enemigo
            if current_hitcount > self.previous_hitcount:
                hitcount_delta = current_hitcount - self.previous_hitcount
                reward_gain = hitcount_delta * self.hit_reward
                custom_reward += reward_gain
                print(f"Recompensa por golpe: {reward_gain}, Golpes hechos: {hitcount_delta}, Recompensa actual: {custom_reward}")
            self.previous_hitcount = current_hitcount

            # Recompensa fija por matar a un enemigo (KILLCOUNT)
            if current_killcount > self.previous_killcount:
                killcount_delta = current_killcount - self.previous_killcount
                reward_gain = killcount_delta * self.kill_reward
                custom_reward += reward_gain
                print(f"Recompensa por matar: {reward_gain}, Enemigos muertos: {killcount_delta}, Recompensa actual: {custom_reward}")
            self.previous_killcount = current_killcount

        print(f"Reward custom: {custom_reward}")
        return custom_reward

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

            curvature = 1.0 # 1.0 for linear // >1.0 for slow decay // <1.0 for fast decay
            adjusted_fraction = fraction ** curvature

            return max(final_epsilon, initial_epsilon - adjusted_fraction * (initial_epsilon - final_epsilon))
    
    return schedule

class EarlyStopCallback(BaseCallback):
    def __init__(self, stop_percentage, total_timesteps, verbose=0):
        super(EarlyStopCallback, self).__init__(verbose)
        # Calcula el número de timesteps en el que detenerse
        self.stop_timesteps = int(stop_percentage * total_timesteps)
        self.verbose = verbose

    def _on_step(self) -> bool:
        # Verifica si se ha alcanzado el límite de timesteps
        if self.num_timesteps >= self.stop_timesteps:
            if self.verbose:
                print(f"Deteniendo el entrenamiento en el {self.stop_timesteps} timesteps ("
                      f"{self.stop_timesteps / self.model._total_timesteps:.1%} del total).")
            return False  # Return False detiene el entrenamiento
        return True

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
        env = gym.wrappers.TransformReward(env, lambda r: r * 0.1)
        env = RewardShapingWrapper(env)
        #env = RewardShapingWrapperDeathMatch(env)
        env = Monitor(env, self.log_dir,)
        return env

if __name__ == "__main__":
    env_kwargs = {
        "frame_skip": FRAME_SKIP,
        #"render_mode": "human"
    }

    start_index = 0

    stop_percentage = 0.4  # Ejemplo de detenerse al 40
    early_stop_callback = EarlyStopCallback(stop_percentage=stop_percentage, total_timesteps=TRAINING_TIMESTEPS, verbose=1)

    for model in MODEL_LIST:
        for map_name, env_name in zip(MAP_LIST[start_index:], ENV_LIST[start_index:]):
            LOG_DIR = f"trains/{map_name}/{model}-{num}"

            if not os.path.exists(LOG_DIR):
                os.makedirs(LOG_DIR)

            log_file_path = f"{LOG_DIR}/training_log.txt"  # Guardar el log en el mismo directorio que LOG_DIR
            
            with open(log_file_path, 'w') as log_file:
                env_wrapper_train = EnvWrapper(log_dir=f"{LOG_DIR}/train_monitor.csv")
                env_wrapper_eval = EnvWrapper(log_dir=f"{LOG_DIR}/eval_monitor.csv")

                train_env = make_vec_env(env_name, n_envs=N_ENVS, wrapper_class=env_wrapper_train, env_kwargs=env_kwargs)
                train_env = VecTransposeImage(train_env)

                eval_env = make_vec_env(env_name, n_envs=N_ENVS, wrapper_class=env_wrapper_eval, env_kwargs=env_kwargs)
                eval_env = VecTransposeImage(eval_env)

                evaluation_callback = callbacks.EvalCallback(
                    eval_env,
                    best_model_save_path=f"{LOG_DIR}/models",
                    log_path=f"{LOG_DIR}/eval",
                    eval_freq=500,
                    render=False,
                    n_eval_episodes=50,
                )

                if model == "dqn":
                    params = DQN_PARAMS.get(env_name, {})

                    total_steps = TRAINING_TIMESTEPS
                    initial_epsilon = 1.0
                    final_epsilon = params.get("exploration_final_eps", 0.05)
                    exploration_fraction = params.get("exploration_fraction", 0.1)
                    learning_starts = params.get("learning_starts", 1000)
                    decay_start_steps = params.get("decay_start_steps", 0)
                    decay_end_steps = params.get("decay_end_steps", total_steps)

                    epsilon_schedule_fn = custom_epsilon_schedule(
                        initial_epsilon=initial_epsilon,
                        final_epsilon=final_epsilon,
                        decay_start_steps=decay_start_steps * total_steps,
                        decay_end_steps=decay_end_steps * total_steps
                    )

                    if old_save:
                        agent = DQN(
                            "CnnPolicy",
                            train_env,
                            batch_size=params.get("batch_size", 64),
                            learning_rate=params.get("learning_rate", 0.0001),
                            buffer_size=params.get("buffer_size", 50000),
                            gamma=params.get("gamma", 0.99),
                            exploration_initial_eps=initial_epsilon,
                            exploration_final_eps=final_epsilon,
                            target_update_interval=100,
                            learning_starts=learning_starts,
                            verbose=1,
                            device='cuda'
                        )
                        agent.load_replay_buffer(f"{old_dir_dqn}/saves/replay_buffer")
                        agent.policy.load(f"{old_dir_dqn}/policy/pesos.zip")
                    else:
                        agent = DQN(
                            "CnnPolicy",
                            train_env,
                            batch_size=params.get("batch_size", 64),
                            learning_rate=params.get("learning_rate", 0.0001),
                            buffer_size=params.get("buffer_size", 50000),
                            gamma=params.get("gamma", 0.99),
                            exploration_initial_eps=initial_epsilon,
                            exploration_final_eps=final_epsilon,
                            target_update_interval=100,
                            learning_starts=learning_starts,
                            verbose=1,
                            device='cuda'
                        )
                        
                    agent.exploration_schedule = epsilon_schedule_fn

                    epsilon_logger = EpsilonLogger(LOG_DIR)

                    agent.learn(total_timesteps=TRAINING_TIMESTEPS, tb_log_name="dqn", callback=[evaluation_callback, epsilon_logger])
                    agent.save(f"{LOG_DIR}/saves/dqn_vizdoom")
                    agent.save_replay_buffer(f"{LOG_DIR}/saves/replay_buffer")
                    if not os.path.exists(f"{LOG_DIR}/policy"): os.makedirs(f"{LOG_DIR}/policy")
                    agent.policy.save(f"{LOG_DIR}/policy/pesos.zip")

                    log_file.write(f"Parameters: {params}\n")

                else:
                    params = PPO_PARAMS.get(env_name, {})

                    if old_save:
                        agent = PPO(
                            "CnnPolicy",
                            train_env,
                            n_steps=params.get("n_steps", 2048),
                            batch_size=params.get("batch_size", 64),
                            learning_rate=params.get("learning_rate", 3e-4),
                            gamma=params.get("gamma", 0.99),
                            gae_lambda=params.get("gae_lambda", 0.95),
                            clip_range=params.get("clip_range", 0.2),
                            ent_coef=params.get("ent_coef", 0.0),
                            vf_coef=params.get("vf_coef", 0.5),
                            clip_range_vf=params.get("clip_range_vf", None),
                            target_kl=params.get("target_kl", 0.01),
                            verbose=1,
                            device='cuda'
                        )
                        agent.policy.load(f"{old_dir_ppo}/policy/pesos.zip")
                    else:
                        agent = PPO(
                            "CnnPolicy",
                            train_env,
                            n_steps=params.get("n_steps", 2048),
                            batch_size=params.get("batch_size", 64),
                            learning_rate=params.get("learning_rate", 3e-4),
                            gamma=params.get("gamma", 0.99),
                            gae_lambda=params.get("gae_lambda", 0.95),
                            clip_range=params.get("clip_range", 0.2),
                            ent_coef=params.get("ent_coef", 0.0),
                            vf_coef=params.get("vf_coef", 0.5),
                            clip_range_vf=params.get("clip_range_vf", None),
                            target_kl=params.get("target_kl", 0.01),
                            verbose=1,
                            device='cuda'
                        )
                    agent.learn(total_timesteps=TRAINING_TIMESTEPS, tb_log_name="ppo", callback=[evaluation_callback])
                    agent.save(f"{LOG_DIR}/saves/ppo_vizdoom")
                    if not os.path.exists(f"{LOG_DIR}/policy"): os.makedirs(f"{LOG_DIR}/policy")
                    agent.policy.save(f"{LOG_DIR}/policy/pesos.zip")

                    log_file.write(f"Parameters: {params}\n")

                log_file.write(f"Training steps: {TRAINING_TIMESTEPS}\n")
                log_file.write(f"Frame skip: {FRAME_SKIP}\n")
                log_file.write(f"Model: {model}-{num}\n")
                
                train_env.close()
                eval_env.close()

                plot_rewards(LOG_DIR, model)



