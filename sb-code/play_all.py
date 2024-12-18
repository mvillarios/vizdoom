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

from utils import plot_results


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
    "random",  # Agregamos la opción de jugar en modo random
    "ppo",
]

# Diccionario de modelos específicos para cada algoritmo, incluyendo el número de modelo y la subcarpeta del modelo
MODEL_PATHS = {
    #"dqn": {"model": "dqn_vizdoom", "num": 6, "subfolder": "saves"},  # dqn usa "dqn_vizdoom" en la subcarpeta "saves"
    #"ppo": {"model": "ppo_vizdoom", "num": "stop-3-3", "subfolder": "saves"},
    "dqn": {"model": "best_model", "num": "stop-2-1", "subfolder": "models"},  # dqn usa "best_model" en la subcarpeta "models"
    "ppo": {"model": "best_model", "num": "stop-3-3", "subfolder": "models"}   # ppo usa "best_model" en la subcarpeta "models"
}

RESOLUTION = (60, 45)
FRAME_SKIP = 4

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
        custom_reward = reward
        game_state = self.env.unwrapped.game.get_state()

        if game_state:
            game_variables = game_state.game_variables
            current_damage_taken = game_variables[1]  # DAMAGE_TAKEN
            current_hitcount = game_variables[2]  # HITCOUNT
            current_ammo = game_variables[3]  # SELECTED_WEAPON_AMMO

            # Penalización por recibir daño
            if current_damage_taken > self.previous_damage_taken:
                damage_taken_delta = current_damage_taken - self.previous_damage_taken
                penalty = damage_taken_delta * self.hit_taken_penalty
                custom_reward += penalty
                #print(f"Penalización por recibir daño: {penalty}, Daño recibido: {damage_taken_delta}, Recompensa actual: {custom_reward}")
            self.previous_damage_taken = current_damage_taken

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
        game_variables = self.env.unwrapped.game.get_state().game_variables

        # Reiniciar las variables necesarias
        self.previous_itemcount = game_variables[5]  # ITEMCOUNT
        self.previous_hitcount = game_variables[6]   # HITCOUNT
        self.previous_killcount = game_variables[0]  # KILLCOUNT

        return obs, info

    def reward(self, reward):
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
                #print(f"Recompensa por ítem: {reward_gain}, Ítems recogidos: {item_delta}, Recompensa actual: {custom_reward}")
            self.previous_itemcount = current_itemcount

            # Recompensa por golpear a un enemigo
            if current_hitcount > self.previous_hitcount:
                hitcount_delta = current_hitcount - self.previous_hitcount
                reward_gain = hitcount_delta * self.hit_reward
                custom_reward += reward_gain
                #print(f"Recompensa por golpe: {reward_gain}, Golpes hechos: {hitcount_delta}, Recompensa actual: {custom_reward}")
            self.previous_hitcount = current_hitcount

            # Recompensa fija por matar a un enemigo (KILLCOUNT)
            if current_killcount > self.previous_killcount:
                killcount_delta = current_killcount - self.previous_killcount
                reward_gain = killcount_delta * self.kill_reward
                custom_reward += reward_gain
                #print(f"Recompensa por matar: {reward_gain}, Enemigos muertos: {killcount_delta}, Recompensa actual: {custom_reward}")
            self.previous_killcount = current_killcount

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
    env_kwargs = {
        "frame_skip": FRAME_SKIP,
        #"render_mode": "human"
    }

    start_index = 0  # Por si quieres iniciar desde un índice particular

    # Itera sobre los modelos
    for model in MODEL_LIST:
        # Itera sobre los mapas y entornos correspondientes
        for map_name, env_name in zip(MAP_LIST[start_index:], ENV_LIST[start_index:]):
            # Selecciona dinámicamente el modelo, número de entrenamiento y la subcarpeta
            if model != "random":
                model_filename = MODEL_PATHS[model]["model"]  # Elige el archivo de modelo que corresponde
                model_num = MODEL_PATHS[model]["num"]        # Elige el número del modelo
                subfolder = MODEL_PATHS[model]["subfolder"]  # Elige la subcarpeta (saves o models)
            else:
                model_filename = None
                model_num = None
                subfolder = None

            # Define el directorio de logs para cada mapa y modelo, utilizando el número seleccionado
            LOG_DIR = f"trains/{map_name}"
            os.makedirs(LOG_DIR, exist_ok=True)  # Crea el directorio si no existe

            # Archivo de puntajes específico para el modelo
            output_file = f"{LOG_DIR}/puntajes_{model}.txt"

            # Carga el agente entrenado o configura el modo aleatorio
            if model == "random":
                agent = None  # No se necesita un agente entrenado para el modo aleatorio
            else:
                # Construye el modelo path de acuerdo al modelo y subcarpeta
                MODEL_PATH = f"trains/{map_name}/{model}-{model_num}/{subfolder}/{model_filename}"

                if model == "dqn":
                    agent = DQN.load(MODEL_PATH, print_system_info=True)
                else:
                    agent = PPO.load(MODEL_PATH, print_system_info=True)

            def wrap_env(env):
                env = ObservationWrapper(env)
                #env = RewardShapingWrapperDeathMatch(env)
                return env

            # Crea el entorno vectorizado
            env = make_vec_env(env_name, wrapper_class=wrap_env, env_kwargs=env_kwargs)

            episode = 0

            # Abre el archivo de puntajes en modo de escritura para el modelo específico
            with open(output_file, "w") as f:
                f.write("Episodio,Puntaje\n")  # Encabezado del archivo

                # Ejecuta 50 episodios por cada mapa y modelo
                for _ in range(2000):
                    obs = env.reset()
                    done = False
                    total_reward = 0
                    episode += 1

                    while not done:
                        if model == "random":
                            action = np.array([env.action_space.sample()])  # Acciones aleatorias
                        else:
                            action, _ = agent.predict(obs, deterministic=True)  # Acciones del agente entrenado
                        
                        obs, reward, done, info = env.step(action)
                        total_reward += reward
                        env.render()

                    # Guarda los puntajes en el archivo
                    f.write(f"{episode},{total_reward}\n")
                    print(f"Episode: {episode} Reward: {total_reward}")

            env.close()
    
    # Grafica los resultados de los puntajes
    plot_results(f"trains/{MAP_LIST[0]}")
