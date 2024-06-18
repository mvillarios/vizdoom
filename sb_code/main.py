import cv2
import imageio
import numpy as np
import vizdoom
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common import policies

from common import envs  # Importa tus definiciones personalizadas de entornos si es necesario

# Funciones de creación de entornos y agente

def create_env(seed: int = None, **kwargs):
    # Create a VizDoom instance.
    game = vizdoom.DoomGame()
    game.load_config('scenarios/basic.cfg')
    game.init()

    # Wrap the environment with the Gym adapter.
    return envs.DoomEnv(game, **kwargs)

def create_vec_env(seed: int = None, **kwargs):
    # Envuelva el entorno en un DummyVecEnv y aplique VecTransposeImage para procesamiento de imágenes
    vec_env = DummyVecEnv([lambda: create_env(seed=seed, **kwargs)])
    return VecTransposeImage(vec_env)

def create_agent(env, **kwargs):
    return PPO(policy=policies.ActorCriticCnnPolicy,
                   env=env,
                   n_steps=4096,
                   batch_size=32,
                   learning_rate=1e-4,
                   tensorboard_log='logs/tensorboard',
                   verbose=0,
                   **kwargs)

# Función para crear un GIF del agente jugando
def make_gif(agent, file_path):
    env = create_vec_env(frame_skip=1)
    env.venv.envs[0].game.set_seed(0)
    
    images = []

    for i in range(20):
        obs = env.reset()

        done = False
        while not done:
            action, _ = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            images.append(env.venv.envs[0].game.get_state().screen_buffer)

    imageio.mimsave(file_path, images, fps=35)

    env.close()

# Ejecución principal del entrenamiento y creación de GIFs
if __name__ == '__main__':

    env_args = {
        'frame_skip': 4, 
        'frame_processor': lambda frame: cv2.resize(
            frame, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA
        )}

    training_env, eval_env = create_vec_env(seed=0, **env_args), create_vec_env(seed=0, **env_args)
    agent = create_agent(training_env)

    evaluation_callback = EvalCallback( 
        eval_env,
        n_eval_episodes=10,
        eval_freq=5000,
        log_path='logs/evaluations/basic',
        best_model_save_path='logs/models/basic'
    )

    agent.learn(total_timesteps=40000, tb_log_name='ppo_basic_simplified', callback=evaluation_callback)

    training_env.close()
    eval_env.close()

    make_gif(agent, 'logs/gifs/basic.gif')
