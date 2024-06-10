import matplotlib.pyplot as plt
import itertools as it
from time import time, sleep
import csv
import os
from datetime import datetime
import numpy as np

import vizdoom as vzd

def create_game(config_file_path, window_visible):
    print("Creating game...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(window_visible)
    game.set_mode(vzd.Mode.ASYNC_PLAYER if window_visible else vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    print("Game created.")
    return game

def train_agent(game, agent, actions, scenario, save_model, STEPS_TO_TRAIN, FRAME_REPEAT, start_decay_steps, end_decay_steps):
    start_time = time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    max_steps = STEPS_TO_TRAIN
    steps_count = 0
    episode_count = 0
    episode_rewards = []
    epsilon_values = []
    max_reward = float('-inf')
    best_episode = -1
    episodes_to_save= []

    # Abrir el archivo CSV para escribir
    with open(f'{scenario}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Start datetime', start_datetime])
        writer.writerow(['Episode', 'Reward'])
        file.flush()

        print("Training")

        while steps_count < max_steps:
            recorded_episode = f"episode{episode_count}_rec.lmp"
            game.new_episode(recorded_episode)
        
            state = agent.preprocess(game.get_state().screen_buffer)
            total_reward = 0
            for step in it.count():
                if steps_count >= max_steps:
                    break
        
                action = agent.select_action(state)
                reward = game.make_action(actions[action], FRAME_REPEAT)
                done = game.is_episode_finished()
                total_reward += reward
                next_state = agent.preprocess(game.get_state().screen_buffer) if not done else None
                agent.remember(state, action, next_state, reward)
                state = next_state
                steps_count += 1

                epsilon = agent.train()
                if epsilon is not None:
                    epsilon_values.append((steps_count, epsilon))

                if steps_count == start_decay_steps or steps_count == end_decay_steps:
                    episodes_to_save.append(episode_count)

                if done:
                    episode_rewards.append(total_reward)
                    writer.writerow([episode_count, total_reward])
                    file.flush()
                    break
            if total_reward >= max_reward:
                max_reward = total_reward
                best_episode = episode_count

            if save_model:
                agent.save_model()

            episode_count += 1

        # Escribir la fecha y hora de finalización en el archivo CSV
        writer.writerow(['End datetime', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    
    episodes_to_save.append(best_episode)
    # Eliminar los episodios grabados
    for episode in range(episode_count):
        if episode not in episodes_to_save:
            file = f"episode{episode}_rec.lmp"
            if os.path.exists(file):
                os.remove(file)

    # Graficar el epsilon, las recompensas y las recompensas suavizadas en el mismo gráfico
    smoothed_rewards = smooth_rewards(episode_rewards, window_size=10)
    plot_rewards_epsilon_and_smoothed(f'{scenario}.png', episode_rewards, epsilon_values, smoothed_rewards, steps_count)

    # Imprimir el tiempo total transcurrido, el mejor episodio y la máxima recompensa
    print("Total elapsed time: %.2f minutes" % ((time() - start_time) / 60.0))
    print("Best episode: ", best_episode)
    print("Max reward: ", max_reward)
    print("Average reward: ", np.mean(episode_rewards))
    print("Steps count: ", steps_count)

def plot_rewards_epsilon_and_smoothed(filename, rewards, epsilon_values, smoothed_rewards, steps_count):
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward', color=color)
    ax1.plot(range(len(rewards)), rewards, color=color, label='Reward')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.plot(range(len(smoothed_rewards)), smoothed_rewards, color='tab:red', linestyle='--', label='Smoothed Reward')

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Epsilon', color=color)
    epsilon_steps, epsilon_vals = zip(*epsilon_values)
    normalized_steps = [step / steps_count * len(rewards) for step in epsilon_steps]
    ax2.plot(normalized_steps, epsilon_vals, color=color, label='Epsilon')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.savefig(filename)
    plt.close()

def smooth_rewards(rewards, window_size):
    return np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

def plot_and_save(filename, xlabel, ylabel, data, x_values=None):
    plt.figure()
    if x_values is None:
        x_values = range(len(data))
    plt.plot(x_values, data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()

def play_game(game, agent, actions, EPISODES_TO_PLAY, FRAME_REPEAT):

    print("Playing game...")
    for _ in range(EPISODES_TO_PLAY):
        game.new_episode()
        while not game.is_episode_finished():
            state = agent.preprocess(game.get_state().screen_buffer)
            best_action_index = agent.select_action(state)

            game.set_action(actions[best_action_index])
            for _ in range(FRAME_REPEAT):
                game.advance_action()

        # Sleep between episodes
        sleep(0.5)
        score = game.get_total_reward()
        print("Total score: ", score)

def play_recorded_game(game, episode_file):
    print("Playing recorded game...")
    game.replay_episode(episode_file)
    while not game.is_episode_finished():
        game.advance_action()

    score = game.get_total_reward()
    print("Total score: ", score)

    print("Episode finished.")
