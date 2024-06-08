import matplotlib.pyplot as plt
import itertools as it
from time import time, sleep
import csv
import os
from datetime import datetime

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


def train_agent(game, agent, actions, scenario, save_model, EPISODES_TO_TRAIN, FRAME_REPEAT):
    start_time = time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    episodes = EPISODES_TO_TRAIN
    episode_rewards = []
    episode_losses = []
    epsilon_values = []
    max_reward = float('-inf')
    best_episode = -1
    steps_count = 0

    # Abrir el archivo CSV para escribir
    with open(f'{scenario}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Start datetime', start_datetime])
        writer.writerow(['Episode', 'Reward'])
        file.flush()

        print("Training")

        for episode in range(episodes):
            recorded_episode = f"episode{episode}_rec.lmp"
            game.new_episode(recorded_episode)
        
            state = agent.preprocess(game.get_state().screen_buffer)

            total_reward = 0
            total_loss = 0
            for step in it.count():
                action = agent.select_action(state)
                reward = game.make_action(actions[action], FRAME_REPEAT)
                done = game.is_episode_finished()
                total_reward += reward
                next_state = agent.preprocess(game.get_state().screen_buffer) if not done else None
                agent.remember(state, action, next_state, reward)
                state = next_state
                steps_count += 1

                loss, epsilon = agent.train()
                epsilon_values.append(epsilon)

                if loss is not None:
                    total_loss += loss.item()

                if done:
                    episode_rewards.append(total_reward)
                    episode_losses.append(total_loss)

                    writer.writerow([episode, total_reward])
                    file.flush()
                    break

            if total_reward >= max_reward:
                max_reward = total_reward
                best_episode = episode

            if save_model:
                agent.save_model()

        # Escribir la fecha y hora de finalización en el archivo CSV
        writer.writerow(['End datetime', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    # Eliminar los archivos de episodios que no son el mejor
    for episode in [f"episode{episode}_rec.lmp" for episode in range(episodes) if episode != best_episode]:
        if os.path.exists(episode):
            os.remove(episode)

    # Graficar y guardar cada serie de datos en una imagen separada
    plot_and_save(f'{scenario}_reward.png', 'Episode', 'Reward', episode_rewards)
    plot_and_save(f'{scenario}_loss.png', 'Episode', 'Loss', episode_losses)
    plot_and_save(f'{scenario}_epsilon.png', 'Step', 'Epsilon', epsilon_values)

    # Imprimir el tiempo total transcurrido, el mejor episodio y la máxima recompensa
    print("Total elapsed time: %.2f minutes" % ((time() - start_time) / 60.0))
    print("Best episode: ", best_episode)
    print("Max reward: ", max_reward)
    print("Steps count: ", steps_count)

def plot_and_save(filename, xlabel, ylabel, data):
    plt.figure(figsize=(12, 8))
    plt.plot(range(len(data)), data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(range(len(data)), [int(x) for x in range(len(data))])
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
