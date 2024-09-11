import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def smooth_rewards(rewards, window_size):
    return np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')

def plot_rewards(log_dir, is_dqn, window=10):
    # Read the CSV file while handling malformed rows
    data = []
    try:
        with open(os.path.join(log_dir, "train_monitor.csv"), 'r') as file:
            for line in file.readlines()[2:]:  # Skip the first two lines
                try:
                    r, l, t = map(float, line.split(','))
                    data.append((r, l, t))
                except ValueError as e:
                    print(f"Skipping malformed line: {line.strip()} (Error: {e})")
                    continue  # Skip malformed rows
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Convert to DataFrame
    if not data:
        print("No valid data found in the CSV file.")
        return

    results_df = pd.DataFrame(data, columns=['r', 'l', 't'])

    # Smooth the rewards
    smoothed_rewards = smooth_rewards(results_df['r'].to_numpy(), window)

    # Calculate average reward
    avg_reward = results_df['r'].mean()

    # Plot all rewards and smoothed rewards
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward', color=color)
    ax1.plot(results_df.index, results_df['r'], color=color, label='Reward')
    ax1.plot(range(len(smoothed_rewards)), smoothed_rewards, color='tab:orange', linestyle='--', label='Smoothed Reward')
    ax1.axhline(avg_reward, color='tab:green', linestyle='-', label='Average Reward')

    ax1.tick_params(axis='y', labelcolor=color)
    plt.grid()

    if is_dqn == "dqn":
        # Load epsilon values
        try:
            epsilon_values = np.load(os.path.join(log_dir, 'epsilons.npy'))
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
        
        # Ignorar el primer valor
        epsilon_values = epsilon_values[1:]

        # Normalize epsilon steps to match the number of episodes
        epsilon_steps, epsilon_vals = zip(*enumerate(epsilon_values))
        normalized_steps = [step / len(epsilon_values) * len(results_df) for step in epsilon_steps]

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Epsilon', color=color)
        ax2.plot(normalized_steps, epsilon_vals, color=color, label='Epsilon')
        ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    if is_dqn == "dqn":
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    else:
        ax1.legend(loc='upper right')

    # Save the limits for reuse in the next plot
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()

    plt.savefig(os.path.join(log_dir, 'reward.png'))
    plt.close()

    # Plot only smoothed rewards and average reward
    fig, ax1 = plt.subplots()

    color = 'tab:orange'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Smoothed Reward')
    ax1.plot(range(len(smoothed_rewards)), smoothed_rewards, color=color, label='Smoothed Reward')
    ax1.axhline(avg_reward, color='tab:green', linestyle='-', label='Average Reward')

    ax1.tick_params(axis='y')

    # Apply the same limits from the combined plot
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)

    fig.tight_layout()
    ax1.legend(loc='upper right')

    plt.grid()

    plt.savefig(os.path.join(log_dir, 'smoothed_reward.png'))
    plt.close()


def plot_comparison(log_dir1, log_dir2, window=10):
    def read_rewards(log_dir):
        data = []
        try:
            with open(os.path.join(log_dir, "train_monitor.csv"), 'r') as file:
                for line in file.readlines()[2:]:  # Skip the first two lines
                    try:
                        r, l, t = map(float, line.split(','))
                        data.append((r, l, t))
                    except ValueError as e:
                        print(f"Skipping malformed line: {line.strip()} (Error: {e})")
                        continue  # Skip malformed rows
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None
        
        if not data:
            print("No valid data found in the CSV file.")
            return None
        
        return pd.DataFrame(data, columns=['r', 'l', 't'])

    results_df1 = read_rewards(log_dir1)
    results_df2 = read_rewards(log_dir2)

    if results_df1 is None or results_df2 is None:
        return

    # Smooth the rewards
    smoothed_rewards1 = smooth_rewards(results_df1['r'].to_numpy(), window)
    smoothed_rewards2 = smooth_rewards(results_df2['r'].to_numpy(), window)

    # Plot all smoothed rewards
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.plot(range(len(smoothed_rewards2)), smoothed_rewards2, color='tab:orange', linestyle='-', label='PPO')
    ax1.plot(range(len(smoothed_rewards1)), smoothed_rewards1, color='tab:blue', linestyle='-', label='DQN')

    ax1.legend(loc='upper right')
    plt.tight_layout()

    plt.grid()

    plt.savefig(os.path.join(log_dir1, 'comparison_reward.png'))
    plt.savefig(os.path.join(log_dir2, 'comparison_reward.png'))
    plt.close()

def plot_side_by_side(log_dir1, log_dir2, window=10):
    def read_rewards(log_dir):
        data = []
        try:
            with open(os.path.join(log_dir, "train_monitor.csv"), 'r') as file:
                for line in file.readlines()[2:]:  # Skip the first two lines
                    try:
                        r, l, t = map(float, line.split(','))
                        data.append((r, l, t))
                    except ValueError as e:
                        print(f"Skipping malformed line: {line.strip()} (Error: {e})")
                        continue  # Skip malformed rows
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None
        
        if not data:
            print("No valid data found in the CSV file.")
            return None
        
        return pd.DataFrame(data, columns=['r', 'l', 't'])

    results_df1 = read_rewards(log_dir1)
    results_df2 = read_rewards(log_dir2)

    if results_df1 is None or results_df2 is None:
        return

    # Smooth the rewards
    smoothed_rewards1 = smooth_rewards(results_df1['r'].to_numpy(), window)
    smoothed_rewards2 = smooth_rewards(results_df2['r'].to_numpy(), window)

    # Plot side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot dataset 1
    ax1.set_title('DQN')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.plot(range(len(smoothed_rewards1)), smoothed_rewards1, color='tab:blue', linestyle='-', label='DQN')
    ax1.legend(loc='upper right')

    # Plot dataset 2
    ax2.set_title('PPO')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward')
    ax2.plot(range(len(smoothed_rewards2)), smoothed_rewards2, color='tab:orange', linestyle='-', label='PPO')
    ax2.legend(loc='upper right')

    plt.tight_layout()

    plt.grid()

    plt.savefig(os.path.join(log_dir1, 'side_by_side_reward.png'))
    plt.savefig(os.path.join(log_dir2, 'side_by_side_reward.png'))
    plt.close()


# Create a plot for only epsilon per steps
def plot_epsilon(log_dir):
    try:
        epsilon_values = np.load(os.path.join(log_dir, 'epsilons.npy'))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    fig, ax = plt.subplots()

    ax.set_xlabel('Step')
    ax.set_ylabel('Epsilon')
    ax.plot(epsilon_values, color='tab:red', label='Epsilon')

    ax.legend(loc='upper right')
    plt.tight_layout()

    plt.savefig(os.path.join(log_dir, 'epsilon.png'))
    plt.close()

# Función para leer los puntajes desde el archivo .txt
def read_scores(file_path):
    data = []
    try:
        with open(file_path, 'r') as file:
            for line in file.readlines()[1:]:  # Saltar la primera línea (cabecera)
                try:
                    episode, score = line.split(',')
                    score = float(score.strip().replace('[', '').replace(']', ''))  # Limpiar el formato de score
                    data.append((int(episode), score))
                except ValueError as e:
                    print(f"Skipping malformed line: {line.strip()} (Error: {e})")
                    continue  # Saltar filas malformadas
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    if not data:
        print("No valid data found in the file.")
        return None

    return pd.DataFrame(data, columns=['Episode', 'Score'])

# Función para graficar la comparación de los tres modelos
def plot_results(map_dir, window=100):
    # Construir los nombres de archivo de puntuaciones para cada modelo
    ppo_file = os.path.join(map_dir, 'puntajes_ppo.txt')
    dqn_file = os.path.join(map_dir, 'puntajes_dqn.txt')
    random_file = os.path.join(map_dir, 'puntajes_random.txt')

    # Leer puntajes de los archivos
    ppo_scores = read_scores(ppo_file)
    dqn_scores = read_scores(dqn_file)
    random_scores = read_scores(random_file)

    if ppo_scores is None or dqn_scores is None or random_scores is None:
        return

    # Suavizar los puntajes
    ppo_smoothed = smooth_rewards(ppo_scores['Score'].to_numpy(), window)
    dqn_smoothed = smooth_rewards(dqn_scores['Score'].to_numpy(), window)
    random_smoothed = smooth_rewards(random_scores['Score'].to_numpy(), window)

    # Crear el gráfico
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    
    # Graficar puntajes suavizados para cada modelo
    ax1.plot(range(len(ppo_smoothed)), ppo_smoothed, color='tab:orange', alpha=0.8, linestyle='-', label='PPO')
    ax1.plot(range(len(dqn_smoothed)), dqn_smoothed, color='tab:blue', alpha=0.8, linestyle='-', label='DQN')
    ax1.plot(range(len(random_smoothed)), random_smoothed, color='tab:green', alpha=0.8, linestyle='-', label='Random')

    # Añadir leyenda y ajustar diseño
    ax1.legend(loc='upper right')
    plt.tight_layout()

    plt.grid()

    # Guardar la gráfica en el directorio del mapa
    comparison_file = os.path.join(map_dir, 'comparison_reward.png')
    plt.savefig(comparison_file)
    plt.close()