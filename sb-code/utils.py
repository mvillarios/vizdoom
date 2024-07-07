import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def smooth_rewards(rewards, window_size):
    return np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')

def plot_rewards(log_dir, is_dqn, window=10):
    # Read the CSV file while handling malformed rows
    data = []
    with open(os.path.join(log_dir, ".monitor.csv"), 'r') as file:
        for line in file.readlines()[2:]:  # Skip the first two lines
            try:
                r, l, t = map(float, line.split(','))
                data.append((r, l, t))
            except ValueError:
                continue  # Skip malformed rows

    # Convert to DataFrame
    results_df = pd.DataFrame(data, columns=['r', 'l', 't'])

    # Smooth the rewards
    smoothed_rewards = smooth_rewards(results_df['r'].to_numpy(), window)

    # Plot the rewards
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward', color=color)
    ax1.plot(results_df.index, results_df['r'], color=color, label='Reward')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.plot(range(len(smoothed_rewards)), smoothed_rewards, color='tab:orange', linestyle='--', label='Smoothed Reward')

    # Set the y-axis limits and ticks
    ax1.set_ylim(-1, 15)
    ax1.set_yticks(np.arange(-1, 16, 1))

    if is_dqn == "dqn":
        # Load epsilon values
        epsilon_values = np.load(os.path.join(log_dir, 'epsilons.npy'))
        
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
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.savefig(os.path.join(log_dir, 'reward.png'))
    plt.close()