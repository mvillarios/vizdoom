import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

LOG_DIR = "saves-tunning/defend-line/ppo-1"

def smooth_rewards(rewards, window_size):
    return np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')

def plot_rewards_smoothed(filename, rewards, smoothed_rewards):
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward', color=color)
    ax1.plot(range(len(rewards)), rewards, color=color, label='Reward')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.plot(range(len(smoothed_rewards)), smoothed_rewards, color='tab:orange', linestyle='--', label='Smoothed Reward')

    fig.tight_layout()
    ax1.legend(loc='upper left')

    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    # Read the CSV file while handling malformed rows
    data = []
    try:
        with open(os.path.join(LOG_DIR, ".monitor.csv"), 'r') as file:
            for line in file.readlines()[2:]:  # Skip the first two lines
                try:
                    r, l, t = map(float, line.split(','))
                    data.append((r, l, t))
                except ValueError as e:
                    print(f"Skipping malformed line: {line.strip()} (Error: {e})")
                    continue  # Skip malformed rows
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit()

    # Convert to DataFrame
    if not data:
        print("No valid data found in the CSV file.")
        exit()

    results_df = pd.DataFrame(data, columns=['r', 'l', 't'])

    # Get the rewards and apply smoothing
    rewards = results_df['r']
    smoothed_rewards = smooth_rewards(rewards, window_size=10)

    # Plot the rewards and smoothed rewards
    plot_rewards_smoothed(os.path.join(LOG_DIR, "reward_per_episode_smoothed.png"), rewards, smoothed_rewards)
