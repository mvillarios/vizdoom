import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_rewards(log_dir, is_dqn, window=10):
    # Read the CSV file while handling malformed rows
    data = []
    with open(log_dir + "/monitor.csv", 'r') as file:
        for line in file.readlines()[2:]:  # Skip the first two lines
            try:
                r, l, t = map(float, line.split(','))
                data.append((r, l, t))
            except ValueError:
                continue  # Skip malformed rows

    # Convert to DataFrame
    results_df = pd.DataFrame(data, columns=['r', 'l', 't'])

    # Smooth the rewards
    results_df['r'] = results_df['r'].rolling(window=window).mean()

    # Plot the rewards and epsilon
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Reward', color='tab:blue')
    ax1.plot(results_df['r'], color='tab:blue', label='Reward')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    if is_dqn:
        # Load epsilon values
        epsilons = np.load(os.path.join(log_dir, 'epsilons.npy'))
        ax2 = ax1.twinx()
        ax2.set_ylabel('Epsilon', color='tab:orange')
        ax2.plot(epsilons, color='tab:orange', label='Epsilon')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.tight_layout()
    plt.title('Reward per Step')
    plt.savefig(os.path.join(log_dir, 'reward.png'))
    plt.show()