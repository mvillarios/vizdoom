import matplotlib.pyplot as plt
import pandas as pd

def plot_rewards(log_dir, window=10):
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

    # Plot the rewards
    rewards = results_df['r']

    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.savefig(f"{log_dir}/reward_per_episode.png")