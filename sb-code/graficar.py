from utils import plot_comparison, plot_side_by_side, plot_rewards, plot_epsilon

# Plot comparison between two models
path = "trains/my-way-home/"

#plot_comparison(f"{path}dqn-3", f"{path}ppo-3", window=10)
#plot_side_by_side(f"{path}dqn-3", f"{path}ppo-3", window=10)
plot_rewards(f"{path}dqn-1", False, window=10)

#plot_epsilon(f"{path}dqn-6")
