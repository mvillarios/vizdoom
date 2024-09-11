from utils import plot_comparison, plot_side_by_side, plot_rewards, plot_epsilon, plot_results

# Plot comparison between two models
path = "trains/take-cover"

#plot_comparison(f"{path}dqn-5", f"{path}ppo-6", window=10)
#plot_side_by_side(f"{path}dqn-1", f"{path}ppo-1", window=10)
#plot_rewards(f"{path}dqn-1", False, window=10)

#plot_epsilon(f"{path}dqn-6")

# Plot results for all models
plot_results(path)