import matplotlib.pyplot as plt
import numpy as np

def plot_training_rewards(game_name, rewards, ma_window=100):
    """
    Show training rewards over time
    overlay with the moving average of the rewards over 100 episode
    """
    plt.figure(figsize=(8, 5))
    plt.plot(rewards, label="Episode Reward", alpha=0.6)
    if len(rewards) >= ma_window:
        moving_avg = np.convolve(rewards, np.ones(ma_window)/ma_window, mode='valid')
        plt.plot(range(ma_window-1, len(rewards)), moving_avg, label=f"{ma_window}-Episode Moving Average", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"{game_name}: Training Rewards over Episodes")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_rollout_histogram(game_name, rewards):
    """
    Show histogram of rewards from 500 evaluation episodes
    """
    plt.figure(figsize=(7, 5))
    plt.hist(rewards, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    plt.title(f"{game_name}: Histogram of Rewards (500 Evaluation Episodes)")
    plt.xlabel("Episode Reward")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"{game_name}: Mean reward over 500 episodes: {mean_reward:.2f}")
    print(f"{game_name}: Std. dev. of reward: {std_reward:.2f}")

def plot_max_q_values(game_name, max_q_values):
    """
    Show maximum Q-values over training episodes
    """
    plt.figure(figsize=(8, 5))
    plt.plot(max_q_values, label="Average Max Q per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Max Q-Value")
    plt.title(f"{game_name}: Maximum Q-Values over Training")
    plt.legend()
    plt.grid(True)
    plt.show()