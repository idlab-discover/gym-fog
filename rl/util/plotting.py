import matplotlib
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt

#matplotlib.use('AGG')

EpisodeStats = namedtuple("episode_stats",["episode_lengths", "episode_rewards", "episode_running_variance", "episode_cost", "episode_time", "episode_requests"])
ILP_Stats = namedtuple("ilp_stats",["execution_time"])
LatencyStats = namedtuple("latency_stats",["latency_rl", "latency_milp", "latency_diff"])

def plot_episode_stats(figName, stats, smoothing_window=10):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Number of Steps")
    plt.title("Episode Length over Time")
    plt.savefig(figName + '_length.png', dpi=250, bbox_inches='tight')

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(figName + '_reward.png', dpi=250, bbox_inches='tight')

    # Plot cost over time
    fig3 = plt.figure(figsize=(10, 5))
    cost_smoothed = pd.Series(stats.episode_cost).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(cost_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Cost Diff between Agent vs MILP (%)")
    plt.title("Avg Cost over Time (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(figName + '_cost.png', dpi=250, bbox_inches='tight')

    # Plot execution time over time
    fig4 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_time)
    plt.xlabel("Episode")
    plt.ylabel("Execution Time (s)")
    plt.title("Episode Execution Time")
    plt.savefig(figName + '_time.png', dpi=250, bbox_inches='tight')

    # Plot % Accepted Requests over time
    fig5 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_requests)
    plt.xlabel("Episode")
    plt.ylabel("Accepted Requests (%)")
    plt.title("Avg Number of Accepted Requests during the episode")
    plt.savefig(figName + '_requests.png', dpi=250, bbox_inches='tight')

    # Plot % Accepted Requests over time
    fig6 = plt.figure(figsize=(10, 5))
    requests_smoothed = pd.Series(stats.episode_requests).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(requests_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Accepted Requests (%)")
    plt.title("Avg Number of Accepted Requests during the episode")
    plt.savefig(figName + '_requests_v2.png', dpi=250, bbox_inches='tight')

    print(requests_smoothed)
    print(cost_smoothed)

    # return fig1, fig2, fig3, fig4

def plot_latency_stats(figName, stats, smoothing_window=10):
    # Plot the latency RL over time
    fig1 = plt.figure(figsize=(10, 5))
    latency_rl_smoothed = pd.Series(stats.latency_rl).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(latency_rl_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Latency (ms)")
    plt.title("Avg. Latency provided by the Agent over Time (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(figName + '_latency_rl.png', dpi=250, bbox_inches='tight')

    # Plot the latency MILP over time
    fig2 = plt.figure(figsize=(10, 5))
    latency_milp_smoothed = pd.Series(stats.latency_milp).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(latency_milp_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Latency (ms)")
    plt.title("Avg. Latency provided by the MILP model over Time (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(figName + '_latency_milp.png', dpi=250, bbox_inches='tight')

    # Plot latency diff over time
    fig3 = plt.figure(figsize=(10, 5))
    lat_smoothed = pd.Series(stats.latency_diff).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(lat_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Latency Diff between Agent vs MILP (%)")
    plt.title("Avg Latency Diff over Time (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(figName + '_latency_diff.png', dpi=250, bbox_inches='tight')

    # return fig1, fig2, fig3

def plot_ilp_stats(figName, stats):
    # Plot execution time over time
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(stats.execution_time)
    plt.xlabel("Number of User Requests")
    plt.ylabel("Execution Time (s)")
    plt.title("MILP execution Time")
    plt.savefig(figName + '_time.png', dpi=250, bbox_inches='tight')

    #return fig1