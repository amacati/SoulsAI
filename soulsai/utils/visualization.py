"""Visualization module for creating training plots from the training statistics."""
from typing import List
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from soulsai.utils.utils import running_mean, running_std


def save_plots(samples: List,
               episodes_rewards: List,
               episodes_steps: List,
               boss_hp: List,
               wins: List,
               path: Path,
               eps: float | None = None,
               N_av: int = 50):
    """Plot and save the training progress dashboard.

    Stats are smoothed by computing the mean over a running window. Confidence intervals are given
    by computing the standard deviation over a running window.

    Args:
        samples: List of the total sample count during training.
        episodes_rewards: List of achieved rewards.
        episodes_steps: List of episode lengths.
        boss_hp: List of boss HP at the end of each episode.
        wins: List of wins.
        path: Save location for the figure.
        eps: Optional list of epsilon values during training.
        N_av: Moving average window for smoothing the plot.
    """
    # t = np.arange(len(episodes_rewards))
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("SoulsAI Dashboard")
    reward_mean = running_mean(episodes_rewards, N_av)
    reward_std = running_std(episodes_rewards, N_av)
    ax[0, 0].plot(samples, reward_mean)
    ax[0, 0].fill_between(samples, reward_mean - reward_std, reward_mean + reward_std, alpha=0.4)
    ax[0, 0].legend(["Mean episode reward", "Std dev episode reward"])
    ax[0, 0].set_title("Total Reward vs Samples")
    ax[0, 0].set_xlabel("Samples")
    ax[0, 0].set_ylabel("Total Reward")
    ax[0, 0].grid(alpha=0.3)
    if len(samples) > N_av:
        lim_low = min(reward_mean[N_av:] - reward_std[N_av:])
        lim_up = max(reward_mean[N_av:] + reward_std[N_av:])
        ax[0, 0].set_ylim([lim_low - 0.1 * abs(lim_low), lim_up + 0.1 * abs(lim_up)])

    steps_mean = running_mean(episodes_steps, N_av)
    steps_std = running_std(episodes_steps, N_av)
    ax[0, 1].plot(samples, steps_mean, label="Mean episode steps")
    lower, upper = steps_mean - steps_std, steps_mean + steps_std
    ax[0, 1].fill_between(samples, lower, upper, alpha=0.4, label="Std dev episode steps")
    if eps is None:
        ax[0, 1].legend()
    ax[0, 1].set_title("Number of Steps vs Samples")
    ax[0, 1].set_xlabel("Samples")
    ax[0, 1].set_ylabel("Number of Steps")
    ax[0, 1].grid(alpha=0.3)
    if len(samples) > N_av:
        lim_low = min(steps_mean[N_av:] - steps_std[N_av:])
        lim_up = max(steps_mean[N_av:] + steps_std[N_av:])
        ax[0, 1].set_ylim([lim_low - abs(lim_low) * 0.1, lim_up + abs(lim_up) * 0.1])

    if eps is not None:
        secax_y = ax[0, 1].twinx()
        secax_y.plot(samples, eps, "orange", label="ε")
        secax_y.set_ylim([-0.05, 1.05])
        secax_y.set_ylabel("Fraction of random actions")
        lines, labels = ax[0, 1].get_legend_handles_labels()
        lines2, labels2 = secax_y.get_legend_handles_labels()
        secax_y.legend(lines + lines2, labels + labels2)

    hp_mean = running_mean(boss_hp, N_av)
    hp_std = running_std(boss_hp, N_av)
    ax[1, 0].plot(samples, hp_mean)
    ax[1, 0].fill_between(samples, hp_mean - hp_std, hp_mean + hp_std, alpha=0.4)
    ax[1, 0].legend(["Mean boss HP", "Std dev boss HP"])
    ax[1, 0].set_title("Boss HP vs Samples")
    ax[1, 0].set_xlabel("Samples")
    ax[1, 0].set_ylabel("Boss HP")
    ax[1, 0].set_ylim([-0.05, 1.05])
    ax[1, 0].grid(alpha=0.3)

    wins = np.array(wins, dtype=np.float64)
    wins_mean = running_mean(wins, N_av)
    ax[1, 1].plot(samples, wins_mean)
    ax[1, 1].legend(["Mean wins"])
    ax[1, 1].set_title("Success Rate vs Samples")
    ax[1, 1].set_xlabel("Samples")
    ax[1, 1].set_ylabel("Success Rate")
    ax[1, 1].set_ylim([0, 1])
    ax[1, 1].grid(alpha=0.3)

    fig.savefig(path)
    plt.close(fig)
