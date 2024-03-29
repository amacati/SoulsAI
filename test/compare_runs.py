"""Script to compare and plot the results of different experiment runs."""

import argparse
import json
import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from soulsai.utils import running_mean

logger = logging.getLogger(__name__)


def plot_comparison(names: List[str], results: dict):
    """Plot all performance graphs into a single figure with mean and standard deviation.

    Saves the figure as `comparison.png`.
    """
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("SoulsAI Multi-Run Comparison")

    reward_ymin, reward_ymax = 0, 0
    steps_ymin, steps_ymax = 0, 0
    for name, result in zip(names, results):
        x = result["n_env_steps"]
        smoothing_window_size = max(int(len(x) * 0.01), 1)

        rewards_mean = running_mean(result["rewards_mean"], smoothing_window_size)
        rewards_std = running_mean(result["rewards_std"], smoothing_window_size)
        lower, upper = rewards_mean - rewards_std, rewards_mean + rewards_std
        ax[0, 0].plot(x, rewards_mean, label="Mean reward " + name)
        ax[0, 0].fill_between(x, lower, upper, alpha=0.4)
        reward_ymin = min(reward_ymin, np.min(lower) - abs(np.min(lower)) * 0.1)
        reward_ymax = max(reward_ymax, np.max(upper) + abs(np.max(upper)) * 0.1)

        steps_mean = running_mean(result["steps_mean"], smoothing_window_size)
        steps_std = running_mean(result["steps_std"], smoothing_window_size)
        lower, upper = steps_mean - steps_std, steps_mean + steps_std
        ax[0, 1].plot(x, steps_mean, label="Mean steps " + name)
        ax[0, 1].fill_between(x, lower, upper, alpha=0.4)
        steps_ymin = min(steps_ymin, np.min(lower) - abs(np.min(lower)) * 0.1)
        steps_ymax = max(steps_ymax, np.max(upper) + abs(np.max(upper)) * 0.1)

        wins_mean = running_mean(result["wins_mean"], smoothing_window_size)
        wins_std = running_mean(result["wins_std"], smoothing_window_size)
        ax[1, 1].plot(x, wins_mean, label="Mean wins " + name)
        ax[1, 1].fill_between(x, wins_mean - wins_std, wins_mean + wins_std, alpha=0.4)

    # Plot settings
    ax[0, 0].set_title("Total reward vs Episodes")
    ax[0, 0].set_xlabel("Episodes")
    ax[0, 0].set_ylabel("Total reward")
    ax[0, 0].grid(alpha=0.3)
    ax[0, 0].set_ylim([reward_ymin, reward_ymax])
    ax[0, 1].set_title("Number of steps vs Episodes")
    ax[0, 1].set_xlabel("Episodes")
    ax[0, 1].set_ylabel("Number of steps")
    ax[0, 1].grid(alpha=0.3)
    ax[0, 1].set_ylim([steps_ymin, steps_ymax])
    ax[1, 0].legend(["N/A", "N/A"])
    ax[1, 0].set_title("N/A")
    ax[1, 0].set_xlabel("Episodes")
    ax[1, 0].set_ylabel("N/A")
    ax[1, 0].set_ylim([0, 1100])
    ax[1, 0].grid(alpha=0.3)
    ax[1, 1].set_title("Success rate vs Episodes")
    ax[1, 1].set_xlabel("Episodes")
    ax[1, 1].set_ylabel("Success rate")
    ax[1, 1].set_ylim([0, 1])
    ax[1, 1].grid(alpha=0.3)
    # Plot legends
    ax[0, 1].legend()
    ax[0, 0].legend()
    ax[1, 1].legend()
    plt.savefig("comparison.png")


def main(folder_names: List[str]):
    """Read the data from all folders and plot the performance graphs into a single figure.

    Args:
        folder_names: The folder names of all experiments. Folders have to be a valid multirun
            experiment with averaged performance data.

    Raises:
        RuntimeError: The specified folder path does not exist.
        FileNotFoundError: The folder does not contain an averaged performance save file.
    """
    # Load results
    results = []
    save_root = Path(__file__).parents[1] / "saves"
    for folder in folder_names:
        save_dir = save_root / folder
        if not save_dir.exists():
            raise RuntimeError(f"Specified folder path {save_dir} does not exist")
        try:
            with open(save_dir / "AveragedStats.json", "r") as f:
                # Load results from json, convert arrays to np arrays
                results.append({key: np.array(value) for key, value in json.load(f).items()})
        except FileNotFoundError as e:
            logger.error(
                ("Missing averaged stats file, specified folder is not a multirun " "experiment.")
            )
            raise e
    # Display results
    plot_comparison(folder_names, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="+")
    args = parser.parse_args()
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    main(args.folders)
