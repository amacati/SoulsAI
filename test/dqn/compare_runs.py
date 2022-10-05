from pathlib import Path
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt


def plot_comparison(names, results):
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("SoulsAI Multi-Run Comparison")
    nepisodes = len(results[0]["run0"]["steps"])
    t = np.arange(nepisodes)
    
    # Plot settings
    ax[0, 0].set_title("Total reward vs Episodes")
    ax[0, 0].set_xlabel("Episodes")
    ax[0, 0].set_ylabel("Total reward")
    ax[0, 0].grid(alpha=0.3)
    ax[0, 0].set_ylim([-350, 350])
    ax[0, 1].set_title("Number of steps vs Episodes")
    ax[0, 1].set_xlabel("Episodes")
    ax[0, 1].set_ylabel("Number of steps")
    ax[0, 1].grid(alpha=0.3)
    ax[0, 1].set_ylim([0, 1100])
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

    for name, result in zip(names, results):
        rewards = np.array([result[run]["rewards"] for run in result])
        reward_mean = np.mean(rewards, axis=0)
        reward_std = np.std(rewards, axis=0)
        ax[0, 0].plot(t, reward_mean, label="Mean reward " + name)
        # ax[0, 0].fill_between(t, reward_mean - reward_std, reward_mean + reward_std, alpha=0.4)

        steps = np.array([result[run]["steps"] for run in result])
        steps_mean = np.mean(steps, axis=0)
        steps_std = np.std(steps, axis=0)
        ax[0, 1].plot(t, steps_mean, label="Mean steps " + name)
        # ax[0, 1].fill_between(t, steps_mean - steps_std, steps_mean + steps_std, alpha=0.4)

        wins = np.array([result[run]["wins"] for run in result], dtype=np.float64)
        wins_mean = np.mean(wins, axis=0)
        wins_std = np.std(wins, axis=0)
        ax[1, 1].plot(t, wins_mean, label="Mean wins " + name)
        # ax[1, 1].fill_between(t, wins_mean - wins_std, wins_mean + wins_std, alpha=0.4)
    # Plot legends
    if results[0]["run0"]["eps"][0] is None:
        ax[0, 1].legend()
    else:
        secax_y = ax[0, 1].twinx()
        for name, result in zip(names, results):
            secax_y.plot(t, result["run0"]["eps"], label="ε " + name)
        secax_y.set_ylim([-0.05, 1.05])
        secax_y.set_ylabel("Fraction of random actions")
        lines, labels = ax[0, 1].get_legend_handles_labels()
        lines2, labels2 = secax_y.get_legend_handles_labels()
        secax_y.legend(lines + lines2, labels + labels2)
    ax[0, 0].legend()
    ax[1, 1].legend()
    plt.savefig("comparison.png")


def main(folder_names):
    # Load results
    results = []
    save_root = Path(__file__).parents[2] / "saves"
    for folder in folder_names:
        save_dir = save_root / folder
        if not save_dir.exists():
            raise RuntimeError(f"Specified folder path {save_dir} does not exist")
        with open(save_dir / "SoulsAIStats.json", "r") as f:
            results.append(json.load(f))
    # Display results
    plot_comparison(folder_names, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('folders', nargs="+")
    args = parser.parse_args()
    main(args.folders)
