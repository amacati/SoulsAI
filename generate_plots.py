from pathlib import Path

import json
import numpy as np
import matplotlib.pyplot as plt

from utils import running_average

if __name__ == "__main__":
    root = Path(__file__).parent
    save = root / "saves" / "results.json"
    if save.exists():
        with open(save, "r") as f:
            results = json.load(f)
        max_len = max([len(x) for x in results["episodes_rewards"]])
        episodes_rewards = np.zeros((len(results["episodes_rewards"]), max_len))
        episodes_steps = np.zeros((len(results["episodes_steps"]), max_len))
        for i in range(len(results["episodes_rewards"])):
            reward = results["episodes_rewards"][i]
            steps = results["episodes_steps"][i]
            episodes_rewards[i, :len(reward)] = reward
            episodes_rewards[i, len(reward):] = running_average(reward, 50)[-1]
            episodes_steps[i, :len(steps)] = steps
            episodes_steps[i, len(steps):] = running_average(steps, 50)[-1]

        reward_mean = running_average(np.mean(episodes_rewards, axis=0), 50)
        reward_std = running_average(np.std(episodes_rewards, axis=0), 50)
        steps_mean = running_average(np.mean(episodes_steps, axis=0), 50)
        steps_std = running_average(np.std(episodes_steps, axis=0), 50)
        t = range(len(reward_mean))
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
        ax[0].plot(t, reward_mean, label='Avg. mean episode reward')
        ax[0].fill_between(t, reward_mean - reward_std, reward_mean + reward_std, alpha=0.2,
                           label="Standard deviation")
        ax[0].set_xlabel('Episodes')
        ax[0].set_ylabel('Total mean reward')
        ax[0].set_title('Total Mean Reward vs Episodes')
        ax[0].legend()
        ax[0].grid(alpha=0.3)
        ax[0].set_ylim([-400, 400])

        ax[1].plot(t, steps_mean, label='Avg. mean number of steps per episode')
        ax[1].fill_between(t, steps_mean - steps_std, steps_mean + steps_std, alpha=0.2,
                           label="Standard deviation")
        ax[1].set_xlabel('Episodes')
        ax[1].set_ylabel('Total mean number of steps')
        ax[1].set_title('Total number of steps vs Episodes')
        ax[1].legend()
        ax[1].grid(alpha=0.3)
        plt.show()
        fig.savefig(root / "combined_stats.png")
