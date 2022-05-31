import matplotlib.pyplot as plt
import numpy as np

from soulsai.core.utils import running_mean, running_std


def save_plots(episodes_rewards, episodes_steps, iudex_hp, wins, path):
    t = np.arange(len(episodes_rewards))
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    reward_mean = running_mean(episodes_rewards, 50)
    reward_std = np.sqrt(running_std(episodes_rewards, 50))
    ax[0, 0].plot(t, reward_mean)
    ax[0, 0].fill_between(t, reward_mean - reward_std, reward_mean + reward_std, alpha=0.4)
    ax[0, 0].legend(["Mean episode reward", "Std deviation episode reward"])
    ax[0, 0].set_title("Total reward vs Episodes")
    ax[0, 0].set_xlabel("Episodes")
    ax[0, 0].set_ylabel("Total reward")
    ax[0, 0].grid(alpha=0.3)
    if len(t) >= 50:
        lim = [min(reward_mean - reward_std) - 100, max(reward_mean + reward_std) + 100]
        ax[0, 0].set_ylim(lim)

    steps_mean = running_mean(episodes_steps, 50)
    steps_std = np.sqrt(running_std(episodes_steps, 50))
    ax[0, 1].plot(t, steps_mean)
    ax[0, 1].fill_between(t, steps_mean - steps_std, steps_mean + steps_std, alpha=0.4)
    ax[0, 1].legend(["Mean episode steps", "Std deviation episode steps"])
    ax[0, 1].set_title("Number of steps vs Episodes")
    ax[0, 1].set_xlabel("Episodes")
    ax[0, 1].set_ylabel("Number of steps")
    ax[0, 1].grid(alpha=0.3)
    if len(t) >= 50:
        ax[0, 1].set_ylim([min(steps_mean - steps_std) - 100, max(steps_mean + steps_std) + 100])

    hp_mean = running_mean(iudex_hp, 50)
    hp_std = np.sqrt(running_std(iudex_hp, 50))
    ax[1, 0].plot(t, hp_mean)
    ax[1, 0].fill_between(t, hp_mean - hp_std, hp_mean + hp_std, alpha=0.4)
    ax[1, 0].legend(["Mean Iudex HP", "Std deviation Iudex HP"])
    ax[1, 0].set_title("Iudex HP vs Episodes")
    ax[1, 0].set_xlabel("Episodes")
    ax[1, 0].set_ylabel("Iudex HP")
    ax[1, 0].set_ylim([0, 1100])
    ax[1, 0].grid(alpha=0.3)

    wins = np.array(wins, dtype=np.float64)
    wins_mean = running_mean(wins, 50)
    wins_std = np.sqrt(running_std(wins, 50))
    ax[1, 1].plot(t, wins_mean)
    ax[1, 1].fill_between(t, wins_mean - wins_std, wins_mean + wins_std, alpha=0.4)
    ax[1, 1].legend(["Mean wins", "Std deviation wins"])
    ax[1, 1].set_title("Success rate vs Episodes")
    ax[1, 1].set_xlabel("Episodes")
    ax[1, 1].set_ylabel("Success rate")
    ax[1, 1].set_ylim([0, 1])
    ax[1, 1].grid(alpha=0.3)

    fig.savefig(path)
    plt.close(fig)
