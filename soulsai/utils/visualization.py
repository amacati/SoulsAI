import matplotlib.pyplot as plt
import numpy as np

from soulsai.utils.utils import running_mean, running_std


def save_plots(episodes_rewards, episodes_steps, iudex_hp, wins, path, eps=None, N_av=50):
    t = np.arange(len(episodes_rewards))
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("SoulsAI Dashboard")
    reward_mean = running_mean(episodes_rewards, N_av)
    reward_std = np.sqrt(running_std(episodes_rewards, N_av))
    ax[0, 0].plot(t, reward_mean)
    ax[0, 0].fill_between(t, reward_mean - reward_std, reward_mean + reward_std, alpha=0.4)
    ax[0, 0].legend(["Mean episode reward", "Std deviation episode reward"])
    ax[0, 0].set_title("Total reward vs Episodes")
    ax[0, 0].set_xlabel("Episodes")
    ax[0, 0].set_ylabel("Total reward")
    ax[0, 0].grid(alpha=0.3)
    if len(t) > N_av:
        lim_low = min(reward_mean[N_av:]) - 100
        lim_up = max(reward_mean[N_av:]) + 100
        ax[0, 0].set_ylim([lim_low, lim_up])

    steps_mean = running_mean(episodes_steps, N_av)
    steps_std = np.sqrt(running_std(episodes_steps, N_av))
    ax[0, 1].plot(t, steps_mean, label="Mean episode steps")
    lower, upper = steps_mean - steps_std, steps_mean + steps_std
    ax[0, 1].fill_between(t, lower, upper, alpha=0.4, label="Std deviation episode steps")
    if eps is None:
        ax[0, 1].legend()
    ax[0, 1].set_title("Number of steps vs Episodes")
    ax[0, 1].set_xlabel("Episodes")
    ax[0, 1].set_ylabel("Number of steps")
    ax[0, 1].grid(alpha=0.3)
    if len(t) > N_av:
        lim_low = min(steps_mean[N_av:] - steps_std[N_av:]) - 100
        lim_up = max(steps_mean[N_av:] + steps_std[N_av:]) + 100
        ax[0, 1].set_ylim([lim_low, lim_up])

    if eps is not None:
        secax_y = ax[0, 1].twinx()
        secax_y.plot(t, eps, "orange", label="ε")
        secax_y.set_ylim([-0.05, 1.05])
        secax_y.set_ylabel("Fraction of random actions")
        lines, labels = ax[0, 1].get_legend_handles_labels()
        lines2, labels2 = secax_y.get_legend_handles_labels()
        secax_y.legend(lines + lines2, labels + labels2)

    hp_mean = running_mean(iudex_hp, N_av)
    hp_std = np.sqrt(running_std(iudex_hp, N_av))
    ax[1, 0].plot(t, hp_mean)
    ax[1, 0].fill_between(t, hp_mean - hp_std, hp_mean + hp_std, alpha=0.4)
    ax[1, 0].legend(["Mean Iudex HP", "Std deviation Iudex HP"])
    ax[1, 0].set_title("Iudex HP vs Episodes")
    ax[1, 0].set_xlabel("Episodes")
    ax[1, 0].set_ylabel("Iudex HP")
    ax[1, 0].set_ylim([0, 1100])
    ax[1, 0].grid(alpha=0.3)

    wins = np.array(wins, dtype=np.float64)
    wins_mean = running_mean(wins, N_av)
    wins_std = np.sqrt(running_std(wins, N_av))
    ax[1, 1].plot(t, wins_mean)
    ax[1, 1].legend(["Mean wins"])
    ax[1, 1].set_title("Success rate vs Episodes")
    ax[1, 1].set_xlabel("Episodes")
    ax[1, 1].set_ylabel("Success rate")
    ax[1, 1].set_ylim([0, 1])
    ax[1, 1].grid(alpha=0.3)

    fig.savefig(path)
    plt.close(fig)
