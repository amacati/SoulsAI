import logging
import json
from pathlib import Path
import time

import redis
import numpy as np
import matplotlib.pyplot as plt

from soulsai.utils.utils import running_mean, running_std, mkdir_date

logger = logging.getLogger(__name__)


class TelemetryNode:

    def __init__(self):
        logger.info("Telemetry node startup")
        # Read redis server secret
        with open(Path(__file__).parents[1] / "redis.secret") as f:
            conf = f.readlines()
        secret = None
        for line in conf:
            if len(line) > 12 and line[0:12] == "requirepass ":
                secret = line[12:]
                break
        if secret is None:
            raise RuntimeError("Missing password configuration for redis in redis.secret")

        self.red = redis.Redis(host='redis', port=6379, password=secret, db=0,
                               decode_responses=True)
        self.sub_telemetry = self.red.pubsub(ignore_subscribe_messages=True)
        self.sub_telemetry.subscribe("telemetry")

        self.rewards = []
        self.steps = []
        self.wins = []
        self.eps = []
        save_dir = mkdir_date(Path(__file__).parent / "save")
        self.figure_path = save_dir / "dashboard.png"
        self.stats_path = save_dir / "stats.json"
        logger.info("Telemetry node startup complete")

    def run(self):
        logger.info("Telemetry node running")
        while True:
            # read new samples
            msg = self.sub_telemetry.get_message()
            if not msg:
                time.sleep(5)
                continue
            sample = json.loads(msg["data"])

            self.rewards.append(sample["reward"])
            self.steps.append(sample["steps"])
            self.wins.append(sample["win"])
            self.eps.append(sample["eps"])

            if len(self.rewards) % 5 == 0:
                logger.info(f"Current mean reward: {running_mean(self.rewards, 50)[-1]}")
                self.update_dashboard()

    def update_dashboard(self):
        self.figure_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_plots(self.rewards, self.steps, self.wins, self.eps, self.figure_path)
        with open(self.stats_path, "w") as f:
            json.dump({"rewards": self.rewards, "steps": self.steps, "wins": self.wins,
                       "eps": self.eps}, f)
        logger.info("Dashboard updated")

    @staticmethod
    def save_plots(rewards, steps, wins, eps, path):
        N = 50
        t = np.arange(len(rewards))
        fig, ax = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("SoulsAI Dashboard")
        reward_mean = running_mean(rewards, N)
        reward_std = np.sqrt(running_std(rewards, N))
        ax[0, 0].plot(t, reward_mean)
        ax[0, 0].fill_between(t, reward_mean - reward_std, reward_mean + reward_std, alpha=0.4)
        ax[0, 0].legend(["Mean episode reward", "Std deviation episode reward"])
        ax[0, 0].set_title("Total reward vs Episodes")
        ax[0, 0].set_xlabel("Episodes")
        ax[0, 0].set_ylabel("Total reward")
        ax[0, 0].grid(alpha=0.3)
        ax[0, 0].set_ylim([-350, 350])

        steps_mean = running_mean(steps, N)
        steps_std = np.sqrt(running_std(steps, N))
        ax[0, 1].plot(t, steps_mean, label="Mean episode steps")
        lower, upper = steps_mean - steps_std, steps_mean + steps_std
        ax[0, 1].fill_between(t, lower, upper, alpha=0.4, label="Std deviation episode steps")
        ax[0, 1].legend()
        ax[0, 1].set_title("Number of steps vs Episodes")
        ax[0, 1].set_xlabel("Episodes")
        ax[0, 1].set_ylabel("Number of steps")
        ax[0, 1].grid(alpha=0.3)
        if len(t) >= N:
            lim_low = max((min(steps_mean - steps_std) - 100, 0))
            lim_up = max(steps_mean + steps_std) + 100
            ax[0, 1].set_ylim([lim_low, lim_up])

        ax[1, 0].plot(t, eps)
        ax[1, 0].legend(["Epsilon"])
        ax[1, 0].set_title("Fraction of random moves")
        ax[1, 0].set_xlabel("Episodes")
        ax[1, 0].set_ylabel("Epsilon")
        ax[1, 0].set_ylim([-0.05, 1.05])
        ax[1, 0].grid(alpha=0.3)

        wins = np.array(wins, dtype=np.float64)
        wins_mean = running_mean(wins, N)
        ax[1, 1].plot(t, wins_mean)
        ax[1, 1].legend(["Mean wins"])
        ax[1, 1].set_title("Success rate vs Episodes")
        ax[1, 1].set_xlabel("Episodes")
        ax[1, 1].set_ylabel("Success rate")
        ax[1, 1].set_ylim([-0.05, 1.05])
        ax[1, 1].grid(alpha=0.3)

        fig.savefig(path)
        plt.close(fig)
