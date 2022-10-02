import subprocess
import time
import argparse
from pathlib import Path
import shutil
import json

import docker
import numpy as np
import matplotlib.pyplot as plt

from soulsai.utils import mkdir_date


def launch_training(dock, n_clients):
    subprocess.Popen(['docker', 'compose',  'up', '--scale', 'client_node=' + str(n_clients)],
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while not len(dock.containers.list(filters={"name": "dqn-client_node"})):
        time.sleep(0.1)


def kill_nodes(dock):
    if len(dock.containers.list()) != 0:
        subprocess.Popen(['docker', 'compose', 'stop'])
        try:
            while len(dock.containers.list()):
                try:
                    for c in dock.containers.list():
                        c.stop()
                except:
                    ...
                time.sleep(0.1)
        except:
            ...

def check_training_done(dock):
    if len(dock.containers.list(filters={"name": "dqn-client_node"})) > 0:
        return False
    return True


def save_plots(results, path):
    nepisodes = len(results["run0"]["steps"])
    t = np.arange(nepisodes)
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("SoulsAI Multi-Run Dashboard")
    
    rewards = np.array([results[run]["rewards"] for run in results])
    reward_mean = np.mean(rewards, axis=0)
    reward_std = np.std(rewards, axis=0)
    ax[0, 0].plot(t, reward_mean)
    ax[0, 0].fill_between(t, reward_mean - reward_std, reward_mean + reward_std, alpha=0.4)
    ax[0, 0].legend(["Mean episode reward", "Std deviation episode reward"])
    ax[0, 0].set_title("Total reward vs Episodes")
    ax[0, 0].set_xlabel("Episodes")
    ax[0, 0].set_ylabel("Total reward")
    ax[0, 0].grid(alpha=0.3)
    lim_low = max(min(reward_mean) - 100, -300)
    lim_up = max(reward_mean) + 100
    ax[0, 0].set_ylim([lim_low, lim_up])

    steps = np.array([results[run]["steps"] for run in results])
    steps_mean = np.mean(steps, axis=0)
    steps_std = np.std(steps, axis=0)
    ax[0, 1].plot(t, steps_mean, label="Mean episode steps")
    lower, upper = steps_mean - steps_std, steps_mean + steps_std
    ax[0, 1].fill_between(t, lower, upper, alpha=0.4, label="Std deviation episode steps")
    if results["run0"]["eps"][0] is None:
        ax[0, 1].legend()
    ax[0, 1].set_title("Number of steps vs Episodes")
    ax[0, 1].set_xlabel("Episodes")
    ax[0, 1].set_ylabel("Number of steps")
    ax[0, 1].grid(alpha=0.3)
    lim_low = max(min(steps_mean - steps_std) - 100, 0)
    lim_up = max(steps_mean + steps_std) + 100
    ax[0, 1].set_ylim([lim_low, lim_up])

    if results["run0"]["eps"][0] is not None:
        secax_y = ax[0, 1].twinx()
        secax_y.plot(t, results["run0"]["eps"], "orange", label="ε")
        secax_y.set_ylim([-0.05, 1.05])
        secax_y.set_ylabel("Fraction of random actions")
        lines, labels = ax[0, 1].get_legend_handles_labels()
        lines2, labels2 = secax_y.get_legend_handles_labels()
        secax_y.legend(lines + lines2, labels + labels2)

    hp_mean = np.zeros_like(t)
    hp_std = np.zeros_like(t)
    ax[1, 0].plot(t, hp_mean)
    ax[1, 0].fill_between(t, hp_mean - hp_std, hp_mean + hp_std, alpha=0.4)
    ax[1, 0].legend(["N/A", "N/A"])
    ax[1, 0].set_title("N/A")
    ax[1, 0].set_xlabel("Episodes")
    ax[1, 0].set_ylabel("N/A")
    ax[1, 0].set_ylim([0, 1100])
    ax[1, 0].grid(alpha=0.3)

    wins = np.array([results[run]["wins"] for run in results], dtype=np.float64)
    wins_mean = np.mean(wins, axis=0)
    wins_std = np.std(wins, axis=0)
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


def main(args):
    dock = docker.from_env()
    # Check if containers still running, kill them
    kill_nodes(dock)
    # Spawn containers
    for i in range(args.nruns):
        print(f"Launching job {i+1}")
        launch_training(dock, args.nclients)
        while not check_training_done(dock):
            time.sleep(1)
        time.sleep(2)  # Give telemetry node time to process the latest samples
        kill_nodes(dock)
    # Summarize results in multirun experiment save
    if args.nruns:
        save_root = Path(__file__).parents[2] / "saves"
        save_dirs = [d for d in save_root.iterdir() if d.is_dir() and d.name[:4].isdigit()]
        run_dirs = sorted(save_dirs)[-args.nruns:]
        save_path = mkdir_date(save_root)
        save_path = save_path.rename(save_path.parent / ("multirun_" + save_path.name))
        # Copy config from first run, save stats into dictionary, create joint results plot
        shutil.copyfile(run_dirs[0] / "config.json", save_path / "config.json")
        results = {}
        for i, run_dir in enumerate(run_dirs):
            with open(run_dir / "SoulsAIStats.json", "r") as f:
                results["run" + str(i)] = json.load(f)
        with open(save_path / "SoulsAIStats.json", "w") as f:
            json.dump(results, f)
        save_plots(results, save_path / "SoulsAIDashboard.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('nruns', type=int, help='Number of training runs')
    parser.add_argument('nclients', type=int, default=1, help='Number of client nodes')
    args = parser.parse_args()
    main(args)
