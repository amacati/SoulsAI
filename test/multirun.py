"""The multirun script enables multiple runs of the same training parameters on simple environments.

It is useful to check the performance of algorithms and parameters with statistical significance
using the ``SoulsAI`` framework.
"""
from __future__ import annotations

import time
import shutil
import json
import argparse
from typing import TYPE_CHECKING
from pathlib import Path
from subprocess import Popen

import docker
import numpy as np
import matplotlib.pyplot as plt
from redis import Redis

from soulsai.utils import mkdir_date, load_redis_secret, running_mean

if TYPE_CHECKING:
    from docker.client import DockerClient


def launch_training(dock: DockerClient, algorithm: str, n_clients: int, profile: str) -> Popen:
    """Launch a training run with docker compose by opening a shell process with subprocesses.

    Args:
        dock: Docker client interface.
        algorithm: Training algorithm. Either ``dqn`` or ``ppo``.
        n_clients: Number of training clients.
        profile: Docker compose profile. If `monitoring` is chosen, also starts the monitoring
            containers.

    Returns:
        The handle to the subprocess.
    """
    path = Path(__file__).parent / algorithm
    profile_cmd = "--profile " + profile if profile else ""
    cmd = f"(cd {path}; docker compose {profile_cmd} up --scale client_node={n_clients})"
    p = Popen(cmd, shell=True)  # Yes, this is hacky af. It works though
    while not dock.containers.list(filters={"name": "client_node"}):
        time.sleep(0.1)
    return p


def shutdown_nodes(dock: DockerClient):
    """Kill all active docker nodes.

    Args:
        dock: Docker client interface.
    """
    if dock.containers.list():
        if dock.containers.list(filters={"name": "redis"}):
            publish_shutdown_cmd()
            while dock.containers.list(filters={"name": ["telemetry_node", "training_node"]}):
                time.sleep(1)
        while dock.containers.list():
            for container in dock.containers.list():
                container.kill()
            time.sleep(1)


def publish_shutdown_cmd():
    """Send the shutdown command to all active client nodes."""
    redis_secret_path = Path(__file__).parents[1] / "config/secrets/redis.secret"
    secret = load_redis_secret(redis_secret_path)
    red = Redis(host="localhost", port=6379, password=secret, db=0, decode_responses=True)
    red.publish("shutdown", 1)


def check_training_done(dock: DockerClient) -> bool:
    """Check if all Docker client nodes have shut down.

    Args:
        dock: Docker client interface.

    Returns:
        True if all nodes have shut down, else false.
    """
    if dock.containers.list(filters={"name": "client_node"}):
        return False
    return True


def save_plots(results: dict, path: Path):
    """Plot the results and save the figure into the summary save folder.

    Args:
        results: A dictionary with the averaged data from all runs.
        path: The save folder path.
    """
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("SoulsAI Multi-Run Dashboard")

    x = results["n_env_steps"]
    smoothing_window_size = max(int(len(x) * 0.01), 1)
    rewards_mean = running_mean(results["rewards_mean"], smoothing_window_size)
    rewards_std = running_mean(results["rewards_std"], smoothing_window_size)
    ax[0, 0].plot(x, rewards_mean)
    lower, upper = rewards_mean - rewards_std, rewards_mean + rewards_std
    ax[0, 0].fill_between(x, lower, upper, alpha=0.4)
    ax[0, 0].legend(["Mean episode reward", "Std deviation episode reward"])
    ax[0, 0].set_title("Total reward vs Episodes")
    ax[0, 0].set_xlabel("Episodes")
    ax[0, 0].set_ylabel("Total reward")
    ax[0, 0].grid(alpha=0.3)
    lower_ylim = np.min(lower) - abs(np.min(lower)) * 0.1
    upper_ylim = np.max(upper) + abs(np.max(upper)) * 0.1
    ax[0, 0].set_ylim([lower_ylim, upper_ylim])

    steps_mean = running_mean(results["steps_mean"], smoothing_window_size)
    steps_std = running_mean(results["steps_std"], smoothing_window_size)
    ax[0, 1].plot(x, steps_mean, label="Mean episode steps")
    lower, upper = steps_mean - steps_std, steps_mean + steps_std
    ax[0, 1].fill_between(x, lower, upper, alpha=0.4, label="Std deviation episode steps")
    ax[0, 1].legend()
    ax[0, 1].set_title("Number of steps vs Episodes")
    ax[0, 1].set_xlabel("Episodes")
    ax[0, 1].set_ylabel("Number of steps")
    ax[0, 1].grid(alpha=0.3)
    lower_ylim = np.min(lower) - abs(np.min(lower)) * 0.1
    upper_ylim = np.max(upper) + abs(np.max(upper)) * 0.1
    ax[0, 1].set_ylim([lower_ylim, upper_ylim])

    ax[1, 0].set_title("N/A")
    ax[1, 0].set_xlabel("N/A")
    ax[1, 0].set_ylabel("N/A")
    ax[1, 0].grid(alpha=0.3)

    wins_mean = running_mean(results["wins_mean"], smoothing_window_size)
    wins_std = running_mean(results["wins_std"], smoothing_window_size)
    ax[1, 1].plot(x, wins_mean)
    ax[1, 1].fill_between(x, wins_mean - wins_std, wins_mean + wins_std, alpha=0.4)
    ax[1, 1].legend(["Mean wins", "Std deviation wins"])
    ax[1, 1].set_title("Success rate vs Episodes")
    ax[1, 1].set_xlabel("Episodes")
    ax[1, 1].set_ylabel("Success rate")
    ax[1, 1].set_ylim([0, 1])
    ax[1, 1].grid(alpha=0.3)

    fig.savefig(path)
    plt.close(fig)


def average_results(results: dict) -> dict:
    """Average the results from multiple runs into a single metric with mean and std deviation.

    Args:
        results: A dictionary of result summaries from each run.

    Returns:
        The averaged datapoints.
    """
    # Calculate min stats
    nsamples = min([max(results[run]["n_env_steps"]) for run in results])
    x = np.linspace(0, nsamples, 1000)
    # Interpolate data with N_episodes datapoints between 0 and nsamples
    # Restrict results to the shortest experiment length
    results = results.copy()
    averaged_results = {}
    for run in results:
        for key in ("rewards", "steps", "wins"):
            results[run][key] = np.interp(x, results[run]["n_env_steps"], results[run][key])
    for key in ("rewards", "steps", "wins"):
        data = np.array([run[key] for run in results.values()])
        averaged_results[key + "_mean"] = np.mean(data, axis=0)
        averaged_results[key + "_std"] = np.std(data, axis=0)
    averaged_results["n_env_steps"] = x
    return averaged_results


def main(args: argparse.Namespace):
    """Execute multiple simulated runs of the same configuration for statistical validation.

    At the end of all runs, the results are averaged and written to an additional save folder.

    Args:
        args: Namespace config for the experiments.
    """
    dock = docker.from_env()
    # Check if containers still running, kill them
    shutdown_nodes(dock)
    # Spawn containers
    for i in range(args.nruns):
        print(f"Launching job {i+1}")
        train_process = launch_training(dock, args.algorithm, args.nclients, args.profile)
        while not check_training_done(dock):
            time.sleep(1)
        time.sleep(3)  # Give telemetry node time to process the latest samples
        train_process.kill()
        shutdown_nodes(dock)
        time.sleep(2)
    # Summarize results in multirun experiment save
    if args.nruns:
        save_root = Path(__file__).parents[1] / "saves"
        save_dirs = [d for d in save_root.iterdir() if d.is_dir() and d.name[:4].isdigit()]
        run_dirs = sorted(save_dirs)[-args.nruns:]
        save_path = mkdir_date(save_root)
        save_path = save_path.rename(save_path.parent / ("multirun_" + save_path.name))
        # Copy config from first run, save stats into dictionary, create joint results plot
        shutil.copyfile(run_dirs[0] / "config.json", save_path / "config.json")
        results = {}
        for i, run_dir in enumerate(run_dirs):
            with open(run_dir / "telemetry.json", "r") as f:
                results["run" + str(i)] = json.load(f)
        with open(save_path / "telemetry.json", "w") as f:
            json.dump(results, f)
        results = average_results(results)
        with open(save_path / "AveragedStats.json", "w") as f:
            json.dump({key: list(value) for key, value in results.items()}, f)
        save_plots(results, save_path / "SoulsAIDashboard.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm',
                        type=str,
                        help='Training algorithm',
                        choices=["ppo", "dqn", "dqn_atari"])
    parser.add_argument('nruns', type=int, help='Number of training runs')
    parser.add_argument('nclients', type=int, default=1, help='Number of client nodes')
    parser.add_argument('--profile',
                        type=str,
                        default="",
                        help='Docker compose profile',
                        required=False)
    args = parser.parse_args()
    main(args)
