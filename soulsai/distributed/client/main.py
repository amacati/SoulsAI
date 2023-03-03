"""This script is the main function for starting the sampling client.

Clients read the local configuration, connect to the training server, and download the current
training configuration. Clients then enter the main training loop and start to generate samples from
the environment.

Note:
    If you train on SoulsGym, the player has to be loaded into the game in order to start training.
    Also remember to not interact with your keyboard in any way during training with SoulsGym.

Example:
    To start training on a client, make sure you have configured the correct Redis address in the
    config files, and the server is running. You can then start the training by running the script
    from the package root folder:

        $ python soulsai/distributed/client/main.py
"""
from pathlib import Path
import logging
from typing import Tuple, List

import torch.multiprocessing as mp
import numpy as np
import soulsgym  # noqa: F401, needs to register SoulsGym envs with gym module

from soulsai.distributed.client.dqn_client import dqn_client
from soulsai.distributed.client.ppo_client import ppo_client
from soulsai.utils import load_config
from soulsai.data.transformation import GameStateTransformer
from soulsai.exception import InvalidConfigError
from soulsai.utils import load_remote_config, load_redis_secret


def dqn_encode_sample(state: np.ndarray, action: int, reward: float, next_state: np.ndarray,
                      done: bool, info: dict) -> Tuple[List, int, float, List, bool, dict]:
    """Encode a sample for messaging with redis.

    The sample can only consist of Python types, therefore we have to convert the state arrays to
    lists.

    Args:
        state: Environment state.
        action: Chosen action.
        reward: Reward.
        next_state: Next environment state
        done: Done flag.
        info: Additional environment info.

    Returns:
        The converted sample.
    """
    return (state.tolist(), action, reward, next_state.tolist(), done, info)


def ppo_encode_sample(state: np.ndarray, action: int, action_prob: float, reward: float,
                      done: bool) -> Tuple[List, int, float, float, bool]:
    """Encode a sample for messaging with redis.

    The sample can only consist of Python types, therefore we have to convert the state arrays to
    lists.

    Args:
        state: Environment state.
        action: Chosen action.
        action_prob: Action probability.
        reward: Reward.
        done: Done flag.

    Returns:
        The converted sample.
    """
    return (state.tolist(), action, action_prob, reward, done)


def dqn_encode_tel(total_reward: float, steps: int, state: np.ndarray, eps: float) -> dict:
    """Encode a telemetry data point for messaging with redis.

    The telemetry node expects dictionaries with predefined entries, so we have to provide a
    function that maps the training statistics to the telemetry format.

    Args:
        total_reward: The total achieved reward in this episode.
        steps: The number of steps during this episode.
        state: The final episode state.
        eps: The current epsilon value.

    Returns:
        A dictionary with the expected telemetry keys.
    """
    return {
        "reward": total_reward,
        "steps": steps,
        "boss_hp": float(state[2]),
        "win": bool(state[2] == 0),
        "eps": eps
    }


def ppo_encode_tel(total_reward: float, steps: int, state: np.ndarray) -> dict:
    """Encode a telemetry data point for messaging with redis.

    The telemetry node expects dictionaries with predefined entries, so we have to provide a
    function that maps the message to the telemetry format.

    Args:
        total_reward: The total episode reward.
        steps: The total episode steps.
        state: The last episode state. Unused in this case.

    Returns:
        A dictionary with the expected telemetry keys.
    """
    return {
        "reward": total_reward,
        "steps": steps,
        "boss_hp": float(state[2]),
        "win": bool(state[2] == 0),
        "eps": 0
    }


if __name__ == "__main__":
    mp.set_start_method("spawn")
    logging.basicConfig()
    node_dir = Path(__file__).parents[3] / "config"
    config = load_config(node_dir / "config_d.yaml", node_dir / "config.yaml")
    secret = load_redis_secret(Path(__file__).parents[3] / "config" / "redis.secret")
    config = load_remote_config(config.redis_address, secret)
    tf_transformer = GameStateTransformer()
    if config.algorithm.lower() == "dqn":
        dqn_client(config,
                   tf_state_callback=tf_transformer.transform,
                   encode_sample=dqn_encode_sample,
                   encode_tel=dqn_encode_tel,
                   episode_end_callback=tf_transformer.reset)
    elif config.algorithm.lower() == "ppo":
        ppo_client(config,
                   tf_state_callback=tf_transformer.transform,
                   encode_sample=ppo_encode_sample,
                   encode_tel=ppo_encode_tel,
                   episode_end_callback=tf_transformer.reset)
    else:
        raise InvalidConfigError(f"Algorithm type {config.algorithm} is not supported")
