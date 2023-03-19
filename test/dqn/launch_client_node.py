"""Launch script for the test version of the DQN training client compatible with LunarLander-v2."""
from pathlib import Path
from typing import Tuple, List

import numpy as np

from soulsai.distributed.client.dqn_client import dqn_client
from soulsai.utils import load_config


def encode_tel(total_reward: float, steps: int, state: np.ndarray, eps: float) -> dict:
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
        "boss_hp": 0,
        "win": bool(total_reward > 200),
        "eps": eps
    }


def encode_sample(state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool,
                  info: dict) -> Tuple[List, int, float, List, bool, dict]:
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


if __name__ == "__main__":
    root_dir = Path(__file__).parents[1]
    config = load_config(root_dir / "common" / "config_d.yaml", root_dir / "dqn" / "config.yaml")
    dqn_client(config,
               tf_obs_callback=lambda x: x,
               encode_sample=encode_sample,
               encode_tel=encode_tel)
