"""Launch script for the test version of the PPO training client compatible with LunarLander-v2."""
from pathlib import Path
from typing import Tuple, List

import numpy as np

from soulsai.distributed.client.ppo_client import ppo_client
from soulsai.utils import load_config


def encode_tel(total_reward: float, steps: int, state: np.ndarray) -> dict:
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
        "boss_hp": 0,
        "win": bool(total_reward > 200),  # Avoid np.bool_
        "eps": 0
    }


def encode_sample(state: np.ndarray, action: int, action_prob: float, reward: float,
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


if __name__ == "__main__":
    root_dir = Path(__file__).parents[1]
    config = load_config(root_dir / "common" / "config_d.yaml", root_dir / "ppo" / "config.yaml")
    ppo_client(config,
               tf_state_callback=lambda x: x,
               encode_sample=encode_sample,
               encode_tel=encode_tel)
