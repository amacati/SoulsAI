"""Launch script for the test version of the PPO training client compatible with LunarLander-v2."""
from pathlib import Path
from typing import Tuple, List, Any

import numpy as np

from soulsai.distributed.client.ppo_client import ppo_client
from soulsai.utils import load_config

SampleMsg = Tuple[str, int, str, int, List]
TelMsg = Tuple[str, float, int, bool]


def tel_callback(total_reward: float, steps: int, state: Any) -> Tuple[float, int, bool]:
    """Generate the telemetry data from the total reward, the steps, and the last state.

    Args:
        total_reward: The total episode reward.
        steps: The total episode steps.
        state: The last episode state. Unused in this case.

    Returns:
        A tuple of the total reward, the steps, and a flag denoting if the environment is considered
        solved.
    """
    return total_reward, steps, bool(total_reward > 200)


def encode_tel(msg: TelMsg) -> dict:
    """Encode a telemetry data point for messaging with redis.

    The telemetry node expects dictionaries with predefined entries, so we have to provide a
    function that maps the message to the telemetry format.

    Args:
        msg: The original telemetry containing (in this order) the message type, the episode reward,
            the episode steps, and the win flag.

    Returns:
        A dictionary with the expected telemetry keys.
    """
    return {"reward": msg[1], "steps": msg[2], "boss_hp": 0, "win": msg[3], "eps": 0}


def encode_sample(msg: SampleMsg) -> Tuple[np.ndarray, int, float, float, bool]:
    """Encode a sample for messaging with redis.

    The sample can only consist of Python types, therefore we have to convert the state arrays to
    lists.

    Args:
        msg: The original message containing (in this order) the message type, the client ID, the
            trajectory ID, the step ID, and the actual sample.

    Returns:
        The converted message.
    """
    msg = msg[4]
    msg[0] = np.float64(msg[0])
    return [list(msg[0]), msg[1], msg[2], msg[3], msg[4]]


if __name__ == "__main__":
    root_dir = Path(__file__).parents[1]
    config = load_config(root_dir / "common" / "config_d.yaml", root_dir / "ppo" / "config.yaml")
    ppo_client(config, tf_state_callback=lambda x: x, tel_callback=tel_callback,
               encode_sample=encode_sample, encode_tel=encode_tel)
