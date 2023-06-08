"""Dockerfile entrypoint to start the train node."""
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from soulsgym.core.static import actions as soulsgym_actions

from soulsai.distributed.server.training_node.dqn import DQNTrainingNode
from soulsai.distributed.server.training_node.ppo import PPOTrainingNode
from soulsai.utils import load_config
from soulsai.exception import InvalidConfigError

logger = logging.getLogger(__name__)
n_actions = len(soulsgym_actions)

# State, action, reward, next_state, done, info
DQNSample = Tuple[np.ndarray, int, float, np.ndarray, bool, dict]
# STate, action, action prob., done, client ID, step ID
PPOSample = Tuple[np.ndarray, int, float, float, bool, int, int]


def decode_ppo_sample(sample: dict) -> PPOSample:
    """Decode a PPO sample.

    Args:
        sample: Sample dictionary.

    Returns:
        The decoded sample.
    """
    trajectory_id, step_id = sample["client_id"], sample["step_id"]
    sample = sample.get("sample")
    sample[0] = np.array(sample[0])
    sample.extend([trajectory_id, step_id])


if __name__ == "__main__":
    config_dir = Path(__file__).parents[4] / "config"
    config = load_config(config_dir / "config_d.yaml", config_dir / "config.yaml")
    path = Path(__file__).parent / "save" / "training_node.log"
    path.parent.mkdir(exist_ok=True)
    logging.basicConfig(level=config.loglevel)

    if config.algorithm.lower() == "dqn":
        training_node = DQNTrainingNode(config)
    elif config.algorithm.lower() == "ppo":
        training_node = PPOTrainingNode(config)
    else:
        raise InvalidConfigError(f"Algorithm {config.algorithm} specified in config"
                                 "is not supported")
    training_node.run()
