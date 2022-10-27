import logging
from pathlib import Path

import numpy as np
from soulsgym.core.static import actions as soulsgym_actions

from soulsai.distributed.server.train_node.dqn import DQNTrainingNode
from soulsai.distributed.server.train_node.ppo import PPOTrainingNode
from soulsai.utils import load_config
from soulsai.exception import InvalidConfigError

logger = logging.getLogger(__name__)
n_actions = len(soulsgym_actions)


def decode_dqn_sample(sample):
    sample = sample.get("sample")
    sample[0] = np.array(sample[0])  # State
    sample[3] = np.array(sample[3])  # Next state
    return sample


def decode_ppo_sample(sample):
    sample = sample.get("sample")
    sample[0] = np.array(sample[0])  # State
    return sample


if __name__ == "__main__":
    config_dir = Path(__file__).parents[4] / "config"
    config = load_config(config_dir / "config_d.yaml", config_dir / "config.yaml")
    path = Path(__file__).parent / "save" / "training_node.log"
    path.parent.mkdir(exist_ok=True)
    logging.basicConfig(level=config.loglevel)

    if config.algorithm.lower() == "dqn":
        training_node = DQNTrainingNode(config, decode_dqn_sample)
    elif config.algorithm.lower() == "ppo":
        training_node = PPOTrainingNode(config, decode_ppo_sample)
    else:
        raise InvalidConfigError(f"Algorithm {config.algorithm} specified in config"
                                 "is not supported")
    training_node.run()
