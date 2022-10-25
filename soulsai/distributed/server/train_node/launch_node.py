import logging
from pathlib import Path

import numpy as np
from soulsgym.core.game_state import GameState
from soulsgym.core.static import actions as soulsgym_actions

from soulsai.distributed.server.train_node.dqn import DQNTrainingNode
from soulsai.distributed.server.train_node.ppo import PPOTrainingNode
from soulsai.utils import load_config
from soulsai.core.utils import gamestate2np
from soulsai.exception import InvalidConfigError

logger = logging.getLogger(__name__)
n_actions = len(soulsgym_actions)

def decode_dqn_sample(sample):
    sample = sample.get("sample")
    sample[0] = gamestate2np(GameState.from_dict(sample[0]))
    sample[3] = gamestate2np(GameState.from_dict(sample[3]))
    return sample


def decode_ppo_sample(sample):
    sample = sample.get("sample")
    sample[0] = gamestate2np(GameState.from_dict(sample[0]))
    sample[2] = np.array(sample[2])  # Decode probabilities
    return sample


if __name__ == "__main__":
    config_dir = Path(__file__).parents[4] / "config"
    config = load_config(config_dir / "config_d.yaml", config_dir / "config.yaml")
    path = Path(__file__).parent / "save" / "training_node.log"
    path.parent.mkdir(exist_ok=True)
    logging.basicConfig(level=config.loglevel)

    if config.train_algorithm == "DQN":
        training_node = DQNTrainingNode(config, decode_dqn_sample)
        if config.fill_buffer:
            training_node.fill_buffer()
    elif config.train_algorithm == "PPO":
        training_node = PPOTrainingNode(config, decode_ppo_sample)
    else:
        raise InvalidConfigError(f"Algorithm {config.train_algorithm} specified in config"
                                 "is not supported")
    training_node.run()
