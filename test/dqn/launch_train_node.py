import logging
from pathlib import Path

import numpy as np

from soulsai.distributed.server.train_node.dqn import DQNTrainingNode
from soulsai.utils import load_config

logger = logging.getLogger(__name__)


def decode_sample(sample):
    experience = sample.get("sample")
    experience[0] = np.array(experience[0])
    experience[3] = np.array(experience[3])
    return experience


if __name__ == "__main__":
    config_dir = Path(__file__).parent
    config = load_config(config_dir / "config_d.yaml", config_dir / "config.yaml")
    logging.basicConfig(level=config.loglevel)

    training_node = DQNTrainingNode(config, decode_sample=decode_sample)
    if config.fill_buffer:
        training_node.fill_buffer()
    training_node.run()
