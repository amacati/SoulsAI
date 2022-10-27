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
    root_dir = Path(__file__).parents[1]
    config = load_config(root_dir / "common" / "config_d.yaml", root_dir / "dqn" / "config.yaml")
    logging.basicConfig(level=config.loglevel)

    training_node = DQNTrainingNode(config, decode_sample=decode_sample)
    training_node.run()
