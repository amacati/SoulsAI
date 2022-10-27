import logging
from pathlib import Path

import numpy as np

from soulsai.distributed.server.train_node.ppo import PPOTrainingNode
from soulsai.utils import load_config

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    root_dir = Path(__file__).parents[1]
    config = load_config(root_dir / "common" / "config_d.yaml", root_dir / "ppo" / "config.yaml")
    logging.basicConfig(level=config.loglevel)

    def decode_sample(sample):
        sample = sample.get("sample")
        sample[0] = np.array(sample[0])
        return sample

    training_node = PPOTrainingNode(config, decode_sample=decode_sample)
    training_node.run()
