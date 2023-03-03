"""Launch script for the test version of the DQN training node compatible with LunarLander-v2."""
import logging
from pathlib import Path
from typing import Tuple

import numpy as np

from soulsai.distributed.server.training_node.dqn import DQNTrainingNode
from soulsai.utils import load_config

logger = logging.getLogger(__name__)


def decode_sample(sample: dict) -> Tuple[np.ndarray, int, float, np.ndarray, bool]:
    """Decode the sample data for the training node.

    Args:
        sample: The sample data.

    Returns:
        The decoded sample. The tuple contains (in that order) the state, action, reward,
        next state, done flag.
    """
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
