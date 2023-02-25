"""Launch script for the test version of the PPO training node compatible with LunarLander-v2."""
import logging
from pathlib import Path
from typing import Tuple

import numpy as np

from soulsai.distributed.server.train_node.ppo import PPOTrainingNode
from soulsai.utils import load_config

logger = logging.getLogger(__name__)


def decode_sample(sample: dict) -> Tuple[np.ndarray, int, float, float, bool, int, int]:
    """Decode the sample data for the training node.

    Args:
        sample: The sample data.

    Returns:
        The decoded sample. The tuple contains (in that order) the state, action, reward, done flag,
        trajectory_id, step_id.
    """
    trajectory_id, step_id = sample["client_id"], sample["step_id"]
    sample = sample.get("sample")
    sample[0] = np.array(sample[0])
    sample.extend([trajectory_id, step_id])
    return sample


if __name__ == "__main__":
    root_dir = Path(__file__).parents[1]
    config = load_config(root_dir / "common" / "config_d.yaml", root_dir / "ppo" / "config.yaml")
    logging.basicConfig(level=config.loglevel)
    training_node = PPOTrainingNode(config, decode_sample=decode_sample)
    training_node.run()
