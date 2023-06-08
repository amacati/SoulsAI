"""Launch script for the test version of the PPO training node compatible with LunarLander-v2."""
import logging
from pathlib import Path

from soulsai.distributed.server.training_node.ppo import PPOTrainingNode
from soulsai.utils import load_config

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    root_dir = Path(__file__).parents[1]
    config = load_config(root_dir / "common" / "config_d.yaml", root_dir / "ppo" / "config.yaml")
    logging.basicConfig(level=config.loglevel)
    training_node = PPOTrainingNode(config)
    training_node.run()
