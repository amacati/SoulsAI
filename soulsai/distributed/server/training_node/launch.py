"""Dockerfile entrypoint to start the train node."""
import logging
from pathlib import Path

from soulsai.distributed.server.training_node.dqn import DQNTrainingNode
from soulsai.distributed.server.training_node.ppo import PPOTrainingNode
from soulsai.distributed.server.training_node.training_node import TrainingNode
from soulsai.utils import load_config
from soulsai.exception import InvalidConfigError

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    config_dir = Path(__file__).parents[4] / "config"
    config = load_config(config_dir / "config_d.yaml", config_dir / "config.yaml")
    path = Path(__file__).parent / "save" / "training_node.log"
    path.parent.mkdir(exist_ok=True)
    logging.basicConfig(level=config.loglevel)

    if config.algorithm.lower() == "dqn":
        training_node: TrainingNode = DQNTrainingNode(config)
    elif config.algorithm.lower() == "ppo":
        training_node = PPOTrainingNode(config)
    else:
        raise InvalidConfigError(f"Algorithm {config.algorithm} specified in config"
                                 "is not supported")
    training_node.run()
