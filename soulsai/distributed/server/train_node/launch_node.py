import logging
from pathlib import Path
from types import SimpleNamespace

import yaml

from soulsai.distributed.server.train_node.training_node import TrainingNode
from soulsai.utils import load_config

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    config_dir = Path(__file__).parents[4] / "config"
    config = load_config(config_dir / "config_d.yaml", config_dir / "config.yaml")
    path = Path(__file__).parent / "save" / "training_node.log"
    path.parent.mkdir(exist_ok=True)
    logging.basicConfig(level=config.loglevel)

    training_node = TrainingNode(config)
    if config.fill_buffer:
        training_node.fill_buffer()
    training_node.run()
