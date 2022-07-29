import logging
from pathlib import Path
from types import SimpleNamespace

import yaml

from test.distributed.dqn.server.train_node.training_node import TrainingNode

logger = logging.getLogger(__name__)


def load_config():
    root_dir = Path(__file__).parent
    with open(root_dir / "config_d.yaml", "r") as f:
        config = yaml.safe_load(f)
    if (root_dir / "config.yaml").is_file():
        with open(root_dir / "config.yaml", "r") as f:
            _config = yaml.safe_load(f)
        if _config is not None:
            config |= _config  # Overwrite default config with keys from user config
    loglvl = config["loglevel"].lower()
    if loglvl == "debug":
        config["loglevel"] = logging.DEBUG
    elif loglvl == "info":
        config["loglevel"] = logging.INFO
    elif loglvl == "warning":
        config["loglevel"] = logging.WARNING
    elif loglvl == "error":
        config["loglevel"] = logging.ERROR
    else:
        raise RuntimeError(f"Loglevel {config['loglevel']} in config not supported!")
    return SimpleNamespace(**config)


if __name__ == "__main__":
    config = load_config()
    logging.basicConfig(level=config.loglevel)
    training_node = TrainingNode()
    if config.fill_buffer:
        training_node.fill_buffer()
    training_node.run()
