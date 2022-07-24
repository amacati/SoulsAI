import logging
from pathlib import Path

import yaml

from soulsai.distributed.server.train_node.training_node import TrainingNode

logger = logging.getLogger(__name__)


def load_config():
    root_dir = Path(__file__).parent
    with open(root_dir / "config_d.yaml", "r") as f:
        config = yaml.safe_load(f)
    if (root_dir / "config.yaml").is_file():
        with open(root_dir / "config.yaml", "r") as f:
            config |= yaml.safe_load(f)  # Overwrite default config with keys from user config
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


if __name__ == "__main__":
    config = load_config()
    path = Path(__file__).parent / "save" / "training_node.log"
    path.parent.mkdir(exist_ok=True)
    logging.basicConfig(level=config["loglevel"])
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)s]  %(message)s")
    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(logFormatter)
    logging.getLogger().addHandler(fileHandler)

    training_node = TrainingNode()
    training_node.run()
