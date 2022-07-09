import logging
from pathlib import Path

from soulsai.distributed.server.train_node.training_node import TrainingNode

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    path = Path(__file__).parent / "save" / "training_node.log"
    logging.basicConfig(level=logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)s]  %(message)s")
    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(logFormatter)
    logging.getLogger().addHandler(fileHandler)

    training_node = TrainingNode()
    training_node.run()
