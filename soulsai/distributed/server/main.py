import logging

from soulsai.distributed.server.training_node import TrainingNode

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    training_node = TrainingNode()
    training_node.run()
