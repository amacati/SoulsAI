import logging

import redis

from training_node import TrainingNode

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    redtest = redis.StrictRedis()
    assert redtest.ping(), "Redis service not running"
    training_node = TrainingNode()
    training_node.run()
