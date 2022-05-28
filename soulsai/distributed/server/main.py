import logging

import redis

from soulsai.distributed.server.training_node import TrainingNode

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    redtest = redis.StrictRedis(host="redis", port=6379)
    assert redtest.ping(), "Redis service not running"
    training_node = TrainingNode()
    training_node.run()
