import logging
from pathlib import Path

from soulsai.distributed.server.telemetry_node.telemetry_node import TelemetryNode
from soulsai.utils import load_config, load_remote_config, load_redis_secret

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    config_dir = Path(__file__).parents[4] / "config"
    config = load_config(config_dir / "config_d.yaml", config_dir / "config.yaml")
    secret = load_redis_secret(config_dir / "redis.secret")
    load_remote_config(config.address, secret)
    logging.basicConfig(level=config.loglevel)
    logger.setLevel(config.loglevel)
    telemetry_node = TelemetryNode(config)
    telemetry_node.run()
