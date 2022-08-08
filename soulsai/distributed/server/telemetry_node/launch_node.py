import logging
from pathlib import Path

from soulsai.distributed.server.telemetry_node.telemetry_node import TelemetryNode
from soulsai.utils import load_config

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    config_dir = Path(__file__).parents[4] / "config"
    config = load_config(config_dir / "config_d.yaml", config_dir / "config.yaml")
    logging.basicConfig(level=config.loglevel)
    logger.setLevel(config.loglevel)
    telemetry_node = TelemetryNode()
    telemetry_node.run()
