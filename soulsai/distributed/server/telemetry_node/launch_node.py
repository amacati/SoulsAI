import logging

from soulsai.distributed.server.telemetry_node.telemetry_node import TelemetryNode

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    telemetry_node = TelemetryNode()
    telemetry_node.run()
