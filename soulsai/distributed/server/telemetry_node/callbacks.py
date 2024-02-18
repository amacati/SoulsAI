from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import logging

from soulsai.utils import module_type_from_string

if TYPE_CHECKING:
    from soulsai.distributed.server.telemetry_node.telemetry_node import TelemetryNode

logger = logging.getLogger(__name__)

telemetry_callback: type[TelemetryCallback] = module_type_from_string(__name__)


class TelemetryCallback(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, telemetry_node: TelemetryNode) -> None:
        """Callback for telemetry samples.

        Args:
            sample: The sample to process.
        """
        ...


class SaveBest(TelemetryCallback):

    def __init__(self, key: str, channel: str):
        super().__init__()
        self.key = key
        self.channel = channel
        self._best = -float("inf")

    def __call__(self, telemetry_node: TelemetryNode):
        if self.key not in telemetry_node.stats:
            logger.warning(f"Callback could not find key '{self.key}' in telemetry stats")
            return
        if telemetry_node.stats[self.key][-1] > self._best:
            self._best = telemetry_node.stats[self.key][-1]
            telemetry_node.red.publish(self.channel, "")  # Notify on the save channel
