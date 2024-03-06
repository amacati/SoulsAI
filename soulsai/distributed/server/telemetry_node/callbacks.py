"""Module for telemetry callbacks.

These callbacks are used to perform special actions when the telemetry node has been updated.
"""
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
    """Abstract class for telemetry callbacks.

    Callbacks are used to trigger actions when the telemetry node has been updated.
    """

    def __init__(self):
        """Initialize the telemetry callback."""
        super().__init__()

    @abstractmethod
    def __call__(self, telemetry_node: TelemetryNode):
        """Execute the callback for the telemetry node.

        Args:
            sample: The sample to process.
        """


class SaveBest(TelemetryCallback):
    """Callback to send a training checkpoint message via Redis when a new best value is found.

    This callback is useful to checkpoint the best model iteration during training based on the best
    value of a given key in the telemetry stats.
    """

    def __init__(self, key: str, channel: str):
        """Initialize the save best callback.

        Args:
            key: The key of the value to check.
            channel: The redis channel to send the save message to.
        """
        super().__init__()
        self.key = key
        self.channel = channel
        self._best = -float("inf")

    def __call__(self, telemetry_node: TelemetryNode):
        """Check if the value of `self.key` is the best so far and send a save message accordingly.

        Args:
            telemetry_node: The telemetry node. Used to easily access the stats.
        """
        if self.key not in telemetry_node.stats:
            logger.warning(f"Callback could not find key '{self.key}' in telemetry stats")
            return
        if telemetry_node.stats[self.key][-1] > self._best:
            self._best = telemetry_node.stats[self.key][-1]
            telemetry_node.red.publish(self.channel, "")  # Notify on the save channel
