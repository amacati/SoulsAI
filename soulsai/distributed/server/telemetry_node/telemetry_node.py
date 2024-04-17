"""The main telemetry node module."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from prometheus_client import start_http_server
from redis import Redis

from soulsai.distributed.common.serialization import deserialize
from soulsai.distributed.server.telemetry_node.callbacks import telemetry_callback
from soulsai.distributed.server.telemetry_node.connectors import (
    FileStorageConnector,
    GrafanaConnector,
    TelemetryConnector,
    WandBConnector,
)
from soulsai.distributed.server.telemetry_node.transforms import (
    TelemetryTransform,
    telemetry_transform,
)
from soulsai.utils import load_redis_secret, load_remote_config

if TYPE_CHECKING:
    from types import SimpleNamespace

logger = logging.getLogger(__name__)


class TelemetryNode:
    """The telemetry node receives telemetry from the training node to track the training progress.

    Samples are received from Redis and added to the telemetry history. Every ``update_interval``
    message, the results are plotted and saved both to a figure and to a json file.

    If live monitoring is enabled, the node also uses a :class:`~.GrafanaConnector` to expose the
    data to a ``Grafana`` instance running within the same network.
    """

    def __init__(self, config: SimpleNamespace):
        """Initialize the connection to Redis and Grafana (if enabled).

        Args:
            config: Training configuration.
        """
        logger.info("Telemetry node startup")

        # Read redis server secret and connect to Redis
        secret = load_redis_secret(Path("/run/secrets/redis_secret"))
        self.red = Redis(host="redis", port=6379, password=secret, db=0)
        self.config = load_remote_config(config.redis_address, secret, self.red)

        # Initialize the subscriber for telemetry information from the training node
        self.sub_telemetry = self.red.pubsub(ignore_subscribe_messages=True)
        self.sub_telemetry.subscribe("telemetry")

        # Listen for shutdown signal from Redis in a separate thread
        self._shutdown = False
        self.cmd_sub = self.red.pubsub(ignore_subscribe_messages=True)
        self.cmd_sub.subscribe(shutdown=self.shutdown)
        self.cmd_sub.run_in_thread(sleep_time=1.0, daemon=True)

        # Start a Prometheus server to make the telemetry node status visible in Grafana
        start_http_server(8080)

        # Add a stats dictionary and initialize the telemetry transforms list. Each time a telemetry
        # message is received, the transforms are applied to the message and the result is added to
        # the stats dictionary. The stats dictionary is then shared with the telemetry connectors.
        # Note that the keys returned by the transforms must be unique.
        self.stats = dict()
        self.telemetry_transforms = self._create_transforms()
        self.telemetry_callbacks = self._create_callbacks()

        self.connectors = self._create_connectors()
        for connector in self.connectors:
            connector.start()

        # Helper variable for saving the best model
        self._best_reward = float("-inf")

        if self.config.checkpoint.load:
            self._load_stats()
            for connector in self.connectors:
                connector.update(self.stats)
        logger.info("Telemetry node startup complete")

    def run(self):
        """Run the telemetry node.

        Receives telemetry messages from clients via Redis and adds them to the training statistics.
        The statistic lists are shared by reference with the :class:`.GrafanaConnector` (if
        enabled). An update of the lists in the main loop therefore automatically updates the stats
        for the GrafanaConnector as well.

        Note:
            Telemetry messages and therefore the stats are not guaranteed to be in order.
        """
        logger.info("Telemetry node running")
        n_episodes = 0
        while not self._shutdown:
            # read new samples
            if not (msg := self.sub_telemetry.get_message()):
                time.sleep(0.1)
                continue
            self._process_msg(msg)
            n_episodes += 1
            if n_episodes % self.config.telemetry.update_interval == 0:
                for connector in self.connectors:
                    connector.update(self.stats)
                self._log_telemetry()  # TODO: move to log connector
                if self.telemetry_callbacks:
                    for callback in self.telemetry_callbacks:
                        callback(self)
                if self.stats["rewards_av"][-1] > self._best_reward:
                    self.red.publish("save_best", "")
                    self._best_reward = self.stats["rewards_av"][-1]

    def _process_msg(self, msg: bytes):
        """Deserialize the message and add the information to the stats dict."""
        sample = deserialize(msg["data"])
        for tf in self.telemetry_transforms:
            key, value = tf(sample)
            if key not in self.stats:
                self.stats[key] = []
                self.stats[key + "_av"] = []
            self.stats[key].append(value)
            self.stats[key + "_av"].append(self._latest_moving_av(self.stats[key]))

    def _log_telemetry(self):
        if keys := self.config.telemetry.log_keys:
            info = " | ".join(f"{k}: {self.stats[k][-1]:.2f}" for k in keys)
            logger.info("Telemetry update: " + info)

    def _create_connectors(self) -> list[TelemetryConnector]:
        """Create the telemetry connectors."""
        connectors = []
        if getattr(self.config.monitoring, "file_storage", None):
            connectors.append(FileStorageConnector(self.config))
            logger.info("Initializing file storage telemetry connector")
        if getattr(self.config.monitoring, "grafana", None):
            connectors.append(GrafanaConnector(self.config))
            logger.info("Initializing Grafana telemetry connector")
        if getattr(self.config.monitoring, "wandb", None):
            connectors.append(WandBConnector(self.config))
            logger.info("Initializing Weights and Biases telemetry connector")
        return connectors

    def _create_transforms(self) -> list[TelemetryTransform]:
        """Create the telemetry transforms."""
        transforms = []
        for transform in self.config.telemetry.transforms:
            transforms.append(telemetry_transform(transform["type"])(**transform.get("kwargs", {})))
        return transforms

    def _create_callbacks(self) -> list[Callable[[TelemetryNode], None]]:
        """Create the telemetry callbacks."""
        callbacks = []
        for callback in self.config.telemetry.callbacks:
            callbacks.append(telemetry_callback(callback["type"])(**callback.get("kwargs", {})))
        return callbacks

    def _load_stats(self):
        path = Path(self.config.monitoring.file_storage.path) / "checkpoint/SoulsAIStats.json"
        if path.exists() and path.is_file():
            with open(path, "r") as f:
                self.stats = json.load(f)
        else:
            logger.warning("Loading from checkpoint, but no previous telemetry found.")

    def shutdown(self, _):
        """Shut down the telemetry node."""
        logger.info("Shutdown signaled")
        self._shutdown = True

    def _latest_moving_av(self, x: list[float]) -> float:
        view = x[-self.config.telemetry.moving_average :]
        return sum(view) / len(view)  # len(view) can be smaller than N
