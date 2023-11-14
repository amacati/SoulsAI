"""The main telemetry node module."""
from __future__ import annotations

import logging
import json
from pathlib import Path
import time
from typing import List, TYPE_CHECKING

from redis import Redis
from prometheus_client import start_http_server

from soulsai.utils import load_redis_secret, load_remote_config
from soulsai.distributed.common.serialization import get_serializer_cls
from soulsai.distributed.server.telemetry_node.connectors import (TelemetryConnector,
                                                                  GrafanaConnector,
                                                                  FileStorageConnector,
                                                                  WandBConnector)

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

    stats = [
        "rewards", "rewards_av", "steps", "steps_av", "boss_hp", "boss_hp_av", "wins", "wins_av",
        "eps", "samples"
    ]

    def __init__(self, config: SimpleNamespace):
        """Initialize the connection to Redis and Grafana (if enabled).

        Args:
            config: Training configuration.
            connectors: List of telemetry connectors that expose the stats to external services or
                files.
        """
        logger.info("Telemetry node startup")

        # Read redis server secret and connect to Redis
        secret = load_redis_secret(Path("/run/secrets/redis_secret"))
        self.red = Redis(host='redis', port=6379, password=secret, db=0)
        self.config = load_remote_config(config.redis_address, secret, self.red)
        self.serializer = get_serializer_cls(self.config.algorithm)(self.config.env.name)

        # Initialize the subscriber for telemetry information from the training node
        self.sub_telemetry = self.red.pubsub(ignore_subscribe_messages=True)
        self.sub_telemetry.subscribe("telemetry")

        # Listen for shutdown signal from Redis in a separate thread
        self._shutdown = False
        self.cmd_sub = self.red.pubsub(ignore_subscribe_messages=True)
        self.cmd_sub.subscribe(shutdown=self.shutdown)
        self.cmd_sub.run_in_thread(sleep_time=1., daemon=True)

        # Start a Prometheus server to make the telemetry node status visible in Grafana
        start_http_server(8080)

        self.stats = {
            "rewards": [],
            "rewards_av": [],
            "steps": [],
            "steps_av": [],
            "boss_hp": [],
            "boss_hp_av": [],
            "wins": [],
            "wins_av": [],
            "eps": [],
            "n_env_steps": []
        }
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
        while not self._shutdown:
            # read new samples
            if not (msg := self.sub_telemetry.get_message()):
                time.sleep(0.1)
                continue
            self._update_stats(self.serializer.deserialize_telemetry(msg["data"]))
            n_episodes = len(self.stats["steps"])
            if n_episodes % self.config.telemetry.update_interval == 0:
                for connector in self.connectors:
                    connector.update(self.stats)
                av_reward, av_steps = self.stats["rewards_av"][-1], self.stats["steps_av"][-1]
                logger.info((f"Telemetry updated, last av. reward: {av_reward:.1f}"
                             f", last av. steps: {av_steps:.0f}"))
                if self.stats["rewards_av"][-1] > self._best_reward:
                    self.red.publish("save_best", "")
                    self._best_reward = self.stats["rewards_av"][-1]

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

    def _update_stats(self, sample: dict):
        """Update the telemetry statistics.

        Args:
            sample: Sample dictionary.
        """
        self.stats["rewards"].append(sample["epReward"])
        self.stats["rewards_av"].append(self._latest_moving_av(self.stats["rewards"]))
        self.stats["steps"].append(sample["epSteps"])
        self.stats["steps_av"].append(self._latest_moving_av(self.stats["steps"]))
        self.stats["boss_hp"].append(sample["bossHp"])
        self.stats["boss_hp_av"].append(self._latest_moving_av(self.stats["boss_hp"]))
        self.stats["wins"].append(int(sample["win"]))
        self.stats["wins_av"].append(self._latest_moving_av(self.stats["wins"]))
        self.stats["eps"].append(sample["eps"])
        self.stats["n_env_steps"].append(sample["totalSteps"])

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

    def _latest_moving_av(self, x: List) -> float:
        view = x[-self.config.telemetry.moving_average:]
        return sum(view) / len(view)  # len(view) can be smaller than N
