"""The connector module allows us to expose the data from the telemetry node to external services.

The module contains a file connector that saves the data to a json file and plots the current stats,
a Grafana connector that exposes the data to a Grafana instance, and a Weights and Biases connector
that sends the data to a Weights and Biases project.
"""
import logging
from threading import Thread
import json
from typing import List
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from flask import Flask, request
from prometheus_client import start_http_server
import wandb

from soulsai.utils import namespace2dict
from soulsai.utils.visualization import save_plots
from soulsai.exception import InvalidConfigError

logger = logging.getLogger(__name__)


class TelemetryConnector(ABC):

    def __init__(self):
        """Initialize the telemetry connector."""

    @abstractmethod
    def start(self):
        """Start the telemetry connector."""

    @abstractmethod
    def update(self, data: dict):
        """Update the data dictionary.

        Args:
            data: Data dictionary.
        """

    @abstractmethod
    def stop(self):
        """Stop the telemetry connector."""


class FileStorageConnector(TelemetryConnector):

    def __init__(self, config: dict):
        """Initialize the file storage connector.

        Args:
            config: Config dictionary.
        """
        super().__init__()
        self._check_config(config)
        self.config = config
        self.plot = config.monitoring.file_storage.plot
        self.path = Path(config.monitoring.file_storage.path) / config.save_dir
        self.path.mkdir(parents=True, exist_ok=True)
        self.stats_path = self.path / "SoulsAIStats.json"
        self.figure_path = self.path / "SoulsAIDashboard.png"

    def start(self):
        ...

    def update(self, data: dict):
        """Update the save files and the plot.

        Args:
            data: Data dictionary.
        """
        with open(self.stats_path, "w") as f:
            json.dump(data, f)
        if self.plot:
            save_plots(data["n_env_steps"], data["rewards"], data["steps"], data["boss_hp"],
                       data["wins"], self.figure_path, data["eps"],
                       self.config.telemetry.moving_average)

    def stop(self):
        ...

    def _check_config(self, config: dict):
        if not hasattr(config.monitoring, "file_storage"):
            raise InvalidConfigError("No file storage config found.")
        for attr in ["path", "plot"]:
            if not hasattr(config.monitoring.file_storage, attr):
                raise InvalidConfigError(
                    f"File storage monitoring config missing attribute `{attr}`.")


class GrafanaConnector(TelemetryConnector):
    """Custom connector to expose the telemetry node as Grafana data source.

    Grafana enables users to define custom data sources by setting up a server that responds to
    predefined queries. This class is a small wrapper around a flask server that allows the
    telemetry node to respond to Grafana queries using the current data.
    """

    def __init__(self, config: dict):
        """Initialize the flask rules for the Grafana connector.

        Args:
            config: Config dictionary.
        """
        super().__init__()
        self.app = Flask(__name__)
        self.app.add_url_rule("/", "index", self.index, methods=["GET"])
        self.app.add_url_rule("/search", "search", self.search, methods=["POST"])
        self.app.add_url_rule("/query", "query", self.query, methods=["POST"])
        self.app.add_url_rule("/annotations", "annotations", self.annotations, methods=["POST"])
        self.app_thread = None
        self.data = {}
        self.config = config
        logging.getLogger('werkzeug').setLevel(logging.ERROR)  # Disable werkzeug logging messages

    def start(self, host: str = "0.0.0.0", port: int = 80):
        """Start the Grafana connector server.

        Args:
            host: Server address.
            port: Server port.
        """
        kwargs = {"host": host, "port": port, "debug": False}
        self.app_thread = Thread(target=self.app.run, kwargs=kwargs, daemon=True)
        self.app_thread.start()
        start_http_server(port=8080)

    def update(self, data: dict):
        """Update the data dictionary.

        Args:
            data: Data dictionary.
        """
        self.data = data

    def stop(self):
        """Stop the Grafana server."""
        if self.app_thread is not None:
            self.app_thread.join()

    def index(self) -> str:
        """Index response.

        Returns:
            An OK string to enable the data source in Grafana.
        """
        return "OK"

    def search(self) -> str:
        """Search response.

        Returns:
            The available data keys.
        """
        return json.dumps(list(set([key.rsplit("_", 1)[0] for key in self.data.keys()])))

    def query(self) -> str:
        """Query response.

        In case we have more samples than the maximum requested data points, we evenly space the
        samples over the whole history.

        Returns:
            The data points for the requested target.
        """
        req = json.loads(request.data.decode("utf-8"))
        key, n_samples = req["targets"][0]["target"], req["maxDataPoints"]
        with self.lock:
            rsp = json.dumps(self._load_data(key, n_samples))
        return rsp

    def annotations(self) -> str:
        """Annotation query response.

        Returns:
            An empty string. We don't support annotations at the moment.
        """
        return ""

    def _load_data(self, key: str, n_samples: int) -> List[dict] | str:
        """Load the data for the key and return it in a suitable format for Grafana.

        If more data points than requested are available, we evenly space the data points of the
        response across the whole range of available data.

        Args:
            key: Data key.
            n_samples: Maximum number of data points.

        Returns:
            The data dictionary or an empty string if the key is not contained in the data.
        """
        if key not in self.data.keys():
            return ""
        keys = [k for k in self.data.keys() if key.lower() in k]
        columns, rows = [], []
        for key in keys:
            if len(self.data[key]) <= n_samples:
                idx = np.arange(0, len(self.data[key]))
            else:
                idx = np.linspace(0, len(self.data[key]) - 1, n_samples, dtype=np.int64)
            columns.append({"text": key, "type": "number"})
            columns.append({"text": key + "_idx", "type": "number"})
            rows.append([self.data[key][i] for i in idx])
            rows.append(idx.tolist())
        columns.append({"text": "samples", "type": "number"})  # Always send samples as X-Axis
        rows.append([self.data["samples"][i] for i in idx])
        rows = list(zip(*rows))
        return [{"columns": columns, "rows": rows, "type": "table"}]


class WandBConnector(TelemetryConnector):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._check_config(config)
        with open(Path("/run/secrets/wandb_api_key"), "r") as f:
            wandb.login(key=f.read())
        self._step = 0
        logger.info("Initialized Weights and Biases telemetry connector")

    def start(self):
        """Start the telemetry connector."""
        save_dir = str(Path(self.config.monitoring.wandb.save_dir))
        self.run = wandb.init(project=self.config.monitoring.wandb.project,
                              entity=self.config.monitoring.wandb.entity,
                              group=getattr(self.config.monitoring.wandb, "group", None),
                              config=namespace2dict(self.config),
                              dir=save_dir)

    def update(self, data: dict):
        """Upload the new stats to Weights and Biases.

        Args:
            data: Data dictionary.
        """
        for i in range(self._step, len(data["n_env_steps"])):
            step = data["n_env_steps"][i]
            self.run.log({key: value[i] for key, value in data.items()}, step=step)
        self._step = len(data["n_env_steps"])

    def stop(self):
        self.run.finish()

    def _check_config(self, config: dict):
        if not hasattr(config.monitoring, "wandb"):
            raise InvalidConfigError("No Weights and Biases config found.")
        for attr in ["project", "entity", "save_dir"]:
            if not hasattr(config.monitoring.wandb, attr):
                raise InvalidConfigError(
                    f"Weights and Biases monitoring config missing attribute `{attr}`.")
