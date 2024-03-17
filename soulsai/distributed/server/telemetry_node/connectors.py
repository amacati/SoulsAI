"""The connector module allows us to expose the data from the telemetry node to external services.

The module contains a file connector that saves the data to a json file and plots the current stats,
a Grafana connector that exposes the data to a Grafana instance, and a Weights and Biases connector
that sends the data to a Weights and Biases project.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Thread
from typing import List

import numpy as np
import wandb
from flask import Flask, request

from soulsai.exception import InvalidConfigError
from soulsai.utils import namespace2dict
from soulsai.utils.visualization import save_plots

logger = logging.getLogger(__name__)


class TelemetryConnector(ABC):
    """Abstract class for telemetry connectors.

    Connectors are used to expose the telemetry data. This allows us to save the same telemetry data
    to the disk, sync it with a Grafana instance, and send it to a Weights and Biases project at the
    same time using a unified API.
    """

    def __init__(self):
        """Initialize the telemetry connector."""

    def start(self):
        """Start the telemetry connector."""

    @abstractmethod
    def update(self, data: dict):
        """Update the data dictionary.

        Args:
            data: Data dictionary.
        """

    def stop(self):
        """Stop the telemetry connector."""


class FileStorageConnector(TelemetryConnector):
    """File storage connector to save the telemetry data to a json file and plot the current stats.

    The connector saves the telemetry data and, if enabled, the current plots to files on the disk.
    """

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
        self.stats_path = self.path / "telemetry.json"
        self.figure_path = self.path / "telemetry.png"

    def update(self, data: dict):
        """Update the save files and the plot.

        Args:
            data: Data dictionary.
        """
        with open(self.stats_path, "w") as f:
            json.dump(data, f)
        if self.plot:
            xkey = self.config.monitoring.file_storage.plot.xkey
            ykeys = self.config.monitoring.file_storage.plot.ykeys
            save_plots(
                x=np.array(data[xkey]),
                ys=[np.array(data[key]) for key in ykeys],
                xlabel=xkey,
                ylabels=ykeys,
                path=self.figure_path,
                N_av=self.config.telemetry.moving_average,
            )

    def _check_config(self, config: dict):
        if not hasattr(config.monitoring, "file_storage"):
            raise InvalidConfigError("No file storage config found.")
        for attr in ["path", "plot"]:
            if not hasattr(config.monitoring.file_storage, attr):
                raise InvalidConfigError(
                    f"File storage monitoring config missing attribute `{attr}`."
                )


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
        self.app_thread = None
        self.data = {}
        self.config = config
        logging.getLogger("werkzeug").setLevel(logging.ERROR)  # Disable werkzeug logging messages

    def start(self, host: str = "0.0.0.0", port: int = 80):
        """Start the Grafana connector server.

        Args:
            host: Server address.
            port: Server port.
        """
        kwargs = {"host": host, "port": port, "debug": False, "use_reloader": False}
        self.app_thread = Thread(target=self.app.run, kwargs=kwargs, daemon=True)
        self.app_thread.start()

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
        if (req := request.data.decode("utf-8")) == "":
            return self.data
        req = json.loads(req)
        if "maxDataPoints" not in req.keys():
            return self.data
        return self._resample_data(req["maxDataPoints"])

    def _resample_data(self, max_samples: int) -> dict[str, List[float]]:
        """Resample the data from `self.data` to contain a maximum of `max_samples` elements.

        If more data points than requested are available, we evenly space the data points of the
        response across the whole range of available data.

        Args:
            max_samples: Maximum number of data points.

        Returns:
            The resampled data dictionary.
        """
        data = {}
        for key in self.data.keys():
            if len(self.data[key]) <= max_samples:
                data[key] = self.data[key]
                continue
            idx = np.linspace(0, len(self.data[key]) - 1, max_samples, dtype=np.int64)
            data[key] = [self.data[key][i] for i in idx]
        return data


class WandBConnector(TelemetryConnector):
    """Custom connector to send telemetry data to Weights and Biases.

    Note:
        This connector requires the Weights and Biases API key to be stored in a file at
        `/run/secrets/wandb_api_key`. When running docker compose, secret files under
        `config/secrets` are mounted to `/run/secrets` in the container.
    """

    def __init__(self, config: dict):
        """Initialize the Weights and Biases telemetry connector.

        Args:
            config: Config dictionary.
        """
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
        self.run = wandb.init(
            project=self.config.monitoring.wandb.project,
            entity=self.config.monitoring.wandb.entity,
            group=getattr(self.config.monitoring.wandb, "group", None),
            config=namespace2dict(self.config),
            dir=save_dir,
        )

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
        """Stop the connector and finish the WandB run."""
        self.run.finish()

    def _check_config(self, config: dict):
        if not hasattr(config.monitoring, "wandb"):
            raise InvalidConfigError("No Weights and Biases config found.")
        for attr in ["project", "entity", "save_dir"]:
            if not hasattr(config.monitoring.wandb, attr):
                raise InvalidConfigError(
                    f"Weights and Biases monitoring config missing attribute `{attr}`."
                )
