"""The grafana connector module allows us to expose the data from the telemetry node to Grafana.

Grafana enables users to define custom data sources by setting up a server that responds to
predefined queries. The :class:`~.GrafanaConnector` is essentially that: A small wrapper class
around a flask server that allows the telemetry node to respond to Grafana queries using the current
data.

When Grafana queries the data from a data key, the Connector aquires the data lock to prevent
inconsistencies and sends the requested data as a json string. Within Grafana, this requires the
configuration of the telemetry server as a JSON source.

All stats are sent with an index array ranging from 1 to N to enable plotting without time stamps.

Note:
    Some features such as annotations are currently not supported.
"""
import logging
from threading import Thread
from _thread import LockType
import json
from typing import List

import numpy as np
from flask import Flask, request

logger = logging.getLogger(__name__)


class GrafanaConnector:
    """Custom connector to expose the telemetry node as Grafana data source.

    The data is shared by adding entries to the ``data`` dictionary. Since lists use references
    underneath the hood, any modification of the data is immediately visible in the connector
    thread.
    """

    def __init__(self, data_lock: LockType):
        """Initialize the flask rules.

        Args:
            data_lock: Lock for the data source shared with the main thread.
        """
        self.app = Flask(__name__)
        self.app.add_url_rule("/", "index", self.index, methods=["GET"])
        self.app.add_url_rule("/search", "search", self.search, methods=["POST"])
        self.app.add_url_rule("/query", "query", self.query, methods=["POST"])
        self.app.add_url_rule("/annotations", "annotations", self.annotations, methods=["POST"])
        self.app_thread = None
        self.data = {}
        self.lock = data_lock
        logging.getLogger('werkzeug').setLevel(logging.ERROR)  # Disable werkzeug logging messages

    def run(self, host: str = "0.0.0.0", port: int = 80):
        """Run the Grafana connector server.

        Args:
            host: Server address.
            port: Server port.
        """
        kwargs = {"host": host, "port": port, "debug": False}
        self.app_thread = Thread(target=self.app.run, kwargs=kwargs, daemon=True)
        self.app_thread.start()

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
