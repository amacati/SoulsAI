import logging
from threading import Thread
import json

import numpy as np
from flask import Flask, request

logger = logging.getLogger(__name__)


class GrafanaConnector:

    def __init__(self, data_lock):
        self.app = Flask(__name__)
        self.app.add_url_rule("/", "index", self.index, methods=["GET"])
        self.app.add_url_rule("/search", "search", self.search, methods=["POST"])
        self.app.add_url_rule("/query", "query", self.query, methods=["POST"])
        self.app.add_url_rule("/annotations", "annotations", self.annotations, methods=["POST"])
        self.app_thread = None
        self.data = {}
        self.lock = data_lock
        logging.getLogger('werkzeug').setLevel(logging.ERROR)  # Disable werkzeug logging messages

    def run(self, host="0.0.0.0", port=80):
        kwargs = {"host": host, "port": port, "debug": False}
        self.app_thread = Thread(target=self.app.run, kwargs=kwargs, daemon=True)
        self.app_thread.start()

    def stop(self):
        if self.app_thread is not None:
            self.app_thread.join()

    def index(self):
        return "OK"

    def search(self):
        return json.dumps(list(set([key.rsplit("_", 1)[0] for key in self.data.keys()])))

    def query(self):
        req = json.loads(request.data.decode("utf-8"))
        key, n_samples = req["targets"][0]["target"], req["maxDataPoints"]
        with self.lock:
            rsp = json.dumps(self._load_data(key, n_samples))
        return rsp

    def annotations(self):
        return ""

    def _load_data(self, key, n_samples):
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
        return [{"columns": columns, "rows": rows, "type":"table"}]
