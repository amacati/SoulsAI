import logging
from threading import Thread
import json

from flask import Flask, request, jsonify

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
        columns = [{"text": key, "type": "number"} for key in keys]
        # Bound checking not necesary with [-idx:]
        rows = list(zip(*[self.data[key][-n_samples:] for key in keys]))
        return [{"columns": columns, "rows": rows, "type":"table"}]
