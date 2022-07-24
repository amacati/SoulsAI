from types import SimpleNamespace
import json
import logging
from pathlib import Path
import multiprocessing as mp
import time

import yaml
import redis

from soulsai.core.agent import ClientAgent

logger = logging.getLogger(__name__)


class Connector:

    def __init__(self):
        self._agent = ClientAgent(72, 20)
        self._buffer_agent = ClientAgent(72, 20)
        self._eps = mp.Value("d", -1.)
        self._lock = mp.Lock()
        self._update_event = mp.Event()
        self._stop_event = mp.Event()
        self.config, secret = self._read_config()
        self.red_msg = redis.Redis(host=self.config.address, password=secret, port=6379, db=0)
        self.red_update = redis.Redis(host=self.config.address, password=secret, port=6379, db=0)
        red_notify = redis.Redis(host=self.config.address, password=secret, port=6379, db=0)
        self.pubsub = red_notify.pubsub()
        self.pubsub.psubscribe(model_update=self._agent_update_callback)
        self.pubsub.run_in_thread(sleep_time=.1, daemon=True)

        self._msg_pipe, _msg_pipe = mp.Pipe()
        self.msg_consumer = mp.Process(target=self._consume_msgs, args=(_msg_pipe, self.red_msg))
        self._agent.share_memory()
        args = (self._update_event, self._stop_event, self._agent, self._eps, self._lock,
                self.red_update)
        self.model_updater = mp.Process(target=self.update_agent, args=args)
        self.msg_consumer.start()
        self.model_updater.start()
        # Block while first model is not here
        logger.info("Waiting for model download to complete")
        while self.model_id == "":
            time.sleep(1)
            logger.warning("Waiting for model download...")
        logger.info("Download complete, connector initialized")

    def __enter__(self):
        self._lock.acquire()

    def __exit__(self, *_):
        self._lock.release()
        return True

    def agent(self, state):
        with self._lock:
            return self._agent(state)

    @property
    def model_id(self):
        with self._lock:
            return self._agent.model_id

    @property
    def eps(self):
        with self._lock:
            return self._eps

    def push_sample(self, model_id, sample):
        self._msg_pipe.send(("sample", model_id, sample))

    def push_telemetry(self, total_reward, steps, boss_hp, win):
        self._msg_pipe.send(("telemetry", total_reward, steps, boss_hp, win))

    def close(self):
        self._stop_event.set()
        self.msg_consumer.join()
        self.model_updater.join()

    def update_agent(self, update_event, stop_event, model: ClientAgent, eps, lock, red):
        logger.debug("Background update process startup")
        _params = red.hgetall("model_params")
        model_params = {key.decode("utf-8"): value for key, value in _params.items()}
        # Deserialize is slower than state_dict load, so we deserialize on a local buffer agent
        # first and then overwrite the tensors of the main agent with load_state_dict
        self._buffer_agent.deserialize(model_params)
        with lock:
            model.load_state_dict(self._buffer_agent.state_dict())
            eps.value = float(model_params["eps"].decode("utf-8"))
        while not stop_event.is_set():
            if not update_event.is_set():
                time.sleep(0.1)
                continue
            update_event.clear()
            _params = red.hgetall("model_params")
            model_params = {key.decode("utf-8"): value for key, value in _params.items()}
            self._buffer_agent.deserialize(model_params)
            # We can use a blocking approach instead of changing references between multiple models
            # as writing the new parameters typically only requires ~1e-3s
            with lock:
                model.load_state_dict(self._buffer_agent.state_dict())
                eps.value = float(model_params["eps"].decode("utf-8"))

    def _read_config(self):
        root_path = Path(__file__).parent
        with open(root_path / "config_d.yaml") as f:
            config = yaml.safe_load(f)
        if (root_path / "config.yaml").is_file():
            with open(root_path / "config.yaml") as f:
                config |= yaml.safe_load(f)  # Overwrite default config with keys from user config

        with open(root_path / "redis.secret") as f:
            _secret_config = f.readlines()
        secret = None
        for line in _secret_config:
            if len(line) > 12 and line[0:12] == "requirepass ":
                secret = line[12:]
                break
        if secret is None:
            raise RuntimeError("Missing password configuration for redis in redis.secret")
        return SimpleNamespace(**config), secret

    def _agent_update_callback(self, _):
        self._update_event.set()

    def _consume_msgs(msg_pipe, red, stop_event):
        while not stop_event.set():
            if not msg_pipe.poll():
                time.sleep(0.1)
                continue
            msg = msg_pipe.recv()
            if msg[0] == "sample":
                sample = [msg[2][0].as_json(), msg[2][1], msg[2][2], msg[2][3].as_json(), msg[2][4]]
                red.publish("samples", json.dumps({"model_id": msg[1], "sample": sample}))
            elif msg[0] == "telemetry":
                telemetry = {"reward": msg[1], "steps": msg[2], "boss_hp": msg[3], "win": msg[4],
                             "eps": msg[5]}
                red.publish("telemetry", json.dumps(telemetry))
            else:
                logger.warning(f"Unknown message type {msg[0]}")
