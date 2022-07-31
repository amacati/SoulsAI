import json
import logging
import multiprocessing as mp
from pathlib import Path
import time

import redis
import numpy as np

from soulsai.core.agent import ClientAgent
from soulsai.core.normalizer import Normalizer

logger = logging.getLogger(__name__)


class Connector:

    def __init__(self, nstates, nactions):
        self._agent = ClientAgent(nstates, nactions)
        self._normalizer = Normalizer(nstates)
        self.nstates, self.nactions = nstates, nactions
        self._eps = mp.Value("d", -1.)
        self._lock = mp.Lock()
        self._update_event = mp.Event()
        self._stop_event = mp.Event()
        secret = self._get_secret()
        red_notify = redis.Redis(host="redis", password=secret, port=6379, db=0)
        self.pubsub = red_notify.pubsub()
        self.pubsub.psubscribe(model_update=self._agent_update_callback)
        self.pubsub.run_in_thread(sleep_time=.05, daemon=True)

        self._msg_pipe, _msg_pipe = mp.Pipe()
        args = (_msg_pipe, secret, self._stop_event)
        self.msg_consumer = mp.Process(target=self._consume_msgs, args=args)
        self._agent.share_memory()
        self._normalizer.share_memory()
        args = (self._update_event, self._stop_event, self._agent, self._eps, self._normalizer,
                self._lock, secret, nstates, nactions)
        self.model_updater = mp.Process(target=self.update_agent, args=args)
        self.model_updater.start()
        self.msg_consumer.start()
        # Block while first model is not here
        logger.info("Waiting for model download...")
        while self.model_id[0] == " ":
            time.sleep(1)
        logger.info("Download complete, connector initialized")

    def __enter__(self):
        self._lock.acquire()

    def __exit__(self, *args):
        self._lock.release()
        if args[0] is None:
            return True

    def agent(self, state):
        return self._agent(self._normalizer(state))  # Default normalizer does not change states

    @property
    def model_id(self):
        return self._agent.model_id

    @property
    def eps(self):
        return self._eps.value

    def push_sample(self, model_id, sample):
        self._msg_pipe.send(("sample", model_id, sample))

    def push_telemetry(self, total_reward, steps, win, eps):
        self._msg_pipe.send(("telemetry", total_reward, steps, win, eps))

    def close(self):
        self._stop_event.set()
        self.msg_consumer.join()
        self.model_updater.join()
        logger.debug("All background processes joined")

    @staticmethod
    def update_agent(update_event, stop_event, model: ClientAgent, eps, normalizer,
                     lock, secret, nstates, nactions):
        red = redis.Redis(host="redis", password=secret, port=6379, db=0)
        logger.debug("Background update process startup")
        _params = red.hgetall("model_params")
        model_params = {key.decode("utf-8"): value for key, value in _params.items()}
        # Deserialize is slower than state_dict load, so we deserialize on a local buffer agent
        # first and then overwrite the tensors of the main agent with load_state_dict
        buffer_agent = ClientAgent(nstates, nactions)
        buffer_agent.deserialize(model_params)
        with lock:
            model.load_state_dict(buffer_agent.state_dict())
            eps.value = float(model_params["eps"].decode("utf-8"))
            if int(model_params['normalize']):
                normalizer.deserialize(model_params)
        while not stop_event.is_set():
            if not update_event.wait(1):
                continue  # Check if stop event has been set
            update_event.clear()
            _params = red.hgetall("model_params")
            model_params = {key.decode("utf-8"): value for key, value in _params.items()}
            buffer_agent.deserialize(model_params)
            # We can use a blocking approach instead of changing references between multiple models
            # as writing the new parameters typically only requires ~1e-3s
            with lock:
                model.load_state_dict(buffer_agent.state_dict())
                eps.value = float(model_params["eps"].decode("utf-8"))
                if int(model_params['normalize']):
                    normalizer.deserialize(model_params)

    def _agent_update_callback(self, _):
        self._update_event.set()

    @staticmethod
    def _consume_msgs(msg_pipe, secret, stop_event):
        logger.debug("Background message consumer process startup")
        red = redis.Redis(host="redis", password=secret, port=6379, db=0)
        while not stop_event.is_set():
            if not msg_pipe.poll(1):
                continue  # Check if stop event has been set
            msg = msg_pipe.recv()
            if msg[0] == "sample":
                msg[2][0], msg[2][3] = np.float64(msg[2][0]), np.float64(msg[2][3])
                sample = [list(msg[2][0]), msg[2][1], msg[2][2], list(msg[2][3]), msg[2][4]]
                red.publish("samples", json.dumps({"model_id": msg[1], "sample": sample}))
            elif msg[0] == "telemetry":
                telemetry = {"reward": msg[1], "steps": msg[2], "win": bool(msg[3]), "eps": msg[4]}
                red.publish("telemetry", json.dumps(telemetry))
            else:
                logger.warning(f"Unknown message type {msg[0]}")

    def _get_secret(self):
        with open(Path(__file__).parent / "redis.secret", "r") as f:
            conf = f.readlines()
        secret = None
        for line in conf:
            if len(line) > 12 and line[0:12] == "requirepass ":
                secret = line[12:]
                break
        if secret is None:
            raise RuntimeError("Missing password configuration for redis in redis.secret")
        return secret
