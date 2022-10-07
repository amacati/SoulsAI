import json
import logging
from pathlib import Path
import torch.multiprocessing as mp
import time
from uuid import uuid4

import redis

from soulsai.core.agent import DQNClientAgent, PPOClientAgent
from soulsai.utils import load_redis_secret
from soulsai.exception import ClientRegistrationError, ServerTimeoutError

logger = logging.getLogger(__name__)


class DQNConnector:

    def __init__(self, config, encode_sample, encode_tel):
        mp.set_start_method("spawn")
        self.config = config
        self._agent = DQNClientAgent(config.network_type, config.network_kwargs)
        self._eps = mp.Value("d", -1.)
        self._lock = mp.Lock()
        self._update_event = mp.Event()
        self._stop_event = mp.Event()
        secret = load_redis_secret(Path(__file__).parents[3] / "config" / "redis.secret")
        address = self.config.redis_address
        red_notify = redis.Redis(host=address, password=secret, port=6379, db=0)
        self.pubsub = red_notify.pubsub()
        self.pubsub.psubscribe(model_update=self._agent_update_callback)
        self.pubsub.run_in_thread(sleep_time=.05, daemon=True)

        self._msg_pipe, _msg_pipe = mp.Pipe()
        args = (_msg_pipe, address, secret, self._stop_event, encode_sample, encode_tel)
        self.msg_consumer = mp.Process(target=self._consume_msgs, args=args)
        self._agent.share_memory()
        args = (self._update_event, self._stop_event, self._agent, self._eps, self._lock,
                secret, config)
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
        return self._agent(state)

    @property
    def model_id(self):
        return self._agent.model_id

    @property
    def eps(self):
        return self._eps.value

    def push_sample(self, model_id, sample):
        self._msg_pipe.send(("sample", model_id, sample))

    def push_telemetry(self, *args):
        self._msg_pipe.send(("telemetry", *args))

    def close(self):
        self._stop_event.set()
        self.msg_consumer.join()
        self.model_updater.join()
        logger.debug("All background processes joined")

    @staticmethod
    def update_agent(update_event, stop_event, model: DQNClientAgent, eps, lock, secret,
                     config):
        logger.debug("Background update process startup")
        red = redis.Redis(host=config.redis_address, password=secret, port=6379, db=0)
        _params = red.hgetall("model_params")
        model_params = {key.decode("utf-8"): value for key, value in _params.items()}
        # Deserialize is slower than state_dict load, so we deserialize on a local buffer agent
        # first and then overwrite the tensors of the main agent with load_state_dict
        buffer_agent = DQNClientAgent(config.network_type, config.network_kwargs)
        buffer_agent.deserialize(model_params)
        with lock:
            model.load_state_dict(buffer_agent.state_dict())
            eps.value = float(model_params["eps"].decode("utf-8"))
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

    def _agent_update_callback(self, _):
        self._update_event.set()

    @staticmethod
    def _consume_msgs(msg_pipe, address, secret, stop_event, encode_sample, encode_tel):
        logger.debug("Background message consumer process startup")
        red = redis.Redis(host=address, password=secret, port=6379, db=0)
        while not stop_event.is_set() or msg_pipe.poll():
            if not msg_pipe.poll(1):
                continue  # Check again if stop event has been set
            msg = msg_pipe.recv()
            if msg[0] == "sample":
                sample = encode_sample(msg)
                red.publish("samples", json.dumps({"model_id": msg[1], "sample": sample}))
            elif msg[0] == "telemetry":
                red.publish("telemetry", json.dumps(encode_tel(msg)))
            else:
                logger.warning(f"Unknown message type {msg[0]}")


class PPOConnector:

    def __init__(self, config, encode_sample, encode_tel):
        self.config = config
        self.agent = PPOClientAgent(config.network_type, config.network_kwargs)

        self._stop_event = mp.Event()
        self._update_event = mp.Event()
        secret = load_redis_secret(Path(__file__).parents[3] / "config" / "redis.secret")
        address = self.config.redis_address

        self.con_id = uuid4()
        self.red = redis.Redis(host=address, password=secret, port=6379, db=0)
        self.heartbeat_thread = mp.Process()
        self.update_sub = self.red.pubsub()
        self.update_sub.subscribe("model_update")
        
        self.heartbeat_p = mp.Process(target=self._heartbeat, args=(address, secret, self.con_id),
                                      daemon=True)
        self.heartbeat_p.start()

        discovery_sub = self.red.pubsub()
        discovery_sub.subscribe(self.con_id)
        self.red.publish("ppo_discovery", self.con_id)
        msg = discovery_sub.get_message(timeout=5)
        if msg[0] != "ACK":
            logger.error("Server discovery rejected client during. Check if server is already full")
            raise ClientRegistrationError("Server rejected client during discovery.")

        self._msg_pipe, _msg_pipe = mp.Pipe()
        args = (_msg_pipe, address, secret, self._stop_event, encode_sample, encode_tel)
        self.msg_consumer = mp.Process(target=self._consume_msgs, args=args)
        self.msg_consumer.start()
        self.sync(300)

    def push_sample(self, model_id, sample):
        self._msg_pipe.send(("sample", model_id, sample))

    def push_telemetry(self, *args):
        self._msg_pipe.send(("telemetry", *args))

    def close(self):
        self._stop_event.set()
        self.msg_consumer.join()
        self.heartbeat.join()
        logger.debug("All background processes joined")

    @staticmethod
    def _heartbeat(address, secret, con_id, stop_flag):
        red = redis.Redis(host=address, password=secret, port=6379, db=0)
        while not stop_flag.wait(1):
            red.publish("ppo_heartbeat", con_id)

    def sync(self, timeout=100.):
        msg = self.update_sub.get_message(timeout=timeout)
        if not msg:
            raise ServerTimeoutError(f"Timeout of {timeout}s exceeded during synchronization")
        raw_params = self.red.hgetall("model_params")
        model_params = {key.decode("utf-8"): value for key, value in raw_params.items()}
        self.agent.deserialize(model_params)

    @staticmethod
    def _consume_msgs(msg_pipe, address, secret, stop_event, encode_sample, encode_tel):
        logger.debug("Background message consumer process startup")
        red = redis.Redis(host=address, password=secret, port=6379, db=0)
        while not stop_event.is_set() or msg_pipe.poll():
            if not msg_pipe.poll(1):
                continue  # Check again if stop event has been set
            msg = msg_pipe.recv()
            if msg[0] == "sample":
                sample = encode_sample(msg)
                red.publish("samples", json.dumps({"model_id": msg[1], "sample": sample}))
            elif msg[0] == "telemetry":
                red.publish("telemetry", json.dumps(encode_tel(msg)))
            else:
                logger.warning(f"Unknown message type {msg[0]}")
