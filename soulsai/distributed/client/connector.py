import json
import logging
from pathlib import Path
import torch.multiprocessing as mp
import time
import queue
from uuid import uuid4

import redis
from redis import Redis

from soulsai.core.agent import DQNClientAgent, PPOClientAgent
from soulsai.core.normalizer import Normalizer
from soulsai.utils import load_redis_secret, namespace2dict
from soulsai.exception import ClientRegistrationError, ServerTimeoutError

logger = logging.getLogger(__name__)


class DQNConnector:

    def __init__(self, config, encode_sample, encode_tel):
        mp.set_start_method("spawn")
        self.config = config
        self.agent = DQNClientAgent(config.dqn.network_type,
                                     namespace2dict(config.dqn.network_kwargs))
        if config.dqn.normalizer_kwargs is not None:
            norm_kwargs = namespace2dict(config.dqn.normalizer_kwargs) 
        else:
            norm_kwargs = {}
        self.normalizer = Normalizer(config.n_states, **norm_kwargs)
        self._eps = mp.Value("d", -1.)
        self._lock = mp.Lock()
        self._update_event = mp.Event()
        self._stop_event = mp.Event()
        secret = load_redis_secret(Path(__file__).parents[3] / "config" / "redis.secret")
        address = self.config.redis_address
        red_notify = Redis(host=address, password=secret, port=6379, db=0)
        self.pubsub = red_notify.pubsub()
        self.pubsub.subscribe(model_update=self._agent_update_callback)
        self.pubsub.run_in_thread(sleep_time=.05, daemon=True)

        self._msg_queue = mp.Queue(maxsize=100)
        args = (self._msg_queue, address, secret, self._stop_event, encode_sample, encode_tel)
        self.msg_consumer = mp.Process(target=self._consume_msgs, args=args)
        self.agent.share_memory()
        self.normalizer.share_memory()
        args = (self._update_event, self._stop_event, self.agent, self.normalizer, self._eps,
                self._lock, secret, config)
        self.model_updater = mp.Process(target=self.update_model, args=args)
        self.model_updater.start()
        self.msg_consumer.start()
        # Block while first model is not here
        logger.info("Waiting for model download...")
        while self.model_id[0] == " ":
            time.sleep(1)
        logger.info("Download complete, connector initialized")
        self.heartbeat = mp.Process(target=self._heartbeat, args=(address, secret,
                                                                  self._stop_event),
                                    daemon=True)
        self.heartbeat.start()
        # Utility attributes
        self._full_queue_warn_time = 0

    def __enter__(self):
        self._lock.acquire()

    def __exit__(self, *args):
        self._lock.release()
        if args[0] is None:
            return True

    @property
    def model_id(self):
        return self.agent.model_id

    @property
    def eps(self):
        return self._eps.value

    def push_msg(self, msg_type, *args):
        try:
            self._msg_queue.put_nowait((msg_type, *args))
        except queue.Full:
            if time.time() - self._full_queue_warn_time > 5:
                self._full_queue_warn_time = time.time()
                logger.warning("Connector queue is full")

    def close(self):
        self._stop_event.set()
        self.msg_consumer.join()
        self.model_updater.join()
        self.heartbeat.join()
        self._msg_queue.cancel_join_thread()
        logger.debug("All background processes joined")

    @staticmethod
    def update_model(update_event, stop_event, agent: DQNClientAgent, normalizer, eps, lock, secret,
                     config):
        logger.debug("Background update process startup")
        red = Redis(host=config.redis_address, password=secret, port=6379, db=0)
        _params = red.hgetall("model_params")
        model_params = {key.decode("utf-8"): value for key, value in _params.items()}
        # Deserialize is slower than state_dict load, so we deserialize on a local buffer agent
        # first and then overwrite the tensors of the main agent with load_state_dict
        buffer_agent = DQNClientAgent(config.dqn.network_type,
                                      namespace2dict(config.dqn.network_kwargs))
        buffer_agent.deserialize(model_params)
        if config.dqn.normalize:
            norm_params = normalizer.deserialize(model_params)
        with lock:
            agent.load_state_dict(buffer_agent.state_dict())
            eps.value = float(model_params["eps"].decode("utf-8"))
            if config.dqn.normalize:
                normalizer.load_params(*norm_params)
        while not stop_event.is_set():
            if not update_event.wait(1):
                continue  # Check if stop event has been set
            update_event.clear()
            try:
                _params = red.hgetall("model_params")
                model_params = {key.decode("utf-8"): value for key, value in _params.items()}
                buffer_agent.deserialize(model_params)
                if config.dqn.normalize:
                    norm_params = normalizer.deserialize(model_params)
                # We can use a blocking approach instead of changing references between multiple
                # models as writing the new parameters typically only requires ~1e-3s
                with lock:
                    agent.load_state_dict(buffer_agent.state_dict())
                    eps.value = float(model_params["eps"].decode("utf-8"))
                    if config.dqn.normalize:
                        normalizer.load_params(*norm_params)
            except redis.exceptions.ConnectionError:
                time.sleep(10)
                red = Redis(host=config.redis_address, password=secret, port=6379, db=0)

    def _agent_update_callback(self, _):
        self._update_event.set()

    @staticmethod
    def _consume_msgs(msg_queue, address, secret, stop_event, encode_sample, encode_tel):
        logger.debug("Background message consumer process startup")
        red = Redis(host=address, password=secret, port=6379, db=0)
        while not stop_event.is_set() or not msg_queue.empty():
            try:
                msg = msg_queue.get(block=True, timeout=1.)
            except queue.Empty:
                continue  # Check again if stop event has been set
            try:
                if msg[0] == "sample":
                    sample = json.dumps({"model_id": msg[1], "sample": encode_sample(msg)})
                    red.publish("samples", sample)
                elif msg[0] == "telemetry":
                    red.publish("telemetry", json.dumps(encode_tel(msg)))
                else:
                    logger.warning(f"Unknown message type {msg[0]}")
            except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
                time.sleep(10)
                red = Redis(host=address, password=secret, port=6379, db=0)

    @staticmethod
    def _heartbeat(address, secret, stop_flag):
        logging.basicConfig(level=logging.INFO)
        red = Redis(host=address, password=secret, port=6379, db=0)
        con_id = str(uuid4())
        disconnect = False
        while not stop_flag.wait(1):
            msg = json.dumps({"client_id": con_id, "timestamp": time.time()})
            try:
                red.publish("dqn_heartbeat", msg)
                if disconnect:
                    logger.info("Connection to server restored")
                    disconnect = False
            except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
                logger.warning("Connection to server interrupted. Trying to reconnect")
                disconnect = True
                time.sleep(10)
                red = Redis(host=address, password=secret, port=6379, db=0)


class PPOConnector:

    def __init__(self, config, encode_sample, encode_tel):
        self.config = config
        self.agent = PPOClientAgent(config.ppo.actor_net_type,
                                    namespace2dict(config.ppo.actor_net_kwargs))
        self._stop_event = mp.Event()
        self._update_event = mp.Event()
        secret = load_redis_secret(Path(__file__).parents[3] / "config" / "redis.secret")
        address = self.config.redis_address

        self.red = Redis(host=address, password=secret, port=6379, db=0)
        self.update_sub = self.red.pubsub(ignore_subscribe_messages=True)
        self.update_sub.subscribe("model_update")

        # Server discovery and client registration
        tmp_id = str(uuid4())  # Temporary ID for server registration
        discovery_sub = self.red.pubsub(ignore_subscribe_messages=True)
        discovery_sub.subscribe(tmp_id)
        self.red.publish("ppo_discovery", tmp_id)
        tstart = time.time()
        while time.time() - tstart < 60:
            msg = discovery_sub.get_message(timeout=60.)
            if not msg:  # ignore_subscribe_messages + timeout doesn't work, so we have to handle it
                time.sleep(0.5)
                continue
            break
        if msg is None:
            logger.error("Server discovery failed. Check if server is already full")
            raise ClientRegistrationError("Server failed to respond")
        self.con_id = json.loads(msg["data"])
        logger.info(f"Client registration successful. New client ID: {self.con_id}")

        self.heartbeat = mp.Process(target=self._heartbeat, args=(address, secret, self.con_id,
                                                                  self._stop_event),
                                    daemon=True)
        self.heartbeat.start()

        self._msg_pipe, _msg_pipe = mp.Pipe()
        args = (_msg_pipe, address, secret, self._stop_event, encode_sample, encode_tel)
        self.msg_consumer = mp.Process(target=self._consume_msgs, args=args)
        self.msg_consumer.start()

    def push_sample(self, model_id, step_id, sample):
        self._msg_pipe.send(("sample", self.con_id, model_id, step_id, sample))

    def push_telemetry(self, *args):
        self._msg_pipe.send(("telemetry", *args))

    def close(self):
        self._stop_event.set()
        self.msg_consumer.join()
        self.heartbeat.join()
        logger.debug("All background processes joined")

    @staticmethod
    def _heartbeat(address, secret, con_id, stop_flag):
        red = Redis(host=address, password=secret, port=6379, db=0)
        while not stop_flag.wait(1):
            msg = json.dumps({"client_id": con_id, "timestamp": time.time()})
            red.publish("ppo_heartbeat", msg)

    def sync(self, timeout=100.):
        tstart = time.time()
        while not time.time() - tstart > timeout:
            msg = self.update_sub.get_message()
            if not msg:  # Redis timeout + ignore subscribe doesn't work properly
                time.sleep(0.01)
                continue
            break
        if not msg:
            raise ServerTimeoutError(f"Timeout of {timeout}s exceeded during synchronization")
        raw_params = self.red.hgetall("model_params")
        model_params = {key.decode("utf-8"): value for key, value in raw_params.items()}
        self.agent.deserialize(model_params)

    @staticmethod
    def _consume_msgs(msg_pipe, address, secret, stop_event, encode_sample, encode_tel):
        logger.debug("Background message consumer process startup")
        red = Redis(host=address, password=secret, port=6379, db=0)
        while not stop_event.is_set() or msg_pipe.poll():
            if not msg_pipe.poll(1):
                continue  # Check again if stop event has been set
            msg = msg_pipe.recv()
            if msg[0] == "sample":
                sample = encode_sample(msg)
                red.publish("samples", json.dumps({"client_id": msg[1], "model_id": msg[2],
                                                   "step_id": msg[3], "sample": sample}))
            elif msg[0] == "telemetry":
                red.publish("telemetry", json.dumps(encode_tel(msg)))
            else:
                logger.warning(f"Unknown message type {msg[0]}")
