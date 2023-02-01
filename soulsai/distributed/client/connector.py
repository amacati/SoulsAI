import json
import logging
from pathlib import Path
import socket
import time
import queue
from uuid import uuid4
import multiprocessing as python_mp

import redis
from redis import Redis
import torch.multiprocessing as mp

from soulsai.core.agent import DQNClientAgent, PPOClientAgent
from soulsai.core.normalizer import Normalizer
from soulsai.utils import load_redis_secret, namespace2dict
from soulsai.exception import ClientRegistrationError, ServerTimeoutError

logger = logging.getLogger(__name__)


class DQNConnector:

    def __init__(self, config, encode_sample, encode_tel):
        cxt = mp.get_context()
        if not isinstance(cxt, python_mp.context.SpawnContext):
            logger.warning(f"Multiprocessing context already set to {type(cxt)}")
            logger.warning("Trying to force spawn method...")
            mp.set_start_method("spawn", force=True)
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

        args = (self._update_event, address, secret, self._stop_event)
        self.update_sub = mp.Process(target=self._update_msg, args=args, daemon=True)
        self.update_sub.start()

        self.shutdown = mp.Event()
        args = (self.shutdown, address, secret)
        self.shutdown_sub = mp.Process(target=self._client_shutdown, args=args, daemon=True)
        self.shutdown_sub.start()

        self._msg_queue = mp.Queue(maxsize=100)
        args = (self._msg_queue, address, secret, self._stop_event, encode_sample, encode_tel)
        self.msg_consumer = mp.Process(target=self._consume_msgs, args=args, daemon=True)
        self.msg_consumer.start()

        self.agent.share_memory()
        self.normalizer.share_memory()
        args = (self._update_event, self._stop_event, self.agent, self.normalizer, self._eps,
                self._lock, secret, config)
        self.model_updater = mp.Process(target=self.update_model, args=args, daemon=True)
        self.model_updater.start()

        # Block while first model is not here
        logger.info("Waiting for model download...")
        while self.model_id[0] == " ":
            time.sleep(1)
        logger.info("Download complete, connector initialized")
        args = (address, secret, self._stop_event)
        self.heartbeat = mp.Process(target=self._heartbeat, args=args, daemon=True)
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
        red = Redis(host=config.redis_address, password=secret, port=6379, db=0,
                    socket_keepalive=True,
                    socket_keepalive_options={socket.TCP_KEEPIDLE: 10, socket.TCP_KEEPINTVL: 60})
        # Deserialize is slower than state_dict load, so we deserialize on a local buffer agent
        # first and then overwrite the tensors of the main agent with load_state_dict
        buffer_agent = DQNClientAgent(config.dqn.network_type,
                                      namespace2dict(config.dqn.network_kwargs))
        update_event.set()  # Ensure load on first start up
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
            except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
                time.sleep(10)
                socket_options = {socket.TCP_KEEPIDLE: 10, socket.TCP_KEEPINTVL: 60}
                red = Redis(host=config.redis_address, password=secret, port=6379, db=0,
                            socket_keepalive=True, socket_keepalive_options=socket_options)
                update_event.set()  # Make sure to get latest model after connection is interrupted

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
                red.publish("heartbeat", msg)
                if disconnect:
                    logger.info("Connection to server restored")
                    disconnect = False
            except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
                logger.warning("Connection to server interrupted. Trying to reconnect")
                disconnect = True
                time.sleep(10)
                red = Redis(host=address, password=secret, port=6379, db=0)

    @staticmethod
    def _client_shutdown(stop_flag, address, secret):
        red = Redis(host=address, password=secret, port=6379, db=0)
        msg_sub = red.pubsub(ignore_subscribe_messages=True)
        msg_sub.subscribe("client_shutdown")
        while True:
            try:
                if msg_sub.get_message(timeout=1) is None:
                    continue
                logger.info("Received shutdown signal from training node. Exiting training")
                stop_flag.set()
                return
            except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
                time.sleep(10)
                red = Redis(host=address, password=secret, port=6379, db=0)
                msg_sub = red.pubsub(ignore_subscribe_messages=True)
                msg_sub.subscribe("client_shutdown")

    @staticmethod
    def _update_msg(update_flag, address, secret, stop_flag):
        red = Redis(host=address, password=secret, port=6379, db=0, socket_timeout=5.)
        msg_sub = red.pubsub(ignore_subscribe_messages=True)
        msg_sub.subscribe("model_update")
        while not stop_flag.is_set():
            try:
                if msg_sub.get_message(timeout=10.) is None:
                    red.ping()  # Periodically check if the connection is still alive
                    continue
                update_flag.set()
            except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
                red = Redis(host=address, password=secret, port=6379, db=0, socket_timeout=5.)
                # Try to reconnect. Don't reenter main loop since msg_sub may have been reset
                # without properly subscribing and get_message() would throw a RuntimeError
                while not stop_flag.is_set():
                    try:
                        msg_sub = red.pubsub(ignore_subscribe_messages=True)
                        msg_sub.subscribe("model_update")
                        # It is likely a model update has been missed during the update time, so we 
                        # reload the model in any case
                        update_flag.set()
                    except redis.exceptions.ConnectionError:
                        time.sleep(10)  # Immediate reconnect failed. Back off and try again later


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

        self.shutdown = mp.Event()
        self._shutdown_sub = self.red.pubsub(ignore_subscribe_messages=True)
        self._shutdown_sub.subscribe(client_shutdown=self._client_shutdown)
        self._shutdown_sub.run_in_thread(sleep_time=1, daemon=True)

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
            red.publish("heartbeat", msg)

    def sync(self, timeout=100.):
        tstart = time.time()
        msg = None
        while not time.time() - tstart > timeout and not self.shutdown.is_set():
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

    def _client_shutdown(self, _):
        logger.info("Received shutdown signal from training node. Exiting training")
        self.shutdown.set()
