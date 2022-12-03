from pathlib import Path
import logging
import multiprocessing as mp
import json
import time
from abc import abstractmethod
from threading import Thread

import numpy as np
from redis import Redis
import torch
from prometheus_client import start_http_server, Info, Counter, Gauge

from soulsai.utils import mkdir_date, load_redis_secret, namespace2dict, dict2namespace

logger = logging.getLogger(__name__)


class TrainingNode:

    def __init__(self, config, decode_sample):
        # Set torch settings: Flush denormal floats. See also:
        # https://discuss.pytorch.org/t/training-time-gets-slower-and-slower-on-cpu/145483
        torch.set_flush_denormal(True)
        self.np_random = np.random.default_rng()  # https://numpy.org/neps/nep-0019-rng-policy.html
        self.decode_sample = decode_sample
        self._shutdown = mp.Event()
        self._lock = mp.Lock()
        # Create unique directory for saves, save to 
        save_root_dir = Path(__file__).parents[4] / "saves"
        save_root_dir.mkdir(exist_ok=True)
        self.save_dir = mkdir_date(save_root_dir)
        # Set config, load from checkpoint if specified
        self.config = config
        if self.config.load_checkpoint_config:
            self.load_config(save_root_dir / "checkpoint")
            logger.info("Config loading complete")
        self.config.save_dir = self.save_dir.name
        self.save_config(self.save_dir)
        # Translate config values that are incompatible with json
        if not self.config.max_env_steps:
            self._max_env_steps = float("inf")
        else:
            self._max_env_steps = self.config.max_env_steps
        # Load redis secret, create redis connection and subscribers
        secret = load_redis_secret(Path(__file__).parents[4] / "config" / "redis.secret")
        self.red = Redis(host='redis', port=6379, password=secret, db=0, decode_responses=True)
        self.sample_sub = self.red.pubsub(ignore_subscribe_messages=True)
        self.sample_sub.subscribe("samples")
        self.cmd_sub = self.red.pubsub(ignore_subscribe_messages=True)
        self.cmd_sub.subscribe(manual_save=lambda _: self.checkpoint(self.save_dir),
                               save_best=lambda _: self.checkpoint(self.save_dir / "best_model"),
                               shutdown=self.shutdown)
        self._cmd_sub_worker = self.cmd_sub.run_in_thread(sleep_time=.1, daemon=True)
        # Initialize monitoring server and metrics
        if self.config.monitoring.enable:
            logger.info("Starting prometheus monitoring server")
            start_http_server(8080)
            self.prom_num_workers = Gauge("soulsai_num_workers",
                                          "Number of registered client nodes")
            self.prom_num_samples = Counter("soulsai_num_samples",
                                            "Total number of received samples")
            self.prom_num_samples_reject = Counter("soulsai_num_samples_reject",
                                                   "Total number of rejected samples")
            self.prom_update_time = Gauge("soulsai_update_duration",
                                          "Processing time for a model update")
            self.prom_config_info = Info("soulsai_config", "SoulsAI configuration")
            self.prom_config_info.info({str(key): str(val) for key, val in
                                        namespace2dict(self.config).items()})
            self._update_client_gauge_thread = Thread(target=self._update_client_gauge, daemon=True)
            self._update_client_gauge_thread.start()

        self._total_env_steps = 0
        self._client_counter = mp.Value("i", 0)
        # Start heartbeat service
        args = (secret, self._shutdown, self._client_counter, self._required_client_ids())
        self.client_heartbeat = mp.Process(target=self._client_heartbeat, daemon=True, args=args)
        # Upload config to redis to share with client and telemetry node
        logger.info("Saving config to redis for synchronization")
        self.red.set("config", json.dumps(namespace2dict(self.config)))

    def run(self):
        self._startup_hook()
        self.client_heartbeat.start()
        self._publish_model()
        while not self._shutdown.is_set():
            if not (msg := self.sample_sub.get_message()):
                time.sleep(0.005)
                continue
            sample = json.loads(msg["data"])
            if not self._validate_sample(sample, monitoring=self.config.monitoring.enable):
                continue
            self._total_env_steps += 1
            with self._lock:
                self.buffer.append(self.decode_sample(sample))
            self._sample_received_hook()
            if self._check_update_cond():
                self._update_model(monitoring=self.config.monitoring.enable)
                self._publish_model()
                self._post_update_hook()
                if self._check_checkpoint_cond():
                    self.checkpoint(self.save_dir)
            if self._max_env_steps < self._total_env_steps:
                logger.info("Maximum samples reached. Shutting down training node.")
                self.red.publish("client_shutdown", "")
                self._shutdown.set()
        self.checkpoint(self.save_dir)
        logger.info("Training node has shut down")

    def save_config(self, path):
        path.mkdir(exist_ok=True)
        with open(path / "config.json", "w")  as f:
            json.dump(namespace2dict(self.config), f)

    def load_config(self, path):
        with open(path / "config.json", "r") as f:
            saved_config = dict2namespace(json.load(f))
        assert saved_config.env == self.config.env, "Config environments do not match"
        assert saved_config.algorithm == self.config.algorithm, "Config algorithms do not match"
        self.config = saved_config

    @staticmethod
    def _client_heartbeat(redis_secret, shutdown_event, client_counter, required_client_ids=[]):
        red = Redis(host='redis', port=6379, password=redis_secret, db=0, decode_responses=True)
        sub = red.pubsub(ignore_subscribe_messages=True)
        sub.subscribe("heartbeat")
        logger.info("Client heartbeat service started")
        t_last_log = 0
        heartbeats = {client_id: time.time() for client_id in required_client_ids}
        while not shutdown_event.is_set():
            if not (msg := sub.get_message()):  # Does not work with timeout and ignore subscribe
                time.sleep(1)
            else:
                msg = json.loads(msg["data"])
                if not msg["client_id"] in heartbeats:
                    logger.info("New client registered")
                if time.time() - msg["timestamp"] < 1e3:
                    # Don't use timestamp from message as clocks may have drifted
                    heartbeats[msg["client_id"]] = time.time()
            tnow = time.time()
            heartbeats = {key: t for key, t in heartbeats.items() if tnow - t < 10}
            logger.debug(f"{time.strftime('%X')}: Heartbeat ok")
            client_counter.value = len(heartbeats)
            if time.time() - t_last_log > 5:
                t_last_log = time.time()
                logger.info(f"n_active_clients: {client_counter.value}")
            if not all(client_id in heartbeats for client_id in required_client_ids):
                logger.error(f"Missing required client heartbeats. Shutting down training")
                shutdown_event.set()

    def _update_client_gauge(self):
        while not self._shutdown.is_set():
            self.prom_num_workers.set(self._client_counter.value)
            if self._client_counter.value == 0:
                self.prom_update_time.set(0)
            time.sleep(1)

    def shutdown(self, _):
        logger.info("Shutdown signaled")
        self._shutdown.set()

    @abstractmethod
    def checkpoint(self, path):
        ...

    @abstractmethod
    def load_checkpoint(self, path):
        ...

    @abstractmethod
    def _validate_sample(self, sample, log):
        ...

    @abstractmethod
    def _update_model(self, log):
        ...

    @abstractmethod
    def _publish_model(self):
        ...

    @abstractmethod
    def _check_update_cond(self):
        ...

    @abstractmethod
    def _check_checkpoint_cond(self):
        ...

    def _startup_hook(self):
        ...

    def _sample_received_hook(self):
        ...

    def _post_update_hook(self):
        ...

    def _required_client_ids(self):
        return []
