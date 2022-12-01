from uuid import uuid4
import logging
import json
from pathlib import Path
from collections import deque
import time
from threading import Lock, Thread
import multiprocessing as mp

from redis import Redis
import torch
from prometheus_client import start_http_server, Counter, Gauge, Info

from soulsai.core.replay_buffer import PerformanceBuffer
from soulsai.core.agent import DQNAgent
from soulsai.core.normalizer import Normalizer
from soulsai.core.scheduler import EpsilonScheduler
from soulsai.utils import load_redis_secret, mkdir_date, dict2namespace, namespace2dict

logger = logging.getLogger(__name__)


class DQNTrainingNode:

    def __init__(self, config, decode_sample):
        logger.info("Training node startup")
        # Set torch settings: Flush denormal floats. See also:
        # https://discuss.pytorch.org/t/training-time-gets-slower-and-slower-on-cpu/145483
        torch.set_flush_denormal(True)
        self.config = config
        self.decode_sample = decode_sample
        self._shutdown_event = mp.Event()
        # Create unique directory
        save_root_dir = Path(__file__).parents[4] / "saves"
        save_root_dir.mkdir(exist_ok=True)
        self.save_dir = mkdir_date(save_root_dir)
        # Load config only if specified
        if self.config.load_checkpoint_config:
            self.load_config(save_root_dir / "checkpoint")
            logger.info("Config loading complete")
        self.config.save_dir = self.save_dir.name
        self.required_samples = self.config.dqn.batch_size # * self.config.dqn.train_epochs
        
        # Read redis server secret
        secret = load_redis_secret(Path(__file__).parents[4] / "config" / "redis.secret")
        self.red = Redis(host='redis', port=6379, password=secret, db=0, decode_responses=True)
        self.sub = self.red.pubsub(ignore_subscribe_messages=True)
        self.cmd_sub = self.red.pubsub(ignore_subscribe_messages=True)
        self.cmd_sub.subscribe(manual_save=lambda _: self.checkpoint(self.save_dir / "manual_save"),
                               save_best=lambda _: self.checkpoint(self.save_dir / "best_model"),
                               shutdown=self.shutdown)
        self.cmd_sub.run_in_thread(sleep_time=1., daemon=True)
        self.lock = Lock()

        self.sub.subscribe("samples")
        self.model_cnt = 0  # Track number of model iterations for checkpoint trigger
        self.model_ids = deque(maxlen=3)  # Also accept samples from recent model iterations

        self.agent = DQNAgent(self.config.dqn.network_type,
                              namespace2dict(self.config.dqn.network_kwargs),
                              self.config.dqn.lr,
                              self.config.gamma,
                              self.config.dqn.multistep,
                              self.config.dqn.grad_clip,
                              self.config.dqn.q_clip)
        self.model_id = str(uuid4())
        self.agent.model_id = self.model_id
        if config.dqn.normalizer_kwargs is not None:
            norm_kwargs = namespace2dict(config.dqn.normalizer_kwargs) 
        else:
            norm_kwargs = {}
        self.normalizer = Normalizer(config.n_states, **norm_kwargs)
        if self.config.load_checkpoint:
            self.load_checkpoint(save_root_dir / "checkpoint")
            logger.info("Checkpoint loading complete")
        self.model_ids.append(self.agent.model_id)
        self.buffer = PerformanceBuffer(self.config.dqn.buffer_size, self.config.n_states,
                                        self.config.n_actions, self.config.dqn.action_masking)
        self.eps_scheduler = EpsilonScheduler(self.config.dqn.eps_max, self.config.dqn.eps_min,
                                              self.config.dqn.eps_steps, zero_ending=True)
        logger.info(f"Initial model ID: {self.model_id}")
        self.checkpoint(self.save_dir)  # Make config accessible for sanity checking

        # Start heartbeat process to track active clients
        self.n_active_clients = mp.Value("i", 0)
        args = (secret, self.n_active_clients, self._shutdown_event)
        self.client_heartbeat = mp.Process(target=self._client_heartbeat, daemon=True, args=args)

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
            self._update_client_count_thread = Thread(target=self._update_client_gauge, daemon=True)
            self._update_client_count_thread.start()  # Manually update value from heartbeat process

        # Upload config to redis to share with client and telemetry node
        logger.info("Saving config to redis for synchronization")
        self.red.set("config", json.dumps(namespace2dict(self.config)))
        self.push_model_update()
        logger.info("Initial model upload successful, startup complete")

    def run(self):
        logger.info("Training node running")
        sample_cnt = 0
        done_cnt = 0
        no_reject = True  # Flag to track if a sample has been rejected during the current iteration
        logger.info("Starting client heartbeat service")
        self.client_heartbeat.start()
        while not self._shutdown_event.is_set():
            msg = self.sub.get_message()
            if not msg:
                time.sleep(0.005)
                continue
            sample = json.loads(msg["data"])
            if not self._check_sample(sample):
                if no_reject:  # Only warn once to avoid log congestion
                    logger.warning("Sample ID rejected")
                    no_reject = False
                    if self.config.monitoring.enable:
                        self.prom_num_samples_reject.inc()
                continue
            sample = self.decode_sample(sample)
            with self.lock:  # Avoid races when checkpointing
                self.buffer.append(sample)
            sample_cnt += 1
            if self.config.monitoring.enable:
                self.prom_num_samples.inc()
            done_cnt += sample[4]
            if (done_cnt / self.config.dqn.multistep) >= 1:
                done_cnt = 0
                with self.lock:  # Avoid races when checkpointing
                    self.eps_scheduler.step()
            sufficient_samples = len(self.buffer) >= self.required_samples
            if sample_cnt >= self.config.dqn.update_samples and sufficient_samples:
                t_start = time.time()
                with self.lock:  # Avoid races when checkpointing
                    self.model_update()
                t_train = time.time() - t_start
                if self.config.monitoring.enable:
                    self.prom_update_time.set(t_train)
                logger.info(f"{time.strftime('%X')}: Model update complete ({t_train:.2f}s)")
                sample_cnt = 0
                no_reject = True
                if self.model_cnt >= self.config.checkpoint_epochs:
                    tstart = time.time()
                    self.checkpoint(self.save_dir)
                    logger.info(f"Training checkpoint successful, took {time.time() - tstart:.2f}s")
            self.model_cnt = 0

        logger.info("Training node has shut down")

    def model_update(self):
        self.train_model()
        self.model_id = str(uuid4())
        self.model_ids.append(self.model_id)
        self.agent.model_id = self.model_id
        self.push_model_update()
        self.model_cnt += 1

    def push_model_update(self):
        logger.debug(f"Publishing new model with ID {self.model_id}")
        model_params = self.agent.serialize()
        model_params["eps"] = self.eps_scheduler.epsilon
        if self.config.dqn.normalize:
            model_params |= self.normalizer.serialize()
        self.red.hset("model_params", mapping=model_params)
        self.red.publish("model_update", self.model_id)
        logger.debug("Model upload successful")

    def _check_sample(self, sample):
        if sample.get("model_id") in self.model_ids:
            return True
        return False

    def train_model(self):
        batches = self.buffer.sample_batches(self.config.dqn.batch_size,
                                             self.config.dqn.train_epochs)
        if self.config.dqn.normalize:
            for batch in batches:
                self.normalizer.update(batch[0])  # Use states to update the normalizer
            for batch in batches:
                batch[0] = self.normalizer.normalize(batch[0])  # Normalize all states for training
                batch[3] = self.normalizer.normalize(batch[3])  # Normalize next states as well
        for batch in batches:
            self.agent.train(*batch)
        self.agent.update_callback()

    def checkpoint(self, path):
        path.mkdir(exist_ok=True)
        with self.lock:
            self.agent.save(path)  # Agent only takes the save directory
            self.buffer.save(path / "buffer.pkl")
            self.eps_scheduler.save(path / "eps_scheduler.json")
            if self.config.dqn.normalize:
                torch.save(self.normalizer.state_dict(), path / "normalizer.pt")
        with open(path / "config.json", "w") as f:
            json.dump(namespace2dict(self.config), f)
        logger.info("Model checkpoint saved")

    def load_checkpoint(self, path):
        self.agent.load(path)
        self.buffer.load(path / "buffer.pkl")
        self.eps_scheduler.load(path / "eps_scheduler.json")
        if self.config.dqn.normalize:
            self.normalizer.load_state_dict(torch.load(path / "normalizer.pt"))

    def load_config(self, path):
        with open(path / "config.json", "r") as f:
            saved_config = dict2namespace(json.load(f))
        assert saved_config.env == self.config.env, "Config environments do not match"
        assert saved_config.algorithm == self.config.algorithm, "Config algorithms do not match"
        self.config = saved_config

    def shutdown(self, _):
        logger.info("Shutdown signaled")
        self._shutdown_event.set()

    def _client_heartbeat(self, redis_secret, n_active_clients, shutdown_event):
        heartbeats = {}
        red = Redis(host='redis', port=6379, password=redis_secret, db=0, decode_responses=True)
        sub = red.pubsub(ignore_subscribe_messages=True)
        sub.subscribe("dqn_heartbeat")
        logger.info("Client heartbeat service started")
        t_last_log = 0
        while not shutdown_event.is_set():
            msg = sub.get_message()  # Does not work with timeout and ignore subscribe
            if not msg:
                time.sleep(1)
            else:
                msg = json.loads(msg["data"])
                if not msg["client_id"] in heartbeats:
                    logger.info("New client registered")
                # Ignore stale messages in the queue
                if time.time() - msg["timestamp"] < 1e3:
                    # Don't use timestamp from message as clocks may have drifted
                    heartbeats[msg["client_id"]] = time.time()
            tnow = time.time()
            heartbeats = {key: t for key, t in heartbeats.items() if tnow - t < 10}
            n_active_clients.value = len(heartbeats)
            if time.time() - t_last_log > 5:
                t_last_log = time.time()
                logger.info(f"n_active_clients: {n_active_clients.value}")

    def _update_client_gauge(self):
        while not self._shutdown_event.is_set():
            self.prom_num_workers.set(self.n_active_clients.value)
            if self.n_active_clients.value == 0:
                self.prom_update_time.set(0)
            time.sleep(1)
