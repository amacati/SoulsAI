from uuid import uuid4
import logging
import json
from pathlib import Path
from collections import deque
from types import SimpleNamespace
import time
from threading import Lock

from redis import Redis

from soulsai.core.replay_buffer import PerformanceBuffer
from soulsai.core.agent import DQNAgent
from soulsai.core.scheduler import EpsilonScheduler
from soulsai.utils import load_redis_secret, mkdir_date

logger = logging.getLogger(__name__)


class DQNTrainingNode:

    def __init__(self, config, decode_sample):
        logger.info("Training node startup")
        self.config = config
        self.decode_sample = decode_sample
        self._shutdown = False
        # Read redis server secret
        secret = load_redis_secret(Path(__file__).parents[4] / "config" / "redis.secret")
        self.red = Redis(host='redis', port=6379, password=secret, db=0, decode_responses=True)
        self.sub = self.red.pubsub(ignore_subscribe_messages=True)
        self.cmd_sub = self.red.pubsub(ignore_subscribe_messages=True)
        self.cmd_sub.psubscribe(manual_save=self.quicksave, shutdown=self.shutdown)
        self.cmd_sub.run_in_thread(sleep_time=1., daemon=True)
        self.lock = Lock()

        # Create unique directory
        save_root_dir = Path(__file__).parents[4] / "saves"
        save_root_dir.mkdir(exist_ok=True)
        self.save_dir = mkdir_date(save_root_dir)

        self.sub.subscribe("samples")
        self.model_cnt = 0  # Track number of model iterations for checkpoint trigger
        self.model_ids = deque(maxlen=3)  # Also accept samples from recent model iterations

        self.agent = DQNAgent(self.config.network_type, self.config.network_kwargs, self.config.lr,
                              self.config.gamma, self.config.dqn_multistep, self.config.grad_clip,
                              self.config.q_clip)
        self.model_id = str(uuid4())
        self.agent.model_id = self.model_id
        self.model_ids.append(self.model_id)
        logger.info(f"Initial model ID: {self.model_id}")

        self.buffer = PerformanceBuffer(self.config.buffer_size, self.config.n_states)
        self.eps_scheduler = EpsilonScheduler(self.config.eps_max, self.config.eps_min,
                                              self.config.eps_steps, zero_ending=True)
        if self.config.load_checkpoint:
            self.load_checkpoint(save_root_dir / "checkpoint")
            logger.info("Checkpoint loading complete")
        else:
            self.checkpoint(self.save_dir)  # Make config accessible for sanity checking
        self.config.save_dir = self.save_dir.name
        # Upload config to redis to share with client and telemetry node
        logger.info("Saving config to redis for synchronization")
        self.red.set("config", json.dumps(vars(self.config)))

        self.push_model_update()
        logger.info("Initial model upload successful, startup complete")

    def run(self):
        logger.info("Training node running")
        sample_cnt = 0
        done_cnt = 0
        no_reject = True  # Flag to track if a sample has been rejected during the current iteration
        while not self._shutdown:
            msg = self.sub.get_message(0.01)
            if not msg:
                continue
            sample = json.loads(msg["data"])
            if not self._check_sample(sample):
                if no_reject:  # Only warn once to avoid log congestion
                    logger.warning("Sample ID rejected")
                    no_reject = False
                continue
            sample = self.decode_sample(sample)
            with self.lock:  # Avoid races when checkpointing
                self.buffer.append(sample)
            sample_cnt += 1
            done_cnt += sample[4]
            if (done_cnt / self.config.dqn_multistep) >= 1:
                done_cnt = 0
                with self.lock:  # Avoid races when checkpointing
                    self.eps_scheduler.step()
            sufficient_samples = len(self.buffer) >= self.config.batch_size
            if sample_cnt >= self.config.update_samples and sufficient_samples:
                with self.lock:  # Avoid races when checkpointing
                    self.model_update()
                logger.info("Model update complete")
                sample_cnt = 0
                no_reject = True
        logger.info("Training node has shut down")

    def model_update(self):
        logger.info("Training model")
        self.train_model()
        self.model_id = str(uuid4())
        self.model_ids.append(self.model_id)
        self.agent.model_id = self.model_id
        self.push_model_update()
        self.model_cnt += 1
        if self.model_cnt >= self.config.checkpoint_epochs:
            tstart = time.time()
            self.checkpoint(self.save_dir)
            logger.info(f"Training checkpoint successful, took {time.time() - tstart:.2f}s")
            self.model_cnt = 0

    def push_model_update(self):
        logger.debug(f"Publishing new model with ID {self.model_id}")
        model_params = self.agent.serialize()
        model_params["eps"] = self.eps_scheduler.epsilon
        self.red.hmset("model_params", model_params)
        self.red.publish("model_update", self.model_id)
        logger.debug("Model upload successful")

    def _check_sample(self, sample):
        if sample.get("model_id") in self.model_ids:
            return True
        return False

    def train_model(self):
        if len(self.buffer) > self.config.batch_size:
            for _ in range(self.config.train_epochs):
                states, actions, rewards, next_states, dones = self.buffer.sample_batch(
                    self.config.batch_size)
                self.agent.train(states, actions, rewards, next_states, dones)
            self.agent.update_callback()

    def checkpoint(self, path):
        logger.info("Checkpointing...")
        path.mkdir(exist_ok=True)
        self.agent.save(path)  # Agent only takes the save directory
        with open(path / "config.json", "w") as f:
            json.dump(vars(self.config), f)
        self.buffer.save(path / "buffer.pkl")
        self.eps_scheduler.save(path / "eps_scheduler.json")
        logger.info("Checkpoint finished")

    def load_checkpoint(self, path):
        self.agent.load(path)
        self.buffer.load(path / "buffer.pkl")
        self.eps_scheduler.load(path / "eps_scheduler.json")
        if self.config.load_checkpoint_config:
            with open(path / "config.json", "r") as f:
                saved_config = SimpleNamespace(**json.load(f))
            assert saved_config.env == self.config.env, "Config environments do not match"
            assert saved_config.algorithm == self.config.algorithm, "Config algorithms do not match"
            self.config = saved_config

    def quicksave(self, _):
        with self.lock:
            self.checkpoint(self.save_dir)

    def shutdown(self, _):
        logger.info("Shutdown signaled")
        self._shutdown = True
