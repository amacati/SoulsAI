from uuid import uuid4
import logging
import json
from pathlib import Path
from collections import deque
from types import SimpleNamespace
import time

from redis import Redis

from soulsai.core.replay_buffer import ExperienceReplayBuffer, PerformanceBuffer
from soulsai.core.agent import DQNAgent
from soulsai.core.scheduler import EpsilonScheduler
from soulsai.utils import load_redis_secret

logger = logging.getLogger(__name__)


class TrainingNode:

    SAVE_PATH = Path(__file__).parent / "save"

    def __init__(self, config, decode_sample):
        logger.info("Training node startup")
        self.config = config
        self.decode_sample = decode_sample
        # Read redis server secret
        secret = load_redis_secret(Path(__file__).parents[4] / "config" / "redis.secret")
        self.red = Redis(host='redis', port=6379, password=secret, db=0, decode_responses=True)
        self.sub = self.red.pubsub(ignore_subscribe_messages=True)
        self.sub.subscribe("samples")
        self.sample_cnt = 0  # Track number of samples for training trigger
        self.model_cnt = 0  # Track number of model iterations for checkpoint trigger
        self.done_cnt = 0  # Track number of completed trajectories for epsilon decay
        self.model_ids = deque(maxlen=3)  # Also accept samples from recent model iterations

        self.agent = DQNAgent(self.config.n_states, self.config.n_actions, self.config.lr,
                              self.config.gamma, self.config.dqn_multistep, self.config.grad_clip,
                              self.config.q_clip, config.layer_width)
        self.model_id = str(uuid4())
        self.agent.model_id = self.model_id
        self.model_ids.append(self.model_id)
        logger.info(f"Initial model ID: {self.model_id}")

        self.buffer = PerformanceBuffer(self.config.buffer_size, self.config.n_states)
        self.eps_scheduler = EpsilonScheduler(self.config.eps_max, self.config.eps_min,
                                              self.config.eps_steps, zero_ending=True)
        if self.config.load_checkpoint:
            self.load_checkpoint()
            logger.info("Checkpoint loading complete")
        else:
            self.checkpoint()  # Checkpoint to make config accessible for sanity checking
        self.push_model_update()
        logger.info("Initial model upload successful, startup complete")

    def run(self):
        logger.info("Training node running")
        while True:
            msg = self.sub.get_message()
            if not msg:
                time.sleep(0.01)
                continue
            sample = json.loads(msg["data"])
            if not self._check_sample(sample):
                continue
            sample = self.decode_sample(sample)
            self.buffer.append(sample)
            self.sample_cnt += 1
            self.done_cnt += sample[4]
            if (self.done_cnt / self.config.dqn_multistep) >= 1:
                self.done_cnt = 0
                self.eps_scheduler.step()
            sufficient_samples = len(self.buffer) >= self.config.batch_size
            if self.sample_cnt >= self.config.update_samples and sufficient_samples:
                self.model_update()
                self.sample_cnt = 0

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
            self.checkpoint()
            logger.info(f"Training checkpoint successful, took {time.time() - tstart:.2f}s")
            self.model_cnt = 0

    def push_model_update(self):
        logger.info(f"Publishing new model with ID {self.model_id}")
        model_params = self.agent.serialize()
        model_params["eps"] = self.eps_scheduler.epsilon
        self.red.hmset("model_params", model_params)
        self.red.publish("model_update", self.model_id)
        logger.info("Model update successful")

    def _check_sample(self, sample):
        if sample.get("model_id") in self.model_ids:
            logger.debug("Sample ID accepted")
            return True
        logger.warning("Sample ID rejected")
        return False

    def train_model(self):
        if len(self.buffer) > self.config.batch_size:
            for _ in range(self.config.train_epochs):
                states, actions, rewards, next_states, dones = self.buffer.sample_batch(
                    self.config.batch_size)
                self.agent.train(states, actions, rewards, next_states, dones)

    def checkpoint(self):
        self.SAVE_PATH.mkdir(exist_ok=True)
        self.agent.save(self.SAVE_PATH)  # Agent only takes the save directory
        with open(self.SAVE_PATH / "config.json", "w") as f:
            json.dump(vars(self.config), f)
        self.buffer.save(self.SAVE_PATH / "buffer.pkl")
        self.eps_scheduler.save(self.SAVE_PATH / "eps_scheduler.json")

    def load_checkpoint(self):
        self.agent.load(self.SAVE_PATH)
        self.buffer.load(self.SAVE_PATH / "buffer.pkl")
        self.eps_scheduler.load(self.SAVE_PATH / "eps_scheduler.json")
        if self.config.load_checkpoint_config:
            with open(self.SAVE_PATH / "config.json", "r") as f:
                self.config = SimpleNamespace(**json.load(f))

    def load_buffer(self, nsamples=None):
        nsamples = nsamples or self.buffer.maxlen
        random_buffer = ExperienceReplayBuffer()
        random_buffer.load(self.SAVE_PATH / "random_buffer.pkl")
        assert len(random_buffer) >= nsamples
        self.buffer.clear()
        while not len(self.buffer) == nsamples:
            self.buffer.append(self._numpify_sample(random_buffer.buffer.pop()))
        logger.info("Loaded buffer with neutral samples")
