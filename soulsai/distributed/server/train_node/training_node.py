from uuid import uuid4
import logging
import json
from pathlib import Path
from collections import deque

import numpy as np
import redis
from soulsgym.core.game_state import GameState

from soulsai.core.replay_buffer import ExperienceReplayBuffer
from soulsai.core.agent import DQNAgent
from soulsai.core.scheduler import EpsilonScheduler
from soulsai.core.utils import gamestate2np

logger = logging.getLogger(__name__)


class TrainingNode:

    def __init__(self):
        logger.info("Training node startup")
        # Read redis server secret
        with open(Path(__file__).parents[1] / "redis.secret") as f:
            conf = f.readlines()
        secret = None
        for line in conf:
            if len(line) > 12 and line[0:12] == "requirepass ":
                secret = line[12:]
                break
        if secret is None:
            raise RuntimeError("Missing password configuration for redis in redis.secret")

        self.red = redis.Redis(host='redis', port=6379, password=secret, db=0,
                               decode_responses=True)
        self.sub = self.red.pubsub(ignore_subscribe_messages=True)
        self.sub.subscribe("samples")
        self.sample_cnt = 0
        self.model_id = str(uuid4())
        self.model_ids = deque(maxlen=3)  # Also accept samples from recent model iterations
        self.model_ids.append(self.model_id)
        self.red.set("model_id", self.model_id)
        logger.info(f"Initial model ID: {self.model_id}")

        # Learning initialization
        lr = 1e-3
        gamma = 0.99
        eps_max = [0.99, 0.05, 0.05]
        eps_min = [0.05, 0.05, 0.01]
        eps_steps = [1500, 1500, 1500]
        grad_clip = 100.  # 1.5
        q_clip = 200.
        buffer_size = 100_000
        n_states = 72
        n_actions = 20
        self.batch_size = 64
        self.train_epochs = 5
        self.n_update_samples = 10

        self.agent = DQNAgent(n_states, n_actions, lr, gamma, grad_clip, q_clip)
        self.buffer = ExperienceReplayBuffer(maxlen=buffer_size)
        self.eps_scheduler = EpsilonScheduler(eps_max, eps_min, eps_steps, zero_ending=True)
        self.push_model_update()
        logger.info("Initial model upload successful, startup complete")

    def run(self):
        logger.info("Training node running")
        while True:
            msg = self.sub.get_message()
            if not msg:
                continue
            sample = json.loads(msg["data"])
            if not self._check_sample(sample):
                continue
            experience = sample.get("sample")
            experience[0] = GameState.from_dict(experience[0])
            experience[3] = GameState.from_dict(experience[3])
            self.buffer.append(experience)
            self.sample_cnt += 1
            if self.sample_cnt >= self.n_update_samples and len(self.buffer) == self.buffer.maxlen:
                self.model_update()
                self.sample_cnt = 0

    def model_update(self):
        logger.info("Training model")
        self.train_model()
        self.red.delete(self.model_id)
        self.model_id = str(uuid4())
        self.model_ids.append(self.model_id)
        self.push_model_update()

    def push_model_update(self):
        logger.info(f"Publishing new model with ID {self.model_id}")
        model_params = self.agent.serialize()
        model_params["eps"] = self.eps_scheduler.epsilon
        self.red.hmset(self.model_id, model_params)
        self.red.set("model_id", self.model_id)
        self.red.publish("model_update", self.model_id)
        logger.info("Model update successful")

    def _check_sample(self, sample):
        if sample.get("model_id") in self.model_ids:
            logger.debug("Sample ID accepted")
            return True
        logger.debug("Sample ID rejected")
        return False

    def train_model(self):
        if len(self.buffer) > self.batch_size:
            for _ in range(self.train_epochs):
                states, actions, rewards, next_states, dones = self.buffer.sample_batch(
                    self.batch_size)
                states = np.array([gamestate2np(state) for state in states])
                next_states = np.array([gamestate2np(next_state) for next_state in next_states])
                actions, rewards, dones = map(np.array, (actions, rewards, dones))
                self.agent.train(states, actions, rewards, next_states, dones)
            self.eps_scheduler.step()
