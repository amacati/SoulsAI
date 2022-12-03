from uuid import uuid4
import logging
from pathlib import Path
from collections import deque
import time

import torch

from soulsai.core.replay_buffer import PerformanceBuffer
from soulsai.core.agent import DQNAgent
from soulsai.core.normalizer import Normalizer
from soulsai.core.scheduler import EpsilonScheduler
from soulsai.distributed.server.train_node.training_node import TrainingNode
from soulsai.utils import namespace2dict

logger = logging.getLogger(__name__)


class DQNTrainingNode(TrainingNode):

    def __init__(self, config, decode_sample):
        logger.info("Training node startup")
        super().__init__(config, decode_sample)
        # Translate config params
        if self.config.dqn.min_samples:
            assert self.config.dqn.min_samples <= self.config.dqn.buffer_size
            self._required_samples = max(self.config.dqn.min_samples, self.config.dqn.batch_size)
        else:
            self._required_samples = self.config.dqn.batch_size * self.config.dqn.train_epochs

        self._log_reject = True
        self._model_iterations = 0  # Track number of model iterations for checkpoint trigger
        self.model_ids = deque(maxlen=3)  # Also accept samples from recent model iterations
        self.agent = DQNAgent(self.config.dqn.network_type,
                              namespace2dict(self.config.dqn.network_kwargs),
                              self.config.dqn.lr,
                              self.config.gamma,
                              self.config.dqn.multistep,
                              self.config.dqn.grad_clip,
                              self.config.dqn.q_clip)
        if self.config.dqn.normalizer_kwargs is not None:
            norm_kwargs = namespace2dict(self.config.dqn.normalizer_kwargs) 
        else:
            norm_kwargs = {}
        self.normalizer = Normalizer(self.config.n_states, **norm_kwargs)
        self.buffer = PerformanceBuffer(self.config.dqn.buffer_size, self.config.n_states,
                                        self.config.n_actions, self.config.dqn.action_masking)
        self.eps_scheduler = EpsilonScheduler(self.config.dqn.eps_max, self.config.dqn.eps_min,
                                              self.config.dqn.eps_steps, zero_ending=True)

        if self.config.load_checkpoint:
            self.load_checkpoint(Path(__file__).parents[4] / "checkpoint")
            logger.info("Checkpoint loading complete")

        self.agent.model_id = str(uuid4())
        self.model_ids.append(self.agent.model_id)
        logger.info(f"Initial model ID: {self.agent.model_id}")
        logger.info("DQN training node startup complete")

    def _validate_sample(self, sample, monitoring):
        valid = sample["model_id"] in self.model_ids
        if not valid and self._log_reject:
            logger.warning("Sample ID rejected")
            self._log_reject = False
        if monitoring:
            self.prom_num_samples.inc() if valid else self.prom_num_samples_reject.inc()
        return valid

    def _sample_received_hook(self):
        if self._total_env_steps % self.config.dqn.eps_samples == 0:
            with self._lock:  # Avoid races when checkpointing
                self.eps_scheduler.step()

    def _check_update_cond(self):
        sufficient_samples = len(self.buffer) >= self._required_samples
        return self._total_env_steps % self.config.dqn.update_samples == 0 and sufficient_samples

    def _update_model(self, monitoring):
        tstart = time.time()
        if monitoring:
            with self.prom_update_time.time():
                self._dqn_step()
        else:
            self._dqn_step()
        self.agent.model_id = str(uuid4())
        self.model_ids.append(self.agent.model_id)
        logger.info((f"{time.strftime('%X')}: Model update complete ({time.time() - tstart:.2f}s)"
                     f"\nTotal env steps: {self._total_env_steps}"))

    def _publish_model(self):
        logger.debug(f"Publishing new model with ID {self.agent.model_id}")
        model_params = self.agent.serialize()
        model_params["eps"] = self.eps_scheduler.epsilon
        if self.config.dqn.normalize:
            model_params |= self.normalizer.serialize()
        self.red.hset("model_params", mapping=model_params)
        self.red.publish("model_update", self.agent.model_id)
        logger.debug("Model upload successful")

    def _post_update_hook(self):
        self._log_reject = True
        self._model_iterations += 1

    def _check_checkpoint_cond(self):
        return self._model_iterations % self.config.checkpoint_epochs == 0

    def _dqn_step(self):
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
        with self._lock:
            self.agent.save(path)  # Agent only takes the save directory
            self.buffer.save(path / "buffer.pkl")
            self.eps_scheduler.save(path / "eps_scheduler.json")
            if self.config.dqn.normalize:
                torch.save(self.normalizer.state_dict(), path / "normalizer.pt")
        logger.info("Model checkpoint saved")

    def load_checkpoint(self, path):
        self.agent.load(path)
        self.buffer.load(path / "buffer.pkl")
        self.eps_scheduler.load(path / "eps_scheduler.json")
        if self.config.dqn.normalize:
            self.normalizer.load_state_dict(torch.load(path / "normalizer.pt"))
