"""The DQNTrainingNode implements an Ape-X like framework for deep Q learning.

It continually receives samples from the clients, trains the model, and broadcasts the new network
to all workers. Since the training is asynchronous, workers will send samples from previous
network iterations. Therefore, we allow samples from the last 3 network iterations into the replay
buffer. Older samples are rejected and discarded.

This approach also allows us to dynamically add and remove worker nodes, making the overall
architecture more resilient against connection failures, client errors etc.
"""
from __future__ import annotations

import logging
from pathlib import Path
from collections import deque
import time
from typing import TYPE_CHECKING
import json
from multiprocessing.synchronize import Event

import torch
from redis import Redis

from soulsai.core.replay_buffer import buffer_cls
from soulsai.core.agent import DistributionalDQNAgent, DQNAgent
from soulsai.core.normalizer import normalizer_cls
from soulsai.core.scheduler import EpsilonScheduler
from soulsai.distributed.common.serialization import serialize, deserialize
from soulsai.distributed.server.training_node.training_node import TrainingNode
from soulsai.distributed.server.training_node.connector import DQNServerConnector
from soulsai.utils import namespace2dict, load_redis_secret

if TYPE_CHECKING:
    from types import SimpleNamespace

    from soulsai.core.normalizer import AbstractNormalizer

logger = logging.getLogger(__name__)


class DQNTrainingNode(TrainingNode):
    """DQN training node for distributed Q learning."""

    def __init__(self, config: SimpleNamespace):
        """Set up the Redis connection, initialize the agent and publish the training config.

        Args:
            config: Training configuration.
        """
        logger.info("DQN training node startup")
        super().__init__(config)
        # Translate config params
        if self.config.dqn.min_samples:
            assert self.config.dqn.min_samples <= self.config.dqn.buffer_size
            self._required_samples = max(self.config.dqn.min_samples, self.config.dqn.batch_size)
        else:
            self._required_samples = self.config.dqn.batch_size * self.config.dqn.train_epochs

        self._log_reject_time = time.time()  # Only log sample rejects once per second
        self._last_model_log = 0  # Reduce number of log messages for model updates
        self._model_iterations = 0  # Track number of model iterations for checkpoint trigger
        # Also accept samples from recent model iterations
        self.model_ids = deque(maxlen=self.config.dqn.max_model_delay)
        if self.config.dqn.variant == "distributional":
            self.agent = DistributionalDQNAgent(self.config.dqn.network_type,
                                                namespace2dict(self.config.dqn.network_kwargs),
                                                self.config.dqn.lr, self.config.gamma,
                                                self.config.dqn.multistep,
                                                self.config.dqn.grad_clip, self.config.dqn.q_clip,
                                                self.config.device)
        elif self.config.dqn.variant == "vanilla":
            self.agent = DQNAgent(self.config.dqn.network_type,
                                  namespace2dict(self.config.dqn.network_kwargs),
                                  self.config.dqn.lr, self.config.gamma, self.config.dqn.multistep,
                                  self.config.dqn.grad_clip, self.config.dqn.q_clip,
                                  self.config.device)
        else:
            raise ValueError(f"DQN variant {self.config.dqn.variant} is not supported")
        # Compile all agent networks
        torch.compile(self.agent.networks, mode="reduce-overhead")

        self.normalizer = None
        if self.config.dqn.normalizer:
            cls = normalizer_cls(self.config.dqn.normalizer)
            normalizer_kwargs = namespace2dict(self.config.dqn.normalizer_kwargs)
            self.normalizer = cls(self.config.env.obs_shape, **normalizer_kwargs)
        buffer_kwargs = {}
        if self.config.dqn.replay_buffer_kwargs is not None:
            buffer_kwargs = namespace2dict(self.config.dqn.replay_buffer_kwargs)
        self.buffer = buffer_cls(self.config.dqn.replay_buffer)(**buffer_kwargs)
        self.eps_scheduler = EpsilonScheduler(self.config.dqn.eps_max,
                                              self.config.dqn.eps_min,
                                              self.config.dqn.eps_steps,
                                              zero_ending=True)

        if self.config.checkpoint.load:
            self.load_checkpoint(Path(__file__).parents[4] / "saves/checkpoint")
            logger.info("Checkpoint loading complete")

        # We upload the network weights to Redis in a separate process to avoid blocking the main
        # training loop. The performance impact for small networks is negligible, but for larger
        # networks it can become significant
        self.agent.share_memory()
        self.eps_scheduler.share_memory()
        cxt = torch.multiprocessing.get_context("spawn")
        self._publish_model_event = cxt.Event()
        async_upload_kwargs = {
            "event": self._publish_model_event,
            "model": self.agent,
            "epsilon_scheduler": self.eps_scheduler
        }
        if self.config.dqn.normalizer:
            self.normalizer.share_memory()
            async_upload_kwargs["normalizer"] = self.normalizer
        self.model_publish_process = cxt.Process(target=self._async_publish_model,
                                                 kwargs=async_upload_kwargs,
                                                 daemon=True)
        self.model_publish_process.start()

        # We also asynchronously receive samples from the clients to interleave the training and
        # sample receiving
        secret = load_redis_secret(Path("/run/secrets/redis_secret"))
        self._sample_connector = DQNServerConnector("redis", secret)

        self.agent.model_id = 0
        self.model_ids.append(self.agent.model_id)
        logger.info(f"Initial model ID: {self.agent.model_id}")
        logger.info("DQN training node startup complete")

    def _get_samples(self) -> list[bytes]:
        return self._sample_connector.msgs()

    def _validate_sample(self, sample: dict, monitoring: bool) -> bool:
        valid = sample["model_id"] in self.model_ids
        if not valid and time.time() - self._log_reject_time > 1:  # Only log once per second
            logger.warning("Sample ID rejected")
            self._log_reject_time = time.time()
        if monitoring:
            self.prom_num_samples.inc() if valid else self.prom_num_samples_reject.inc()
        return valid

    def _sample_received_hook(self, _: dict):
        self.eps_scheduler.step()

    def _check_update_cond(self) -> bool:
        sufficient_samples = len(self.buffer) >= self._required_samples
        return self._total_env_steps % self.config.dqn.update_samples == 0 and sufficient_samples

    def _update_model(self):
        t1 = time.time()
        self._dqn_step()
        t2 = time.time()
        self.agent.model_id += 1
        self.model_ids.append(self.agent.model_id)
        if time.time() - self._last_model_log > 10:
            logger.info((f"{time.strftime('%X')}: Model update complete."
                         f"\nTotal env steps: {self._total_env_steps}"))
            logger.info(f"Model update took {t2 - t1:.2f} seconds")
            self._last_model_log = time.time()

    def _publish_model(self):
        logger.debug(f"Publishing new model iteration {self.agent.model_id}")
        self._publish_model_event.set()

    @staticmethod
    def _async_publish_model(event: Event,
                             model: DQNAgent,
                             epsilon_scheduler: EpsilonScheduler,
                             normalizer: AbstractNormalizer | None = None):
        """Upload the model to Redis in a separate process.

        Args:
            event: Event to signal when a new model is available.
            model: The neural network.
            epsilon_scheduler: The epsilon scheduler for epsilon-greedy policies.
            normalizer: The optional normalizer.
        """
        secret = load_redis_secret(Path("/run/secrets/redis_secret"))
        red = Redis(host='redis', port=6379, password=secret, db=0)
        last_log = time.time()  # Reduce number of log messages for model updates
        while True:
            t = time.perf_counter()
            event.wait()
            # If the model update is faster than the upload, this flag will be set to True when the
            # loop starts again. This will cause model generations to be skipped, which might be
            # problematic. Therefore we log a warning
            if time.perf_counter() - t < 1e-5:
                if time.time() - last_log > 1:  # Only log once per second
                    logger.warning("Model upload can't keep up with model updates!")
                    last_log = time.time()
            event.clear()
            model_params = model.serialize()
            model_params["eps"] = epsilon_scheduler.epsilon
            if normalizer is not None:
                model_params |= normalizer.serialize()
            red.hset("model_params", mapping=model_params)
            red.publish("model_update", model.model_id)
            logger.debug("Model upload successful")

    def _post_update_hook(self):
        self._model_iterations += 1

    def _check_checkpoint_cond(self) -> bool:
        if not self.config.checkpoint.epochs:
            return False
        return self._model_iterations % self.config.checkpoint.epochs == 0

    def _episode_info_callback(self, episode_info: bytes):
        episode_info = deserialize(episode_info)
        if not episode_info["model_id"] in self.model_ids:
            logger.warning("Stale episode info rejected")
            return
        episode_info["total_steps"] = torch.tensor([self._total_env_steps])
        self.red.publish("telemetry", serialize(episode_info))

    def _dqn_step(self):
        """Sample batches, normalize the values if applicable, and update the agent."""
        batches = self.buffer.sample_batches(self.config.dqn.batch_size,
                                             self.config.dqn.train_epochs)
        if self.config.dqn.normalizer:
            for batch in batches:
                self.normalizer.update(batch[0])  # Use observations to update the normalizer
            for batch in batches:
                batch[0] = self.normalizer.normalize(batch[0])  # Normalize all observations
                batch[3] = self.normalizer.normalize(batch[3])  # Normalize next observations too
        for batch in batches:
            if self.config.dqn.replay_buffer == "PrioritizedReplayBuffer":
                # -2 because the last two elements are the weights and indices
                td_errors = self.agent.train(*batch[:-2], weights=batch[-2])
            else:
                self.agent.train(*batch)
            if self.config.dqn.replay_buffer == "PrioritizedReplayBuffer":
                self.buffer.update_priorities(batch[-1], td_errors)
        self.agent.update_callback()

    def checkpoint(self, path: Path, options: dict = {}):
        """Create a training checkpoint.

        Args:
            path: Path to the save folder.
            options: Additional options dictionary to customize checkpointing.
        """
        if not options.get("manual", False) and not self.config.checkpoint.save:
            logger.info("Checkpoint saving disabled")
            return
        path.mkdir(exist_ok=True)
        with self._lock:
            self.agent.save(path / "agent.pt")
            if self.config.checkpoint.save_buffer or options.get("save_buffer", False):
                self.buffer.save(path / "buffer.pkl")
            self.eps_scheduler.save(path / "eps_scheduler.json")
            if self.config.dqn.normalizer:
                torch.save(self.normalizer.state_dict(), path / "normalizer.pt")
            training_stats = {
                "_total_env_steps": self._total_env_steps,
                "_model_iterations": self._model_iterations
            }
            with open(path / "training_stats.json", "w") as f:
                json.dump(training_stats, f)
        logger.info("Model checkpoint saved")

    def load_checkpoint(self, path: Path):
        """Load a training checkpoint from the folder.

        Args:
            path: Path to the save folder.
        """
        self.agent.load(path / "agent.pt")
        if self.config.checkpoint.load_buffer:
            self.buffer.load(path / "buffer.pkl")
        self.eps_scheduler.load(path / "eps_scheduler.json")
        if self.config.dqn.normalizer:
            self.normalizer.load_state_dict(torch.load(path / "normalizer.pt"))
        with open(path / "training_stats.json", "r") as f:
            stats = json.load(f)
        self._total_env_steps = stats["_total_env_steps"]
        self.prom_num_samples.inc(self._total_env_steps)
        self._model_iterations = stats["_model_iterations"]
