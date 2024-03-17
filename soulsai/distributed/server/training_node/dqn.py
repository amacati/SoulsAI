"""The DQNTrainingNode implements an Ape-X like framework for deep Q learning.

It continually receives samples from the clients, trains the model, and broadcasts the new network
to all workers. Since the training is asynchronous, workers will send samples from previous
network iterations. Therefore, we allow samples from the last 3 network iterations into the replay
buffer. Older samples are rejected and discarded.

This approach also allows us to dynamically add and remove worker nodes, making the overall
architecture more resilient against connection failures, client errors etc.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from redis import Redis

from soulsai.core.agent import Agent, agent_cls
from soulsai.core.replay_buffer import buffer_cls
from soulsai.core.scheduler import Scheduler
from soulsai.core.transform import transform_cls
from soulsai.distributed.common.serialization import deserialize, serialize
from soulsai.distributed.server.training_node.connector import DQNServerConnector
from soulsai.distributed.server.training_node.training_node import TrainingNode
from soulsai.utils import load_redis_secret, namespace2dict

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event
    from types import SimpleNamespace

    from soulsai.core.transform import Transform

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
        # Reduce the number of log messages for logging events
        self._log_times = {"reject_sample": 0, "model_update": 0, "checkpoint": 0}
        self._last_model_log = 0  # Reduce number of log messages for model updates
        self._model_iterations = 0  # Track number of model iterations for checkpoint trigger
        # Also accept samples from recent model iterations
        self.model_ids = deque(maxlen=self.config.dqn.max_model_delay)

        self.agent = agent_cls(self.config.dqn.agent.type)(
            self.config.dqn.network.type,
            namespace2dict(self.config.dqn.network.kwargs),
            **namespace2dict(self.config.dqn.agent.kwargs),
        )
        # Compile all agent networks
        torch.compile(self.agent.networks, mode="reduce-overhead")

        self.transforms: nn.ModuleDict[str, Transform] = nn.ModuleDict()
        kwargs = namespace2dict(getattr(config.dqn.observation_transform, "kwargs", None))
        self.transforms["obs"] = transform_cls(config.dqn.observation_transform.type)(**kwargs)
        kwargs = namespace2dict(getattr(config.dqn.action_transform, "kwargs", None))
        self.transforms["action"] = transform_cls(config.dqn.action_transform.type)(**kwargs)
        self.transforms.to(self.agent.device)

        self.buffer = buffer_cls(self.config.dqn.replay_buffer.type)(
            **namespace2dict(self.config.dqn.replay_buffer.kwargs)
        )

        # Translate config params
        if self.config.dqn.min_samples:
            assert self.config.dqn.min_samples <= self.buffer.size
            self._required_samples = max(self.config.dqn.min_samples, self.config.dqn.batch_size)
        else:
            self._required_samples = self.config.dqn.batch_size * self.config.dqn.train_epochs

        if self.config.checkpoint.load:
            self.load_checkpoint(Path(__file__).parents[4] / "saves/checkpoint")
            logger.info("Checkpoint loading complete")

        # We upload the network weights to Redis in a separate process to avoid blocking the main
        # training loop. The performance impact for small networks is negligible, but for larger
        # networks it can become significant
        self.agent.share_memory()
        self.transforms.share_memory()
        cxt = torch.multiprocessing.get_context("spawn")
        self._publish_model_event = cxt.Event()
        async_upload_kwargs = {
            "event": self._publish_model_event,
            "model": self.agent,
            "transforms": self.transforms,
        }
        self.model_publish_process = cxt.Process(
            target=self._async_publish_model, kwargs=async_upload_kwargs, daemon=True
        )
        self.model_publish_process.start()

        # We also asynchronously receive samples from the clients to interleave the training and
        # sample receiving
        secret = load_redis_secret(Path("/run/secrets/redis_secret"))
        self._sample_connector = DQNServerConnector("redis", secret)

        self.agent.model_id.copy_(torch.tensor([0], dtype=torch.int64))
        self.model_ids.append(self.agent.model_id.item())
        logger.info(f"Initial model ID: {self.agent.model_id.item()}")
        logger.info("DQN training node startup complete")

    def _get_samples(self) -> list[bytes]:
        return self._sample_connector.msgs()

    def _validate_sample(self, sample: dict, monitoring: bool) -> bool:
        # We don't need to lock the model_ids deque here, since sample validation and model id
        # appending are happening in the same thread
        valid = sample["model_id"].item() in self.model_ids
        # Only log once per second
        if not valid and time.time() - self._log_times["reject_sample"] > 1:
            logger.warning("Sample ID rejected")
            self._log_times["reject_sample"] = time.time()
        if monitoring:
            self.prom_num_samples.inc() if valid else self.prom_num_samples_reject.inc()
        return valid

    def _check_update_cond(self) -> bool:
        sufficient_samples = len(self.buffer) >= self._required_samples
        return self._total_env_steps % self.config.dqn.update_samples == 0 and sufficient_samples

    def _update_model(self):
        t1 = time.time()
        self._dqn_step()
        t2 = time.time()
        self.agent.model_id += 1
        with self._lock:  # Prevent deque mutated during iteration error in other threads
            self.model_ids.append(self.agent.model_id.item())
        if time.time() - self._log_times["model_update"] > 10:
            logger.info(
                (
                    f"{time.strftime('%X')}: Model update complete."
                    f"\nTotal env steps: {self._total_env_steps}"
                )
            )
            logger.info(f"Model update took {t2 - t1:.2f} seconds")
            self._log_times["model_update"] = time.time()

    def _publish_model(self):
        logger.debug(f"Publishing new model iteration {self.agent.model_id}")
        self._publish_model_event.set()

    @staticmethod
    def _async_publish_model(event: Event, model: Agent, transforms: nn.ModuleDict):
        """Upload the model to Redis in a separate process.

        Args:
            event: Event to signal when a new model is available.
            model: The neural network.
            transforms: ModuleDict of observation, action etc. transforms.
        """
        secret = load_redis_secret(Path("/run/secrets/redis_secret"))
        red = Redis(host="redis", port=6379, password=secret, db=0)
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
            red.set("model_state_dict", serialize(model.client_state_dict()))
            red.hset(
                "transforms_state_dict",
                mapping={k: serialize(v.state_dict()) for k, v in transforms.items()},
            )
            red.publish("model_update", model.model_id.item())
            logger.debug("Model upload successful")

    def _post_update_hook(self):
        self._model_iterations += 1

    def _check_checkpoint_cond(self) -> bool:
        if not self.config.checkpoint.epochs:
            return False
        return self._model_iterations % self.config.checkpoint.epochs == 0

    def _episode_info_callback(self, episode_info: bytes):
        episode_info = deserialize(episode_info)
        with self._lock:  # Prevent deque mutated during iteration error
            if episode_info["model_id"] not in self.model_ids:
                logger.warning("Stale episode info rejected")
                return
        assert "total_steps" not in episode_info.keys(), "total_steps is a reserved key!"
        episode_info["total_steps"] = torch.tensor([self._total_env_steps])
        # Log the scheduler info to telemetry. Check if any transform contains a scheduler, and if
        # so, add its current value to the episode info
        i = 0
        for name, tf in self.transforms.items():
            for m in tf.modules():
                if isinstance(m, Scheduler):
                    key = name + f".scheduler.{i}"
                    assert key not in episode_info.keys(), f"{key} is already in episode_info!"
                    episode_info[key] = m().unsqueeze(0)
                    i += 1
        # Cast to CPU to avoid deserializing on CUDA on the telemetry server
        self.red.publish("telemetry", serialize(episode_info.cpu()))

    def _dqn_step(self):
        """Sample batches, transform the observations and update the agent."""
        batches = self.buffer.sample_batches(
            self.config.dqn.batch_size, self.config.dqn.train_epochs
        ).to(self.agent.device)
        for batch in batches:
            for tf in self.transforms.values():
                tf.update(batch)  # Update transforms with the new training batches
        # Transform the observations. We have to transform all batches in one go. Otherwise, the
        # updated values can get lost, e.g. if the transformation casts the tensor to a different
        # dtype such as from uint8 to float32. In that case, the float32 tensor would be cast back
        # to uint8, because we are only writing to parts of the tensor. In addition, transforming
        # all batches in one go is faster
        batches = self.transforms["obs"](batches)
        # Use the observation transform also for the next observation
        batches = self.transforms["obs"](batches, keys_mapping={"obs": "next_obs"})

        for batch in batches:
            batch = self.agent.train(batch)
            if self.config.dqn.replay_buffer.type == "PrioritizedReplayBuffer":
                self.buffer.update_priorities(batch)
        self.agent.update_callback()

    def checkpoint(self, path: Path, options: dict = {}):
        """Create a training checkpoint.

        Args:
            path: Path to the save folder.
            options: Additional options dictionary to customize checkpointing.
        """
        if not options.get("manual", False) and not self.config.checkpoint.save:
            if time.time() - self._log_times["checkpoint"] > 10:
                logger.info("Checkpoint saving disabled")
                self._log_times["checkpoint"] = time.time()
            return
        path.mkdir(exist_ok=True)
        with self._lock:
            self.agent.save(path / "agent.pt")
            if self.config.checkpoint.save_buffer or options.get("save_buffer", False):
                self.buffer.save(path / "buffer.pkl")
            for name, tf in self.transforms.items():
                torch.save(tf.state_dict(), path / f"{name}_transform.pt")
            training_stats = {
                "_total_env_steps": self._total_env_steps,
                "_model_iterations": self._model_iterations,
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
        for name, tf in self.transforms.items():
            tf.load_state_dict(torch.load(path / f"{name}_transform.pt"))
        if self.config.checkpoint.load_buffer:
            self.buffer.load(path / "buffer.pkl")
        with open(path / "training_stats.json", "r") as f:
            stats = json.load(f)
        self._total_env_steps = stats["_total_env_steps"]
        self.prom_num_samples.inc(self._total_env_steps)
        self._model_iterations = stats["_model_iterations"]
