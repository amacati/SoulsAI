"""The PPOTrainingNode implements the classic synchronous PPO algorithm with multiple workers.

It continually receives samples from the clients, trains the model, and broadcasts the new network
to all workers. The workers wait for the new model, and then start to sample the next batch of
trajectories. The algorithm requires all workers to stay connected and is therefore not resilient to
network errors etc.

In our PPO implementation, we use General Advantage Estimation with the design decisions recommended
in https://arxiv.org/pdf/2006.05990.pdf.
"""
from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING

import numpy as np
import torch

from soulsai.core.agent import PPOAgent
from soulsai.core.replay_buffer import TrajectoryBuffer
from soulsai.distributed.server.training_node.training_node import TrainingNode
from soulsai.distributed.common.serialization import serialize, deserialize
from soulsai.utils import namespace2dict
from soulsai.exception import ServerDiscoveryTimeout

if TYPE_CHECKING:
    from types import SimpleNamespace

logger = logging.getLogger(__name__)


class PPOTrainingNode(TrainingNode):
    """PPO training node for distributed, synchronized proximal policy optimization."""

    def __init__(self, config: SimpleNamespace):
        """Set up the Redis connection, initialize the agent and publish the training config.

        Args:
            config: Training configuration.
        """
        logger.info("PPO training node startup")
        super().__init__(config)
        self.agent = PPOAgent(self.config.ppo.actor_net.type,
                              namespace2dict(self.config.ppo.actor_net.kwargs),
                              self.config.ppo.critic_net.type,
                              namespace2dict(self.config.ppo.critic_net.kwargs),
                              self.config.ppo.actor_lr, self.config.ppo.critic_lr, config.device)
        self.agent.model_id = 0
        if self.config.checkpoint.load:
            self.load_checkpoint(Path(__file__).parents[4] / "saves" / "checkpoint")
            logger.info("Checkpoint loading complete")

        logger.info(f"Initial model ID: {self.agent.model_id}")
        self.buffer = TrajectoryBuffer(self.config.ppo.n_clients, self.config.ppo.n_steps,
                                       self.config.env.obs_shape)
        self._model_iterations = 0
        logger.info("PPO training node startup complete")

    def _startup_hook(self):
        logger.info("Starting discovery phase")
        self._discover_clients()
        logger.info("Discovery complete, starting training")

    def _get_samples(self) -> list[bytes]:
        return self.red.rpop("samples", 10)  # Batch receive samples

    def _validate_sample(self, sample: dict, monitoring: bool) -> bool:
        valid = sample["modelId"] == self.agent.model_id
        if valid:
            logger.debug(f"Received sample {sample['clientId']}:{sample['stepId']}")
        else:
            logger.warning("Unexpected sample with outdated model ID")
        if monitoring:
            self.prom_num_samples.inc() if valid else self.prom_num_samples_reject.inc()
        return valid

    def _check_update_cond(self) -> bool:
        return self.buffer.buffer_complete

    def _update_model(self):
        tstart = time.time()
        self._ppo_step()
        self.agent.model_id += 1
        logger.info((f"{time.strftime('%X')}: Model update complete ({time.time() - tstart:.2f}s)"
                     f"\nTotal env steps: {self._total_env_steps}"))

    def _publish_model(self):
        logger.debug(f"Publishing new model iteration {self.agent.model_id}")
        self.red.hset("model_params", mapping=self.agent.serialize(serialize_critic=False))
        self.red.publish("model_update", self.agent.model_id)
        logger.debug("Model upload successful")

    def _post_update_hook(self):
        self.buffer.clear()
        self._model_iterations += 1

    def _check_checkpoint_cond(self) -> bool:
        if not self.config.checkpoint.epochs:
            return False
        return self._model_iterations % self.config.checkpoint.epochs == 0

    def _ppo_step(self):
        # Training algorithm based on Cx recommendations from https://arxiv.org/pdf/2006.05990.pdf
        b_idx = np.arange(self.buffer.n_batch_samples)
        dev = self.config.device
        for _ in range(self.config.ppo.train_epochs):
            # Compute GAE advantage (C6) in each epoch (C5)
            self.buffer.compute_advantages_and_values(self.agent, self.config.gamma,
                                                      self.config.ppo.gae_lambda)
            advantages = self.buffer.advantages.to(dev)
            returns = (self.buffer.advantages + self.buffer.values).to(dev)
            obs = self.buffer.obs.to(dev)
            probs = self.buffer.probs.to(dev)
            actions = self.buffer.actions.to(dev)
            self.np_random.shuffle(b_idx)
            for j in range(0, self.buffer.n_batch_samples, self.config.ppo.minibatch_size):
                mb_idx = b_idx[j:j + self.config.ppo.minibatch_size]
                new_probs = self.agent.get_probs(obs[mb_idx])
                new_probs = torch.gather(new_probs, 1, actions[mb_idx])
                ratio = new_probs / probs[mb_idx]
                # Compute policy (actor) loss
                mb_advantages = advantages[mb_idx]
                policy_loss_1 = -mb_advantages * ratio
                policy_loss_2 = -mb_advantages * torch.clamp(ratio, 1 - self.config.ppo.clip_range,
                                                             1 + self.config.ppo.clip_range)
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
                # Compute value (critic) loss
                v_estimate = self.agent.get_values(obs[mb_idx])
                value_loss = ((v_estimate - returns[mb_idx])**2).mean()
                value_loss *= 0.5 * self.config.ppo.vf_coef
                # Update agent
                self.agent.critic_opt.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.networks["critic"].parameters(),
                                               self.config.ppo.max_grad_norm)
                with self._lock:
                    self.agent.critic_opt.step()
                self.agent.actor_opt.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.networks["actor"].parameters(),
                                               self.config.ppo.max_grad_norm)
                with self._lock:
                    self.agent.actor_opt.step()

    def _discover_clients(self, timeout: float = 60.):
        discovery_sub = self.red.pubsub(ignore_subscribe_messages=True)
        discovery_sub.subscribe("ppo_discovery")
        n_registered = 0
        tstart = time.time()
        while not time.time() - tstart > timeout:
            if not (msg := discovery_sub.get_message()):
                time.sleep(0.5)
                continue
            self.red.publish(msg["data"], n_registered)
            n_registered += 1
            if self.config.monitoring.prometheus:
                self.prom_num_workers.inc()
            logger.info(f"Discovered client {n_registered}/{self.config.ppo.n_clients}")
            if n_registered == self.config.ppo.n_clients:
                return
        raise ServerDiscoveryTimeout("Discovery phase failed to register the required clients")

    def checkpoint(self, path: Path, options: dict = {}):
        """Create a training checkpoint.

        Args:
            path: Path to the save folder.
            options: Additional options dictionary to customize checkpointing.
        """
        path.mkdir(exist_ok=True)
        with self._lock:
            self.agent.save(path / "agent.pt")
        logger.info("Model checkpoint saved")

    def load_checkpoint(self, path: Path):
        """Load a training checkpoint from the folder.

        Args:
            path: Path to the save folder.
        """
        with self._lock:
            self.agent.load(path / "agent.pt")

    def _required_client_ids(self) -> list[int]:
        return list(range(self.config.ppo.n_clients))

    def _episode_info_callback(self, episode_info: bytes):
        data = deserialize(episode_info)
        data["total_steps"] = self._total_env_steps
        self.red.publish("telemetry", serialize(data))
