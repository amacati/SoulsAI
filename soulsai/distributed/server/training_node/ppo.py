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
                              **namespace2dict(self.config.ppo.agent.kwargs))
        self.agent.model_id.copy_(torch.tensor([0], dtype=torch.int64))
        if self.config.checkpoint.load:
            self.load_checkpoint(Path(__file__).parents[4] / "saves" / "checkpoint")
            logger.info("Checkpoint loading complete")

        logger.info(f"Initial model ID: {self.agent.model_id}")
        self.buffer = TrajectoryBuffer(**namespace2dict(self.config.ppo.buffer.kwargs))
        self._model_iterations = 0
        logger.info("PPO training node startup complete")

    def _startup_hook(self):
        logger.info("Starting discovery phase")
        self._discover_clients()
        logger.info("Discovery complete, starting training")

    def _get_samples(self) -> list[bytes]:
        return self.red.rpop("samples", 10)  # Batch receive samples

    def _validate_sample(self, sample: dict, monitoring: bool) -> bool:
        valid = sample["model_id"] == self.agent.model_id.item()
        if valid:
            logger.debug(f"Received sample {sample['client_id']}:{sample['step_id']}")
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
        self.red.set("model_state_dict", serialize(self.agent.client_state_dict()))
        self.red.publish("model_update", self.agent.model_id.item())
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
        n_trajectories, n_samples = self.buffer.n_trajectories, self.buffer.n_samples
        t_idx = torch.arange(n_trajectories).repeat(n_samples, 1).T.flatten()
        s_idx = torch.arange(n_samples).repeat(n_trajectories, 1).flatten()
        for _ in range(self.config.ppo.train_epochs):
            # Compute GAE advantage (C6) in each epoch (C5)
            advantage, value = self._compute_advantages()
            assert advantage.shape == (n_trajectories, n_samples), "Advantage shape mismatch"
            assert advantage.shape == value.shape, "advantage and value must have the same shape"
            returns = (advantage + value)
            obs = self.buffer.buffer["obs"].to(self.agent.device)
            probs = self.buffer.buffer["prob"].to(self.agent.device)
            actions = self.buffer.buffer["action"].to(self.agent.device)
            rand_idx = torch.randperm(n_samples * n_trajectories)
            s_idx, t_idx = s_idx[rand_idx], t_idx[rand_idx]
            assert s_idx.shape == (n_samples * n_trajectories,), "s_idx shape mismatch"
            assert s_idx.shape == t_idx.shape, "s_idx and t_idx must have the same shape"
            for j in range(0, self.buffer.n_batch_samples, self.config.ppo.minibatch_size):
                bt_idx = t_idx[j:j + self.config.ppo.minibatch_size]
                bs_idx = s_idx[j:j + self.config.ppo.minibatch_size]
                new_probs = self.agent.get_probs(obs[bt_idx, bs_idx])
                new_probs = torch.gather(new_probs, 1, actions[bt_idx, bs_idx].unsqueeze(-1)).T
                prev_probs = probs[bt_idx, bs_idx].unsqueeze(0)
                assert new_probs.shape == prev_probs.shape, "new_probs shape mismatch"
                ratio = new_probs / prev_probs
                assert ratio.shape == (1, self.config.ppo.minibatch_size), "ratio shape mismatch"
                # Compute policy (actor) loss
                b_advantage = advantage[bt_idx, bs_idx].unsqueeze(0)
                assert b_advantage.shape == (
                    1, self.config.ppo.minibatch_size), "b_advantage shape mismatch"
                policy_loss_1 = -b_advantage * ratio
                policy_loss_2 = -b_advantage * torch.clamp(ratio, 1 - self.config.ppo.clip_range,
                                                           1 + self.config.ppo.clip_range)
                assert policy_loss_1.shape == b_advantage.shape, "policy_loss shape mismatch"
                assert policy_loss_1.shape == policy_loss_2.shape, "policy_loss shape mismatch"
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
                # Compute value (critic) loss
                v_estimate = self.agent.get_values(obs[bt_idx, bs_idx])
                b_returns = returns[bt_idx, bs_idx].unsqueeze(1)
                assert v_estimate.shape == b_returns.shape, "v_estimate | b_returns shape mismatch"
                value_loss = ((v_estimate - b_returns)**2).mean()
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

    def _compute_advantages(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE) advantages from the current buffer."""
        assert self.buffer.buffer_complete, "Buffer must be complete to compute advantages"
        obs = self.buffer.buffer["obs"].to(self.agent.device)
        final_obs = self.buffer.final_buffer["obs"].to(self.agent.device)
        terminated = self.buffer.buffer["terminated"].to(self.agent.device)
        final_terminated = self.buffer.final_buffer["terminated"].to(self.agent.device)
        rewards = self.buffer.buffer["reward"].to(self.agent.device)
        values = self.agent.get_values(obs, requires_grad=False)
        final_values = self.agent.get_values(final_obs, requires_grad=False)
        values = values.squeeze(-1)  # Remove the last dimension
        advantages = torch.zeros_like(values)
        n_trajectories, n_samples = values.shape[0], values.shape[1]
        last_advantage = torch.zeros(n_trajectories, 1, device=values.device)
        step_shape = (n_trajectories, 1)
        for step_id in reversed(range(n_samples)):
            # The estimation computes the advantage values in reverse order by using the
            # value and advantage estimate of time t + 1 for the observation at time t
            if step_id == n_samples - 1:  # Terminal sample. Use actual end values
                not_terminated = (1. - final_terminated.unsqueeze(-1).float())
                next_value = final_values
            else:
                not_terminated = (1. - terminated[:, step_id].unsqueeze(-1).float())
                next_value = values[:, step_id + 1].unsqueeze(-1)
            reward = rewards[:, step_id].unsqueeze(-1)
            assert reward.shape == step_shape, f"Reward shape mismatch ({reward.shape})"
            assert next_value.shape == step_shape, "Next value shape mismatch"
            assert not_terminated.shape == step_shape, "Not terminated shape mismatch"
            td_target = reward + self.config.ppo.gamma * next_value * not_terminated
            assert td_target.shape == step_shape, "TD target shape mismatch"
            value = values[:, step_id].unsqueeze(-1)
            assert value.shape == step_shape, "Value shape mismatch"
            td_error = td_target - value
            assert td_error.shape == step_shape, "TD error shape mismatch"
            last_advantage = td_error + self.config.ppo.gamma * self.config.ppo.gae_lambda * not_terminated * last_advantage  # noqa: E501
            assert last_advantage.shape == step_shape, "Last advantage shape mismatch"
            advantages[:, step_id] = last_advantage.squeeze(-1)
        assert advantages.shape == (n_trajectories, n_samples), "Advantage shape mismatch"
        assert advantages.shape == values.shape, "Advantage shape mismatch"
        return advantages, values

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
            torch.save(self.agent.state_dict(), path / "agent_state_dict.pt")
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
        data["total_steps"] = torch.tensor([self._total_env_steps])
        self.red.publish("telemetry", serialize(data))
