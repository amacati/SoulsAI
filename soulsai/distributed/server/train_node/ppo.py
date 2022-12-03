import logging
from uuid import uuid4
from pathlib import Path
import time

import numpy as np
import torch

from soulsai.core.agent import PPOAgent
from soulsai.core.replay_buffer import TrajectoryBuffer
from soulsai.distributed.server.train_node.training_node import TrainingNode
from soulsai.utils import namespace2dict
from soulsai.exception import ServerDiscoveryTimeout

logger = logging.getLogger(__name__)


class PPOTrainingNode(TrainingNode):

    def __init__(self, config, decode_sample):
        logger.info("PPO training node startup")
        super().__init__(config, decode_sample)
        self.agent = PPOAgent(self.config.ppo.actor_net_type,
                              namespace2dict(self.config.ppo.actor_net_kwargs),
                              self.config.ppo.critic_net_type,
                              namespace2dict(self.config.ppo.critic_net_kwargs),
                              self.config.ppo.actor_lr,
                              self.config.ppo.critic_lr)
        self.agent.model_id = str(uuid4())
        if self.config.load_checkpoint:
            self.load_checkpoint(Path(__file__).parents[4] / "saves" / "checkpoint")
            logger.info("Checkpoint loading complete")

        logger.info(f"Initial model ID: {self.agent.model_id}")
        self.buffer = TrajectoryBuffer(self.config.ppo.n_clients, self.config.ppo.n_steps,
                                       self.config.n_states, self.config.n_actions)
        self._model_iterations = 0
        logger.info("PPO training node startup complete")

    def _startup_hook(self):
        logger.info("Starting discovery phase")
        self._discover_clients()
        logger.info("Discovery complete, starting training")

    def _validate_sample(self, sample, monitoring):
        valid = sample["model_id"] == self.agent.model_id
        if valid:
            logger.debug(f"Received sample {sample['client_id']}:{sample['step_id']}")
        else:
            logger.warning("Unexpected sample with outdated model ID")
        if monitoring:
            self.prom_num_samples.inc() if valid else self.prom_num_samples_reject.inc()
        return valid

    def _check_update_cond(self):
        return self.buffer.buffer_complete

    def _update_model(self, monitoring):
        tstart = time.time()
        if monitoring:
            with self.prom_update_time.time():
                self._ppo_step()
        else:
            self._ppo_step()
        self.agent.model_id = str(uuid4())
        logger.info((f"{time.strftime('%X')}: Model update complete ({time.time() - tstart:.2f}s)"
                     f"\nTotal env steps: {self._total_env_steps}"))

    def _publish_model(self):
        logger.debug(f"Publishing new model with ID {self.agent.model_id}")
        self.red.hset("model_params", mapping=self.agent.serialize(serialize_critic=False))
        self.red.publish("model_update", self.agent.model_id)
        logger.debug("Model upload successful")

    def _post_update_hook(self):
        self.buffer.clear()
        self._model_iterations += 1

    def _check_checkpoint_cond(self):
        return self._model_iterations % self.config.checkpoint_epochs == 0

    def _ppo_step(self):
        # Training algorithm based on Cx recommendations from https://arxiv.org/pdf/2006.05990.pdf
        b_idx = np.arange(self.buffer.n_batch_samples)
        for _ in range(self.config.ppo.train_epochs):
            # Compute GAE advantage (C6) in each epoch (C5)
            self.buffer.compute_advantages_and_values(self.agent, self.config.gamma,
                                                      self.config.ppo.gae_lambda)
            returns = self.buffer.advantages + self.buffer.values
            self.np_random.shuffle(b_idx)
            for j in range(0, self.buffer.n_batch_samples, self.config.ppo.minibatch_size):
                mb_idx = b_idx[j:j + self.config.ppo.minibatch_size]
                new_prob = self.agent.get_probs(self.buffer.states[mb_idx])
                new_prob = torch.gather(new_prob, 1, self.buffer.actions[mb_idx])
                ratio = new_prob / self.buffer.probs[mb_idx]
                # Compute policy (actor) loss
                mb_advantages = self.buffer.advantages[mb_idx]
                policy_loss_1 = - mb_advantages * ratio
                policy_loss_2 = - mb_advantages * torch.clamp(ratio,
                                                              1 - self.config.ppo.clip_range,
                                                              1 + self.config.ppo.clip_range)
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
                # Compute value (critic) loss
                v_estimate = self.agent.get_values(self.buffer.states[mb_idx])
                value_loss = ((v_estimate - returns[mb_idx])**2).mean()
                value_loss *= 0.5 * self.config.ppo.vf_coef
                # Update agent
                self.agent.critic_opt.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(),
                                               self.config.ppo.max_grad_norm)
                with self._lock:
                    self.agent.critic_opt.step()
                self.agent.actor_opt.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(),
                                               self.config.ppo.max_grad_norm)
                with self._lock:
                    self.agent.actor_opt.step()

    def _discover_clients(self, timeout=60):
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
            if self.config.monitoring.enable:
                self.prom_num_workers.inc()
            logger.info(f"Discovered client {n_registered}/{self.config.ppo.n_clients}")
            if n_registered == self.config.ppo.n_clients:
                return
        raise ServerDiscoveryTimeout("Discovery phase failed to register the required clients")

    def checkpoint(self, path):
        path.mkdir(exist_ok=True)
        with self._lock:
            self.agent.save(path)
        logger.info("Model checkpoint saved")

    def load_checkpoint(self, path):
        with self._lock:
            self.agent.load(path)

    def _required_client_ids(self):
        return list(range(self.config.ppo.n_clients))
