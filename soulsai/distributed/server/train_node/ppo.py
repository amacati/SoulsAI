import logging
from types import SimpleNamespace
from uuid import uuid4
from pathlib import Path
from threading import Lock
import json
import multiprocessing as mp
import time
import uuid

from redis import Redis
import numpy as np
import torch

from soulsai.core.agent import PPOAgent
from soulsai.core.replay_buffer import TrajectoryBuffer
from soulsai.utils import load_redis_secret, mkdir_date
from soulsai.exception import ServerDiscoveryTimeout

logger = logging.getLogger(__name__)


class PPOTrainingNode:

    def __init__(self, config, decode_sample):
        logger.info("PPO training node startup")
        self.config = config
        self.decode_sample = decode_sample
        self._shutdown = mp.Event()
        # Read redis server secret
        secret = load_redis_secret(Path(__file__).parents[4] / "config" / "redis.secret")
        self.red = Redis(host='redis', port=6379, password=secret, db=0, decode_responses=True)
        self.sub = self.red.pubsub(ignore_subscribe_messages=True)
        self.cmd_sub = self.red.pubsub(ignore_subscribe_messages=True)
        self.cmd_sub.psubscribe(manual_save=self.quicksave, shutdown=self.shutdown)
        self.cmd_sub.run_in_thread(sleep_time=1., daemon=True)
        self.lock = Lock()

        args = (self.config.ppo["n_clients"], secret, self._shutdown)
        self.client_heartbeat = mp.Process(target=self._client_heartbeat, daemon=True, args=args)

        # Create unique directory
        save_root_dir = Path(__file__).parents[4] / "saves"
        save_root_dir.mkdir(exist_ok=True)
        self.save_dir = mkdir_date(save_root_dir)

        self.sub.subscribe("samples")
        self.total_env_steps = 0

        self.agent = PPOAgent(self.config.ppo["actor_net_type"],
                              self.config.ppo["actor_net_kwargs"],
                              self.config.ppo["critic_net_type"],
                              self.config.ppo["critic_net_kwargs"],
                              self.config.ppo["actor_lr"],
                              self.config.ppo["critic_lr"],
                              self.config.gamma,
                              self.config.grad_clip)
        self.agent.model_id = str(uuid4())
        logger.info(f"Initial model ID: {self.agent.model_id}")
        self.buffer = TrajectoryBuffer(self.config.ppo["n_clients"], self.config.ppo["n_steps"],
                                       self.config.n_states, self.config.n_actions)
        if self.config.load_checkpoint:
            self.load_checkpoint(save_root_dir / "checkpoint")
            logger.info("Checkpoint loading complete")
        else:
            self.checkpoint(self.save_dir)  # Make config accessible for sanity checking
        self.config.save_dir = self.save_dir.name
        self.np_random = np.random.default_rng()  # https://numpy.org/neps/nep-0019-rng-policy.html
        # Upload config to redis to share with client and telemetry node
        logger.info("Saving config to redis for synchronization")
        self.red.set("config", json.dumps(vars(self.config)))
        logger.info("Startup complete")

    def run(self):
        logger.info("Starting discovery phase")
        self.discover_clients()
        logger.info("Discovery complete, starting training")
        self.client_heartbeat.start()
        self.push_model_update()
        while not self._shutdown.is_set():
            msg = self.sub.get_message()
            if not msg:
                time.sleep(0.005)
                continue
            sample = json.loads(msg["data"])
            if not sample.get("model_id") == self.agent.model_id:
                logger.warning("Unexpected sample with outdated model ID")
                continue
            client_id, step_id = sample.get("client_id"), sample.get("step_id")
            sample = self.decode_sample(sample)
            logger.debug(f"Received sample with client_id: {client_id}, step_id: {step_id}")
            self.buffer.append(sample, trajectory_id=client_id, step_id=step_id)
            if self.buffer.buffer_complete:
                self.total_env_steps += self.config.ppo["n_clients"] * self.config.ppo["n_steps"]
                t_train_start = time.time()
                with self.lock:
                    self._model_update()
                t_train = time.time() - t_train_start
                logger.info(f"{time.strftime('%X')}: Model update complete ({t_train:.2f}s)")
                logger.info(f"Total env steps: {self.total_env_steps}")
                self.buffer.clear()
        logger.info("Training node has shut down")

    def push_model_update(self):
        logger.debug(f"Publishing new model with ID {self.agent.model_id}")
        self.red.hmset("model_params", self.agent.serialize(serialize_critic=False))
        self.red.publish("model_update", self.agent.model_id)
        logger.debug("Model upload successful")

    def _model_update(self):
        # Training algorithm based on Cx recommendations from https://arxiv.org/pdf/2006.05990.pdf
        b_idx = np.arange(self.buffer.n_batch_samples)
        for _ in range(self.config.ppo["train_epochs"]):
            # Compute GAE advantage (C6) in each epoch (C5)
            self.buffer.compute_advantages_and_values(self.agent, self.config.gamma,
                                                      self.config.ppo["lambda"])
            returns = self.buffer.advantages + self.buffer.values
            self.np_random.shuffle(b_idx)
            for j in range(0, self.config.batch_size, self.config.ppo["minibatch_size"]):
                mb_idx = b_idx[j:j + self.config.ppo["minibatch_size"]]
                new_prob = self.agent.get_probs(self.buffer.states[mb_idx])
                new_prob = torch.gather(new_prob, 1, self.buffer.actions[mb_idx])
                ratio = new_prob / self.buffer.probs[mb_idx]
                # Compute policy (actor) loss
                mb_advantages = self.buffer.advantages[mb_idx]
                policy_loss_1 = - mb_advantages * ratio
                policy_loss_2 = - mb_advantages * torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
                # Compute value (critic) loss
                v_estimate = self.agent.get_values(self.buffer.states[mb_idx])
                value_loss = 0.5 * ((v_estimate - returns[mb_idx])**2).mean()
                # Update agent
                self.agent.critic_opt.zero_grad()
                value_loss.backward()
                self.agent.critic_opt.step()
                self.agent.actor_opt.zero_grad()
                policy_loss.backward()
                self.agent.actor_opt.step()
        self.agent.model_id = str(uuid4())
        self.push_model_update()

    def _client_heartbeat(self, n_clients, redis_secret, shutdown_event):
        last_heartbeat = np.zeros((n_clients, ), dtype=np.float64)
        red = Redis(host='redis', port=6379, password=redis_secret, db=0, decode_responses=True)
        sub = red.pubsub(ignore_subscribe_messages=True)
        sub.subscribe("ppo_heartbeat")
        last_heartbeat[:] = time.time()
        logger.info("Client heartbeat service started")
        while not shutdown_event.is_set():
            msg = sub.get_message()  # Does not work with timeout and ignore subscribe
            if not msg:
                time.sleep(1)
            else:
                msg = json.loads(msg["data"])
                if msg["client_id"] < n_clients:
                    if time.time() - msg["timestamp"] < 5:  # Ignore stale messages in the queue
                        last_heartbeat[msg["client_id"]] = msg["timestamp"]
            if np.all(time.time() - last_heartbeat < 10):
                logger.debug(f"{time.strftime('%X')}: Heartbeat ok")
                continue
            stale_ids = list(np.where(time.time() - last_heartbeat >= 10)[0])
            logger.error(f"Missing heartbeat for clients {stale_ids}. Shutting down training")
            shutdown_event.set()

    def discover_clients(self, timeout=60):
        discovery_sub = self.red.pubsub(ignore_subscribe_messages=True)
        discovery_sub.subscribe("ppo_discovery")
        n_registered = 0
        tstart = time.time()
        while not time.time() - tstart > timeout:
            msg = discovery_sub.get_message()
            if not msg:
                time.sleep(0.5)
                continue
            self.red.publish(msg["data"], n_registered)
            n_registered += 1
            logger.info(f"Discovered client {n_registered}/{self.config.ppo['n_clients']}")
            if n_registered == self.config.ppo["n_clients"]:
                return
        raise ServerDiscoveryTimeout("Discovery phase failed to register the required clients")

    def checkpoint(self, path):
        logger.info("Checkpointing...")
        path.mkdir(exist_ok=True)
        self.agent.save(path)
        with open(path / "config.json", "w")  as f:
            json.dump(vars(self.config), f)
        logger.info("Checkpoint finished")

    def load_checkpoint(self, path):
        self.agent.load(path)
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
        self._shutdown.set()
