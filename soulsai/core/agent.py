"""The agent module contains server and client side implementations of DQN and PPO agents.

Since models are trained on the server, all server agents have to support the serialization of
parameters into a format suitable for uploading it to the Redis data base. Client agents then have
to deserialize this data and load the new weights into their policy or value networks.
"""
from __future__ import annotations

import io
import random
import logging
from typing import Tuple, TYPE_CHECKING

import torch
import torch.nn.functional as F
import numpy as np

from soulsai.core.networks import net_cls

if TYPE_CHECKING:
    from tensordict import TensorDict

logger = logging.getLogger(__name__)


class Agent(torch.nn.Module):
    """Base class for agents.

    All agents should inherit from this class. Agents are Modules to allow for easy serialization
    and deserialization of their parameters. The model ID is used to keep track of the current
    version of the model. The update callback can be used to reset noisy networks after an update.
    """

    def __init__(self, dev: torch.device = torch.device("cpu")):
        """Initialize the agent.

        Args:
            dev: Torch device for the networks.
        """
        super().__init__()
        self.dev = dev
        self.networks = torch.nn.ModuleDict()
        self.model_id = torch.nn.Parameter(torch.tensor([-1], dtype=torch.int64),
                                           requires_grad=False)

    def update_callback(self):
        """Update callback for networks with special requirements."""


class DQNAgent(Agent):
    """Deep Q learning agent class for training and sharing Q networks.

    The agent uses a dueling Q network algorithm, where two Q networks are trained at the same time.
    Networks are assigned to either estimate the future state value for the TD error, or to estimate
    the current value. The network estimating the current value then gets updated. In order to share
    the current weights with client agents, networks can be serialized into dictionaries of byte
    arrays.
    """

    def __init__(self, network_type: str, network_kwargs: dict, lr: float, gamma: float,
                 multistep: int, grad_clip: float, q_clip: float, dev: torch.device):
        """Initialize the networks and optimizers.

        Args:
            network_type: The network type name.
            network_kwargs: Keyword arguments for the network.
            lr: Network learning rate.
            gamma: Reward discount factor.
            multistep: Number of multi-step returns considered in the TD update.
            grad_clip: Gradient clipping value for the Q networks.
            q_clip: Maximal value of the estimator network during training.
            dev: Torch device for the networks.
        """
        super().__init__(dev)
        self.q_clip = q_clip
        self.network_type = network_type
        self.networks.add_module("dqn1", net_cls(network_type)(**network_kwargs).to(self.dev))
        self.networks.add_module("dqn2", net_cls(network_type)(**network_kwargs).to(self.dev))
        self.dqn1_opt = torch.optim.Adam(self.networks["dqn1"].parameters(), lr=lr)
        self.dqn2_opt = torch.optim.Adam(self.networks["dqn2"].parameters(), lr=lr)
        self.gamma = gamma
        self.multistep = multistep
        self.grad_clip = grad_clip

    def train(self, sample: TensorDict) -> np.ndarray:
        """Train the agent with dueling Q networks and optional action masks.

        Calculates the TD error between the predictions from the trained network and the data with
        a Q(s+1, a) estimate from the estimation network and takes an optimization step for the
        train network. ``dqn1`` and ``dqn2`` are randomly assigned their role as estimation or train
        network.

        Args:
            sample: A training sample as TensorDict containing observations, actions, rewards etc.
                as keys.

        Returns:
            The TD error for each sample in the batch.
        """
        self.networks.train()
        batch_size = sample.batch_size[0]
        coin = random.choice([True, False])
        train_net, estimate_net = ("dqn1", "dqn2") if coin else ("dqn2", "dqn1")
        train_net, estimate_net = self.networks[train_net], self.networks[estimate_net]
        self.dqn1_opt.zero_grad()
        self.dqn2_opt.zero_grad()
        train_opt = self.dqn1_opt if coin else self.dqn2_opt
        obs = torch.as_tensor(sample["obs"], dtype=torch.float32).to(self.dev)
        rewards = torch.as_tensor(sample["reward"], dtype=torch.float32).to(self.dev)
        next_obs = torch.as_tensor(sample["next_obs"], dtype=torch.float32).to(self.dev)
        actions = torch.as_tensor(sample["action"])
        terminated = torch.as_tensor(sample["terminated"], dtype=torch.float32).to(self.dev)
        action_masks = None
        if "action_mask" in sample.keys():
            action_masks = torch.as_tensor(sample["action_mask"], dtype=torch.bool).to(self.dev)
        q_a = train_net(obs)[range(batch_size), actions]
        with torch.no_grad():
            q_next = train_net(next_obs)
            if action_masks is not None:
                q_next = torch.where(action_masks, q_next, -torch.inf)
            a_next = torch.max(q_next, 1).indices
            q_a_next = estimate_net(next_obs)[range(batch_size), a_next]
            q_a_next = torch.clamp(q_a_next, -self.q_clip, self.q_clip)
            q_td = rewards + self.gamma**self.multistep * q_a_next * (1 - terminated)
        sample_loss = (q_a - q_td)**2
        if "weights" in sample.keys():
            assert sample["weights"].shape == (batch_size,)
            sample_loss = sample_loss * torch.tensor(sample["weights"]).to(self.dev)
        loss = sample_loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_net.parameters(), self.grad_clip)
        train_opt.step()
        self.networks.eval()
        return sample_loss.detach().cpu().numpy()

    def update_callback(self):
        """Reset noisy networks after an update."""
        if "Noisy" in self.network_type:
            self.dqn1.reset_noise()
            self.dqn2.reset_noise()


class DistributionalDQNAgent(Agent):
    """QR DQN agent."""

    def __init__(self, network_type: str, network_kwargs: dict, lr: float, gamma: float,
                 multistep: int, grad_clip: float, q_clip: float, dev: torch.device):
        """Initialize the networks and optimizers.

        Args:
            network_type: The network type name.
            network_kwargs: Keyword arguments for the network.
            lr: Network learning rate.
            gamma: Reward discount factor.
            multistep: Number of multi-step returns considered in the TD update.
            grad_clip: Gradient clipping value for the Q networks.
            q_clip: Maximal value of the estimator network during training.
            dev: Torch device for the networks.
        """
        super().__init__(dev)
        self.network_type = network_type
        self.networks.add_module("qr_dqn1", net_cls(network_type)(**network_kwargs).to(self.dev))
        self.networks.add_module("qr_dqn2", net_cls(network_type)(**network_kwargs).to(self.dev))
        self.opt1 = torch.optim.Adam(self.networks["qr_dqn1"].parameters(), lr)
        self.opt2 = torch.optim.Adam(self.networks["qr_dqn2"].parameters(), lr)
        self.gamma = gamma
        self.multistep = multistep
        self.grad_clip = grad_clip
        self.q_clip = q_clip
        N = self.networks["qr_dqn1"].n_quantiles
        self.quantile_tau = torch.tensor([i / N for i in range(1, N + 1)]).float().to(self.dev)

    def train(self, sample: TensorDict) -> np.ndarray:
        """Train the agent with dual quantile regression DQN.

        Calculates the TD error between the predictions from the trained network and the data with
        a Q(s+1, a) estimate from the estimation network and takes an optimization step for the
        train network. ``dqn1`` and ``dqn2`` are randomly assigned their role as estimation or train
        network.

        Args:
            sample: A training sample as TensorDict containing observations, actions, rewards etc.
                as keys.

        Returns:
            The TD error for each sample in the batch.
        """
        self.networks.train()
        batch_size, N = sample.batch_size[0], self.networks["qr_dqn1"].n_quantiles
        coin = random.choice([True, False])
        train_net, estimate_net = ("qr_dqn1", "qr_dqn2") if coin else ("qr_dqn2", "qr_dqn1")
        train_net, estimate_net = self.networks[train_net], self.networks[estimate_net]
        self.opt1.zero_grad()
        self.opt2.zero_grad()
        train_opt = self.opt1 if coin else self.opt2
        # Move data to tensors. Unsqueeze rewards and terminated in preparation for broadcasting
        obs = torch.as_tensor(sample["obs"], dtype=torch.float32).to(self.dev)
        rewards = torch.as_tensor(sample["reward"], dtype=torch.float32).unsqueeze(-1).to(self.dev)
        next_obs = torch.as_tensor(sample["next_obs"], dtype=torch.float32).to(self.dev)
        actions = torch.as_tensor(sample["action"])
        terminated = torch.as_tensor(sample["terminated"],
                                     dtype=torch.float32).unsqueeze(-1).to(self.dev)
        action_masks = None
        if "action_mask" in sample.keys():
            action_masks = torch.as_tensor(sample["action_mask"], dtype=torch.bool).to(self.dev)
        q_a = train_net(obs)[range(batch_size), actions, :]
        with torch.no_grad():
            q_next = train_net(next_obs)  # Let train net choose actions
            if action_masks is not None:
                assert action_masks.shape == (batch_size, self.networks["qr_dqn1"].output_dims)
                action_masks = action_masks.unsqueeze(1).expand(q_next.shape)
                q_next = torch.where(action_masks, q_next, -torch.inf)
            a_next = torch.argmax(q_next.mean(dim=-1), dim=-1)
            # Estimate quantiles of actions chosen by train net with estimate net to avoid
            # overestimation
            q_a_next = estimate_net(next_obs)[range(batch_size), a_next, :]
            assert q_a_next.shape == (batch_size, N), f"Unexpected shape {q_a_next.shape}"
            q_a_next = torch.clamp(q_a_next, -self.q_clip, self.q_clip)
            q_targets = rewards + self.gamma**self.multistep * q_a_next * (1 - terminated)
            assert q_targets.shape == (batch_size, N)
        td_error = q_targets[:, None, :] - q_a[..., None]  # Broadcast to shape [B N N]
        assert td_error.shape == (batch_size, N, N), f"Unexpected shape {td_error.shape}"
        huber_loss = F.huber_loss(td_error, torch.zeros_like(td_error), reduction="none", delta=1.)
        quantile_loss = abs(self.quantile_tau - (td_error.detach() < 0).float()) * huber_loss
        assert quantile_loss.shape == (batch_size, N, N), quantile_loss.shape
        summed_quantile_loss = quantile_loss.mean(dim=2).sum(1)
        if "weights" in sample.keys():
            assert sample["weights"].shape == (batch_size,)
            weights = torch.tensor(sample["weights"]).to(self.dev)
            summed_quantile_loss = summed_quantile_loss * weights
        loss = summed_quantile_loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_net.parameters(), self.grad_clip)
        train_opt.step()
        self.networks.eval()
        return summed_quantile_loss.detach().cpu().numpy()


class DistributionalR2D2Agent(Agent):
    """QR R2D2 agent.

    The agent uses a dueling Q network algorithm, where two Q networks are trained at the same time.
    The networks predict a distribution of quantiles for each action instead of single values. The
    networks also contain a recurrent layer to allow for a latent network state that compansates
    potential non-markovian elements in the environment.
    """

    def __init__(self, network_type: str, network_kwargs: dict, lr: float, gamma: float,
                 multistep: int, grad_clip: float, q_clip: float, dev: torch.device):
        """Initialize the networks and optimizers.

        Args:
            network_type: The network type name.
            network_kwargs: Keyword arguments for the network.
            lr: Network learning rate.
            gamma: Reward discount factor.
            multistep: Number of multi-step returns considered in the TD update.
            grad_clip: Gradient clipping value for the Q networks.
            q_clip: Maximal value of the estimator network during training.
            dev: Torch device for the networks.
        """
        super().__init__(dev)
        self.network_type = network_type
        self.networks.add_module("qr_dqn1", net_cls(network_type)(**network_kwargs).to(self.dev))
        self.networks.add_module("qr_dqn2", net_cls(network_type)(**network_kwargs).to(self.dev))
        self.opt1 = torch.optim.Adam(self.networks["qr_dqn1"].parameters(), lr)
        self.opt2 = torch.optim.Adam(self.networks["qr_dqn2"].parameters(), lr)
        self.gamma = gamma
        self.multistep = multistep
        self.grad_clip = grad_clip
        self.q_clip = q_clip
        N = self.networks["qr_dqn1"].n_quantiles
        self.quantile_tau = torch.tensor([i / N for i in range(1, N + 1)]).float().to(self.dev)

    def train(self,
              obs: np.ndarray,
              actions: np.ndarray,
              rewards: np.ndarray,
              next_obs: np.ndarray,
              terminated: np.ndarray,
              action_masks: np.ndarray | None = None,
              weights: np.ndarray | None = None) -> np.ndarray:
        """Train the agent with quantile regression DQN and a target network.

        Args:
            obs: Batch of observations.
            actions: A batch of actions.
            rewards: A batch of rewards.
            next_obs: A batch of next observations.
            terminated: A batch of episode termination flags.
            action_masks: Optional batch of mask for actions.
            weights: Optional batch of weights for prioritized experience replay.

        Returns:
            The TD error for each sample in the batch.
        """
        batch_size, N = obs.shape[0], self.networks["qr_dqn1"].n_quantiles
        coin = random.choice([True, False])
        train_net, estimate_net = ("qr_dqn1", "qr_dqn2") if coin else ("qr_dqn2", "qr_dqn1")
        train_net, estimate_net = self.networks[train_net], self.networks[estimate_net]
        self.opt1.zero_grad()
        self.opt2.zero_grad()
        train_opt = self.opt1 if coin else self.opt2
        # Move data to tensors. Unsqueeze rewards and terminated in preparation for broadcasting
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.dev)
        rewards = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(self.dev)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32).to(self.dev)
        terminated = torch.as_tensor(terminated, dtype=torch.float32).unsqueeze(-1).to(self.dev)
        if action_masks is not None:
            action_masks = torch.as_tensor(action_masks, dtype=torch.bool).to(self.dev)
        q_a = train_net(obs)[range(batch_size), :, actions]
        with torch.no_grad():
            q_next = train_net(next_obs)  # Let train net choose actions
            if action_masks is not None:
                assert action_masks.shape == (batch_size, self.networks["qr_dqn1"].output_dims)
                action_masks = action_masks.unsqueeze(1).expand(q_next.shape)
                q_next = torch.where(action_masks, q_next, -torch.inf)
            a_next = torch.argmax(q_next.mean(dim=1), dim=1)
            # Estimate quantiles of actions chosen by train net with estimate net to avoid
            # overestimation
            q_a_next = estimate_net(next_obs)[range(batch_size), :, a_next]
            assert q_a_next.shape == (batch_size, N)
            q_a_next = torch.clamp(q_a_next, -self.q_clip, self.q_clip)
            q_targets = rewards + self.gamma**self.multistep * q_a_next * (1 - terminated)
        td_error = q_targets[:, None, :] - q_a[..., None]  # Broadcast to shape [B N N]
        assert td_error.shape == (batch_size, N, N)
        huber_loss = F.huber_loss(td_error, torch.zeros_like(td_error), reduction="none", delta=1.)
        quantile_loss = abs(self.quantile_tau - (td_error.detach() < 0).float()) * huber_loss
        assert quantile_loss.shape == (batch_size, N, N), quantile_loss.shape
        summed_quantile_loss = quantile_loss.mean(dim=2).sum(1)
        if weights is not None:
            assert weights.shape == (batch_size,)
            weights = torch.tensor(weights).to(self.dev)
            summed_quantile_loss = summed_quantile_loss * weights
        loss = summed_quantile_loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_net.parameters(), self.grad_clip)
        train_opt.step()
        return summed_quantile_loss.detach().cpu().numpy()


class DQNClientAgent(Agent):
    """DQN agent implementation for clients.

    The client agent should only be called for inference on training nodes. In order to update the
    networks, client agents can share the neural networks' and model ID's memory. A separate update
    process can then continuously download the current network weights and load them into the
    Q-networks.
    """

    def __init__(self, network_type: str, network_kwargs: dict, dev: torch.device):
        """Initialize the client agent.

        Args:
            network_type: The network type name.
            network_kwargs: Keyword arguments for the network.
            dev: Torch device for the networks.
        """
        super().__init__(dev)
        self.network_type = network_type
        self.networks.add_module("dqn1", net_cls(network_type)(**network_kwargs).to(self.dev))
        self.networks.add_module("dqn2", net_cls(network_type)(**network_kwargs).to(self.dev))
        self.networks.eval()

    def __call__(self, x: np.ndarray, action_mask: np.ndarray | None = None) -> torch.IntTensor:
        """Calculate the current best action by averaging the values from both networks.

        Args:
            x: Network input.
            action_mask: Optional mask to restrict the network to a set of permitted actions.

        Returns:
            The chosen action.
        """
        with torch.no_grad():
            x = torch.as_tensor(x).to(self.dev)
            qvalues = self.networks["dqn1"](x) + self.networks["dqn2"](x)
            if action_mask is not None:
                c = torch.as_tensor(action_mask, dtype=torch.bool, device=self.dev)
                qvalues = torch.where(c, qvalues, -torch.inf)
            return torch.argmax(qvalues, dim=-1, keepdim=True)


class DistributionalDQNClientAgent(Agent):
    """QR DQN client agent for inference on worker nodes.

    The client agent is missing the optimizer and training methods, but can be used for inference
    on worker nodes.
    """

    def __init__(self, network_type: str, network_kwargs: dict, dev: torch.device):
        """Initialize the client agent.

        Args:
            network_type: The network type name.
            network_kwargs: Keyword arguments for the network.
            dev: Torch device for the networks.
        """
        super().__init__(dev)
        self.network_type = network_type
        self.networks.add_module("qr_dqn1", net_cls(network_type)(**network_kwargs).to(self.dev))
        self.networks.add_module("qr_dqn2", net_cls(network_type)(**network_kwargs).to(self.dev))
        self.networks.eval()

    def __call__(self, x: np.ndarray, action_mask: np.ndarray | None = None) -> torch.IntTensor:
        """Calculate the current best action.

        Args:
            x: Network input.
            action_mask: Optional mask to restrict the network to a set of permitted actions.

        Returns:
            The chosen action.
        """
        with torch.no_grad():
            x = torch.as_tensor(x).to(self.dev)
            qvalues = (self.networks["qr_dqn1"](x) + self.networks["qr_dqn2"](x)).mean(dim=-1)
            if action_mask is not None:
                c = torch.as_tensor(action_mask, dtype=torch.bool, device=self.dev)
                qvalues = torch.where(c, qvalues, -torch.inf)
            return torch.argmax(qvalues, dim=-1, keepdim=True)


class PPOAgent(Agent):
    """PPO agent for server-side training.

    Uses a critic for general advantage estimation (see https://arxiv.org/pdf/2006.05990.pdf).
    """

    def __init__(self, actor_net: str, actor_net_kwargs: dict, critic_net: str,
                 critic_net_kwargs: dict, actor_lr: float, critic_lr: float, dev: torch.device):
        """Initialize the actor and critic networks.

        Args:
            actor_net: The actor network type name.
            actor_net_kwargs: Keyword arguments for the actor network.
            critic_net: The critic network type name.
            critic_net_kwargs: Keyword arguments for the critic network.
            actor_lr: Actor learning rate.
            critic_lr: Critic learning rate.
            dev: Torch device for the networks.
        """
        super().__init__(dev=dev)
        self.actor_net_type, self.critic_net_type = actor_net, critic_net
        self.networks.add_module("actor", net_cls(actor_net)(**actor_net_kwargs).to(dev))
        self.networks.add_module("critic", net_cls(critic_net)(**critic_net_kwargs).to(dev))

        self.actor_opt = torch.optim.Adam(self.networks["actor"].parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.networks["critic"].parameters(), lr=critic_lr)

    def get_action(self, x: np.ndarray) -> Tuple[int, float]:
        """Get the action and the action probability.

        Note:
            The probability is given as an actual probability, not a logit.

        Args:
            x: The network input.

        Returns:
            A tuple of the chosen action and its associated probability.
        """
        with torch.no_grad():
            probs = self.networks["actor"](torch.as_tensor(x).to(self.dev))
        action = torch.multinomial(probs, 1).item()
        return action, probs[action].item()

    def get_values(self, x: torch.Tensor, requires_grad: bool = True) -> torch.Tensor:
        """Get the state value for the input x.

        Args:
            x: Input tensor.
            requires_grad: Disables the computation of gradients if true.

        Returns:
            The current state-action value.
        """
        if requires_grad:
            return self.networks["critic"](x.to(self.dev))
        with torch.no_grad():
            return self.networks["critic"](x.to(self.dev))

    def get_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get the action probabilities for the input x.

        Args:
            x: Input tensor.

        Returns:
            The action probabilities.
        """
        return self.networks["actor"](x.to(self.dev))

    def serialize(self, serialize_critic: bool = False) -> dict:
        """Serialize the network parameters into a dictionary of byte arrays.

        Args:
            serialize_critic: Serialize the critic network as well if set to true.

        Returns:
            The serialized parameter dictionary.
        """
        assert self.model_id is not None
        if serialize_critic:
            return super().serialize()
        actor_buff = io.BytesIO()
        torch.save(self.networks["actor"], actor_buff)
        actor_buff.seek(0)
        return {"actor": actor_buff.read(), "model_id": self.model_id}

    def deserialize(self, serialization: dict, deserialize_critic: bool = False):
        """Deserialize the parameter dictionary and load the values into the networks.

        Args:
            serialization: The serialized parameter dictionary.
            deserialize_critic: Deserialize the critic network as well if set to true.
        """
        if deserialize_critic:
            super().deserialize(serialization)
        actor_buff = io.BytesIO(serialization["actor"])
        actor_buff.seek(0)
        self.networks["actor"] = torch.load(actor_buff)
        self.model_id = serialization["model_id"]

    def update_callback(self):
        """Update callback after a training step to reset noisy nets if used."""
        if self.actor_net_type == "NoisyNet":
            self.networks["actor"].reset_noise()
        if self.critic_net_type == "NoisyNet":
            self.networks["critic"].reset_noise()


class PPOClientAgent(PPOAgent, Agent):
    """PPO client agent for inference on worker nodes."""

    def __init__(self, network_type: str, network_kwargs: dict, dev: torch.device):
        """Initialize the policy network.

        Args:
            network_type: The policy network type name.
            network_kwargs: Keyword arguments for the policy network.
            dev: Torch device for the networks.
        """
        # Skip PPOAgent __init__, we don't want the critic on clients
        super(PPOAgent, self).__init__(dev)
        self.actor_net_type = network_type
        self.networks.add_module("actor", net_cls(network_type)(**network_kwargs).to(dev))
        self.model_id = None
