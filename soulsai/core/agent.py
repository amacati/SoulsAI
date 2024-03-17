"""The agent module contains implementations of DQN and PPO agents.

Since models are trained on the server, all server agents have to support the serialization of
parameters into a format suitable for uploading it to the Redis data base. To reduce bandwidth,
agents can choose which parameters to serialize and which to keep local.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F

from soulsai.core.networks import net_cls, polyak_update
from soulsai.utils import module_type_from_string

if TYPE_CHECKING:
    import numpy as np
    from tensordict import TensorDict

    from soulsai.core.transform import Mask

logger = logging.getLogger(__name__)
agent_cls = module_type_from_string(__name__)


class Agent(torch.nn.Module):
    """Base class for agents.

    All agents should inherit from this class. Agents are Modules to allow for easy serialization
    and deserialization of their parameters. The model ID is used to keep track of the current
    version of the model. The update callback can be used to reset noisy networks after an update.
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        """Initialize the agent.

        Args:
            device: Torch device for the networks.
        """
        super().__init__()
        self.device = device
        self.networks = torch.nn.ModuleDict()
        self.model_id = torch.nn.Parameter(
            torch.tensor([-1], dtype=torch.int64), requires_grad=False
        )

    def update_callback(self):
        """Update callback for networks with special requirements."""

    def client_state_dict(self) -> dict:
        """Get the state dictionary of the agent.

        By default, the state dictionary is the same as the state dictionary of the module. Based on
        the inference requirements, the state dictionary can be modified to exclude unnecessary.

        Returns:
            The state dictionary of the agent.
        """
        return self.state_dict()


class DQNAgent(Agent):
    """Deep Q learning agent class for training and sharing Q networks.

    The agent uses a dueling Q network algorithm, where two Q networks are trained at the same time.
    Networks are assigned to either estimate the future state value for the TD error, or to estimate
    the current value. The network estimating the current value then gets updated.
    """

    def __init__(
        self,
        network_type: str,
        network_kwargs: dict,
        lr: float,
        gamma: float,
        multistep: int,
        grad_clip: float,
        q_clip: float,
        device: torch.device,
    ):
        """Initialize the networks and optimizers.

        Args:
            network_type: The network type name.
            network_kwargs: Keyword arguments for the network.
            lr: Network learning rate.
            gamma: Reward discount factor.
            multistep: Number of multi-step returns considered in the TD update.
            grad_clip: Gradient clipping value for the Q networks.
            q_clip: Maximal value of the estimator network during training.
            device: Torch device for the networks.
        """
        super().__init__(device)
        self.q_clip = q_clip
        self.network_type = network_type
        self.networks.add_module("dqn1", net_cls(network_type)(**network_kwargs).to(self.device))
        self.networks.add_module("dqn2", net_cls(network_type)(**network_kwargs).to(self.device))
        self.dqn1_opt = torch.optim.Adam(self.networks["dqn1"].parameters(), lr=lr)
        self.dqn2_opt = torch.optim.Adam(self.networks["dqn2"].parameters(), lr=lr)
        self.gamma = gamma
        self.multistep = multistep
        self.grad_clip = grad_clip

    @torch.no_grad()
    def __call__(self, sample: TensorDict) -> TensorDict:
        """Calculate the current best action by averaging the values from both networks.

        Args:
            sample: Sample containing the 'obs' key.

        Returns:
            The sample with the value tensor at the '__value__' key.
        """
        x = sample["obs"].to(self.device)
        sample["__value__"] = self.networks["dqn1"](x) + self.networks["dqn2"](x)
        return sample

    def train(self, batch: TensorDict, mask: Mask | None = None) -> TensorDict:
        """Train the agent with double DQN.

        Calculates the TD error between the predictions from the trained network and the data with
        a Q(s+1, a) estimate from the estimation network and takes an optimization step for the
        train network. ``dqn1`` and ``dqn2`` are randomly assigned their role as estimation or train
        network.

        Args:
            batch: A TensorDict training batch containing observations, actions, rewards etc.
            mask: Optional mask for the value network.

        Returns:
            The batch, optionally with additional info from the training.
        """
        self.networks.train()
        batch_size = batch.batch_size[0]
        coin = random.choice([True, False])
        train_net, estimate_net = ("dqn1", "dqn2") if coin else ("dqn2", "dqn1")
        train_net, estimate_net = self.networks[train_net], self.networks[estimate_net]
        self.dqn1_opt.zero_grad()
        self.dqn2_opt.zero_grad()
        train_opt = self.dqn1_opt if coin else self.dqn2_opt
        obs = batch["obs"].float().to(self.device)
        reward = batch["reward"].float().to(self.device)
        next_obs = batch["next_obs"].float().to(self.device)
        action = batch["action"].to(self.device)
        terminated = batch["terminated"].float().to(self.device)
        q_a = train_net(obs)[range(batch_size), action]
        assert q_a.shape == (batch_size,), f"Unexpected shape {q_a.shape}"
        with torch.no_grad():
            q_next = train_net(next_obs)
            # Optionally mask the estimated Q values
            q_next = q_next if mask is None else mask.mask_tensor(q_next, batch)
            assert q_next.ndim == 2, f"Unexpected shape {q_next.shape}"
            assert q_next.shape[0] == batch_size, f"Unexpected shape {q_next.shape}"
            a_next = torch.max(q_next, 1).indices
            q_a_next = estimate_net(next_obs)[range(batch_size), a_next]
            assert q_a_next.shape == (batch_size,), f"Unexpected shape {q_a_next.shape}"
            q_a_next = torch.clamp(q_a_next, -self.q_clip, self.q_clip)
            q_td = reward + self.gamma**self.multistep * q_a_next * (1 - terminated)
            assert q_td.shape == (batch_size,), f"Unexpected shape {q_td.shape}"
        td_error = q_a - q_td
        huber_loss = F.huber_loss(td_error, torch.zeros_like(td_error), reduction="none", delta=1.0)
        assert huber_loss.shape == (batch_size,), f"Unexpected shape {huber_loss.shape}"
        if "__weight__" in batch.keys():
            assert batch["__weight__"].shape == (batch_size,)
            batch["__priority__"] = huber_loss.detach()
            huber_loss = huber_loss * batch["__priority__"].to(self.device)
        huber_loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(train_net.parameters(), self.grad_clip)
        train_opt.step()
        self.networks.eval()
        return batch

    def update_callback(self):
        """Reset noisy networks after an update."""
        if "Noisy" in self.network_type:
            self.dqn1.reset_noise()
            self.dqn2.reset_noise()


class DistributionalDQNAgent(Agent):
    """QR DQN agent."""

    def __init__(
        self,
        network_type: str,
        network_kwargs: dict,
        lr: float,
        gamma: float,
        multistep: int,
        grad_clip: float,
        q_clip: float,
        tau: float,
        device: torch.device,
    ):
        """Initialize the networks and optimizers.

        Args:
            network_type: The network type name.
            network_kwargs: Keyword arguments for the network.
            lr: Network learning rate.
            gamma: Reward discount factor.
            multistep: Number of multi-step returns considered in the TD update.
            grad_clip: Gradient clipping value for the Q networks.
            q_clip: Maximal value of the estimator network during training.
            tau: Polyak averaging factor for the target network.
            device: Torch device for the networks.
        """
        super().__init__(device)
        self.network_type = network_type
        self.networks.add_module("dqn", net_cls(network_type)(**network_kwargs).to(self.device))
        self.networks.add_module(
            "target_dqn", net_cls(network_type)(**network_kwargs).to(self.device)
        )
        self.networks["target_dqn"].load_state_dict(self.networks["dqn"].state_dict())
        self.networks["target_dqn"].requires_grad_(False)
        self.opt = torch.optim.AdamW(self.networks["dqn"].parameters(), lr)
        self.gamma = gamma
        self.tau = tau
        self.multistep = multistep
        self.grad_clip = grad_clip
        self.q_clip = q_clip
        N = self.networks["dqn"].n_quantiles
        self.quantile_tau = torch.tensor([i / N for i in range(1, N + 1)]).float().to(self.device)

    @torch.no_grad()
    def __call__(self, sample: TensorDict) -> TensorDict:
        """Calculate the current best action.

        Args:
            sample: Sample containing the 'obs' key.

        Returns:
            The sample with the value tensor at the '__value__' key.
        """
        sample["__value__"] = self.networks["dqn"](sample["obs"].to(self.device)).mean(dim=-1)
        return sample

    def train(self, batch: TensorDict, mask: Mask | None = None) -> TensorDict:
        """Train the agent with quantile regression DQN.

        Calculates the TD error between the predictions from the main network and the data with
        a Q(s+1, a) estimate from the target network and takes an optimization step for the
        train network. The action for the next state is chosen by the main network as in double DQN.

        Args:
            batch: A TensorDict training batch containing observations, actions, rewards etc.
            mask: Optional mask for the value network.

        Returns:
            The batch, optionally with additional info from the training.
        """
        self.networks.train()
        batch_size, N = batch.batch_size[0], self.networks["dqn"].n_quantiles
        self.opt.zero_grad()
        # Move data to tensors. Unsqueeze rewards and terminated in preparation for broadcasting
        obs = batch["obs"].float().to(self.device)
        reward = batch["reward"].float().unsqueeze(-1).to(self.device)
        next_obs = batch["next_obs"].float().to(self.device)
        action = batch["action"].to(self.device)
        terminated = batch["terminated"].float().unsqueeze(-1).to(self.device)
        q_a = self.networks["dqn"](obs)[range(batch_size), action, :]
        with torch.no_grad():
            q_next = self.networks["dqn"](next_obs)  # Let train net choose actions
            # Optionally mask the estimated Q values
            q_next = q_next if mask is None else mask.mask_tensor(q_next, batch, auto_expand=True)
            assert q_next.shape == (batch_size, self.networks["dqn"].output_dims, N)
            a_next = torch.argmax(q_next.mean(dim=-1), dim=-1)
            # Estimate quantiles of actions chosen by train net with target net to avoid
            # overestimation
            q_a_next = self.networks["target_dqn"](next_obs)[range(batch_size), a_next, :]
            assert q_a_next.shape == (batch_size, N), f"Unexpected shape {q_a_next.shape}"
            q_a_next = torch.clamp(q_a_next, -self.q_clip, self.q_clip)
            q_targets = reward + self.gamma**self.multistep * q_a_next * (1 - terminated)
            assert q_targets.shape == (batch_size, N)
        td_error = q_targets[:, None, :] - q_a[..., None]  # Broadcast to shape [B N N]
        assert td_error.shape == (batch_size, N, N), f"Unexpected shape {td_error.shape}"
        huber_loss = F.huber_loss(td_error, torch.zeros_like(td_error), reduction="none", delta=1.0)
        quantile_loss = abs(self.quantile_tau - (td_error.detach() < 0).float()) * huber_loss
        assert quantile_loss.shape == (batch_size, N, N), quantile_loss.shape
        summed_quantile_loss = quantile_loss.mean(dim=2).sum(1)
        if "__weight__" in batch.keys():
            assert batch["__weight__"].shape == (batch_size,)
            batch["__priority__"] = summed_quantile_loss.detach()
            summed_quantile_loss = summed_quantile_loss * batch["__weight__"].to(self.device)
        loss = summed_quantile_loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.networks["dqn"].parameters(), self.grad_clip)
        self.opt.step()
        polyak_update(self.networks["target_dqn"], self.networks["dqn"], self.tau)
        self.networks.eval()
        return batch

    def client_state_dict(self) -> dict:
        """Get the state dictionary of the agent.

        Removes the target network, which is not needed for inference on the client.

        Returns:
            The state dictionary of the agent.
        """
        return {k: v for k, v in self.state_dict().items() if "target_dqn" not in k}


class DistributionalR2D2Agent(Agent):
    """QR R2D2 agent.

    The agent uses a dueling Q network algorithm, where two Q networks are trained at the same time.
    The networks predict a distribution of quantiles for each action instead of single values. The
    networks also contain a recurrent layer to allow for a latent network state that compansates
    potential non-markovian elements in the environment.
    """

    def __init__(
        self,
        network_type: str,
        network_kwargs: dict,
        lr: float,
        gamma: float,
        multistep: int,
        grad_clip: float,
        q_clip: float,
        device: torch.device,
    ):
        """Initialize the networks and optimizers.

        Args:
            network_type: The network type name.
            network_kwargs: Keyword arguments for the network.
            lr: Network learning rate.
            gamma: Reward discount factor.
            multistep: Number of multi-step returns considered in the TD update.
            grad_clip: Gradient clipping value for the Q networks.
            q_clip: Maximal value of the estimator network during training.
            device: Torch device for the networks.
        """
        super().__init__(device)
        self.network_type = network_type
        self.networks.add_module("qr_dqn1", net_cls(network_type)(**network_kwargs).to(self.device))
        self.networks.add_module("qr_dqn2", net_cls(network_type)(**network_kwargs).to(self.device))
        self.opt1 = torch.optim.Adam(self.networks["qr_dqn1"].parameters(), lr)
        self.opt2 = torch.optim.Adam(self.networks["qr_dqn2"].parameters(), lr)
        self.gamma = gamma
        self.multistep = multistep
        self.grad_clip = grad_clip
        self.q_clip = q_clip
        N = self.networks["qr_dqn1"].n_quantiles
        self.quantile_tau = torch.tensor([i / N for i in range(1, N + 1)]).float().to(self.device)

    def train(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        terminated: np.ndarray,
        action_masks: np.ndarray | None = None,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
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
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(self.device)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32).to(self.device)
        terminated = torch.as_tensor(terminated, dtype=torch.float32).unsqueeze(-1).to(self.device)
        if action_masks is not None:
            action_masks = torch.as_tensor(action_masks, dtype=torch.bool).to(self.device)
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
        huber_loss = F.huber_loss(td_error, torch.zeros_like(td_error), reduction="none", delta=1.0)
        quantile_loss = abs(self.quantile_tau - (td_error.detach() < 0).float()) * huber_loss
        assert quantile_loss.shape == (batch_size, N, N), quantile_loss.shape
        summed_quantile_loss = quantile_loss.mean(dim=2).sum(1)
        if weights is not None:
            assert weights.shape == (batch_size,)
            weights = torch.tensor(weights).to(self.device)
            summed_quantile_loss = summed_quantile_loss * weights
        loss = summed_quantile_loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_net.parameters(), self.grad_clip)
        train_opt.step()
        return summed_quantile_loss.detach().cpu().numpy()


class PPOAgent(Agent):
    """PPO agent for server-side training.

    Uses a critic for general advantage estimation (see https://arxiv.org/pdf/2006.05990.pdf).
    """

    def __init__(
        self,
        actor_net: str,
        actor_net_kwargs: dict,
        critic_net: str,
        critic_net_kwargs: dict,
        actor_lr: float,
        critic_lr: float,
        device: torch.device,
    ):
        """Initialize the actor and critic networks.

        Args:
            actor_net: The actor network type name.
            actor_net_kwargs: Keyword arguments for the actor network.
            critic_net: The critic network type name.
            critic_net_kwargs: Keyword arguments for the critic network.
            actor_lr: Actor learning rate.
            critic_lr: Critic learning rate.
            device: Torch device for the networks.
        """
        super().__init__(device=device)
        self.actor_net_type, self.critic_net_type = actor_net, critic_net
        self.networks.add_module("actor", net_cls(actor_net)(**actor_net_kwargs).to(device))
        self.networks.add_module("critic", net_cls(critic_net)(**critic_net_kwargs).to(device))

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
            probs = self.networks["actor"](torch.as_tensor(x).to(self.device))
        action = torch.multinomial(probs, 1).squeeze(-1)
        return action, probs[range(action.shape[0]), action]

    def get_values(self, x: torch.Tensor, requires_grad: bool = True) -> torch.Tensor:
        """Get the state value for the input x.

        Args:
            x: Input tensor.
            requires_grad: Disables the computation of gradients if true.

        Returns:
            The current state-action value.
        """
        if requires_grad:
            return self.networks["critic"](x.to(self.device))
        with torch.no_grad():
            return self.networks["critic"](x.to(self.device))

    def get_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get the action probabilities for the input x.

        Args:
            x: Input tensor.

        Returns:
            The action probabilities.
        """
        return self.networks["actor"](x.to(self.device))

    def update_callback(self):
        """Update callback after a training step to reset noisy nets if used."""
        if self.actor_net_type == "NoisyNet":
            self.networks["actor"].reset_noise()
        if self.critic_net_type == "NoisyNet":
            self.networks["critic"].reset_noise()

    def client_state_dict(self) -> dict:
        """Get the state dictionary of the agent.

        By default, the state dictionary is the same as the state dictionary of the module. Based on
        the inference requirements, the state dictionary can be modified to exclude unnecessary.

        Returns:
            The state dictionary of the agent.
        """
        return {k: v for k, v in self.state_dict().items() if "critic" not in k}
