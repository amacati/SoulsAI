"""The agent module contains server and client side implementations of DQN and PPO agents.

Since models are trained on the server, all server agents have to support the serialization of
parameters into a format suitable for uploading it to the Redis data base. Client agents then have
to deserialize this data and load the new weights into their policy or value networks.
"""
import logging
import random
import multiprocessing as mp
import io
from typing import Tuple, Any
from pathlib import Path

import torch
import numpy as np

from soulsai.core.networks import get_net_class

logger = logging.getLogger(__name__)


class DQNAgent:
    """Deep Q learning agent class for training and sharing Q networks.

    The agent uses a dueling Q network algorithm, where two Q networks are trained at the same time.
    Networks are assigned to either estimate the future state value for the TD error, or to estimate
    the current value. The network estimating the current value then gets updated. In order to share
    the current weights with client agents, networks can be serialized into dictionaries of byte
    arrays.
    """

    def __init__(self, network_type: str, network_kwargs: dict, lr: float, gamma: float,
                 multistep: int, grad_clip: float, q_clip: float):
        """Initialize the networks and optimizers.

        Args:
            network_type: The network type name.
            network_kwargs: Keyword arguments for the network.
            lr: Network learning rate.
            gamma: Reward discount factor.
            multistep: Number of multi-step returns considered in the TD update.
            grad_clip: Gradient clipping value for the Q networks.
            q_clip: Maximal value of the estimator network during training.
        """
        self.dev = torch.device("cpu")  # CPU is faster for small networks
        self.q_clip = q_clip
        self.network_type = network_type
        Net = get_net_class(network_type)
        self.dqn1 = Net(**network_kwargs).to(self.dev)
        self.dqn2 = Net(**network_kwargs).to(self.dev)
        self.dqn1_opt = torch.optim.Adam(self.dqn1.parameters(), lr=lr)
        self.dqn2_opt = torch.optim.Adam(self.dqn2.parameters(), lr=lr)
        self.gamma = gamma
        self.multistep = multistep
        self.grad_clip = grad_clip
        self.model_id = None

    def __call__(self, x: np.ndarray, action_mask: np.ndarray | None = None) -> int:
        """Calculate the current best action by averaging the values from both networks.

        Args:
            x: Network input.
            action_mask: Optional mask to restrict the network to a set of permitted actions.

        Returns:
            The chosen action.
        """
        with torch.no_grad():
            x = torch.as_tensor(x).to(self.dev)
            qvalues = self.dqn1(x) + self.dqn2(x)
            if action_mask is not None:
                c = torch.as_tensor(action_mask, dtype=torch.bool)
                qvalues = torch.where(c, qvalues, torch.tensor([-torch.inf], dtype=torch.float32))
            return torch.argmax(qvalues).item()

    def train(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
              next_states: np.ndarray, dones: np.ndarray, action_masks: np.ndarray | None = None):
        """Train the agent with dueling Q networks and optional action masks.

        Calculates the TD error between the predictions from the trained network and the data with
        a Q(s+1, a) estimate from the estimation network and takes an optimization step for the
        train network. ``dqn1`` and ``dqn2`` are randomly assigned their role as estimation or train
        network.

        Args:
            states: A batch of states.
            actions: A batch of actions.
            rewards: A batch of rewards.
            next_states: A batch of next states.
            dones: A batch of episode termination flags.
            action_masks: Optional batch of mask for actions.
        """
        batch_size = states.shape[0]
        coin = random.choice([True, False])
        train_net, estimate_net = (self.dqn1, self.dqn2) if coin else (self.dqn2, self.dqn1)
        self.dqn1_opt.zero_grad()
        self.dqn2_opt.zero_grad()
        train_opt = self.dqn1_opt if coin else self.dqn2_opt
        states = torch.as_tensor(states, dtype=torch.float32).to(self.dev)
        rewards = torch.as_tensor(rewards, dtype=torch.float32).to(self.dev)
        next_states = torch.as_tensor(next_states, dtype=torch.float32).to(self.dev)
        dones = torch.as_tensor(dones, dtype=torch.float32).to(self.dev)
        if action_masks is not None:
            action_masks = torch.as_tensor(action_masks, dtype=torch.bool)
        q_a = train_net(states)[range(batch_size), actions]
        with torch.no_grad():
            q_next = train_net(next_states)
            if action_masks is not None:
                q_next = torch.where(action_masks, q_next, -torch.inf)
            a_next = torch.max(q_next, 1).indices
            q_a_next = torch.clamp(estimate_net(next_states)[range(batch_size), a_next],
                                   -self.q_clip, self.q_clip)
            q_td = rewards + self.gamma ** self.multistep * q_a_next * (1 - dones)
        loss = torch.nn.functional.mse_loss(q_a, q_td)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_net.parameters(), self.grad_clip)
        train_opt.step()

    def save(self, path: Path):
        """Save both Q networks to save files within the supplied folder.

        Args:
            path: The path of the root folder for the saves.
        """
        torch.save(self.dqn1, path / "actor_dqn1.pt")
        torch.save(self.dqn2, path / "actor_dqn2.pt")

    def load(self, path: Path):
        """Load both the actor and the critic from save files.

        Args:
            path: The path of the root folder containing the actor_dqn1.pt and actor_dqn2.pt files.
        """
        self.dqn1 = torch.load(path / "actor_dqn1.pt").to(self.dev)
        self.dqn2 = torch.load(path / "actor_dqn2.pt").to(self.dev)

    def load_state_dict(self, state_dicts: dict):
        """Load a state dict of the agent.

        Args:
            state_dicts: The dictionary of state dicts for the actor and critic networks.
        """
        self.dqn1.load_state_dict(state_dicts["dqn1"])
        self.dqn2.load_state_dict(state_dicts["dqn2"])
        self.model_id = state_dicts["model_id"]

    def state_dict(self) -> dict:
        """Create a state dict of the agent.

        Returns:
            A state dict with the state dicts of both Q networks, as well as the model ID.
        """
        return {"dqn1": self.dqn1.state_dict(), "dqn2": self.dqn2.state_dict(),
                "model_id": self.model_id}

    def serialize(self) -> dict:
        """Serialize the network parameters into a dictionary of byte arrays.

        Args:
            serialize_critic: Serialize the critic network as well if set to true.

        Returns:
            The serialized parameter dictionary.
        """
        assert self.model_id is not None
        dqn1_buff = io.BytesIO()
        torch.save(self.dqn1, dqn1_buff)
        dqn1_buff.seek(0)
        dqn2_buff = io.BytesIO()
        torch.save(self.dqn2, dqn2_buff)
        dqn2_buff.seek(0)
        return {"dqn1": dqn1_buff.read(), "dqn2": dqn2_buff.read(), "model_id": self.model_id}

    def deserialize(self, serialization: dict):
        """Deserialize the parameter dictionary and load the values into the networks.

        Args:
            serialization: The serialized parameter dictionary.
        """
        dqn1_buff = io.BytesIO(serialization["dqn1"])
        dqn1_buff.seek(0)
        self.dqn1 = torch.load(dqn1_buff)
        dqn2_buff = io.BytesIO(serialization["dqn2"])
        dqn2_buff.seek(0)
        self.dqn2 = torch.load(dqn2_buff)
        self.model_id = serialization["model_id"].decode("utf-8")

    def update_callback(self):
        """Reset noisy networks after an update."""
        if "Noisy" in self.network_type:
            self.dqn1.reset_noise()
            self.dqn2.reset_noise()


class DQNClientAgent(DQNAgent):
    """DQN agent implementation for clients.

    The client agent should only be called for inference on training nodes. In order to update the
    networks, client agents can share the neural networks' and model ID's memory. A separate update
    process can then continuously download the current network weights and load them into the
    Q-networks.
    """

    def __init__(self, network_type: str, network_kwargs: dict):
        """Initialize the Q networks.

        Args:
            network_type: The Q-network type name.
            network_kwargs: Keyword arguments for the Q-network.
        """
        self.shared = False  # Shared before super init since model_id property gets called
        super().__init__(network_type, network_kwargs, 0, 0, 0, 0, 0)
        self._model_id = None

    def share_memory(self):
        """Share the client network and model ID memory."""
        self.dqn1.share_memory()
        self.dqn2.share_memory()
        self.shared = True
        self._model_id = mp.Array("B", 36)  # uuid4 string holds 36 chars
        self._model_id[:] = bytes(36 * " ", encoding="utf-8")

    def train(self, *_: Any):
        """Superseed the inhereted training function to ensure clients never train locally.

        Raises:
            NotImplementedError: The train function is not supported on client agents.
        """
        raise NotImplementedError("Client agent does not support training")

    @property
    def model_id(self) -> str:
        """The current model ID identifies each unique iteration of the agent.

        Returns:
            The model ID.
        """
        if self.shared:
            return bytes(self._model_id[:]).decode("utf-8")
        return self._model_id

    @model_id.setter
    def model_id(self, value: str):
        if self.shared:
            self._model_id[:] = bytes(value, encoding="utf-8")
        else:
            self._model_id = value


class PPOAgent:
    """PPO agent for server-side training.

    Uses a critic for general advantage estimation (see https://arxiv.org/pdf/2006.05990.pdf).
    """

    def __init__(self, actor_net: str, actor_net_kwargs: dict, critic_net: str,
                 critic_net_kwargs: dict, actor_lr: float, critic_lr: float):
        """Initialize the actor and critic networks.

        Args:
            actor_net: The actor network type name.
            actor_net_kwargs: Keyword arguments for the actor network.
            critic_net: The critic network type name.
            critic_net_kwargs: Keyword arguments for the critic network.
            actor_lr: Actor learning rate.
            critic_lr: Critic learning rate.
        """
        self.dev = torch.device("cpu")  # CPU is faster for small networks
        self.actor_net_type, self.critic_net_type = actor_net, critic_net
        self.actor = get_net_class(actor_net)(**actor_net_kwargs)
        self.critic = get_net_class(critic_net)(**critic_net_kwargs)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.model_id = None

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
            probs = self.actor(torch.as_tensor(x).to(self.dev))
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
            return self.critic(x)
        with torch.no_grad():
            return self.critic(x)

    def get_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get the action probabilities for the input x.

        Args:
            x: Input tensor.

        Returns:
            The action probabilities.
        """
        return self.actor(x)

    def save(self, path: Path):
        """Save both the actor and the critic to save files within the supplied folder.

        Args:
            path: The path of the root folder for the saves.
        """
        torch.save(self.actor, path / "actor_ppo.pt")
        torch.save(self.critic, path / "critic_ppo.pt")

    def load(self, path: Path):
        """Load both the actor and the critic from save files.

        Args:
            path: The path of the root folder containing the actor_ppo.pt and critic_ppo.pt files.
        """
        self.actor = torch.load(path / "actor_ppo.pt").to(self.dev)
        self.critic = torch.load(path / "critic_ppo.pt").to(self.dev)

    def load_state_dict(self, state_dicts: dict):
        """Load a state dict of the agent.

        Args:
            state_dicts: The dictionary of state dicts for the actor and critic networks.
        """
        self.actor.load_state_dict(state_dicts["actor"])
        self.critic.load_state_dict(state_dicts["critic"])
        self.model_id = state_dicts["model_id"]

    def state_dict(self) -> dict:
        """Create a state dict of the agent.

        Returns:
            A state dict with the state dicts of the actor and the critic, as well as the model ID.
        """
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "model_id": self.model_id}

    def serialize(self, serialize_critic: bool = False) -> dict:
        """Serialize the network parameters into a dictionary of byte arrays.

        Args:
            serialize_critic: Serialize the critic network as well if set to true.

        Returns:
            The serialized parameter dictionary.
        """
        assert self.model_id is not None
        actor_buff = io.BytesIO()
        torch.save(self.actor, actor_buff)
        actor_buff.seek(0)
        serialization = {"actor": actor_buff.read(), "model_id": self.model_id}
        if serialize_critic:
            critic_buff = io.BytesIO()
            torch.save(self.critic, critic_buff)
            critic_buff.seek(0)
            serialization["critic"] = critic_buff.read()
        return serialization

    def deserialize(self, serialization: dict, deserialize_critic: bool = False):
        """Deserialize the parameter dictionary and load the values into the networks.

        Args:
            serialization: The serialized parameter dictionary.
            deserialize_critic: Deserialize the critic network as well if set to true.
        """
        actor_buff = io.BytesIO(serialization["actor"])
        actor_buff.seek(0)
        self.actor = torch.load(actor_buff)
        self.model_id = serialization["model_id"].decode("utf-8")
        if deserialize_critic:
            critic_buff = io.BytesIO(serialization["critic"])
            critic_buff.seek(0)
            self.critic = torch.load(critic_buff)

    def update_callback(self):
        """Update callback after a training step to reset noisy nets if used."""
        if self.actor_net_type == "NoisyNet":
            self.actor.reset_noise()
        if self.critic_net_type == "NoisyNet":
            self.critic.reset_noise()


class PPOClientAgent(PPOAgent):
    """PPO client agent for inference on worker nodes."""

    def __init__(self, network_type: str, network_kwargs: dict):
        """Initialize the policy network.

        Args:
            network_type: The policy network type name.
            network_kwargs: Keyword arguments for the policy network.
        """
        self.dev = torch.device("cpu")  # CPU is faster for small networks
        self.actor_net_type = network_type
        self.actor = get_net_class(network_type)(**network_kwargs)
        self.model_id = None
