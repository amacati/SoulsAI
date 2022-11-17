import logging
import random
import multiprocessing as mp
import io

import torch

from soulsai.core.networks import DQN, AdvantageDQN, NoisyDQN, PPOActor, PPOCritic

logger = logging.getLogger(__name__)


def get_net_class(network_type):
    if network_type == "DQN":
        return DQN
    if network_type == "AdvantageDQN":
        return AdvantageDQN
    if network_type == "NoisyDQN":
        return NoisyDQN
    if network_type == "PPOActor":
        return PPOActor
    if network_type == "PPOCritic":
        return PPOCritic
    raise ValueError(f"Net type {network_type} not supported!")


class DQNAgent:

    def __init__(self, network_type, network_kwargs, lr, gamma, multistep, grad_clip, q_clip):
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

    def __call__(self, x, action_mask=None):
        with torch.no_grad():
            x = torch.as_tensor(x).to(self.dev)
            qvalues = self.dqn1(x)+self.dqn2(x)
            if action_mask is not None:
                c = torch.as_tensor(action_mask, dtype=torch.bool)
                qvalues = torch.where(c, qvalues, torch.tensor([-torch.inf], dtype=torch.float32))
            return torch.argmax(qvalues).item()

    def train(self, states, actions, rewards, next_states, dones, action_masks=None):
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

    def save(self, path):
        torch.save(self.dqn1, path / "actor_dqn1.pt")
        torch.save(self.dqn2, path / "actor_dqn2.pt")

    def load(self, path):
        self.dqn1 = torch.load(path / "actor_dqn1.pt").to(self.dev)
        self.dqn2 = torch.load(path / "actor_dqn2.pt").to(self.dev)

    def load_state_dict(self, state_dicts):
        self.dqn1.load_state_dict(state_dicts["dqn1"])
        self.dqn2.load_state_dict(state_dicts["dqn2"])
        self.model_id = state_dicts["model_id"]

    def state_dict(self):
        return {"dqn1": self.dqn1.state_dict(), "dqn2": self.dqn2.state_dict(),
                "model_id": self.model_id}

    def serialize(self):
        assert self.model_id is not None
        dqn1_buff = io.BytesIO()
        torch.save(self.dqn1, dqn1_buff)
        dqn1_buff.seek(0)
        dqn2_buff = io.BytesIO()
        torch.save(self.dqn2, dqn2_buff)
        dqn2_buff.seek(0)
        return {"dqn1": dqn1_buff.read(), "dqn2": dqn2_buff.read(), "model_id": self.model_id}

    def deserialize(self, serialization):
        dqn1_buff = io.BytesIO(serialization["dqn1"])
        dqn1_buff.seek(0)
        self.dqn1 = torch.load(dqn1_buff)
        dqn2_buff = io.BytesIO(serialization["dqn2"])
        dqn2_buff.seek(0)
        self.dqn2 = torch.load(dqn2_buff)
        self.model_id = serialization["model_id"].decode("utf-8")

    def update_callback(self):
        if self.network_type == "NoisyDQN":
            self.dqn1.reset_noise()
            self.dqn2.reset_noise()


class DQNClientAgent(DQNAgent):

    def __init__(self, network_type, network_kwargs):
        self.shared = False  # Shared before super init since model_id property gets called
        super().__init__(network_type, network_kwargs, 0, 0, 0, 0, 0)
        self._model_id = None

    def share_memory(self):
        self.dqn1.share_memory()
        self.dqn2.share_memory()
        self.shared = True
        self._model_id = mp.Array("B", 36)  # uuid4 string holds 36 chars
        self._model_id[:] = bytes(36*" ", encoding="utf-8")

    def train(self, *_):
        raise NotImplementedError("Client agent does not support training")

    @property
    def model_id(self) -> str:
        if self.shared:
            return bytes(self._model_id[:]).decode("utf-8")
        return self._model_id

    @model_id.setter
    def model_id(self, value):
        if self.shared:
            self._model_id[:] = bytes(value, encoding="utf-8")
        else:
            self._model_id = value


class PPOAgent:

    def __init__(self, actor_net, actor_net_kwargs, critic_net, critic_net_kwargs, actor_lr,
                 critic_lr):
        self.dev = torch.device("cpu")  # CPU is faster for small networks
        self.actor_net_type, self.critic_net_type = actor_net, critic_net
        self.actor = get_net_class(actor_net)(**actor_net_kwargs)
        self.critic = get_net_class(critic_net)(**critic_net_kwargs)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.model_id = None

    def get_action(self, x):
        with torch.no_grad():
            probs = self.actor(torch.as_tensor(x).to(self.dev))
        action = torch.multinomial(probs, 1).item()
        return action, probs[action].item()

    def get_values(self, x, requires_grad=True):
        if requires_grad:
            return self.critic(x)
        with torch.no_grad():
            return self.critic(x)

    def get_probs(self, x):
        return self.actor(x)

    def save(self, path):
        torch.save(self.actor, path / "actor_ppo.pt")
        torch.save(self.critic, path / "critic_ppo.pt")

    def load(self, path):
        self.actor = torch.load(path / "actor_ppo.pt").to(self.dev)
        self.critic = torch.load(path / "critic_ppo.pt").to(self.dev)

    def load_state_dict(self, state_dicts):
        self.actor.load_state_dict(state_dicts["actor"])
        self.critic.load_state_dict(state_dicts["critic"])
        self.model_id = state_dicts["model_id"]

    def state_dict(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "model_id": self.model_id}

    def serialize(self, serialize_critic=True):
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

    def deserialize(self, serialization, deserialize_critic=True):
        actor_buff = io.BytesIO(serialization["actor"])
        actor_buff.seek(0)
        self.actor = torch.load(actor_buff)
        self.model_id = serialization["model_id"].decode("utf-8")
        if deserialize_critic:
            critic_buff = io.BytesIO(serialization["critic"])
            critic_buff.seek(0)
            self.critic = torch.load(critic_buff)

    def update_callback(self):
        if self.actor_net_type == "NoisyNet":
            self.actor.reset_noise()
        if self.critic_net_type == "NoisyNet":
            self.critic.reset_noise()


class PPOClientAgent:

    def __init__(self, network_type, network_kwargs):
        self.dev = torch.device("cpu")  # CPU is faster for small networks
        self.actor_net_type = network_type
        self.actor = get_net_class(network_type)(**network_kwargs)
        self.model_id = None

    def get_action(self, x):
        with torch.no_grad():
            probs = self.actor(torch.as_tensor(x).to(self.dev))
        action = torch.multinomial(probs, 1).item()
        return action, probs[action].item()

    def serialize(self):
        assert self.model_id is not None
        actor_buff = io.BytesIO()
        torch.save(self.actor, actor_buff)
        actor_buff.seek(0)
        return {"actor": actor_buff.read(), "model_id": self.model_id}

    def deserialize(self, serialization):
        actor_buff = io.BytesIO(serialization["actor"])
        actor_buff.seek(0)
        self.actor = torch.load(actor_buff)
        self.model_id = serialization["model_id"].decode("utf-8")
