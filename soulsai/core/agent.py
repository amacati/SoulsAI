import logging
import random
import multiprocessing as mp
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


def get_net_class(network_type):
    if network_type == "DQN":
        return DQN
    if network_type == "AdvantageDQN":
        return AdvantageDQN
    if network_type == "NoisyDQN":
        return NoisyDQN
    raise ValueError(f"Net type {network_type} not supported!")


class DQNAgent:

    def __init__(self, network_type, size_s, size_a, lr, gamma, multistep, grad_clip, q_clip,
                 size_n=128):
        self.dev = torch.device("cpu")  # CPU is faster for small networks
        self.q_clip = q_clip
        self.network_type = network_type
        Net = get_net_class(network_type)
        self.dqn1 = Net(size_s, size_a, size_n).to(self.dev)
        self.dqn2 = Net(size_s, size_a, size_n).to(self.dev)
        self.dqn1_opt = torch.optim.Adam(self.dqn1.parameters(), lr=lr)
        self.dqn2_opt = torch.optim.Adam(self.dqn2.parameters(), lr=lr)
        self.gamma = gamma
        self.multistep = multistep
        self.grad_clip = grad_clip
        self.model_id = None

    def __call__(self, x):
        with torch.no_grad():
            x = torch.as_tensor(x).to(self.dev)
            return torch.argmax(self.dqn1(x)+self.dqn2(x)).item()

    def train(self, states, actions, rewards, next_states, dones):
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
        q_a = train_net(states)[range(batch_size), actions]
        with torch.no_grad():
            a_next = torch.max(train_net(next_states), 1).indices
            q_a_next = torch.clamp(estimate_net(next_states)[range(batch_size), a_next],
                                   -self.q_clip, self.q_clip)
            q_td = rewards + self.gamma ** self.multistep * q_a_next * (1 - dones)
        loss = F.mse_loss(q_a, q_td)
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

    def update_callback(self):
        if self.network_type == "NoisyDQN":
            self.dqn1.reset_noise()
            self.dqn2.reset_noise()


class ClientAgent:

    def __init__(self, network_type, size_s, size_a, size_n=128):
        self.dev = torch.device("cpu")  # CPU is faster for small networks
        Net = get_net_class(network_type)
        self.dqn1 = Net(size_s, size_a, size_n).to(self.dev)
        self.dqn2 = Net(size_s, size_a, size_n).to(self.dev)
        self._model_id = None
        self.shared = False

    def __call__(self, x):
        with torch.no_grad():
            x = torch.as_tensor(x).to(self.dev)
            return torch.argmax(self.dqn1(x)+self.dqn2(x)).item()

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

    def load_state_dict(self, state_dicts):
        self.dqn1.load_state_dict(state_dicts["dqn1"])
        self.dqn2.load_state_dict(state_dicts["dqn2"])
        self.model_id = state_dicts["model_id"]

    def state_dict(self):
        return {"dqn1": self.dqn1.state_dict(), "dqn2": self.dqn2.state_dict(),
                "model_id": self.model_id}

    def share_memory(self):
        self.dqn1.share_memory()
        self.dqn2.share_memory()
        self.shared = True
        self._model_id = mp.Array("B", 36)  # uuid4 string holds 36 chars
        self._model_id[:] = bytes(36*" ", encoding="utf-8")

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


class DQN(nn.Module):

    def __init__(self, size_s, size_a, size_n=128):
        super().__init__()
        self.linear1 = nn.Linear(size_s, size_n)
        self.linear2 = nn.Linear(size_n, size_n)
        self.linear3 = nn.Linear(size_n, size_n)
        self.output = nn.Linear(size_n, size_a)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        return self.output(x)


class AdvantageDQN(nn.Module):

    def __init__(self, size_s, size_a, size_n=128):
        super().__init__()
        self.linear1 = nn.Linear(size_s, size_n)
        self.linear2 = nn.Linear(size_n, size_n)
        self.linear3 = nn.Linear(size_n, size_n)
        self.baseline = nn.Linear(size_n, 1)
        self.advantage = nn.Linear(size_n, size_a)
        for layer in (self.linear1, self.linear2, self.linear3):
            torch.nn.init.orthogonal_(layer.weight, gain=np.sqrt(2.))
            torch.nn.init.constant_(layer.bias, val=0.)
        for layer in (self.baseline, self.advantage):
            torch.nn.init.orthogonal_(layer.weight, gain=1.)
            torch.nn.init.constant_(layer.bias, val=0.)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        v_s = self.baseline(x)
        a_s = self.advantage(x)
        return a_s + v_s - torch.mean(a_s, dim=-1, keepdim=True)


class NoisyDQN(nn.Module):

    def __init__(self, size_s, size_a, size_n=128):
        super().__init__()
        self.linear1 = nn.Linear(size_s, size_n)
        self.noisy1 = NoisyLinear(size_n, size_n)
        self.noisy2 = NoisyLinear(size_n, size_a)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.noisy1(x))
        return self.noisy2(x)

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()


class NoisyLinear(nn.Module):

    def __init__(self, size_in: int, size_out: int, std_init: float = 0.5):
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(size_out, size_in))
        self.weight_sigma = nn.Parameter(torch.Tensor(size_out, size_in))
        self.register_buffer("weight_epsilon", torch.Tensor(size_out, size_in))

        self.bias_mu = nn.Parameter(torch.Tensor(size_out))
        self.bias_sigma = nn.Parameter(torch.Tensor(size_out))
        self.register_buffer("bias_epsilon", torch.Tensor(size_out))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.size_in)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        # torch.nn.init.orthogonal_(self.weight_mu)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.size_in))
        # torch.nn.init.constant_(self.weight_sigma, self.std_init / np.sqrt(self.size_in))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        # torch.nn.init.uniform_(self.bias_mu, -mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.size_out))
        # torch.nn.init.constant_(self.bias_sigma, self.std_init / np.sqrt(self.size_out))

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.size_in)
        epsilon_out = self.scale_noise(self.size_out)
        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return F.linear(x, weight, bias)

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
