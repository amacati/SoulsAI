import random
import json
import torch
import torch.nn as nn
import io


class DQNAgent:

    def __init__(self, size_s, size_a, lr, gamma, grad_clip, q_clip):
        self.dev = torch.device("cpu")  # CPU is faster for small networks
        self.q_clip = q_clip
        self.dqn1 = AdvantageDQN(size_s, size_a).to(self.dev)
        self.dqn2 = AdvantageDQN(size_s, size_a).to(self.dev)
        self.dqn1_opt = torch.optim.Adam(self.dqn1.parameters(), lr=lr)
        self.dqn2_opt = torch.optim.Adam(self.dqn2.parameters(), lr=lr)
        self.gamma = gamma
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
            q_td = rewards + self.gamma * q_a_next * (1 - dones)
        loss = torch.nn.functional.mse_loss(q_a, q_td)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_net.parameters(), self.grad_clip)
        train_opt.step()

    def save(self, path):
        torch.save(self.dqn1, path / "dqn1.pt")
        torch.save(self.dqn2, path / "dqn2.pt")

    def load(self, path):
        self.dqn1 = torch.load(path / "dqn1.pt").to(self.dev)
        self.dqn2 = torch.load(path / "dqn2.pt").to(self.dev)

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


class ClientAgent:

    def __init__(self, size_s, size_a):
        self.dev = torch.device("cpu")  # CPU is faster for small networks
        self.dqn1 = AdvantageDQN(size_s, size_a).to(self.dev)
        self.dqn2 = AdvantageDQN(size_s, size_a).to(self.dev)

    def __call__(self, x):
        with torch.no_grad():
            x = torch.as_tensor(x).to(self.dev)
            return torch.argmax(self.dqn1(x)+self.dqn2(x)).item()

    def serialize(self):
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


class DQN(nn.Module):

    def __init__(self, size_s, size_a):
        super().__init__()
        self.linear1 = nn.Linear(size_s, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 128)
        self.output = nn.Linear(128, size_a)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        return self.output(x)


class AdvantageDQN(nn.Module):

    def __init__(self, size_s, size_a):
        super().__init__()
        self.linear1 = nn.Linear(size_s, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 128)
        self.baseline = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, size_a)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        v_s = self.baseline(x)
        a_s = self.advantage(x)
        return a_s + v_s - torch.mean(a_s, dim=-1, keepdim=True)
