import torch.nn as nn
import torch
import numpy as np


def clip(x, epsilon):
    return torch.maximum(torch.ones_like(x)-epsilon, torch.minimum(x, torch.ones_like(x) + epsilon))


class PPOAgent:

    def __init__(self, size_s, size_a, train_epochs, gamma, _lambda, epsilon, lr_actor, lr_critic,
                 entropy_coef=0.):
        self.actor = Actor(size_s, size_a)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic = Critic(size_s)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.train_epochs = train_epochs
        self.gamma = gamma
        self._lambda = _lambda
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef

    def __call__(self, x):
        with torch.no_grad():
            return self.actor(torch.as_tensor(x))

    def train(self, states, actions, rewards):
        T = states.shape[0]
        np_gamma = np.flip(np.array([self.gamma**t for t in range(T)]))
        g_t = torch.as_tensor(np.convolve(np_gamma, rewards)[T-1:], dtype=torch.float32)
        states_t = torch.as_tensor(states)
        with torch.no_grad():
            p_a = self.actor(states_t)
            pi_theta_old = p_a[range(T), actions]
            advantage = g_t - self.critic(states_t).squeeze()
        for _ in range(self.train_epochs):
            self.critic_optim.zero_grad()
            v_s = self.critic(states_t).squeeze()
            loss = torch.nn.functional.mse_loss(v_s, g_t)
            loss.backward()
            self.critic_optim.step()

            self.actor_optim.zero_grad()
            p_a = self.actor(states_t)
            pi_theta = p_a[range(T), actions]
            L_unconstrained = (pi_theta/pi_theta_old)*advantage
            L_constrained = clip(pi_theta/pi_theta_old, self.epsilon)*advantage
            # Encourage exploration
            entropy = - torch.sum(p_a * torch.log(p_a))
            L_entropy = entropy * self.entropy_coef
            loss = -torch.mean(torch.minimum(L_unconstrained, L_constrained)) - L_entropy
            loss.backward()
            self.actor_optim.step()

    def save(self, path):
        torch.save(self.actor, path / "ppo_actor.pt")
        torch.save(self.critic, path / "ppo_critic.pt")

    def load(self, path):
        self.actor = torch.load(path / "ppo_actor.pt")
        self.critic = torch.load(path / "ppo_critic.pt")


class Actor(nn.Module):

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
        return torch.softmax(self.output(x), dim=-1)


class Critic(nn.Module):

    def __init__(self, size_s):
        super().__init__()
        self.linear1 = nn.Linear(size_s, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 128)
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        return self.output(x)
