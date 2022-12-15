import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_net_class(network_type):
    if network_type == "DQN":
        return DQN
    if network_type == "AdvantageDQN":
        return AdvantageDQN
    if network_type == "NoisyDQN":
        return NoisyDQN
    if network_type == "NoisyAdvantageDQN":
        return NoisyAdvantageDQN
    if network_type == "NoisyAdvantageSkipDQN":
        return NoisyAdvantageSkipDQN
    if network_type == "PPOActor":
        return PPOActor
    if network_type == "PPOCritic":
        return PPOCritic
    raise ValueError(f"Net type {network_type} not supported!")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DQN(nn.Module):

    def __init__(self, input_dims, output_dims, layer_dims):
        super().__init__()
        self.linear1 = nn.Linear(input_dims, layer_dims)
        self.linear2 = nn.Linear(layer_dims, layer_dims)
        self.linear3 = nn.Linear(layer_dims, layer_dims)
        self.output = nn.Linear(layer_dims, output_dims)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        return self.output(x)


class AdvantageDQN(nn.Module):

    def __init__(self, input_dims, output_dims, layer_dims, nlayers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(nlayers):
            size_in = input_dims if i == 0 else layer_dims
            layer = nn.Linear(size_in, layer_dims)
            torch.nn.init.orthogonal_(layer.weight, gain=np.sqrt(2.))
            self.layers.append(layer)
            self.layers.append(nn.ReLU())
        self.baseline = nn.Linear(layer_dims, 1)
        self.advantage = nn.Linear(layer_dims, output_dims)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        v_s = self.baseline(x)
        a_s = self.advantage(x)
        return a_s + v_s - torch.mean(a_s, dim=-1, keepdim=True)


class NoisyDQN(nn.Module):

    def __init__(self, input_dims, output_dims, layer_dims):
        super().__init__()
        self.linear1 = nn.Linear(input_dims, layer_dims)
        self.noisy1 = NoisyLinear(layer_dims, layer_dims)
        self.noisy2 = NoisyLinear(layer_dims, output_dims)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.noisy1(x))
        return self.noisy2(x)

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()


class NoisyAdvantageDQN(nn.Module):
    
    def __init__(self, input_dims, output_dims, layer_dims):
        super().__init__()
        self.linear1 = nn.Linear(input_dims, layer_dims)
        self.noisy1 = NoisyLinear(layer_dims, layer_dims)
        self.noisy2 = NoisyLinear(layer_dims, layer_dims)
        self.baseline = nn.Linear(layer_dims, 1)
        self.advantage = nn.Linear(layer_dims, output_dims)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.noisy1(x))
        x = torch.relu(self.noisy2(x))
        v_s = self.baseline(x)
        a_s = self.advantage(x)
        return a_s + v_s - torch.mean(a_s, dim=-1, keepdim=True)

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()


class NoisyAdvantageSkipDQN(nn.Module):
    
    def __init__(self, input_dims, output_dims, layer_dims):
        super().__init__()
        self.linear1 = nn.Linear(input_dims, layer_dims)
        self.skip_layer = nn.Linear(layer_dims, layer_dims)
        self.noisy1 = NoisyLinear(layer_dims, layer_dims)
        self.noisy2 = NoisyLinear(layer_dims, layer_dims)
        self.baseline = nn.Linear(layer_dims * 2, 1)
        self.advantage = nn.Linear(layer_dims * 2, output_dims)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        skip = self.skip_layer(x)
        x = torch.relu(self.noisy1(x))
        x = self.noisy2(x)
        x = torch.relu(torch.cat((x, skip), dim=-1))
        v_s = self.baseline(x)
        a_s = self.advantage(x)
        return a_s + v_s - torch.mean(a_s, dim=-1, keepdim=True)

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()


class PPOActor(nn.Module):

    def __init__(self, input_dims, output_dims, layer_dims):
        super().__init__()
        self.linear1 = layer_init(nn.Linear(input_dims, layer_dims))
        self.linear2 = layer_init(nn.Linear(layer_dims, layer_dims))
        self.output = layer_init(nn.Linear(layer_dims, output_dims), std=0.01)

    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        return torch.softmax(self.output(x), dim=-1)


class PPOCritic(nn.Module):

    def __init__(self, input_dims, layer_dims):
        super().__init__()
        self.linear1 = layer_init(nn.Linear(input_dims, layer_dims))
        self.linear2 = layer_init(nn.Linear(layer_dims, layer_dims))
        self.output = layer_init(nn.Linear(layer_dims, 1), std=1.)

    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        return self.output(x)


class NoisyLinear(nn.Module):

    def __init__(self, input_dims: int, output_dims: int, std_init: float = 0.5):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(output_dims, input_dims))
        self.weight_sigma = nn.Parameter(torch.Tensor(output_dims, input_dims))
        self.register_buffer("weight_epsilon", torch.Tensor(output_dims, input_dims))

        self.bias_mu = nn.Parameter(torch.Tensor(output_dims))
        self.bias_sigma = nn.Parameter(torch.Tensor(output_dims))
        self.register_buffer("bias_epsilon", torch.Tensor(output_dims))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.input_dims)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        # torch.nn.init.orthogonal_(self.weight_mu)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.input_dims))
        # torch.nn.init.constant_(self.weight_sigma, self.std_init / np.sqrt(self.input_dims))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        # torch.nn.init.uniform_(self.bias_mu, -mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.output_dims))
        # torch.nn.init.constant_(self.bias_sigma, self.std_init / np.sqrt(self.output_dims))

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.input_dims)
        epsilon_out = self.scale_noise(self.output_dims)
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
