"""The ``networks`` module is a collection of neural network architectures used in RL.

While the architecture is usually less important for agent performance while staying within
reasonable hyperparameter regimes and mostly dense networks, users may want to experiment with
different network styles such as noisy nets for exploration.
"""
import sys
from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_net_class(network_type: str) -> Type[nn.Module]:
    """Get the network class from the network string.

    Note:
        This function returns a type rather than an instance!

    Args:
        network_type: The network type name.

    Returns:
        The network type.

    Raises:
        AttributeError: The specified network type does not exist.
    """
    return getattr(sys.modules[__name__], network_type)


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Initialize a linear layer with orthogonal weights and constant bias.

    Note:
        This is an in-place function. The returned reference is just for convenience.

    Args:
        layer: The network layer.
        std: The standard deviation of the orthogonal weights.
        bias_const: The constant bias value.

    Returns:
        The initialized layer.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DQN(nn.Module):
    """Deep Q network class.

    The network has four layers with variable input, layer and output dimensions. Uses ReLU as
    non-linearities.
    """

    def __init__(self, input_dims: int, output_dims: int, layer_dims: int):
        """Initialize the network layers with specified dimensions.

        Args:
            input_dims: Number of input nodes.
            output_dims: Number of network outputs.
            layer_dims: Number of nodes in the hidden layers.
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dims, layer_dims)
        self.linear2 = nn.Linear(layer_dims, layer_dims)
        self.linear3 = nn.Linear(layer_dims, layer_dims)
        self.output = nn.Linear(layer_dims, output_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the network.

        Args:
            x: Network input.

        Returns:
            The network output.
        """
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        return self.output(x)


class AdvantageDQN(nn.Module):
    """Advantage deep Q network class.

    The network has a configurable number of hidden layers with variable input, layer and output
    dimensions. Uses ReLU as non-linearities. Calculates a baseline value as well as advantage
    values, adds the values and substracts the mean of the advantage values. Layers are initialized
    with orthogonal weights.
    """

    def __init__(self, input_dims: int, output_dims: int, layer_dims: int, nlayers: int = 2):
        """Initialize the network layers and add them to the module list.

        Note:
            An advantage and baseline layer will be added on top of the hidden layers.

        Args:
            input_dims: Input layer dimension.
            output_dims: Output layer dimension.
            layer_dims: Hidden layers dimension.
            nlayers: Number of hidden layers in the network.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the network.

        Args:
            x: Network input.

        Returns:
            The network output.
        """
        for layer in self.layers:
            x = layer(x)
        v_s = self.baseline(x)
        a_s = self.advantage(x)
        return a_s + v_s - torch.mean(a_s, dim=-1, keepdim=True)


class NoisyDQN(nn.Module):
    """Noisy deep Q network class.

    The network has two noisy hidden layers with variable input, layer and output dimensions. Uses
    ReLU as non-linearities.

    See https://arxiv.org/abs/1706.10295.
    """

    def __init__(self, input_dims: int, output_dims: int, layer_dims: int):
        """Initialize the network layers.

        Args:
            input_dims: Input layer dimension.
            output_dims: Output layer dimension.
            layer_dims: Hidden layers dimension.
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dims, layer_dims)
        self.noisy1 = NoisyLinear(layer_dims, layer_dims)
        self.noisy2 = NoisyLinear(layer_dims, output_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the network.

        Args:
            x: Network input.

        Returns:
            The network output.
        """
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.noisy1(x))
        return self.noisy2(x)

    def reset_noise(self):
        """Reset the noise in all network layers."""
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()


class NoisyAdvantageDQN(nn.Module):
    """Noisy advantage deep Q network class.

    The network has two noisy hidden layers with variable input, layer and output dimensions. Uses
    ReLU as non-linearities. Calculates a baseline value as well as advantage values, adds the
    values and substracts the mean of the advantage values.
    """

    def __init__(self, input_dims: int, output_dims: int, layer_dims: int):
        """Initialize the network layers.

        Args:
            input_dims: Input layer dimension.
            output_dims: Output layer dimension.
            layer_dims: Hidden layers dimension.
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dims, layer_dims)
        self.noisy1 = NoisyLinear(layer_dims, layer_dims)
        self.noisy2 = NoisyLinear(layer_dims, layer_dims)
        self.baseline = nn.Linear(layer_dims, 1)
        self.advantage = nn.Linear(layer_dims, output_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the network.

        Args:
            x: Network input.

        Returns:
            The network output.
        """
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.noisy1(x))
        x = torch.relu(self.noisy2(x))
        v_s = self.baseline(x)
        a_s = self.advantage(x)
        return a_s + v_s - torch.mean(a_s, dim=-1, keepdim=True)

    def reset_noise(self):
        """Reset the noise in all network layers."""
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()


class NoisyAdvantageSkipDQN(nn.Module):
    """Noisy advantage deep Q network class with skip connections.

    The network has two noisy layers and a skip connection. Uses ReLU as non-linearities. Calculates
    a baseline value as well as advantage values, adds the values and substracts the mean of the
    advantage values.
    """

    def __init__(self, input_dims: int, output_dims: int, layer_dims: int):
        """Initialize the network layers.

        Args:
            input_dims: Input layer dimension.
            output_dims: Output layer dimension.
            layer_dims: Hidden layers dimension.
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dims, layer_dims)
        self.skip_layer = nn.Linear(layer_dims, layer_dims)
        self.noisy1 = NoisyLinear(layer_dims, layer_dims)
        self.noisy2 = NoisyLinear(layer_dims, layer_dims)
        self.baseline = nn.Linear(layer_dims * 2, 1)
        self.advantage = nn.Linear(layer_dims * 2, output_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the network.

        Args:
            x: Network input.

        Returns:
            The network output.
        """
        x = torch.relu(self.linear1(x))
        skip = self.skip_layer(x)
        x = torch.relu(self.noisy1(x))
        x = self.noisy2(x)
        x = torch.relu(torch.cat((x, skip), dim=-1))
        v_s = self.baseline(x)
        a_s = self.advantage(x)
        return a_s + v_s - torch.mean(a_s, dim=-1, keepdim=True)

    def reset_noise(self):
        """Reset the noise in all network layers."""
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()


class PPOActor(nn.Module):
    """PPOActor network parameterizing a stochasic policy from state inputs."""

    def __init__(self, input_dims: int, output_dims: int, layer_dims: int):
        """Initialize the layers of the actor network.

        Args:
            input_dims: Network input dimensions.
            output_dims: Network output dimensions.
            layer_dims: Network layer dimensions.
        """
        super().__init__()
        self.linear1 = layer_init(nn.Linear(input_dims, layer_dims))
        self.linear2 = layer_init(nn.Linear(layer_dims, layer_dims))
        self.output = layer_init(nn.Linear(layer_dims, output_dims), std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the network.

        Args:
            x: Network input.

        Returns:
            The network output.
        """
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        return torch.softmax(self.output(x), dim=-1)


class PPOCritic(nn.Module):
    """PPO critic network to generate value estimates from states."""

    def __init__(self, input_dims: int, layer_dims: int):
        """Initialize the layers of the critic network.

        Args:
            input_dims: Network input dimensions.
            layer_dims: Network layer dimensions.
        """
        super().__init__()
        self.linear1 = layer_init(nn.Linear(input_dims, layer_dims))
        self.linear2 = layer_init(nn.Linear(layer_dims, layer_dims))
        self.output = layer_init(nn.Linear(layer_dims, 1), std=1.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the network.

        Args:
            x: Network input.

        Returns:
            The network output.
        """
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        return self.output(x)


class NoisyLinear(nn.Module):
    """A noisy linear layer to create noisy Q networks.

    For more details, see https://arxiv.org/pdf/1706.10295.pdf.
    """

    def __init__(self, input_dims: int, output_dims: int, std_init: float = 0.5):
        """Initialize the weight tensors and the noise weights.

        Args:
            input_dims: Layer input dimensions.
            output_dims: Layer output dimensions.
            std_init: Standard deviation for weight initialization.
        """
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
        """Reset the mu and sigma parameter weights."""
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
        """Resample the noise in the layer and update the weights."""
        epsilon_in = self.scale_noise(self.input_dims)
        epsilon_out = self.scale_noise(self.output_dims)
        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the layer.

        Args:
            x: Layer input.

        Returns:
            The layer output.
        """
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return F.linear(x, weight, bias)

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Create a random tensor scaled by the square root of the absolute of its elements."""
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
