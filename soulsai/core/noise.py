"""The noise module provides sampler objects that generate random actions.

By providing a common interface for all noise samplers, algorithms can sample exploration actions
independently of the noise type chosen.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from soulsai.utils import module_type_from_string

noise_cls: Callable[[str], type[Noise]] = module_type_from_string(__name__)


class Noise(ABC, nn.Module):
    """Abstract base class for noise sampling processes."""

    def __init__(self):
        """Initialize the numpy random generator."""
        super().__init__()
        self.np_random = np.random.default_rng()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method of the noise sampler."""
        return self.sample(x)

    @abstractmethod
    def sample(self) -> int:
        """Generate a single noise sample."""

    @abstractmethod
    def reset(self):
        """Reset the noise process in case of stateful noise."""


class UniformDiscreteNoise(Noise):
    """Discrete action noise sampler sampling uniformly over the interval of [0, ``size_n``)."""

    def __init__(self, size_n: int):
        """Initialize the base class.

        Args:
            size_n: The number of possible actions.
        """
        super().__init__()
        assert size_n > 0
        self.size_n = torch.nn.Parameter(
            torch.tensor(size_n, dtype=torch.int32), requires_grad=False
        )

    def sample(self, _: torch.Tensor) -> torch.Tensor:
        """Sample a random action in the range of [0, ``size_n``).

        Returns:
            The random action.
        """
        return torch.randint(0, self.size_n, (1,))

    def reset(self):
        """Reset the noise process in case of stateful noise."""


class MaskedDiscreteNoise(Noise):
    """Discrete action noise sampler omitting masked actions in the sample process.

    Actions are uniformly sampled in the set of valid actions defined by the action mask.
    """

    def __init__(self, size_n: int):
        """Initialize the base class.

        Args:
            size_n: The number of possible actions.
        """
        super().__init__()
        assert size_n > 0
        self.size_n = torch.nn.Parameter(
            torch.tensor(size_n, dtype=torch.int32), requires_grad=False
        )

    def sample(self, mask: torch.BoolTensor) -> torch.Tensor:
        """Sample a random action in the range of [0, ``size_n``) while omitting masked actions.

        Args:
            mask: A torch bool tensor of size ``size_n``.

        Returns:
            The random action.
        """
        return torch.argmax(torch.rand(self.size_n) * mask)

    def reset(self):
        """Reset the noise process in case of stateful noise."""
