"""The noise module provides sampler objects that generate random actions.

By providing a common interface for all noise samplers, algorithms can sample exploration actions
independently of the noise type chosen.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
import torch.nn as nn

from soulsai.utils import module_type_from_string

if TYPE_CHECKING:
    from tensordict import NestedKey, TensorDict


noise_cls: Callable[[str], type[Noise]] = module_type_from_string(__name__)


class Noise(ABC, nn.Module):
    """Abstract base class for noise sampling processes."""

    def __init__(self):
        """Initialize the numpy random generator."""
        super().__init__()
        self.np_random = np.random.default_rng()

    def forward(self, sample: TensorDict) -> torch.Tensor:
        """Alias for `Noise.sample(sample)`.

        Args:
            sample: The sample TensorDict the noise is sampled for.

        Returns:
            The random action.
        """
        return self.sample(sample)

    @abstractmethod
    def sample(self) -> int:
        """Generate a single noise sample."""

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

    def sample(self, _: TensorDict) -> torch.Tensor:
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

    def __init__(self, size_n: int, mask_key: NestedKey):
        """Initialize the base class.

        Args:
            size_n: The number of possible actions.
            mask_key: The key to the action mask in the sample TensorDict.
        """
        super().__init__()
        assert size_n > 0
        self.size_n = torch.nn.Parameter(
            torch.tensor(size_n, dtype=torch.int32), requires_grad=False
        )
        self._mask_key = mask_key if isinstance(mask_key, str) else tuple(mask_key)

    def sample(self, sample: TensorDict) -> torch.Tensor:
        """Sample a random action in the range of [0, ``size_n``) while omitting masked actions.

        Args:
            sample: The sample TensorDict the noise is sampled for.

        Returns:
            The random action.
        """
        return torch.argmax(torch.rand(self.size_n) * sample[self._mask_key]).reshape(1)
