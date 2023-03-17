"""The noise module provides sampler objects that generate random actions.

By providing a common interface for all noise samplers, algorithms can sample exploration actions
independently of the noise type chosen.
"""
from abc import ABC, abstractmethod
import sys

import numpy as np


class Noise(ABC):
    """Abstract base class for noise sampling processes."""

    def __init__(self):
        """Initialize the numpy random generator."""
        self.np_random = np.random.default_rng()

    @abstractmethod
    def sample(self):
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
        self.size_n = size_n

    def sample(self) -> int:
        """Sample a random action in the range of [0, ``size_n``).

        Returns:
            The random action.
        """
        return self.np_random.integers(0, self.size_n).item()

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
        self.size_n = size_n

    def sample(self, mask: np.ndarray) -> int:
        """Sample a random action in the range of [0, ``size_n``) while omitting masked actions.

        Args:
            mask: A numpy array of 0s and 1s of size ``size_n``, with 1s denoting valid actions.

        Returns:
            The random action.
        """
        return np.argmax(self.np_random.random(self.size_n) * mask).item()

    def reset(self):
        """Reset the noise process in case of stateful noise."""


def get_noise_class(noise_type: str) -> Noise:
    """Get the noise class from the noise name.

    Note:
        This function returns a type rather than an instance!

    Args:
        noise_type: The noise type name.

    Returns:
        The noise type.
    """
    return getattr(sys.modules[__name__], noise_type)
